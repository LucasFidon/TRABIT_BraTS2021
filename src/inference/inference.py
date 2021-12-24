# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
import torch
import torch.nn.functional as F
import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset, DataLoader, NiftiSaver
from monai.transforms import Zoomd, CenterSpatialCropd
from src.networks.factory import get_network
from src.data.factory import get_single_case_dataloader
from src.utils.definitions import *

def _check_input_path(data_config, input_path_dict):
    for key in data_config['info']['image_keys']:
        assert key in list(input_path_dict.keys()), 'Input key %s not found in the input paths provided' % key


def segment(config, data_config, model_path, input_path_dict, save_folder):
    def pad_if_needed(img, patch_size):
        # Define my own dummy padding function because the one from MONAI
        # does not retain the padding values, and as a result
        # we cannot unpad after inference...
        img_np = img.cpu().numpy()
        shape = img.shape[2:]
        need_padding = np.any(shape < np.array(patch_size))
        if not need_padding:
            pad_list = [(0, 0)] * 3
            return img, np.array(pad_list)
        else:
            pad_list = []
            for dim in range(3):
                diff = patch_size[dim] - shape[dim]
                if diff > 0:
                    margin = diff // 2
                    pad_dim = (margin, diff - margin)
                    pad_list.append(pad_dim)
                else:
                    pad_list.append((0, 0))
            padded_array = np.pad(
                img_np,
                [(0, 0), (0, 0)] + pad_list,  # pad only the spatial dimensions
                'constant',
                constant_values=[(0, 0)] * 5,
            )
            padded_img = torch.tensor(padded_array).float()
            return padded_img, np.array(pad_list)

    # Check that the provided input paths and the data config correspond
    _check_input_path(data_config, input_path_dict)

    device = torch.device("cuda:0")

    # Create the dataloader for the single case to segment
    dataloader = get_single_case_dataloader(
        config=config,
        data_config=data_config,
        input_path_dict=input_path_dict,
    )

    # Create the network and load the checkpoint
    net = get_network(
        config=config,
        in_channels=data_config['info']['in_channels'],
        n_class=data_config['info']['n_class'],
        device=device,
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    # The inferer is in charge of taking a full volumetric input
    # and run the window-based prediction using the network.
    inferer = SlidingWindowInferer(
        roi_size=config['data']['patch_size'],  # patch size to use for inference
        sw_batch_size=1,  # max number of windows per network inference iteration
        overlap=0.5,  # amount of overlap between windows (in [0, 1])
        mode="gaussian",  # how to blend output of overlapping windows
        sigma_scale=0.125,  # sigma for the Gaussian blending. MONAI default=0.125
        padding_mode="constant",  # for when ``roi_size`` is larger than inputs
        cval=0.,  # fill value to use for padding
    )

    torch.cuda.empty_cache()

    net.eval()  # Put the CNN in evaluation mode
    with torch.no_grad():  # we do not need to compute the gradient during inference
        # Load and prepare the full image
        data = [d for d in dataloader][0]  # load the full image
        input = torch.cat(tuple([data[key] for key in data_config['info']['image_keys']]), 1)
        input, pad_values = pad_if_needed(input, config['data']['patch_size'])
        input = input.to(device)
        pred = inferer(inputs=input, network=net)
        n_pred = 1
        # Perform test-time flipping augmentation
        flip_dims = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
        for dims in flip_dims:
            flip_input = torch.flip(input, dims=dims)
            pred += torch.flip(
                inferer(inputs=flip_input, network=net),
                dims=dims,
            )
            n_pred += 1
        pred /= n_pred
    seg = pred.argmax(dim=1, keepdims=True).float()

    # Post-processing
    if data_config['name'] == 'BraTS2020':
        # Convert the segmentation label to the original BraTS labels
        seg[seg == 3] = 4

        # Remove small connected components for BraTS
        num_ET_voxels = torch.sum(seg == data_config['info']['labels']['ET'])
        if num_ET_voxels < THRESHOLD_ET and num_ET_voxels > 0:
            print('')
            print('Only %d voxels were predicted as ET.' % num_ET_voxels)
            print('Change all ET predictions to NCR/NET')
            seg[seg == data_config['info']['labels']['ET']] = data_config['info']['labels']['NCR_NET']

    # Unpad the prediction
    seg = seg[:, :, pad_values[0,0]:seg.size(2)-pad_values[0,1], pad_values[1,0]:seg.size(3)-pad_values[1,1], pad_values[2,0]:seg.size(4)-pad_values[2,1]]

    # Insert the segmentation in the original image size
    meta_data = data['%s_meta_dict' % data_config['info']['image_keys'][0]]
    dim = meta_data['spatial_shape'].cpu().numpy()
    full_dim = [1, 1, dim[0,0], dim[0,1], dim[0,2]]
    fg_start = data['foreground_start_coord'][0]
    fg_end = data['foreground_end_coord'][0]
    full_seg = torch.zeros(full_dim)
    full_seg[:, :, fg_start[0]:fg_end[0], fg_start[1]:fg_end[1], fg_start[2]:fg_end[2]] = seg

    # Save the segmentation
    saver = NiftiSaver(output_dir=save_folder, mode="nearest", output_postfix="")
    saver.save_batch(full_seg, meta_data=meta_data)


def segment_brats_21(config, data_config, model_paths_list, input_path_dict, save_folder,
                     zoom_aug=False, no_label_conversion=False):
    def pad_if_needed(img, patch_size):
        # Define my own dummy padding function because the one from MONAI
        # does not retain the padding values, and as a result
        # we cannot unpad after inference...
        # img_np = img.cpu().numpy()
        shape = img.shape[2:]
        need_padding = np.any(shape < np.array(patch_size))
        if not need_padding:
            pad_list = [(0, 0)] * 3
            return img, np.array(pad_list)
        else:
            img_np = img.cpu().numpy()
            pad_list = []
            for dim in range(3):
                diff = patch_size[dim] - shape[dim]
                if diff > 0:
                    margin = diff // 2
                    pad_dim = (margin, diff - margin)
                    pad_list.append(pad_dim)
                else:
                    pad_list.append((0, 0))
            padded_array = np.pad(
                img_np,
                [(0, 0), (0, 0)] + pad_list,  # pad only the spatial dimensions
                'constant',
                constant_values=[(0, 0)] * 5,
            )
            padded_img = torch.tensor(padded_array).float()
            return padded_img, np.array(pad_list)

    def need_cropping(img, patch_size):
        shape = img.shape[2:]
        need_cropping = np.any(shape > np.array(patch_size))
        return need_cropping

    def crop_coord_from_center(center, patch_size, ori_size):
        roi_start = np.maximum(center - np.floor_divide(patch_size, 2), 0)
        roi_end = np.minimum(roi_start + patch_size, ori_size)
        roi_start = roi_end - patch_size  # need this in case roi_end values were clipped
        return roi_start, roi_end

    def compute_crop_around_tumor_coord(seg, patch_size):
        from scipy import ndimage
        # Compute the center of gravity of the predicted whole tumor segmentation
        seg_np = np.squeeze(seg.cpu().numpy())  # keep only x,y,z dimensions
        num_foreground_voxels = np.sum(seg_np)
        if num_foreground_voxels == 0:  # Empty tumor segmentation
            center_of_mass = np.array(patch_size, dtype=np.int16) // 2
        else:
            center_of_mass = np.asarray(ndimage.measurements.center_of_mass(seg_np), dtype=np.int16)
        roi_start, roi_end = crop_coord_from_center(
            center_of_mass,
            patch_size=patch_size,
            ori_size=seg_np.shape
        )
        return roi_start, roi_end

    SCALE_FACTOR = 1.125  # chosen to be 1 + 1 / 2**p for some p integer >= 0

    # Check that the provided input paths and the data config correspond
    _check_input_path(data_config, input_path_dict)

    device = torch.device("cuda:0")

    # Create the dataloader for the single case to segment
    dataloader = get_single_case_dataloader(
        config=config,
        data_config=data_config,
        input_path_dict=input_path_dict,
    )

    # Create the network and load the checkpoint
    net = get_network(
        config=config,
        in_channels=data_config['info']['in_channels'],
        n_class=data_config['info']['n_class'],
        device=device,
    )

    # The inferer is in charge of taking a full volumetric input
    # and run the window-based prediction using the network.
    inferer = SlidingWindowInferer(
        roi_size=config['data']['patch_size'],  # patch size to use for inference
        sw_batch_size=1,  # max number of windows per network inference iteration
        overlap=0.5,  # amount of overlap between windows (in [0, 1])
        mode="gaussian",  # how to blend output of overlapping windows
        sigma_scale=0.125,  # sigma for the Gaussian blending. MONAI default=0.125
        padding_mode="constant",  # for when ``roi_size`` is larger than inputs
        cval=0.,  # fill value to use for padding
    )

    torch.cuda.empty_cache()

    net.eval()  # Put the CNN in evaluation mode

    # Load the full image after cropping the fg
    # We pad if needed to make sure the image in larger than the patch size
    data = [d for d in dataloader][0]
    input = torch.cat(tuple([data[key] for key in data_config['info']['image_keys']]), 1)
    input, pad_values = pad_if_needed(input, config['data']['patch_size'])
    input = input.to(device)
    flip_input = torch.flip(input, dims=(2,))

    # Compute the first segmentation
    model_localization = model_paths_list[0]
    checkpoint = torch.load(model_localization, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    with torch.no_grad():
        pred = 0.5 * inferer(inputs=input, network=net)
        pred += 0.5 * torch.flip(
            inferer(inputs=flip_input, network=net),
            dims=(2,),
        )
    first_seg = pred.argmax(dim=1, keepdims=True)

    # Compute the final segmentation
    if len(model_paths_list) == 1:  # Single model. We are done.
        seg = first_seg.float()
    else:  # Ensemble of models
        do_cropping = need_cropping(input, config['data']['patch_size'])
        if do_cropping:
            # Crop around the tumor for the cropped input to fit in one patch
            print('Crop around the tumor for the cropped input to fit in one patch')
            roi_start, roi_end = compute_crop_around_tumor_coord(
                first_seg,
                config['data']['patch_size'],
            )
            input_crop = input[:,:,roi_start[0]:roi_end[0],roi_start[1]:roi_end[1],roi_start[2]:roi_end[2]]
            flip_input_crop = torch.flip(input_crop, dims=(2,))
        else:
            input_crop = input
            flip_input_crop = flip_input

        # Ensemble-based segmentation for the cropped region
        softmax = 0
        for model_path in model_paths_list:
            # Load the parameters of the model
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            with torch.no_grad():  # we do not need to compute the gradient during inference
                # Perform test-time flipping augmentation
                pred = 0.1 * inferer(inputs=input_crop, network=net)  # temperature param before softmax
                softmax += F.softmax(pred, dim=1)
                pred += 0.1 * torch.flip(
                    inferer(inputs=flip_input_crop, network=net),
                    dims=(2,),
                )
                softmax += F.softmax(pred, dim=1)
        softmax /= 2 * len(model_paths_list)

        # Ensemble with test-time zoom augmentation
        if zoom_aug:
            print('Apply test-time zoom with scale factor %f' % SCALE_FACTOR)
            # Zoom in
            input_crop_zoom = F.interpolate(
                input_crop,
                scale_factor=SCALE_FACTOR,
                mode='trilinear',
                align_corners=True,
                recompute_scale_factor=False,
            )
            size_after_zoom = np.array([i for i in input_crop_zoom.size()[2:]])
            center_after_zoom = size_after_zoom // 2
            zoom_start, zoom_end = crop_coord_from_center(
                center=center_after_zoom,
                patch_size=config['data']['patch_size'],
                ori_size=size_after_zoom,
            )
            input_crop_zoom = input_crop_zoom[:,:,zoom_start[0]:zoom_end[0],zoom_start[1]:zoom_end[1],zoom_start[2]:zoom_end[2]]
            flip_input_crop_zoom = torch.flip(input_crop_zoom, dims=(2,))

            # Segmentation prediction
            softmax_zoom = 0
            for model_path in model_paths_list:
                # Load the parameters of the model
                checkpoint = torch.load(model_path, map_location='cpu')
                net.load_state_dict(checkpoint['net'])
                with torch.no_grad():  # we do not need to compute the gradient during inference
                    # Perform test-time flipping augmentation
                    pred = 0.1 * inferer(inputs=input_crop_zoom, network=net)  # temperature param before softmax
                    softmax_zoom += F.softmax(pred, dim=1)
                    pred += 0.1 * torch.flip(
                        inferer(inputs=flip_input_crop_zoom, network=net),
                        dims=(2,),
                    )
                    softmax_zoom += F.softmax(pred, dim=1)
            softmax_zoom /= 2 * len(model_paths_list)

            # Zoom out
            softmax_zoom_padded = torch.zeros(
                [softmax_zoom.size(0), softmax_zoom.size(1) + 1] + size_after_zoom.tolist()
            ).to(softmax_zoom.device)
            # Add an extra label for padding
            softmax_zoom_padded[-1,:,:,:,:] = 1.
            # Copy the predicted proba
            softmax_zoom_padded[:,:-1,zoom_start[0]:zoom_end[0],zoom_start[1]:zoom_end[1],zoom_start[2]:zoom_end[2]] = softmax_zoom
            softmax_zoom_padded[:,-1,zoom_start[0]:zoom_end[0],zoom_start[1]:zoom_end[1],zoom_start[2]:zoom_end[2]] = 0
            softmax_zoom_out = F.interpolate(
                softmax_zoom_padded,
                scale_factor=1./SCALE_FACTOR,
                mode='trilinear',
                align_corners=True,
                recompute_scale_factor=False,
            )

            # Average the softmax probability values for the non padded voxels
            mask = (softmax_zoom_out[:,-1,:,:,:] == 0).float()  # for non padded voxels the padding proba is equal to 0
            softmax = (1. - mask) * softmax + 0.5 * mask * (softmax + softmax_zoom_out[:,:-1,:,:,:])

        seg_crop = softmax.argmax(dim=1, keepdims=True).float()

        # Uncrop the segmentation (if needed)
        if do_cropping:
            seg = torch.zeros_like(first_seg)
            seg[:,:,roi_start[0]:roi_end[0],roi_start[1]:roi_end[1],roi_start[2]:roi_end[2]] = seg_crop
        else:
            seg = seg_crop

    # Post-processing
    if not no_label_conversion:
        # Convert the segmentation label to the original BraTS labels
        seg[seg == 3] = 4

    # Remove small connected components for BraTS
    num_ET_voxels = torch.sum(seg == data_config['info']['labels']['ET'])
    if num_ET_voxels < THRESHOLD_ET and num_ET_voxels > 0:
        print('\033[91mOnly %d voxels were predicted as ET.' % num_ET_voxels)
        print('Change all ET predictions to NCR/NET\033[0m')
        seg[seg == data_config['info']['labels']['ET']] = data_config['info']['labels']['NCR_NET']

    # Unpad the prediction
    seg = seg[:, :, pad_values[0,0]:seg.size(2)-pad_values[0,1], pad_values[1,0]:seg.size(3)-pad_values[1,1], pad_values[2,0]:seg.size(4)-pad_values[2,1]]

    # Insert the segmentation in the original image size
    meta_data = data['%s_meta_dict' % data_config['info']['image_keys'][0]]
    dim = meta_data['spatial_shape'].cpu().numpy()
    full_dim = [1, 1, dim[0,0], dim[0,1], dim[0,2]]
    fg_start = data['foreground_start_coord'][0]
    fg_end = data['foreground_end_coord'][0]
    full_seg = torch.zeros(full_dim)
    full_seg[:, :, fg_start[0]:fg_end[0], fg_start[1]:fg_end[1], fg_start[2]:fg_end[2]] = seg

    # Save the segmentation
    saver = NiftiSaver(output_dir=save_folder, mode="nearest", output_postfix="")
    saver.save_batch(full_seg, meta_data=meta_data)
