# Copyright 2021 Lucas Fidon

"""
@brief  Script for performing segmentation inference
"""

import os
import time
from argparse import ArgumentParser
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation
from config.loader import load_config
from src.inference.inference import segment
from dataset_config.loader import load_feta_challenge_data_config

NUM_ITER_MASK_DILATION = 5

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config.')
parser.add_argument('--model', help='Path to the checkpoint to use for inference.')
parser.add_argument('--save_folder', help='Folder where to save the predicted segmentations.')


def preprocessing(img_path, mask_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    img_nii = nib.load(img_path)
    img_np = img_nii.get_fdata().astype(np.float32)
    mask_nii = nib.load(mask_path)
    mask_np = mask_nii.get_fdata().astype(np.uint8)

    # Mask the Nans
    if np.count_nonzero(np.isnan(mask_np)) > 0:
        mask_np[np.isnan(mask_np)] = 0

    # Dilate the mask
    mask_dilated_np = binary_dilation(mask_np, iterations=NUM_ITER_MASK_DILATION)

    # Mask the image
    img_np[mask_dilated_np == 0] = 0

    # Clip high intensities
    p_999 = np.percentile(img_np, 99.9)
    img_np[img_np > p_999] = p_999

    # Save the preprocessed image
    new_img_nii = nib.Nifti1Image(img_np, img_nii.affine)
    save_path = os.path.join(save_folder, 'srr_preprocessed.nii.gz')
    nib.save(new_img_nii, save_path)

    return save_path


def main(args):
    """
    Run inference for all the testing data
    """
    config = load_config(args.config)
    data_config = load_feta_challenge_data_config()
    case_names = [
        n for n in os.listdir(data_config['path']['test'])
        if not '.' in n
    ]
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    for i, case_name in enumerate(case_names):
        case_folder = os.path.join(data_config['path']['test'], case_name)
        if os.path.exists(os.path.join(args.save_folder, case_name)) or os.path.exists(os.path.join(args.save_folder, '%s_autoseg.nii.gz' % case_name)):
            print('\n%s segmentation already exists. Skip inference.' % case_name)
            continue
        srr = os.path.join(case_folder, 'srr.nii.gz')
        mask = os.path.join(case_folder, 'mask.nii.gz')

        save_folder = os.path.join(args.save_folder, case_name)
        srr_pre = preprocessing(srr, mask, save_folder=save_folder)

        input_path_dict = {'srr': srr_pre}
        print('\n(%d/%d) Compute the automatic segmentation for the case %s...' % (i+1, len(case_names), case_name))
        t0 = time.time()
        segment(
            config=config,
            data_config=data_config,
            model_path=args.model,
            input_path_dict=input_path_dict,
            save_folder=os.path.join(args.save_folder, case_name),
        )

        #TODO: convert the labels to match the one used in the FeTA challenge

        print('\nInference done in %.2f sec\n' % (time.time() - t0))
        # MONAI saves the prediction at save_folder/patID/patID.nii.gz



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
