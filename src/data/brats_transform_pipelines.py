import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
)
from src.data.transforms import *


def brats_nnunet_train_transform(config, image_keys, all_keys):
    train_transform = Compose([
        LoadImaged(keys=all_keys),
        AddChanneld(keys=all_keys),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        # Crop based on thresholding > 0 in source_key
        CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Spacingd(
            keys=all_keys,
            pixdim=config['data']['spacing'],
            mode=("bilinear",) * len(image_keys) + ("nearest",),
        ),
        #TODO: change orientation for the default orientation used in BraTS
        Orientationd(keys=all_keys, axcodes="RAS"),
        SpatialPadd(keys=all_keys, spatial_size=config['data']['patch_size']),
        #TODO: is the zoom ratio computed for each dimension independently?
        RandZoomd(
            keys=all_keys,
            min_zoom=0.7,
            max_zoom=1.5,
            mode=("trilinear",) * len(image_keys) + ("nearest",),
            align_corners=(True,) * len(image_keys) + (None,),
            prob=0.3,
        ),
        RandRotated(
            keys=all_keys,
            range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            mode=("bilinear",) * len(image_keys) + ("nearest",),
            align_corners=(True,) * len(image_keys) + (None,),
            padding_mode=("border", ) * len(all_keys),
            prob=0.3,
        ),
        # Return a batch of spatial size equal to roi_size
        # that is strictly contained inside the input batch.
        # The position of the output batch is selected randomly (random_center=True).
        # I suppose it si better to do the random crop after the zoom and rotation.
        RandSpatialCropd(
            keys=all_keys,
            roi_size=config['data']['patch_size'],
            random_center=True,
            random_size=False,
        ),
        RandGaussianNoised(keys=image_keys, mean=0., std=0.1, prob=0.2),
        RandGaussianSmoothd(
            keys=image_keys,
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.2,
        ),
        RandAdjustContrastd(  # same as Gamma in nnU-Net
            keys=image_keys,
            gamma=(0.7, 1.5),
            prob=0.3,
        ),
        # RandFlipd(all_keys, spatial_axis=[0, 1, 2], prob=0.5),
        RandFlipd(all_keys, spatial_axis=[0], prob=0.5),  # Only right-left flip
        # if we put NormalizeIntensityd at the end, it solves validation dice oscillation
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return train_transform


def brats_nnunet_valid_transform(config, image_keys, all_keys):
    val_transform = Compose([
        LoadImaged(keys=all_keys),
        AddChanneld(keys=all_keys),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Spacingd(
            keys=all_keys,
            pixdim=config['data']['spacing'],
            mode=("bilinear",) * len(image_keys) + ("nearest",),
        ),
        Orientationd(keys=all_keys, axcodes="RAS"),
        SpatialPadd(keys=all_keys, spatial_size=config['data']['patch_size']),
        NormalizeIntensityd(keys=image_keys, nonzero=False, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return val_transform


def brats_inference_transform(config, image_keys):
    #todo: make sure we revert the image spacing after inference...
    #todo: same problem with orientation as for padding: don't know how to revert it after inference...
    # Those things are only important for compatibility with data that are not from BraTS.
    # Waiting for some updates in MONAI for this.
    # Please let the transformations commented for now.
    inference_transform = Compose([
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        # Spacingd(
        #     keys=IMAGE_KEYS,
        #     pixdim=SPACING,
        #     mode=("bilinear", "bilinear", "bilinear", "bilinear"),
        # ),
        #  Orientationd(keys=IMAGE_KEYS, axcodes="RAS"),  #todo for BraTS 2021
        #  SpatialPadd(keys=IMAGE_KEYS, spatial_size=PATCH_SIZE),  # padding coord are not stored so we cannot unpas afterward...
        NormalizeIntensityd(keys=image_keys, nonzero=False, channel_wise=True),
        CastToTyped(keys=image_keys, dtype=(np.float32, np.float32, np.float32, np.float32)),
        ToTensord(keys=image_keys),
    ])
    return inference_transform
