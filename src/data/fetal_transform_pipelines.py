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
    RandCropByPosNegLabeld,
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


def fetal_train_transform(config, image_keys, all_keys):
    train_transform = Compose([
        LoadImaged(keys=all_keys),
        AddChanneld(keys=all_keys),
        CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Spacingd(
            keys=all_keys,
            pixdim=config['data']['spacing'],
            mode=("bilinear",) * len(image_keys) + ("nearest",),
        ),
        # TODO do we need the orientation preprocessing?
        #  Probably not as long as we perform the flip for all axis
        #  and even less with the preprocessing that is done offline
        # Orientationd(keys=all_keys, axcodes="LPI"),
        SpatialPadd(keys=all_keys, spatial_size=config['data']['patch_size']),
        RandCropByPosNegLabeld(  # crop with center in label>0 with proba pos / (neg + pos)
            keys=all_keys,
            label_key="label",
            spatial_size=config['data']['patch_size'],
            pos=1,
            neg=0,  # never center in background voxels
            num_samples=1,
            image_key=None,  # for no restriction with image thresholding
            image_threshold=0,
        ),
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
        RandFlipd(all_keys, spatial_axis=[0, 1, 2], prob=0.5),
        # if we put NormalizeIntensityd at the end, it solves validation dice oscillation
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return train_transform


def fetal_valid_transform(config, image_keys, all_keys):
    val_transform = Compose([
        LoadImaged(keys=all_keys),
        AddChanneld(keys=all_keys),
        CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Spacingd(
            keys=all_keys,
            pixdim=config['data']['spacing'],
            mode=("bilinear",) * len(image_keys) + ("nearest",),
        ),
        Orientationd(keys=all_keys, axcodes="LPI"),
        SpatialPadd(keys=all_keys, spatial_size=config['data']['patch_size']),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return val_transform


def fetal_inference_transform(config, image_keys):
    # NB: do not pad; this is done in the segment function for inference
    inference_transform = Compose([
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        # Spacingd(
        #     keys=IMAGE_KEYS,
        #     pixdim=SPACING,
        #     mode="bilinear",
        # ),
        #  Orientationd(keys=IMAGE_KEYS, axcodes="RAS"),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=image_keys, dtype=(np.float32,)),
        ToTensord(keys=image_keys),
    ])
    return inference_transform
