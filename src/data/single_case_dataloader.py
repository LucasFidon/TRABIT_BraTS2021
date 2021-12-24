# Copyright 2021 Lucas Fidon and Suprosanna Shit
"""
Data loader for a single case.
This is typically used for inference.
"""

import torch
from monai.data import Dataset, DataLoader


def single_case_dataloader(inference_transform, input_path_dict):
    """
    :param inference_transform
    :param input_path_dict: dict; keys=image_keys, values=paths
    :return:
    """
    data_dicts = [input_path_dict]
    ds = Dataset(
        data=data_dicts,
        transform=inference_transform,
    )
    loader = DataLoader(
        ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
