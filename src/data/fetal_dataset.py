# Copyright 2021 Lucas Fidon

import os
import random
from monai.data import PersistentDataset, Dataset
from src.data.fetal_transform_pipelines import *
from src.utils.definitions import *


def get_fetal_dataset(config, data_config, seed, mode='training', use_persistent_dataset=False):
    # Create the Fetal data dictionary
    data_dir = data_config['path']['train']
    srr_ = []
    labels_ = []
    folders = [f for f in os.listdir(data_dir) if not '.' in f]
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        srr_path = os.path.join(folder_path, 'srr.nii.gz')
        srr_.append(srr_path)
        seg_path = os.path.join(folder_path, 'parcellation.nii.gz')
        labels_.append(seg_path)
    data_dicts = [
        {"srr": srr, "label": label_name}
        for srr, label_name in zip(srr_, labels_)
    ]

    # Create the data augmentation/preprocessing pipelines
    if config['data']['data_augmentation'] == 'nnUNet':
        train_transform = fetal_train_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
            all_keys=data_config['info']['all_keys'],
        )
        val_transform = fetal_valid_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
            all_keys=data_config['info']['all_keys'],
        )
    else:
        raise NotImplementedError(
            'Unknown data augmentation pipeline %s' % \
            config['data']['data_augmentation']
        )

    # Create the dataset
    cache_dir = os.path.join(CACHE_DIR, data_config['log']['name'])
    if use_persistent_dataset:
        print('\n*** Persistent dataset stored in %s' % cache_dir)
    if mode == 'training':
        if use_persistent_dataset:
            ds = PersistentDataset(
                data=data_dicts,
                transform=train_transform,
                cache_dir=cache_dir,
            )
        else:
            ds = Dataset(
                data=data_dicts,
                transform=train_transform,
            )
        return ds
    elif mode == 'val':
        if use_persistent_dataset:
            ds = PersistentDataset(
                data=data_dicts,
                transform=val_transform,
                cache_dir=cache_dir,
            )
        else:
            ds = Dataset(
                data=data_dicts,
                transform=val_transform,
            )
        return ds
    elif mode == 'split':
        # TODO: it would be useful to save the split
        random.seed(seed)
        random.shuffle(data_dicts)
        train_split = int(config['data']['split'] * len(data_dicts))
        train_files = data_dicts[:train_split]
        val_files = data_dicts[train_split:]
        if use_persistent_dataset:
            train_ds = PersistentDataset(
                data=train_files,
                transform=train_transform,
                cache_dir=cache_dir,
            )
            val_ds = PersistentDataset(
                data=val_files,
                transform=val_transform,
                cache_dir=cache_dir,
            )
        else:
            train_ds = Dataset(
                data=train_files,
                transform=train_transform,
            )
            val_ds = Dataset(
                data=val_files,
                transform=val_transform,
            )
        return train_ds, val_ds
