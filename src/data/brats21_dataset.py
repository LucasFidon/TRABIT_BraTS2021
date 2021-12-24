# Copyright 2021 Lucas Fidon and Suprosanna Shit

import glob
import os
import random
from monai.data import PersistentDataset, Dataset
from src.data.brats_transform_pipelines import *
from src.utils.definitions import *


def get_brats21_dataset(config, data_config=None, seed=112, mode='training', use_persistent_dataset=False):
    if data_config is None:
        from dataset_config.loader import load_brats21_data_config
        data_config = load_brats21_data_config()

    # Create the BraTS data dictionary
    data_dir = data_config['path']['train']
    flair_ = []
    t1_ = []
    labels_ = []
    t1ce_ = []
    t2_ = []
    exam_folders = [
        f for f in os.listdir(data_dir)
        if not '.' in f
    ]
    print('Found %d cases in %s' % (len(exam_folders), data_dir))
    for folder_name in exam_folders:
        folder_path = os.path.join(data_dir, folder_name)
        flair_path = os.path.join(folder_path, '%s_flair.nii.gz' % folder_name)
        assert os.path.exists(flair_path)
        flair_.append(flair_path)
        t1_path = os.path.join(folder_path, '%s_t1.nii.gz' % folder_name)
        assert os.path.exists(t1_path)
        t1_.append(t1_path)
        t1ce_path = os.path.join(folder_path, '%s_t1ce.nii.gz' % folder_name)
        assert os.path.exists(t1ce_path)
        t1ce_.append(t1ce_path)
        t2_path = os.path.join(folder_path, '%s_t2.nii.gz' % folder_name)
        assert os.path.exists(t2_path)
        t2_.append(t2_path)
        labels_path = os.path.join(folder_path, '%s_seg.nii.gz' % folder_name)
        assert os.path.exists(labels_path)
        labels_.append(labels_path)
    data_dicts = [
        {"flair": flair, "t1": t1, "t1ce": t1ce, "t2":t2, "label": label_name}
        for flair, t1, t1ce, t2, label_name in zip(flair_, t1_, t1ce_, t2_, labels_)
    ]

    # Create the data augmentation/preprocessing pipelines
    if config['data']['data_augmentation'] == 'nnUNet':
        train_transform = brats_nnunet_train_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
            all_keys=data_config['info']['all_keys'],
        )
        val_transform = brats_nnunet_valid_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
            all_keys=data_config['info']['all_keys'],
        )
    else:
        train_transform = None
        val_transform = None

    # Create the datasets
    cache_dir = os.path.join(CACHE_DIR, data_config['log']['name'])
    if use_persistent_dataset:
        print('\n*** Persistent dataset stored in %s' % cache_dir)
    if mode=='training':
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
    elif mode=='val':
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
    elif mode=='split':
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
