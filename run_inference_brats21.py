# Copyright 2021 Lucas Fidon and Suprosanna Shit

"""
@brief  Script for performing segmentation inference on one case
        using a CNN that was previously trained.
        This is the script run by the docker submission.
"""

import os
import time
from argparse import ArgumentParser
from config.loader import load_config
from src.inference.inference import segment_brats_21
from dataset_config.loader import load_dataset_config

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config.')
parser.add_argument('--data_config',
                    required=True,
                    help='config file (.yml) containing the hyper-parameters for the dataset. '
                         'See /dataset_config for examples.')
parser.add_argument('--model',
                    nargs='+',
                    help='Paths to the checkpoints to use for inference separated by a space.')
parser.add_argument('--save_folder', help='Folder where to save the predicted segmentations.')
parser.add_argument('--zoom', action='store_true', help='Apply test-time zoom augmentation')


def main(args):
    config = load_config(args.config)
    data_config = load_dataset_config(config_file=args.data_config, verbose=True)

    case_folder = '/input'
    nifti_files = [ f for f in os.listdir(case_folder) if '.nii.gz' in f]
    case_ID = nifti_files[0].split('_')[1]
    t1 = os.path.join(case_folder, 'BraTS2021_%s_t1.nii.gz' % case_ID)
    t1ce = os.path.join(case_folder, 'BraTS2021_%s_t1ce.nii.gz' % case_ID)
    t2 = os.path.join(case_folder, 'BraTS2021_%s_t2.nii.gz' % case_ID)
    flair = os.path.join(case_folder, 'BraTS2021_%s_flair.nii.gz' % case_ID)
    input = {'t1': t1, 't1ce': t1ce, 't2': t2, 'flair': flair}
    print('Input:')
    print(input)

    print('\nRun the segmentation...')
    segment_brats_21(
        config=config,
        data_config=data_config,
        model_paths_list=args.model,
        input_path_dict=input,
        save_folder=args.save_folder,
        zoom_aug=args.zoom,
    )
    print('Segmentation done.')

    # MONAI saves the prediction at save_folder/patID/patID.nii.gz
    # but we want to save them at save_folder/patID.nii.gz for online evaluation
    # so we have to move the files around...
    monai_save_path = os.path.join(
        args.save_folder,
        'BraTS2021_%s_flair' % case_ID,
        'BraTS2021_%s_flair.nii.gz' % case_ID,
    )
    save_path = os.path.join(args.save_folder, '%s.nii.gz' % case_ID)
    os.system('cp %s %s' % (monai_save_path, save_path))
    os.system('rm -r %s' % os.path.dirname(monai_save_path))


if __name__ == '__main__':
    args = parser.parse_args()
    t0 = time.time()
    main(args)
    print('\nInference done in %.2f sec\n' % (time.time() - t0))
