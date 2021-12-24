# Copyright 2021 Lucas Fidon and Suprosanna Shit

"""
@brief  Script for performing segmentation inference
        for all the BraTS validation cases
        using a CNN that was previously trained.
"""

import os
import time
from argparse import ArgumentParser
from config.loader import load_config
from src.inference.inference import segment_brats_21
from dataset_config.loader import load_dataset_config


parser = ArgumentParser()
#TODO the same confg is used for all the models at the moment
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
parser.add_argument('--no_label_conversion', action='store_true', help='Use directly the label numbers produced by the model')


def main(args):
    """
    Run inference for all the testing data
    """
    t_start = time.time()
    config = load_config(args.config)
    data_config = load_dataset_config(config_file=args.data_config, verbose=True)
    case_names = [
        n for n in os.listdir(data_config['path']['test'])
        if not '.' in n
    ]
    for i, case_name in enumerate(case_names):
        case_folder = os.path.join(data_config['path']['test'], case_name)
        t1 = os.path.join(case_folder, '%s_t1.nii.gz' % case_name)
        t1ce = os.path.join(case_folder, '%s_t1ce.nii.gz' % case_name)
        t2 = os.path.join(case_folder, '%s_t2.nii.gz' % case_name)
        flair = os.path.join(case_folder, '%s_flair.nii.gz' % case_name)
        input = {'t1': t1, 't1ce': t1ce, 't2': t2, 'flair': flair}
        print('\n(%d/%d) Compute the automatic segmentation for case %s...' % (i+1, len(case_names), case_name))
        t0 = time.time()
        segment_brats_21(
            config=config,
            data_config=data_config,
            model_paths_list=args.model,
            input_path_dict=input,
            save_folder=args.save_folder,
            zoom_aug=args.zoom,
            no_label_conversion=args.no_label_conversion,
        )
        print('\nInference done in %.2f sec\n' % (time.time() - t0))
        # MONAI saves the prediction at save_folder/patID/patID.nii.gz
        # but we want to save them at save_folder/patID.nii.gz for online evaluation
        # so we have to move the files around...
        monai_save_path = os.path.join(args.save_folder, '%s_flair' % case_name, '%s_flair.nii.gz' % case_name)
        save_path = os.path.join(args.save_folder, '%s.nii.gz' % case_name)
        os.system('cp %s %s' % (monai_save_path, save_path))
        os.system('rm -r %s' % os.path.dirname(monai_save_path))
    ave_time = int((time.time() - t_start) / len(case_names))
    print('\nAverage time per case: %dmin%dsec' % (ave_time / 60, ave_time % 60))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
