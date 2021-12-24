# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
import yaml


def load_dataset_config(config_file=None, verbose=False):
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Dataset config file %s not found' % config_file)

    if verbose:
        print('\n*** Dataset config file')
        print(config_file)

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set the number of input channels
    config['info']['in_channels'] = len(config['info']['image_keys'])

    # Set the number of classes
    n_class = len(config['info']['labels'].keys())
    config['info']['n_class'] = n_class

    # Total number of classes (including superset; only for partial supervision)
    n_class_all = n_class
    if 'labels_superset' in config['info'].keys():
        n_class_all += len(config['info']['labels_superset'].keys())
    else:
        config['info']['labels_superset'] = None
    config['info']['n_class_all'] = n_class_all

    if verbose:
        print(config['log']['message'])

    return config


def load_brats_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "brats.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config

def load_brats21_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "brats21.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config

def load_covid19_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config


def load_fetal_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fetal_partialseg.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config

def load_feta_challenge_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fetal_challenge.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config
