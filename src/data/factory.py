from src.data.brats_dataset import get_brats_dataset
from src.data.brats21_dataset import get_brats21_dataset
from src.data.covid19_dataset import get_covid19_dataset
from src.data.fetal_dataset import get_fetal_dataset
from src.data.brats_transform_pipelines import brats_inference_transform
from src.data.covid19_transform_pipelines import covid19_inference_transform
from src.data.fetal_transform_pipelines import fetal_inference_transform
from src.data.single_case_dataloader import single_case_dataloader

SUPPORTED_DATASET = [
    'BraTS2021',
    'FeTAChallenge',
]

SUPPORTED_DATA_AUGMENTATION_PIPELINES = [
    'nnUNet',
]

def get_dataset(config, data_config, seed, mode='split', use_persistent_dataset=False):
    """
    Typically used for training / validation.
    :param config:
    :param data_config:
    :param mode:
    :return:
    """
    dataset_name = data_config['name']

    # Check the dataset name
    assert dataset_name in SUPPORTED_DATASET, \
        'Found dataset %s. But only %s are supported for training.' % \
        (dataset_name, str(SUPPORTED_DATASET))

    # Check data augmentation hyper-parameters
    data_aug_pipeline = config['data']['data_augmentation']
    if not data_aug_pipeline in SUPPORTED_DATA_AUGMENTATION_PIPELINES:
        raise ArgumentError(
            'Data augmentation %s is not supported. Please choose one in %s' % \
            (data_aug_pipeline, SUPPORTED_DATA_AUGMENTATION_PIPELINES)
        )

    # Return the dataloader for the dataset
    if dataset_name == 'BraTS2021':
        return get_brats21_dataset(config, data_config, seed, mode, use_persistent_dataset)
    elif dataset_name == 'FeTAChallenge':
        return get_fetal_dataset(config, data_config, seed, mode, use_persistent_dataset)
    else:
        raise NotImplementedError('Unknown dataset %s' % dataset_name)


def get_single_case_dataloader(config, data_config, input_path_dict):
    """
    Typically used for inference.
    :param config:
    :param data_config:
    :param input_path_dict:
    :return:
    """
    dataset_name = data_config['name']

    # Check the dataset name
    assert dataset_name in SUPPORTED_DATASET, \
        'Found dataset %s. But only %s are supported for inference.' % \
        (dataset_name, str(SUPPORTED_DATASET))

    # Get the inference transform pipeline for the dataset
    if dataset_name == 'BraTS2021':
        inference_pipeline = brats_inference_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
        )
    elif dataset_name == 'FeTAChallenge':
        inference_pipeline = fetal_inference_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
        )
    else:
        inference_pipeline = None
        NotImplementedError('Unknown dataset for inference transform: %s' % dataset_name)

    # Create the dataloader
    dataloader = single_case_dataloader(
        inference_transform=inference_pipeline,
        input_path_dict=input_path_dict,
    )

    return dataloader


