# TRABIT team for BraTS 2021 and FeTA 2021 using MONAI

Our code reproduces the segmentation performance of nnU-Net using MONAI.

## Installation with Docker (recommended)
Create the docker image ```brats21/trabit``` by running
```bash
sh build docker.sh
```
Your nvidia driver needs to be version 470.42 or higher.

You can then create a docker container named ```trabit``` using a command like
```bash
nvidia-docker run --ipc=host -it -v ~/workspace:/workspace -v ~/data:/data --name trabit brats21/trabit:latest
```
Feel free to change which folders you mount with ```-v ```.

## Installation (without Docker)
Install MONAI and all its dependencies with
```bash
python -m pip install 'monai[all]'==0.5.2
```

(Optional) To use the Generalized Wasserstein Dice loss you have to run in a terminal
```bash
python -m pip install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
```

(Optional) To use the AdamP or SGDP optimizers first install the adamp library using
 ```bash
python -m pip install adamp
```

(Optional) to use partially supervised learning (as we did for the FeTA 2021 challenge) you have to install
the label-set loss functions package
 ```bash
python -m pip install git+https://github.com/LucasFidon/label-set-loss-functions.git
```

## Instruction to use the code for BraTS (What you have to change)
You need to change the paths in /dataset_config/brats.yml

## Instruction for distributed training e.g. using GPU 0 3 5
`python3 -m torch.distributed.launch --nproc_per_node=3 run_train.py --cuda_visible_device 0 3 5 --config <config file> --data_config <data config file>`

## Instruction to extend the code to a new segmentation application (What you have to change)
For training:
* You need to create a dataset config file for your application in /dataset_config
* You need to define a data augmentation/preprocessing pipeline for your dataset in /data
* You need to define a dataloader pipeline for your dataset in /data
* You need to update data/factory to use your data augmentation pipeline and dataloader for your dataset

## BraTS inference
The weights of our model for the BraTS 2021 challenge can be downloaded at
https://drive.google.com/drive/folders/1b9rZDv9HHCxiEtiHz4h4I1vvjzScivpg?usp=sharing

Please put the weights in the folder ```\trained_weights``` and make sure the subfolders are named ```model_i``` for i integer from 1 to 7

## How to cite
If you find this repository useful for your research, please cite our work
* L. Fidon, S. Shit, I. Ezhov, J. C. Paetzold, S. Ourselin, T. Vercauteren.
[Generalized Wasserstein Dice Loss, Test-time Augmentation, and Transformers for the BraTS 2021 challenge.][brats2021]
BraTS 2021.
* L. Fidon, M. Aertsen, S. Shit, P. Demaerel, S. Ourselin, J. Deprest, T. Vercauteren.
[Partial supervision for the FeTA challenge 2021.][feta2021]
FeTA Challenge 2021.

BibTeX:
```
@article{fidon2021generalized,
  title={Generalized Wasserstein Dice Loss, Test-time Augmentation, and Transformers for the BraTS 2021 challenge},
  author={Fidon, Lucas and Shit, Suprosanna and Ezhov, Ivan and Paetzold, Johannes C and Ourselin, S{\'e}bastien and Vercauteren, Tom},
  journal={arXiv preprint arXiv:2112.13054},
  year={2021}
}
@article{fidon2021partial,
  title={Partial supervision for the FeTA challenge 2021},
  author={Fidon, Lucas and Aertsen, Michael and Shit, Suprosanna and Demaerel, Philippe and Ourselin, S{\'e}bastien and Deprest, Jan and Vercauteren, Tom},
  journal={arXiv preprint arXiv:2111.02408},
  year={2021}
}
```

[brats2021]: https://arxiv.org/abs/2112.13054
[feta2021]: https://arxiv.org/abs/2111.02408