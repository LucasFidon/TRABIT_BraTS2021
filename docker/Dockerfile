# @brief  Basic MONAI image
# @author Lucas Fidon (lucas.fidon@gmail.com).
# @date   August 2021.

# Image info
#FROM nvcr.io/nvidia/pytorch:21.07-py3
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="lucas.fidon@gmail.com"
LABEL version="0.1"
LABEL description="Docker for MONAI."

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    aptitude sudo vim zsh wget curl git net-tools \
    rxvt-unicode tmux byobu thunar locales build-essential cmake \
    python3-pip
RUN echo "alias python=python3" >> ~/.bash_aliases

# Remove unnecesary packages
RUN DEBIAN_FRONTEND=noninteractive apt remove -y --purge gdm3 xserver-xephyr

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install Pytorch
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install MONAI
RUN pip3 install 'monai[all]'==0.5.2 tensorboardX pyyaml

# Install optional Libraries
RUN pip3 install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
RUN pip3 install adamp

WORKDIR /

# copy necessary files
COPY src /src
COPY config/brats_config.yml /config/brats_config.yml
COPY config/loader.py /config/loader.py
COPY dataset_config/brats21_docker.yml /dataset_config/brats21_docker.yml
COPY dataset_config/loader.py /dataset_config/loader.py
COPY run_inference_brats21.py /run_inference_brats21.py
COPY trained_weights /trained_weights

# ==================================================================
# RUN CODE
# ------------------------------------------------------------------

ENTRYPOINT [ "python3","-u", "run_inference_brats21.py", "--config", "config/brats_config.yml", "--data_config", "dataset_config/brats21_docker.yml", "--zoom", "--save_folder", "/output/", "--model", "trained_weights/model_1/checkpoint_epoch=500.pt", "trained_weights/model_2/checkpoint_epoch=500.pt", "trained_weights/model_3/checkpoint_epoch=500.pt", "trained_weights/model_4/checkpoint_epoch=500.pt", "trained_weights/model_5/checkpoint_epoch=500.pt", "trained_weights/model_6/checkpoint_epoch=500.pt", "trained_weights/model_7/checkpoint_epoch=500.pt"]