# PDANS

This repo contains the project code for the paper **Point Cloud Upsampling Using Conditional Diffusion Module with Adaptive Noise Suppression** and is intended solely for reviewers to reference the code while evaluating the paper.

## The Overall Framework 
<img src="assets/image2.pdf" alt="pdans" width="900"/> 

## Overview
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)


## Installation

### Requirements
The following environment is recommended for running **_PDANS_** (an NVIDIA 4080 GPU):
- Ubuntu: 22.04 and above
- CUDA: 11.8 and above
- PyTorch: 2.0.0 and above
- python: 3.8 and above

### Environment

- Base environment
```
conda create -n pdans python=3.8 -y
conda activate pdans

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install cuda -c nvidia/label/cuda-11.8.0

pip install -r requirements.txt

# For installing pytorch3d, please follow:
1. download pytorch3d-0.7.4-py38_cu118_pyt200.tar.bz2 from https://anaconda.org/pytorch3d/pytorch3d/files?page=6

2. conda install pytorch3d-0.7.4-py38_cu118_pyt200.tar.bz2

# compile C++ extension packages
sh compile.sh
```

Some code is still being refined, please wait.
