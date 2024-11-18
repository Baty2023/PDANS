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

## Data Preparation
Please download [ [PU1K](https://github.com/guochengqian/PU-GCN) ] and [ [PUGAN](https://github.com/liruihui/PU-GAN) ]. 
```
# For generating test data, please see **PDANS-main/pointnet2/dataloder/prepare_dataset.py**
cd PDANS-main/pointnet2/dataloder

# For example 1, we can generate 4x test set of PUGAN:
python prepare_dataset.py --input_pts_num 2048 --R 4 --mesh_dir mesh_dir --save_dir save_dir

# For example 2, we can generate 4x test set of PUGAN with 0.1 Gaussion noise:
python prepare_dataset.py --input_pts_num 2048 --R 4 --noise_level 0.1 --noise_type gaussian --mesh_dir mesh_dir --save_dir save_dir
```


## Quick Start
### Example
We provide some examples. There examples are in the **PDANS-main/pointnet2/example** folder. The results are in the **PDANS-main/pointnet2/test/example** folder.
```bash
# For example, we can run 30 steps (DDIM) to generate 4x point cloud on PU1K with the pre-trained model of PUGAN.
# We provide the function (bin2xyz) of converting *.bin to *.xyz in **PDANS-main/pointnet2/dataloder/dataset_utils.py**.
cd PDANS-main/pointnet2
python example_samples.py --dataset PUGAN --R 4 --step 30 --example_file ./example/.xyz
