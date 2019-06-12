# Collaborative Generative Adversarial Networks for Missing MR contrast imputation

### An implementation of "Which Contrast Does Matter? Towards a Deep Understanding of MR Contrast using Collaborative GAN", arXiv:1905.04105

The main codes have two parts: one is Collaborative Generative Adversarial Networks for MR contrast imputation problem and the other is brain tumor segmentation network.
The Collaborative GAN is a deep learning model for missing image data imputation (Dongwook Lee et al. CVPR 2019. oral). The concept for the missing image imputation is applied for MR contrast problem and this is the implementation of that using tensorflow.
The segmentation code is the modified version of 3D MRI brain tumor segmentation using autoencoder regularization(Andriy Myronenko, 2018, arXiv:1810.11654) due to the limited GPU memory issue.

This repository provides a tensorflow implementation of CollaGAN for missing MR contrast imputation as described in the paper:
> Which Contrast Does Matter? Towards a Deep Understanding of MR Contrast using Collaborative GAN,
> Dongwook Lee, Won-Jin Moon, Jong Chul Ye (arXiv:1905.04105)
> [[Paper]](https://arxiv.org/pdf/1905.04105.pdf)

## OS
The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:
Linux: Ubuntu 16.04

## Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
tensorflow 		  1.10.1
tqdm			  4.28.1
numpy			  1.14.5
scipy			  1.1.0
argparse		  1.1
logging 	 	  0.5.1.2
ipdb 			  0.11
cv2 			  3.4.3
```
## Datasets
Dataset for Multimodal brain tumor segmentation challenge
BRATS2015 (https://www.smir.ch/BRATS/Start2015)

## Main train files
```
train.py
seg_train.py
```
These files are handled by the `scripts/train_CollaGAN_BRATS.sh` and `scripts/train_Segmentation_BRATS.sh` files with following commands:
```
sh scripts/train_CollaGAN_BRATS.sh
sh scripts/train_Segmentation_BRATS.sh
```

## Input and output options
The explanation of the input and output options for CollaGAN model and Segmentation model for training are introduced in following files, respectively:
```
options/colla_options.py
options/seg_option.py
```

