# Deep Residual Learning for Image Recognition

By Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

Microsoft Research Asia (MSRA).

### Table of Contents
0. [Introduction](#introduction)
0. [Disclaimer and Known Issues](#disclaimer-and-known-issues)
0. [Results](#results)
0. [Third-party Re-implementations](#third-party-re-implementations)

### Introduction

This repository contains the original models (ResNet-50, ResNet-101, and ResNet-152) described in the paper "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385). These models are those used in [ILSVRC] (http://image-net.org/challenges/LSVRC/2015/) and [COCO](http://mscoco.org/dataset/#detections-challenge2015) 2015 competitions, which won the 1st places in: ImageNet classification, ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

If you use these models in your research, please cite:

	@article{He2015,
		author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
		title = {Deep Residual Learning for Image Recognition},
		journal = {arXiv preprint arXiv:1512.03385},
		year = {2015}
	}

### Disclaimer and Known Issues

0. These models are converted from our own implementation to a recent version of Caffe. There might be numerical differences.
0. These models are for the usage of testing or fine-tuning.
0. These models were **not** trained using this version of Caffe.
0. If you want to train these models using this version of Caffe without modifications, please notice that:
	- GPU memory might be insufficient for extremely deep models.
	- Implementation of data augmentation might be different (see our paper about the data augmentation we used).
	- Changes of mini-batch size should impact accuracy (we use a mini-batch of 256 images on 8 GPUs, that is, 32 images per GPU).
0. In our BN layers, the provided mean and variance are strictly computed using average (**not** moving average) on a sufficiently large training batch after the training procedure. Using moving average might lead to different results.
0. We use Caffe's implementation of SGD: W := momentum\*W + lr\*g. **If you want to port these models to other libraries (e.g., Torch), please pay careful attention to the possibly different implementation of SGD**: W := momentum\*W + (1-momentum)\*lr\*g, which changes the effective learning rates.
	
### Results

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%
	
### Third-party Re-implementations

Deep residual networks are very easy to implement and train. We recommend to see also the following third-party re-implementations and extensions:

0. https://github.com/gcr/torch-residual-networks