# Retinal Blood Vessel Segmentation
This project focuses on the segmentation of retinal blood vessels using deep learning techniques, specifically the U-Net and FR-U-Net models. The primary objective is to accurately identify and segment blood vessels in retinal images, which is crucial for diagnosing various eye diseases.

## Task Overview
Retinal blood vessel segmentation is an essential task in medical image analysis. Accurate segmentation can aid in the early detection and treatment of conditions such as diabetic retinopathy, hypertension, and glaucoma. The goal of this project is to develop robust models that can effectively segment blood vessels from retinal fundus images.

## Dataset
The dataset used for this project is the DRIVE (Digital Retinal Images for Vessel Extraction) dataset. This dataset consists of a collection of retinal images along with corresponding ground truth segmentation masks, which provide a benchmark for evaluating segmentation performance. The DRIVE dataset is widely used in the research community for testing and comparing segmentation algorithms.

<img src="https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/drive%20image.png" alt="img" width="400"/>
<img src="https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/drive-mask.png" alt="mask" width="400"/>


Models
- [U-Net](https://arxiv.org/abs/1505.04597)

- [FR-Unet](https://ieeexplore.ieee.org/abstract/document/9815506)

![res](https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/unet-patches-finetuned-traindata.png)
