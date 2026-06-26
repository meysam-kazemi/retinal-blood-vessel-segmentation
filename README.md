# Retinal Blood Vessel Segmentation

This project focuses on the segmentation of retinal blood vessels using deep learning techniques, specifically the U-Net and FR-U-Net models. The primary objective is to accurately identify and segment blood vessels in retinal images, which is crucial for diagnosing various eye diseases.

## Task Overview
Retinal blood vessel segmentation is an essential task in medical image analysis. Accurate segmentation can aid in the early detection and treatment of conditions such as diabetic retinopathy, hypertension, and glaucoma. The goal of this project is to develop robust models that can effectively segment blood vessels from retinal fundus images.

## Dataset
The dataset used for this project is the **DRIVE** (Digital Retinal Images for Vessel Extraction) dataset. It consists of retinal images along with corresponding ground-truth segmentation masks, which provide a benchmark for evaluating segmentation performance. DRIVE is widely used in the research community for testing and comparing segmentation algorithms.

The dataset is not included in this repository. Download it from the [DRIVE challenge page](https://drive.grand-challenge.org/) and arrange it as follows (this layout matches `.gitignore`):

```
data/
└── Drive/
    ├── train-png/
    │   ├── image/
    │   └── mask/
    └── test-png/
        ├── image/
        └── mask/
```

<img src="https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/drive%20image.png" alt="img" width="400"/>
<img src="https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/drive-mask.png" alt="mask" width="400"/>

## Models
- [U-Net](https://arxiv.org/abs/1505.04597)
- [FR-U-Net](https://ieeexplore.ieee.org/abstract/document/9815506)

## Installation

```bash
git clone https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation.git
cd retinal-blood-vessel-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training / experiments
The training pipelines, patch extraction, augmentation, and evaluation live in the
[`notebooks/`](notebooks) folder. The most relevant entry points are:

| Notebook | Purpose |
|----------|---------|
| `notebooks/unet.ipynb` | Train the baseline U-Net on full images |
| `notebooks/unet_patches.ipynb` | Train U-Net on image patches |
| `notebooks/fr_unet.ipynb` | Train the FR-U-Net model |
| `notebooks/patches_data.ipynb` | Build the patch dataset |
| `notebooks/augment.ipynb` | Data augmentation |
| `notebooks/for-test.ipynb` | Evaluation / metrics |

The shared U-Net architecture is defined once in [`src/model.py`](src/model.py) so it can be
imported from both the notebooks and the demo app.

### Demo app
After training, run the Gradio demo to segment vessels from an uploaded fundus image:

```bash
python app.py --weights models/unet.pth
```

This launches a local web UI where you can upload a retinal image and view the predicted
vessel mask. The model runs on GPU if available, otherwise CPU.

## Results

![res](https://github.com/meysam-kazemi/retinal-blood-vessel-segmentation/blob/main/figs/unet-patches-finetuned-traindata.png)

> Metrics (Dice / F1, AUC, accuracy) are computed in `notebooks/for-test.ipynb`.
> Update the table below with your latest numbers:

| Model | Dice / F1 | AUC | Accuracy |
|-------|-----------|-----|----------|
| U-Net | – | – | – |
| FR-U-Net | – | – | – |

## License
This project is licensed under the [MIT License](LICENSE).
