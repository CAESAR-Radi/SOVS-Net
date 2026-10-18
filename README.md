# SOVS-Net: Open-Vocabulary Segmentation for SAR Imagery

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange.svg)](https://pytorch.org/)

## 🚀 Overview

**SOVS-Net** is an open-vocabulary semantic segmentation framework designed for Synthetic Aperture Radar (SAR) imagery.  
It aims to enable pixel-level semantic understanding of SAR scenes using natural-language category definitions.

SOVS-Net follows an **encoding–alignment–decoding** paradigm and introduces SAR-oriented vision-language alignment and decoding components to address the modality gap between SAR backscattering characteristics and textual semantics.

The repository also provides the **SAR-OVSeg** benchmark, a unified dataset for SAR open-vocabulary semantic segmentation.

---

## ✨ Key Features

- **SAR-oriented open-vocabulary segmentation framework**  
  Implements SOVS-Net, an encoding-alignment-decoding framework for SAR OVSS with natural-language category definitions.

- **SAR-OVSeg benchmark**  
  Provides a standardized SAR OVSS benchmark with over 50,000 image-label pairs from multiple sources, covering diverse resolutions, geographic regions, and representative land-cover categories.

---

## 📂 Project Structure

```text
SOVS-Net/
├── configs/
│   ├── config.yaml
│   └── convnextB_768.yaml
├── datasets/
│   └── SAR-OVSeg_release/
│       ├── train/
│       │   ├── images/
│       │   └── annotations/
│       └── test/
│           ├── images/
│           └── annotations/
├── sovs/
│   ├── data/
│   ├── modeling/
│   ├── third_party/
│   └── utils/
├── train_net.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠 Installation

### Environment Requirements

- Python >= 3.8
- PyTorch >= 1.13
- A PyTorch-compatible version of torchvision and Detectron2
- GDAL for reading SAR GeoTIFF data

### Install Dependencies

Please install a CUDA-compatible PyTorch/torchvision pair and Detectron2
according to their official installation instructions before running the code:

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

GDAL is required but is not installed through `requirements.txt`.
Installing it with Conda is recommended:

```bash
conda install -c conda-forge "gdal>=3.6,<4.0"
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

Please download the SAR-OVSeg dataset from the following link:

- **SAR-OVSeg Dataset:** [🔗 Baidu Netdisk](https://pan.baidu.com/s/1oFJHUm1vVyyYtlHu1HJZTw?pwd=7ghx) (Extraction code: `7ghx`)

After downloading, place the dataset under the `datasets/` directory:

```text
datasets/
└── SAR-OVSeg_release/
    ├── train/
    │   ├── images/
    │   └── annotations/
    └── test/
        ├── images/
        └── annotations/
```

The dataset contains 11 semantic categories:

```text
water, building, road, cropland, forest, shrubland, grassland,
wetland, bareland, snow and ice, airport
```

The annotation TIFF files use label IDs from `0` to `10` for the 11 semantic categories, respectively, and `255` is used as the ignore label.

---

## 💾 Pretrained Weights

Please download the pretrained weights and tokenizer files from the following link:

- **SARCLIP ConvNeXt-B Weights:** [🔗 Baidu Netdisk](https://pan.baidu.com/s/1rZxkOIU1fB52L9NOheHDgQ?pwd=ntmr) (Extraction code: `ntmr`)

Please place the pretrained files under the `pretrain/` directory.

Expected structure:

```text
pretrain/
├── convnextb_model.pth
├── vocab.json
└── merges.txt
```

---

## 🚩 Quick Start

### Training

```bash
python train_net.py \
  --config-file configs/convnextB_768.yaml \
  --num-gpus 1 \
  --dist-url auto \
  OUTPUT_DIR output/
```

### Evaluation

```bash
python train_net.py \
  --config-file configs/convnextB_768.yaml \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR output/eval \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  MODEL.WEIGHTS path/to/model_checkpoint.pth
```

---

## ⚙️ Configuration Notes

The main configuration file is:

```text
configs/convnextB_768.yaml
```

Some configuration fields are inherited from the original Detectron2/SED training pipeline and are retained for compatibility.  
The main active settings for SOVS-Net include:

```yaml
MODEL:
  META_ARCHITECTURE: "SOVSNet"
  BACKBONE:
    NAME: "CLIP"
  SEM_SEG_HEAD:
    NAME: "SOVSNetHead"
    NUM_CLASSES: 11
    IGNORE_VALUE: 255
```

The data mapper is adapted for SAR float/TIF inputs and reads SAR images using GDAL.

---

## 📌 Codebase and Acknowledgements

This repository is built upon and adapted from the official implementation of **SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**, which is licensed under the Apache License 2.0.

We sincerely thank the authors of the following projects for their excellent open-source contributions:

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MaskFormer](https://github.com/facebookresearch/MaskFormer)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SED](https://github.com/xb534/SED)

---

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).

Parts of the code are adapted from third-party open-source projects.  
Please refer to the license and attribution notices in the corresponding files.

---

## 📚 Citation

If you find this repository useful, please cite our work:

```bibtex
@misc{sovsnet2026,
  title  = {SOVS-Net: Open-Vocabulary Segmentation for SAR Imagery via Cross-Modal Alignment and Hierarchical Semantic Reasoning},
  author = {CAESAR-Radi},
  year   = {2026},
  url    = {https://github.com/CAESAR-Radi/SOVS-Net}
}
```

The official citation will be updated after publication.

---
