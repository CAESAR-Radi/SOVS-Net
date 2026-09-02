# 🛰️ SOVS-Net

**SOVS-Net: Open-Vocabulary Semantic Segmentation for SAR Imagery**

## 📢 News

- **[Revision Update]** A revised version of the manuscript has been submitted in response to the latest reviewer comments. (September 2026)
- **[Planned Release]** The official implementation of **SOVS-Net** and the **SAR-OVSeg** benchmark will be made publicly available in this repository upon paper acceptance.
  
> 🚧 Note: This repository is currently under construction. The complete implementation and benchmark resources have been prepared for release and will be made publicly available upon paper acceptance.

## 🧭 Overview

SOVS-Net is an open-vocabulary semantic segmentation framework tailored for Synthetic Aperture Radar (SAR) imagery.  
By representing semantic categories using natural language, SOVS-Net enables pixel-level land-cover parsing under text-defined label spaces, overcoming the limitations of conventional closed-set SAR segmentation.

The framework follows an **encoding–alignment–decoding** paradigm and is specifically designed to address the modality gap between SAR backscattering characteristics and natural language semantics.

## 🧠 Key Characteristics

- 🔹 Pixel-level open-vocabulary semantic segmentation for SAR imagery  
- 🔹 Cross-modal alignment between SAR visual features and textual semantics  
- 🔹 Robust spatial decoding under complex scattering and noise conditions  
- 🔹 Flexible adaptation to unseen datasets and dynamic label configurations  

## 📊 SAR-OVSeg Benchmark

This project also introduces **SAR-OVSeg**, an open-vocabulary SAR semantic segmentation benchmark constructed by harmonizing multiple public SAR datasets into a unified land-cover taxonomy.

- 🔹 Over **50,000** SAR image–label pairs  
- 🔹 Multi-sensor, multi-resolution, and multi-region coverage  
- 🔹 Pronounced long-tailed land-cover category distribution  
- 🔹 Designed for open-vocabulary and cross-dataset evaluation  

## 📈 Experimental Scope

SOVS-Net is evaluated under diverse and challenging settings, including:

- 🧩 Open-vocabulary SAR semantic segmentation  
- 🌍 Cross-dataset generalization to unseen SAR datasets  
- 🗺️ Large-scale whole-scene SAR mapping  
- 📝 Text-driven semantic querying with flexible prompts  

## 📦 Planned Release

Upon acceptance of the paper, we will release the following:

- 📁 Full SOVS-Net training and inference code  
- 📊 The SAR-OVSeg benchmark
- ⚙️ Configuration files and scripts for reproduction  
