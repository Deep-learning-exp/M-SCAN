# M-SCAN: A Multistage Framework for Lumbar Spinal Canal Stenosis Grading Using Multi-View Cross Attention

## Introduction
Lumbar spinal canal stenosis (SCS) is a prevalent condition, particularly among the elderly, leading to significant challenges in diagnosis and treatment. This repository contains the implementation of **M-SCAN**, a novel deep-learning framework designed to fully automate the grading of lumbar spinal canal stenosis using MRI images.

## Framework Overview
M-SCAN introduces a multistage approach that leverages both Sagittal and Axial MRI images through a sequence-based architecture and multi-view cross-attention mechanism. The framework has demonstrated state-of-the-art performance, achieving high accuracy in predicting the severity of SCS across multiple spinal levels.

### Key Features
- **Multistage Approach:** The framework employs a three-stage design:
  - **Stage One:** A U-Net-based model identifies SCS points on Sagittal images, projecting them into 3D space to locate corresponding Axial slices.
  - **Stage Two:** Pre-trained CNN models classify cropped Sagittal and Axial slices based on SCS severity.
  - **Stage Three:** A multi-view cross-attention model integrates the outputs from the previous stages, providing accurate grading for each spinal level.

- **Multi-View Cross Attention:** The architecture includes separate processing paths for Sagittal and Axial images, with cross-attention mechanisms to enhance feature extraction and alignment across views.

- **High Performance:** The framework achieves a predictive accuracy of 93.80% and an AUROC of 0.971, outperforming existing benchmarks.

## Dataset
The framework is trained on the RSNA 2024 Lumbar Spine Degenerative dataset, comprising 1,975 studies labeled across five intervertebral disc levels (L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1). Each study includes Sagittal and Axial MRI slices, categorized into Normal/Mild, Moderate, or Severe SCS.

## Installation

To run the M-SCAN framework, you need to set up a Python environment with the following dependencies:

- `os`: Standard Python library for interacting with the operating system.
- `gc`: Standard Python library for garbage collection.
- `sys`: Standard Python library for system-specific parameters and functions.
- `Pillow==10.0.0`: For handling images.
- `opencv-python-headless`: For image processing without GUI dependencies.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `glob2`: For file pattern matching and file handling.
- `tqdm`: For progress bars and progress tracking.
- `matplotlib`: For creating static, animated, and interactive visualizations.
- `scikit-learn`: For machine learning utilities, including KFold cross-validation.
- `pydicom`: For handling DICOM files, commonly used in medical imaging.
- `torch`: PyTorch framework for deep learning.
- `torchvision`: Provides datasets, model architectures, and image transformations for computer vision.
- `timm`: Pretrained vision models and utilities.
- `transformers`: Hugging Face's library for state-of-the-art natural language processing models.
- `ultralytics`: YOLO models and utilities for object detection.
- `natsort`: For natural sorting of file names.
- `albumentations`: Fast image augmentation library.

To install these dependencies, you can use the following command:

## Results
The framework was trained and evaluated on 80% of the dataset for training and 20% for testing. The final results are as follows:

| Model                              | Accuracy | WCE Loss | AUROC |
|------------------------------------|----------|----------|-------|
| CNN (Sagittal only) + Concat       | 89.82%   | 0.392    | 0.851 |
| CNN (Sagittal only) + GRU          | 91.37%   | 0.336    | 0.863 |
| CNN (Multi-View) + GRU             | 92.60%   | 0.321    | 0.891 |
| **Multi-View Cross Attention (Ours)** | **93.80%**   | **0.282**    | **0.971** |


