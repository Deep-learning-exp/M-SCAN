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
To run the M-SCAN framework, you need to set up a Python environment with the following dependencies.
os
gc
sys
Pillow==10.0.0          # For handling images
opencv-python-headless  # For image processing
numpy                   # For numerical computations
pandas                  # For data manipulation
glob2                   # For file pattern matching
tqdm                    # For progress bars
matplotlib              # For plotting
scikit-learn            # For KFold cross-validation
pydicom                 # For DICOM file handling
torch                   # PyTorch framework for deep learning
torchvision             # For PyTorch vision utilities
timm                    # Pretrained vision models
transformers            # Hugging Face Transformers library
ultralytics             # YOLO models and utilities
natsort                 # For natural sorting of file names
albumentations          # For image augmentation
albumentations[imgaug]  # Extra augmentation features


## Results
The framework was trained and evaluated on 80% of the dataset for training and 20% for testing. The final results are as follows:

| Model                              | Accuracy | WCE Loss | AUROC |
|------------------------------------|----------|----------|-------|
| CNN (Sagittal only) + Concat       | 89.82%   | 0.392    | 0.851 |
| CNN (Sagittal only) + GRU          | 91.37%   | 0.336    | 0.863 |
| CNN (Multi-View) + GRU             | 92.60%   | 0.321    | 0.891 |
| **Multi-View Cross Attention (Ours)** | **93.80%**   | **0.282**    | **0.971** |

## Citation
If you find this work useful in your research, please cite the following paper:
```bibtex
@inproceedings{batra2024mscan,
  title={M-SCAN: A Multistage Framework for Lumbar Spinal Canal Stenosis Grading Using Multi-View Cross Attention},
  author={Batra, Arnesh and Gumber, Arush and Kumar, Anushk},
  booktitle={Proceedings of the Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2024}
}
