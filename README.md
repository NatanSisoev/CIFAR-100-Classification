# CIFAR-100 Classification

## Overview

This project implements a lightweight Residual Network (ResNet-style) for image classification on the CIFAR-100 dataset using PyTorch. The goal is to build, train, and evaluate a compact convolutional neural network (~100k trainable parameters) while maintaining good performance and clean engineering practices.

---

## Project Structure

```
CIFAR-100-Classification/
│
├── data/
│   ├── datasets.py        # Dataset loading (CIFAR-100, custom datasets)
│   └── transforms.py      # Data augmentation / preprocessing (CutMix, normalization, etc.)
│
├── models/
│   └── resnet01.py        # first acrhitecture
│
├── training/
│   └── train01.py         # first acrhitecture
│
├── configs/
│   └── config.py          # Hyperparameters and constants
│
├── notebooks/
│   └── model01.ipynb      # first acrhitecture
│
├── results/               # plots with the results
│
├── requirements.txt
├── README.md
├── all_in_one.py          # All codebase without local imports
└── gather.py              # Gather all codebase (for colab initial cell)
```

---


## Author

Natan Sisoev

- natan.sisoev@gmail.com

