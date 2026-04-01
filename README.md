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
│   └── resnet01.py        # SmallResNet and ResidualBlock definitions
│
├── training/
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation utilities
│   └── utils.py           # Metrics, helpers
│
├── configs/
│   └── config.py          # Hyperparameters and constants
│
├── notebooks/
│   └── experiments.ipynb  # Optional experimentation
│
├── requirements.txt
├── README.md
└── main.py                # Entry point
```

---

## Installation

```bash
git clone <repo-url>
cd CIFAR-100-Classification
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python main.py
```

---


## Author

Natan Sisoev

- natan.sisoev@gmail.com

