# CIFAR-100 Classification

## Overview

This project implements small neural networks for image classification on the CIFAR-100 dataset using PyTorch. The goal is to build, train, and evaluate a compact convolutional neural network (~100k trainable parameters) while maintaining good performance and clean engineering practices.

---

## Project Structure

```
CIFAR-100-Classification/
│
├── data/
│   ├── datasets.py        # Dataset loading (CIFAR-100, custom datasets)
│   ├── transforms.py      # Data augmentation / preprocessing (CutMix, normalization, etc.)
│   └── utils.py           # Dataset visualization
│
├── models/
│   ├── base.py            # Base abstract model with `init` and `restore` methods
│   ├── smallresnet.py     # Small Residual Network
│   ├── train.py           # Training routine
│   └── utils.py           # Model parameter count, result visualization and device
│
├── configs/
│   └── config.py          # Hyperparameters and constants
│
├── notebooks/
│   ├── dataset.py         # Dataset exploration
│   └── main.ipynb         # Main and final model training
│
├── requirements.txt
└── README.md
```

---

## Author

Natan Sisoev

- natan.sisoev@gmail.com

