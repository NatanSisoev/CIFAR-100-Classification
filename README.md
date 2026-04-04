# CIFAR-100 Classification

## Overview

This project implements small neural networks for image classification on the CIFAR-100 dataset using PyTorch. The goal is to build, train, and evaluate a compact convolutional neural network (~100k trainable parameters) while maintaining good performance and clean engineering practices.

---

## Project Structure

```
CIFAR-100-Classification/
│
├── artifacts/                             # Saved pre-trained models
│   ├── plots/                             # Evaluation and history plots
│   │   ├── history_data_augmentation.png  # Training with data augmentation plot
│   │   ├── history_final.png              # Training final 300 epochs plot
│   │   ├── history_past_cosine.png        # Training past CosineAnnealingLR's T_max plot
│   │   └── top_k.png                      # Top k accuracy vs k final model plot
│   ├── smallresnet_300.py                 # SmallResNet after 300 epochs
│   └── smallresnet_best.py                # best SmallResNet after 300 epochs
│
├── configs/
│   └── config.py                          # Hyperparameters and constants
│
├── data/
│   ├── datasets.py                        # Dataset loading (CIFAR-100 train and test)
│   ├── transforms.py                      # Data augmentation / preprocessing
│   └── utils.py                           # Dataset visualization
│
├── docs/
│   ├── src/                               # Source LaTeX code for the report
│   └── report.pdf                         # Project report (in catalan)
│
├── models/
│   ├── smallresnet.py                     # Small Residual Network
│   ├── train.py                           # Training routine
│   └── utils.py                           # Model parameter count, result visualization and device
│
├── notebooks/
│   ├── final.py                           # Final model evaluation
│   ├── main_executed.py                   # Training notebook with outputs (40MB)
│   └── main.ipynb                         # Training notebook (for Modal)
│
├── README.md
└── requirements.txt
```

---

## Training

You can train the model using the main notebook `notebooks/main.ipynb`.

It is designed such that it can be ran in cloud services (like the one I used, Modal, or Google Colab), clonning this exact repository to get all the source code. It also stores all the artifacts there (to use Google Colab, setup the drive mount).

To run locally just remove the first cell and adjust the paths in `configs/config.py`.

---

## Author

Natan Sisoev

- natan.sisoev@gmail.com

