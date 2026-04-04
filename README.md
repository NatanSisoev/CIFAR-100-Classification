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

You can train the model using the main notebook:

1. Open `notebooks/train_smallresnet.ipynb`.
2. Ensure the required packages are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Set training parameters if needed (epochs, batch size, learning rate).
4. Run the notebook cells sequentially to:
    - Load CIFAR-100
    - Apply data augmentation
    - Train SmallResNet
    - Save the trained model and plots
5. Evaluate on the test set to see metrics and Top-k accuracies.

---

## Author

Natan Sisoev

- natan.sisoev@gmail.com

