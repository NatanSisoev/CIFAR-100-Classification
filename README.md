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

## References

- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv. https://arxiv.org/abs/1704.04861
- Hu, J., Shen, L., Albanie, S., Sun, G., & Wu, E. (2017). Squeeze-and-Excitation Networks. arXiv. https://arxiv.org/abs/1709.01507
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. arXiv. https://arxiv.org/abs/1807.06521
- Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization strategy to train strong classifiers with localizable features. arXiv. https://arxiv.org/abs/1905.04899
- Müller, S. G., & Hutter, F. (2021). TrivialAugment: Tuning-free yet state-of-the-art data augmentation. arXiv. https://arxiv.org/abs/2103.10158

---

## Author

Natan Sisoev

- natan.sisoev@gmail.com

