import torch.nn as nn
from configs import BASE_DIR

from matlplotlib import pyplot as plt
from loguru import logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(
    history: dict, save_path: str = BASE_DIR + "training_history.png"
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train", linewidth=2)
    ax1.plot(history["test_loss"], label="Test", linewidth=2)
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history["train_acc"], label="Train", linewidth=2)
    ax2.plot(history["test_acc"], label="Test", linewidth=2)
    ax2.set_title("Top-1 Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("CIFAR-100 SmallResNet Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    logger.info(f"Training plot saved to '{save_path}'")
