import time

import torch.nn as nn
import torch
from loguru import logger
from torch.utils.data import DataLoader
from google.colab import drive

from configs import REMOTE_BASE_DIR
from models import count_parameters, plot_history


def train(
    *,
    device,
    model: nn.Module,
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 50,
    save_path: str = REMOTE_BASE_DIR + "best_model.pth",
    resumed: bool = False,
    **kwargs,
) -> tuple[nn.Module, dict]:
    drive.mount("/content/drive")

    n_params = count_parameters(model)

    if not resumed:
        logger.info(f"Training started - {n_params:,} trainable parameters")
        history = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
        }
        best_test_acc = 0.0
    else:
        logger.info(f"Training resumed - {n_params:,} trainable parameters")
        history = kwargs.get("history")
        best_test_acc = kwargs.get("best_test_acc")

    init_time = time.time()

    init_epochs = 0 if not resumed else kwargs.get("start_epoch")

    for epoch in range(init_epochs, num_epochs):
        for phase in ["train", "test"]:
            is_train = phase == "train"
            loader = dataloader_train if is_train else dataloader_test

            model.train() if is_train else model.eval()

            running_loss = 0
            correct = 0
            total = 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += preds.eq(labels).sum().item()
                total += images.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total * 100

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            logger.info(
                f"Epoch {epoch + 1:3d}/{num_epochs} | {phase:5s} | "
                f"loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%"
            )

            if not is_train and epoch_acc > best_test_acc:
                best_test_acc = epoch_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "t_max": scheduler.T_max,
                        "best_test_acc": best_test_acc,
                        "history": history,
                    },
                    save_path,
                )
                logger.info(
                    f" ★ New best test accuracy: {best_test_acc:.2f}% - model saved to '{save_path}'"
                )

        scheduler.step()

        plot_history(history)

    elapsed = time.time() - init_time
    logger.info(
        f"Training finished in {elapsed / 60:.1f} min | Best test acc: {best_test_acc:.2f}%"
    )

    return model, history
