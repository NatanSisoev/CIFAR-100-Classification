import time

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from .utils import plot_history
from configs import NUM_CLASSES

from models.utils import count_parameters


def train(
    *,
    device,
    model: nn.Module,
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    best_save_path: str,
    last_save_path: str,
    grad_clip: float = 1.0,
    resumed: bool = False,
    **kwargs,
) -> tuple[nn.Module, dict]:

    n_params = count_parameters(model)
    logger.info(
        f"Training {'resumed' if resumed else 'started'} — {n_params:,} parameters"
    )

    if not resumed:
        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        best_test_acc = 0.0
        init_epoch = 0
    else:
        history = kwargs["history"]
        best_test_acc = kwargs["best_test_acc"]
        init_epoch = kwargs["start_epoch"]

    cutmix = transforms.CutMix(num_classes=NUM_CLASSES)
    mixup = transforms.MixUp(num_classes=NUM_CLASSES)
    aug = transforms.RandomChoice([cutmix, mixup], [1/5, 1/5])

    init_time = time.time()

    for epoch in range(init_epoch, num_epochs):
        for phase in ("train", "test"):
            is_train = phase == "train"
            loader = dataloader_train if is_train else dataloader_test

            model.train() if is_train else model.eval()

            running_loss = correct = total = 0.0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    if is_train and kwargs.get("data_augmentation", False):
                        images, labels = aug(images, labels)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if is_train:
                        loss.backward()
                        if grad_clip is not None:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                if labels.ndim == 1:
                    correct += preds.eq(labels).sum().item()
                else:
                    correct += (labels[torch.arange(labels.size(0)), preds] > 0).sum().item()
                total += images.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total * 100

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            logger.info(
                f"Epoch {epoch + 1:3d}/{num_epochs} | {phase:5s} | "
                f"loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%"
            )

            if not is_train:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_test_acc": best_test_acc,
                        "history": history,
                    },
                    last_save_path,
                )
                logger.info(f"Accuracy: {epoch_acc:.2f}% — saved to '{last_save_path}'")

                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_test_acc": best_test_acc,
                            "history": history,
                        },
                        best_save_path,
                    )
                    logger.info(
                        f" ★ New best: {best_test_acc:.2f}% — saved to '{best_save_path}'"
                    )

        scheduler.step()
        plot_history(history)

    elapsed = time.time() - init_time
    logger.info(f"Finished in {elapsed / 60:.1f} min | Best: {best_test_acc:.2f}%")

    return model, history
