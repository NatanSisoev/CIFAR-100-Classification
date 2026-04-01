import torch
import torch.nn as nn
from loguru import logger

from configs import NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY
from .utils import count_parameters


class BaseModel(nn.Module):

    @classmethod
    def init(cls, device, **kwargs):
        model = cls(**kwargs).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS
        )

        logger.info(
            f"Initiated {cls.__name__} with {count_parameters(model):,} parameters"
        )

        return model, criterion, optimizer, scheduler

    @classmethod
    def restore(cls, device, save_path, **kwargs):
        checkpoint = torch.load(save_path, map_location=device)

        model = cls(**kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=checkpoint["t_max"]
        )

        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        start_epoch   = checkpoint["epoch"] + 1
        best_test_acc = checkpoint["best_test_acc"]
        history       = checkpoint["history"]

        logger.info(
            f"Restored {cls.__name__} from '{save_path}' "
            f"(epoch {start_epoch}, best acc {best_test_acc:.2f}%)"
        )

        return {
            "model":         model,
            "criterion":     criterion,
            "optimizer":     optimizer,
            "scheduler":     scheduler,
            "history":       history,
            "best_test_acc": best_test_acc,
            "start_epoch":   start_epoch,
            "save_path":     save_path,
            "resumed":       True,
        }
