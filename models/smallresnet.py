import torch
import torch.nn as nn
from torch import Tensor

from loguru import logger

from configs import NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, NUM_CLASSES
from .utils import count_parameters


class ResidualBlock(nn.Module):
    """
    Standard residual block with two 3x3 convolutions.

    If input and output shapes differ (channels or stride > 1),
    a 1x1 shortcut convolution matches dimensions before the residual addition.

    Input shape:  (B, in_channels,  H,        W)
    Output shape: (B, out_channels, H/stride, W/stride)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.block(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """
    Lightweight ResNet for CIFAR-100 (~100K parameters).

    Architecture
    ────────────
    Stem:    3  →   8 ch,  32x32
    Stage 1: 8  →  16 ch,  32x32  (2 blocks, stride 1)
    Stage 2: 16 →  32 ch,  16x16  (2 blocks, stride 2 then stride 2)
    Stage 3: 32 →  64 ch,   8x8   (1 block,  stride 2)
    Head:    GlobalAvgPool → Dropout(0.3) → Linear(64, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        BASE = 8

        self.stem = nn.Sequential(
            nn.Conv2d(3, BASE, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(BASE),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(BASE, BASE * 2, stride=1),
            ResidualBlock(BASE * 2, BASE * 2, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(BASE * 2, BASE * 4, stride=2),
            ResidualBlock(BASE * 4, BASE * 4, stride=2),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(BASE * 4, BASE * 8, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(BASE * 8, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    @classmethod
    def init(cls, device, **kwargs):
        model = cls(**kwargs).to(device)
        criterion = lambda x, y: nn.functional.corss_entropy(x, y, label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
        
        warmup  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=5
        )
        cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - 5
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[5]
        )

        logger.info(f"Initiated {cls.__name__} with {count_parameters(model):,} params")

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

        start_epoch = checkpoint["epoch"] + 1
        best_test_acc = checkpoint["best_test_acc"]
        history = checkpoint["history"]

        logger.info(
            f"Restored {cls.__name__} from '{save_path}' "
            f"(epoch {start_epoch}, best acc {best_test_acc:.2f}%)"
        )

        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "history": history,
            "best_test_acc": best_test_acc,
            "start_epoch": start_epoch,
            "save_path": save_path,
            "resumed": True,
        }
