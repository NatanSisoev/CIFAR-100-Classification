import torch.nn as nn
from torch import Tensor
import torch
from .utils import count_parameters
from loguru import logger
from configs import NUM_EPOCHS, BASE_DIR


class ResidualBlock(nn.Module):
    """
    Standard residual block with two 3x3 convolutions.

    If the input and output shapes differ (different channels or stride > 1),
    a 1x1 "shortcut" convolution is used to match dimensions before the addition.

    Input shape:  (B, in_channels, H,   W)
    Output shape: (B, out_channels, H/stride, W/stride)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # MAIN: Conv -> batch normalization -> ReLU → Conv → batch normalization
        self.block = nn.Sequential(
            # First conv: may reduce spatial resolution via stride
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),  # bias=False because BN has its own bias
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv: same spatial size
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

        # RESIDUAL CONNECTION
        if stride != 1 or in_channels != out_channels:
            # 1x1 kernel: project input to output size
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # identity: free skip, no parameters
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.block(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """
    Lightweight ResNet.

    Architecture
    ────────────
    Stem:    3  →  BASE channels, BASExBASE
    Stage 1: BASE →  BASE channels, BASExBASE  (2 blocks, stride 1)
    Stage 2: BASE →  64 channels, 16x16  (2 blocks, stride 2)
    Stage 3: 64 → 128 channels,  8x8   (2 blocks, stride 2)
    Head:    GlobalAvgPool(1x1) → Dropout(0.3) → Linear(128, 100)
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        BASE = 8

        self.stem = nn.Sequential(
            nn.Conv2d(3, BASE, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(BASE),
            nn.ReLU(inplace=True),
        )

        # Residual stages
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

        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 128, 8, 8) → (B, 128, 1, 1)
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
        x = self.stem(x)  # (B,  3, 32, 32) → (B,  32, 32, 32)
        x = self.stage1(x)  #                 → (B,  32, 32, 32)
        x = self.stage2(x)  #                 → (B,  64, 16, 16)
        x = self.stage3(x)  #                 → (B, 128,  8,  8)
        x = self.pool(x)  #                 → (B, 128,  1,  1)
        x = torch.flatten(x, 1)  #                 → (B, 128)
        x = self.dropout(x)
        x = self.fc(x)  #                 → (B, 100)
        return x


def init_smallresnet(device):
    net = SmallResNet().to(device)
    n = count_parameters(net)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    logger.info(f"{n:,} parameters")

    return net, criterion, optimizer, scheduler


def resume_smallresnet(
    num_classes,
    device,
    num_epochs: int = NUM_EPOCHS,
    save_path: str = BASE_DIR + "best_model.pth",
):
    checkpoint = torch.load(save_path, map_location=device)

    net = SmallResNet(num_classes=num_classes).to(device)
    net.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
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

    return {
        "model": net,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "num_epochs": num_epochs,
        "history": history,
        "best_test_acc": best_test_acc,
        "start_epoch": start_epoch,
        "save_path": save_path,
        "resumed": True,
    }
