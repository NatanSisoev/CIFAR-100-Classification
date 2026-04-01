from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_test_transforms
from configs import BATCH_SIZE
from loguru import logger


def get_cifar100_loaders():
    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=get_test_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.success("Succesfully loaded train loader.")
    logger.success("Succesfully loaded test loader.")

    return train_loader, test_loader
