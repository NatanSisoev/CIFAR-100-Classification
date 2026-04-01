import torchvision.transforms.v2 as transforms
from configs import INPUT_SIZE, MEAN, STD


def get_train_transforms(version: str = None):
    return transforms.Compose(
        [
            transforms.RandomCrop(INPUT_SIZE[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
