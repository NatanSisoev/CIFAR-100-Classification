import torchvision.transforms as transforms
from configs import INPUT_SIZE, MEAN, STD


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomCrop(INPUT_SIZE[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
