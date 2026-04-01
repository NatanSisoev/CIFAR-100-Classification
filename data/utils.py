import numpy as np
from matplotlib import pyplot as plt
from configs import STD, MEAN


def visualize_images(
    trainset,
    imgs: tuple[int, int] = (3, 5),
    show_label: bool = True,
    seed: int = None,
    category: int | str | None = None,
):
    rng = np.random.default_rng(seed)
    classes = trainset.classes

    rows, cols = imgs
    total = rows * cols

    if category is None:
        valid_indices = np.arange(len(trainset))
    else:
        if isinstance(category, str):
            if category not in classes:
                raise ValueError(f"Category '{category}' not found.")
            category = classes.index(category)

        targets = np.array(trainset.targets)
        valid_indices = np.where(targets == category)[0]

        if len(valid_indices) == 0:
            raise ValueError("No images found for this category.")

    plt.figure(figsize=(4 * cols, 3 * rows))

    for i in range(total):
        plt.subplot(rows, cols, i + 1)

        idx = rng.choice(valid_indices)
        image, label = trainset[idx]

        img = image.permute(1, 2, 0).numpy()
        img = img * STD + MEAN
        img = np.clip(img, 0, 1)

        plt.imshow(img)

        if show_label:
            plt.title(f"{idx} - {classes[label]}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()
