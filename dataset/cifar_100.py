from typing import List, Tuple, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform


def get_val_test_tranform():
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform


def get_cifar_datasets(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    rng = np.random.default_rng(seed=seed)
    train_transform = get_train_transform()
    val_test_transform = get_val_test_tranform()

    data_root = "/content/data"

    full_train_dataset = CIFAR100(root=data_root, train=True, download=True)
    test_dataset = CIFAR100(root=data_root, train=False, transform=val_test_transform)

    # Create train/val split indices
    len_train = len(full_train_dataset)
    indices = list(range(len_train))
    split = int(0.8 * len_train)

    rng.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = CIFAR100(root=data_root, train=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_root, train=True, transform=val_test_transform)

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    dataset: Dataset,
    indices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return a DataLoader for a given dataset, optionally using a subset of the data.

    Args:
        dataset (Dataset): The dataset to be loaded.
        indices (List[int], optional): A list of indices to create a subset of the dataset. If None, the entire dataset is used. Defaults to None.
        batch_size (int, optional): The number of samples per batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
            NB Should be set to False for test/val dataset and to True for train datasets as in general we do not want
            any testing procedure to be non-deterministic.

    Returns:
        DataLoader: A DataLoader for the given dataset, potentially using a subset of the data.
    """
    batch_size = batch_size or 32  # if batch_size is None, use 32
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_cifar_dataloaders(batch_size=None):
    trainset, valset, _ = get_cifar_datasets()
    train_dataloader, val_dataloader = get_dataloader(
        trainset, batch_size=batch_size
    ), get_dataloader(valset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
