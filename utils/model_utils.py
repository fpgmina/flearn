from __future__ import annotations

import os
import torch
from collections import defaultdict, Counter
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data import Subset, DataLoader, Dataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_forward_pass(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, num_classes: int = 200
):
    device = get_device()
    sample_batch, _ = next(iter(dataloader))  # Get one batch
    sample_batch = sample_batch.to(device)

    output = model(sample_batch)
    assert output.shape == (dataloader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")


def get_subset_loader(
    dataloader: torch.utils.data.DataLoader, subset_size: int = 1000
) -> torch.utils.data.DataLoader:
    dataset_size = len(dataloader.dataset)  # type: ignore
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    return subset_loader


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filename: Optional[str] = None,
):

    # Check if running in Colab
    is_colab = os.path.exists("/content/drive")

    # If not running on Colab, use the local directory or default
    checkpoint_dir = (
        Path("/content/drive/MyDrive/checkpoints/")
        if is_colab
        else Path("./checkpoints/")
    )

    checkpoint_dir = Path(checkpoint_dir) or Path("/content/drive/MyDrive/checkpoints/")
    filename = filename or "model_checkpoint.pth"

    if is_colab:
        from google.colab import drive

        drive.mount("/content/drive")

    assert checkpoint_dir.exists()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_dir / filename)


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str
):
    assert Path(checkpoint_path).exists()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss


def iid_sharding(
    dataset: Dataset, num_clients: int, seed: Optional[int] = 42
) -> Dict[int, List[int]]:
    # Split the dataset into num_clients equal parts, each with samples from all classes
    data_len = len(dataset)  # type: ignore
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(data_len)
    client_data = defaultdict(list)

    for i in range(data_len):
        client_id = i % num_clients
        client_data[client_id].append(indices[i])

    return client_data


def non_iid_sharding(
    dataset: Dataset,
    num_clients: int,
    num_classes: int,
    seed: Optional[int] = 42,
) -> Dict[int, List[int]]:
    """
    Non-iid sharding with resampling:
    Each client gets `len(dataset) // num_clients` samples from `num_classes` classes, e.g.
    if I set num_classes=1 and num_clients=100 on a dataset with 40000 samples,
    each client will have 400 samples all belonging to a single class. Note that if there aren't enough
    samples in the class (e.g. class 0 has only 390 samples), sample with replacement.
    Classes can be assigned to multiple clients, and samples can be resampled.


    Returns:
        Dict[client_id, List[int]]
    """
    rng = np.random.default_rng(seed)

    # Group sample indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for cls in class_indices:
        rng.shuffle(class_indices[cls])

    all_classes = list(class_indices.keys())
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    samples_per_class = samples_per_client // num_classes

    client_data = {}

    for client_id in range(num_clients):
        # select classes for client_id
        selected_classes = rng.choice(all_classes, size=num_classes, replace=False)
        client_samples = []

        for cls in selected_classes:
            pool = class_indices[cls]
            # If not enough samples in class, sample with replacement
            if len(pool) >= samples_per_class:
                sampled = rng.choice(pool, size=samples_per_class, replace=False)
            else:
                sampled = rng.choice(pool, size=samples_per_class, replace=True)
            client_samples.extend(sampled)

        client_data[client_id] = client_samples

    return client_data


def count_labels_per_dataset(dataset: Dataset) -> Dict[int, int]:
    return dict(Counter((label for _, label in dataset)))


def _adapt_fisher_mask(
    mask_full: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Adapt a Fisher-based mask created for a pre-trained model
    to a new model that might have a different architecture (head).

    Args:
        mask_full (Dict[str, torch.Tensor]): Mask dictionary from the original model.
        model (nn.Module): New model with possibly different architecture.

    Returns:
        Dict[str, torch.Tensor]: Adapted mask for the new model.
    """
    adapted_mask = {
        name: (
            mask_full[name]
            if (name in mask_full and mask_full[name].shape == param.shape)
            else torch.ones_like(param, device=param.device)
        )
        for name, param in model.named_parameters()
    }
    return adapted_mask
