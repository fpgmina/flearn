import os
import torch
from collections import defaultdict
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
    Split the dataset into non-i.i.d. shards.

    Each client receives samples from exactly `num_classes` classes.

    Args:
        dataset (Dataset): Dataset to be split.
        num_clients (int): Number of clients.
        num_classes (int): Number of classes per client.
        seed (Optional[int]): Random seed.

    Returns:
        Dict[int, List[int]]: Mapping client ID to list of sample indices.
    """
    client_data = defaultdict(list)
    class_indices = defaultdict(list)

    # Group sample indices by class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    all_classes = list(class_indices.keys())
    total_classes = len(all_classes)

    if num_classes > total_classes:
        raise ValueError(
            f"Requested {num_classes} classes per client, "
            f"but dataset only has {total_classes} classes."
        )

    if num_clients * num_classes > total_classes * len(class_indices[0]):
        print("Warning: There may be overlapping class assignments among clients.")

    rng = np.random.default_rng(seed)

    # Shuffle class list for randomness
    rng.shuffle(all_classes)

    # Assign classes to clients
    class_pool = all_classes.copy()
    for client_id in range(num_clients):
        if len(class_pool) < num_classes:
            class_pool = all_classes.copy()
            rng.shuffle(class_pool)
        selected_classes = class_pool[:num_classes]
        class_pool = class_pool[num_classes:]

        # Assign samples from selected classes
        for cls in selected_classes:
            samples = class_indices[cls]
            rng.shuffle(samples)
            client_data[client_id].extend(samples)

    return dict(client_data)
