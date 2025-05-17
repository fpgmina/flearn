from collections import Counter
from typing import List

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import os
from torch.utils.data import Dataset

from utils.model_utils import save_checkpoint, iid_sharding, non_iid_sharding


class SimpleDataset(Dataset):
    def __init__(self, data: List[int], labels: List[int]):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.mark.parametrize("is_colab", [False])  # only test for local machine
def test_save_checkpoint_local(model_and_optimizer, is_colab):
    model, optimizer = model_and_optimizer

    # Mock os.path.exists to simulate the local environment (is_colab = False)
    mock_exists = MagicMock()
    mock_exists.return_value = (
        False  # Simulate local directory (it doesn't exist in Colab)
    )

    # Simulate saving to the checkpoint directory
    checkpoint_dir = "./checkpoints/"
    checkpoint_path = Path(checkpoint_dir) / "model_checkpoint.pth"

    # Mock the behavior of os.path.exists
    os.path.exists = mock_exists

    save_checkpoint(model, optimizer, epoch=1, loss=0.25)

    # Assert that the checkpoint file is saved to the correct directory
    assert checkpoint_path.exists(), f"Checkpoint was not saved to {checkpoint_path}"

    # Cleanup: Remove the checkpoint file after the test
    checkpoint_path.unlink()  # This deletes the file


def test_iid_sharding(binary_dataset):
    num_clients = 2
    shard_data = iid_sharding(binary_dataset, num_clients, seed=42)

    # Ensure correct number of clients
    assert (
        len(shard_data) == num_clients
    ), "Number of clients in the sharded data is incorrect"

    # Ensure each client has 5 data points
    for client_id, indices in shard_data.items():
        assert len(indices) == 5, f"Client {client_id} does not have 5 data points"

    # Ensure no data points are repeated and all are accounted for
    all_indices = sum(list(shard_data.values()), [])
    assert len(set(all_indices)) == len(
        binary_dataset
    ), "Some data points are missing or duplicated in the sharded data"


def test_non_iid_sharding(dataset):
    num_clients = 10
    num_classes = 5
    shard_data = non_iid_sharding(
        dataset, num_clients=num_clients, num_classes=num_classes
    )
    assert (
        len(shard_data) == num_clients
    ), "Number of clients in the sharded data is incorrect"
    for k, v in shard_data.items():
        assert (
            len(v) == len(dataset) / num_clients
        ), f"Client {k} does not have the correct number of samples"  # each client gets 400 samples
        count = Counter([dataset[i][1] for i in shard_data[k]])
        assert (
            len(count) == num_classes
        ), f"Client {k} has samples corresponding to more than {num_classes} classes"  # each client gets samples of exactly 5 classes


def test_iid_reproducibility(binary_dataset):
    num_clients = 2
    # Generate two shardings with the same seed
    shard_data_1 = iid_sharding(binary_dataset, num_clients, seed=42)
    shard_data_2 = iid_sharding(binary_dataset, num_clients, seed=42)

    # Assert that both shardings are the same (reproducibility)
    assert (
        shard_data_1 == shard_data_2
    ), "Sharding results are not reproducible with the same seed"


def test_non_iid_reproducibility(dataset):
    num_clients = 2
    num_classes = 2  # Each client will get 2 classes
    # Generate two shardings with the same seed
    shard_data_1 = non_iid_sharding(dataset, num_clients, num_classes, seed=42)
    shard_data_2 = non_iid_sharding(dataset, num_clients, num_classes, seed=42)

    # Assert that both shardings are the same (reproducibility)
    assert (
        shard_data_1 == shard_data_2
    ), "Sharding results are not reproducible with the same seed"
