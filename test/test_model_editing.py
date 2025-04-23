import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.model_editing import compute_fisher_diagonal, create_fisher_mask


def test_compute_fisher_diagonal(tiny_cnn):
    # Create dummy dataset (batch of 8, 1x8x8 images, 10 classes)
    X = torch.randn(8, 1, 8, 8)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)

    model = tiny_cnn
    loss_fn = nn.CrossEntropyLoss()

    fisher_diag = compute_fisher_diagonal(model, dataloader, loss_fn, num_batches=2)

    total_params = sum(p.numel() for p in model.parameters())  # 1490
    assert isinstance(fisher_diag, torch.Tensor)
    assert fisher_diag.ndim == 1
    assert fisher_diag.numel() == total_params
    assert (fisher_diag >= 0).all(), "Fisher scores must be non-negative"


def test_create_fisher_mask_shapes_and_counts(tiny_mlp):
    model = tiny_mlp
    total_params = sum(p.numel() for p in model.parameters())

    # Fake Fisher vector with increasing importance
    fisher_diag = torch.linspace(0, 1, steps=total_params)

    # Keep top 20%
    keep_ratio = 0.2
    masks = create_fisher_mask(fisher_diag, model, keep_ratio=keep_ratio)

    # 1. Check correct number of masks
    param_names = [name for name, _ in model.named_parameters()]
    assert set(masks.keys()) == set(param_names), "Mask keys must match parameter names"

    # 2. Check that each mask shape matches its corresponding parameter
    for name, param in model.named_parameters():
        assert name in masks, f"Missing mask for parameter: {name}"
        assert masks[name].shape == param.shape, f"Shape mismatch for {name}"

    # 3. Check total number of ones in all masks equals expected count
    total_ones = sum(mask.sum().item() for mask in masks.values())
    expected_ones = int(total_params * keep_ratio)
    assert (
        total_ones == expected_ones
    ), f"Expected {expected_ones} ones, got {total_ones}"
