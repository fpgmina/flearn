import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.model_editing import (
    compute_fisher_diagonal,
    create_fisher_mask,
    _adapt_fisher_mask,
)


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

    # Keep top 20% (i.e., freeze top 20%, unfreeze 80%)
    keep_ratio = 0.2
    masks = create_fisher_mask(fisher_diag, model, keep_ratio=keep_ratio)

    # 1. Check correct number of masks
    param_names = [name for name, _ in model.named_parameters()]
    assert set(masks.keys()) == set(param_names), "Mask keys must match parameter names"

    # 2. Check that each mask shape matches its corresponding parameter
    for name, param in model.named_parameters():
        assert name in masks, f"Missing mask for parameter: {name}"
        assert masks[name].shape == param.shape, f"Shape mismatch for {name}"

    # 3. Check total number of *zeros* matches expected keep count
    total_zeros = sum((mask == 0).sum().item() for mask in masks.values())
    expected_zeros = int(total_params * keep_ratio)
    assert (
        total_zeros == expected_zeros
    ), f"Expected {expected_zeros} zeros (frozen), got {total_zeros}"


def test_adapt_fisher_mask(tiny_mlp):
    old_model = tiny_mlp
    mask_full = {
        name: torch.ones_like(param) for name, param in old_model.named_parameters()
    }

    # Create new model (tiny_mlp from fixture) and modify head slightly
    new_model = tiny_mlp

    # Let's simulate a model with a different head: reinitialize last Linear
    new_model.net[-1] = nn.Linear(20, 3)  # Change output dimension from 5 -> 3

    # Adapt the mask
    adapted_mask = _adapt_fisher_mask(mask_full, new_model)

    # Check that all parameter names are present
    assert set(adapted_mask.keys()) == set(
        name for name, _ in new_model.named_parameters()
    )

    # Backbone mask should match shape
    assert adapted_mask["net.0.weight"].shape == old_model.net[0].weight.shape
    assert adapted_mask["net.0.bias"].shape == old_model.net[0].bias.shape

    # Head mask must be newly created (ones), not copied (shape mismatch expected)
    assert adapted_mask["net.2.weight"].shape == new_model.net[2].weight.shape
    assert adapted_mask["net.2.bias"].shape == new_model.net[2].bias.shape

    # Check that the head masks are all ones
    assert torch.all(adapted_mask["net.2.weight"] == 1.0)
    assert torch.all(adapted_mask["net.2.bias"] == 1.0)
