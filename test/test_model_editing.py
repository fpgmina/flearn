import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.model_editing import (
    compute_fisher_diagonal,
    create_fisher_mask,
    progressive_mask_calibration,
    Mask,
    PruningType,
)
from utils.model_utils import get_device, _adapt_fisher_mask


@pytest.fixture
def sample_mask_dict():
    return {
        "layer1.weight": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "layer2.bias": torch.tensor([0.0, 1.0, 1.0]),
    }


@pytest.fixture
def mask(sample_mask_dict):
    return Mask(mask_dict=sample_mask_dict)


def test_mask_getitem(mask, sample_mask_dict):
    assert torch.equal(mask["layer1.weight"], sample_mask_dict["layer1.weight"])
    assert torch.equal(mask["layer2.bias"], sample_mask_dict["layer2.bias"])


def test_mask_get(mask, sample_mask_dict):
    assert torch.equal(mask.get("layer1.weight"), sample_mask_dict["layer1.weight"])
    assert mask.get("nonexistent", default=None) is None


def test_mask_contains(mask):
    assert "layer1.weight" in mask
    assert "layer2.bias" in mask
    assert "nonexistent" not in mask


def test_mask_iter(mask, sample_mask_dict):
    keys = list(iter(mask))
    expected_keys = list(sample_mask_dict.keys())
    assert set(keys) == set(expected_keys)


def test_mask_len(mask, sample_mask_dict):
    assert len(mask) == len(sample_mask_dict)


def test_mask_keys(mask, sample_mask_dict):
    assert set(mask.keys()) == set(sample_mask_dict.keys())


def test_mask_values(mask, sample_mask_dict):
    for val1, val2 in zip(mask.values(), sample_mask_dict.values()):
        assert torch.equal(val1, val2)


def test_mask_items(mask, sample_mask_dict):
    for (k1, v1), (k2, v2) in zip(mask.items(), sample_mask_dict.items()):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_mask_sparsity(tiny_mlp):
    model = tiny_mlp
    device = get_device()
    ones_mask_dict = {
        name: torch.ones_like(param, device=device)
        for name, param in model.named_parameters()
    }
    mask = Mask(mask_dict=ones_mask_dict)
    assert np.allclose(mask.sparsity, 0.0)


def test_mask_update():
    # Create two example masks for a fake model with two parameters
    mask_a = {
        "layer1.weight": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "layer2.bias": torch.tensor([1.0, 0.0]),
    }

    mask_b = {
        "layer1.weight": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        "layer2.bias": torch.tensor([0.0, 1.0]),
    }

    # Expected output = element-wise product (logical AND on 1s)
    expected_mask = {
        "layer1.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "layer2.bias": torch.tensor([0.0, 0.0]),
    }

    m1 = Mask(mask_dict=mask_a)
    m2 = Mask(mask_dict=mask_b)
    updated = m1.update(m2)

    # Check the masks match
    for name in expected_mask:
        assert torch.equal(
            updated.mask_dict[name], expected_mask[name]
        ), f"Mismatch in {name}: expected {expected_mask[name]}, got {updated.mask_dict[name]}"


@pytest.mark.parametrize(
    "pruning_type", [PruningType.FISHER, PruningType.HESSIAN_PARAM_SQUARED]
)
def test_compute_fisher_diagonal(tiny_cnn, pruning_type):
    # Create dummy dataset (batch of 8, 1x8x8 images, 10 classes)
    X = torch.randn(8, 1, 8, 8)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)

    model = tiny_cnn
    loss_fn = nn.CrossEntropyLoss()

    fisher_diag = compute_fisher_diagonal(
        model, dataloader, loss_fn, num_batches=2, pruning_type=pruning_type
    )

    total_params = sum(p.numel() for p in model.parameters())  # 1490
    assert isinstance(fisher_diag, torch.Tensor)
    assert fisher_diag.ndim == 1
    assert fisher_diag.numel() == total_params
    assert (fisher_diag >= 0).all(), "Fisher scores must be non-negative"


@pytest.mark.parametrize("sparsity", [0.2, 0.4, 0.6, 0.7, 0.85, 0.9])
def test_create_fisher_mask_shapes_and_counts(tiny_mlp, sparsity):
    model = tiny_mlp
    total_params = sum(p.numel() for p in model.parameters())

    # Fake Fisher vector with increasing importance
    fisher_diag = torch.linspace(0, 1, steps=total_params)

    masks = create_fisher_mask(fisher_diag, model, sparsity=sparsity)

    # 1. Check correct number of masks
    param_names = [name for name, _ in model.named_parameters()]
    assert set(masks.keys()) == set(param_names), "Mask keys must match parameter names"

    # 2. Check that each mask shape matches its corresponding parameter
    for name, param in model.named_parameters():
        assert name in masks, f"Missing mask for parameter: {name}"
        assert masks[name].shape == param.shape, f"Shape mismatch for {name}"

    # 3. Check total number of ones in all masks equals expected count
    total_ones = sum(mask.sum().item() for mask in masks.values())
    expected_ones = total_params - round(total_params * sparsity)
    assert (
        total_ones == expected_ones
    ), f"Expected {expected_ones} ones, got {total_ones}"


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


@pytest.mark.parametrize("target_sparsity", [0.9, 0.8])
def test_progressive_mask_calibration(tiny_mlp, dummy_dataloader, target_sparsity):
    model = tiny_mlp
    loss_fn = nn.CrossEntropyLoss()
    dataloader = dummy_dataloader

    try:
        mask = progressive_mask_calibration(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            target_sparsity=target_sparsity,
            rounds=10,
            warn_tolerance=0.01,
        )
    except RuntimeError as e:
        pytest.fail(f"progressive_mask_calibration raised an unexpected error: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    masked_params = sum((v == 0).sum().item() for v in mask.values())
    actual_sparsity = masked_params / total_params

    print(
        f"[Test] Final sparsity: {actual_sparsity:.4f} vs Target: {target_sparsity:.4f}"
    )
    assert (
        abs(actual_sparsity - target_sparsity) / target_sparsity <= 0.01
    ), "Final sparsity deviates beyond allowed tolerance"
