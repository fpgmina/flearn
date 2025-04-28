import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.model_editing import create_fisher_mask
from optim.ssgd import SparseSGDM


def test_sparse_sgdm_step(tiny_mlp):
    data = torch.randn(32, 10)
    labels = torch.randint(0, 5, (32,))
    dataloader = DataLoader(TensorDataset(data, labels), batch_size=8)
    loss_fn = nn.CrossEntropyLoss()
    model = tiny_mlp

    # Create fake Fisher vector
    total_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.linspace(0, 1, steps=total_params)

    # Create binary masks
    mask = create_fisher_mask(fisher_diag, model, keep_ratio=0.2)

    # named_params = {name: param for name, param in model.named_parameters()}
    named_params = dict(model.named_parameters())

    optimizer = SparseSGDM(
        model.parameters(), named_params=named_params, grad_mask=mask, lr=0.01
    )

    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Clone grads before step
        grads_before = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        optimizer.step()

        # Verify that masked gradients were zeroed
        for name, param in model.named_parameters():
            if param.grad is not None and name in mask:
                masked_grad = grads_before[name] * mask[name]
                actual_grad = param.grad  # After masking step
                assert torch.allclose(
                    actual_grad, masked_grad
                ), f"Mask not correctly applied to {name}"


def test_sparse_sgdm_respects_mask_in_parameter_updates(tiny_mlp):
    data = torch.randn(16, 10)
    labels = torch.randint(0, 5, (16,))
    dataloader = DataLoader(TensorDataset(data, labels), batch_size=8)
    loss_fn = nn.CrossEntropyLoss()
    model = tiny_mlp

    # Create fake Fisher vector
    total_params = sum(p.numel() for p in model.parameters())
    fisher_diag = torch.linspace(0, 1, steps=total_params)

    # Create a mask that keeps only 20% of parameters trainable
    mask = create_fisher_mask(fisher_diag, model, keep_ratio=0.2)
    named_params = dict(model.named_parameters())

    optimizer = SparseSGDM(
        model.parameters(),
        named_params=named_params,
        grad_mask=mask,
        lr=0.1,  # slightly bigger lr to amplify changes
    )

    model.train()
    inputs, targets = next(iter(dataloader))

    # Save a copy of original parameters
    original_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    # Now check:
    #  - non-frozen parameters should have changed
    #  - frozen parameters should not have changed
    for name, param in model.named_parameters():
        if name in mask:
            frozen_mask = mask[name] == 0
            trainable_mask = mask[name] == 1

            if frozen_mask.any():
                # Frozen part: should not change
                assert torch.allclose(
                    param.detach()[frozen_mask],
                    original_params[name][frozen_mask],
                    atol=1e-6,
                ), f"Frozen part of {name} changed!"

            if trainable_mask.any():
                # Trainable part: should have changed
                assert not torch.allclose(
                    param.detach()[trainable_mask],
                    original_params[name][trainable_mask],
                    atol=1e-5,
                ), f"Trainable part of {name} did not change!"
