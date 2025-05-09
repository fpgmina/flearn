import copy
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.model_utils import get_device


# Loss function (averaged over N samples):
#
#     L(θ) = (1 / N) * ∑_{i=1}^{N} L^{(i)}(θ) where L^{(i)} := L(y^i, f_θ(x^i)) i.e. the loss on the i-th data point
#
# Gradient of the loss with respect to model parameters θ:
#
#     ∇L(θ) = (1 / N) * ∑_{i=1}^{N} ∇L^{(i)}(θ)
#
# Where:
#     - L^{(i)}(θ) is the loss on the i-th data point (x_i, y_i)
#     - ∇L^{(i)}(θ) = ∂L^{(i)} / ∂θ is the gradient of the loss for sample i
#     - N is the number of samples in the dataset or minibatch
#


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Approximate the diagonal of the Fisher Information Matrix (FIM) using squared gradients computed
    over mini-batches.

    The Fisher Information is given by the expected value of the squared gradient of the loss function:

        Fisher(θ) = E_{(x, y) ~ D} [ (∂L(f_θ(x), y) / ∂θ)^2 ]

    The sample approximation of the above expectation should be:

        Fisher(θ) ≈ (1 / N) * ∑_{i=1}^{N} (∇_θ L^{(i)}(θ))²

    Where:
        - L^{(i)}(θ) is the loss on the i-th data point
        - ∇_θ L^{(i)}(θ) is the gradient of the loss with respect to parameters θ
        - N is the number of samples (or mini-batches)
        - The square is element-wise and gives the diagonal approximation

    However, this function only computes an approximation of the Fisher diagonal and returns:

        F_diag ≈ (1/N) ∑_b (∇_θ L_b)²

    where:
        - b indexes batches from the dataset
        - N is the number of batches
        - L_b = (1 / |B|) ∑_{i∈B} ℓ(f_θ(x_i), y_i) is the **average loss over a mini-batch**
        - ℓ(·,·) is the per-sample loss (e.g., negative log-likelihood)
        - ∇_θ L_b is the gradient of the average batch loss with respect to the model parameters

    In other words, for computational simplicity, we square the gradient of the average loss over the mini batch,
    while in fact we should square the gradient of the loss over the individual samples.

    Notes:
    - This method does **not** compute per-sample gradients.
    - It squares the gradient of the averaged batch loss, which introduces bias.
    - It underestimates the true Fisher diagonal, as E[g]² < E[g²].
    - It's computationally efficient and works with any model

    Args:
        model (nn.Module): The model whose parameters are being analyzed.
        dataloader (DataLoader): DataLoader providing input–target pairs.
        loss_fn (nn.Module): Loss function used to compute gradients.
        num_batches (Optional[int]): If set, only the first `num_batches` are used
            to estimate the Fisher information. Useful for faster computation.

    Returns:
        fisher_diag (torch.Tensor): A flattened tensor containing the Fisher diagonal estimate,
        one element per parameter.
    """

    model.eval()
    device = get_device()
    model.to(device)

    fisher_diag = [torch.zeros_like(p, device=device) for p in model.parameters()]
    total_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)  # L_b = average over the batch

        # Approximation: we compute ∇_θ L_b, then square it,
        # which is not equal to the average of per-sample (∇_θ ℓ_i)^2
        loss.backward()

        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                fisher_diag[i] += p.grad.detach() ** 2

        total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches processed for Fisher approximation.")

    # Normalize by the number of batches (approximate Fisher expectation over data)
    fisher_diag = [f / total_batches for f in fisher_diag]

    # Flatten and concatenate all parameter diagonals into one tensor
    return torch.cat([f.flatten() for f in fisher_diag])


def create_fisher_mask(
    fisher_diag: torch.Tensor, model: nn.Module, sparsity: float = 0.9
) -> Dict[str, torch.Tensor]:
    """
    Create a dictionary of binary gradient masks based on Fisher importance scores.

    Keeps sparsity% of the most important parameters masking them to 0 and sets
    the rest to 1 (sets their gradients to zero during training).

    Args:
        fisher_diag (torch.Tensor): Flattened tensor of Fisher Information scores (1D).
        model (nn.Module): The model whose parameters will be masked.
        sparsity (float): Fraction of total parameters that are frozen (set mask=0).

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to binary masks
                                 with the same shape as the parameter tensors.
    """
    assert 0 < sparsity < 1, "sparsity needs to be between 0 and 1"

    k = int(len(fisher_diag) * sparsity)

    # Default: all parameters allowed to update
    flat_mask = torch.ones_like(fisher_diag)

    # Find top-k important indices to freeze
    # 1 for unimportant paramters, 0 for the important ones so that gradient does not update in SparseSGD

    if k > 0:
        important_indices = torch.topk(
            fisher_diag, k=k, largest=True
        ).indices  # top k scores
        flat_mask[important_indices] = 0.0

    param_sizes = [p.numel() for _, p in model.named_parameters()]
    param_shapes = [p.shape for _, p in model.named_parameters()]
    param_names = [name for name, _ in model.named_parameters()]
    split_masks = torch.split(flat_mask, param_sizes)

    return {
        name: mask.view(shape)
        for name, mask, shape in zip(param_names, split_masks, param_shapes)
    }


# Notes on apply_mask: Why a parameter being zero does NOT imply its gradient is zero
#
# Consider a simple linear model: y_hat = w1 * x1 + w2 * x2 and suppose we prune w1
# by setting it to zero: w1 = 0, w2 != 0
#
# In the forward pass:
#   y_hat = w1 * x1 + w2 * x2 = w2 * x2 (i.e. w1 does not contribute anything in the forward pass!)
#
# In the loss function (e.g., mean squared error):
#   L = (y - y_hat)^2
#
# Now, we compute the gradients using backpropagation.
#
# Despite w1 being zero, its gradient is:
#   ∂L/∂w1 = ∂L/∂y * ∂y/∂w1 = 2 * (y - y_hat) * x1
#
# => So, the gradient of w1 is NON-ZERO even though its value is zero.
#
# This happens because the gradient reflects how the loss would change if w1 changed —
# even if it's currently zero, the computation graph is still intact and tracks how
# sensitive the loss is to it.
#
# => Therefore: Simply setting a parameter to zero (e.g., param *= mask) is NOT enough
# to keep it frozen — the optimizer could still update it if its gradient is non-zero.
# You must ALSO zero the gradient before each optimizer step (e.g., param.grad *= mask)
# to truly "freeze" the parameter and prevent it from being updated.

# -------------------------------------------------------------------------------
# WHY ZEROING THE PARAMETER IS OFTEN "GOOD ENOUGH" for parameter sensitivity:
#
# In practice, even if we don’t zero the gradients explicitly, setting parameters to 0
# often *suppresses* their gradient magnitude over time.
#
# 1. Zeroed parameters contribute nothing to the output during the forward pass.
# 2. As a result, the loss becomes less sensitive to those parameters.
# 3. This leads to smaller gradients (or zero gradients) over time.
#
# So although it's not guaranteed, once zeroed, parameters tend to stay small or shrink further.
#
# => However, if strict sparsity is required (e.g., for efficiency or compression),
# it's safer to mask both parameter values and gradients explicitly.


def apply_mask(model: nn.Module, mask_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Return a new model with the given binary mask applied to its parameters (zeroed out).
    Zeroed out parameters naturally accumulate smaller Fisher scores, see comments above on apply_mask and
    comments on progressive pruning within progressive_mask_calibration.
    """
    model_copy = copy.deepcopy(model).to(next(model.parameters()).device)
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name])
    return model_copy


def progressive_mask_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    target_sparsity: float = 0.9,
    rounds: int = 5,
    warn_tolerance: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """
    Progressively create a gradient mask using Fisher info, applying pruning at each round.

    This method is based on:
        "The Lottery Ticket Hypothesis: finding sparse, trainable neural networks" by Frankle and Carbin.
         https://arxiv.org/abs/1803.03635.

    Note: Emits a warning if the final sparsity deviates significantly from the target.

    Args:
        model (nn.Module): Model to prune.
        dataloader (DataLoader): DataLoader (not used in dummy logic).
        loss_fn (nn.Module): Loss function (not used in dummy logic).
        target_sparsity (float): Target sparsity at the end of pruning.
        rounds (int): Number of pruning rounds.
        warn_tolerance (float): Relative deviation tolerance to trigger a warning.
    """
    device = get_device()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    grad_mask = {
        name: torch.ones_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    sparsity_targets = np.geomspace(0.1, target_sparsity, rounds)

    for r, sparsity in enumerate(sparsity_targets):
        # current_sparsity = target_sparsity * r / rounds
        print(f"Current sparsity in round {r}: {sparsity}")
        # Apply current mask to model
        masked_model = apply_mask(model, grad_mask)

        # Recompute Fisher based on the masked model
        fisher_diag = compute_fisher_diagonal(masked_model, dataloader, loss_fn)

        # Create new mask (0 = freeze, 1 = allow update)
        new_mask = create_fisher_mask(fisher_diag, model, sparsity=sparsity)
        print(f"zeros in mask: {sum((v==0).sum().item() for v in new_mask.values())}")

        # Progressive pruning.
        # Update cumulative mask (once frozen, always frozen) i.e. once a parameter has been set to zero because it's
        # important it's going to stay set to zero. If it is unimportant in the old mask (grad_mask=1) and important in the
        # new_mask (new_mask=0), update it so that grad_mask=0, thus gradually increasing sparsity.
        grad_mask = {name: grad_mask[name] * new_mask[name] for name in grad_mask}
        masked = sum((v == 0).sum().item() for v in grad_mask.values())
        print(f"[Round {r}] Masked: {masked} / {total_params}")

    # Final sparsity check and warning
    masked_params = sum((v == 0).sum().item() for v in grad_mask.values())
    actual_sparsity = masked_params / total_params
    rel_error = abs(actual_sparsity - target_sparsity) / target_sparsity

    if rel_error > warn_tolerance:
        warnings.warn(
            f"[WARNING] Final sparsity {actual_sparsity:.4f} deviates from target {target_sparsity:.4f} "
            f"by {rel_error:.2%} (> {warn_tolerance:.2%} relative tolerance)."
        )
    return grad_mask


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
