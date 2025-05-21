from __future__ import annotations

import attr
import logging

import numpy as np
import torch
from werkzeug.utils import cached_property
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union, Iterable, Tuple, List
from utils.model_utils import get_device


@attr.s(frozen=True)
class Mask:
    """
    Container for a parameter mask dictionary used in pruning or sparse training. Wraps Dict[str, torch.Tensor].

    Each mask is a tensor with binary values:
      - `1` indicates a trainable (unmasked) parameter
      - `0` indicates a frozen (masked out) parameter

    The class provides utilities for validation, analysis, and application of the mask.

    Example:
    >>> mask = Mask(mask_dict=some_dict)
    >>> mask.validate_against(model)
    >>> print(mask.sparsity)
    >>> print(mask.per_layer_sparsity)
    >>> torch.save(mask.state_dict(), "mask.pt")
    >>> loaded = Mask.load_state_dict(torch.load("mask.pt"))
    >>> param_mask = mask["layer1.weight"]
    """

    mask_dict: Dict[str, torch.Tensor] = attr.ib(kw_only=True)

    def __attrs_post_init__(self):
        assert isinstance(
            self.mask_dict, dict
        ), f"mask must be of type: Dict[str, torch.Tensor] and not of type:{type(self.mask_dict)}"

        # Validation: all masks must be binary (0 or 1), float or bool, and on the same device
        devices = set()
        for name, tensor in self.mask_dict.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"Mask for '{name}' is not a tensor.")
            if tensor.dtype not in (torch.float32, torch.bool):
                raise TypeError(
                    f"Mask for '{name}' must be float32 or bool, got {tensor.dtype}."
                )
            if not ((tensor == 0) | (tensor == 1)).all():
                raise ValueError(f"Mask for '{name}' must be binary (0 or 1).")
            devices.add(tensor.device)

        if len(devices) > 1:
            raise ValueError(
                f"All mask tensors must be on the same device. Found: {devices}"
            )

    @property
    def device(self) -> torch.device:
        # device of the first mask tensor (already checked they all are on the same device)
        return next(iter(self.mask_dict.values())).device

    @cached_property
    def num_total_parameters(self) -> int:
        return sum(t.numel() for t in self.mask_dict.values())

    @cached_property
    def num_zeroed_parameters(self) -> int:
        return sum((t == 0).sum().item() for t in self.mask_dict.values())

    @property
    def sparsity(self) -> float:
        """
        Compute total sparsity (fraction of zeros) across all parameters.
        """
        return self.num_zeroed_parameters / self.num_total_parameters

    @property
    def per_layer_sparsity(self) -> Dict[str, float]:
        return {
            name: (mask == 0).sum().item() / mask.numel()
            for name, mask in self.mask_dict.items()
        }

    def update(self, other: Mask) -> Mask:
        """
        Return new MaskDict with parameters frozen if frozen in either mask (logical AND on 1s).
        """
        assert isinstance(other, Mask)
        return Mask(
            mask_dict={
                k: self.mask_dict[k] * other.mask_dict[k]
                for k in self.mask_dict
                if k in other.mask_dict
            }
        )

    def to(self, device: torch.device) -> Mask:
        """
        Move all mask tensors to specified device.
        """
        return Mask(mask_dict={k: v.to(device) for k, v in self.mask_dict.items()})

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # saveable version of the mask dictionary.
        return self.mask_dict

    @classmethod
    def load_state_dict(cls, state: Dict[str, torch.Tensor]) -> Mask:
        return cls(mask_dict=state)

    def unmask_layers(self, layers: Union[str, List[str]]) -> Mask:
        """
        Set the masks for the specified layers to 1, meaning that all parameters
        of these layers are unmasked and trainable.

        Args:
            layers (Union[str, List[str]]): The name(s) of the layer(s) whose mask should be set to 1.

        Returns:
            Mask: A new Mask object with the specified layers' masks updated to 1.
        """
        # Ensure layers is a list, even if it's a single string
        if isinstance(layers, str):
            layers = [layers]

        # Check that each layer exists in the mask
        for layer_name in layers:
            if layer_name not in self.mask_dict:
                raise ValueError(f"Layer '{layer_name}' not found in the mask.")

        # Create a new mask where the specified layers' masks are set to all 1's
        updated_mask_dict = self.mask_dict.copy()
        for layer_name in layers:
            updated_mask_dict[layer_name] = torch.ones_like(self.mask_dict[layer_name], dtype=torch.float32)

        return Mask(mask_dict=updated_mask_dict)

    def validate_against(
        self,
        named_params: Union[
            Iterable[Tuple[str, nn.Parameter]], Dict[str, nn.Parameter]
        ],
    ) -> None:
        """
        Validate mask against model.named_parameters().
        Check that:
          - All parameter names in the model exactly match those in the mask.
          - All mask shapes match model parameter shapes.
          - The model parameters and the mask tensors are on the same device.

        Raises:
            AssertionError/ValueError if any of the above conditions fail.
        """
        if not isinstance(named_params, dict):
            named_params = dict(named_params)

        model_param_names = set(named_params.keys())
        mask_param_names = set(self.mask_dict.keys())

        # Check key match
        assert model_param_names == mask_param_names, (
            f"Mismatch between model parameters and mask keys.\n"
            f"Missing in mask: {model_param_names - mask_param_names}\n"
            f"Extra in mask: {mask_param_names - model_param_names}"
        )

        # Check shape match
        for name in model_param_names:
            model_shape = named_params[name].shape
            mask_shape = self.mask_dict[name].shape
            if model_shape != mask_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': mask shape {mask_shape}, model shape {model_shape}"
                )

        param_device = next(p.device for p in named_params.values() if p.requires_grad)

        # Check mask is on the same device as the model
        if self.device != param_device:
            raise ValueError(
                f"Device mismatch: model parameters on {param_device}, "
                f"mask on {self.device}. Please call `mask.to({param_device})` before passing it in."
            )

    # Dictionary-style interface
    def __getitem__(self, name: str) -> torch.Tensor:
        return self.mask_dict[name]

    def __iter__(self):
        return iter(self.mask_dict)

    def __len__(self) -> int:
        return len(self.mask_dict)

    def keys(self):
        return self.mask_dict.keys()

    def values(self):
        return self.mask_dict.values()

    def items(self):
        return self.mask_dict.items()

    def get(self, name: str, default=None) -> torch.Tensor:
        return self.mask_dict.get(name, default)


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


# Notes on compute_fisher_diagonal: Why a parameter being zero does NOT imply its gradient is zero
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


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_batches: Optional[int] = None,
    mask: Optional[Mask] = None,
) -> torch.Tensor:
    """
    Approximate the diagonal of the Fisher Information Matrix (FIM) using squared gradients computed
    over mini-batches. If `mask_dict` is provided, gradients of pruned parameters will be zeroed out
    before accumulation, preserving previously frozen weights.

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
        mask (Optional[Mask]): If provided, compute gradients only for parameters that are not frozen.

    Returns:
        fisher_diag (torch.Tensor): A flattened tensor containing the Fisher diagonal estimate,
        one element per parameter.
    """
    if mask is not None:
        assert isinstance(mask, Mask)

    model.eval()
    device = get_device()
    model.to(device)
    if mask is not None:
        mask.to(device)
        mask.validate_against(model.named_parameters())

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

        for i, (name, p) in enumerate(model.named_parameters()):
            if p.grad is not None:
                grad = p.grad.detach()
                if mask is not None and name in mask:
                    grad = grad * mask[name]
                fisher_diag[i] += grad**2

        total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches processed for Fisher approximation.")

    # Normalize by the number of batches (approximate Fisher expectation over data)
    fisher_diag = [f / total_batches for f in fisher_diag]

    # Flatten and concatenate all parameter diagonals into one tensor
    return torch.cat([f.flatten() for f in fisher_diag])


def create_fisher_mask(
    fisher_diag: torch.Tensor, model: nn.Module, sparsity: float = 0.9
) -> Mask:
    """
    Create a dictionary of binary gradient masks based on Fisher importance scores.

    Keeps sparsity% of the most important parameters masking them to 0 and sets
    the rest to 1 (sets their gradients to zero during training).

    Args:
        fisher_diag (torch.Tensor): Flattened tensor of Fisher Information scores (1D).
        model (nn.Module): The model whose parameters will be masked.
        sparsity (float): Fraction of total parameters that are frozen (set mask=0).

    Returns:
        Mask:  Dictionary mapping parameter names to binary masks
                                 with the same shape as the parameter tensors.
    """
    assert 0 < sparsity < 1, "sparsity needs to be between 0 and 1"

    k = round(len(fisher_diag) * sparsity)

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

    return Mask(
        mask_dict={
            name: mask.view(shape)
            for name, mask, shape in zip(param_names, split_masks, param_shapes)
        }
    )


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

    Note: Raises a RuntimeError if the final sparsity deviates significantly from the target.

    Args:
        model (nn.Module): Model to prune.
        dataloader (DataLoader): DataLoader (not used in dummy logic).
        loss_fn (nn.Module): Loss function (not used in dummy logic).
        target_sparsity (float): Target sparsity at the end of pruning.
        rounds (int): Number of pruning rounds.
        warn_tolerance (float): Relative deviation tolerance to trigger a warning.
    """
    assert isinstance(rounds, int)
    assert rounds > 1, "rounds needs to be greater than 1"
    device = get_device()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    grad_mask = Mask(
        mask_dict={
            name: torch.ones_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    )

    sparsity_targets = np.geomspace(0.1, target_sparsity, rounds)

    for r, sparsity_target in enumerate(sparsity_targets):
        print(f"[Round {r}] Target sparsity: {sparsity_target}")

        # Recompute Fisher based on the masked model
        fisher_diag = compute_fisher_diagonal(
            model, dataloader, loss_fn, mask=grad_mask
        )

        # Count how many parameters are already frozen and how many parameters are still to mask
        already_masked = grad_mask.num_zeroed_parameters
        parameters_to_mask = round(total_params * sparsity_target)
        adjusted_sparsity = (parameters_to_mask - already_masked) / total_params
        logging.debug(f"[Round {r}]: Target Adjusted sparsity: {adjusted_sparsity}")
        # Create new mask (0 = freeze, 1 = allow update)
        new_mask = create_fisher_mask(fisher_diag, model, sparsity=adjusted_sparsity)
        new_sparsity = new_mask.num_zeroed_parameters
        logging.debug(
            f"[Round {r}] Actual Adjusted Sparsity: {new_sparsity/total_params}."
        )

        # Progressive pruning.
        # Update cumulative mask (once frozen, always frozen) i.e. once a parameter has been set to zero because it's
        # important it's going to stay set to zero. If it is unimportant in the old mask (grad_mask=1) and important in the
        # new_mask (new_mask=0), update it so that grad_mask=0, thus gradually increasing sparsity.
        grad_mask = grad_mask.update(new_mask)
        masked = grad_mask.num_zeroed_parameters
        print(
            f"[Round {r}] Actual Sparsity: {masked/total_params}. Masked: {masked} / {total_params}. "
        )

    # Final sparsity check and warning
    actual_sparsity = grad_mask.sparsity
    rel_error = abs(actual_sparsity - target_sparsity) / target_sparsity

    if rel_error > warn_tolerance:
        raise RuntimeError(
            f"Final sparsity {actual_sparsity:.4f} deviates from target {target_sparsity:.4f} "
            f"by {rel_error:.2%} (exceeds allowed tolerance of {warn_tolerance:.2%})."
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
