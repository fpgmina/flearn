import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, Optional

from utils.model_utils import get_device


# Loss function (averaged over N samples):
#
#     L(θ) = (1 / N) * ∑_{i=1}^{N} L^{(i)}(θ) where L^{(i)} = L(y^i, x^i) i.e. the loss on the i-th data point
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
    Compute the diagonal of the Fisher Information Matrix using squared gradients.

    The Fisher Information is given by the expected value of the squared gradient of the loss function:

        Fisher(θ) = E_{(x, y) ~ D} [ (∂L(f_θ(x), y) / ∂θ)^2 ]

    This function approximates and returns the diagonal of the Fisher Information Matrix:

        Fisher(θ) ≈ (1 / N) * ∑_{i=1}^{N} (∇_θ L^{(i)}(θ))²

    Where:
        - L^{(i)}(θ) is the loss on the i-th data point
        - ∇_θ L^{(i)}(θ) is the gradient of the loss with respect to parameters θ
        - N is the number of samples (or mini-batches)
        - The square is element-wise and gives the diagonal approximation

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
    fisher_diag = torch.cat(
        [torch.zeros_like(p.data.flatten()) for p in model.parameters()]
    )
    total_samples = 0
    device = get_device()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                # g^(i) = ∂L^(i) / ∂θ the gradient of the mini batch (a p dimensional vector
                # where p is the number of parameters)
                grads.append(p.grad.detach().clone().flatten())
            else:
                grads.append(torch.zeros_like(p.data.flatten()))

        grad_vector = torch.cat(grads)
        fisher_diag += grad_vector.pow(2)
        total_samples += inputs.size(0)

    fisher_diag /= total_samples
    return fisher_diag


def create_fisher_mask(
    fisher_diag: torch.Tensor, model: nn.Module, keep_ratio: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Create a dictionary of binary gradient masks based on Fisher importance scores.

    Keeps the top-k% most important parameters (based on the Fisher Information diagonal),
    and masks the rest (sets their gradients to zero during training).

    Args:
        fisher_diag (torch.Tensor): Flattened tensor of Fisher Information scores (1D).
        model (nn.Module): The model whose parameters will be masked.
        keep_ratio (float): Fraction of total parameters to keep (set mask=1).

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping parameter names to binary masks
                                 with the same shape as the parameter tensors.
    """
    k = int(len(fisher_diag) * keep_ratio)
    threshold = torch.topk(fisher_diag, k, largest=True).values[-1]
    flat_mask = (fisher_diag >= threshold).float()

    param_sizes = [p.numel() for _, p in model.named_parameters()]
    param_shapes = [p.shape for _, p in model.named_parameters()]
    param_names = [name for name, _ in model.named_parameters()]
    split_masks = torch.split(flat_mask, param_sizes)

    return {
        name: mask.view(shape)
        for name, mask, shape in zip(param_names, split_masks, param_shapes)
    }


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


def calibrate_talos_mask(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    final_sparsity: float,
    R: int,
) -> Dict[str, torch.Tensor]:
    """
    TaLoS-style sparse fine-tuning mask calibration using iterative Fisher scoring.

    Args:
        model: Pretrained torch.nn.Module (θ₀)
        dataloader: Dataset Dt
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        final_sparsity: Target sparsity (0 < final_sparsity < 1)
        R: Number of pruning rounds

    Returns:
        c: A dict mapping parameter names to binary 0/1 masks
    """
    device = get_device()
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()

    c: Dict[str, torch.Tensor] = {
        name: torch.ones_like(p, dtype=torch.float32, device=device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    m = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for r in range(1, R + 1):
        curr_sparsity = final_sparsity * (r / R)

        s: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        for x, _ in dataloader:
            x = x.to(device)
            model.zero_grad()

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            y_sampled = torch.multinomial(probs, num_samples=1).squeeze()

            loss = loss_fn(logits, y_sampled)
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None:
                    s[name] += (p.grad.detach() ** 2) * c[name]

        num_to_keep = min(int((1 - curr_sparsity) * m), m - 1)

        all_scores_flat = torch.cat([v.view(-1) for v in s.values()])
        sorted_scores, _ = torch.sort(all_scores_flat, descending=True)
        threshold_value = sorted_scores[num_to_keep - 1]  # retain top scores

        for name in s:
            c[name] = torch.where(
                s[name] >= threshold_value,
                torch.ones_like(c[name]),
                torch.zeros_like(c[name]),
            )

    return c
