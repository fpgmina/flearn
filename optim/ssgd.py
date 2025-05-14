import torch
from torch.optim import SGD
from typing import Dict, Iterable

from core.model_editing import Mask


class SparseSGDM(SGD):
    """
    SparseSGDM is an extension of PyTorch's SGD optimizer that applies
    a gradient mask during the optimization step. This allows you to freeze
    certain model parameters by zeroing out their gradients before updating.

    It is typically used in sparse training or fine-tuning workflows where only
    a subset of parameters (e.g. those selected by Fisher Information) are allowed to update.

    Args:
        params (Iterable[nn.Parameters]): Iterable of parameters to be optimized.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        dampening (float): Dampening for momentum.
        weight_decay (float): Weight decay (L2 penalty).
        nesterov (bool): Enables Nesterov momentum.
        named_params (Dict[str, torch.nn.Parameter]): Dictionary mapping parameter names
            to their corresponding tensors. Used to match parameters to their gradient masks.
        grad_mask (Mask): Mask object mapping parameter names to binary masks
            of the same shape. A value of 1 allows gradient updates; 0 freezes the parameter.

    Note on parameter freezing: During each call to .step(), it multiplies the gradient of each parameter by the mask,
    effectively zeroing gradients where the mask is 0. This prevents any update to the parameter when super().step()
    is eventually called.

    Example:
        >>> optimizer = SparseSGDM(named_params, grad_mask, lr=0.01)
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        - The optimizer assumes that `named_params` includes all parameters passed to `SGD`.
        - Parameters not included in `grad_mask` will not be masked.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        grad_mask: Mask,
        named_params: Dict[str, torch.nn.Parameter],
    ):

        param_list = list(params)  # Materialize generator safely
        named_param_list = list(named_params.values())

        super(SparseSGDM, self).__init__(
            param_list,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        # check that named_params and grad_mask have the same keys
        model_param_names = set(named_params.keys())
        grad_mask_names = set(grad_mask.keys())
        assert model_param_names == grad_mask_names, (
            f"Mismatch between model parameters and gradient mask keys.\n"
            f"Missing in grad_mask: {model_param_names - grad_mask_names}\n"
            f"Extra in grad_mask: {grad_mask_names - model_param_names}"
        )

        # check that params and named_params.values() have the same items
        assert len(param_list) == len(
            named_param_list
        ), f"params and named_params have different lengths: {len(param_list)} vs {len(named_param_list)}"
        for p1, p2 in zip(param_list, named_param_list):
            assert (
                p1 is p2
            ), f"params and named_params.values() mismatch: {p1} is not {p2}"

        self.named_params = named_params
        self.param_id_to_name = {id(p): n for n, p in named_params.items()}
        self.grad_mask = grad_mask

    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            for param in group["params"]:
                # Applying the gradient mask only if that name is in the mask
                name = self.param_id_to_name.get(id(param))
                if param.grad is not None and name in self.grad_mask:
                    if param.grad.shape != self.grad_mask[name].shape:
                        raise ValueError(
                            f"Gradient shape {param.grad.shape} does not match mask shape {self.grad_mask[name].shape} "
                            f"for parameter '{name}'"
                        )
                    param.grad.data *= self.grad_mask[name]

        return super().step(closure)
