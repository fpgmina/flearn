import attr
import torch
from torch import nn
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from optim.ssgd import SparseSGDM


def is_nn_module(instance, attribute, value):
    """Validator to check if the value is an instance of nn.Module or its subclass."""
    if not isinstance(value, nn.Module):
        raise TypeError(
            f"{attribute.name} must be an instance of nn.Module or its subclass."
        )
    return value


def is_optimizer_class(instance, attribute, value):
    """Validator to check if the value is a subclass of torch.optim.Optimizer."""
    if not issubclass(value, Optimizer):
        raise TypeError(
            f"{attribute.name} must be a subclass of torch.optim.Optimizer."
        )
    return value


def is_scheduler_class(instance, attribute, value):
    """Validator to check if the value is a subclass of torch.optim.lr_scheduler.LRScheduler"""
    if value is not None:
        if not issubclass(value, LRScheduler):
            raise TypeError(
                f"{attribute.name} must be a subclass of torch.optim.lr_scheduler.LRScheduler"
            )
    return value


@attr.s(frozen=True, kw_only=True)
class TrainingParams:
    """
    A class to store the parameters required for core a model.

    Attributes:
        training_name (str): A name for the core experiment.
        epochs (int): The number of epochs for core.
        learning_rate (float): The learning rate for the optimizer.
        model (nn.Module): The model to be trained.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to be used for core.
        loss_function (nn.Module): The loss function to be used.
        max_steps (Optional[int]): The maximum number of optimization steps to be taken.
        optimizer_params (Optional[Dict[str, Any]]): A dictionary of additional optimizer parameters (optional).
    """

    training_name: str = attr.ib(validator=attr.validators.instance_of(str))
    epochs: int = attr.ib(validator=attr.validators.instance_of(int))
    learning_rate: float = attr.ib(validator=attr.validators.ge(0.0))
    model: nn.Module = attr.ib(
        validator=is_nn_module
    )  # Custom validation to pass instance check (instance_of checks for exact type and not for superclasses)
    loss_function: nn.Module = attr.ib(validator=is_nn_module)  # Custom validation
    optimizer_class: torch.optim.Optimizer = attr.ib(validator=is_optimizer_class)
    scheduler_class: Optional[torch.optim.lr_scheduler.LRScheduler] = attr.ib(
        validator=is_scheduler_class, default=None
    )
    max_steps: Optional[int] = attr.ib(default=None)
    optimizer_params: Optional[Dict[str, Any]] = attr.ib(default=None)
    scheduler_params: Optional[Dict[str, Any]] = attr.ib(default=None)

    def __attrs_post_init__(self):
        if self.optimizer_class is SparseSGDM:
            if not isinstance(self.optimizer_params, dict):
                raise ValueError(
                    "optimizer_params must be a dictionary when using SparseSGDM."
                )

            if "named_params" not in self.optimizer_params:
                raise ValueError(
                    "SparseSGDM requires 'named_params' in optimizer_params."
                )

            if "grad_mask" not in self.optimizer_params:
                raise ValueError("SparseSGDM requires 'grad_mask' in optimizer_params.")

    @property
    def optimizer(self):
        optimizer_params = self.optimizer_params or {}
        return self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **optimizer_params
        )

    @property
    def scheduler(self):
        scheduler_params = self.scheduler_params or {}
        if self.scheduler_class:
            return self.scheduler_class(
                self.optimizer, **{"T_max": 50, **scheduler_params}
            )
