import logging
import os
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
from core.train_params import TrainingParams
from utils.model_utils import get_device, get_subset_loader

__all__ = ["train_model", "compute_predictions", "train_on_subset"]


os.environ["WANDB_MODE"] = "online"


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # format like a print
    handlers=[logging.StreamHandler()],  # output logs to console
)


def _train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    """
    Note: .loss.item() returns the average per batch loss (not the sum of the losses of the elements in the batch).
    """
    device = get_device()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        preds = model(inputs)
        loss = loss_func(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # loss.item is average loss per batch
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss = running_loss / len(
        train_loader
    )  # NB len(train_loader) is the total number of batches
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


def compute_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[str] = None,
    loss_function: Optional[nn.Module] = None,
):
    """
    Compute predictions for a given dataloader using the trained model.
    Note: .loss.item() returns the average per batch loss (not the sum of the losses of the elements in the batch).

    Args:
        model: The trained model.
        dataloader: The DataLoader containing the test or train dataset.
        device: The device to use ('cpu' or 'cuda').
        loss_function: The loss function to minimize in core.

    Returns:
        predictions: Tensor of predictions.
        labels: Tensor of true labels.
        loss: Computed loss for the given data.
        accuracy: Computed accuracy for the given data.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    labels = []
    device = device or get_device()

    loss = 0.0

    # Disable gradient computation during inference
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(
                device
            )  # Move to the appropriate device

            # Forward pass
            preds = model(inputs)  # Get raw model predictions

            if loss_function is not None:
                loss += loss_function(preds, targets).item()

            # Get predicted class (class with the highest score)
            _, predicted = torch.max(preds, 1)

            predictions.append(predicted)
            labels.append(targets)

    # Concatenate all predictions and labels
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    loss = loss / len(dataloader)

    return predictions, labels, loss, accuracy


def train_model(
    *,
    training_params: TrainingParams,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
) -> Dict[str, Union[torch.nn.Module, float]]:
    """
    Train a pytorch-based model with the given training parameters and data loaders.

    Args:
        training_params (TrainingParams): Training parameters including model, optimizer, loss function, and scheduler.
        train_loader (DataLoader): PyTorch DataLoader for the training dataset.
        val_loader (Optional[DataLoader], optional): PyTorch DataLoader for the validation dataset, defaults to None.
        project_name (str, optional): Name of the WandB project, defaults to "flearn".
        wandb_log (bool): Use wandb to log training output, otherwise fallback to default logger. Defaults to True.
        wandb_save (bool): Use wandb to save trained model, otherwise do not save. Defaults to True.

    Returns:
        Dict[str, torch.nn.Module, float]: A dictionary containing the trained model and the best validation accuracy.
            - 'model' (torch.nn.Module): The trained model.
            - 'best_accuracy' (float): The highest validation accuracy achieved during training.
    """

    assert isinstance(training_params, TrainingParams)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    use_wandb = wandb_log or wandb_save
    if use_wandb:
        if project_name is None:
            raise ValueError(
                "project name cannot be None with either wandb_log or wandb_save set to True."
            )

    if use_wandb:
        wandb.init(
            project=project_name,
            name=training_params.training_name,
            config={
                "epochs": training_params.epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": training_params.learning_rate,
                "architecture": training_params.model.__class__.__name__,
                "optimizer_class": training_params.optimizer_class.__name__,
                "loss_function": training_params.loss_function.__class__.__name__,
                **(training_params.optimizer_params or {}),
            },
        )
    device = get_device()
    model = training_params.model.to(device)
    loss_func = training_params.loss_function
    optimizer = training_params.optimizer
    scheduler = training_params.scheduler

    best_acc = 0
    num_epochs = training_params.epochs

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = _train(
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if wandb_log:
            # Log core metrics to wandb
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                }
            )
        else:
            logging.info(
                f"Epoch: {epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}"
            )

        if val_loader:
            _, _, val_loss, val_accuracy = compute_predictions(
                model=model, dataloader=val_loader, loss_function=loss_func
            )
            if wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Validation Loss": val_loss,
                        "Validation Accuracy": val_accuracy,
                    }
                )
            else:
                logging.info(
                    f"Epoch: {epoch}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
                )

            # Save the model with the best validation accuracy
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                if wandb_save:
                    model_name = f"{training_params.training_name}_best.pth"
                    torch.save(model.state_dict(), model_name)  # local save
                    artifact = wandb.Artifact(
                        name=f"{training_params.training_name}", type="model"
                    )
                    artifact.add_file(model_name)
                    wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    res_dict = {"model": model, "best_accuracy": best_acc}
    return res_dict


def train_on_subset(training_params, train_loader, val_loader=None, epochs=2, **kwargs):
    assert isinstance(training_params, TrainingParams)
    training_params_subset = TrainingParams(
        **{
            **training_params.__dict__,
            **{
                "training_name": f"{training_params.training_name}_SUBSET",
                "epochs": epochs,
            },
        }
    )
    train_loader_subset = get_subset_loader(train_loader)
    train_model(
        training_params=training_params_subset,
        train_loader=train_loader_subset,
        val_loader=val_loader,
        **kwargs,
    )
    logging.info("Finished core on subset.")
