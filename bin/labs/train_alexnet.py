import optuna
import torch
import torchvision
from torch import nn
from dataset.imagenet import get_imagenet_dataloaders
from models.alexnet import AlexNet
from core.train import train_model, train_on_subset
from core.train_params import TrainingParams
from utils.model_utils import check_forward_pass


def train_alexnet(batch_size: int, learning_rate: float) -> float:
    """
    Train alexnet for a given batch size and learning rate.
    """
    training_params = TrainingParams(
        training_name=f"alexnet_train_lr_{learning_rate}_batchsize_{batch_size}",
        epochs=10,
        learning_rate=learning_rate,
        model=AlexNet(),
        optimizer_class=torch.optim.SGD,  # type: ignore
        loss_function=nn.CrossEntropyLoss(),
        optimizer_params={"weight_decay": 5e-4, "momentum": 0.9},
    )

    train_loader, val_loader = get_imagenet_dataloaders(batch_size=batch_size)
    accuracy = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        training_params=training_params,
    )
    return accuracy


def objective(trial: optuna.trial.Trial) -> float:
    batch_size = trial.suggest_int("batch_size", 16, 64)
    learning_rate = trial.suggest_loguniform(
        "learning_rate", 1e-5, 1e-1
    )  # Learning rate between 1e-5 and 1e-1 (log scale)

    accuracy = train_alexnet(batch_size, learning_rate)

    return accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)


if __name__ == "__main__":
    # learning_rates = [0.1, 0.001, 0.0001]
    # batch_sizes = [16, 32, 64]

    # # Loop over hyperparameters
    # for lr in learning_rates:
    #     for batch_size in batch_sizes:
    #         # Initialize the network
    #         training_params = TrainingParams(
    #             training_name=f"alexnet_train_lr_{lr}_batchsize_{batch_size}",
    #             epochs=10,
    #             learning_rate=lr,
    #             model=AlexNet(),
    #             optimizer_class=torch.optim.SGD,
    #             loss_function=nn.CrossEntropyLoss(),
    #             optimizer_params={"weight_decay": 5e-4, "momentum": 0.9},
    #         )

    #         train_loader, val_loader = get_imagenet_dataloaders(batch_size=batch_size)
    #         train_model(
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             training_params=training_params,
    #         )
    train_subset = False
    training_params = TrainingParams(
        training_name="alexnet-training_adam_torchvision",
        epochs=10,
        learning_rate=0.001,
        model=torchvision.models.alexnet(
            pretrained=False, num_classes=200
        ),  # AlexNet(),
        optimizer_class=torch.optim.Adam,
        loss_function=nn.CrossEntropyLoss(),
    )

    train_loader, val_loader = get_imagenet_dataloaders(batch_size=32)

    check_forward_pass(training_params.model.cuda(), train_loader, num_classes=200)

    if train_subset:
        train_on_subset(
            training_params,
            train_loader,
            val_loader,
            epochs=2,
            project_name="mldl_lab4",
        )

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        training_params=training_params,
        project_name="mldl_lab4",
    )
