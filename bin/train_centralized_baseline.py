import optuna
import torch
import torch.nn as nn

from core.train import train_model
from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_dataloaders,
)
from models.dino_backbone import get_dino_backbone_model


def objective(trial: optuna.trial.Trial):
    train_dataloader, val_dataloader = get_cifar_dataloaders()
    model = get_dino_backbone_model()

    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    # tune only weight decay and learning rate
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    params = TrainingParams(
        training_name=f"centralized_baseline_momentum_{momentum:.2f}_wdecay_{weight_decay:.2f}_lr_{learning_rate:.2f}_cosineLR",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=learning_rate,
        optimizer_class=torch.optim.SGD,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=3,
        optimizer_params={"momentum": momentum, "weight_decay": weight_decay},
    )
    best_acc = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_baseline",
    )
    return best_acc


def optimize():
    # create a study object (maximize accuracy)
    study = optuna.create_study(direction="maximize")
    # run the optimization for 10 trials
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)


def run_single(*, lr=1e-3, momentum=0.9, weight_decay=5e-4, batch_size=64):
    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=batch_size)
    model = get_dino_backbone_model(freeze_backbone=True)
    params = TrainingParams(
        training_name=f"centralized_baseline_bs_{batch_size}_momentum_{momentum:.2f}_wdecay_{weight_decay:.2f}_lr_{lr:.2f}_cosineLR",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=lr,
        optimizer_class=torch.optim.SGD,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=10,
        optimizer_params={"momentum": momentum, "weight_decay": weight_decay},
        scheduler_params={"T_max": 20},
    )
    res_dict = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="centralized_baseline",
    )
    return res_dict["best_accuracy"]


if __name__ == "__main__":
    # optimize()
    run_single()
