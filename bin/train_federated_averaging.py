import torch
from torch import nn
from core.federated_averaging import FederatedAveraging, ShardingType
from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_dataloaders,
    get_cifar_100_train_valset_datasets,
    get_cifar_100_datasets,
)
from models.dino_backbone import get_dino_backbone_model

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=32)
    model = get_dino_backbone_model()
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)

    client_training_params = TrainingParams(
        training_name="fl_client_training_params",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        optimizer_class=torch.optim.SGD,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=5,
        optimizer_params={
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        scheduler_params={"T_max": 10},
    )

    fedav = FederatedAveraging(
        global_model=model,
        trainset=trainset,
        valset=valset,
        client_training_params=client_training_params,
        sharding_type=ShardingType.IID,
    )
    fedav.train()
