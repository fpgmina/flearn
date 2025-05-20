import torch
from torch import nn
from core.federated_averaging import FederatedAveraging, ShardingType
from core.model_editing import Mask
from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_dataloaders,
    get_cifar_datasets,
)
from models.dino_backbone import get_dino_backbone_model
from utils.model_utils import get_device

if __name__ == "__main__":
    batch_size = 32
    momentum = 0.9
    weight_decay = 5e-4
    lr = 1e-3

    mask_path = "/content/drive/MyDrive/progressive_fisher_mask_90.pth"
    trainset, valset, _ = get_cifar_datasets()
    loss_fn = nn.CrossEntropyLoss()
    # Load model with editable backbone and modified head to fit on CIFAR100
    model = get_dino_backbone_model(freeze_backbone=False)
    mask = Mask.load_state_dict(torch.load(mask_path))

    named_params = dict(model.named_parameters())
    mask.validate_against(named_params)

    device = get_device()
    model.to(device)
    mask.to(device)

    client_training_params = TrainingParams(
        training_name="fl_client_training_params_model_edit",
        model=model,
        loss_function=loss_fn,
        learning_rate=lr,
        optimizer_class=SparseSGDM,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=5,
        optimizer_params={
            "momentum": momentum,
            "weight_decay": weight_decay,
            "grad_mask": mask,
            "named_params": named_params,
        },
        scheduler_params={"T_max": 10},
    )

    fedav = FederatedAveraging(
        global_model=model,
        trainset=trainset,
        valset=valset,
        client_training_params=client_training_params,
        sharding_type=ShardingType.IID,
        wandb_project_name="fl_iid_model_edit",
    )
    fedav.train()
