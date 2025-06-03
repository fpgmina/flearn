import argparse

import torch
from torch import nn
from core.federated_averaging import FederatedAveraging, ShardingType
from core.model_editing import Mask
from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_datasets,
)
from optim.ssgd import SparseSGDM
from models.dino_backbone import get_dino_backbone_model
from utils.model_utils import get_device

if __name__ == "__main__":
    momentum = 0.9
    weight_decay = 5e-4
    lr = 1e-3

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        default=1,
        help="Number of epochs for client",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        default=60,
        help="Number of rounds of communications between server and clients",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        required=True,
        default=4,
    )
    args = parser.parse_args()

    mask_path = "/content/drive/MyDrive/progressive_fisher_mask_90.pth"
    trainset, valset, _ = get_cifar_datasets()
    loss_fn = nn.CrossEntropyLoss()
    # Load model with editable backbone and modified head to fit on CIFAR100
    model = get_dino_backbone_model(freeze_backbone=False)
    mask = Mask.load_state_dict(torch.load(mask_path))
    #  cambia la head della mask cosi' che abbia 1s in modo che sia consistente con il model.
    #  (model ha random weights nella head, voglio poterli update)
    mask = mask.unmask_layers(["head.weight", "head.bias"])

    device = get_device()
    model.to(device)
    mask.to(device)

    named_params = dict(model.named_parameters())
    mask.validate_against(named_params)

    client_training_params = TrainingParams(
        training_name="fl_client_training_params_model_edit",
        model=model,
        loss_function=loss_fn,
        learning_rate=lr,
        optimizer_class=SparseSGDM,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=args.epochs,
        optimizer_params={
            "momentum": momentum,
            "weight_decay": weight_decay,
            "grad_mask": mask,
            "named_params": named_params,
        },
        max_steps=args.max_steps,
        scheduler_params={"T_max": 10},
    )

    fedav = FederatedAveraging(
        global_model=model,
        trainset=trainset,
        valset=valset,
        client_training_params=client_training_params,
        sharding_type=ShardingType.IID,
        rounds=args.rounds,
        wandb_project_name="fl_iid_model_edit_max_step_4",
    )
    fedav.train()
