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

# in FederatedAveraging I sample 10 clients per round.
# for each client: number of local steps: epochs * (500 / batch_size) e.g. epochs=1, batch_size=64 ==> local_steps=8
# Keep total number of optimization steps constant: #rounds * 10 * 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_labels",
        type=int,
        required=True,
        help="Number of labels each client gets",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        default=1e-3,
        help="Learning Rate of the model",
    )
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
        default=40,
        help="Number of rounds of communications between server and clients",
    )
    args = parser.parse_args()

    momentum = 0.9
    weight_decay = 5e-4

    mask_path = "/content/drive/MyDrive/progressive_fisher_mask_90.pth"
    trainset, valset, _ = get_cifar_datasets()
    loss_fn = nn.CrossEntropyLoss()
    # Load model with editable backbone and modified head to fit on CIFAR100
    model = get_dino_backbone_model(freeze_backbone=False)
    mask = Mask.load_state_dict(torch.load(mask_path))
    # cambia la head della mask cosi' che abbia 1s in modo che sia consistente con il model.
    # (model ha random weights nella head, voglio poterli update)
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
        learning_rate=args.learning_rate,
        optimizer_class=SparseSGDM,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=args.epochs,
        optimizer_params={
            "momentum": momentum,
            "weight_decay": weight_decay,
            "grad_mask": mask,
            "named_params": named_params,
        },
        max_steps=4,
        scheduler_params={"T_max": 10, "eta_min": 1e-5},
    )

    fedav = FederatedAveraging(
        global_model=model,
        trainset=trainset,
        valset=valset,
        rounds=args.rounds,
        client_training_params=client_training_params,
        sharding_type=ShardingType.NON_IID,
        num_classes=args.client_labels,  # only samples of #client_label labels on average
        wandb_project_name="fl_non_iid_model_edit_MAX_STEP",
    )
    fedav.train()
