import argparse
import timm
import torch
from torch import nn

from core.model_editing import create_fisher_mask
from core.train import train_model
from core.train_params import TrainingParams
from dataset.cifar_100 import get_cifar_dataloaders
from models.dino_backbone import get_dino_backbone_model
from optim.ssgd import SparseSGDM
from utils.model_utils import get_device


def run_single(
    *,
    fisher_path: str,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    keep_ratio: float = 0.1,
) -> float:
    fisher_diag = torch.load(fisher_path, map_location=get_device())

    # Create backbone and mask
    backbone_model = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    mask = create_fisher_mask(
        fisher_diag=fisher_diag,
        model=backbone_model,
        keep_ratio=keep_ratio,
    )

    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=batch_size)

    # Load model with editable backbone
    model = get_dino_backbone_model(freeze_backbone=False)
    named_params = dict(model.named_parameters())

    _training_name = (
        f"centralized_baseline_bs_{batch_size}_momentum_{momentum:.2f}_wdecay_"
        f"{weight_decay:.2f}_lr_{lr:.2f}_cosineLR"
    )

    params = TrainingParams(
        training_name=_training_name,
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=lr,
        optimizer_class=SparseSGDM,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=10,
        optimizer_params={
            "momentum": momentum,
            "weight_decay": weight_decay,
            "grad_mask": mask,
            "named_params": named_params,
        },
        scheduler_params={"T_max": 20},
    )

    res_dict = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_model_edited_baseline",
    )

    return res_dict["best_accuracy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fisher_path", type=str, required=True, help="Path to Fisher diagonal .pth file")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--keep_ratio", type=float, default=0.1)

    args = parser.parse_args()

    best_acc = run_single(
        fisher_path=args.fisher_path,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        keep_ratio=args.keep_ratio,
    )

    print(f"✅ Best validation accuracy: {best_acc:.4f}")


# from google.colab import drive
# drive.mount('/content/drive')
#
# fisher_path = "/content/drive/MyDrive/checkpoints/fisher_diag.pth"
# !python -m module_name --fisher_path /content/fisher_diag.pth --batch_size 64 --keep_ratio 0.1