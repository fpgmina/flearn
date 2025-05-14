import argparse
import torch
from torch import nn

from core.model_editing import create_fisher_mask, compute_fisher_diagonal
from core.train import train_model
from core.train_params import TrainingParams
from dataset.cifar_100 import get_cifar_dataloaders
from models.dino_backbone import get_dino_backbone_model
from optim.ssgd import SparseSGDM


def run_single(
    *,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 64,
    sparsity: float = 0.9,
) -> float:
    # fisher_diag = torch.load(fisher_path, map_location=get_device())

    # Create backbone and mask
    # backbone_model = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    # mask = create_fisher_mask(
    #     fisher_diag=fisher_diag,
    #     model=backbone_model,
    #     keep_ratio=keep_ratio,
    # ) # this mask is based on a fisher diagonal computed on Imagenet100. We don't need to use Imagenet100
    # in fact the data used should not matter at all (as the model was pre-trained on a large corpus of data and
    # thus it should encode shared structure in the gradients); we can therefore simplify this logic and compute directly
    # the mask on CIFAR100

    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    # Load model with editable backbone and modified head to fit on CIFAR100
    model = get_dino_backbone_model(freeze_backbone=False)
    print("⏳ Computing Fisher diagonal...")
    fisher_diag = compute_fisher_diagonal(
        model=model, dataloader=train_dataloader, loss_fn=loss_fn, num_batches=None
    )
    print("✅ Fisher diagonal computed. Shape:", fisher_diag.shape)
    mask = create_fisher_mask(fisher_diag=fisher_diag, model=model, sparsity=sparsity)
    # alternatively use a recursive mask function that calls within it compute_fisher_diag
    named_params = dict(model.named_parameters())

    _training_name = (
        f"centralized_baseline_bs_{batch_size}_momentum_{momentum:.2f}_wdecay_"
        f"{weight_decay:.2f}_lr_{lr:.2f}_cosineLR_MODEL_EDIT_{sparsity}"
    )

    params = TrainingParams(
        training_name=_training_name,
        model=model,
        loss_function=loss_fn,
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
    print("⏳ Train Model...")
    res_dict = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_model_edited_baseline",
    )
    best_acc = res_dict["best_accuracy"]
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.9)

    args = parser.parse_args()

    best_acc = run_single(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        sparsity=args.sparsity,
    )

    print(f"✅ Best validation accuracy: {best_acc:.4f}")
