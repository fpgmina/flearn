import torch
import timm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from core.model_editing import compute_fisher_diagonal, create_fisher_mask
from dataset.cifar_100 import get_cifar_dataloaders
from models.dino_backbone import get_dino_backbone_model
from utils.model_utils import get_device

if __name__ == "__main__":

    train_loader, val_loader = get_cifar_dataloaders(batch_size=32)

    device = get_device()

    model = get_dino_backbone_model(freeze_backbone=False)
    model = model.to(device)

    # --- COMPUTE FISHER DIAGONAL ---

    loss_fn = nn.CrossEntropyLoss()

    print("⏳ Computing Fisher diagonal...")
    fisher_diag = compute_fisher_diagonal(
        model=model, dataloader=train_loader, loss_fn=loss_fn, num_batches=None
    )

    print("✅ Fisher diagonal computed. Shape:", fisher_diag.shape)

    # Save Fisher diagonal
    torch.save(fisher_diag, "fisher_diag_cifar100.pth")
    print("✅ Fisher diagonal saved to fisher_diag_cifar100.pth")

    # --- CREATE FISHER MASK ---

    fisher_mask = create_fisher_mask(
        fisher_diag=fisher_diag, model=model, keep_ratio=0.2
    )

    # Save Fisher mask
    torch.save(fisher_mask, "fisher_mask_cifar100.pth")
    print("✅ Fisher mask saved to fisher_mask_cifar100.pth")
