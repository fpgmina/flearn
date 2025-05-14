import torch
import timm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from core.model_editing import compute_fisher_diagonal, create_fisher_mask
from utils.model_utils import get_device

if __name__ == "__main__":

    IMAGENET100_VAL_DIR = "/content/drive/MyDrive/datasets/imagenet100/val/"

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    imagenet100_val = datasets.ImageFolder(
        root=IMAGENET100_VAL_DIR, transform=transform
    )

    val_loader = DataLoader(
        imagenet100_val, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )

    # --- LOAD MODEL ---

    device = get_device()

    model = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    model = model.to(device)

    # --- COMPUTE FISHER DIAGONAL ---

    loss_fn = nn.CrossEntropyLoss()

    print("⏳ Computing Fisher diagonal...")
    fisher_diag = compute_fisher_diagonal(
        model=model, dataloader=val_loader, loss_fn=loss_fn, num_batches=None
    )

    print("✅ Fisher diagonal computed. Shape:", fisher_diag.shape)

    # Save Fisher diagonal
    torch.save(fisher_diag, "fisher_diag_imagenet100.pth")
    print("✅ Fisher diagonal saved to fisher_diag_imagenet100.pth")

    # --- CREATE FISHER MASK ---

    fisher_mask = create_fisher_mask(fisher_diag=fisher_diag, model=model, sparsity=0.8)

    # Save Fisher mask
    torch.save(fisher_mask, "fisher_mask_imagenet100.pth")
    print("✅ Fisher mask saved to fisher_mask_imagenet100.pth")
