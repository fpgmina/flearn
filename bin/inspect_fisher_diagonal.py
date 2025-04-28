import torch
import matplotlib.pyplot as plt

from utils.model_utils import get_device

if __name__ == "__main__":
    SAVE_PATH = "../checkpoints/fisher_diag_histogram.png"
    fisher_diag = torch.load("../checkpoints/fisher_diag_imagenet100.pth", map_location=get_device())

    print(f"✅ Loaded Fisher diagonal.")
    print(f"Shape: {fisher_diag.shape}")
    print(f"Dtype: {fisher_diag.dtype}")
    print(f"Min value: {fisher_diag.min().item():.5e}")
    print(f"Max value: {fisher_diag.max().item():.5e}")
    print(f"Mean value: {fisher_diag.mean().item():.5e}")
    print(f"Std deviation: {fisher_diag.std().item():.5e}")
    print(f"Number of elements: {fisher_diag.numel()}")

    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(fisher_diag.numpy(), bins=100, log=True)
    plt.title("Distribution of Fisher Diagonal Values")
    plt.xlabel("Fisher Score")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True)

    # Save plot
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
