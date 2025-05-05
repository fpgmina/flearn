import argparse
import torch
import matplotlib.pyplot as plt

from utils.model_utils import get_device


def _inspect_fisher(fisher_diag):
    print("✅ Loaded Fisher diagonal.")
    print(f"Shape: {fisher_diag.shape}")
    print(f"Dtype: {fisher_diag.dtype}")
    print(f"Min value: {fisher_diag.min().item():.5e}")
    print(f"Max value: {fisher_diag.max().item():.5e}")
    print(f"Mean value: {fisher_diag.mean().item():.5e}")
    print(f"Std deviation: {fisher_diag.std().item():.5e}")
    print(f"Number of elements: {fisher_diag.numel()}")


def _plot_fisher(fisher_diag, save_path):
    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(fisher_diag.numpy(), bins=100, log=True)
    plt.title("Distribution of Fisher Diagonal Values")
    plt.xlabel("Fisher Score")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True)

    # Save plot
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fisher_path",
        type=str,
        required=True,
        help="Path to Fisher diagonal .pth file",
    )

    args = parser.parse_args()
    fisher_diag = torch.load(args.fisher_path, map_location=get_device())

    SAVE_PATH = args.fisher_path.split("/")[-1].split(".pth")[0] + "_histogram.png"
    _inspect_fisher(fisher_diag)
    _plot_fisher(fisher_diag, SAVE_PATH)


# python ./bin/inspect_fisher_diagonal.py --fisher_path "./checkpoints/fisher_diag_cifar100.pth"
