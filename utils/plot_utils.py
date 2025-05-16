import argparse
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional
from math import ceil
from matplotlib.gridspec import GridSpec


def plot_mask(
    mask, save_path: Optional[Path] = None, per_layer_heatmaps: bool = False
) -> plt.Figure:
    """
    Visualize a Mask object:
      - Per-layer sparsity with color-coded module type and param count overlay
      - Optional: Heatmaps of 2D masks

    Args:
        mask (Mask): Dict-like object with {param_name: mask_tensor}
        save_path (Optional[Path]): If set, saves the figure
        per_layer_heatmaps (bool): If True, show 2D mask heatmaps

    Returns:
        fig (matplotlib.figure.Figure): The resulting figure
    """
    # Compute sparsity and parameter count
    sparsity = {name: (m == 0).float().mean().item() for name, m in mask.items()}
    param_counts = {name: m.numel() for name, m in mask.items()}
    heatmap_items = (
        [(name, m) for name, m in mask.items() if m.ndim == 2]
        if per_layer_heatmaps
        else []
    )

    # Classify modules by name
    def classify(name):
        if "attn" in name.lower():
            return "attn"
        if "mlp" in name.lower():
            return "mlp"
        if "norm" in name.lower():
            return "norm"
        if "head" in name.lower():
            return "head"
        return "other"

    color_map = {
        "attn": "steelblue",
        "mlp": "darkorange",
        "norm": "green",
        "head": "crimson",
        "other": "gray",
    }

    # Layout: 1 row for barplot + remaining rows for heatmaps
    ncols = 2
    n_heatmap_rows = ceil(len(heatmap_items) / ncols) if per_layer_heatmaps else 0
    nrows = 1 + n_heatmap_rows

    fig = plt.figure(figsize=(7.5 * ncols, 3 * nrows))
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)

    # Per-layer sparsity bar plot (row 0, full width)
    ax = fig.add_subplot(gs[0, :])
    keys = list(sparsity.keys())
    sparsity_vals = [sparsity[k] for k in keys]
    param_vals = [param_counts[k] for k in keys]
    module_types = [classify(k) for k in keys]
    bar_colors = [color_map[t] for t in module_types]

    ax.bar(range(len(keys)), sparsity_vals, color=bar_colors, label="Sparsity")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=6)
    ax.set_ylabel("Sparsity (fraction masked)")
    ax.set_title("Per-Layer Sparsity (with Parameter Count)")
    ax.grid(True, axis="y")

    # Overlay parameter count on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        range(len(keys)),
        param_vals,
        color="black",
        linestyle="--",
        marker="o",
        markersize=2,
        linewidth=1,
        label="Param count",
    )
    ax2.set_ylabel("Parameter count")

    # Custom legend
    from matplotlib.patches import Patch

    legend_patches = [Patch(color=color_map[t], label=t) for t in color_map]
    ax.legend(handles=legend_patches + [ax2.lines[0]], loc="upper right", fontsize=8)

    # Optional heatmaps
    for i, (name, m) in enumerate(heatmap_items):
        row = 1 + i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(m.cpu().numpy(), cmap="Greys", aspect="auto")
        ax.set_title(f"Mask Heatmap: {name}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_wandb_metrics(wandb_path: str, save_path: Path, title: Optional[str] = None):
    api = wandb.Api()
    run = api.run(wandb_path)
    df = run.history()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot Accuracy
    sns.lineplot(
        ax=axes[0], x=df["Epoch"], y=df["Train Accuracy"], label="Train Accuracy"
    )
    sns.lineplot(
        ax=axes[0], x=df["Epoch"], y=df["Validation Accuracy"], label="Validation Accuracy"
    )
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Loss
    sns.lineplot(ax=axes[1], x=df["Epoch"], y=df["Train Loss"], label="Train Loss")
    sns.lineplot(ax=axes[1], x=df["Epoch"], y=df["Validation Loss"], label="Validation Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    save_path = save_path.with_suffix(".png")
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot W&B metrics and save as figure.")
    parser.add_argument("wandb_path", type=str, help="W&B run path: username/project/run_id")
    parser.add_argument("--save_path", type=Path, default=Path.cwd() / "centralized_baseline_wandb_plot.png")
    parser.add_argument("--title", type=str, default=None, help="Title for the overall figure")
    args = parser.parse_args()

    plot_wandb_metrics(args.wandb_path, args.save_path, args.title)

# python utils/plot_utils.py "francesco-mina-fpgm/centralized_baseline/runs/ho4h2mic" --title "CENTRALIZED BASELINE"
