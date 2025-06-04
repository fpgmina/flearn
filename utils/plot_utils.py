import argparse
from pathlib import Path

import pandas as pd
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


def plot_wandb_metrics(
    wandb_path: str,
    save_path: Path,
    title: Optional[str] = None,
    is_federated: bool = False,
):
    api = wandb.Api()
    run = api.run(wandb_path)
    df = run.history()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    epoch_or_round = "Round" if is_federated else "Epoch"

    # Plot Accuracy
    sns.lineplot(
        ax=axes[0], x=df[epoch_or_round], y=df["Train Accuracy"], label="Train Accuracy"
    )
    sns.lineplot(
        ax=axes[0],
        x=df[epoch_or_round],
        y=df["Validation Accuracy"],
        label="Validation Accuracy",
    )
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel(epoch_or_round)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Loss
    sns.lineplot(
        ax=axes[1], x=df[epoch_or_round], y=df["Train Loss"], label="Train Loss"
    )
    sns.lineplot(
        ax=axes[1],
        x=df[epoch_or_round],
        y=df["Validation Loss"],
        label="Validation Loss",
    )
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel(epoch_or_round)
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    save_path = save_path.with_suffix(".png")
    plt.savefig(save_path, bbox_inches="tight")


def plot_wandb_comparison(
    project_path: str,
    save_path: str,
    title: str,
    metric_keys=None,
    rename_runs=None,
):
    """
    Plots loss and accuracy metrics for all runs in a given W&B project.

    Args:
        project_path (str): The project path in the format "username/project_name".
        save_path (str): Path to save the figure.
        title (str): Title of the plot.
        metric_keys (dict): Mapping of plot titles to W&B metric keys, e.g.:
            {
                "Train Loss": "Train Loss",
                "Validation Loss": "Validation Loss",
                "Train Accuracy": "Train Accuracy",
                "Validation Accuracy": "Validation Accuracy"
            }
        rename_runs (dict, optional): A dictionary that maps run names to new names (e.g. {"run1": "Model A", ...}).
    """
    if metric_keys is None:
        metric_keys = {
            "Train Loss": "Train Loss",
            "Validation Loss": "Validation Loss",
            "Train Accuracy": "Train Accuracy",
            "Validation Accuracy": "Validation Accuracy",
        }

    api = wandb.Api()

    runs = api.runs(project_path)

    if not rename_runs:
        rename_runs = dict()
        for run in runs:
            rename_runs[run.name] = "_".join(run.name.split("_")[-3:])

    # Prepare data for plotting
    all_data = []
    for run in runs:
        try:
            history = run.history()
            history["run"] = run.name

            if rename_runs and run.name in rename_runs:
                history["run"] = rename_runs[run.name]

            all_data.append(history)
        except Exception as e:
            print(f"Skipping {run.name}: {e}")

    if not all_data:
        raise ValueError("No run data found.")

    df_all = pd.concat(all_data)

    # Convert df_all from wide format to long format using pd.melt()
    df_long = df_all.melt(
        id_vars=["_step", "run"],
        value_vars=[
            metric_keys["Train Loss"],
            metric_keys["Validation Loss"],
            metric_keys["Train Accuracy"],
            metric_keys["Validation Accuracy"],
        ],
        var_name="metric",
        value_name="value",
    )

    # Create a new 'type' column to differentiate between train and validation
    df_long["type"] = df_long["metric"].apply(
        lambda x: "train" if "Train" in x else "validation"
    )

    # Set the plot style
    sns.set_theme(style="whitegrid")

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Loss with distinct Train and Validation (different line styles)
    sns.lineplot(
        data=df_long[df_long["metric"].str.contains("Loss")],
        x="_step",
        y="value",
        hue="run",
        style="type",
        markers=True,
        ax=axes[0],
        legend="full",
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Plot Accuracy with distinct Train and Validation (different line styles)
    sns.lineplot(
        data=df_long[df_long["metric"].str.contains("Accuracy")],
        x="_step",
        y="value",
        hue="run",
        style="type",
        markers=True,
        ax=axes[1],
        legend="full",
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")


def fetch_val_metrics(wandb_path):
    """
    Fetch validation accuracy and loss per epoch from a specific W&B run.

    Args:
        wandb_path (str): Path to the run in the format "entity/project/run_id".

    Returns:
        pd.DataFrame: Table with columns [epoch, Validation Loss, Validation Accuracy]
    """
    # Keys to extract from run history
    metric_keys = {
        "Validation Loss": "Validation Loss",
        "Validation Accuracy": "Validation Accuracy",
    }

    api = wandb.Api()
    run = api.run(wandb_path)
    history = run.history(samples=10000)
    print(history)

    # Use 'Epoch' as the index if it exists
    df = history[["Validation Loss", "Validation Accuracy"]].copy()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot W&B metrics and save as figure.")
    parser.add_argument(
        "wandb_path", type=str, help="W&B run path: username/project/run_id"
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path.cwd() / "wandb_plot.png",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Title for the overall figure"
    )
    parser.add_argument(
        "--is_federated",
        type=bool,
        default=False,
        help="Whether the wandb run uses Federated Averaging",
    )
    args = parser.parse_args()

    plot_wandb_metrics(args.wandb_path, args.save_path, args.title, args.is_federated)

# python utils/plot_utils.py "francesco-mina-fpgm/centralized_baseline/runs/ho4h2mic" --title "CENTRALIZED BASELINE"
# python utils/plot_utils.py francesco-mina-fpgm/fl/runs/e7bt2wc0 --title "FEDERATED IID" --is_federated True
# python utils/plot_utils.py francesco-mina-fpgm/centralized_model_edited_baseline/runs/4latjoqr --title "CENTRALIZED MODEL EDITED"  # mask with 0 in the head
# python utils/plot_utils.py francesco-mina-fpgm/centralized_model_edited_baseline/runs/w3oyu1d9 --title "CENTRALIZED MODEL EDITED"  # mask with 1 in the head
# python utils/plot_utils.py francesco-mina-fpgm/fl_iid_model_edit/runs/u6nusf04 --title "FEDERATED MODEL EDITED IID" --is_federated True
# python utils/plot_utils.py "francesco-mina-fpgm/fl_iid_max_step_4/runs/u6x9ql4q" --title "FEDERATED IID" --is_federated True
# python utils/plot_utils.py "francesco-mina-fpgm/fl_iid_model_edit_max_step_4/runs/kaw063k0" --title "FEDERATED MODEL EDITED IID" --is_federated True
