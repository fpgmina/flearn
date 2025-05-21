import argparse
from pathlib import Path

from utils.plot_utils import plot_wandb_comparison

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
    args = parser.parse_args()

    plot_wandb_comparison(args.wandb_path, args.save_path, args.title)

# python bin/plot_federated.py francesco-mina-fpgm/fl_non_iid --title "FEDERATED NON-IID"
# python bin/plot_federated.py francesco-mina-fpgm/fl_non_iid_model_edit --title "FEDERATED MODEL EDITED NON-IID
