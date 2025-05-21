import argparse
from pathlib import Path
import torch

from core.model_editing import Mask
from utils.model_utils import get_device
from utils.plot_utils import plot_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to Fisher mask .pth file",
    )

    args = parser.parse_args()
    fisher_mask = Mask.load_state_dict(
        torch.load(args.mask_path, map_location=get_device())
    )

    save_path = args.mask_path.split("/")[-1].split(".pth")[0] + "_plot.png"

    plot_mask(fisher_mask, Path(save_path))


# python ./bin/inspect_mask.py --mask_path "./checkpoints/progressive_fisher_mask_90.pth"
# python ./bin/inspect_mask.py --mask_path "./checkpoints/progressive_fisher_mask_90_NO_WARMUP.pth"
