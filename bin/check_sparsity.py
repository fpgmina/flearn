import torch
from pathlib import Path

from core.model_editing import Mask

if __name__ == "__main__":

    root = Path("../checkpoints")
    mask_file = root / "progressive_fisher_mask_90_HESSIAN_PARAM_SQUARED.pth"

    mask = Mask.load_state_dict(torch.load(mask_file, map_location=torch.device("cpu")))

    actual_sparsity, total_params, zeroed_params = mask.check_sparsity(
        target_sparsity=0.9, tolerance=0.02
    )

    print(
        f"PROGRESSIVE FISHER HESSIAN PARAM SQUARED -- Sparsity: {actual_sparsity:.2%}, Total params: {total_params}, Zeroed params: {zeroed_params}"
    )

    mask_file = root / "progressive_fisher_mask_90.pth"
    mask = Mask.load_state_dict(torch.load(mask_file, map_location=torch.device("cpu")))

    actual_sparsity, total_params, zeroed_params = mask.check_sparsity(
        target_sparsity=0.9, tolerance=0.02
    )
    print(
        f"PROGRESSIVE FISHER STANDARD -- Sparsity: {actual_sparsity:.2%}, Total params: {total_params}, Zeroed params: {zeroed_params}"
    )
