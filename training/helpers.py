"""
Helper functions for DINOv2 training.
Includes block-mask generation and data loading utilities.
"""

import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import math

import numpy as np
from typing import Tuple


class BlockMaskGenerator:
    """
    Generate spatially coherent block masks for iBOT training.
    Based on official DINOv2 MaskingGenerator.

    Args:
        input_size: Tuple of (height, width) in patches
        min_num_patches: Minimum patches per block
        min_aspect: Minimum aspect ratio for blocks
        max_aspect: Maximum aspect ratio (default: 1/min_aspect)
    """
    def __init__(self, input_size, min_num_patches=4, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        """Attempt to place a rectangular block on the mask."""
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect = math.exp(random.uniform(*self.log_aspect))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))

            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top:top+h, left:left+w].sum()

                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top:top+h, left:left+w] = 1
                    delta += h * w - num_masked
                    if delta >= max_mask_patches:
                        break
        return delta

    def __call__(self, num_masking_patches):
        """Generate a single mask with num_masking_patches masked tokens."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            delta = self._mask(mask, num_masking_patches - mask_count)
            if delta == 0:
                break
            mask_count += delta
        return mask.flatten()


def generate_block_masks(
    batch_size: int,
    n_patches_h: int,
    n_patches_w: int,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    mask_sample_probability: float = 0.5,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate block masks for a batch with variable ratios.

    Args:
        batch_size: Number of samples
        n_patches_h: Number of patches in height
        n_patches_w: Number of patches in width
        mask_ratio_min: Minimum mask ratio (e.g., 0.1 = 10%)
        mask_ratio_max: Maximum mask ratio (e.g., 0.5 = 50%)
        mask_sample_probability: Fraction of samples to mask (e.g., 0.5)
        device: Device for output tensors

    Returns:
        masks: [B, N] boolean tensor where True = masked
        masks_weight: [B] tensor with 1/num_masked for loss weighting
    """
    n_patches = n_patches_h * n_patches_w
    n_masked_samples = int(batch_size * mask_sample_probability)

    mask_generator = BlockMaskGenerator(
        input_size=(n_patches_h, n_patches_w),
        min_num_patches=4,
        min_aspect=0.3
    )

    # Generate variable-ratio masks for masked samples
    masks = []
    probs = np.linspace(mask_ratio_min, mask_ratio_max, n_masked_samples + 1)

    for i in range(n_masked_samples):
        prob_min, prob_max = probs[i], probs[i + 1]
        ratio = random.uniform(prob_min, prob_max)
        num_to_mask = int(n_patches * ratio)
        masks.append(mask_generator(num_to_mask))

    # Empty masks for unmasked samples
    for _ in range(n_masked_samples, batch_size):
        masks.append(np.zeros(n_patches, dtype=bool))

    # Shuffle to distribute randomly
    random.shuffle(masks)

    masks_tensor = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)

    # Compute weights: 1 / num_masked_per_sample (0 for unmasked samples)
    num_masked = masks_tensor.sum(dim=1).float()
    masks_weight = torch.where(
        num_masked > 0,
        1.0 / num_masked,
        torch.zeros_like(num_masked)
    )

    return masks_tensor, masks_weight


def calculate_total_student_views(args):
    """
    Calculate total number of student views for DINO CLS loss.
    """
    total = 0

    # Global views (always included unless explicitly set to 0)
    if args.global_views > 0:
        total += args.global_views

    # Standard local crops
    total += args.n_standard_local_crops

    return total


def worker_init_fn(worker_id):
    """
    Initialize worker with proper random seeding.

    Args:
        worker_id: Worker ID
    """
    import numpy as np
    import torch
    from torch.utils.data import get_worker_info

    worker_info = get_worker_info()
    dataset = worker_info.dataset

    if hasattr(dataset, 'set_worker_info'):
        dataset.set_worker_info(worker_info.id, worker_info.num_workers)
    seed = getattr(dataset, 'seed', 0)

    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_ddp_model(model, args, find_unused=False):
    """
    Setup model for distributed data parallel training.

    Args:
        model: Model to wrap
        args: Arguments with GPU info
        find_unused: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    # Enable gradient checkpointing BEFORE wrapping
    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print(f"Enabled gradient checkpointing before wrapping")

    # Hand-rolled data parallelism instead of DDP: DDP wraps the autograd graph and
    # breaks the torch.compile + gradient-checkpointing nesting (CheckpointError,
    # pytorch#144035). HandRolledDP does a manual, bit-equivalent grad all-reduce
    # after backward() (find_unused is handled by its presence-mask -> None logic).
    from training.hand_rolled_dp import HandRolledDP
    return HandRolledDP(model)

