"""
iBOT patch-level loss for masked token prediction.
Optimized: Vectorized masking eliminates per-sample loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class iBOTPatchLoss(nn.Module):
    """
    Canonical iBOT patch loss implementation.
    
    Args:
        student_temp: Student temperature
        n_iterations: Sinkhorn-Knopp iterations
    """
    def __init__(self, student_temp=0.1, n_iterations=3):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        teacher_temp=0.07
    ):
        """
        Vectorized cross-entropy between teacher and student on masked patches.
        
        Args:
            student_patch_tokens_masked: [B, N, D] student patch tokens
            teacher_patch_tokens_masked: [B, N, D] teacher patch tokens
            student_masks_flat: [B, N] boolean mask (True = masked)
            n_masked_patches: Optional number of masked patches (unused, for compatibility)
            masks_weight: Optional [B] per-sample weights (1/num_masked per sample)
            teacher_temp: Teacher temperature
            
        Returns:
            Loss value
        """
        B, N, D = student_patch_tokens_masked.shape
        device = student_patch_tokens_masked.device
        dtype = student_patch_tokens_masked.dtype
        
        # Flatten everything for vectorized operations
        student_flat = student_patch_tokens_masked.reshape(B * N, D)
        teacher_flat = teacher_patch_tokens_masked.reshape(B * N, D)
        mask_flat = student_masks_flat.reshape(-1)  # [B*N]
        
        # SINGLE nonzero call (was B calls in the loop)
        masked_indices = mask_flat.nonzero(as_tuple=True)[0]
        
        M = masked_indices.numel()
        if M == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Gather all masked tokens at once
        student_masked = student_flat[masked_indices]  # [M, D]
        teacher_masked = teacher_flat[masked_indices]  # [M, D]
        
        # Compute per-token weights
        # Which sample each masked token belongs to
        sample_idx = masked_indices // N  # [M]
        
        if masks_weight is None:
            # Per-sample normalization: weight = 1 / num_masked_in_sample
            num_masked_per_sample = mask_flat.reshape(B, N).sum(dim=1).clamp(min=1.0)  # [B]
            weights = 1.0 / num_masked_per_sample[sample_idx]  # [M]
        else:
            # masks_weight is [B] per-sample weights, map to per-token
            weights = masks_weight[sample_idx]  # [M]
        
        # Sinkhorn-Knopp normalization on teacher
        teacher_normalized = self.sinkhorn_knopp_normalization(teacher_masked, teacher_temp)
        
        # Student log probabilities
        student_log_probs = F.log_softmax(student_masked / self.student_temp, dim=-1)
        
        # Cross-entropy loss per token
        loss_per_token = -torch.sum(teacher_normalized.detach() * student_log_probs, dim=-1)  # [M]
        
        # Weighted mean: sum(loss * weight) / B gives equal contribution per sample
        weighted_loss = (loss_per_token * weights).sum() / B
        
        return weighted_loss

    def forward_gathered(
        self,
        student_proj,
        teacher_proj,
        weights,
        batch_size,
        teacher_temp=0.07
    ):
        """
        Compute iBOT loss on pre-gathered, pre-projected masked tokens.
        Use this when tokens have already been gathered and projected through
        the patchhead externally (to avoid projecting all B*N tokens).
        
        Mathematically identical to forward_masked — just skips the internal
        gather step since caller already did it.
        
        Args:
            student_proj: [M, out_dim] student patchhead output for masked tokens
            teacher_proj: [M, out_dim] teacher patchhead output for masked tokens
            weights: [M] per-token weights (typically 1/num_masked_per_sample)
            batch_size: int, B, for loss normalization
            teacher_temp: Teacher temperature for Sinkhorn-Knopp
            
        Returns:
            Loss value (scalar)
        """
        M = student_proj.shape[0]
        device = student_proj.device
        dtype = student_proj.dtype
        
        if M == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Sinkhorn-Knopp normalization on teacher
        teacher_normalized = self.sinkhorn_knopp_normalization(teacher_proj, teacher_temp)
        
        # Student log probabilities
        student_log_probs = F.log_softmax(student_proj / self.student_temp, dim=-1)
        
        # Cross-entropy loss per token
        loss_per_token = -torch.sum(teacher_normalized.detach() * student_log_probs, dim=-1)  # [M]
        
        # Weighted mean: sum(loss * weight) / B
        weighted_loss = (loss_per_token * weights).sum() / batch_size
        
        return weighted_loss

    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, teacher_output, teacher_temp, n_iterations=None):
        """Apply Sinkhorn-Knopp normalization to teacher outputs.

        Memory note: the fp32 copy from .float() is a fresh tensor we own,
        so divide and exp are done in-place on that copy to avoid two
        ~6 GB transient allocations at out_dim=65536, M~22k.
        """
        if n_iterations is None:
            n_iterations = self.n_iterations

        teacher_output = teacher_output.float()

        Q = teacher_output.div_(teacher_temp).exp_().t()
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        B = Q.shape[1] * world_size
        K = Q.shape[0]
        
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        
        for it in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B
        return Q.t()