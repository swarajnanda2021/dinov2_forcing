"""
Patch prototype clustering loss with CAPI-inspired doubly stochastic Sinkhorn-Knopp.
Uses global processing (all tokens together) with doubly stochastic constraints
to eliminate positional bias while maintaining efficient computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .koleo_loss import KoLeoLoss


class PatchPrototypeLoss(nn.Module):
    """
    CAPI-inspired patch clustering loss using doubly stochastic optimal transport.
    
    Key differences from original:
    - Doubly stochastic constraints (both samples and prototypes sum to 1)
    - Improved numerical stability across dtypes
    - Epsilon protection against division by zero
    - Eliminates spatial positional bias in prototype assignments
    
    Args:
        num_prototypes: Number of prototype vectors
        embed_dim: Embedding dimension
        teacher_temp: Teacher temperature
        student_temp: Student temperature
    """
    def __init__(
        self, 
        num_prototypes=8192, 
        embed_dim=768,
        teacher_temp=0.07, 
        student_temp=0.1
    ):
        super().__init__()
        
        self.k = num_prototypes
        self.embed_dim = embed_dim
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        
        # KoLeo for preventing collapse
        self.koleo_loss = KoLeoLoss()

        # Expensive diagnostics (bincount over K + a usage all-reduce) are computed
        # only every metric_log_freq steps; self.last_* hold the most recent value.
        self.metric_log_freq = 20

        # Tracking metrics
        self.last_prediction_loss = 0.0
        self.last_arrangement_loss = 0.0
        self.last_koleo_loss = 0.0
        self.last_entropy = 0.0
        self.last_usage_std = 0.0
        self.last_num_masked = 0
    
    def teacher_targets(self, teacher_patch_tokens, prototype_bank, teacher_temp, with_koleo=True):
        """Teacher path ONLY: Sinkhorn assignments Q + arrangement loss (+ optional koleo).

        Depends solely on the teacher tokens and the bank, NOT on the student view or
        mask -- so it is computed ONCE per distinct teacher input and reused. The
        semantic-prototype call feeds the same teacher_patch_tokens_g1, so reusing g1's
        Q here is bit-identical to recomputing it, and avoids a full bank matmul + a
        Sinkhorn (4 collectives).

        Returns: (Q_tilde_all [B,N,K], teacher_proto_loss, koleo_proto_loss_or_None)
        """
        B, N, D = teacher_patch_tokens.shape
        teacher_norm = F.normalize(teacher_patch_tokens, p=2, dim=-1)
        teacher_logits_all = prototype_bank(teacher_norm)               # [B, N, K]
        teacher_logits_flat = teacher_logits_all.reshape(B * N, -1)
        with torch.no_grad():
            Q_tilde_all = self.sinkhorn_knopp(teacher_logits_flat, teacher_temp).reshape(B, N, -1)
        teacher_log_probs_all = F.log_softmax(teacher_logits_all / teacher_temp, dim=-1)
        teacher_proto_loss = -torch.sum(Q_tilde_all.detach() * teacher_log_probs_all) / (B * N)
        koleo_proto_loss = None
        if with_koleo:
            weight_normalized = F.normalize(prototype_bank.module.proto_layer.weight, p=2, dim=1)
            koleo_proto_loss = self.koleo_loss(weight_normalized)
        return Q_tilde_all, teacher_proto_loss, koleo_proto_loss

    def student_prediction(self, student_patch_tokens, token_masks, Q_tilde_all,
                           prototype_bank, current_iteration, masks_weight=None):
        """Student prediction loss given PRECOMPUTED teacher targets Q_tilde_all [B,N,K].

        This is the only part that differs between the block-mask and semantic calls
        (different student view + mask), so only this is recomputed for semantic."""
        B, N, D = student_patch_tokens.shape
        student_norm = F.normalize(student_patch_tokens, p=2, dim=-1)

        Q_tilde_masked = Q_tilde_all[token_masks].detach()             # [M, K]
        student_norm_masked = student_norm[token_masks]                # [M, D]
        M_total = student_norm_masked.shape[0]
        if M_total == 0:
            return torch.tensor(0.0, device=student_patch_tokens.device), 0

        student_logits_masked = prototype_bank(student_norm_masked)    # [M, K]
        student_log_probs_masked = F.log_softmax(student_logits_masked / self.student_temp, dim=-1)
        per_token_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked, dim=-1)  # [M]

        if masks_weight is not None:
            sample_indices = token_masks.nonzero(as_tuple=True)[0]     # [M] sample idx per masked token
            per_token_weight = masks_weight[sample_indices]
            clustering_loss = (per_token_loss * per_token_weight).sum() / B
        else:
            clustering_loss = per_token_loss.mean()

        # cheap scalar (kept every step); expensive diagnostics gated below
        self.last_prediction_loss = clustering_loss.item()
        self.last_num_masked = M_total
        if current_iteration % self.metric_log_freq == 0:
            with torch.no_grad():
                student_probs = torch.exp(student_log_probs_masked)
                entropy = -(student_probs * student_log_probs_masked).sum(dim=-1).mean()
                self.last_entropy = (entropy / math.log(self.k)).item()
                assignments = torch.argmax(Q_tilde_masked, dim=-1)
                usage = torch.bincount(assignments, minlength=self.k).float()
                if dist.is_initialized():
                    dist.all_reduce(usage)                             # collective: log-steps only
                self.last_usage_std = (usage.std() / (usage.mean() + 1e-6)).item()
        return clustering_loss, M_total

    def forward(self, teacher_patch_tokens, student_patch_tokens, token_masks,
                prototype_bank, current_iteration, teacher_temp, masks_weight=None):
        """Full path (teacher targets + student prediction). Numerically unchanged vs
        the pre-refactor loss; additionally returns Q_tilde_all so callers with the same
        teacher input (semantic prototype) can reuse it instead of recomputing.

        Returns: (clustering_loss, teacher_proto_loss, koleo_proto_loss, Q_tilde_all)
        """
        Q_tilde_all, teacher_proto_loss, koleo_proto_loss = self.teacher_targets(
            teacher_patch_tokens, prototype_bank, teacher_temp, with_koleo=True)
        self.last_arrangement_loss = teacher_proto_loss.item()
        self.last_koleo_loss = koleo_proto_loss.item()

        clustering_loss, M_total = self.student_prediction(
            student_patch_tokens, token_masks, Q_tilde_all, prototype_bank,
            current_iteration, masks_weight=masks_weight)

        # Preserve the original M==0 semantics: arrangement + koleo are zeroed when the
        # crop has no masked tokens (the pre-refactor forward early-returned all zeros;
        # M_total is derived from the masked-select shape, no extra host sync).
        if M_total == 0:
            z = torch.tensor(0.0, device=teacher_patch_tokens.device)
            return z, z, z, Q_tilde_all
        return clustering_loss, teacher_proto_loss, koleo_proto_loss, Q_tilde_all

    @torch.no_grad()
    def sinkhorn_knopp(self, teacher_output, teacher_temp, n_iterations=3, eps=1e-8):
        """
        CAPI-inspired doubly stochastic Sinkhorn-Knopp.
        Matches the official CAPI implementation exactly.
        
        Args:
            teacher_output: [M, K] logits where M = B*N tokens
            teacher_temp: Temperature for softmax
            n_iterations: Number of SK iterations
            eps: Epsilon for numerical stability
            
        Returns:
            Q: [M, K] doubly stochastic assignment matrix
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Numerical stability: shift by global max before exp
        M = teacher_output / teacher_temp
        M_max = M.max()
        if dist.is_initialized():
            dist.all_reduce(M_max, op=dist.ReduceOp.MAX)
        M = M - M_max
        
        # Transpose for easier iteration: [K, M]
        Q = torch.exp(M).t()
        
        # Doubly stochastic iterations (matches CAPI exactly)
        for _ in range(n_iterations):
            # Normalize over samples (each prototype distribution sums to 1)
            # All-reduce needed because samples are distributed across GPUs
            sum_over_samples = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_over_samples)
            Q /= (sum_over_samples + eps)
            
            # Normalize over prototypes (each sample distribution sums to 1)
            # No all-reduce needed because prototypes are replicated
            sum_over_prototypes = torch.sum(Q, dim=0, keepdim=True)
            Q /= (sum_over_prototypes + eps)
        
        # Transpose back to [M, K]
        return Q.t()