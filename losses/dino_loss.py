"""
DINO CLS token loss with Sinkhorn-Knopp normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    """
    DINO loss with teacher temperature scheduling and Sinkhorn-Knopp normalization.
    
    Args:
        ncrops: Number of crops
        warmup_teacher_temp: Initial teacher temperature
        teacher_temp: Final teacher temperature
        warmup_teacher_temp_iters: Warmup iterations for temperature
        student_temp: Student temperature (fixed)
        n_iterations: Sinkhorn-Knopp iterations
    """
    def __init__(
        self, 
        ncrops, 
        warmup_teacher_temp, 
        teacher_temp,
        warmup_teacher_temp_iters, 
        student_temp=0.1,
        n_iterations=3
    ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.n_iterations = n_iterations
                
        warmup_iters = float(warmup_teacher_temp_iters)
        self.teacher_temp_schedule = lambda it: teacher_temp + (warmup_teacher_temp - teacher_temp) * \
            (1 + math.cos(math.pi * min(it, warmup_iters) / warmup_iters)) / 2
    
    def forward(self, student_output, teacher_output, current_iteration,
                sample_temperatures=None, sample_weights=None):
        """
        Compute DINO loss.

        Args:
            student_output: Student predictions [B*ncrops, out_dim]
            teacher_output: Teacher predictions [B*2, out_dim] (2 global crops)
            current_iteration: Current training iteration
            sample_temperatures: Optional per-sample temperature tensor of shape [B].
                If provided, applied per-sample instead of the scalar student_temp.
                Used by the typicality dampening 'adaptive_temp' modulation.
            sample_weights: Optional per-sample weight tensor of shape [B].
                If provided, the per-sample loss is weighted before averaging
                (weighted sum divided by weight sum). Used by the typicality
                dampening 'weighted_loss' modulation.

        Returns:
            Loss value
        """
        temp = self.teacher_temp_schedule(current_iteration)

        normalized_teacher = self.sinkhorn_knopp_normalization(teacher_output, temp)
        normalized_teacher = normalized_teacher.detach().chunk(2)

        # Temperature: per-sample or scalar
        if sample_temperatures is not None:
            tau_repeated = sample_temperatures.repeat(self.ncrops)
            student_out = student_output / tau_repeated.unsqueeze(1)
        else:
            student_out = student_output / self.student_temp

        student_out = student_out.chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(normalized_teacher):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)

                if sample_weights is not None:
                    total_loss += (loss * sample_weights).sum() / sample_weights.sum()
                else:
                    total_loss += loss.mean()

                n_loss_terms += 1

        total_loss /= n_loss_terms
        return total_loss
    
    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, teacher_output, teacher_temp, n_iterations=None):
        """Apply Sinkhorn-Knopp normalization."""
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        Q = torch.exp(teacher_output / teacher_temp).t()
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