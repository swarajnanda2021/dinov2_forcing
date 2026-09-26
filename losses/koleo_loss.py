"""
KoLeo regularization loss.
Encourages uniform distribution of features in embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """
    KoLeo loss for preventing feature collapse.
    Encourages features to be uniformly distributed.
    """
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
    
    def pairwise_NNs_inner(self, x):
        """Find nearest neighbors using inner product."""
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)
        _, I = torch.max(dots, dim=1)
        return I
    
    def forward(self, student_output, eps=1e-8):
        """
        Compute KoLeo loss.
        
        Args:
            student_output: Feature vectors [B, D]
            eps: Small constant for numerical stability
            
        Returns:
            Loss value encouraging uniform distribution
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[I])
            loss = -torch.log(distances + eps).mean()
            return loss