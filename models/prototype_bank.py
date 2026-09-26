"""
Prototype Bank Module for patch-level clustering.
CAPI-style soft prototypes using a linear layer.
"""

import torch
import torch.nn as nn


class LinearPrototypeBank(nn.Module):
    """
    CAPI-style soft prototypes using a linear layer.
    The weight matrix acts as learnable prototypes.
    
    Args:
        num_prototypes: Number of prototype vectors
        embed_dim: Dimension of embeddings
        bias: Whether to use bias in linear layer
    """
    def __init__(self, num_prototypes=8192, embed_dim=768, bias=False):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        
        # Linear layer where weights are the "prototypes"
        self.proto_layer = nn.Linear(embed_dim, num_prototypes, bias=bias)
        
        # Initialize weights with reasonable scale
        nn.init.normal_(self.proto_layer.weight, std=1)
        if bias:
            nn.init.zeros_(self.proto_layer.bias)
    
    def forward(self, x):
        """
        Compute similarities to prototypes.
        
        Args:
            x: [B, N, D] or [M, D] normalized patch features
            
        Returns:
            logits: [B, N, K] or [M, K] similarities to prototypes
        """
        return self.proto_layer(x)
    
    def get_stats(self):
        """Return statistics for monitoring."""
        with torch.no_grad():
            weight = self.proto_layer.weight
            weight_norms = weight.norm(dim=1)
            return {
                'weight_norm_mean': weight_norms.mean().item(),
                'weight_norm_std': weight_norms.std().item(),
                'weight_norm_min': weight_norms.min().item(),
                'weight_norm_max': weight_norms.max().item(),
            }