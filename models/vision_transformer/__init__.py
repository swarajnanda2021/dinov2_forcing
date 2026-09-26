"""
Vision Transformer implementations.
Modern xformers-based ViT plus the DINO projection head.
"""

from .modern_vit import VisionTransformer, DINOHead

__all__ = [
    'VisionTransformer',
    'DINOHead',
]