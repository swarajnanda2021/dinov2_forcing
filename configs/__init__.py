"""
Configuration management for DINOv2 training.
"""

from .config import get_args_parser, apply_vit_variant, VIT_CONFIGS

__all__ = ['get_args_parser', 'apply_vit_variant', 'VIT_CONFIGS']