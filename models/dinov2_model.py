"""
Combined model for DINOv2 with iBOT using sequence packing.
Delegates packing to the backbone for efficient multi-crop processing.
"""

import torch
import torch.nn as nn


class CombinedModelDINO(nn.Module):
    """
    Combined model for DINOv2 with iBOT using sequence packing.
    Now delegates packing to the backbone.
    
    Args:
        backbone: Vision Transformer backbone
        classhead: Projection head for CLS tokens (DINO)
        patchhead: Projection head for patch tokens (iBOT)
        num_masks: Number of semantic masks (legacy, kept for compatibility)
        patch_size: Patch size of the backbone
    """
    def __init__(self, backbone, classhead, patchhead, num_masks=6, patch_size=16):
        super().__init__()
        
        # Remove fc and head if they exist
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
            
        self.backbone = backbone
        self.classhead = classhead
        self.patchhead = patchhead
        self.num_masks = num_masks
        self.patch_size = patch_size
    
    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing in the backbone."""
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
            print(f"✓ Gradient checkpointing {'enabled' if enable else 'disabled'} in CombinedModelDINO backbone")
        else:
            print(f"⚠ Warning: Backbone does not support gradient checkpointing")

    def forward(self, crops, token_masks=None, mode='dino', return_bottleneck=False,
                bottleneck_only=False):
        """
        Unified forward supporting both DINO and iBOT modes.

        Args:
            crops: Either:
                - List of tensors [crop1, crop2, ...] for multi-crop (DINO)
                - Single tensor [B, C, H, W] for single image (iBOT)
            token_masks: Optional token masks for iBOT
                - List of masks matching crops (or list of None)
                - Single mask tensor [B, N] for iBOT
                - None for no masking
            mode: 'dino' or 'ibot' (mostly for clarity)
            return_bottleneck: If True (multi-crop / DINO path only), also return
                the pre-prototype bottleneck representation from the classhead.
                Used by the typicality dampening feature.
            bottleneck_only: If True (multi-crop / DINO path only), return ONLY the classhead
                bottleneck and SKIP the out_dim prototype layer entirely. The returned bottleneck
                is bit-identical to the one from return_bottleneck=True. Used by the thinned
                scout forward, which consumes nothing but the bottleneck.

        Returns:
            Dictionary with keys depending on mode:
            - DINO: {'cls_outputs': tensor, 'features_list': list of dicts[, 'bottleneck': tensor]}
            - DINO + bottleneck_only: {'bottleneck': tensor, 'features_list': list of dicts}
              (no 'cls_outputs' key -- the prototype projection is never computed)
            - iBOT: {'patch_outputs': tensor, 'features': dict, 'cls_output': tensor}
        """

        # Determine if we're doing multi-crop (list input) or single image
        is_multi_crop = isinstance(crops, list)

        if is_multi_crop:
            # ========== MULTI-CROP MODE (DINO) ==========
            # Ensure masks is also a list
            if token_masks is None:
                token_masks = [None] * len(crops)
            elif not isinstance(token_masks, list):
                # Single mask provided, assume it's for first crop
                token_masks = [token_masks] + [None] * (len(crops) - 1)

            # Forward through backbone with packing
            # backbone.forward() will detect list and call forward_features_list()
            outputs_list = self.backbone(crops, token_masks=token_masks)
            # outputs_list: [{'clstoken': [B,D], 'patchtokens': [B,N,D], ...}, ...]

            # Collect all CLS tokens and apply head
            all_cls_tokens = []
            for output_dict in outputs_list:
                all_cls_tokens.append(output_dict['clstoken'])

            # Concatenate all CLS tokens: [B1+B2+...+BN, D]
            cls_tokens_cat = torch.cat(all_cls_tokens, dim=0)

            # Apply DINO head
            if bottleneck_only:
                # Scout path: mlp + normalize only, no out_dim prototype matmul.
                return {
                    'bottleneck': self.classhead.forward_bottleneck(cls_tokens_cat),
                    'features_list': outputs_list,
                }
            if return_bottleneck:
                cls_outputs, bottleneck = self.classhead(cls_tokens_cat, return_bottleneck=True)
            else:
                cls_outputs = self.classhead(cls_tokens_cat)

            result = {
                'cls_outputs': cls_outputs,  # [total_crops, out_dim]
                'features_list': outputs_list,  # List of dicts
            }
            if return_bottleneck:
                result['bottleneck'] = bottleneck
            return result
        
        else:
            # ========== SINGLE IMAGE MODE (iBOT) ==========
            # crops is a single tensor [B, C, H, W]

            # Forward through backbone
            output_dict = self.backbone(crops, token_masks=token_masks, return_dict=True)

            # Apply heads (postnorm tokens preserve the previous head input dist).
            cls_output = self.classhead(output_dict['clstoken_postnorm'])
            patch_outputs = self.patchhead(output_dict['patchtokens_postnorm'])
            
            return {
                'cls_output': cls_output,  # [B, out_dim]
                'patch_outputs': patch_outputs,  # [B, N, out_dim]
                'features': output_dict,  # Full feature dict
            }