"""
Modern Vision Transformer with xformers memory-efficient attention.
Supports sequence packing, register tokens, and masking for DINOv2.
"""

import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchDropout, trunc_normal_

import xformers.ops as xops
from xformers.ops import fmha

from timm.models._manipulate import checkpoint_seq

# Cache for attention bias to avoid recomputation
attn_bias_cache = {}


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Per-sample stochastic depth. No-op in eval or when drop_prob=0."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth (drops whole residual branch)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_attn_bias_and_cat(x_list, branges=None):
    """
    Pack multiple sequences and create block-diagonal attention mask.
    
    Args:
        x_list: List of tensors, each [B,N,D], N can differ
        branges: Optional batch ranges for stochastic depth
        
    Returns:
        attn_bias: Block diagonal matrix (BlockDiagonalMask)
        cat_tensors: Packed tensor [1, total_tokens, D]
    """
    batch_sizes = [x.shape[0] for x in x_list]
    
    # Create tuple of shapes for caching
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    
    if all_shapes not in attn_bias_cache:
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    
    # Concatenate all tensors
    tensors_bs1 = tuple(x.reshape(1, -1, x.shape[-1]) for x in x_list)
    cat_tensors = torch.cat(tensors_bs1, dim=1)
    
    return attn_bias_cache[all_shapes], cat_tensors


class PatchEmbed(nn.Module):
    """Convert image into patch embeddings with optional dual normalization."""
    def __init__(
        self, 
        img_size, 
        patch_size, 
        in_channels=3, 
        embed_dim=768, 
        dual_norm=False, 
        norm_layer=None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size)**2
        self.dual_norm = dual_norm
        
        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        if dual_norm and norm_layer is not None:
            self.pre_norm = norm_layer(in_channels)
            self.post_norm = norm_layer(embed_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.dual_norm:
            x_flat = x.flatten(2).transpose(1, 2)
            x_flat = self.pre_norm(x_flat)
            x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.project(x)
        x = x.flatten(2).transpose(1, 2)
        
        if self.dual_norm:
            x = self.post_norm(x)
            
        return x


class SwiGLUFFN(nn.Module):
    """Optimized SwiGLU implementation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU with optimized hidden dimension sizing."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class Mlp(nn.Module):
    """Standard MLP FFN (Linear -> act -> Linear). DINOv2 ssl_default uses this
    in place of SwiGLU. Drop-in for `mlp_layer`: same call signature as
    SwiGLUFFNFused (in_features / hidden_features / out_features / drop / bias);
    act_layer defaults to GELU since the TransformerBlock call does not pass it."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with xformers attention and SwiGLU MLP."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=None,
        mlp_bias=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, bias=False)
            self.k_norm = nn.LayerNorm(self.head_dim, bias=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_layer = mlp_layer or SwiGLUFFNFused
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=proj_drop,
            bias=mlp_bias,
        )
        
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: Either [B,N,D] for regular or [1,total_tokens,D] for packed
            attn_bias: None for regular, BlockDiagonalMask for packed
        """
        shortcut = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        
        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            scale=self.scale
        )
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.gamma_1 is not None:
            x = shortcut + self.drop_path(self.gamma_1 * x)
        else:
            x = shortcut + self.drop_path(x)
        
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class VisionTransformer(nn.Module):
    """
    Modern Vision Transformer with:
    - XFormers memory-efficient attention
    - SwiGLU MLP activation
    - Dynamic position embeddings
    - Register tokens
    - Mask token for DINOv2 training
    - Sequence packing support
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        mlp_bias=True,
        qk_norm=False,
        dual_norm=False,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4,
        drop_path_uniform=False,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        block_fn=TransformerBlock,
        mlp_layer=SwiGLUFFNFused,
        num_register_tokens=4,
        layerscale_init=None,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0 
        self.no_embed_class = no_embed_class
        self.numregisters = num_register_tokens

        # Mask token for DINOv2
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            dual_norm=dual_norm,
            norm_layer=norm_layer,
        )

        self.grad_checkpointing = False
        
        if isinstance(img_size, tuple):
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            num_patches = (img_size // patch_size) ** 2

        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        # Position embeddings
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        
        # Dropouts
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.patch_drop = PatchDropout(
            patch_drop_rate,
            num_prefix_tokens=self.num_prefix_tokens,
        ) if patch_drop_rate > 0 else nn.Identity()
        
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Stochastic depth
        if drop_path_uniform:
            dpr = [drop_path_rate] * depth   # canonical DINOv2: flat across depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # CaiT ramp

        # LayerScale initialization
        if layerscale_init is not None:
            layer_init_values = [layerscale_init] * depth
        else:
            layer_init_values = []
            for i in range(depth):
                if depth < 18:
                    layer_init_values.append(0.1)
                elif depth < 24:
                    layer_init_values.append(1e-5)
                else:
                    layer_init_values.append(1e-6)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                mlp_bias=mlp_bias,
                qk_norm=qk_norm,
                init_values=layer_init_values[i],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()

        if weight_init != "skip":
            self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)
        
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "register_tokens", "mask_token"}
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def interpolate_pos_embed(self, x, h, w):
        """Interpolate position embeddings for different image sizes."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if num_patches == N:
            return self.pos_embed
            
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        w0, h0 = w0 + 0.1, h0 + 0.1
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens_with_masks(self, x, token_masks=None):
        """Prepare tokens with optional masking for DINOv2."""
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        if token_masks is not None:
            x = torch.where(token_masks.unsqueeze(-1), 
                            self.mask_token.to(x.dtype).unsqueeze(0), 
                            x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_embed(x, H, W)
        
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((x[:, 0:1], reg_tokens, x[:, 1:]), dim=1)
        
        return self.pos_drop(x)

    def prepare_tokens(self, x):
        """Prepare tokens with register tokens and position embeddings."""
        B, C, H, W = x.shape 
        
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1) if self.cls_token is not None else torch.zeros(B, 0, self.embed_dim, device=x.device)
        x = torch.cat((cls_token, x), dim=1) if self.cls_token is not None else x
        
        pos_embed = self.interpolate_pos_embed(x, H, W)
        x = x + pos_embed
        
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        if self.cls_token is not None:
            x = torch.cat((x[:, 0:1], reg_tokens, x[:, 1:]), dim=1)
        else:
            x = torch.cat((reg_tokens, x), dim=1)
        
        return self.pos_drop(x)

    def get_intermediate_layers(self, x):
        """Extract features at specified points in the network."""
        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []
        total_blocks = len(self.blocks)
        extraction_points = [
            (total_blocks // 4) - 1, 
            (total_blocks // 2) - 1, 
            (3 * total_blocks // 4) - 1, 
            total_blocks - 1
        ]

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in extraction_points:
                features.append(x)

        return features        
    
    def forward_features(self, x):
        """Forward pass through features."""
        x = self.prepare_tokens(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        x = self.blocks(x)
            
        x = self.norm(x)
        return x

    def forward_features_list(self, x, masks_list):
        """
        Process multiple crops using sequence packing.
        
        Args:
            x: list of image tensors
            masks_list: list of mask tensors (can be None)
            
        Returns:
            List of output dictionaries, one per crop type
        """
        x_packed = []
        for img, masks in zip(x, masks_list):
            if masks is not None:
                x_prep = self.prepare_tokens_with_masks(img, masks)
            else:
                x_prep = self.prepare_tokens(img)
            x_packed.append(x_prep)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            # Checkpoint-friendly version
            for blk in self.blocks:
                attn_bias, x_cat = get_attn_bias_and_cat(x_packed)
                x_cat = torch.utils.checkpoint.checkpoint(
                    blk,
                    x_cat,
                    attn_bias,
                    use_reentrant=False
                )
                x_packed = attn_bias.split(x_cat)
        else:
            for blk in self.blocks:
                attn_bias, x_cat = get_attn_bias_and_cat(x_packed)
                x_cat = blk(x_cat, attn_bias=attn_bias)
                x_packed = attn_bias.split(x_cat)

        outputs = []
        for x, masks in zip(x_packed, masks_list):
            x_norm = self.norm(x)
            outputs.append({
                "clstoken": x_norm[:, 0],
                "regtokens": x_norm[:, 1:self.numregisters+1],
                "patchtokens": x_norm[:, self.numregisters+1:],
                "masks": masks
            })

        return outputs

    def forward(self, x, token_masks=None, return_dict: bool = False):
        """
        Forward with automatic detection of single vs multi-crop input.

        Args:
            x: Either single tensor [B, C, H, W] or list of tensors
            token_masks: Either None, single mask, or list of masks
            return_dict: Single-image path only. False (default) returns the
                pre-self.norm CLS token of shape [B, D]. True returns a dict
                with prenorm and postnorm versions of CLS, patches, and
                register tokens. Ignored on the list-input path.

        Returns:
            - If list: List of dicts (unchanged)
            - If single, return_dict=False: Tensor [B, D] (pre-norm CLS)
            - If single, return_dict=True: Dict with prenorm/postnorm tokens
        """
        if isinstance(x, list):
            if token_masks is None:
                token_masks = [None] * len(x)
            elif not isinstance(token_masks, list):
                token_masks = [token_masks] + [None] * (len(x) - 1)
            
            return self.forward_features_list(x, token_masks)
        
        else:
            if token_masks is not None:
                x = self.prepare_tokens_with_masks(x, token_masks)
            else:
                x = self.prepare_tokens(x)
            
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                for blk in self.blocks:
                    x = blk(x, attn_bias=None)

            # x at this point is the post-blocks residual stream (pre-self.norm).
            if not return_dict:
                return x[:, 0]  # pre-norm CLS, [B, D]

            x_norm = self.norm(x)
            return {
                'clstoken_prenorm':     x[:, 0],
                'clstoken_postnorm':    x_norm[:, 0],
                'patchtokens_prenorm':  x[:, self.numregisters + 1:],
                'patchtokens_postnorm': x_norm[:, self.numregisters + 1:],
                'regtokens_prenorm':    x[:, 1:self.numregisters + 1],
                'regtokens_postnorm':   x_norm[:, 1:self.numregisters + 1],
                'masks':                token_masks,
            }




class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # Clustering layers inspired by SwAV
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

