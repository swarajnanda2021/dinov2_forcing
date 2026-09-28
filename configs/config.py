"""
Argument parser and configuration for DINOv2 training.
"""

import argparse
import utils


# Standard DINOv2 sizes: each row fixes (embeddingdim, vitdepth, vitheads).
VIT_CONFIGS = {
    "S": dict(embeddingdim=384,  vitdepth=12, vitheads=6),
    "B": dict(embeddingdim=768,  vitdepth=12, vitheads=12),
    "L": dict(embeddingdim=1024, vitdepth=24, vitheads=16),
    "H": dict(embeddingdim=1280, vitdepth=32, vitheads=16),
    "G": dict(embeddingdim=1536, vitdepth=40, vitheads=24),
}


def apply_vit_variant(args):
    """If args.vit_variant is set, overwrite embeddingdim / vitdepth / vitheads from VIT_CONFIGS."""
    variant = getattr(args, 'vit_variant', None)
    if variant:
        for k, v in VIT_CONFIGS[variant].items():
            setattr(args, k, v)
    depth = getattr(args, 'depth', None)
    if depth is not None:
        args.vitdepth = depth          # --depth overrides the variant's block count (width, heads unchanged)
    return args


def get_args_parser():
    """
    Create argument parser with all training configuration options.

    Returns:
        ArgumentParser with all training arguments
    """
    parser = argparse.ArgumentParser('DINOv2 forcing study', add_help=False)

    # ========== Model parameters ==========
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for vision transformer')
    parser.add_argument('--embeddingdim', default=768, type=int,
                        help='Embedding dimension')
    parser.add_argument('--vitheads', default=12, type=int,
                        help='Number of attention heads')
    parser.add_argument('--vitdepth', default=12, type=int,
                        help='Number of transformer blocks')
    parser.add_argument('--vit_variant', default=None, type=str, choices=list(VIT_CONFIGS),
                        help='ViT size (S/B/L/H/G). When set, overrides --embeddingdim, '
                             '--vitdepth and --vitheads with the standard DINOv2 values.')
    parser.add_argument('--depth', default=None, type=int,
                        help='Number of transformer blocks. None keeps the variant\'s depth (--vitdepth); '
                             'a value overrides it at the variant\'s width and heads.')
    parser.add_argument('--out_dim', default=65536, type=int,
                        help='Output dimension of projection heads')
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
                        help='Normalize the DINO head last layer (frozen weight-norm trick). '
                             'Default: False (corrected DINOv2 behavior). Pass True to '
                             'reproduce the older runs that used the frozen last layer.')
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help='Use batch normalization in projection head')
    parser.add_argument('--ffn_type', default='swiglu', choices=['swiglu', 'mlp'],
                        help="Transformer FFN type. 'swiglu' (default, current fork) uses "
                             "gated SwiGLU/SiLU. 'mlp' uses a standard Linear->GELU->Linear "
                             "MLP (DINOv2 ssl_default behavior).")
    parser.add_argument("--layerscale_init", default=1e-5, type=float,
                        help="Uniform LayerScale init for ALL blocks. Only consulted when "
                             "--layerscale_schedule=uniform (the default). Default: 1e-5 "
                             "(corrected DINOv2 behavior).")
    parser.add_argument("--layerscale_schedule", default="uniform",
                        choices=["uniform", "cait"],
                        help="LayerScale init policy. 'uniform' (default) uses the constant "
                             "--layerscale_init for every block (corrected DINOv2 behavior). "
                             "'cait' reproduces the old depth-based schedule (0.1 for "
                             "depth<18, 1e-5 for 18<=depth<24, 1e-6 for depth>=24); "
                             "--layerscale_init is ignored in this mode.")
    parser.add_argument('--drop_path_rate', default=0.4, type=float,
                        help='Stochastic depth rate. Default 0.4 preserves the '
                             'historical hard-coded value at the training call '
                             'site. Canonical DINOv2 uses 0.3 for ViT-L with '
                             '--drop_path_uniform=True.')
    parser.add_argument('--drop_path_uniform', default=False, type=utils.bool_flag,
                        help='If True, every block uses drop_path_rate (canonical '
                             'DINOv2 ViT-L). If False (default, current fork '
                             'behavior), use a linear ramp linspace(0, '
                             'drop_path_rate, depth) — block 0 gets 0%, last '
                             'block gets the full rate.')

    # ========== Flexible augmentation parameters ==========
    parser.add_argument('--global_views', default=2, type=int,
                        help='Number of global views')
    parser.add_argument('--n_standard_local_crops', default=3, type=int,
                        help='Number of standard local crops')
    parser.add_argument('--local_crop_size', default=96, type=int,
                        help='Size of local crops')

    # ========== Loss parameters ==========
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help='EMA momentum for teacher update')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help='Initial teacher temperature')
    parser.add_argument('--teacher_temp', default=0.07, type=float,
                        help='Final teacher temperature')
    parser.add_argument('--teacher_temp_warmup_iters', default=30000, type=int,
                        help='Teacher temperature warmup iterations')
    parser.add_argument('--koleo_loss_weight', default=0.1, type=float,
                        help='Weight for KoLeo regularization loss')
    parser.add_argument('--ibot_loss_weight', default=1.0, type=float,
                        help='Weight for iBOT patch loss')
    parser.add_argument('--mask_ratio_min', default=0.1, type=float,
                        help='Minimum mask ratio for iBOT block masking')
    parser.add_argument('--mask_ratio_max', default=0.5, type=float,
                        help='Maximum mask ratio for iBOT block masking')
    parser.add_argument('--mask_sample_probability', default=0.5, type=float,
                        help='Fraction of samples in batch to apply masking')

    # ========== Patch Prototype Clustering parameters ==========
    parser.add_argument('--use_prototype_clustering', default=True, type=utils.bool_flag,
                    help='Enable patch prototype clustering loss')
    parser.add_argument('--num_prototypes', default=8192, type=int,
                        help='Number of prototypes for clustering')
    parser.add_argument('--clustering_weight', default=1.0, type=float,
                        help='Weight for prototype clustering loss')
    parser.add_argument('--clustering_teacher_temp', default=0.07, type=float,
                        help='Teacher temperature for clustering')
    parser.add_argument('--clustering_student_temp', default=0.1, type=float,
                        help='Student temperature for clustering')

    # ========== Training parameters ==========
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--total_iterations', default=300000, type=int,
                        help='Total number of training iterations')
    parser.add_argument('--warmup_iterations', default=10000, type=int,
                        help='Number of warmup iterations')
    parser.add_argument('--freeze_last_layer_iters', default=5000, type=int,
                        help='Freeze last layer for this many iterations')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
                        help='Use mixed precision training')
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help='Gradient clipping value')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Initial weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='Final weight decay')
    # ---- schedule shapes (forcing study knobs) ----
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'constant'],
                        help='cosine: linear warmup then cosine decay to --min_lr. constant: linear '
                             'warmup then hold the peak; --min_lr is ignored.')
    parser.add_argument('--wd_schedule', default='cosine', choices=['cosine', 'constant'],
                        help='cosine: --weight_decay -> --weight_decay_end. constant: --weight_decay '
                             'for the whole run; --weight_decay_end is ignored.')
    parser.add_argument('--momentum_schedule', default='cosine', choices=['cosine', 'constant'],
                        help='cosine: --momentum_teacher -> --momentum_teacher_end. constant: '
                             '--momentum_teacher for the whole run; --momentum_teacher_end is ignored.')
    parser.add_argument('--momentum_teacher_end', default=1.0, type=float,
                        help='Final teacher EMA momentum (cosine momentum schedule only)')
    parser.add_argument('--lr_decay_rate', default=0.9, type=float,
                        help='Layer-wise LR decay rate (1.0 = no decay, 0.9 = typical)')
    parser.add_argument('--patch_embed_lr_mult', default=0.2, type=float,
                        help='LR multiplier applied to the patch_embed param group ONLY '
                             '(on top of layer-wise decay). DINOv2 ssl_default_config uses '
                             '0.2; set 1.0 to disable. Rationale: MoCo v3 patch-projection '
                             'stability.')
    # ---- speed (opt-in; default OFF so nothing changes unless asked) ----
    parser.add_argument('--compile_blocks', default=False, type=utils.bool_flag,
                        help="torch.compile each TransformerBlock (NOT the whole backbone: the "
                             "xformers attn_bias_cache global makes a full-backbone compile fail a "
                             "dynamo guard). Measured -23.5%% on the fwd+bwd phase; ~80 s warm-up. "
                             "Changes fp results at the 1e-3 level (kernel fusion) -- not bit-identical.")
    parser.add_argument('--grad_checkpointing', default=False, type=utils.bool_flag,
                    help='Enable gradient checkpointing to reduce memory at cost of ~40% speed')

    # ========== Dataset and I/O ==========
    parser.add_argument('--dataset_sources', type=str, nargs='+',
                        help='Dataset sources in format NAME:BASE_DIR:INDEX_FILE')
    parser.add_argument('--output_dir', default=".", type=str,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_checkpoint_freq', default=2000, type=int,
                        help='Periodic checkpoint frequency (checkpoint_iter_<it>.pth, kept)')
    parser.add_argument('--rolling_checkpoint_freq', default=5000, type=int,
                        help='Rolling checkpoint frequency (checkpoint.pth, overwritten; the resume file)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')

    # ========== Patch-geometry diagnostics ==========
    parser.add_argument('--diag_every', default=2000, type=int,
                        help='Run the patch-geometry diagnostics at iteration 0 and every this many iterations')
    parser.add_argument('--diag_probe_manifest', default=None, type=str,
                        help='JSON manifest of the fixed probe tiles (created on first use, then shared)')
    parser.add_argument('--diag_probe_size', default=1024, type=int,
                        help='Number of probe tiles drawn from the training stream when the manifest is created')
    parser.add_argument('--diag_ibot_tokens', default=None, type=int,
                        help='Patch tokens per probe tile fed to the iBOT head for ibot_tok_H / ibot_img_H '
                             '(fixed random subset, seed 0). Default: 49 on CUDA, 16 on CPU.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers')

    # ========== Distributed training ==========
    parser.add_argument("--dist_url", default="env://", type=str,
                        help='URL for distributed training setup')
    parser.add_argument("--local_rank", default=0, type=int,
                        help='Local rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')

    parser.add_argument('--qk_norm', default=None, type=utils.bool_flag_or_none,
                        help='Enable QK normalization in attention. If None (default), '
                             'the trainer enables it. Explicit True/False wins.')

    parser.add_argument('--num_register_tokens', default=4, type=int,
                        help='Number of register tokens [Darcet et al. 2023].')

    return parser
