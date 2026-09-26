"""
Main training loop for DINOv2 with iBOT and prototype clustering.
"""

import os
import sys
import time
import datetime
import json
from pathlib import Path
from copy import deepcopy
import gc
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
from models.vision_transformer.modern_vit import Mlp, SwiGLUFFNFused
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, KDELoss, PatchPrototypeLoss
from data import ProportionalMultiDatasetWrapper
from .helpers import (
    load_pretrained_mask_model,
    load_pretrained_cellvit_model,
    apply_masks_to_images,
    apply_cellvit_masks,
    extract_local_crops_from_masked,
    extract_crops_from_cellvit_channel,
    generate_random_image_masks,
    generate_block_masks,
    convert_semantic_masks_to_token_masks,
    calculate_total_student_views,
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)
from typicality import RepresentativePrototypes, TypicalityScorer, CountedCoverageBank
# Vectorized drop-in for the bank's hot path. BIT-IDENTICAL state and return values (verified on
# GPU: full state_dict torch.equal every step, matured bank, 40 steps x 3 shape configs); it only
# removes the per-row cdist/.item() launches in _update (~40k device syncs/step at B=6144).
from typicality.counted_coverage_bank_fast import FastCountedCoverageBank


def _gather_and_compute_weights(patch_tokens, mask, masks_weight=None):
    """
    Gather masked tokens from [B, N, D] using boolean mask [B, N].
    Returns gathered tokens [M, D], per-token weights [M], and batch size B.
    
    Args:
        patch_tokens: [B, N, D] patch tokens
        mask: [B, N] boolean mask (True = masked)
        masks_weight: Optional [B] per-sample weights. If None, uses 1/num_masked.
        
    Returns:
        gathered: [M, D] masked tokens
        weights: [M] per-token weights
        B: batch size
    """
    B, N, D = patch_tokens.shape
    
    mask_flat = mask.reshape(-1)  # [B*N]
    masked_indices = mask_flat.nonzero(as_tuple=True)[0]  # [M]
    M = masked_indices.numel()
    
    if M == 0:
        return None, None, B
    
    gathered = patch_tokens.reshape(B * N, D)[masked_indices]  # [M, D]
    
    # Per-token weights
    sample_idx = masked_indices // N  # [M]
    if masks_weight is not None:
        weights = masks_weight[sample_idx]  # [M]
    else:
        num_masked_per_sample = mask.sum(dim=1).float().clamp(min=1.0)  # [B]
        weights = 1.0 / num_masked_per_sample[sample_idx]  # [M]
    
    return gathered, weights, B


def _all_gather_signatures(s_local):
    """All-gather [B_local, K'] signatures across data-parallel ranks into
    [world_size*B_local, K'] (rank order preserved -> identical on every rank).
    Falls back to the local tensor when distributed is unavailable or world_size == 1."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return s_local
    world = dist.get_world_size()
    gathered = [torch.empty_like(s_local) for _ in range(world)]
    dist.all_gather(gathered, s_local.contiguous())
    return torch.cat(gathered, dim=0)


def _local_rows(x_global, b_local):
    """Take this rank's contiguous B_local rows out of a gathered [world*B_local, ...]
    tensor. Row order matches all_gather: rank r -> rows [r*b_local:(r+1)*b_local].
    Identity when distributed is unavailable or world_size == 1."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x_global
    r = dist.get_rank()
    return x_global[r * b_local:(r + 1) * b_local]


def _assert_bank_synced(bank, tag=""):
    """Cross-rank insurance: the global bank must be byte-identical on every rank.
    Evict-nearest churn (topk/argmin + the CPU dedup dict) is deterministic only if all
    ranks see the identical gathered batch AND dict-insertion order is fixed; this checks
    it held. Compares a (sum, sum-of-squares) fingerprint via MIN/MAX all-reduce. No-op on
    a single rank; runtime output comes from a >=2-rank run. If it ever prints False, make
    the dedup order-deterministic (sort claimed keys / fix tie-breaks) before trusting the
    global bank."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    fp = torch.stack([bank.double().sum(), bank.double().pow(2).sum()])
    lo = fp.clone(); hi = fp.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if utils.is_main_process():
        ok = bool(torch.allclose(lo, hi))
        print(f"[typicality bank sync{tag}] cross-rank identical: {ok} "
              f"(fp_min={lo.tolist()}, fp_max={hi.tolist()})")


def _thin_profile_err(committed_phat, pool_phat, p_ref, a=0.5, c_frac=0.25, nbins=20):
    """Cheap binned-histogram L1 between the achieved committed-batch p_hat distribution and the
    target thinned profile a*mu (section 3.9). mu is the candidate-pool p_hat histogram; the target
    weights each bin by a(p) = w/w_max with w = (p + c_frac*p_ref)^-a. Bins over log10(p_hat).
    a / c_frac MUST match the admission tilt (args.typicality_a / args.typicality_c_frac) or the
    target is built for the wrong tilt and the metric is meaningless. Both histograms are normalised
    to sum 1; returns the L1 distance (0 = perfect match)."""
    eps = 1e-8
    lp_pool = torch.log10(pool_phat + eps)
    lo = float(lp_pool.min().item()); hi = float(lp_pool.max().item())
    if not (hi > lo):
        return 0.0
    mu = torch.histc(lp_pool, bins=nbins, min=lo, max=hi)
    com = torch.histc(torch.log10(committed_phat + eps), bins=nbins, min=lo, max=hi)
    centers = torch.linspace(lo, hi, nbins, device=pool_phat.device) + 0.5 * (hi - lo) / nbins
    w_c = 1.0 / (10.0 ** centers + c_frac * p_ref).clamp(min=eps).pow(a)
    a_c = w_c / w_c.max().clamp(min=eps)
    target = a_c * mu
    com_n = com / com.sum().clamp(min=eps)
    tgt_n = target / target.sum().clamp(min=eps)
    return float((com_n - tgt_n).abs().sum().item())


def train_dinov2(args):
    """
    Main training function for DINOv2 with iBOT and prototype clustering.

    Args:
        args: Training arguments namespace
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))

    # ============ Pathology FM recipe: resolve overrides and auto-gate ============
    # Applied BEFORE model/loss/optimizer construction so downstream uses the
    # effective values. No-op unless args.use_pathology_recipe=True.
    auto_gate_active = getattr(args, 'use_pathology_recipe', False) and args.embeddingdim >= 1280

    if getattr(args, 'use_pathology_recipe', False):
        # patch_size 14 is the community standard for pathology FMs
        if args.patch_size != 14:
            print(f"[pathology recipe] Overriding patch_size {args.patch_size} -> 14 "
                  f"(community standard for pathology FMs)")
            args.patch_size = 14
        # Nudge koleo_loss_weight to 0.05 only if user left the default 0.1
        if abs(args.koleo_loss_weight - 0.1) < 1e-6:
            args.koleo_loss_weight = 0.05
            print(f"[pathology recipe] Set koleo_loss_weight=0.05 (Virchow2 KDE default lambda)")

    if auto_gate_active:
        # qk_norm: auto-enable if not explicitly set
        if args.qk_norm is None:
            args.qk_norm = True
        # register tokens: monotonic max with 8
        if args.num_register_tokens < 8:
            args.num_register_tokens = 8
            print(f"[pathology recipe auto-gate] Bumped num_register_tokens to 8 "
                  f"(Virchow2G scaling package)")
        # out_dim 131072 was Virchow v1's choice at ViT-H scale, carried by
        # Virchow2 / Virchow2G at ViT-H/G. It is a scaled-regime change, not
        # a universal pathology-recipe component, so it lives in the auto-gate.
        if args.out_dim != 131072:
            print(f"[pathology recipe auto-gate] Overriding out_dim {args.out_dim} -> 131072 "
                  f"(Virchow v1 Methods, Paige standard)")
            args.out_dim = 131072
        print(f"[pathology recipe auto-gate] qk_norm={args.qk_norm}, "
              f"register_tokens={args.num_register_tokens}, "
              f"out_dim={args.out_dim}, StableAdamW active")
    elif getattr(args, 'qk_norm', None) is None:
        # Corrected DINOv2 default: qk_norm on at every model size when the
        # user did not set it explicitly. The auto-gate above still wins
        # (it already sets True for ViT-H/G + pathology recipe). Explicit
        # True/False from the CLI / launch script always wins because both
        # values fail the `is None` check on the auto-gate's gating line.
        # Pass --qk_norm False (or args.qk_norm = False in the launcher) to
        # reproduce the older no-QK-norm runs.
        args.qk_norm = True

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Stream-rebalancing mode (section 3.9). Thinned mode reuses the whole typicality stack
    # (bank, representative prototypes, R optimizer, checkpointing); it only relocates the bank
    # update to a scout pass and drops the loss weighting. So it requires the stack to be built.
    balance_mode = getattr(args, 'balance_mode', 'weighted')
    if balance_mode == 'thinned':
        assert args.use_typicality_dampening, (
            "--balance_mode thinned requires --use_typicality_dampening True "
            "(thinned mode reuses the counted-coverage bank for the scout density)")
    print(f"Balance mode: {balance_mode}")
    # --use_gpu_augmentation: a weighted/off baseline reuses the thinned raw-tile loader + GPU
    # augmentation path (no scout/bank/thinning). No-op under thinned (which already does this).
    use_gpu_aug = bool(getattr(args, 'use_gpu_augmentation', False)) and balance_mode != 'thinned'
    if use_gpu_aug:
        print("GPU augmentation (baseline): raw-tile loader + kornia GPU crops, no typicality machinery")

    # Augmentation configuration
    augmentation_free_mode = (args.global_views == 0)

    print("\n========== Augmentation Configuration ==========")
    print(f"Global views (teacher/student): {args.global_views}")
    print(f"Standard local crops: {args.n_standard_local_crops}")
    print(f"Local crop size: {args.local_crop_size}x{args.local_crop_size}")

    if args.use_semantic_ibot:
        print("Semantic iBOT: ENABLED")
        print(f"  Mask model: {args.mask_model_arch}")
        print(f"  Semantic channels: {args.num_masks}")
        print(f"  Channels per iteration: {args.semantic_masks_per_iteration}")
        print(f"  Semantic iBOT weight: {args.semantic_ibot_weight}")
        if args.use_semantic_prototypes:
            print(f"  Semantic prototype loss: ENABLED (weight={args.semantic_clustering_weight})")
    else:
        print("Semantic iBOT: DISABLED")

    # Adversarial mask-as-student-view augmentation
    if args.use_adversarial_mask_augmentation:
        print(f"\nAdversarial Mask Augmentation: ENABLED")
        print(f"  Number of masks: {args.num_masks}")
        print(f"  Crops per mask: {args.crops_per_mask}")
    else:
        print(f"\nAdversarial Mask Augmentation: DISABLED")

    # CellViT augmentation
    if args.use_cellvit_augmentation:
        print(f"\nCellViT Augmentation: ENABLED")
        print(f"  Crops per channel: {args.cellvit_crops_per_channel}")
    else:
        print(f"\nCellViT Augmentation: DISABLED")

    # Random rectangular mask augmentation
    if args.use_random_mask_augmentation:
        print(f"\nRandom Mask Augmentation: ENABLED")
        print(f"  Number of masks: {args.random_num_masks}")
        print(f"  Crops per mask: {args.random_crops_per_mask}")
    else:
        print(f"\nRandom Mask Augmentation: DISABLED")

    total_student_views = calculate_total_student_views(args)
    print(f"\nTotal student views: {total_student_views}")
    print("================================================\n")

    # ============ Load pre-trained mask model (shared by semantic iBOT and adversarial mask aug) ============
    # Either feature enables this single frozen mask model. Its soft masks are consumed
    # differently by each: semantic iBOT uses them as token masks inside the iBOT loss,
    # while adversarial mask augmentation uses them as image-level masks to create
    # additional student views.
    mask_model_frozen = None
    needs_mask_model = args.use_semantic_ibot or args.use_adversarial_mask_augmentation
    if needs_mask_model:
        if args.mask_checkpoint is None:
            raise ValueError("--mask_checkpoint is required when --use_semantic_ibot or "
                             "--use_adversarial_mask_augmentation is True")

        if args.num_masks <= 0:
            raise ValueError("--num_masks must be > 0 when the mask model is enabled")

        mask_model_frozen = load_pretrained_mask_model(
                                                        args.mask_checkpoint,
                                                        args.num_masks,
                                                        mask_model_arch=args.mask_model_arch,
                                                        mask_encoder_dim=args.mask_encoder_dim
                                                    )
        mask_model_frozen = mask_model_frozen.cuda()
        mask_model_frozen.eval()

        for param in mask_model_frozen.parameters():
            param.requires_grad = False

        consumers = []
        if args.use_semantic_ibot:
            consumers.append("semantic iBOT")
        if args.use_adversarial_mask_augmentation:
            consumers.append("adversarial mask augmentation")
        print(f"Loaded and froze mask model with {args.num_masks} channels "
              f"(consumers: {', '.join(consumers)})")
    else:
        print("Mask model not loaded (semantic iBOT and adversarial mask augmentation both disabled)")

    # ============ Load pre-trained CellViT model (if enabled) ============
    cellvit_model_frozen = None
    if args.use_cellvit_augmentation:
        if args.cellvit_checkpoint is None:
            raise ValueError("--use_cellvit_augmentation is True but --cellvit_checkpoint not provided")

        cellvit_model_frozen = load_pretrained_cellvit_model(args.cellvit_checkpoint, device='cuda')
        cellvit_model_frozen.eval()

        for param in cellvit_model_frozen.parameters():
            param.requires_grad = False

        print(f"Loaded and froze CellViT model for nuclei/background segmentation")
        print(f"  CellViT crops per channel: {args.cellvit_crops_per_channel}")
    else:
        print("CellViT augmentation disabled (--use_cellvit_augmentation=False)")

    # ============ Create dataset ============
    dataset_configs = []
    for source in args.dataset_sources:
        parts = source.split(':')
        name, base_dir, index_file = parts
        index_path = os.path.join(base_dir, index_file)
        metadata_path = index_path.replace('.pkl', '_metadata.pkl')
        dataset_configs.append({
            'name': name,
            'base_dir': base_dir,
            'index_file': index_file
        })

    trainset = ProportionalMultiDatasetWrapper(
        dataset_configs=dataset_configs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        n_standard_local_crops=args.n_standard_local_crops,
        global_views=args.global_views,
        local_crop_size=args.local_crop_size,
        worker_id=0,  # vestigial: __iter__ reads worker info from get_worker_info()
        num_workers=args.num_workers,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        seed=args.seed,
        global_size=224,
        use_pathology_recipe=getattr(args, 'use_pathology_recipe', False),
        ect_probability=getattr(args, 'ect_probability', 0.4),
        # Thinned mode (section 3.9): the loader emits ONLY the raw Resize(224) uint8 tile per tile
        # (scout_pool_mode) -- no CPU augmentation. The trainer normalizes it for the density bank and
        # GPU-augments (kornia) ONLY the thinned survivors, so the 5/6 of the over-drawn pool that gets
        # discarded costs no augmentation. weighted/off keep the full CPU crop pipeline -> byte-identical.
        emit_scout=False,
        scout_pool_mode=(getattr(args, 'balance_mode', 'weighted') == 'thinned'
                         or getattr(args, 'use_gpu_augmentation', False)),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    # ============ Initialize models ============
    # Layerscale schedule resolution. Defaults to 'uniform' with
    # --layerscale_init=1e-5 (corrected DINOv2 behavior). 'cait' translates
    # to layerscale_init=None at the constructor level, which the
    # VisionTransformer already interprets as "use the depth-based CaiT
    # schedule" (0.1 / 1e-5 / 1e-6 by depth). Keeping the schedule flag
    # trainer-side leaves the constructor signature unchanged, so the
    # dashboard/PCA checkpoint loaders (which don't pass this kwarg) keep
    # their existing CaiT-schedule behavior automatically.
    if getattr(args, 'layerscale_schedule', 'uniform') == 'cait':
        effective_layerscale_init = None
        if args.layerscale_init is not None:
            print(f"[layerscale] schedule='cait' → using depth-based CaiT init; "
                  f"--layerscale_init={args.layerscale_init} is ignored.")
    else:
        effective_layerscale_init = args.layerscale_init

    # FFN type: DINOv2 ssl_default uses a standard MLP+GELU; SwiGLU is the fork default.
    mlp_layer_cls = Mlp if getattr(args, 'ffn_type', 'swiglu') == 'mlp' else SwiGLUFFNFused

    student_encoder = ModernViT(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embeddingdim,
        depth=args.vitdepth,
        num_heads=args.vitheads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=bool(args.qk_norm),
        dual_norm=False,
        drop_path_rate=args.drop_path_rate,
        drop_path_uniform=args.drop_path_uniform,
        pre_norm=False,
        num_register_tokens=args.num_register_tokens,
        layerscale_init=effective_layerscale_init,
        mlp_layer=mlp_layer_cls,
    )

    teacher_encoder = deepcopy(student_encoder)

    student_classhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    teacher_classhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
    )

    student_patchhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    teacher_patchhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
    )

    student = CombinedModelDINO(
        backbone=student_encoder,
        classhead=student_classhead,
        patchhead=student_patchhead,
        num_masks=args.num_masks,
        patch_size=args.patch_size,
    )

    teacher = CombinedModelDINO(
        backbone=teacher_encoder,
        classhead=teacher_classhead,
        patchhead=teacher_patchhead,
        num_masks=args.num_masks,
        patch_size=args.patch_size,
    )

    # ============ Create Prototype Bank (Optional) ============
    prototype_bank = None
    if args.use_prototype_clustering:
        prototype_bank = LinearPrototypeBank(
            num_prototypes=args.num_prototypes,
            embed_dim=args.embeddingdim,
            bias=True
        )
        prototype_bank = prototype_bank.cuda()

        print(f"Created LinearPrototypeBank with {args.num_prototypes} soft prototypes")
    else:
        print("Prototype clustering disabled (--use_prototype_clustering=False)")

    # ============ Create Typicality Dampening (Optional) ============
    repr_protos = None
    typicality_bank = None
    richardson_bank = None
    if args.use_typicality_dampening:
        repr_protos = RepresentativePrototypes(
            K_prime=args.typicality_K_prime,
            bottleneck_dim=256,  # DINOHead bottleneck_dim
        )
        repr_protos = repr_protos.cuda()

        # Thinned mode pays the bank on the FULL over-drawn pool every step, so it uses the
        # vectorized subclass. weighted/off keep the original class untouched (byte-identical arms).
        _BankCls = (FastCountedCoverageBank
                    if getattr(args, 'balance_mode', 'weighted') == 'thinned'
                    else CountedCoverageBank)
        typicality_bank = _BankCls(
            M=args.typicality_bank_size,
            K_prime=args.typicality_K_prime,
            pool_j=args.typicality_pool_j,
            halflife_steps=args.typicality_halflife_steps,
            reserve_residency=args.typicality_reserve_residency,
            reserve_size=args.typicality_reserve_size,
            s_buffer_size=args.typicality_s_buffer_size,
            s_sweep_interval=args.typicality_s_sweep_interval,
            s_grid_points=args.typicality_s_grid_points,
            grid_span=tuple(args.typicality_s_grid_span),
            s_ema_alpha=args.typicality_s_ema_alpha,
            s_min_buffer=args.typicality_s_min_buffer,
            s_headroom=args.typicality_s_headroom,
            graduation_hits=args.typicality_graduation_hits,
            pool_selftune=args.typicality_pool_selftune,
            pool_rse_target=args.typicality_pool_rse_target,
            pool_max=args.typicality_pool_max,
            pool_ema=args.typicality_pool_ema,
            radius_mult=args.typicality_radius_mult,
        ).cuda()

        print(f"Created Typicality Dampening (counted-coverage bank):")
        print(f"  K' = {args.typicality_K_prime}, Bank M = {args.typicality_bank_size}")
        print(f"  Modulation: {args.typicality_modulation}")
        print(f"  Warmup: {args.typicality_warmup_iters} iterations")

        # Richardson two-scale bias probe (section 3.9), thinned mode only. A second, COARSE
        # covering bank at ~2s spacing. M' = M * 2^-D (D~9.3) ~ 13 is below the class's working
        # regime (reserve/sweep/variogram assume M in the thousands), so floor M' to 64; the coarse
        # bank self-tunes its own (larger) s and yields p_hat_2s. beta_hat = log p_2s - log p_s is a
        # CRUDE, measure-only bias indicator (not calibrated). Built with the same knobs but M=M'.
        if balance_mode == 'thinned':
            M_coarse = max(64, int(args.typicality_bank_size / (2.0 ** 9.3)))
            richardson_bank = _BankCls(
                M=M_coarse,
                K_prime=args.typicality_K_prime,
                pool_j=args.typicality_pool_j,
                halflife_steps=args.typicality_halflife_steps,
                reserve_residency=args.typicality_reserve_residency,
                reserve_size=args.typicality_reserve_size,
                s_buffer_size=args.typicality_s_buffer_size,
                s_sweep_interval=args.typicality_s_sweep_interval,
                s_grid_points=args.typicality_s_grid_points,
                grid_span=tuple(args.typicality_s_grid_span),
                s_ema_alpha=args.typicality_s_ema_alpha,
                s_min_buffer=args.typicality_s_min_buffer,
                s_headroom=args.typicality_s_headroom,
                graduation_hits=args.typicality_graduation_hits,
                pool_selftune=args.typicality_pool_selftune,
                pool_rse_target=args.typicality_pool_rse_target,
                pool_max=args.typicality_pool_max,
                pool_ema=args.typicality_pool_ema,
                radius_mult=args.typicality_radius_mult,
            ).cuda()
            print(f"  Richardson coarse bank: M' = {M_coarse} (crude two-scale bias probe, measure-only)")
    else:
        print("Typicality dampening disabled (--use_typicality_dampening=False)")

    student = student.cuda()
    teacher = teacher.cuda()

    # Per-block torch.compile. Compiling backbone-as-a-whole trips a dynamo guard on the module-global
    # attn_bias_cache in models/vision_transformer/modern_vit.py ("Duplicate tensors found"); compiling
    # the blocks leaves the sequence-packing eager and sidesteps it entirely.
    if getattr(args, 'compile_blocks', False):
        for _blk in student.backbone.blocks:
            _blk.forward = torch.compile(_blk.forward, dynamic=False)
        for _blk in teacher.backbone.blocks:
            _blk.forward = torch.compile(_blk.forward, dynamic=False)
        print("[speed] torch.compile applied per TransformerBlock (student+teacher)")

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    from training.hand_rolled_dp import HandRolledDP
    # Hand-rolled data parallel (see hand_rolled_dp.py): replaces DDP so torch.compile
    # + gradient checkpointing works. student grads are synced via student.sync_grads()
    # after backward(); teacher carries no grad (EMA), prototype_bank has its own step.
    student = setup_ddp_model(student, args, find_unused=True)   # -> HandRolledDP
    teacher = HandRolledDP(teacher, broadcast=False)             # loaded from student below

    if args.use_prototype_clustering:
        prototype_bank = HandRolledDP(prototype_bank)

    teacher_without_ddp = teacher.module

    print("Hand-rolled data parallel active (student sync_grads after backward)")

    teacher_without_ddp.backbone.load_state_dict(student.module.backbone.state_dict())
    teacher_without_ddp.classhead.load_state_dict(student.module.classhead.state_dict())
    teacher_without_ddp.patchhead.load_state_dict(student.module.patchhead.state_dict())
    teacher.requires_grad_(False)

    # ============ Initialize losses ============
    # teacher_temp=0.04 fixed is a Virchow2G ViT-G scaling-package choice that
    # pairs with out_dim=131k. Both live in the auto-gate; do not apply at
    # ViT-B/L where the standard warmup schedule converges.
    if auto_gate_active:
        warmup_teacher_temp_effective = 0.04
        teacher_temp_effective = 0.04
        print(f"[pathology recipe auto-gate] Teacher temperature fixed at 0.04. "
              f"Source: Virchow2G Section 5.1.")
    else:
        warmup_teacher_temp_effective = args.warmup_teacher_temp
        teacher_temp_effective = args.teacher_temp

    dino_class_loss = DINOLoss(
        ncrops=total_student_views,
        warmup_teacher_temp=warmup_teacher_temp_effective,
        teacher_temp=teacher_temp_effective,
        warmup_teacher_temp_iters=args.teacher_temp_warmup_iters,
        n_iterations=5,
        student_temp=0.1,
    ).cuda()

    ibot_patch_loss = iBOTPatchLoss(
        student_temp=0.1,
        n_iterations=3,
    ).cuda()

    if getattr(args, 'use_pathology_recipe', False):
        dino_koleo_loss = KDELoss(kappa=args.kde_kappa).cuda()
        print(f"Using KDE regularizer (kappa={args.kde_kappa}, all-gather across "
              f"{dist.get_world_size()} GPUs). Source: Virchow2 Section 5.2.")
    else:
        dino_koleo_loss = KoLeoLoss().cuda()
        print("Using KoLeo regularizer (DINOv2 default)")

    patch_prototype_loss = None
    if args.use_prototype_clustering:
        patch_prototype_loss = PatchPrototypeLoss(
            num_prototypes=args.num_prototypes,
            embed_dim=args.embeddingdim,
            teacher_temp=args.clustering_teacher_temp,
            student_temp=args.clustering_student_temp,
        ).cuda()

        print(f"Initialized PatchPrototypeLoss with {args.num_prototypes} prototypes")
    else:
        print("Patch prototype clustering disabled")

    # ============ Create fp16_scaler ============
    if getattr(args, 'use_pathology_recipe', False):
        fp16_scaler = None
        bf16_mode = True
        print("Using bf16 end-to-end (no GradScaler). H100-appropriate; "
              "avoids fp16 NaN issues flagged by Virchow2G.")
    else:
        # bf16 autocast needs no loss scaling; a default GradScaler doubles its scale
        # every 2000 clean steps unbounded under bf16 and overflows fp32 near it~190k
        # (observed scale 2^111 at 190k iters, then NaN collapse). Disabled.
        fp16_scaler = None
        bf16_mode = False

    # ============ Create optimizers ============
    backbone_params = utils.get_params_groups_with_layer_decay(
        student.module.backbone,
        lr_decay_rate=args.lr_decay_rate,
        num_layers=args.vitdepth,
        patch_embed_lr_mult=args.patch_embed_lr_mult,
    )

    classhead_params = utils.get_params_groups_with_decay_for_heads(student.module.classhead)
    patchhead_params = utils.get_params_groups_with_decay_for_heads(student.module.patchhead)

    all_param_groups = backbone_params + classhead_params + patchhead_params

    if auto_gate_active:
        optimizer_student = utils.StableAdamW(all_param_groups, betas=(0.9, 0.95))
        print(f"Using StableAdamW (betas=(0.9, 0.95)). Source: Virchow2G Section 6.")
    else:
        optimizer_student = torch.optim.AdamW(all_param_groups)

    if utils.is_main_process():
        print(f"\n=== Layer-wise LR Decay (rate={args.lr_decay_rate}) ===")
        for i, pg in enumerate(backbone_params):
            print(f"  Group {i}: lr_mult={pg['lr_multiplier']:.4f}, wd_mult={pg['wd_multiplier']}, params={len(pg['params'])}")
        print(f"  Head groups: {len(classhead_params) + len(patchhead_params)} groups with lr_mult=1.0")
        print(f"  Total param groups: {len(all_param_groups)}")
        print("=" * 50 + "\n")

    optimizer_prototypes = None
    if args.use_prototype_clustering:
        optimizer_prototypes = torch.optim.AdamW(
            prototype_bank.module.parameters(),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        print(f"Created optimizers (including prototype optimizer)")
    else:
        print(f"Created optimizer (student only)")

    # R optimizer for representative prototypes (typicality)
    R_optimizer = None
    if args.use_typicality_dampening and repr_protos is not None:
        R_optimizer = torch.optim.AdamW(
            repr_protos.parameters(),
            lr=args.typicality_repr_lr,
            weight_decay=0.0,
        )
        print(f"Created R optimizer (lr={args.typicality_repr_lr}, no weight decay)")

    # ============ Create schedulers ============
    student_lr_schedule = utils.cosine_scheduler(
        base_value=args.lr * math.sqrt(args.batch_size_per_gpu * utils.get_world_size() / 1024.0),
        final_value=args.min_lr,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
        start_warmup_value=0
    )

    proto_lr_schedule = None
    if args.use_prototype_clustering:
        proto_lr_schedule = utils.cosine_scheduler(
            base_value=args.lr * 0.5,
            final_value=0,
            total_iters=args.total_iterations,
            warmup_iters=args.warmup_iterations,
            start_warmup_value=0
        )

    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
        start_warmup_value=args.weight_decay
    )

    momentum_schedule = utils.cosine_scheduler(
        base_value=args.momentum_teacher,
        final_value=1.0,
        total_iters=args.total_iterations,
        warmup_iters=0,
        start_warmup_value=args.momentum_teacher
    )

    # ============ Load checkpoint ============
    to_restore = {"iteration": 0, "dataset_position": 0}

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    loaded_checkpoint = None
    if os.path.exists(checkpoint_path):
        try:
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Pre-loaded checkpoint from iteration {loaded_checkpoint.get('iteration', 'N/A')}.")
        except Exception as e:
            print(f"Could not pre-load checkpoint. Starting fresh. Error: {e}")
            loaded_checkpoint = None

    checkpoint_kwargs = {
        'student': student,
        'teacher': teacher,
        'optimizer_student': optimizer_student,
        'fp16_scaler': fp16_scaler,
        'dino_class_loss': dino_class_loss,
    }

    if args.use_prototype_clustering:
        checkpoint_kwargs['prototype_bank'] = prototype_bank
        checkpoint_kwargs['optimizer_prototypes'] = optimizer_prototypes
        checkpoint_kwargs['patch_prototype_loss'] = patch_prototype_loss

    # Add typicality-related modules only if enabled
    if args.use_typicality_dampening:
        checkpoint_kwargs['repr_protos'] = repr_protos
        checkpoint_kwargs['R_optimizer'] = R_optimizer
        checkpoint_kwargs['typicality_bank'] = typicality_bank
        if richardson_bank is not None:
            checkpoint_kwargs['richardson_bank'] = richardson_bank

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        **checkpoint_kwargs
    )

    current_iteration = to_restore["iteration"]
    dataset_position = to_restore.get("dataset_position", 0)

    # ============ Set resume position in dataset ============
    if current_iteration > 0:
        global_samples_processed = current_iteration * args.batch_size_per_gpu * dist.get_world_size()
        trainset.set_resume_position(global_samples_processed)
        print(f"Resuming from iteration {current_iteration}")

    # ============ Restore RNGs ============
    if loaded_checkpoint and 'torch_rng_state' in loaded_checkpoint:
        try:
            print("Restoring RNG states from checkpoint...")
            torch.set_rng_state(loaded_checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state_all(loaded_checkpoint['cuda_rng_state'])
            np.random.set_state(loaded_checkpoint['numpy_rng_state'])
            random.setstate(loaded_checkpoint['random_rng_state'])
            print(f"Successfully restored all RNG states to iteration {current_iteration}.")
        except Exception as e:
            print(f"WARNING: Failed to restore RNG states. Re-seeding. Error: {e}")
            utils.fix_random_seeds(args.seed + utils.get_rank())
    else:
        if current_iteration == 0:
            print("Starting from scratch. Fixing random seeds.")
        else:
            print(f"WARNING: Checkpoint found but no RNG state. Re-seeding.")
        utils.fix_random_seeds(args.seed + utils.get_rank())

    # ============ Verify checkpoint ============
    if utils.is_main_process() and current_iteration > 0:
        print(f"\n=== Checkpoint Loaded at Iteration {current_iteration} ===")
        if args.use_prototype_clustering:
            proto_stats = prototype_bank.module.get_stats()
            print(f"Prototype Bank Statistics:")
            print(f"  Weight norm mean: {proto_stats['weight_norm_mean']:.6f}")
            print(f"  Weight norm std: {proto_stats['weight_norm_std']:.6f}")
        if args.use_typicality_dampening and typicality_bank is not None:
            _st = typicality_bank.stats()
            print(f"Typicality Bank (counted): n_est={_st['n_est']}/{typicality_bank.M} "
                  f"reserve={_st['n_reserve']} graduations={_st['graduations']} "
                  f"s={_st['s']:.4f} j={_st['j']} p_ref={_st['p_ref']:.4g}")
        print("="*50 + "\n")

    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    metric_logger.start_time = time.time()
    # ETA state: seconds/iter over a RECENT window (EMA), robust to (a) resume -- eta_ref_iter is the
    # RESUMED iteration, not 0, so the estimate is not diluted by pre-restart progress -- and (b) the
    # fast->slow step-time change when thinning's over-draw + scout engage. The previous cumulative
    # elapsed/total-progress estimate was wrong on both counts.
    eta_sec_per_it, eta_ref_time, eta_ref_iter = None, metric_logger.start_time, current_iteration
    typ_last_grad = 0                        # for the per-interval graduations/step diff (compact log)
    typ_last_grad_iter = int(getattr(args, 'typicality_warmup_iters', 0))

    data_iterator = iter(train_loader)

    loader_len = len(train_loader) if len(train_loader) > 0 else 1
    dataset_passes = dataset_position // loader_len
    max_passes = 5

    # ---- stream-thinning gate state (section 3.9; thinned mode only) ----
    # w_max is now the ANALYTIC bound (c_frac*p_ref)^(-a) computed each step (see the acceptance
    # block below), so there is no measurement window / freeze: acceptance is live the moment the
    # gate opens. The old frozen running-max went stale as p_ref grew and starved acceptance.
    thin_gate_open = False        # latches True once the bank is FILLED for K consecutive steps
    thin_ready_streak = 0         # consecutive steps with bank_ready true (resets on any miss)
    thin_active = 0               # 0 = gate closed; 1 = acceptance live (immediate once gate opens)
    THIN_GATE_K = 5              # consecutive bank_ready steps required to open the gate

    # ---- thinned GPU augmentation (kornia): the loader emits raw uint8 tiles (scout_pool_mode); the
    # scout is the NORMALIZED raw (fed to the bank), and ONLY the committed survivors are augmented,
    # on-GPU -- so the 5/6 of the over-drawn pool discarded by thinning is never augmented. weighted/off
    # never construct this (they keep the exact CPU crop pipeline) unless --use_gpu_augmentation. ----
    gpu_augment = None
    _raw_mean = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1)  # PATHOLOGY norm (NOT ImageNet);
    _raw_std = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1)   #   matches the dataloader default.
    if balance_mode == 'thinned' or use_gpu_aug:
        # grid_sample augmenter: ~-386 ms/step vs the kornia path, distribution-matched with exact
        # torchvision HSV hue. Different RNG stream than kornia (the thinned arm was never pixel-
        # comparable to weighted anyway); adopted deliberately for the speed.
        from data.gpu_augment_fast import FastCropAugment as GPUCropAugment
        gpu_augment = GPUCropAugment(global_size=224, local_size=args.local_crop_size,
                                     n_local_crops=args.n_standard_local_crops).cuda()
        _raw_mean = _raw_mean.cuda(); _raw_std = _raw_std.cuda()

    def _normalize_raw(raw_u8):
        """Raw uint8 [B,3,224,224] -> normalized float on GPU = the scout view fed to the bank. Same
        pathology mean/std as the retired self.scout (Resize+ToTensor+Normalize)."""
        x = raw_u8.to('cuda', non_blocking=True).float().div_(255.0)
        return (x - _raw_mean) / _raw_std

    def _draw_one():
        """Draw one loader batch, handling StopIteration -> pass increment (same bookkeeping as
        the weighted/off path). Returns (batch, stop). Only used by the thinned over-draw."""
        nonlocal data_iterator, dataset_position, dataset_passes
        try:
            b = next(data_iterator); dataset_position += 1
        except StopIteration:
            dataset_passes += 1
            if dataset_passes >= max_passes:
                return None, True
            data_iterator = iter(train_loader)
            b = next(data_iterator)
            dataset_position = dataset_passes * loader_len
            print(f"Starting pass {dataset_passes + 1} at iteration {current_iteration}")
        return b, False

    def _scout_and_bank(scout_crop, cur_iter):
        """Thinned mode: forward the unaugmented scout crop no-grad, update the PRIMARY and the
        coarse Richardson banks on the all-gathered scout signatures (the only bank updates), and
        return (p_hat_local, p_hat_gathered, mean|beta_hat|). p_hat is debiased to u*=2u_s-u_2s
        only when --thin_richardson_correct (default off)."""
        with torch.no_grad():
            # bottleneck_only: the scout consumes so['bottleneck'] and nothing else, so skip the
            # out_dim prototype layer over the whole over-drawn pool. Bottleneck is bit-identical.
            # (Measured: this saves ~0 ms -- the skipped GEMM is tiny -- but it avoids materializing a
            # [pool, out_dim] fp32 activation, ~400 MB at pool=1536/out_dim=65536.)
            _sc = scout_crop.cuda(non_blocking=True)
            if getattr(args, 'scout_amp_bf16', False):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    so = student([_sc], token_masks=[None], mode='dino', bottleneck_only=True)
            else:
                so = student([_sc], token_masks=[None], mode='dino', bottleneck_only=True)
            sg = _all_gather_signatures(
                repr_protos.compute_signatures(so['bottleneck'].detach()).detach())
            bo = typicality_bank.score_and_update(sg, cur_iter)
            if bo['ready'] and cur_iter % 2000 == 0:
                _assert_bank_synced(typicality_bank.sync_fingerprint(), tag=f"@it{cur_iter}(scout)")
            co = richardson_bank.score_and_update(sg, cur_iter) if richardson_bank is not None else {}
            pg = bo.get('p_hat')
            bias = None
            if pg is not None and co.get('p_hat') is not None:
                us = torch.log(pg + 1e-8); u2 = torch.log(co['p_hat'] + 1e-8)
                bias = float((u2 - us).abs().mean().item())
                if args.thin_richardson_correct:
                    pg = torch.exp(2.0 * us - u2)
            nrows = scout_crop.shape[0]
            return (_local_rows(pg, nrows) if pg is not None else None), pg, bias

    if utils.is_main_process():
        print(f"Starting training at iteration {current_iteration}")

    # ============ Training loop ============
    print("Starting training!")

    while current_iteration < args.total_iterations:
        # ========== Get batch (weighted/off: one batch; thinned: over-drawn pool -> N) ==========
        thin_seen_val = None; thin_bias_val = None; thin_accept_val = None
        thin_committed_phat = None; thin_pool_phat = None
        if balance_mode != 'thinned':
            # ===== weighted / off: draw exactly one full-crop batch (UNCHANGED, byte-identical) =====
            try:
                batch_data = next(data_iterator)
                dataset_position += 1
            except StopIteration:
                dataset_passes += 1
                if dataset_passes >= max_passes:
                    print(f"Reached maximum passes ({max_passes}). Stopping.")
                    break

                data_iterator = iter(train_loader)
                batch_data = next(data_iterator)
                dataset_position = dataset_passes * loader_len
                print(f"Starting pass {dataset_passes + 1} at iteration {current_iteration}")
            if use_gpu_aug:
                # --use_gpu_augmentation: the loader emitted ONE raw Resize(224) uint8 tile per sample
                # (scout_pool_mode). Commit all N and GPU-augment them -> [g1, g2, l1..lL] normalized
                # on GPU -- the same transform sequence as the thinned pre-warmup path below (raw
                # uint8 -> gpu_augment, which normalizes internally). No over-draw, no scout, no bank,
                # no thin_* telemetry (thin_active stays at its baseline 0).
                batch_data = gpu_augment(batch_data[0].to('cuda', non_blocking=True))
        else:
            # ===== THINNED (raw-carry + GPU augmentation, section 3.9) =====
            # The loader yields ONE raw Resize(224) uint8 tile per sample (scout_pool_mode). We
            # normalize it for the density scout/bank, admit survivors by density, and GPU-augment
            # (kornia) ONLY the committed N -- the 5/6 of the over-drawn pool discarded by thinning is
            # never augmented (that was the ~6x CPU-augmentation cost). `survivor_raw` is the uint8
            # tile set that gets augmented; batch_data = [g1, g2, l1..lL] normalized on GPU, the exact
            # format the crop extraction below already consumes.
            N = args.batch_size_per_gpu
            if not thin_gate_open:
                # pre-warmup OR gate-closed post-warmup: PLAIN BASELINE -- one raw batch, commit all N
                # (no over-draw, no acceptance). The scout STILL fills the bank once past warmup; the
                # gate opens after bank_ready (p_ref>0 AND n_est==M AND reserve full) holds K steps.
                b, _stop = _draw_one()
                if _stop:
                    print(f"Reached maximum passes ({max_passes}). Stopping.")
                    break
                pool_raw = b[0]                                   # [N,3,224,224] uint8 (CPU)
                if current_iteration >= args.typicality_warmup_iters:
                    _pl, _pg, bias = _scout_and_bank(_normalize_raw(pool_raw), current_iteration)
                    thin_bias_val = bias
                    thin_seen_val = N                             # no over-draw; committed = drawn = N
                    thin_accept_val = 1.0                         # plain baseline: everything committed
                    _pref = float(typicality_bank.p_ref.item())
                    bank_ready = (_pref > 0.0
                                  and typicality_bank._ne() == typicality_bank.M
                                  and typicality_bank._nr() == typicality_bank.reserve_cap)
                    thin_ready_streak = thin_ready_streak + 1 if bank_ready else 0
                    if thin_ready_streak >= THIN_GATE_K:
                        thin_gate_open = True
                        if utils.is_main_process():
                            print(f"[thinned] gate OPEN at iter {current_iteration}: bank filled "
                                  f"(p_ref={_pref:.4g}, n_est={typicality_bank._ne()}/{typicality_bank.M}, "
                                  f"reserve={typicality_bank._nr()}/{typicality_bank.reserve_cap}); "
                                  f"acceptance LIVE (analytic w_max = (c_frac*p_ref)^-a)")
                survivor_raw = pool_raw                           # commit all N (no thinning)
            else:
                # ===== GATE OPEN: ONE over-drawn raw pool -> scout (single all_gather) -> admit by
                # density a(p_hat)=w/w_max -> fixed to EXACTLY N. No "pull more" loop (that desynced
                # NCCL); a rare under-fill is topped up from the SAME pool by highest-a rejects. w_max
                # is the ANALYTIC bound (c_frac*p_ref)^(-a), recomputed each step (see below). =====
                n_draws = max(1, math.ceil(args.thin_oversample_factor))
                pool_batches = []; _stop = False
                for _ in range(n_draws):
                    b, _stop = _draw_one()
                    if _stop:
                        break
                    pool_batches.append(b)
                if _stop:
                    print(f"Reached maximum passes ({max_passes}). Stopping.")
                    break
                pool_raw = torch.cat([pb[0] for pb in pool_batches], dim=0)   # [pool_n,3,224,224] uint8
                p_local, p_gathered, bias = _scout_and_bank(_normalize_raw(pool_raw), current_iteration)
                thin_bias_val = bias
                pool_n = pool_raw.shape[0]
                thin_seen_val = pool_n
                if p_local is None:
                    # defensive: the gate requires a matured bank, so this should not happen post-gate.
                    keep_idx = torch.arange(min(N, pool_n))
                    thin_accept_val = 1.0
                else:
                    thin_pool_phat = p_local
                    # Acceptance tilt reads the SAME knobs as the weighted path
                    # (args.typicality_a / args.typicality_c_frac), so the hi thinned arm now
                    # admits at its configured a=1.0. Previously a_tilt was hardcoded to 0.5,
                    # which made hi accept like lo and silently cancelled the thinned tilt sweep.
                    # a_tilt drives BOTH w_local and the analytic w_max below, so the two stay
                    # consistent by construction (change one -> both move).
                    a_tilt, c_frac_tilt = args.typicality_a, args.typicality_c_frac
                    w_local = TypicalityScorer.absolute_weights(
                        p_local, typicality_bank.p_ref, a_tilt, c_frac_tilt)
                    # Analytic w_max = sup_{p>=0} w = (c_frac*p_ref)^(-a): the exact upper bound of w,
                    # attained as p_hat->0 (absolute_weights docstring: "1/c^a, no clipping required").
                    # Recomputed EACH STEP so it tracks p_ref. Provably >= max(w) => a=w/w_max never
                    # clamps => the admitted distribution is w_max-INVARIANT (w_max cancels in the
                    # top-N selection); only the acceptance RATE changes. This replaces the frozen
                    # 200-step running max, which stayed pinned to the gate-open p_ref scale while
                    # p_ref grew ~10x, inflating the normalizer ~3.3x and starving accept to ~0.15
                    # (85% under-fill + top-up, which breaks the Bernoulli matched-mass guarantee).
                    # Because w depends only on p_hat/p_ref, accept is scale-invariant (~0.49) at every
                    # p_ref, so there is no early-or-late under-fill.
                    _pref_now = typicality_bank.p_ref
                    _pref_now = float(_pref_now.item()) if torch.is_tensor(_pref_now) else float(_pref_now)
                    wmax = (c_frac_tilt * max(_pref_now, 1e-8)) ** (-a_tilt)
                    a = (w_local / max(wmax, 1e-8)).clamp(0.0, 1.0)
                    accepted = torch.nonzero(torch.rand_like(a) < a, as_tuple=False).flatten()
                    thin_accept_val = accepted.numel() / max(1, pool_n)   # realized acceptance ~ 1/chi
                    if accepted.numel() >= N:
                        keep_idx = accepted[:N]
                    else:
                        # rare under-fill: top up to N with the highest-a rejects (deterministic, no
                        # extra draw/collective). Persistent -> raise --thin_oversample_factor.
                        rej = torch.ones(pool_n, dtype=torch.bool, device=a.device)
                        rej[accepted] = False
                        rej_idx = torch.nonzero(rej, as_tuple=False).flatten()
                        order = torch.argsort(a[rej_idx], descending=True)
                        keep_idx = torch.cat([accepted, rej_idx[order[:N - accepted.numel()]]])
                        if utils.is_main_process():
                            print(f"[thinned] under-filled (accepted {accepted.numel()} < N={N}) at iter "
                                  f"{current_iteration}; topped up from pool -- consider raising "
                                  f"--thin_oversample_factor")
                    thin_committed_phat = p_local[keep_idx]
                survivor_raw = pool_raw[keep_idx.detach().to(pool_raw.device)]   # [N,3,224,224] uint8
            # GPU-augment the committed survivors: raw uint8 -> [g1, g2, l1..lL] normalized, on GPU.
            batch_data = gpu_augment(survivor_raw.to('cuda', non_blocking=True))
        # thin_active: 1 as soon as the gate is open (analytic w_max -> no calibration window).
        thin_active = 1 if (balance_mode == 'thinned' and thin_gate_open) else 0

        # === 1. Extract crops from batch ===
        idx = 0

        teacher_global_crops = []
        for i in range(args.global_views):
            teacher_global_crops.append(batch_data[idx].cuda(non_blocking=True))
            idx += 1

        student_all_crops = []
        for crop in teacher_global_crops:
            student_all_crops.append(crop)

        student_local_crops = []
        for i in range(args.n_standard_local_crops):
            crop = batch_data[idx].cuda(non_blocking=True)
            student_local_crops.append(crop)
            student_all_crops.append(crop)
            idx += 1

        # First global crop feeds every frozen mask model in the augmentation blocks below.
        mask_model_input = teacher_global_crops[0]

        # === 2. Generate block masks for standard iBOT ===
        batch_size = teacher_global_crops[0].shape[0]
        n_patches_h = n_patches_w = 224 // args.patch_size

        block_masks_1, masks_weight_1 = generate_block_masks(
            batch_size, n_patches_h, n_patches_w,
            mask_ratio_min=args.mask_ratio_min,
            mask_ratio_max=args.mask_ratio_max,
            mask_sample_probability=args.mask_sample_probability,
            device=teacher_global_crops[0].device
        )

        block_masks_2, masks_weight_2 = generate_block_masks(
            batch_size, n_patches_h, n_patches_w,
            mask_ratio_min=args.mask_ratio_min,
            mask_ratio_max=args.mask_ratio_max,
            mask_sample_probability=args.mask_sample_probability,
            device=teacher_global_crops[0].device
        )

        # === 3. Generate semantic token masks (if enabled) ===
        semantic_token_masks = None
        semantic_masks_weights = None
        selected_channels = None

        if args.use_semantic_ibot and mask_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16 or bf16_mode):
                    mask_output = mask_model_frozen(teacher_global_crops[0])
                    soft_masks = mask_output['masks'].float()

            all_token_masks, all_masks_weights = convert_semantic_masks_to_token_masks(
                soft_masks, patch_size=args.patch_size
            )

            if args.semantic_masks_per_iteration >= args.num_masks:
                selected_channels = list(range(args.num_masks))
            else:
                selected_channels = random.sample(
                    range(args.num_masks), args.semantic_masks_per_iteration
                )

            semantic_token_masks = [all_token_masks[:, c, :] for c in selected_channels]
            semantic_masks_weights = [all_masks_weights[:, c] for c in selected_channels]

            # Apply mask_sample_probability gating (same as block masks)
            for i in range(len(semantic_token_masks)):
                keep = torch.rand(batch_size, device=semantic_token_masks[i].device) < args.mask_sample_probability
                semantic_token_masks[i] = semantic_token_masks[i] & keep.unsqueeze(1)
                num_masked = semantic_token_masks[i].sum(dim=1).float()
                semantic_masks_weights[i] = torch.where(
                    num_masked > 0, 1.0 / num_masked, torch.zeros_like(num_masked)
                )

        # === 4. Adversarial mask-as-student-view augmentation (optional) ===
        # Applies the adversarial mask model's soft masks as image-level masks,
        # producing additional student views. Distinct from semantic iBOT, which
        # uses the same model's output as token-level masks inside the iBOT loss.
        masked_global_crops = []
        masked_local_crops_all = []

        if args.use_adversarial_mask_augmentation and mask_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16 or bf16_mode):
                    mask_output = mask_model_frozen(mask_model_input)
                    masks = mask_output['masks'].float()  # Cast back to float32 for downstream ops

            # Apply masks to create masked global views
            masked_images = apply_masks_to_images(mask_model_input, masks)
            masked_global_crops = masked_images

            # Add masked global crops to student views
            student_all_crops.extend(masked_global_crops)

            # Extract local crops from masked images
            if args.crops_per_mask > 0:
                for masked_img in masked_images:
                    crops = extract_local_crops_from_masked(
                        masked_img,
                        n_crops=args.crops_per_mask,
                        crop_size=args.local_crop_size
                    )
                    masked_local_crops_all.extend(crops)

                # Add masked local crops to student views
                student_all_crops.extend(masked_local_crops_all)

        # === 5. CellViT (nuclei / background) augmentation (optional) ===
        cellvit_nuclei_global = None
        cellvit_background_global = None
        cellvit_nuclei_crops = []
        cellvit_background_crops = []

        if args.use_cellvit_augmentation and cellvit_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    cellvit_output = cellvit_model_frozen(mask_model_input)
                    cellvit_masks = cellvit_output['masks'].float()  # [B, 2, H, W]

            del cellvit_output

            # Apply masks to create nuclei and background views
            nuclei_images, background_images = apply_cellvit_masks(mask_model_input, cellvit_masks)
            del cellvit_masks

            # Global views (224x224)
            cellvit_nuclei_global = nuclei_images.clone()
            cellvit_background_global = background_images.clone()

            # Add global views to student crops
            student_all_crops.append(cellvit_nuclei_global)
            student_all_crops.append(cellvit_background_global)

            # Extract local crops (local_crop_size) from each channel
            if args.cellvit_crops_per_channel > 0:
                nuclei_crops = extract_crops_from_cellvit_channel(
                    nuclei_images,
                    n_crops=args.cellvit_crops_per_channel,
                    crop_size=args.local_crop_size
                )
                background_crops = extract_crops_from_cellvit_channel(
                    background_images,
                    n_crops=args.cellvit_crops_per_channel,
                    crop_size=args.local_crop_size
                )

                cellvit_nuclei_crops = nuclei_crops
                cellvit_background_crops = background_crops
                student_all_crops.extend(nuclei_crops)
                student_all_crops.extend(background_crops)

                del nuclei_crops, background_crops

            del nuclei_images, background_images

        # === 6. Random rectangular mask augmentation (optional) ===
        random_masked_global_crops = []
        random_masked_local_crops = []

        if args.use_random_mask_augmentation:
            random_masks = generate_random_image_masks(
                batch_size=mask_model_input.shape[0],
                num_masks=args.random_num_masks,
                height=224,
                width=224,
                device=mask_model_input.device,
            )

            random_masked_images = apply_masks_to_images(mask_model_input, random_masks)
            random_masked_global_crops = random_masked_images

            student_all_crops.extend(random_masked_global_crops)

            if args.random_crops_per_mask > 0:
                for masked_img in random_masked_images:
                    crops = extract_local_crops_from_masked(
                        masked_img,
                        n_crops=args.random_crops_per_mask,
                        crop_size=args.local_crop_size
                    )
                    random_masked_local_crops.extend(crops)

                student_all_crops.extend(random_masked_local_crops)

        # ========== Debug: Print shapes on first iteration ==========
        if current_iteration == 0 and utils.is_main_process():
            print("\n=== Crop Organization (First Iteration) ===")
            print(f"Teacher global crops: {len(teacher_global_crops)} crops")
            for i, crop in enumerate(teacher_global_crops):
                print(f"  Teacher crop {i}: {crop.shape}")
            print(f"Student total crops: {len(student_all_crops)} crops")
            for i, crop in enumerate(student_all_crops):
                print(f"  Student crop {i}: {crop.shape}")
            print(f"Mask model input: {mask_model_input.shape}")
            print(f"Block masks 1: {block_masks_1.shape}, masks_weight_1: {masks_weight_1.shape}")
            print(f"Block masks 2: {block_masks_2.shape}, masks_weight_2: {masks_weight_2.shape}")

            if args.use_semantic_ibot and semantic_token_masks is not None:
                print(f"\nSemantic iBOT:")
                print(f"  Selected channels: {selected_channels}")
                for ch_idx, (sm, sw) in enumerate(zip(semantic_token_masks, semantic_masks_weights)):
                    n_masked = sm.sum(dim=1).float().mean().item()
                    print(f"  Channel {selected_channels[ch_idx]}: mask shape {sm.shape}, avg masked patches: {n_masked:.1f}")

            if args.use_adversarial_mask_augmentation:
                print(f"\nAdversarial Mask Augmentation:")
                print(f"  Masked global crops: {len(masked_global_crops)}")
                print(f"  Masked local crops: {len(masked_local_crops_all)}")

            if args.use_cellvit_augmentation:
                print(f"\nCellViT Augmentation:")
                print(f"  Nuclei global: {cellvit_nuclei_global.shape if cellvit_nuclei_global is not None else 'None'}")
                print(f"  Background global: {cellvit_background_global.shape if cellvit_background_global is not None else 'None'}")
                print(f"  Nuclei crops: {len(cellvit_nuclei_crops)} x {cellvit_nuclei_crops[0].shape if cellvit_nuclei_crops else 'None'}")
                print(f"  Background crops: {len(cellvit_background_crops)} x {cellvit_background_crops[0].shape if cellvit_background_crops else 'None'}")

            if args.use_random_mask_augmentation:
                print(f"\nRandom Mask Augmentation:")
                print(f"  Random masked global crops: {len(random_masked_global_crops)}")
                print(f"  Random masked local crops: {len(random_masked_local_crops)}")

            if args.use_typicality_dampening:
                print(f"\nTypicality Dampening:")
                print(f"  K' = {args.typicality_K_prime}, Bank M = {args.typicality_bank_size}")
                print(f"  Modulation: {args.typicality_modulation}")
                print(f"  return_bottleneck = True")

            print("="*50 + "\n")

        # ========== Update learning rates ==========
        for i, param_group in enumerate(optimizer_student.param_groups):
            base_lr = student_lr_schedule[current_iteration]
            lr_mult = param_group.get("lr_multiplier", 1.0)
            param_group["lr"] = base_lr * lr_mult

            wd_mult = param_group.get("wd_multiplier", 1.0)
            if wd_mult > 0:
                param_group["weight_decay"] = wd_schedule[current_iteration] * wd_mult

        if args.use_prototype_clustering and optimizer_prototypes is not None:
            for param_group in optimizer_prototypes.param_groups:
                param_group["lr"] = proto_lr_schedule[current_iteration]

        optimizer_student.zero_grad()
        if args.use_prototype_clustering and optimizer_prototypes is not None:
            optimizer_prototypes.zero_grad()

        # ========== Forward passes and loss computation ==========
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16 or bf16_mode):
            # ========== DINO Loss with Sequence Packing ==========

            student_masks = [block_masks_1, block_masks_2] + [None] * len(student_local_crops)

            # Teacher forward (unmasked targets)
            with torch.no_grad():
                teacher_output = teacher(teacher_global_crops, token_masks=[None, None], mode='dino')
                teacher_cls_outputs = teacher_output['cls_outputs']
                teacher_patch_tokens_g1 = teacher_output['features_list'][0]['patchtokens']
                teacher_patch_tokens_g2 = teacher_output['features_list'][1]['patchtokens']

            # Student forward (all crops, masks on global crops)
            student_output = student(student_all_crops, token_masks=student_masks, mode='dino',
                                     return_bottleneck=args.use_typicality_dampening)
            student_cls_outputs = student_output['cls_outputs']
            student_patch_tokens_g1 = student_output['features_list'][0]['patchtokens']
            student_patch_tokens_g2 = student_output['features_list'][1]['patchtokens']

            # ========== Typicality Dampening ==========
            typicality_weights = None
            out = {'ready': False}
            p_hat = None

            # In thinned mode the bank was already updated on the scout pool above and the loss is
            # unweighted, so this in-place update + weighting is skipped entirely (relocation).
            if (args.use_typicality_dampening and repr_protos is not None
                    and balance_mode != 'thinned'
                    and current_iteration >= args.typicality_warmup_iters):
                with torch.no_grad():
                    # Extract bottleneck from global crop 1: first B entries
                    z_global1 = student_output['bottleneck'][:batch_size].detach()

                    # Compute morphology signatures (local, detached)
                    s_batch = repr_protos.compute_signatures(z_global1)

                    # All-gather signatures so the bank runs identically on every rank (rank
                    # order preserved -> byte-identical, fingerprint-checked below). The bank
                    # returns p_hat over the gathered batch; take this rank's local rows for the
                    # local loss. The gate is iteration-based/rank-invariant, so all ranks
                    # reach the collective together.
                    s_global = _all_gather_signatures(s_batch.detach())
                    out = typicality_bank.score_and_update(s_global, current_iteration)
                    if out['ready'] and current_iteration % 2000 == 0:
                        _assert_bank_synced(typicality_bank.sync_fingerprint(), tag=f"@it{current_iteration}")

                    # Fixed-radius readout: out['p_hat'] is an absolute density (None during the
                    # cold-start phase -> no modulation, explicit). The counted bank drives the
                    # weighted-loss weight w = 1/(p_hat + c)^a with c = c_frac * p_ref (the only
                    # modulation; adaptive temperature was removed).
                    if out.get('p_hat') is not None and current_iteration >= args.typicality_warmup_iters:
                        p_hat = _local_rows(out['p_hat'], batch_size)
                        typicality_weights = TypicalityScorer.absolute_weights(
                            p_hat, typicality_bank.p_ref,
                            args.typicality_a, args.typicality_c_frac,
                        )

            # DINO CLS loss. In thinned mode the rebalancing is realized by admission (the batch is
            # already density-thinned), so the loss is UNWEIGHTED: the typicality block above is
            # skipped and typicality_weights stays None -> sample_weights=None. (weighted mode keeps
            # its per-tile weights; off has none.)
            if balance_mode == 'thinned':
                assert typicality_weights is None, "thinned mode must not weight the committed loss"
            dino_class_loss_val = dino_class_loss(
                student_cls_outputs,
                teacher_cls_outputs,
                current_iteration,
                sample_weights=typicality_weights,
            )

            # ========== KoLeo Loss ==========
            # Matches morphological_rarity: if adversarial-mask-as-student-view is enabled,
            # its masked globals are included in the KoLeo regularizer (CellViT / random
            # masked globals are left out).
            num_global_total = args.global_views
            if args.use_adversarial_mask_augmentation:
                num_global_total += args.num_masks
            global_features_list = student_output['features_list'][:num_global_total]
            global_cls_tokens = [feat_dict['clstoken'] for feat_dict in global_features_list]

            koleo_loss_val = torch.tensor(0.0).cuda()
            if len(global_cls_tokens) > 0:
                # Canonical DINOv2 / Virchow2: SUM the regularizer over the global
                # crops (do NOT average). Applies to both KoLeo and KDE.
                koleo_loss_val = sum(dino_koleo_loss(token) for token in global_cls_tokens)

            # ================================================================
            # iBOT Loss — gather-then-project to avoid [B, N, 65536] tensors.
            # Projects only masked tokens (~7k) instead of all tokens (~50k).
            # Semantic iBOT integrated here, reusing teacher backbone tokens.
            # ================================================================
            current_teacher_temp_ibot = dino_class_loss.teacher_temp_schedule(current_iteration)

            semantic_ibot_loss_val = torch.tensor(0.0, device='cuda')
            semantic_backbone_outputs = []  # Raw [B, N, D] tokens for prototype section

            # ---------- Block iBOT: Global crop 1 ----------
            s_gathered_1, weights_1, B = _gather_and_compute_weights(
                student_patch_tokens_g1, block_masks_1, masks_weight_1
            )

            if s_gathered_1 is not None:
                with torch.no_grad():
                    t_gathered_1 = teacher_patch_tokens_g1.reshape(-1, teacher_patch_tokens_g1.shape[-1])[
                        block_masks_1.reshape(-1).nonzero(as_tuple=True)[0]
                    ]
                    t_proj_1 = teacher.module.patchhead(t_gathered_1)

                s_proj_1 = student.module.patchhead(s_gathered_1)

                ibot_loss_g1 = ibot_patch_loss.forward_gathered(
                    s_proj_1, t_proj_1, weights_1, B, current_teacher_temp_ibot
                )
                del s_proj_1, t_proj_1, s_gathered_1, t_gathered_1
            else:
                ibot_loss_g1 = torch.tensor(0.0, device='cuda')

            # ---------- Semantic iBOT on global crop 1 ----------
            # Memory optimization: only one channel is randomly selected
            # downstream for the prototype-clustering path (see prototype
            # block). Pre-pick that channel here so the loop can release
            # the other (N-1) backbone graphs as soon as their iBOT loss
            # is computed, instead of holding all N graphs alive until
            # the prototype block. Saves ~3 GB peak at 3 channels, ViT-B/14.
            if args.use_semantic_ibot and semantic_token_masks is not None:
                ibot_accum = 0.0
                n_ch = len(semantic_token_masks)

                # Pre-pick which channel to retain for the prototype path.
                if args.use_semantic_prototypes:
                    retain_idx = random.randint(0, n_ch - 1)
                else:
                    retain_idx = -1  # no retention

                # BATCHED semantic backbone: one forward over ALL channels at once,
                # replacing n_ch sequential full-backbone forwards. teacher_global_crops[0]
                # is the same input for every channel, so tile it n_ch times and stack the
                # per-channel masks; row (c*B + b) is sample b masked by channel c. The ViT
                # backbone is per-sample (LayerNorm + attention, no BatchNorm), so this is
                # bit-identical to n_ch separate forwards. Trade-off: all n_ch graphs stay
                # alive (the previous per-channel graph-release is dropped) -- funded by the
                # compile + checkpoint memory savings on this branch.
                sem_input = teacher_global_crops[0].repeat(n_ch, 1, 1, 1)          # [n_ch*B, C, H, W]
                sem_mask_all = torch.cat(list(semantic_token_masks), dim=0)        # [n_ch*B, num_tokens]
                sem_backbone_out = student.module.backbone(
                    sem_input, token_masks=sem_mask_all, return_dict=True
                )
                sem_patch_raw_all = sem_backbone_out['patchtokens_postnorm']       # [n_ch*B, N, D]

                for ch_idx, (sem_mask, sem_weight) in enumerate(
                    zip(semantic_token_masks, semantic_masks_weights)
                ):
                    sem_patch_raw = sem_patch_raw_all[ch_idx * B:(ch_idx + 1) * B]  # [B, N, D] view

                    # Store for prototype section ONLY if this is the retain channel.
                    if ch_idx == retain_idx:
                        semantic_backbone_outputs.append((sem_patch_raw, sem_mask, sem_weight))

                    # Gather only masked tokens, then project
                    sem_s_gathered, sem_weights, _ = _gather_and_compute_weights(
                        sem_patch_raw, sem_mask, sem_weight
                    )

                    if sem_s_gathered is not None:
                        with torch.no_grad():
                            sem_t_gathered = teacher_patch_tokens_g1.reshape(-1, teacher_patch_tokens_g1.shape[-1])[
                                sem_mask.reshape(-1).nonzero(as_tuple=True)[0]
                            ]
                            sem_t_proj = teacher.module.patchhead(sem_t_gathered)

                        sem_s_proj = student.module.patchhead(sem_s_gathered)

                        loss_ibot = ibot_patch_loss.forward_gathered(
                            sem_s_proj, sem_t_proj, sem_weights, B, current_teacher_temp_ibot
                        )
                        ibot_accum += loss_ibot
                        del sem_s_proj, sem_t_proj, sem_s_gathered, sem_t_gathered

                # sem_patch_raw_all is kept alive (via the retained view in
                # semantic_backbone_outputs) until the prototype block; drop the dict wrapper.
                del sem_backbone_out
                semantic_ibot_loss_val = ibot_accum / n_ch

            # ---------- Block iBOT: Global crop 2 ----------
            s_gathered_2, weights_2, _ = _gather_and_compute_weights(
                student_patch_tokens_g2, block_masks_2, masks_weight_2
            )

            if s_gathered_2 is not None:
                with torch.no_grad():
                    t_gathered_2 = teacher_patch_tokens_g2.reshape(-1, teacher_patch_tokens_g2.shape[-1])[
                        block_masks_2.reshape(-1).nonzero(as_tuple=True)[0]
                    ]
                    t_proj_2 = teacher.module.patchhead(t_gathered_2)

                s_proj_2 = student.module.patchhead(s_gathered_2)

                ibot_loss_g2 = ibot_patch_loss.forward_gathered(
                    s_proj_2, t_proj_2, weights_2, B, current_teacher_temp_ibot
                )
                del s_proj_2, t_proj_2, s_gathered_2, t_gathered_2
            else:
                ibot_loss_g2 = torch.tensor(0.0, device='cuda')

            ibot_loss_val = (ibot_loss_g1 + ibot_loss_g2) / 2.0

        # ================================================================
        # Patch Prototype Clustering
        # Operates on backbone-dim [B, N, 768] — no memory concern.
        # Semantic prototype loss integrated here.
        # ================================================================
        semantic_clustering_loss = torch.tensor(0.0, device='cuda')
        semantic_teacher_proto_loss = torch.tensor(0.0, device='cuda')
        semantic_koleo_proto_loss = torch.tensor(0.0, device='cuda')

        if args.use_prototype_clustering:
            current_teacher_temp = dino_class_loss.teacher_temp_schedule(current_iteration)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16 or bf16_mode):
                # ---------- Block mask prototype: Global crop 1 ----------
                # Also returns Q_g1 (teacher assignments) so the semantic call below,
                # which feeds the IDENTICAL teacher input, can reuse it instead of
                # recomputing the full teacher path (bank matmul + Sinkhorn = 4 collectives).
                clust_loss_g1, proto_loss_g1, koleo_proto_g1, Q_g1 = patch_prototype_loss(
                    teacher_patch_tokens_g1,
                    student_patch_tokens_g1,
                    block_masks_1,
                    prototype_bank,
                    current_iteration,
                    current_teacher_temp,
                    masks_weight=masks_weight_1
                )

                # ---------- Semantic prototype on global crop 1 ----------
                # Reuse Q_g1 (bit-identical to recomputing it from teacher_patch_tokens_g1)
                # and run ONLY the student prediction. The teacher arrangement + koleo are
                # NOT re-added to prototype_loss: they are identical to g1's and were
                # previously double-counted (g1 weighted 1.5x / koleo 2x). Only the distinct
                # semantic student term feeds student_loss (via semantic_clustering_weight).
                if args.use_semantic_prototypes and len(semantic_backbone_outputs) > 0:
                    # Randomly select one semantic channel for prototype loss (for economical reasons)
                    sem_idx = random.randint(0, len(semantic_backbone_outputs) - 1)
                    sem_patch_raw, sem_mask, sem_weight = semantic_backbone_outputs[sem_idx]

                    semantic_clustering_loss, _ = patch_prototype_loss.student_prediction(
                        sem_patch_raw,
                        sem_mask,
                        Q_g1,
                        prototype_bank,
                        current_iteration,
                        masks_weight=sem_weight
                    )

                # Free semantic backbone outputs
                del semantic_backbone_outputs
                semantic_backbone_outputs = []

                # ---------- Block mask prototype: Global crop 2 ----------
                clust_loss_g2, proto_loss_g2, koleo_proto_g2, Q_g2 = patch_prototype_loss(
                    teacher_patch_tokens_g2,
                    student_patch_tokens_g2,
                    block_masks_2,
                    prototype_bank,
                    current_iteration,
                    current_teacher_temp,
                    masks_weight=masks_weight_2
                )

                clustering_loss = (clust_loss_g1 + clust_loss_g2) / 2.0
                teacher_proto_loss = (proto_loss_g1 + proto_loss_g2) / 2.0
                koleo_proto_loss = (koleo_proto_g1 + koleo_proto_g2) / 2.0

            # Double-count removed: the semantic call's teacher arrangement + koleo (which
            # equal g1's) are no longer added here. The bank now sees each teacher crop's
            # arrangement once, averaged over g1/g2, plus koleo once averaged over g1/g2.
            prototype_loss = teacher_proto_loss + koleo_proto_loss
        else:
            del semantic_backbone_outputs

            clustering_loss = torch.tensor(0.0).cuda()
            teacher_proto_loss = torch.tensor(0.0).cuda()
            koleo_proto_loss = torch.tensor(0.0).cuda()
            prototype_loss = torch.tensor(0.0).cuda()

        # ========== Compute Total Losses ==========
        student_loss = (
            dino_class_loss_val +
            args.koleo_loss_weight * koleo_loss_val +
            args.ibot_loss_weight * ibot_loss_val +
            args.clustering_weight * clustering_loss +
            args.semantic_ibot_weight * semantic_ibot_loss_val +
            args.semantic_clustering_weight * semantic_clustering_loss
        )

        # ========== Backward and optimizer steps ==========
        if fp16_scaler is None:
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                prototype_bank.sync_grads()          # hand-rolled DP: reduce bank grads
                optimizer_prototypes.step()

            optimizer_student.zero_grad()
            student_loss.backward()
            student.sync_grads()                     # hand-rolled DP: reduce BEFORE clip/step

            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.module.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.module.patchhead, args.freeze_last_layer_iters)

            optimizer_student.step()

        else:
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                prototype_bank.sync_grads()          # hand-rolled DP: reduce bank grads
                optimizer_prototypes.step()

            optimizer_student.zero_grad()
            fp16_scaler.scale(student_loss).backward()
            fp16_scaler.unscale_(optimizer_student)
            student.sync_grads()                     # reduce unscaled grads BEFORE clip/step

            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.module.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.module.patchhead, args.freeze_last_layer_iters)

            fp16_scaler.step(optimizer_student)
            fp16_scaler.update()

        # ========== Representative prototype update (typicality) ==========
        if args.use_typicality_dampening and repr_protos is not None:
            P = student.module.classhead.last_layer.weight.detach()
            repr_loss, l_nn, l_cov = repr_protos.compute_loss(P)
            R_optimizer.zero_grad()
            repr_loss.backward()
            R_optimizer.step()
            repr_protos.project_to_sphere()

        # ========== EMA update teacher ==========
        with torch.no_grad():
            m = momentum_schedule[current_iteration]

            for param_q, param_k in zip(student.module.backbone.parameters(),
                                    teacher_without_ddp.backbone.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            for param_q, param_k in zip(student.module.classhead.parameters(),
                                    teacher_without_ddp.classhead.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            for param_q, param_k in zip(student.module.patchhead.parameters(),
                                    teacher_without_ddp.patchhead.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # ========== Clean cache periodically ==========
        if current_iteration % 100 == 0:
            torch.cuda.empty_cache()

        # ========== Visualize masks ==========
        if current_iteration % args.visualization_freq == 0 and current_iteration < 5000:
            if args.use_semantic_ibot and mask_model_frozen is not None:
                sample_image = teacher_global_crops[0][:1]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16 or bf16_mode):
                        vis_masks = mask_model_frozen(sample_image)['masks']
                    save_iteration_masks_efficient(
                        sample_image,
                        vis_masks,
                        current_iteration,
                        os.path.join(args.output_dir, 'semantic_mask_visualizations'),
                        num_samples=1
                    )
                    vis_token_masks, _ = convert_semantic_masks_to_token_masks(
                        vis_masks.float(), patch_size=args.patch_size
                    )
                    h_patches = w_patches = 224 // args.patch_size
                    vis_token_grid = vis_token_masks.float().reshape(1, args.num_masks, h_patches, w_patches)
                    vis_token_upsampled = F.interpolate(
                        vis_token_grid, size=(224, 224), mode='nearest'
                    )
                    save_iteration_masks_efficient(
                        sample_image,
                        vis_token_upsampled,
                        current_iteration,
                        os.path.join(args.output_dir, 'semantic_token_mask_visualizations'),
                        num_samples=1
                    )

        # ========== Logging ==========
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(dino_class_loss=dino_class_loss_val.item())
        metric_logger.update(koleo_loss=koleo_loss_val.item())
        metric_logger.update(ibot_loss=ibot_loss_val.item())

        if args.use_prototype_clustering:
            metric_logger.update(clustering_loss=clustering_loss.item())
            metric_logger.update(proto_koleo_loss=koleo_proto_loss.item())
            metric_logger.update(teacher_proto_arrangement_loss=teacher_proto_loss.item())
            metric_logger.update(clustering_entropy=patch_prototype_loss.last_entropy)

        if args.use_semantic_ibot:
            metric_logger.update(semantic_ibot_loss=semantic_ibot_loss_val.item())
        if args.use_semantic_prototypes:
            metric_logger.update(semantic_clustering_loss=semantic_clustering_loss.item())
            metric_logger.update(semantic_proto_arrangement=semantic_teacher_proto_loss.item())
            metric_logger.update(semantic_proto_koleo=semantic_koleo_proto_loss.item())

        if args.use_typicality_dampening and repr_protos is not None:
            metric_logger.update(repr_L_nn=l_nn.item())
            metric_logger.update(repr_L_cov=l_cov.item())
            if current_iteration >= args.typicality_warmup_iters:
                # Compact scalar meters for plot_typicality.py, with a STABLE column set logged
                # every step -- the readout meters below carry cold-start defaults so the log
                # format does not change when the bank matures. The one-line 'typ | ...' status
                # is printed alongside 'It .../...' below and reuses these same values.
                h = typicality_bank.health()
                metric_logger.update(
                    typ_s=h['s'], typ_j=h['j'], typ_h_over_L=h['h_over_L'],
                    typ_n_est=h['n_est'], typ_reserve=h['n_reserve'], typ_hit_frac=h['hit_frac'],
                    typ_lam_q10=h['lam_q10'], typ_lam_median=h['lam_median'], typ_lam_q90=h['lam_q90'],
                    typ_lam_spread=h['lam_spread'], typ_evict_over_q10=h['evict_over_q10'],
                    typ_underresolved=h['underresolved'], typ_graduations=h['graduations'],
                    # migrated from the retired 'typ | ...' pipe line so no information is lost:
                    typ_s_delta=h['s_delta'], typ_reserve_cap=h['reserve_cap'],
                    typ_exit_grad=h['step_grad'], typ_exit_expired=h['step_expired'],
                    typ_exit_flushed=h['step_flushed'], typ_exit_age=h['exit_age_mean'],
                )
                # Fixed-radius readout: p_median/p_ref are the density and its reference scale
                # (rank-invariant, from the gathered batch); w_mean/ESS describe the realized
                # weights. Defaults hold during cold start / non-weighted arms -- no modulation is
                # a uniform weight of 1 and a full effective batch. typ_ess = sum(w)^2/(sum(w^2)*B)
                # is the health signal (t_std was uninformative: a rank statistic is uniform on
                # [0,1] by construction, so it sat at sqrt(1/12) and never moved -- ESS can move).
                typ_pmed, typ_pref = 0.0, float(typicality_bank.p_ref.item())
                typ_wmean, typ_ess = 1.0, 1.0
                if out.get('p_hat') is not None:
                    _ph = out['p_hat']
                    typ_pmed = _ph.median().item()
                    if args.typicality_modulation == 'weighted_loss':
                        _w = TypicalityScorer.absolute_weights(
                            _ph, typicality_bank.p_ref, args.typicality_a, args.typicality_c_frac)
                        typ_wmean = _w.mean().item()
                        typ_ess = (_w.sum() ** 2 / (_w.pow(2).sum() + 1e-12)).item() / _w.numel()
                metric_logger.update(typ_p_median=typ_pmed, typ_p_ref=typ_pref,
                                     typ_w_mean=typ_wmean, typ_ess=typ_ess)
                if current_iteration == args.typicality_warmup_iters and utils.is_main_process():
                    # n_eff = (1+eta)/(1-eta): the estimator's variance floor, set by the
                    # half-life alone (independent of stream length).
                    print(f"[typicality] activated: n_eff = (1+eta)/(1-eta) = {h['n_eff']:.1f} "
                          f"(variance floor set by the half-life)")

                # Stream-thinning diagnostics (section 3.9), canonical line, thinned mode only.
                # thin_seen: candidates scouted to fill this step (~chi*N). thin_accept = N/seen
                # (~1/chi; drift is the miscalibration alarm). thin_bias: mean|beta_hat| (crude).
                # thin_prof_err: L1 between the committed p_hat histogram and the target a*mu.
                if balance_mode == 'thinned':
                    # thin_active: 0 while the gate is closed / during the w_max window, 1 once
                    # acceptance is live (frozen w_max). Distinguishes "thinning engaged but doing
                    # nothing" (gate closed / calibrating) from "thinning not yet started".
                    metric_logger.update(thin_active=float(thin_active))
                    if thin_seen_val:
                        # thin_accept is the REALIZED acceptance rate (accepted/pool ~ 1/chi), the
                        # calibration signal -- NOT N/pool (which is the fixed 1/factor). thin_seen
                        # is the pool size scouted this step.
                        metric_logger.update(thin_seen=float(thin_seen_val))
                        if thin_accept_val is not None:
                            metric_logger.update(thin_accept=float(thin_accept_val))
                        if thin_bias_val is not None:
                            metric_logger.update(thin_bias=float(thin_bias_val))
                        if (thin_committed_phat is not None and thin_pool_phat is not None
                                and thin_pool_phat.numel() > 1):
                            metric_logger.update(thin_prof_err=_thin_profile_err(
                                thin_committed_phat, thin_pool_phat,
                                float(typicality_bank.p_ref.item()),
                                a=args.typicality_a, c_frac=args.typicality_c_frac))

        metric_logger.update(lr=optimizer_student.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer_student.param_groups[0]["weight_decay"])

        if utils.is_main_process() and current_iteration % 10 == 0:
            # recent-window rate (survives resume + the thinning fast->slow phase change): sec/iter
            # measured over the last log window, EMA-smoothed, applied to the iterations REMAINING.
            _now = time.time()
            _d_it = current_iteration - eta_ref_iter
            if _d_it > 0:
                _inst = (_now - eta_ref_time) / _d_it
                eta_sec_per_it = _inst if eta_sec_per_it is None else 0.8 * eta_sec_per_it + 0.2 * _inst
                eta_ref_time, eta_ref_iter = _now, current_iteration
            progress = current_iteration / args.total_iterations
            eta_seconds = (eta_sec_per_it or 0.0) * (args.total_iterations - current_iteration)
            eta_string = (str(datetime.timedelta(seconds=int(eta_seconds)))
                          if eta_sec_per_it else "estimating...")

            if torch.cuda.is_available():
                memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory = 0

            metric_logger.synchronize_between_processes()
            print(f"It {current_iteration}/{args.total_iterations/1000:.0f}k (ETA {eta_string}), "
                f"Progress: {progress*100:.1f}%, max mem: {memory/1000:.1f} GB : {metric_logger}")

            # compact typicality status line (Part 4): dimensionless health at a glance
            if (args.use_typicality_dampening and repr_protos is not None
                    and current_iteration >= args.typicality_warmup_iters):
                _gnow = int(typicality_bank.graduations.item())
                _di = max(current_iteration - typ_last_grad_iter, 1)
                _gps = (_gnow - typ_last_grad) / _di            # graduations/step over the interval
                typ_last_grad = _gnow
                typ_last_grad_iter = current_iteration
                _turn = typicality_bank.M / _gps if _gps > 1e-9 else 0.0
                _neff = float(typicality_bank.n_eff)
                metric_logger.update(typ_grad_per_step=_gps, typ_turnover=_turn,
                                     typ_turn_ratio=(_turn / _neff if _neff > 0 else 0.0))
                # The custom pipe-delimited 'typ | ...' status line is retired: it duplicated the
                # canonical metric line, which now carries every one of its fields (s_delta,
                # reserve_cap, turn_ratio, the reserve-exit breakdown grad/exp/flush, exit age; the
                # churn/evict/fill/ess flags are thresholds on those canonical fields). No print here.

        # ========== Write to log file ==========
        if utils.is_main_process() and current_iteration % 100 == 0:
            log_stats = {
                **{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()},
                'iteration': current_iteration,
                'total_iterations': args.total_iterations,
                'progress_percentage': (current_iteration / args.total_iterations) * 100,
                'augmentation_config': {
                    'global_views': args.global_views,
                    'n_standard_local_crops': args.n_standard_local_crops,
                    'use_semantic_ibot': args.use_semantic_ibot,
                    'use_semantic_prototypes': args.use_semantic_prototypes,
                    'semantic_ibot_weight': args.semantic_ibot_weight if args.use_semantic_ibot else 0,
                    'semantic_clustering_weight': args.semantic_clustering_weight if args.use_semantic_prototypes else 0,
                    'semantic_masks_per_iteration': args.semantic_masks_per_iteration if args.use_semantic_ibot else 0,
                    'num_masks': args.num_masks if (args.use_semantic_ibot or args.use_adversarial_mask_augmentation) else 0,
                    'adversarial_mask_augmentation': args.use_adversarial_mask_augmentation,
                    'crops_per_mask': args.crops_per_mask if args.use_adversarial_mask_augmentation else 0,
                    'cellvit_augmentation': args.use_cellvit_augmentation,
                    'cellvit_crops_per_channel': args.cellvit_crops_per_channel if args.use_cellvit_augmentation else 0,
                    'random_mask_augmentation': args.use_random_mask_augmentation,
                    'random_num_masks': args.random_num_masks if args.use_random_mask_augmentation else 0,
                    'random_crops_per_mask': args.random_crops_per_mask if args.use_random_mask_augmentation else 0,
                    'total_student_views': total_student_views,
                    'typicality_dampening': args.use_typicality_dampening,
                }
            }

            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # ========== Save checkpoints ==========
        if current_iteration % args.save_checkpoint_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'dino_class_loss': dino_class_loss.state_dict(),
                'optimizer_student': optimizer_student.state_dict(),
                'iteration': current_iteration,
                'dataset_position': dataset_position,
                'args': args,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
            }

            if args.use_prototype_clustering:
                if prototype_bank is not None:
                    save_dict['prototype_bank'] = prototype_bank.state_dict()
                if patch_prototype_loss is not None:
                    save_dict['patch_prototype_loss'] = patch_prototype_loss.state_dict()
                if optimizer_prototypes is not None:
                    save_dict['optimizer_prototypes'] = optimizer_prototypes.state_dict()

            # Add typicality-related state only if enabled
            if args.use_typicality_dampening:
                if repr_protos is not None:
                    save_dict['repr_protos'] = repr_protos.state_dict()
                if R_optimizer is not None:
                    save_dict['R_optimizer'] = R_optimizer.state_dict()
                if typicality_bank is not None:
                    save_dict['typicality_bank'] = typicality_bank.state_dict()
                    # Counted-coverage bank health (logged per checkpoint).
                    if hasattr(typicality_bank, 'stats'):
                        _st = typicality_bank.stats()
                        print(f"  [counted bank] n_est={_st['n_est']}/{args.typicality_bank_size} "
                              f"reserve={_st['n_reserve']} graduations={_st['graduations']} "
                              f"s={_st['s']:.4f}")
                if richardson_bank is not None:
                    save_dict['richardson_bank'] = richardson_bank.state_dict()

            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_iter_{current_iteration:08d}.pth'))
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        current_iteration += 1

        if current_iteration % 100 == 0:
            if dist.is_initialized():
                dist.barrier()

    # ========== Final checkpoint and log ==========
    if utils.is_main_process():
        final_log_stats = {
            **{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()},
            'iteration': args.total_iterations,
            'training_completed': True,
        }

        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(final_log_stats) + "\n")

    print("Training Complete!")