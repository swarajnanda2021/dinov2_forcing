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
from configs import apply_vit_variant
from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
from models.vision_transformer.modern_vit import Mlp, SwiGLUFFNFused
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, PatchPrototypeLoss
from data import ProportionalMultiDatasetWrapper
from .helpers import (
    generate_block_masks,
    calculate_total_student_views,
    worker_init_fn,
    setup_ddp_model,
)


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
    apply_vit_variant(args)

    if getattr(args, 'qk_norm', None) is None:
        # Corrected DINOv2 default: qk_norm on at every model size when the
        # user did not set it explicitly. Explicit True/False from the CLI /
        # launch script always wins. Pass --qk_norm False (or args.qk_norm =
        # False in the launcher) to reproduce the older no-QK-norm runs.
        args.qk_norm = True

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    # Device: CUDA on the cluster; CPU only for the local dry run (scripts/local_dryrun.sh).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_device_type = device.type

    # Augmentation configuration
    augmentation_free_mode = (args.global_views == 0)

    print("\n========== Augmentation Configuration ==========")
    print(f"Global views (teacher/student): {args.global_views}")
    print(f"Standard local crops: {args.n_standard_local_crops}")
    print(f"Local crop size: {args.local_crop_size}x{args.local_crop_size}")

    total_student_views = calculate_total_student_views(args)
    print(f"\nTotal student views: {total_student_views}")
    print("================================================\n")

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
        patch_size=args.patch_size,
    )

    teacher = CombinedModelDINO(
        backbone=teacher_encoder,
        classhead=teacher_classhead,
        patchhead=teacher_patchhead,
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
        prototype_bank = prototype_bank.to(device)

        print(f"Created LinearPrototypeBank with {args.num_prototypes} soft prototypes")
    else:
        print("Prototype clustering disabled (--use_prototype_clustering=False)")

    student = student.to(device)
    teacher = teacher.to(device)

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
    dino_class_loss = DINOLoss(
        ncrops=total_student_views,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_iters=args.teacher_temp_warmup_iters,
        n_iterations=5,
        student_temp=0.1,
    ).to(device)

    ibot_patch_loss = iBOTPatchLoss(
        student_temp=0.1,
        n_iterations=3,
    ).to(device)

    dino_koleo_loss = KoLeoLoss().to(device)
    print("Using KoLeo regularizer (DINOv2 default)")

    patch_prototype_loss = None
    if args.use_prototype_clustering:
        patch_prototype_loss = PatchPrototypeLoss(
            num_prototypes=args.num_prototypes,
            embed_dim=args.embeddingdim,
            teacher_temp=args.clustering_teacher_temp,
            student_temp=args.clustering_student_temp,
        ).to(device)

        print(f"Initialized PatchPrototypeLoss with {args.num_prototypes} prototypes")
    else:
        print("Patch prototype clustering disabled")

    # ============ Create fp16_scaler ============
    # bf16 autocast needs no loss scaling; a default GradScaler doubles its scale
    # every 2000 clean steps unbounded under bf16 and overflows fp32 near it~190k
    # (observed scale 2^111 at 190k iters, then NaN collapse). Disabled.
    fp16_scaler = None

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

    # ============ Create schedulers ============
    # Peak lr keeps the source rule: base lr x sqrt(global_batch / 1024).
    peak_lr = args.lr * math.sqrt(args.batch_size_per_gpu * utils.get_world_size() / 1024.0)
    lr_schedule_kind = getattr(args, 'lr_schedule', 'cosine')
    wd_schedule_kind = getattr(args, 'wd_schedule', 'cosine')
    momentum_schedule_kind = getattr(args, 'momentum_schedule', 'cosine')
    momentum_teacher_end = getattr(args, 'momentum_teacher_end', 1.0)

    if lr_schedule_kind == 'constant':
        # linear warmup 0 -> peak over warmup_iterations, then hold the peak (min_lr ignored)
        student_lr_schedule = utils.cosine_scheduler(
            base_value=peak_lr,
            final_value=peak_lr,
            total_iters=args.total_iterations,
            warmup_iters=args.warmup_iterations,
            start_warmup_value=0
        )
    else:
        student_lr_schedule = utils.cosine_scheduler(
            base_value=peak_lr,
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

    if wd_schedule_kind == 'constant':
        wd_schedule = utils.cosine_scheduler(
            base_value=args.weight_decay,
            final_value=args.weight_decay,
            total_iters=args.total_iterations,
            warmup_iters=0,
            start_warmup_value=args.weight_decay
        )
    else:
        wd_schedule = utils.cosine_scheduler(
            base_value=args.weight_decay,
            final_value=args.weight_decay_end,
            total_iters=args.total_iterations,
            warmup_iters=args.warmup_iterations,
            start_warmup_value=args.weight_decay
        )

    if momentum_schedule_kind == 'constant':
        momentum_schedule = utils.cosine_scheduler(
            base_value=args.momentum_teacher,
            final_value=args.momentum_teacher,
            total_iters=args.total_iterations,
            warmup_iters=0,
            start_warmup_value=args.momentum_teacher
        )
    else:
        momentum_schedule = utils.cosine_scheduler(
            base_value=args.momentum_teacher,
            final_value=momentum_teacher_end,
            total_iters=args.total_iterations,
            warmup_iters=0,
            start_warmup_value=args.momentum_teacher
        )

    if utils.is_main_process():
        print(f"[sched-config] lr: {lr_schedule_kind} peak={peak_lr:.6g} "
              f"min={(peak_lr if lr_schedule_kind == 'constant' else args.min_lr):.6g} "
              f"warmup={args.warmup_iterations}")
        print(f"[sched-config] wd: {wd_schedule_kind} peak={args.weight_decay:.6g} "
              f"min={(args.weight_decay if wd_schedule_kind == 'constant' else args.weight_decay_end):.6g} "
              f"warmup={(0 if wd_schedule_kind == 'constant' else args.warmup_iterations)}")
        print(f"[sched-config] momentum: {momentum_schedule_kind} peak={args.momentum_teacher:.6g} "
              f"min={(args.momentum_teacher if momentum_schedule_kind == 'constant' else momentum_teacher_end):.6g} "
              f"warmup=0")

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
            if torch.cuda.is_available():
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
        print("="*50 + "\n")

    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    metric_logger.start_time = time.time()
    # ETA state: seconds/iter over a RECENT window (EMA), robust to resume -- eta_ref_iter is the
    # RESUMED iteration, not 0, so the estimate is not diluted by pre-restart progress.
    eta_sec_per_it, eta_ref_time, eta_ref_iter = None, metric_logger.start_time, current_iteration

    data_iterator = iter(train_loader)

    loader_len = len(train_loader) if len(train_loader) > 0 else 1
    dataset_passes = dataset_position // loader_len
    max_passes = 5

    if utils.is_main_process():
        print(f"Starting training at iteration {current_iteration}")

    # ============ Training loop ============
    print("Starting training!")

    while current_iteration < args.total_iterations:
        # ========== Get batch ==========
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
        # === 1. Extract crops from batch ===
        idx = 0

        teacher_global_crops = []
        for i in range(args.global_views):
            teacher_global_crops.append(batch_data[idx].to(device, non_blocking=True))
            idx += 1

        student_all_crops = []
        for crop in teacher_global_crops:
            student_all_crops.append(crop)

        student_local_crops = []
        for i in range(args.n_standard_local_crops):
            crop = batch_data[idx].to(device, non_blocking=True)
            student_local_crops.append(crop)
            student_all_crops.append(crop)
            idx += 1

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

        # ========== Debug: Print shapes on first iteration ==========
        if current_iteration == 0 and utils.is_main_process():
            print("\n=== Crop Organization (First Iteration) ===")
            print(f"Teacher global crops: {len(teacher_global_crops)} crops")
            for i, crop in enumerate(teacher_global_crops):
                print(f"  Teacher crop {i}: {crop.shape}")
            print(f"Student total crops: {len(student_all_crops)} crops")
            for i, crop in enumerate(student_all_crops):
                print(f"  Student crop {i}: {crop.shape}")
            print(f"Block masks 1: {block_masks_1.shape}, masks_weight_1: {masks_weight_1.shape}")
            print(f"Block masks 2: {block_masks_2.shape}, masks_weight_2: {masks_weight_2.shape}")

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
        with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16, enabled=args.use_fp16):
            # ========== DINO Loss with Sequence Packing ==========

            student_masks = [block_masks_1, block_masks_2] + [None] * len(student_local_crops)

            # Teacher forward (unmasked targets)
            with torch.no_grad():
                teacher_output = teacher(teacher_global_crops, token_masks=[None, None], mode='dino')
                teacher_cls_outputs = teacher_output['cls_outputs']
                teacher_patch_tokens_g1 = teacher_output['features_list'][0]['patchtokens']
                teacher_patch_tokens_g2 = teacher_output['features_list'][1]['patchtokens']

            # Student forward (all crops, masks on global crops)
            student_output = student(student_all_crops, token_masks=student_masks, mode='dino')
            student_cls_outputs = student_output['cls_outputs']
            student_patch_tokens_g1 = student_output['features_list'][0]['patchtokens']
            student_patch_tokens_g2 = student_output['features_list'][1]['patchtokens']

            dino_class_loss_val = dino_class_loss(
                student_cls_outputs,
                teacher_cls_outputs,
                current_iteration,
            )

            # ========== KoLeo Loss ==========
            num_global_total = args.global_views
            global_features_list = student_output['features_list'][:num_global_total]
            global_cls_tokens = [feat_dict['clstoken'] for feat_dict in global_features_list]

            koleo_loss_val = torch.tensor(0.0, device=device)
            if len(global_cls_tokens) > 0:
                # Canonical DINOv2 / Virchow2: SUM the regularizer over the global
                # crops (do NOT average). Applies to both KoLeo and KDE.
                koleo_loss_val = sum(dino_koleo_loss(token) for token in global_cls_tokens)

            # ================================================================
            # iBOT Loss — gather-then-project to avoid [B, N, 65536] tensors.
            # Projects only masked tokens (~7k) instead of all tokens (~50k).
            # ================================================================
            current_teacher_temp_ibot = dino_class_loss.teacher_temp_schedule(current_iteration)

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
                ibot_loss_g1 = torch.tensor(0.0, device=device)

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
                ibot_loss_g2 = torch.tensor(0.0, device=device)

            ibot_loss_val = (ibot_loss_g1 + ibot_loss_g2) / 2.0

        # ================================================================
        # Patch Prototype Clustering
        # Operates on backbone-dim [B, N, 768] — no memory concern.
        # ================================================================
        if args.use_prototype_clustering:
            current_teacher_temp = dino_class_loss.teacher_temp_schedule(current_iteration)

            with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16, enabled=args.use_fp16):
                # ---------- Block mask prototype: Global crop 1 ----------
                clust_loss_g1, proto_loss_g1, koleo_proto_g1, Q_g1 = patch_prototype_loss(
                    teacher_patch_tokens_g1,
                    student_patch_tokens_g1,
                    block_masks_1,
                    prototype_bank,
                    current_iteration,
                    current_teacher_temp,
                    masks_weight=masks_weight_1
                )

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

            # The bank sees each teacher crop's arrangement once, averaged over g1/g2,
            # plus koleo once averaged over g1/g2.
            prototype_loss = teacher_proto_loss + koleo_proto_loss
        else:
            clustering_loss = torch.tensor(0.0, device=device)
            teacher_proto_loss = torch.tensor(0.0, device=device)
            koleo_proto_loss = torch.tensor(0.0, device=device)
            prototype_loss = torch.tensor(0.0, device=device)

        # ========== Compute Total Losses ==========
        student_loss = (
            dino_class_loss_val +
            args.koleo_loss_weight * koleo_loss_val +
            args.ibot_loss_weight * ibot_loss_val +
            args.clustering_weight * clustering_loss
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
        if current_iteration % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                    'total_student_views': total_student_views,
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
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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