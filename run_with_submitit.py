"""
Submitit launcher for DINOv2 training on SLURM clusters.
"""

import argparse
import os
import uuid
import datetime
from pathlib import Path

import submitit

from configs import get_args_parser
from training import train_dinov2
from training.helpers import calculate_total_student_views


def parse_args():
    """Parse submitit and training arguments."""
    parser = argparse.ArgumentParser(
        "Submitit for DINOv2",
        parents=[get_args_parser()]
    )

    # Submitit specific arguments
    parser.add_argument("--ngpus", default=4, type=int,
                        help="Number of GPUs per node")
    parser.add_argument("--nodes", default=1, type=int,
                        help="Number of nodes")
    parser.add_argument("--timeout", default=10000, type=int,
                        help="Job duration in minutes")
    parser.add_argument("--partition", default="vanderbc_gpu", type=str,
                        help="Partition name")

    return parser.parse_args()


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints relative to script location."""
    script_dir = Path(__file__).parent.resolve()
    p = script_dir / "logs"
    p.mkdir(exist_ok=True)
    return p


def get_init_file():
    """Create unique init file for distributed training."""
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    """Wrapper class for submitit job."""
    def __init__(self, args):
        self.args = args

    def __call__(self):
        """Main training call."""
        self._setup_gpu_args()
        train_dinov2(self.args)

    def checkpoint(self):
        """Checkpoint for job preemption."""
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        """Setup GPU arguments from submitit environment."""
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


# PROBABILISTIC ECT - MAGNIFICATION TABLE
#
# Formula: apparent_mag = source_mag * output_side / (sqrt(scale) * source_side)
#
# 40x tiles (source >= 448, native 40x at MPP 0.25):
#   ECT branch (p=0.4) - preserves cellular morphology:
#     Global (224 out): scale=(0.203, 0.303), ratio=(0.95, 1.05)
#     Local  (96  out): scale=(0.037, 0.056), ratio=(0.95, 1.05)
#     Apparent mag: globals 36.3-44.4x, locals 36.2-44.5x
#   Standard branch (p=0.6):
#     Global (224 out): scale=(0.32, 1.0),    ratio=(0.75, 1.33)
#     Local  (96  out): scale=(0.05, 0.32),   ratio=(0.75, 1.33)
#     Apparent mag: globals 20.0-35.4x, locals 15.2-38.3x
#
# 20x tiles (source == 224, native 20x at MPP 0.50):
#   Standard branch always:
#     Global (224 out): scale=(0.32, 1.0),    ratio=(0.75, 1.33)
#     Local  (96  out): scale=(0.05, 0.32),   ratio=(0.75, 1.33)
#     Apparent mag: globals 20.0-35.4x, locals 15.2-38.3x
#
# Design notes:
#  - ECT branch (Virchow2 recipe) lives at native 40x +/- 10%. Model sees
#    cells at correct physical scale; no aggressive resize.
#  - Standard branch spans 20x-35x globally, 15x-38x locally on both tile
#    types. This includes the downstream evaluation magnification (20x).
#  - Small gap at 35-36x where neither branch covers densely. Acceptable
#    trade-off: widening standard would defeat ECT's morphology guarantee.


def main():
    """Main submitit launcher."""
    args = parse_args()

    # Set output directory
    args.output_dir = str(get_shared_folder())

    # Setup executor
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    # Job name with timestamp
    job_name = f"dinov2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Slurm parameters
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=256,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{args.ngpus}',
        slurm_constraint='h100',
        slurm_setup=[
            'ulimit -l unlimited',
            f'export OMP_NUM_THREADS=8',
            # PyTorch CUDA allocator: grow existing segments instead of carving fresh
            # fixed-size ones. Collapses fragmentation from multi-crop forwards, xformers
            # packed variable-length sequences, and iBOT [M, D] gathers where M varies
            # per batch — the three biggest fragmentation sources in this training loop.
            f'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
            f'export NCCL_DEBUG=INFO',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={num_gpus_per_node * nodes}',
        ]
    )

    executor.update_parameters(name=job_name)

    args.dist_url = get_init_file().as_uri()

    # ========== Default training configuration ==========
    # Architecture - pick the variant. Each row of _VIT_CONFIGS fixes
    # (embeddingdim, vitdepth, vitheads) to the standard DINOv2 sizes:
    #   S:  embed=384,  depth=12, heads=6    (tiny)
    #   B:  embed=768,  depth=12, heads=12   (base)
    #   L:  embed=1024, depth=24, heads=16   (large)
    #   H:  embed=1280, depth=32, heads=16   (huge)  -- triggers pathology-recipe auto-gate
    #   G:  embed=1536, depth=40, heads=24   (giant) -- triggers pathology-recipe auto-gate
    # patch_size is orthogonal to the variant; the pathology recipe overrides it to 14 at runtime.
    args.vit_variant = "B"
    _VIT_CONFIGS = {
        "S": dict(embeddingdim=384,  vitdepth=12, vitheads=6),
        "B": dict(embeddingdim=768,  vitdepth=12, vitheads=12),
        "L": dict(embeddingdim=1024, vitdepth=24, vitheads=16),
        "H": dict(embeddingdim=1280, vitdepth=32, vitheads=16),
        "G": dict(embeddingdim=1536, vitdepth=40, vitheads=24),
    }
    for k, v in _VIT_CONFIGS[args.vit_variant].items():
        setattr(args, k, v)
    args.patch_size = 16

    # ---- Architecture corrections (see LayerScale/qk-norm investigation) ----
    args.layerscale_init = 1e-5     # uniform; NOT the CaiT depth-schedule (which broke ViT-L)
    args.norm_last_layer = False    # DINOv2 behavior; True = frozen prototype magnitude (old bug)
    args.qk_norm         = True     # caps attention-logit blow-up; False = block-10 sink at depth
    args.drop_path_rate    = 0.1    # DINOv2 ssl_default / UNI supplementary (was 0.4)
    args.drop_path_uniform = True   # DINOv2 ssl_default: flat rate across depth (was False/CaiT ramp)
    args.ffn_type          = "mlp"  # DINOv2 ssl_default uses standard MLP+GELU (was SwiGLU)

    # ========== Augmentation Configuration ==========
    args.global_views = 2
    args.n_standard_local_crops = 8
    args.local_crop_size = 96

    # Semantic iBOT
    args.use_semantic_ibot = False
    args.use_semantic_prototypes = False
    args.semantic_ibot_weight = 1.0
    args.semantic_clustering_weight = 1.0
    args.semantic_masks_per_iteration = 1

    # Mask model (used by semantic iBOT)
    args.mask_checkpoint = "/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth"
    args.num_masks = 3
    args.mask_model_arch = 'vit_unet'
    args.mask_encoder_dim = 192

    # DINO parameters
    args.out_dim = 65536
    # args.norm_last_layer is set in the "Architecture corrections" block
    # near the variant/patch_size definitions above.
    args.use_bn_in_head = False

    # DINOv2 parameters
    args.koleo_loss_weight = 0.1
    args.ibot_loss_weight = 1.0

    # Prototype clustering
    args.use_prototype_clustering = False
    args.num_prototypes = 16384
    args.clustering_weight = 1.0
    args.clustering_teacher_temp = 0.07
    args.clustering_student_temp = 0.1

    # ============ Typicality dampening (counted-coverage bank; fixed-radius readout) ============
    # A running bank estimates each tile's local density p_hat; the stream is then rebalanced either by
    # WEIGHTING the loss (rev7) or by THINNING the batch (rev8). EVERY knob is surfaced here, grouped, so
    # this file is the single source of truth. Values are the weighted/baseline defaults (weighted arms
    # stay byte-identical); the launcher overrides per arm and scales the bank 4x for thinned arms.
    # Weight/acceptance:  w = 1/(p_hat + c)^a,  c = c_frac * p_ref,  readout radius R_rad = radius_mult * s.
    args.use_typicality_dampening = False   # master switch; launcher flips ON for every bc_* arm.

    # ---- bank capacity & counters (structural dials) ----
    args.typicality_bank_size         = 8192   # M: established signatures the bank holds. THINNED arms scale
                                               #    this 4x in the launcher -- the scout feeds the bank the whole
                                               #    ~4x over-draw pool each step, so it needs ~4x the slots to
                                               #    keep turnover / lam_spread in the weighted-healthy range.
    args.typicality_reserve_size      = 550    # probation buffer on top of M (thinned arms scale 4x too).
    args.typicality_K_prime           = 256    # signature dimensionality K'.
    args.typicality_halflife_steps    = 250    # counter decay half-life H (eta = 0.5**(1/H)); fixes n_eff.
    args.typicality_reserve_residency = 300    # T_need: steps a reserve entry may sit before it expires.
    args.typicality_graduation_hits   = 2      # hits a reserve entry needs before graduating into the bank.
    args.typicality_pool_j            = 64     # INITIAL kernel-sum neighbor count j (self-tuned online).

    # ---- fixed-radius readout / weight ----
    args.typicality_modulation   = 'weighted_loss'  # only modulation wired to the counted bank.
    args.typicality_a            = 1.0    # tilt exponent; rarer tiles up-weighted more as a grows (lo=0.5, hi=1.0).
    args.typicality_c_frac       = 0.25   # weight floor c = c_frac * p_ref (keeps w finite at p_hat=0).
    args.typicality_radius_mult  = 1.5    # readout radius R_rad = radius_mult * s.

    # ---- rebalancing MODE: weighted loss (rev7) vs stream thinning / scouted (rev8) ----
    #   'weighted' -> scale the DINO loss per tile by w (rev7, byte-unchanged).
    #   'thinned'  -> UNWEIGHTED loss; admit tiles by density a(p_hat)=w/w_max over an over-drawn scout pool.
    #   'off'      -> no rebalancing. Launcher sets per arm (bc_weightedloss_*->weighted; bc_thinned_*->thinned).
    args.balance_mode            = 'weighted'
    args.thin_oversample_factor  = 3.0    # thinned only: candidate pool = ceil(factor)*N; MUST exceed the
                                          #    measured chi or the step under-fills (launcher: 6 lo / 12 hi).
    args.thin_richardson_correct = False  # thinned only: debias p_hat via u*=2u_s-u_2s (off; probe is measure-only).
    args.use_gpu_augmentation    = False  # opt-in for baselines: thinned raw-tile loader + GPU augmentation, no thinning.

    # ---- activation / representation head ----
    args.typicality_warmup_iters = 50000  # bank inert until here (late-fill; bc arms warm-start from a 50k ckpt).
    args.typicality_repr_lr      = 1e-3   # LR for the projection head that maps features -> signatures.

    # ---- self-tuning hit radius s & pool count j (advanced; self-tuned online, defaults are fine) ----
    args.typicality_s_buffer_size    = 60000      # rolling signature ring-buffer feeding the s edge-sweep.
    args.typicality_s_sweep_interval = 500        # steps between s edge-sweeps.
    args.typicality_s_grid_points    = 9          # radius grid points per sweep (before one bisection).
    args.typicality_s_grid_span      = [0.3, 2.0] # sweep span as (lo, hi) multiples of the live s.
    args.typicality_s_ema_alpha      = 0.2        # EMA weight for s <- (1-a)*s + a*edge.
    args.typicality_s_min_buffer     = 60000      # signatures buffered before the first sweep fixes s.
    args.typicality_s_headroom       = 0.0        # optional fraction to sit under the swept edge.
    args.typicality_pool_selftune    = True       # self-tune j from the variogram L estimate (False pins j).
    args.typicality_pool_rse_target  = 0.05       # target relative SE on log p_hat that sets the j floor.
    args.typicality_pool_max         = 256        # upper clamp on the self-tuned j.
    args.typicality_pool_ema         = 0.2        # EMA weight for j across sweeps.

    # Adversarial-mask-as-student-view augmentation (re-uses mask_checkpoint above)
    args.use_adversarial_mask_augmentation = False
    args.crops_per_mask = 0

    # CellViT (nuclei / background) augmentation
    args.use_cellvit_augmentation = False
    args.cellvit_checkpoint = "/data1/vanderbc/nandas1/CellViT_models/TCGA_Dinov2_ViT-B_run2/model.pth"
    args.cellvit_crops_per_channel = 0

    # Random rectangular mask augmentation
    args.use_random_mask_augmentation = False
    args.random_num_masks = 2
    args.random_crops_per_mask = 0

    # Teacher parameters
    args.momentum_teacher = 0.992
    args.teacher_temp = 0.07
    args.warmup_teacher_temp = 0.04
    args.teacher_temp_warmup_iters = 37_500

    # Optimization
    args.batch_size_per_gpu = 256
    args.warmup_iterations = 12_500
    args.total_iterations = 125_001
    args.freeze_last_layer_iters = 1_250
    args.lr = 2e-3      # base_lr under sqrt_wrt_1024 rule -> ~3.46e-3 applied at bs 3072 (384x8)
    args.min_lr = 1e-6
    args.weight_decay = 0.04
    args.weight_decay_end = 0.4
    args.lr_decay_rate = 1.0    # DINOv2 ssl_default layerwise_decay=1.0 for ViT-L (was 0.9)

    # Training setup
    args.use_fp16 = True
    args.clip_grad = 3.0
    args.save_checkpoint_freq = 2_000
    args.num_workers = 10
    args.visualization_freq = 10000
    args.grad_checkpointing = True

    # Dataset
    args.dataset_sources = [
        "TCGA:/data1/vanderbc/foundation_model_training_images/TCGA:TCGA_dataset_index.pkl",
        "CPTAC:/data1/vanderbc/foundation_model_training_images/CPTAC:CPTAC_dataset_index.pkl",
        "IMPACT:/data1/vanderbc/foundation_model_training_images/IMPACT:IMPACT_dataset_index.pkl"
    ]

    # ================================================================
    # PATHOLOGY FM RECIPE - toggle
    # ================================================================
    # Flip use_pathology_recipe to True to enable the Virchow2-derived
    # bundle. ect_probability and kde_kappa are only consulted when the
    # recipe is on; leaving them at their defaults here is harmless.
    #
    # Recipe includes:
    #   - KDE regularizer replaces KoLeo        [Virchow2 Sec 5.2]
    #   - Probabilistic ECT augmentation         [Virchow2 Sec 5.1 + user variation]
    #   - Teacher temp fixed at 0.04             [Virchow2G Sec 5.1]
    #   - patch_size=14                          [pathology FM community standard]
    #   - bf16 end-to-end                        [Virchow2G retrospective]
    #   - Solarization off, V-flip, 90-deg rot   [Virchow2/RudolfV/Hibou convergence]
    #
    # When embeddingdim >= 1280 (ViT-H/G), the auto-gate additionally enables:
    #   - qk_norm=True                           [Virchow2G Sec 6]
    #   - num_register_tokens >= 8               [Virchow2G + UNI2-h]
    #   - out_dim=131,072                        [Virchow v1 Methods, Paige standard]
    #   - StableAdamW with beta2=0.95            [Virchow2G Sec 6]
    #
    # Full probabilistic ECT magnification table is documented in a
    # module-level comment block at the top of this file.
    # ================================================================
    args.use_pathology_recipe = False
    args.ect_probability = 0.4
    args.kde_kappa = 5.0

    # Patch-embed LR throttle (DINOv2 ssl_default_config: 0.2; MoCo v3 stability).
    # Applied to the patch_embed param group only, on top of layer-wise decay.
    args.patch_embed_lr_mult = 0.2

    # Save configuration
    with open(os.path.join(args.output_dir, f"{job_name}_config.txt"), "w") as f:
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    # Create and submit trainer
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Job name: {job_name}")
    print(f"Logs and checkpoints: {args.output_dir}")

    # Calculate total views
    total_views = calculate_total_student_views(args)

    # Reflect the pathology-recipe patch_size override in the summary so the
    # printed architecture matches what train_dinov2 will actually build.
    effective_patch_size = 14 if args.use_pathology_recipe else args.patch_size

    print("\n" + "="*80)
    print("Configuration Summary:")
    print(f"  Architecture: ViT-{args.vit_variant}/{effective_patch_size}")
    print(f"    layerscale_init = {args.layerscale_init}  "
          f"(schedule = {getattr(args, 'layerscale_schedule', 'uniform')})")
    print(f"    norm_last_layer = {args.norm_last_layer}")
    print(f"    qk_norm         = {args.qk_norm}")
    print(f"  Global crops: {args.global_views}")
    print(f"  Standard local crops: {args.n_standard_local_crops}")

    if args.use_semantic_ibot:
        print(f"  Semantic iBOT: ENABLED")
        print(f"    Mask model: {args.mask_model_arch}")
        print(f"    Semantic channels: {args.num_masks}")
        print(f"    Channels per iteration: {args.semantic_masks_per_iteration}")
        print(f"    Semantic iBOT weight: {args.semantic_ibot_weight}")
        if args.use_semantic_prototypes:
            print(f"    Semantic prototype loss: ENABLED (weight={args.semantic_clustering_weight})")

    if args.use_typicality_dampening:
        print(f"  Typicality Dampening: ENABLED")
        print(f"    K' = {args.typicality_K_prime}, Bank M = {args.typicality_bank_size}")
        print(f"    Modulation: {args.typicality_modulation}")
        print(f"    warmup = {args.typicality_warmup_iters}")

    if args.use_adversarial_mask_augmentation:
        print(f"  Adversarial Mask Augmentation: ENABLED")
        print(f"    num_masks = {args.num_masks}, crops_per_mask = {args.crops_per_mask}")

    if args.use_cellvit_augmentation:
        print(f"  CellViT Augmentation: ENABLED")
        print(f"    cellvit_crops_per_channel = {args.cellvit_crops_per_channel}")

    if args.use_random_mask_augmentation:
        print(f"  Random Mask Augmentation: ENABLED")
        print(f"    random_num_masks = {args.random_num_masks}, random_crops_per_mask = {args.random_crops_per_mask}")

    print(f"  Total student views (DINO CLS): {total_views}")
    print(f"  Batch size per GPU: {args.batch_size_per_gpu}")
    print(f"  Total GPUs: {args.ngpus * args.nodes}")
    print(f"  Effective batch size: {args.batch_size_per_gpu * args.ngpus * args.nodes}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
