#!/bin/bash
# scripts/run_forcing_suite.sh <ARM> -- set up one arm of the ViT-S forcing study.
#
# Mechanics follow run_sub_stab_suite_rev13.sh of the source fork: one experiment directory
# per arm under BASE_DIR, a fresh clone of this repository into it, CLONED_COMMIT.txt from
# `git log -1`, an ensure_arg sed helper that sets `args.<key> = <value>` lines in
# run_with_submitit.py, the log-path patch, the SLURM constraint patch, and a printed launch
# command. This script never submits; run the printed command by hand (or scripts/launch_all.sh).
# It refuses to touch an existing experiment directory (exit 1): remove it by hand first.
#
# Every arm: ViT-S, batch 512 on one GPU, 1_000_001 iterations, periodic checkpoints every
# 50_000 iterations (checkpoint_iter_*.pth) and a rolling checkpoint.pth every 5_000.
# SMOKE arms: 501 iterations, both checkpoint periods 250.
#
# Overridable from the environment (used by the local acceptance test):
#   BASE_DIR       experiment root            (default /data1/vanderbc/test_dinov2_swaraj)
#   FORCING_REPO   git clone source           (default the public GitHub repository)
#   PARTITION      SLURM partition            (default vanderbc_gpu)
set -e

RUN="${1:-}"
GITHUB_REPO="${FORCING_REPO:-https://github.com/swarajnanda2021/dinov2_forcing.git}"
BRANCH="main"
BASE_DIR="${BASE_DIR:-/data1/vanderbc/test_dinov2_swaraj}"
PARTITION="${PARTITION:-vanderbc_gpu}"
SLURM_CONSTRAINT="h100"          # the one-line constraint patch; run_with_submitit.py ships with 'h100'

ARMS="R0 R1 R2 R3 R4 R5 R6 R7 R1_PROTO R0_PROTO SMOKE SMOKE_PROTO"
case " $ARMS " in
  *" $RUN "*) ;;
  *) echo "Usage: $0 <ARM>"; echo "  ARM: $ARMS"; exit 1 ;;
esac

exp_name="forcing_ViT-S_${RUN}"
exp_dir="$BASE_DIR/$exp_name"

echo "========================================"
echo "Setup: $exp_name  (branch: $BRANCH)"
echo "  ViT-S/16 forcing study, one knob per arm: $RUN"
echo "========================================"

[ -d "$exp_dir" ] && { echo "exists: $exp_dir"; exit 1; }
mkdir -p "$exp_dir"; cd "$exp_dir"

echo "  Cloning ($BRANCH from $GITHUB_REPO)..."
git clone -q -b "$BRANCH" "$GITHUB_REPO" .

echo "  Recording clone commit..."
git log -1 --oneline | tee "$exp_dir/CLONED_COMMIT.txt"

echo "  Verifying the forcing-study code is present..."
for a in lr_schedule wd_schedule momentum_schedule momentum_teacher_end diag_every diag_probe_manifest diag_probe_size vit_variant; do
    if ! grep -q -- "--$a" configs/config.py; then
        echo "  FATAL: configs/config.py has no --$a -- stale clone. Aborting."; exit 1
    fi
done
[ -f diagnostics/patch_geometry.py ] || { echo "  FATAL: diagnostics/patch_geometry.py missing. Aborting."; exit 1; }
grep -q "run_diagnostics" training/trainer.py || { echo "  FATAL: trainer does not call run_diagnostics. Aborting."; exit 1; }
echo "    verified."

# portable in-place sed (GNU on the cluster, BSD on a macOS workstation)
sedi () { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }

echo "  Patching log path + GPU constraint..."
sedi "s|p = script_dir / \"logs\"|p = Path(\"$exp_dir/logs\")|" run_with_submitit.py
sedi "s@slurm_constraint='h100'@slurm_constraint='$SLURM_CONSTRAINT'@" run_with_submitit.py

ensure_arg () {
    local key="args.$1"; local esc="args\\.$1"
    if grep -qE "^[[:space:]]*${esc}[[:space:]]*=" run_with_submitit.py; then
        sedi -E "s|^([[:space:]]*)${esc}[[:space:]]*=.*|\\1${key} = $2|" run_with_submitit.py
    else
        # append a new line after the patch_embed_lr_mult anchor (awk: same on GNU and BSD)
        awk -v line="    ${key} = $2" '{ print } /^[[:space:]]*args\.patch_embed_lr_mult[[:space:]]*=/ { print line }' \
            run_with_submitit.py > run_with_submitit.py.tmp && mv run_with_submitit.py.tmp run_with_submitit.py
    fi
}

echo "  Common settings (every arm)..."
ensure_arg vit_variant              '"S"'
ensure_arg batch_size_per_gpu       512
ensure_arg num_workers              10
ensure_arg total_iterations         1_000_001
ensure_arg warmup_iterations        10_000
ensure_arg save_checkpoint_freq     50_000      # periodic checkpoint_iter_*.pth
ensure_arg rolling_checkpoint_freq  5_000       # rolling checkpoint.pth
ensure_arg momentum_teacher         0.992
ensure_arg momentum_teacher_end     1.0
ensure_arg weight_decay             0.04
ensure_arg weight_decay_end         0.4
ensure_arg min_lr                   1e-6
ensure_arg lr                       2e-4        # base at global batch 1024; the trainer scales by sqrt(512/1024) -> 1.41e-4 on one GPU
ensure_arg drop_path_rate           0.1         # source recipe launcher value (with drop_path_uniform=True)
ensure_arg n_standard_local_crops   8
ensure_arg local_crop_size          96
ensure_arg ibot_loss_weight         1.0
ensure_arg mask_ratio_min           0.1
ensure_arg mask_ratio_max           0.5
ensure_arg mask_sample_probability  0.5
ensure_arg koleo_loss_weight        0.1
ensure_arg use_prototype_clustering False
ensure_arg diag_every               2000
ensure_arg diag_probe_manifest      "\"$BASE_DIR/probe_manifest.json\""
ensure_arg diag_probe_size          1024
ensure_arg seed                     0
# recipe control shapes (R0); R1-based arms override below
ensure_arg lr_schedule              '"cosine"'
ensure_arg wd_schedule              '"cosine"'
ensure_arg momentum_schedule        '"cosine"'

r1_base () {
    ensure_arg lr_schedule          '"constant"'
    ensure_arg wd_schedule          '"constant"'
    ensure_arg weight_decay         0.4
    ensure_arg momentum_schedule    '"constant"'
}

echo "  Arm: $RUN"
case "$RUN" in
  R0) ;;                                   # recipe control: cosine lr / wd / momentum
  R1) r1_base ;;
  R2) r1_base; ensure_arg weight_decay 1.0 ;;
  R3) r1_base; ensure_arg weight_decay 0.1 ;;
  R4) r1_base; ensure_arg lr 1e-4 ;;                                    # 0.5 x the R1 base lr
  R5) r1_base; ensure_arg ibot_loss_weight 0.5 ;;
  R6) r1_base; ensure_arg n_standard_local_crops 16; ensure_arg local_crop_size 64 ;;
  R7) r1_base; ensure_arg mask_ratio_min 0.5; ensure_arg mask_ratio_max 0.75 ;;
  R1_PROTO) r1_base; ensure_arg use_prototype_clustering True; ensure_arg num_prototypes 4096; ensure_arg clustering_weight 1.0 ;;
  R0_PROTO)          ensure_arg use_prototype_clustering True; ensure_arg num_prototypes 4096; ensure_arg clustering_weight 1.0 ;;
  SMOKE)       r1_base; ensure_arg total_iterations 501; ensure_arg warmup_iterations 100; ensure_arg diag_every 100
               ensure_arg save_checkpoint_freq 250; ensure_arg rolling_checkpoint_freq 250 ;;
  SMOKE_PROTO) r1_base; ensure_arg total_iterations 501; ensure_arg warmup_iterations 100; ensure_arg diag_every 100
               ensure_arg save_checkpoint_freq 250; ensure_arg rolling_checkpoint_freq 250
               ensure_arg use_prototype_clustering True ;;
esac

mkdir -p "$exp_dir/logs"

echo ""
echo "  Resolved settings ('MISSING' => fix before launch):"
for kv in \
    args.vit_variant args.batch_size_per_gpu args.num_workers args.total_iterations args.warmup_iterations \
    args.save_checkpoint_freq args.rolling_checkpoint_freq args.momentum_teacher args.momentum_teacher_end args.weight_decay args.weight_decay_end \
    args.min_lr args.lr args.drop_path_rate args.n_standard_local_crops args.local_crop_size \
    args.ibot_loss_weight args.mask_ratio_min args.mask_ratio_max args.mask_sample_probability \
    args.koleo_loss_weight args.use_prototype_clustering args.num_prototypes args.clustering_weight \
    args.lr_schedule args.wd_schedule args.momentum_schedule \
    args.diag_every args.diag_probe_manifest args.diag_probe_size args.seed ; do
        hit=$(grep -nE "^[[:space:]]*${kv//./\\.}[[:space:]]*=" run_with_submitit.py | head -1)
        printf "    %-34s %s\n" "$kv" "${hit:-MISSING}"
done
echo "    GPU constraint -> $(grep -n "slurm_constraint" run_with_submitit.py | head -1)"
echo "    Log path       -> $(grep -n 'p = Path(' run_with_submitit.py | head -1)"
echo "    Clone commit   -> $(cat "$exp_dir/CLONED_COMMIT.txt")"

LAUNCH_CMD="cd $exp_dir && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_with_submitit.py --nodes 1 --ngpus 1 --partition $PARTITION"
echo ""
echo "  NOT submitting. Launch manually with:"
echo "    $LAUNCH_CMD"
echo "$LAUNCH_CMD" > "$exp_dir/LAUNCH_CMD.txt"
echo ""
echo "  Read the run with:"
echo "    grep -h '^\\[diag\\]' $exp_dir/logs/*.out | grep 'branch=teacher'"
echo "    grep -h '^\\[sched\\]' $exp_dir/logs/*.out"
