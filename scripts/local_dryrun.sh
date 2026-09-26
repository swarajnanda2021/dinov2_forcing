#!/bin/bash
# scripts/local_dryrun.sh -- CPU (or single local GPU) dry run of the forcing-study trainer.
#
# Builds a synthetic dataset in the exact format data/datasets.py reads (zip files of .webp
# tiles + an index pickle of (zip_path, [member names]) pairs), then runs main_train.py for
# 6 iterations on R0, R1 and R1_PROTO settings, then kills-and-resumes R1 from the
# iteration-3 checkpoint. Exits non-zero if any expectation fails.
#
# Usage: bash scripts/local_dryrun.sh [WORK_DIR]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-${TMPDIR:-/tmp}/dinov2_forcing_dryrun}"
PY="${PYTHON:-python}"
ARMS="${DRYRUN_ARMS:-R0 R1 R1_PROTO}"    # e.g. DRYRUN_ARMS="R1 R1_PROTO"
N_IMAGES=64
TILE=224
# bf16 autocast on CPU is ~20x slower than fp32 for the backward pass (measured on an M1: 57 s vs
# 1.8 s per step), so the CPU dry run uses fp32. A local CUDA GPU keeps the training bf16 path.
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then USE_FP16=True; else USE_FP16=False; fi

rm -rf "$WORK"; mkdir -p "$WORK/data" "$WORK/runs"
cd "$REPO"

echo "== building synthetic dataset in $WORK/data =="
"$PY" - "$WORK/data" "$N_IMAGES" "$TILE" <<'PYEOF'
import io, os, pickle, sys, zipfile
import numpy as np
from PIL import Image
base, n, tile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
zips, per_zip = [], 16
for zi in range(n // per_zip):
    zpath = os.path.join(base, f"tiles_{zi:03d}.zip")
    names = []
    with zipfile.ZipFile(zpath, 'w') as zf:
        for k in range(per_zip):
            arr = rng.integers(0, 256, size=(tile, tile, 3), dtype=np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format='WEBP', quality=90)
            name = f"tile_{zi:03d}_{k:03d}.webp"
            zf.writestr(name, buf.getvalue()); names.append(name)
    zips.append((zpath, names))
with open(os.path.join(base, "synthetic_dataset_index.pkl"), 'wb') as f:
    pickle.dump(zips, f)
print(f"wrote {len(zips)} zips, {sum(len(n) for _, n in zips)} images")
PYEOF

SRC="SYN:$WORK/data:synthetic_dataset_index.pkl"
COMMON=(--dataset_sources "$SRC" --vit_variant S --batch_size_per_gpu 4 --n_standard_local_crops 2
        --local_crop_size 96 --warmup_iterations 2 --total_iterations 7 --diag_every 2
        --diag_probe_size 16 --diag_probe_manifest "$WORK/probe_manifest.json" --save_checkpoint_freq 6 --rolling_checkpoint_freq 3
        --num_workers 2 --seed 0 --min_lr 1e-6 --lr 2e-4 --momentum_teacher 0.992
        --ibot_loss_weight 1.0 --mask_ratio_min 0.1 --mask_ratio_max 0.5 --mask_sample_probability 0.5
        --koleo_loss_weight 0.1 --drop_path_rate 0.1 --drop_path_uniform True --ffn_type mlp
        --layerscale_init 1e-5 --norm_last_layer False --qk_norm True --lr_decay_rate 1.0
        --freeze_last_layer_iters 1 --teacher_temp_warmup_iters 4 --grad_checkpointing False
        --use_fp16 "$USE_FP16")
R0=(--lr_schedule cosine --wd_schedule cosine --momentum_schedule cosine --weight_decay 0.04 --weight_decay_end 0.4 --use_prototype_clustering False)
R1=(--lr_schedule constant --wd_schedule constant --momentum_schedule constant --weight_decay 0.4 --weight_decay_end 0.4 --use_prototype_clustering False)
R1_PROTO=("${R1[@]:0:10}" --use_prototype_clustering True --num_prototypes 16384 --clustering_weight 1.0)

run_arm () {
    local name="$1"; shift
    local out="$WORK/runs/$name"; mkdir -p "$out"
    echo; echo "== arm $name -> $out =="
    (cd "$out" && "$PY" "$REPO/main_train.py" "${COMMON[@]}" "$@" --output_dir "$out" > "$out/train.log" 2>&1) \
        || { echo "FAILED: see $out/train.log"; tail -50 "$out/train.log"; exit 1; }
    grep -h '^\[sched-config\]' "$out/train.log"
    grep -h '^\[sched\]' "$out/train.log"
    grep -h '^\[diag\]' "$out/train.log"
    grep -h '^\[diag-time\]' "$out/train.log"
    grep -h '^\[ckpt\]' "$out/train.log"
    grep -h '^It ' "$out/train.log" | tail -1
}

expect () {  # expect <n> <pattern> <file> <label>
    local got; got=$(grep -c -- "$2" "$3" || true)
    if [ "$got" -ne "$1" ]; then echo "EXPECTATION FAILED: $4: found $got, expected $1 ($2 in $3)"; exit 1; fi
    echo "  ok: $4 ($got)"
}

FIRST_ARM=""
for a in $ARMS; do
    [ -z "$FIRST_ARM" ] && FIRST_ARM="$a"
    case "$a" in
      R0) run_arm R0 "${R0[@]}" ;;
      R1) run_arm R1 "${R1[@]}" ;;
      R1_PROTO) run_arm R1_PROTO "${R1_PROTO[@]}" ;;
      *) echo "unknown arm $a"; exit 1 ;;
    esac
done

echo; echo "== checks =="
for a in $ARMS; do
    L="$WORK/runs/$a/train.log"
    expect 3 '^\[sched-config\]' "$L" "$a sched-config lines"
    expect 4 '^\[sched\]' "$L" "$a sched lines (it 0,2,4,6)"
    expect 4 '^\[diag\] it=[0246] branch=student' "$L" "$a student diag lines"
    expect 4 '^\[diag\] it=[0246] branch=teacher' "$L" "$a teacher diag lines"
    expect 8 '"branch"' "$WORK/runs/$a/diag.jsonl" "$a diag.jsonl records"
    expect 2 '^\[ckpt\] it=[36] rolling=1' "$WORK/runs/$a/train.log" "$a rolling checkpoint.pth at 3 and 6"
    expect 1 '^\[ckpt\] it=6 rolling=1 periodic=1' "$WORK/runs/$a/train.log" "$a periodic checkpoint at 6"
    test -f "$WORK/runs/$a/checkpoint.pth" || { echo "missing checkpoint.pth for $a"; exit 1; }
    test -f "$WORK/runs/$a/checkpoint_iter_00000006.pth" || { echo "missing iteration-6 checkpoint for $a"; exit 1; }
    test ! -f "$WORK/runs/$a/checkpoint_iter_00000003.pth" || { echo "unexpected periodic checkpoint at 3 for $a"; exit 1; }
done
# constant schedules on R1: lr after warmup == peak (2e-4 * sqrt(4/1024)), wd 0.4, m 0.992 at every [sched]
has_arm () { case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
if has_arm R1; then
    expect 3 '^\[sched\] it=[246] lr=1.25e-05 wd=0.4 m=0.992' "$WORK/runs/R1/train.log" "R1 constant lr/wd/m after warmup"
    expect 1 '^\[sched-config\] lr: constant peak=1.25e-05 min=1.25e-05 warmup=2' "$WORK/runs/R1/train.log" "R1 lr sched-config"
fi
if has_arm R0; then
    expect 1 '^\[sched-config\] lr: cosine peak=1.25e-05 min=1e-06 warmup=2' "$WORK/runs/R0/train.log" "R0 lr sched-config"
    expect 1 '^\[sched-config\] wd: cosine peak=0.04 min=0.4' "$WORK/runs/R0/train.log" "R0 wd sched-config"
fi
if has_arm R1_PROTO; then
    expect 1 'clustering_entropy' "$WORK/runs/R1_PROTO/train.log" "R1_PROTO clustering_entropy logged"
    if grep -q -i 'nan' <(grep '^It ' "$WORK/runs/R1_PROTO/train.log"); then echo "NaN in R1_PROTO loss line"; exit 1; fi
    echo "  ok: no NaN in R1_PROTO loss lines"
fi
expect 1 'wrote manifest' "$WORK/runs/$FIRST_ARM/train.log" "manifest written once ($FIRST_ARM)"
for a in $ARMS; do [ "$a" = "$FIRST_ARM" ] || expect 1 'loaded manifest' "$WORK/runs/$a/train.log" "manifest reused ($a)"; done

if ! has_arm R1; then echo; echo "DRY RUN PASSED  (work dir: $WORK; no R1 arm, resume test skipped)"; exit 0; fi
echo; echo "== kill-and-resume: R1 from the rolling checkpoint.pth written at iteration 3 =="
# A first pass stopped after iteration 3 (total_iterations 4) leaves only the rolling checkpoint.pth
# (rolling every 3, periodic every 6 -> no checkpoint_iter file); the second pass resumes from it.
OUT="$WORK/runs/R1_resume"; mkdir -p "$OUT"
cp "$WORK/probe_manifest.json" "$WORK/probe_manifest_before.json"
COMMON_KILL=("${COMMON[@]}"); for i in "${!COMMON_KILL[@]}"; do [ "${COMMON_KILL[$i]}" = "--total_iterations" ] && COMMON_KILL[$((i+1))]=4; done
(cd "$OUT" && "$PY" "$REPO/main_train.py" "${COMMON_KILL[@]}" "${R1[@]}" --output_dir "$OUT" > "$OUT/train_first.log" 2>&1) \
    || { echo "FIRST PASS FAILED"; tail -50 "$OUT/train_first.log"; exit 1; }
grep -h '^\[ckpt\]' "$OUT/train_first.log"
expect 1 '^\[ckpt\] it=3 rolling=1 periodic=0' "$OUT/train_first.log" "first pass: rolling checkpoint.pth at 3, no periodic"
test -f "$OUT/checkpoint.pth" || { echo "missing checkpoint.pth after first pass"; exit 1; }
test ! -f "$OUT/checkpoint_iter_00000003.pth" || { echo "unexpected periodic checkpoint at 3 after first pass"; exit 1; }
(cd "$OUT" && "$PY" "$REPO/main_train.py" "${COMMON[@]}" "${R1[@]}" --output_dir "$OUT" > "$OUT/train_resume.log" 2>&1) \
    || { echo "RESUME FAILED"; tail -50 "$OUT/train_resume.log"; exit 1; }
grep -h 'Resuming from iteration\|Starting training at iteration' "$OUT/train_resume.log"
grep -h '^\[sched-config\]\|^\[sched\]\|^\[diag\]\|^\[ckpt\]' "$OUT/train_resume.log"
expect 1 'Resuming from iteration 3' "$OUT/train_resume.log" "resume at iteration 3"
expect 2 '^\[sched\] it=[46] ' "$OUT/train_resume.log" "sched lines at 4 and 6 after resume"
expect 2 '^\[diag\] it=[46] branch=teacher' "$OUT/train_resume.log" "teacher diag at 4 and 6 after resume"
expect 1 '^\[ckpt\] it=6 rolling=1 periodic=1' "$OUT/train_resume.log" "resume: rolling + periodic checkpoint at 6"
diff <(grep -h '^\[sched-config\]' "$WORK/runs/R1/train.log") <(grep -h '^\[sched-config\]' "$OUT/train_resume.log") && echo "  ok: sched-config unchanged after resume (vs the uninterrupted R1 run)"
diff <(grep -h '^\[sched\] it=[46]' "$WORK/runs/R1/train.log") <(grep -h '^\[sched\] it=[46]' "$OUT/train_resume.log") && echo "  ok: [sched] values at 4 and 6 unchanged after resume (vs the uninterrupted R1 run)"
cmp "$WORK/probe_manifest_before.json" "$WORK/probe_manifest.json" && echo "  ok: probe manifest unchanged after resume"
expect 1 'loaded manifest' "$OUT/train_resume.log" "manifest reused on resume"
test -f "$OUT/checkpoint_iter_00000006.pth" && echo "  ok: periodic iteration-6 checkpoint written after resume"
echo; echo "DRY RUN PASSED  (work dir: $WORK)"
