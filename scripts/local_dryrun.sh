#!/bin/bash
# scripts/local_dryrun.sh -- CPU (or single local GPU) dry run of the forcing-study trainer.
#
# Builds a synthetic dataset in the exact format data/datasets.py reads (zip files of .webp
# tiles + an index pickle of (zip_path, [member names]) pairs), then runs main_train.py for
# 6 iterations on BASE, DEPTH36 and LSCALE settings (PROTO optional), then kills-and-resumes
# BASE from the rolling iteration-3 checkpoint. Exits non-zero if any expectation fails.
#
# Usage: bash scripts/local_dryrun.sh [WORK_DIR]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-${TMPDIR:-/tmp}/dinov2_forcing_dryrun}"
PY="${PYTHON:-python}"
ARMS="${DRYRUN_ARMS:-BASE DEPTH36 LSCALE}"    # e.g. DRYRUN_ARMS="DEPTH36 LSCALE"; PROTO = BASE + prototype clustering
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
VANILLA=(--lr_schedule cosine --wd_schedule cosine --momentum_schedule cosine --weight_decay 0.04 --weight_decay_end 0.4)
BASE=("${VANILLA[@]}" --use_prototype_clustering False)
DEPTH36=("${VANILLA[@]}" --use_prototype_clustering False --depth 36)
LSCALE=("${VANILLA[@]}" --use_prototype_clustering False --layerscale_init 1e-2)
PROTO=("${VANILLA[@]}" --use_prototype_clustering True --num_prototypes 4096 --clustering_weight 1.0)

run_arm () {
    local name="$1"; shift
    local out="$WORK/runs/$name"; mkdir -p "$out"
    echo; echo "== arm $name -> $out =="
    (cd "$out" && "$PY" "$REPO/main_train.py" "${COMMON[@]}" "$@" --output_dir "$out" > "$out/train.log" 2>&1) \
        || { echo "FAILED: see $out/train.log"; tail -50 "$out/train.log"; exit 1; }
    grep -h '^\[model\]' "$out/train.log"
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
      BASE) run_arm BASE "${BASE[@]}" ;;
      DEPTH36) run_arm DEPTH36 "${DEPTH36[@]}" ;;
      LSCALE) run_arm LSCALE "${LSCALE[@]}" ;;
      PROTO) run_arm PROTO "${PROTO[@]}" ;;
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
    expect 8 ' attn_entropy=[0-9.e+-]* attn_entropy_frac=[0-9.e+-]* cos_patch_tilemean=[0-9.e+-]* cos_cls_tilemean=[0-9.e+-]* cos_cls_embed=[0-9.e+-]*$' "$L" "$a diag lines end with the four new keys"
    expect 1 '^\[model\] variant=S ' "$L" "$a model line"
    expect 1 '^\[sched-config\] lr: cosine peak=1.25e-05 min=1e-06 warmup=2' "$L" "$a lr sched-config (cosine, peak 2e-4*sqrt(4/1024))"
    expect 1 '^\[sched-config\] wd: cosine peak=0.04 min=0.4' "$L" "$a wd sched-config"
    expect 1 '^\[sched-config\] momentum: cosine peak=0.992 min=1' "$L" "$a momentum sched-config"
    expect 2 '^\[ckpt\] it=[36] rolling=1' "$WORK/runs/$a/train.log" "$a rolling checkpoint.pth at 3 and 6"
    expect 1 '^\[ckpt\] it=6 rolling=1 periodic=1' "$WORK/runs/$a/train.log" "$a periodic checkpoint at 6"
    test -f "$WORK/runs/$a/checkpoint.pth" || { echo "missing checkpoint.pth for $a"; exit 1; }
    test -f "$WORK/runs/$a/checkpoint_iter_00000006.pth" || { echo "missing iteration-6 checkpoint for $a"; exit 1; }
    test ! -f "$WORK/runs/$a/checkpoint_iter_00000003.pth" || { echo "unexpected periodic checkpoint at 3 for $a"; exit 1; }
done
has_arm () { case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
has_arm BASE    && expect 1 '^\[model\] variant=S depth=12 embed=384 heads=6 layerscale_init=1e-05' "$WORK/runs/BASE/train.log" "BASE model line (depth 12, layerscale 1e-5)"
has_arm DEPTH36 && expect 1 '^\[model\] variant=S depth=36 embed=384 heads=6 layerscale_init=1e-05' "$WORK/runs/DEPTH36/train.log" "DEPTH36 model line (depth 36)"
has_arm LSCALE  && expect 1 '^\[model\] variant=S depth=12 embed=384 heads=6 layerscale_init=0.01' "$WORK/runs/LSCALE/train.log" "LSCALE model line (layerscale_init 0.01)"
if has_arm PROTO; then
    expect 1 'clustering_entropy' "$WORK/runs/PROTO/train.log" "PROTO clustering_entropy logged"
    if grep -q -i 'nan' <(grep '^It ' "$WORK/runs/PROTO/train.log"); then echo "NaN in PROTO loss line"; exit 1; fi
    echo "  ok: no NaN in PROTO loss lines"
fi
expect 1 'wrote manifest' "$WORK/runs/$FIRST_ARM/train.log" "manifest written once ($FIRST_ARM)"
for a in $ARMS; do [ "$a" = "$FIRST_ARM" ] || expect 1 'loaded manifest' "$WORK/runs/$a/train.log" "manifest reused ($a)"; done

if ! has_arm BASE; then echo; echo "DRY RUN PASSED  (work dir: $WORK; no BASE arm, resume test skipped)"; exit 0; fi
echo; echo "== kill-and-resume: BASE from the rolling checkpoint.pth written at iteration 3 =="
# A first pass stopped after iteration 3 (total_iterations 4) leaves only the rolling checkpoint.pth
# (rolling every 3, periodic every 6 -> no checkpoint_iter file); the second pass resumes from it.
OUT="$WORK/runs/BASE_resume"; mkdir -p "$OUT"
cp "$WORK/probe_manifest.json" "$WORK/probe_manifest_before.json"
COMMON_KILL=("${COMMON[@]}"); for i in "${!COMMON_KILL[@]}"; do [ "${COMMON_KILL[$i]}" = "--total_iterations" ] && COMMON_KILL[$((i+1))]=4; done
(cd "$OUT" && "$PY" "$REPO/main_train.py" "${COMMON_KILL[@]}" "${BASE[@]}" --output_dir "$OUT" > "$OUT/train_first.log" 2>&1) \
    || { echo "FIRST PASS FAILED"; tail -50 "$OUT/train_first.log"; exit 1; }
grep -h '^\[ckpt\]' "$OUT/train_first.log"
expect 1 '^\[ckpt\] it=3 rolling=1 periodic=0' "$OUT/train_first.log" "first pass: rolling checkpoint.pth at 3, no periodic"
test -f "$OUT/checkpoint.pth" || { echo "missing checkpoint.pth after first pass"; exit 1; }
test ! -f "$OUT/checkpoint_iter_00000003.pth" || { echo "unexpected periodic checkpoint at 3 after first pass"; exit 1; }
(cd "$OUT" && "$PY" "$REPO/main_train.py" "${COMMON[@]}" "${BASE[@]}" --output_dir "$OUT" > "$OUT/train_resume.log" 2>&1) \
    || { echo "RESUME FAILED"; tail -50 "$OUT/train_resume.log"; exit 1; }
grep -h 'Resuming from iteration\|Starting training at iteration' "$OUT/train_resume.log"
grep -h '^\[sched-config\]\|^\[sched\]\|^\[diag\]\|^\[ckpt\]' "$OUT/train_resume.log"
expect 1 'Resuming from iteration 3' "$OUT/train_resume.log" "resume at iteration 3"
expect 2 '^\[sched\] it=[46] ' "$OUT/train_resume.log" "sched lines at 4 and 6 after resume"
expect 2 '^\[diag\] it=[46] branch=teacher' "$OUT/train_resume.log" "teacher diag at 4 and 6 after resume"
expect 1 '^\[ckpt\] it=6 rolling=1 periodic=1' "$OUT/train_resume.log" "resume: rolling + periodic checkpoint at 6"
diff <(grep -h '^\[sched-config\]' "$WORK/runs/BASE/train.log") <(grep -h '^\[sched-config\]' "$OUT/train_resume.log") && echo "  ok: sched-config unchanged after resume (vs the uninterrupted BASE run)"
diff <(grep -h '^\[sched\] it=[46]' "$WORK/runs/BASE/train.log") <(grep -h '^\[sched\] it=[46]' "$OUT/train_resume.log") && echo "  ok: [sched] values at 4 and 6 unchanged after resume (vs the uninterrupted BASE run)"
cmp "$WORK/probe_manifest_before.json" "$WORK/probe_manifest.json" && echo "  ok: probe manifest unchanged after resume"
expect 1 'loaded manifest' "$OUT/train_resume.log" "manifest reused on resume"
test -f "$OUT/checkpoint_iter_00000006.pth" && echo "  ok: periodic iteration-6 checkpoint written after resume"
echo; echo "DRY RUN PASSED  (work dir: $WORK)"
