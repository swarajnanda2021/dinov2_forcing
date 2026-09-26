# dinov2_forcing

A minimal DINOv2 training loop that exposes a small set of training knobs and logs the
geometry of patch tokens against the class token during training. The experiment is a
forcing study at ViT-S scale: each run turns one knob, and the log shows whether patch
tokens drift toward the class token. There is no evaluation in this repository: no MIL,
no kNN, no feature extraction. The `[diag]` log lines are the result.

## Provenance

Extracted by deletion from `https://github.com/swarajnanda2021/Dinov2_modified`, branch
`pathology-fm-recipe-tuned`, commit `c08262c`. The dataloader, the ViT, the hand-rolled data
parallel, the checkpoint save and resume, the submitit launcher and the log printing are the
source's, minus the removed code paths (semantic iBOT and the mask model, adversarial /
CellViT / random-mask augmentation, typicality dampening and stream thinning, the pathology
recipe bundle and the KDE loss, GPU augmentation, evaluation hooks). KoLeo is the only
regularizer. `DINOHead` was moved into `models/vision_transformer/modern_vit.py` because the
file that held it (`auxiliary_models.py`) was not carried over.

Two local-only changes exist so the code imports and runs on a CPU workstation: the xformers
import in `modern_vit.py` is guarded with a scaled-dot-product-attention fallback, and
`utils.init_distributed_mode` starts a single gloo process when no GPU is present. On the
cluster (CUDA + xformers) the training forward is unchanged.

## Knobs

Existing arguments, kept as named in `configs/config.py`:

| argument | meaning |
|---|---|
| `--lr`, `--min_lr` | base lr at global batch 1024 (peak = lr x sqrt(global_batch / 1024)); floor of the cosine lr schedule |
| `--weight_decay`, `--weight_decay_end` | start and end of the cosine weight-decay schedule |
| `--momentum_teacher` | teacher EMA momentum (start value) |
| `--warmup_iterations`, `--total_iterations` | linear lr warmup length; run length |
| `--batch_size_per_gpu`, `--vit_variant` | per-GPU batch; ViT size S/B/L/H/G (sets embeddingdim, vitdepth, vitheads) |
| `--n_standard_local_crops`, `--local_crop_size` | local crop count and size |
| `--ibot_loss_weight`, `--mask_ratio_min`, `--mask_ratio_max`, `--mask_sample_probability` | iBOT weight and block-mask sampling |
| `--koleo_loss_weight` | KoLeo weight on the global class tokens |
| `--use_prototype_clustering`, `--num_prototypes`, `--clustering_weight`, `--clustering_teacher_temp`, `--clustering_student_temp` | patch prototype clustering loss |
| `--save_checkpoint_freq`, `--num_workers`, `--seed` | checkpoint period, loader workers, seed |

New arguments:

| argument | default | meaning |
|---|---|---|
| `--lr_schedule {cosine,constant}` | cosine | constant: linear warmup 0 to peak over `warmup_iterations`, then hold the peak; `min_lr` ignored |
| `--wd_schedule {cosine,constant}` | cosine | constant: `weight_decay` from iteration 0 to the end; `weight_decay_end` ignored |
| `--momentum_schedule {cosine,constant}` | cosine | constant: `momentum_teacher` for the whole run; `momentum_teacher_end` ignored |
| `--momentum_teacher_end` | 1.0 | end value of the cosine momentum schedule |
| `--diag_every` | 2000 | diagnostics at iteration 0 and every this many iterations |
| `--diag_probe_manifest` | none | JSON manifest of the shared probe tiles (created on first use) |
| `--diag_probe_size` | 1024 | number of probe tiles drawn when the manifest is created |
| `--diag_ibot_tokens` | 49 on CUDA, 16 on CPU | patch tokens per probe tile fed to the iBOT head for the entropy metrics |

The schedules are built in `training/trainer.py` next to the source's `cosine_scheduler`
calls. At startup the resolved schedules are printed, one line each:

```
[sched-config] lr: <cosine|constant> peak=<value> min=<value> warmup=<n>
[sched-config] wd: ...
[sched-config] momentum: ...
```

At every diagnostics step the values actually applied are printed, read from the optimizer
param groups (max over groups, which is the unscaled base value) and the momentum schedule
entry used by the EMA update at that iteration:

```
[sched] it=<n> lr=<value> wd=<value> m=<value>
```

## Arms

Common settings for every arm (`scripts/run_forcing_suite.sh`): `vit_variant "S"`,
`batch_size_per_gpu 256`, `num_workers 10`, `total_iterations 200_001`,
`warmup_iterations 10_000`, `save_checkpoint_freq 10_000`, `momentum_teacher 0.992`,
`weight_decay 0.04`, `weight_decay_end 0.4`, `min_lr 1e-6`, `lr 2e-4` (1e-4 applied at one
GPU x 256), `drop_path_rate 0.1` (the source recipe launcher value, with `drop_path_uniform True`), `n_standard_local_crops 8`,
`local_crop_size 96`, `ibot_loss_weight 1.0`, `mask_ratio_min 0.1`, `mask_ratio_max 0.5`,
`mask_sample_probability 0.5`, `koleo_loss_weight 0.1`, `use_prototype_clustering False`,
`diag_every 2000`, `diag_probe_manifest "$BASE_DIR/probe_manifest.json"`, `seed 0`.
Launch: `python run_with_submitit.py --nodes 1 --ngpus 1 --partition gpu`.

| arm | change relative to the common settings |
|---|---|
| R0 | recipe control: `lr_schedule cosine`, `wd_schedule cosine`, `momentum_schedule cosine` |
| R1 | `lr_schedule constant`, `wd_schedule constant`, `weight_decay 0.4`, `momentum_schedule constant` |
| R2 | R1 + `weight_decay 1.0` |
| R3 | R1 + `weight_decay 0.1` |
| R4 | R1 + `lr 1e-4` (0.5 x the R1 base lr) |
| R5 | R1 + `ibot_loss_weight 0.5` |
| R6 | R1 + `n_standard_local_crops 16`, `local_crop_size 64` |
| R7 | R1 + `mask_ratio_min 0.5`, `mask_ratio_max 0.75` |
| R1_PROTO | R1 + `use_prototype_clustering True`, `num_prototypes 16384`, `clustering_weight 1.0` |
| R0_PROTO | R0 + `use_prototype_clustering True`, `num_prototypes 16384`, `clustering_weight 1.0` |
| SMOKE | R1 with `total_iterations 501`, `warmup_iterations 100`, `diag_every 100`, `save_checkpoint_freq 250` |
| SMOKE_PROTO | SMOKE + `use_prototype_clustering True` |

`scripts/launch_all.sh` sets up R0 to R7 (eight GPUs, one each) and prints the eight launch
commands; it submits them only with `AUTO_SUBMIT=yes`. The PROTO arms are a second wave.

## Diagnostics

`diagnostics/probe_set.py` draws `diag_probe_size` tiles once from the training stream with
seed 0, split across the datasets by their stream proportions and spread over many zips, and
writes them to `--diag_probe_manifest` as `{zip, member}` pairs. If the manifest exists it is
loaded, never redrawn, so every arm under one `BASE_DIR` shares the same probe. The probe loader
applies Resize to 224, ToTensor and Normalize with the training mean and std, no augmentation,
fixed order, batches of 128.

`diagnostics/patch_geometry.py` runs on rank 0 at iteration 0 and every `diag_every`
iterations, under `torch.no_grad()` and the training autocast dtype, on the student and the
teacher backbone in eval mode. The tokens are the post-norm outputs (after `backbone.norm`),
which is what the losses consume in `trainer.py`: `clstoken` feeds the DINO head and
`patchtokens` feed the iBOT head (the single-image keys `clstoken_postnorm` and
`patchtokens_postnorm`). The one exception is the `pnorm_*` outlier guard, which reads the
pre-norm residual stream (`patchtokens_prenorm`): after the final LayerNorm every token has
norm close to sqrt(d), so high-norm outlier tokens are only visible before it.

For each tile b, `X_b` is its patch tokens (P x d) and `c_b` its class token. All cosine and
rank quantities use L2-normalized rows. effrk = exp(H(p)) with p the normalized squared
singular values (there was no effective-rank function in `utils.py`).

| key | definition |
|---|---|
| `cls_patch_cos` | mean over b of mean over i of cos(x_bi, c_b) |
| `within_between` | tr(S_W) / tr(S_B), mu_b the per-tile mean, S_W = sum_b sum_i (x_bi - mu_b)(x_bi - mu_b)^T, S_B = sum_b P mu_b mu_b^T |
| `within_effrk` | mean over b of effrk(X_b - mu_b) |
| `patch_effrk` | effrk of all probe patch tokens stacked |
| `cls_effrk` | effrk of all probe class tokens stacked |
| `locality` | mean over b and i of [mean cos to the 4-neighbour patches of i minus mean cos to 8 random patches of the same tile at Chebyshev grid distance >= 4]; the random far patches are fixed per grid position (seed 0) |
| `ibot_tok_H` | teacher iBOT head on the probe patch tokens, Sinkhorn-Knopp as in training over the probe set at the current teacher temperature; mean per-token entropy of the targets |
| `ibot_img_H` | mean over b of the entropy of the per-tile marginal (mean over i of t_bi) |
| `pnorm_p50`, `pnorm_p99`, `pnorm_max_over_p50` | percentiles and max/p50 of the pre-norm patch token norms |
| `reg_route`, `patch_route`, `cls_route` | last block only: mean attention mass from patch queries to register keys, patch keys, the class key |
| `aw_p99` | last block only: 99th percentile of the maximum attention weight per patch-query row |

Two implementation notes. The training attention is a fused kernel, so the last block's
attention is recomputed explicitly from its q and k for the probe batch only (a forward
pre-hook captures the block input; the training forward is not touched). The iBOT target
entropy uses the teacher iBOT head for both branches and a fixed random subset of
`--diag_ibot_tokens` tokens per tile (seed 0), because the full probe set at out_dim 65536 in fp32 would need over 50 GB;
the Sinkhorn is the training routine's arithmetic without its distributed all-reduces, since
the diagnostics run on one rank.

Output, one line per branch per diagnostics step, fixed key order:

```
[diag] it=<n> branch=<student|teacher> cls_patch_cos=... within_between=... within_effrk=... patch_effrk=... cls_effrk=... locality=... ibot_tok_H=... ibot_img_H=... pnorm_p50=... pnorm_p99=... pnorm_max_over_p50=... reg_route=... patch_route=... cls_route=... aw_p99=...
```

The same record is appended as one JSON object per line to `<output_dir>/diag.jsonl`, and a
`[diag-time] it=<n> seconds=<s>` line reports the cost of the step. Budget: under one percent
of training time at the default settings; the H100 number comes from the SMOKE arm.

## Reading a run

```
grep -h '^\[diag\]' logs/*.out | grep 'branch=teacher'
grep -h '^\[sched\]' logs/*.out
```

Expected signature of forcing: `cls_patch_cos` rising; `within_between`, `within_effrk` and
`locality` falling; `ibot_img_H` falling; the guards (`pnorm_*`, `reg_route`, `patch_route`,
`cls_route`, `aw_p99`) flat. A run in which `pnorm_max_over_p50` or `aw_p99` blows up is a
different failure (outlier tokens or attention saturation), not forcing, and is discarded.

## Local checks (workstation, no cluster)

```
python -m diagnostics.test_patch_geometry     # metric unit tests on synthetic tensors
bash scripts/local_dryrun.sh                  # synthetic zip dataset, R0 / R1 / R1_PROTO for 6 iterations, kill-and-resume
```

The dry run uses fp32 on a CPU (bf16 autocast on a CPU is about twenty times slower for the
backward pass); a local CUDA GPU keeps the bf16 path.

## Cluster checklist

1. On the cluster: `git clone https://github.com/swarajnanda2021/dinov2_forcing.git` (pull only),
   `conda activate ssl-v1`.
2. `scripts/run_forcing_suite.sh SMOKE`, then the printed launch command.
3. Expect in the log: `[sched-config]`, `[sched]` with constant lr, wd, m after warmup,
   `[diag]` for both branches at 0, 100, 200, 300, 400, 500, `diag.jsonl` with 12 records,
   a checkpoint at 250 and 500. Kill at about 300, relaunch, confirm resume at 300.
4. `scripts/run_forcing_suite.sh SMOKE_PROTO`, launch; expect prototype loss values and
   `clustering_entropy` in the log, no NaN.
5. Note iterations per second for both smoke arms and the diagnostics step cost. These size
   the 200k runs.
6. Any failure: paste the traceback and the last 50 log lines back to the workstation session.
   The fix is pushed; pull and rerun.
7. When both smoke arms pass: `scripts/launch_all.sh` and submit the eight printed commands.
