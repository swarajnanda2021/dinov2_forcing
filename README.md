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
| `--depth`, `--layerscale_init` | block count override at the variant's width and heads (default None); LayerScale initial value (default 1e-5, recipe block 1e-5) |
| `--n_standard_local_crops`, `--local_crop_size` | local crop count and size |
| `--ibot_loss_weight`, `--mask_ratio_min`, `--mask_ratio_max`, `--mask_sample_probability` | iBOT weight and block-mask sampling |
| `--koleo_loss_weight` | KoLeo weight on the global class tokens |
| `--use_prototype_clustering`, `--num_prototypes`, `--clustering_weight`, `--clustering_teacher_temp`, `--clustering_student_temp` | patch prototype clustering loss |
| `--save_checkpoint_freq`, `--num_workers`, `--seed` | periodic checkpoint period (`checkpoint_iter_<it>.pth`, kept), loader workers, seed |

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
| `--rolling_checkpoint_freq` | 5000 | rolling `checkpoint.pth` period (overwritten; the file resume reads) |

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

## Arms (wave 2: scale mimicry on the vanilla recipe)

Wave 1 (R0 to R7 and the PROTO arms) is finished. Wave 2 asks one question: does the DINOv3
patch-to-class-token drift come from scale, and can two scale-like changes reproduce it at
ViT-S width. Three arms, each the vanilla recipe plus at most one change, all at global batch
512 and 400 001 iterations.

| Arm | Change from vanilla | GPUs x batch per GPU |
|---|---|---|
| BASE | none | 2 x 256 |
| DEPTH36 | 36 transformer blocks instead of 12, width and heads unchanged (`--depth 36`) | 4 x 128 |
| LSCALE | LayerScale initial value 1e-2 instead of 1e-5 (`--layerscale_init 1e-2`) | 2 x 256 |

Motivation. Depth is the scale variable that separates ViT-L from ViT-g in DINOv3's report,
and each block adds one more averaging step across tokens. A larger LayerScale start lets the
attention content dominate the class token's residual stream from the beginning, which is the
state a deep, long run reaches late. BASE is the control at identical batch, learning rate and
length.

Vanilla recipe means every value in the recipe block of `run_with_submitit.py` plus the suite's
common settings (`scripts/run_forcing_suite.sh`): `vit_variant "S"`, `total_iterations 400_001`,
`warmup_iterations 10_000`, `lr_schedule`, `wd_schedule` and `momentum_schedule` all `cosine`,
`weight_decay 0.04` to `weight_decay_end 0.4`, `momentum_teacher 0.992` to `momentum_teacher_end 1.0`,
base `lr 2e-4` scaled by sqrt(global_batch / 1024) (1.41e-4 at 512), `min_lr 1e-6`,
`drop_path_rate 0.1` with `drop_path_uniform True`, `n_standard_local_crops 8` at
`local_crop_size 96`, iBOT block masking `mask_ratio_min 0.1` to `mask_ratio_max 0.5` on
`mask_sample_probability 0.5` of the tiles, `koleo_loss_weight 0.1`, `use_prototype_clustering False`,
`save_checkpoint_freq 50_000` (periodic, kept), `rolling_checkpoint_freq 5_000` (rolling
`checkpoint.pth`), `diag_every 2000`, `diag_probe_manifest "$BASE_DIR/probe_manifest.json"`,
`num_workers 10`, `seed 0`. The LayerScale value is the existing `--layerscale_init` argument
(the recipe block sets it to 1e-5; LSCALE overrides that line). `--depth` overrides the
variant's block count after the variant mapping; the ViT is constructed in `training/trainer.py`
from `args.vitdepth`, and `layerscale_init` reaches every block's `gamma_1` and `gamma_2` in
`models/vision_transformer/modern_vit.py`. The peak learning rate uses the global batch,
`lr * sqrt(batch_size_per_gpu * world_size / 1024)` with `world_size` the job's GPU count, so all
three arms print the same `[sched-config] lr: cosine peak=0.000141421 ...`. At startup the
trainer prints one line: `[model] variant=S depth=<n> embed=<d> heads=<h> layerscale_init=<v>`.

Launch: `python run_with_submitit.py --nodes 1 --ngpus <2|4|2> --partition vanderbc_gpu`, printed
by the suite. The suite script refuses to overwrite an existing experiment directory (it prints
`exists: <dir>` and exits 1). `scripts/launch_all.sh` sets up the three arms and prints the three
launch commands; it submits them only with `AUTO_SUBMIT=yes`.

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
| `attn_entropy` | last block only: mean over patch queries and heads of the entropy (nats) of the attention row, on the same recomputed attention as the routing metrics |
| `attn_entropy_frac` | `attn_entropy` divided by log of the number of keys; 1.0 is uniform attention |
| `cos_patch_tilemean` | mean over tiles and patches of cos(x_bi, mu_b), L2-normalised post-norm patch tokens against their tile mean |
| `cos_cls_tilemean` | mean over tiles of cos(c_b, mu_b), the post-norm class token against that tile mean |
| `cos_cls_embed` | mean over tiles of cos(c_b, e_cls), the post-norm class token against the learned class-token parameter `backbone.cls_token` (how much of the class output is still its private embedding) |

Two implementation notes. The training attention is a fused kernel, so the last block's
attention is recomputed explicitly from its q and k for the probe batch only (a forward
pre-hook captures the block input; the training forward is not touched). The iBOT target
entropy uses the teacher iBOT head for both branches and a fixed random subset of
`--diag_ibot_tokens` tokens per tile (seed 0), because the full probe set at out_dim 65536 in fp32 would need over 50 GB;
the Sinkhorn is the training routine's arithmetic without its distributed all-reduces, since
the diagnostics run on one rank.

Output, one line per branch per diagnostics step, fixed key order:

```
[diag] it=<n> branch=<student|teacher> cls_patch_cos=... within_between=... within_effrk=... patch_effrk=... cls_effrk=... locality=... ibot_tok_H=... ibot_img_H=... pnorm_p50=... pnorm_p99=... pnorm_max_over_p50=... reg_route=... patch_route=... cls_route=... aw_p99=... attn_entropy=... attn_entropy_frac=... cos_patch_tilemean=... cos_cls_tilemean=... cos_cls_embed=...
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

Reading the four wave-2 metrics for the drift: `attn_entropy_frac` up is uniformisation of the
last block's attention; down while the attention guard (`aw_p99`) goes up is a sink.
`cos_patch_tilemean` up is within-tile collapse. `cos_cls_tilemean` up is the class token
becoming the tile average. `cos_cls_embed` down is the private class embedding losing dominance.

## Local checks (workstation, no cluster)

```
python -m diagnostics.test_patch_geometry     # metric unit tests on synthetic tensors
bash scripts/local_dryrun.sh                  # synthetic zip dataset, BASE / DEPTH36 / LSCALE for 6 iterations, kill-and-resume (DRYRUN_ARMS selects arms)
```

The dry run uses fp32 on a CPU (bf16 autocast on a CPU is about twenty times slower for the
backward pass); a local CUDA GPU keeps the bf16 path.

## Cluster checklist

1. Cancel every job of wave 1 and delete the wave-1 directories under
   `/data1/vanderbc/test_dinov2_swaraj` (forcing_ViT-S_R0 to R7, the PROTO arms, SMOKE and
   *_failed_* directories). `probe_manifest.json`, `plot_forcing.py`, `forcing_monitor.sh`
   and `forcing_monitor/` stay.
2. Clone this commit, verify the suite defines exactly BASE, DEPTH36, LSCALE and that
   `--depth` and the LayerScale argument (`--layerscale_init`) exist.
3. Set up and launch the three arms with their GPU counts on `vanderbc_gpu`, no smoke arms.
4. Verify each shows `[model]`, `[sched-config]` with peak lr 1.41e-4 and cosine kinds, and
   `[diag]` at iteration 0 with the four new keys; report memory and iterations per second.
