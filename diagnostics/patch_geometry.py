"""
Patch-token geometry against the class token, measured on a fixed probe set.

Runs on rank 0 only, under torch.no_grad() and the training autocast dtype, on the student
and the teacher backbone in eval mode. The tokens are the post-norm outputs (backbone.norm)
that the losses consume in training: 'clstoken' feeds the DINO head and 'patchtokens' feed
the iBOT head (single-image keys: clstoken_postnorm / patchtokens_postnorm). The pnorm_* outlier
guard alone reads the pre-norm residual stream (patchtokens_prenorm): after backbone.norm every
token has norm close to sqrt(d), so high-norm outlier tokens are only visible before it.

Sinkhorn-Knopp here is the training routine's arithmetic (losses/ibot_loss.py) without the
distributed all-reduces, because the diagnostics run on one rank over the probe set.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F


DIAG_KEYS = [
    'cls_patch_cos', 'within_between', 'within_effrk', 'patch_effrk', 'cls_effrk', 'locality',
    'ibot_tok_H', 'ibot_img_H', 'pnorm_p50', 'pnorm_p99', 'pnorm_max_over_p50',
    'reg_route', 'patch_route', 'cls_route', 'aw_p99',
    'attn_entropy', 'attn_entropy_frac', 'cos_patch_tilemean', 'cos_cls_tilemean', 'cos_cls_embed',
]

# iBOT target entropy is measured on a fixed random subset of tokens per tile (seed 0): the
# full probe set (1024 tiles x 196 tokens x out_dim 65536 in fp32) would need >50 GB.
# The count comes from --diag_ibot_tokens (trainer default: 49 on CUDA, 16 on CPU).
IBOT_TOKENS_PER_TILE = 16
LOCALITY_FAR_SAMPLES = 8
LOCALITY_FAR_MIN_DIST = 4   # Chebyshev grid distance


# ---------------------------------------------------------------- pure metric functions

def effective_rank_from_gram(G):
    """effrk = exp(H(p)), p = normalized eigenvalues of the Gram matrix (= squared singular values)."""
    ev = torch.linalg.eigvalsh(G.double()).clamp(min=0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def effective_rank(X):
    """X: [n, d] -> exp of the entropy of the normalized squared singular values."""
    X = X.double()
    G = X.t() @ X if X.shape[0] >= X.shape[1] else X @ X.t()
    return effective_rank_from_gram(G)


def batched_effective_rank(X):
    """X: [B, n, d] -> tensor [B] of effective ranks."""
    s = torch.linalg.svdvals(X.double())          # [B, min(n, d)]
    p = s.pow(2)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-30)
    H = -(p * torch.log(p.clamp(min=1e-30))).sum(dim=1)
    return torch.exp(H)


def cls_patch_cos(Xn, cn):
    """Xn: [B, P, d] L2-normalized patch tokens; cn: [B, d] L2-normalized class token.
    mean over b of mean over i of cos(x_bi, c_b)."""
    return float((Xn @ cn.unsqueeze(-1)).squeeze(-1).mean().item())


def within_between_parts(Xn):
    """Returns (tr(S_W), tr(S_B)) for one batch of tiles. Xn: [B, P, d] L2-normalized.
    S_W = sum_b sum_i (x_bi - mu_b)(x_bi - mu_b)^T,  S_B = sum_b P mu_b mu_b^T."""
    mu = Xn.mean(dim=1, keepdim=True)                       # [B, 1, d]
    tr_w = (Xn - mu).pow(2).sum()
    tr_b = Xn.shape[1] * mu.squeeze(1).pow(2).sum()
    return float(tr_w.item()), float(tr_b.item())


def within_effrk_sum(Xn):
    """Sum over tiles of effrk(X_b - mu_b); divide by the tile count for the mean."""
    mu = Xn.mean(dim=1, keepdim=True)
    return float(batched_effective_rank(Xn - mu).sum().item())


def locality_index_sets(grid_h, grid_w, n_far=LOCALITY_FAR_SAMPLES, min_dist=LOCALITY_FAR_MIN_DIST, seed=0):
    """Fixed neighbour and far-patch index sets for a grid.
    Returns (nbr_idx [P, 4], nbr_valid [P, 4] bool, far_idx [P, n_far])."""
    P = grid_h * grid_w
    g = torch.Generator().manual_seed(seed)
    nbr_idx = torch.zeros(P, 4, dtype=torch.long)
    nbr_valid = torch.zeros(P, 4, dtype=torch.bool)
    far_idx = torch.zeros(P, n_far, dtype=torch.long)
    for i in range(P):
        r, c = divmod(i, grid_w)
        for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid_h and 0 <= cc < grid_w:
                nbr_idx[i, k] = rr * grid_w + cc
                nbr_valid[i, k] = True
        cand = [j for j in range(P)
                if max(abs(j // grid_w - r), abs(j % grid_w - c)) >= min_dist]
        perm = torch.randperm(len(cand), generator=g)[:n_far]
        far_idx[i] = torch.tensor([cand[p] for p in perm.tolist()], dtype=torch.long)
    return nbr_idx, nbr_valid, far_idx


def locality(Xn, nbr_idx, nbr_valid, far_idx):
    """mean over b, i of [mean cos to the valid 4-neighbours of i minus mean cos to n_far random
    patches of the same tile at grid distance >= min_dist]. Xn: [B, P, d] L2-normalized."""
    B, P, d = Xn.shape
    nbr = Xn[:, nbr_idx]                                    # [B, P, 4, d]
    far = Xn[:, far_idx]                                    # [B, P, n_far, d]
    cos_nbr = (nbr * Xn.unsqueeze(2)).sum(-1)               # [B, P, 4]
    cos_far = (far * Xn.unsqueeze(2)).sum(-1)               # [B, P, n_far]
    valid = nbr_valid.to(Xn.device).unsqueeze(0).float()
    mean_nbr = (cos_nbr * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return float((mean_nbr - cos_far.mean(-1)).mean().item())


@torch.no_grad()
def sinkhorn_knopp_local(teacher_output, teacher_temp, n_iterations=3):
    """Same arithmetic as iBOTPatchLoss.sinkhorn_knopp_normalization with world_size = 1
    (no collectives). teacher_output: [M, K] logits -> [M, K] targets."""
    Q = teacher_output.float().div(teacher_temp).exp().t()  # [K, M]
    B = Q.shape[1]
    K = Q.shape[0]
    Q /= torch.sum(Q)
    for _ in range(n_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


def target_entropies(T, tokens_per_tile):
    """T: [n_tiles * tokens_per_tile, K] targets, tile-major. Returns (ibot_tok_H, ibot_img_H):
    mean per-token entropy, and mean over tiles of the entropy of the per-tile marginal."""
    eps = 1e-12
    tok_H = -(T * torch.log(T.clamp(min=eps))).sum(-1).mean()
    marg = T.reshape(-1, tokens_per_tile, T.shape[-1]).mean(dim=1)     # [n_tiles, K]
    img_H = -(marg * torch.log(marg.clamp(min=eps))).sum(-1).mean()
    return float(tok_H.item()), float(img_H.item())


def pnorm_stats(norms):
    """norms: [n] token norms -> (p50, p99, max/p50)."""
    n = norms.float()
    p50 = float(torch.quantile(n, 0.50).item())
    p99 = float(torch.quantile(n, 0.99).item())
    return p50, p99, float(n.max().item() / max(p50, 1e-12))


def attention_routing(attn, num_reg):
    """attn: [B, H, N, N] softmax rows (queries x keys), token order [cls, reg..., patches].
    Returns per-batch sums so batches can be averaged: (reg_mass, patch_mass, cls_mass, count, row_max [rows])."""
    p0 = num_reg + 1
    rows = attn[:, :, p0:, :]                               # patch queries
    cls_mass = rows[..., 0].sum()
    reg_mass = rows[..., 1:p0].sum(-1).sum()
    patch_mass = rows[..., p0:].sum(-1).sum()
    count = rows.shape[0] * rows.shape[1] * rows.shape[2]
    row_max = rows.max(dim=-1).values.reshape(-1)
    return float(reg_mass.item()), float(patch_mass.item()), float(cls_mass.item()), count, row_max


def attention_entropy(attn, num_reg):
    """attn: [B, H, N, N] softmax rows. Returns (sum of row entropies in nats over patch queries,
    heads and batch; count of those rows; number of keys N)."""
    rows = attn[:, :, num_reg + 1:, :].float()
    H = -(rows * torch.log(rows.clamp(min=1e-12))).sum(-1)      # [B, H, Pq]
    return float(H.sum().item()), H.numel(), attn.shape[-1]


def tile_mean_cosines(Xn, cn):
    """Xn: [B, P, d] L2-normalized patch tokens; cn: [B, d] L2-normalized class token.
    mu_b = tile mean of Xn. Returns per-batch SUMS over tiles of
    (mean_i cos(x_bi, mu_b), cos(c_b, mu_b)) so batches can be averaged."""
    mu = F.normalize(Xn.mean(dim=1), dim=-1)                    # [B, d]
    cos_patch = (Xn @ mu.unsqueeze(-1)).squeeze(-1).mean(dim=1) # [B]
    cos_cls = (cn * mu).sum(-1)                                 # [B]
    return float(cos_patch.sum().item()), float(cos_cls.sum().item())


def cls_embed_cosine_sum(cn, cls_embed):
    """Sum over tiles of cos(c_b, e_cls); cls_embed: [d] learned class-token parameter."""
    e = F.normalize(cls_embed.float().reshape(-1), dim=0)
    return float((cn @ e).sum().item())


# ---------------------------------------------------------------- backbone probing

def _last_block_attention(block, x_in):
    """Recompute the last block's attention weights explicitly from its q and k (the training forward
    uses a fused kernel and is not touched). x_in: the block's input [B, N, C]."""
    x = block.norm1(x_in)
    B, N, C = x.shape
    qkv = block.qkv(x).reshape(B, N, 3, block.num_heads, block.head_dim)
    q, k, _ = qkv.unbind(dim=2)
    q = block.q_norm(q).float()
    k = block.k_norm(k).float()
    logits = torch.einsum('bnhd,bmhd->bhnm', q, k) * block.scale
    return logits.softmax(dim=-1)                            # [B, H, N, N]


@torch.no_grad()
def probe_backbone(backbone, patchhead, loader, device, amp_enabled, teacher_temp,
                   tokens_per_tile=IBOT_TOKENS_PER_TILE):
    """Compute every DIAG_KEYS metric for one backbone over the probe loader."""
    was_training = backbone.training
    backbone.eval()
    head_was_training = patchhead.training
    patchhead.eval()

    captured = {}
    def _pre_hook(_m, inputs):
        captured['x'] = inputs[0]
    handle = backbone.blocks[-1].register_forward_pre_hook(_pre_hook)

    num_reg = backbone.numregisters
    d = backbone.embed_dim
    gram_patch = torch.zeros(d, d, dtype=torch.float64, device=device)
    cls_all = []
    sum_cos = 0.0; n_tiles = 0
    tr_w = 0.0; tr_b = 0.0
    within_sum = 0.0
    loc_sum = 0.0; loc_batches = 0
    norms_all = []
    ibot_logits = []
    reg_m = patch_m = cls_m = 0.0; route_count = 0
    row_max_all = []
    ent_sum = 0.0; ent_count = 0; n_keys = 1
    cos_pt_sum = 0.0; cos_ct_sum = 0.0; cos_ce_sum = 0.0
    cls_embed = backbone.cls_token.detach().reshape(-1)
    loc_sets = None
    tok_gen = torch.Generator().manual_seed(0)

    try:
        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                out = backbone(x, token_masks=None, return_dict=True)
                X_raw = out['patchtokens_postnorm']
                c_raw = out['clstoken_postnorm']
                X_pre = out['patchtokens_prenorm']
                attn = _last_block_attention(backbone.blocks[-1], captured['x'])
                # iBOT head on a fixed random subset of tokens per tile
                Bc, P, _ = X_raw.shape
                tok_idx = torch.stack([torch.randperm(P, generator=tok_gen)[:tokens_per_tile]
                                       for _ in range(Bc)]).to(device)          # [Bc, t]
                sub = torch.gather(X_raw, 1, tok_idx.unsqueeze(-1).expand(-1, -1, d))
                ibot_logits.append(patchhead(sub.reshape(-1, d)).float())

            X = X_raw.float(); c = c_raw.float()
            B, P, _ = X.shape
            Xn = F.normalize(X, dim=-1); cn = F.normalize(c, dim=-1)

            sum_cos += cls_patch_cos(Xn, cn) * B; n_tiles += B
            w, b = within_between_parts(Xn); tr_w += w; tr_b += b
            within_sum += within_effrk_sum(Xn)
            gram_patch += Xn.reshape(-1, d).double().t() @ Xn.reshape(-1, d).double()
            cls_all.append(cn)
            if loc_sets is None:
                g = int(math.sqrt(P)); assert g * g == P, f"non-square patch grid P={P}"
                loc_sets = locality_index_sets(g, g)
                loc_sets = tuple(t.to(device) for t in loc_sets)
            loc_sum += locality(Xn, *loc_sets) * B
            norms_all.append(X_pre.float().norm(dim=-1).reshape(-1))
            rm, pm, cm, cnt, row_max = attention_routing(attn, num_reg)
            reg_m += rm; patch_m += pm; cls_m += cm; route_count += cnt
            row_max_all.append(row_max.float())
            es, ec, n_keys = attention_entropy(attn, num_reg); ent_sum += es; ent_count += ec
            cp, cc = tile_mean_cosines(Xn, cn); cos_pt_sum += cp; cos_ct_sum += cc
            cos_ce_sum += cls_embed_cosine_sum(cn, cls_embed)
    finally:
        handle.remove()
        backbone.train(was_training)
        patchhead.train(head_was_training)

    cls_cat = torch.cat(cls_all, 0)
    T = sinkhorn_knopp_local(torch.cat(ibot_logits, 0), teacher_temp)
    tok_H, img_H = target_entropies(T, tokens_per_tile)
    p50, p99, ratio = pnorm_stats(torch.cat(norms_all, 0))
    row_max_cat = torch.cat(row_max_all, 0)

    return {
        'cls_patch_cos': sum_cos / max(n_tiles, 1),
        'within_between': tr_w / max(tr_b, 1e-12),
        'within_effrk': within_sum / max(n_tiles, 1),
        'patch_effrk': effective_rank_from_gram(gram_patch),
        'cls_effrk': effective_rank(cls_cat),
        'locality': loc_sum / max(n_tiles, 1),
        'ibot_tok_H': tok_H,
        'ibot_img_H': img_H,
        'pnorm_p50': p50,
        'pnorm_p99': p99,
        'pnorm_max_over_p50': ratio,
        'reg_route': reg_m / max(route_count, 1),
        'patch_route': patch_m / max(route_count, 1),
        'cls_route': cls_m / max(route_count, 1),
        'aw_p99': float(torch.quantile(row_max_cat, 0.99).item()),
        'attn_entropy': ent_sum / max(ent_count, 1),
        'attn_entropy_frac': (ent_sum / max(ent_count, 1)) / math.log(max(n_keys, 2)),
        'cos_patch_tilemean': cos_pt_sum / max(n_tiles, 1),
        'cos_cls_tilemean': cos_ct_sum / max(n_tiles, 1),
        'cos_cls_embed': cos_ce_sum / max(n_tiles, 1),
    }


def format_diag_line(it, branch, metrics):
    parts = [f"[diag] it={it}", f"branch={branch}"] + [f"{k}={metrics[k]:.6g}" for k in DIAG_KEYS]
    return " ".join(parts)


def run_diagnostics(it, student_backbone, teacher_backbone, teacher_patchhead, probe_loader,
                    device, amp_enabled, teacher_temp, output_dir, tokens_per_tile=IBOT_TOKENS_PER_TILE):
    """Print one [diag] line per branch and append one JSON record per branch to diag.jsonl.
    The teacher iBOT head is used for both branches (the training targets come from it)."""
    t0 = time.time()
    records = []
    for branch, backbone in (('student', student_backbone), ('teacher', teacher_backbone)):
        m = probe_backbone(backbone, teacher_patchhead, probe_loader, device, amp_enabled, teacher_temp,
                           tokens_per_tile=tokens_per_tile)
        print(format_diag_line(it, branch, m), flush=True)
        rec = {'it': int(it), 'branch': branch}
        rec.update({k: float(m[k]) for k in DIAG_KEYS})
        records.append(rec)
    with open(os.path.join(output_dir, 'diag.jsonl'), 'a') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[diag-time] it={it} seconds={time.time() - t0:.2f}", flush=True)
    return records
