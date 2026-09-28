"""
Unit tests for diagnostics/patch_geometry.py on synthetic tensors with known answers.
Run:  python -m diagnostics.test_patch_geometry
"""

import math
import torch
import torch.nn.functional as F

from diagnostics.patch_geometry import (
    cls_patch_cos, within_between_parts, within_effrk_sum, effective_rank,
    locality, locality_index_sets, sinkhorn_knopp_local, target_entropies,
    pnorm_stats, attention_routing, attention_entropy, tile_mean_cosines, cls_embed_cosine_sum,
)


def check(name, value, expect, tol):
    ok = abs(value - expect) <= tol
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got {value:.6g}, expected {expect:.6g} (tol {tol})")
    return ok


def main():
    torch.manual_seed(0)
    B, P, d = 8, 196, 64
    results = []

    # 1. identical patch tokens within each tile -> within_between = 0, within_effrk = 1
    base = F.normalize(torch.randn(B, 1, d), dim=-1).expand(B, P, d).contiguous()
    tr_w, tr_b = within_between_parts(base)
    results.append(check("within_between (identical patches)", tr_w / tr_b, 0.0, 1e-9))
    # X_b - mu_b is exactly zero -> all singular values 0 -> H = 0 -> effrk = 1
    results.append(check("within_effrk (identical patches)", within_effrk_sum(base) / B, 1.0, 1e-6))
    # one perturbed row -> centered tile is exactly rank 1 -> effrk = 1 (double precision)
    rank1 = base.double().clone()
    rank1[:, 0] += F.normalize(torch.randn(B, d, dtype=torch.float64), dim=-1)
    results.append(check("within_effrk (rank-1 centered tile)", within_effrk_sum(rank1) / B, 1.0, 1e-6))

    # 2. patches equal to the class token -> cls_patch_cos = 1
    c = F.normalize(torch.randn(B, d), dim=-1)
    X = c.unsqueeze(1).expand(B, P, d).contiguous()
    results.append(check("cls_patch_cos (patches == cls)", cls_patch_cos(X, c), 1.0, 1e-6))

    # 3. i.i.d. random patches -> locality near 0
    Xr = F.normalize(torch.randn(64, P, d), dim=-1)
    sets = locality_index_sets(14, 14)
    results.append(check("locality (iid random patches)", locality(Xr, *sets), 0.0, 0.02))
    # sanity: a smooth field gives positive locality
    grid = torch.linspace(0, 1, 14)
    field = torch.stack(torch.meshgrid(grid, grid, indexing='ij'), -1).reshape(1, P, 2)
    smooth = F.normalize(torch.cat([field.expand(4, P, 2), 0.05 * torch.randn(4, P, d - 2)], -1), dim=-1)
    loc_smooth = locality(smooth, *sets)
    results.append(check("locality (smooth field) > 0.1", max(loc_smooth, 0.1), loc_smooth, 1e-9))

    # 4. targets identical across positions -> ibot_img_H == ibot_tok_H
    K, t = 512, 16
    logits_one = torch.randn(1, K)
    T = sinkhorn_knopp_local(logits_one.expand(4 * t, K).contiguous(), 0.07)   # every row identical
    tok_H, img_H = target_entropies(T, t)
    results.append(check("ibot_img_H == ibot_tok_H (identical targets)", img_H, tok_H, 1e-6))
    # marginal entropy is never below the mean token entropy (concavity)
    T2 = sinkhorn_knopp_local(torch.randn(4 * t, K), 0.07)
    tok2, img2 = target_entropies(T2, t)
    results.append(check("ibot_img_H >= ibot_tok_H (random targets)", max(img2, tok2), img2, 1e-9))
    # Sinkhorn output rows sum to 1
    results.append(check("sinkhorn rows sum to 1", float(T2.sum(-1).mean()), 1.0, 1e-4))

    # 5. effective rank of an orthonormal 10-column matrix is 10
    Q, _ = torch.linalg.qr(torch.randn(100, 10))
    results.append(check("effective_rank (orthonormal, 10 cols)", effective_rank(Q), 10.0, 1e-6))

    # 6. pnorm stats on a known vector
    p50, p99, ratio = pnorm_stats(torch.arange(1, 101).float())
    results.append(check("pnorm_p50 (1..100)", p50, 50.5, 1e-6))
    results.append(check("pnorm_max_over_p50 (1..100)", ratio, 100 / 50.5, 1e-6))

    # 7. attention routing on a uniform attention matrix with 4 registers, 1 cls, 3 patches
    N = 1 + 4 + 3
    attn = torch.full((2, 3, N, N), 1.0 / N)
    rm, pm, cm, cnt, row_max = attention_routing(attn, num_reg=4)
    results.append(check("reg_route (uniform attention)", rm / cnt, 4 / N, 1e-6))
    results.append(check("patch_route (uniform attention)", pm / cnt, 3 / N, 1e-6))
    results.append(check("cls_route (uniform attention)", cm / cnt, 1 / N, 1e-6))
    results.append(check("row max (uniform attention)", float(row_max.max()), 1 / N, 1e-6))

    # 8. identical patches -> cos_patch_tilemean = 1 and cos_cls_tilemean = cos(c, x)
    cp, cc = tile_mean_cosines(base, c)
    results.append(check("cos_patch_tilemean (identical patches)", cp / B, 1.0, 1e-6))
    results.append(check("cos_cls_tilemean (identical patches) == cos(c, x)", cc / B,
                         float((c * base[:, 0]).sum(-1).mean()), 1e-6))
    # class token equal to its embedding -> cos_cls_embed = 1
    results.append(check("cos_cls_embed (cls == embedding)", cls_embed_cosine_sum(c[:1], c[0]) / 1, 1.0, 1e-6))
    # 9. uniform attention row -> attn_entropy = log N, attn_entropy_frac = 1.0
    es, ec, nk = attention_entropy(attn, num_reg=4)
    results.append(check("attn_entropy (uniform rows) == log N", es / ec, math.log(N), 1e-6))
    results.append(check("attn_entropy_frac (uniform rows)", (es / ec) / math.log(nk), 1.0, 1e-6))
    one_hot = torch.zeros(2, 3, N, N); one_hot[..., 0] = 1.0
    es1, ec1, _ = attention_entropy(one_hot, num_reg=4)
    results.append(check("attn_entropy (one-hot rows)", es1 / ec1, 0.0, 1e-6))

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
