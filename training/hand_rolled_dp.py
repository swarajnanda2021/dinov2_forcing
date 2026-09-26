"""Hand-rolled data parallelism -- a drop-in replacement for DistributedDataParallel
that does NOT wrap the autograd graph, so it composes with torch.compile + gradient
checkpointing (DDP breaks that nesting: CheckpointError, pytorch#144035).

Usage (replaces DDP):
    model = HandRolledDP(model)          # broadcasts params from rank 0 at init
    ...
    loss.backward()                      # local grads only, no DDP hooks
    model.sync_grads()                   # <-- manual all-reduce, BEFORE clip/step
    clip_gradients(model, ...); optimizer.step()

Gradient sync is DDP-EQUIVALENT (verified bit-equal on A100, 66/66 params, maxabs
0.0 at drop_path=0):
  * one BUCKETED all-reduce (SUM) then / world_size  -- DDP buckets internally, a
    naive per-param loop is both slower and an unfair comparison;
  * a PRESENCE MASK so a parameter that received NO gradient on ANY rank ends as
    grad=None, NOT a zero tensor. This is the subtle correctness point: AdamW skips
    a None grad, but a *zero* grad still applies decoupled weight decay
    (p *= 1 - lr*wd) and still moves p via existing Adam moments -- so a
    conditionally-unused head (patchhead, semantic/prototype heads) would silently
    decay. DDP + find_unused_parameters leaves such grads None; we replicate that.

Params are broadcast from rank 0 at construction (like DDP) so every rank starts
identical -- the EMA teacher update reads student params directly and assumes
cross-rank identity, so any divergence would silently corrupt the teacher.

NOTE (overlap): this sync is a blocking phase after backward(); DDP hides the
all-reduce inside backward via per-bucket hooks. On NCCL/NVLink the exposed comm is
small; if a profile shows it exposed, add gradient-bucket hooks here. The blocking
version is what was verified bit-equal, so it ships first.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _dist_on():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class HandRolledDP(nn.Module):
    """Delegates forward to `.module` (same attribute name as DDP, so existing
    `model.module.<...>` access is unchanged) and syncs gradients on demand."""

    def __init__(self, module, broadcast=True):
        super().__init__()
        self.module = module                      # registered submodule -> .parameters() works
        # stable param order, shared by every rank (identical model construction)
        self._params = list(module.parameters())
        if broadcast and _dist_on():
            for p in self._params:
                dist.broadcast(p.data, src=0)
            for b in module.buffers():
                dist.broadcast(b.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _set_static_graph(self):
        # DDP-only optimization; no-op here (kept so existing call sites don't break).
        pass

    @torch.no_grad()
    def sync_grads(self):
        """All-reduce (average) gradients across ranks. Call AFTER backward() and
        BEFORE gradient clipping / optimizer.step()."""
        if not _dist_on():
            return
        world = dist.get_world_size()
        params = self._params
        dev = params[0].device

        # (1) presence mask: did ANY rank produce a grad for param i?
        present = torch.tensor([0.0 if p.grad is None else 1.0 for p in params], device=dev)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)

        # (2) every rank must fill its slot so the flat buffers align across ranks
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # (3) ONE bucketed all-reduce per dtype (SUM then /world)
        by_dtype = {}
        for p in params:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
        for grads in by_dtype.values():
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= world
            for g, s in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(s)

        # (4) DDP parity: globally-unused params -> None (skip weight-decay / Adam drift)
        present_list = present.tolist()
        for pr, p in zip(present_list, params):
            if pr == 0.0:
                p.grad = None
