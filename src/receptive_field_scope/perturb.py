"""Perturbation-based receptive field: no gradients needed (works through quantizers, argmax, black boxes).

Add noise to one patch of the input at a time and record how much one output cell changes."""
from __future__ import annotations

import torch

from .core import make_input, result_from_map


@torch.no_grad()
def perturbation_receptive_field(fn, in_shape=None, center=None, *, input=None, patch: int = 4,
                                 step: int | None = None, window: int | None = None, noise: float = 1.0,
                                 batch: int = 32, rel_tol: float = 1e-4, energy: float = 0.95,
                                 scale: float = 1.0, seed: int = 0):
    """Returns an RFResult built from a perturbation-sensitivity map (resolution = `step` pixels).

    center: output-grid (y, x), default middle.   window: only probe +-window input pixels around the
    centre (much cheaper than the whole image).   rel_tol: support threshold relative to the max change;
    batched forwards can differ from single ones by float dust, so do not set it to ~0.
    Distances are in input pixels * scale."""
    x = make_input(in_shape, input, seed=seed)
    H, W = x.shape[-2:]
    step = step or patch
    y0 = fn(x)
    h, w = y0.shape[-2:]
    cy, cx = center or (h // 2, w // 2)
    ry, rx = (cy + 0.5) * H / h, (cx + 0.5) * W / w
    lo_y, hi_y = (0, H) if window is None else (max(0, int(ry - window)), min(H, int(ry + window)))
    lo_x, hi_x = (0, W) if window is None else (max(0, int(rx - window)), min(W, int(rx + window)))
    ys = list(range(lo_y, hi_y - patch + 1, step))
    xs = list(range(lo_x, hi_x - patch + 1, step))
    g = torch.Generator().manual_seed(seed + 1)
    base = y0[0, :, cy, cx]
    deltas = torch.zeros(len(ys) * len(xs))
    pos = [(a, b) for a in ys for b in xs]
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for k, (a, b) in enumerate(chunk):
            xb[k, :, a:a + patch, b:b + patch] += noise * torch.randn(
                x.shape[1], patch, patch, generator=g, dtype=x.dtype).to(x.device)
        yb = fn(xb)[:, :, cy, cx]
        deltas[i:i + len(chunk)] = (yb - base).norm(dim=1).cpu()
    # paint patch-level deltas back onto an input-resolution map (max over overlapping patches)
    m = torch.zeros(H, W)
    for (a, b), d in zip(pos, deltas):
        m[a:a + patch, b:b + patch] = torch.maximum(m[a:a + patch, b:b + patch], d)
    return result_from_map(m, (cy, cx), (ry, rx), scale=scale, eps=rel_tol, energy=energy, keep_map=True)
