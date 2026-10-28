"""Perturbation-based receptive field: no gradients needed (works through quantizers, argmax, black boxes).

Add noise to one patch of the input at a time and record how much one output cell changes. Any number of
spatial dimensions."""
from __future__ import annotations

from itertools import product

import torch

from .core import make_input, result_from_map


@torch.no_grad()
def perturbation_receptive_field(fn, in_shape=None, center=None, *, input=None, patch=4, step=None,
                                 window: int | None = None, noise: float = 1.0, batch: int = 32,
                                 rel_tol: float = 1e-4, energy: float = 0.95, power: float = 1.0, scale=1.0, seed: int = 0):
    """Returns an RFResult built from a perturbation-sensitivity map (resolution = `step` input cells).

    center: output-grid cell (default middle). patch/step: int or one per axis. window: only probe +-window
    input cells around the centre (much cheaper than the whole input). rel_tol: support threshold relative
    to the max change; batched forwards can differ from single ones by float dust, so do not set it to ~0."""
    x = make_input(in_shape, input, seed=seed)
    S = tuple(x.shape[2:])
    D = len(S)
    patch = tuple(patch) if hasattr(patch, "__len__") else (patch,) * D
    step = patch if step is None else (tuple(step) if hasattr(step, "__len__") else (step,) * D)
    y0 = fn(x)
    s = tuple(y0.shape[2:])
    c = tuple(v // 2 for v in s) if center is None else ((center,) if isinstance(center, int) else tuple(center))
    ref = tuple((c[i] + 0.5) * S[i] / s[i] for i in range(D))
    rngs = []
    for i in range(D):
        lo, hi = (0, S[i]) if window is None else (max(0, int(ref[i] - window)), min(S[i], int(ref[i] + window)))
        rngs.append(list(range(lo, hi - patch[i] + 1, step[i])))
    pos = list(product(*rngs))
    g = torch.Generator().manual_seed(seed + 1)
    base = y0[(0, slice(None), *c)]
    deltas = torch.zeros(len(pos))
    sl = lambda p: (slice(None), slice(None), *[slice(a, a + patch[i]) for i, a in enumerate(p)])
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = x.repeat(len(chunk), *([1] * (D + 1)))
        for k, p in enumerate(chunk):
            blk = xb[k:k + 1][sl(p)]
            xb[k:k + 1][sl(p)] = blk + noise * torch.randn(blk.shape, generator=g, dtype=x.dtype).to(x.device)
        yb = fn(xb)[(slice(None), slice(None), *c)]
        deltas[i:i + len(chunk)] = (yb - base).norm(dim=1).cpu()
    m = torch.zeros(S)
    for p, d in zip(pos, deltas):
        idx = sl(p)[2:]
        m[idx] = torch.maximum(m[idx], d)
    return result_from_map(m, c, ref, scale=scale, eps=rel_tol, energy=energy, power=power, keep_map=True)
