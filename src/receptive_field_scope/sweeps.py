"""Sweeps: alignment offsets, position maps (border / register-token effects), input dependence."""
from __future__ import annotations

from itertools import product

import torch

from .core import make_input, measure_receptive_field


def _tup(v, d):
    return tuple(v) if hasattr(v, "__len__") else (v,) * d


def sweep_centers(out_shape, period=1, margin: int = 0, anchor=None):
    """All output-grid offsets inside one `period`-sized block near `anchor` (default grid middle,
    rounded down to a multiple of `period`). `out_shape` is the output grid, e.g. (32, 32) or (T,);
    `period` is an int or one int per axis (the block size on the OUTPUT grid). Exposes block/stride
    alignment dependence."""
    shape = (out_shape,) if isinstance(out_shape, int) else tuple(out_shape)
    D = len(shape)
    per = _tup(period, D)
    anc = tuple(n // 2 for n in shape) if anchor is None else tuple(anchor)
    anc = tuple(a - a % p for a, p in zip(anc, per))
    assert all(margin <= a and a + p <= n - margin for a, p, n in zip(anc, per, shape)), \
        "grid too small for this margin/period"
    return [tuple(a + d for a, d in zip(anc, off)) for off in product(*[range(p) for p in per])]


def grid_centers(out_shape, step: int = 1, margin: int = 0):
    """Regular grid of output cells, for position maps."""
    shape = (out_shape,) if isinstance(out_shape, int) else tuple(out_shape)
    return [c for c in product(*[range(margin, n - margin, step) for n in shape])]


def position_map(fn, in_shape=None, *, input=None, step: int = 1, margin: int = 0, **kw):
    """RF vs position: dict of tensors shaped like the probed grid (support radius, effective radius,
    asymmetry along x/time = asym_x, and along y = asym_y when D >= 2).

    Use to see border effects and to spot tokens whose reach is out of line with their neighbours
    (e.g. register-like tokens in models that are meant to be local)."""
    x = make_input(in_shape, input)
    with torch.no_grad():
        s = tuple(fn(x).shape[2:])
    cs = grid_centers(s, step, margin)
    rs = measure_receptive_field(fn, input=x, centers=cs, **kw)
    gshape = tuple(len(range(margin, n - margin, step)) for n in s)
    t = lambda f: torch.tensor([f(r) for r in rs], dtype=torch.float32).view(gshape)
    out = {"support_radius": t(lambda r: r.radius), "effective_radius": t(lambda r: r.eff_radius),
           "asym_x": t(lambda r: r.asymmetry[0]), "centers": cs}
    if len(s) >= 2:
        out["asym_y"] = t(lambda r: r.asymmetry[1])
    return out


def over_images(fn, images, centers=None, **kw):
    """How much does the RF depend on the input (image / audio clip)? Per-input results and mean/std of the
    effective radius (support is usually input-independent for local models; effective RF is not).
    `images`: iterable of [1, C, *S] tensors."""
    per = [measure_receptive_field(fn, input=im, centers=centers, **kw) for im in images]
    eff = torch.tensor([[r.eff_radius for r in rs] for rs in per])        # [n_inputs, n_centres]
    sup = torch.tensor([[r.radius for r in rs] for rs in per])
    zero = torch.tensor(0.)
    return {"results": per, "eff_mean": eff.mean(), "eff_std": eff.std() if eff.numel() > 1 else zero,
            "support_mean": sup.mean(), "support_std": sup.std() if sup.numel() > 1 else zero}


over_clips = over_images      # alias for audio
