"""Sweeps: alignment offsets, position maps (border / register-token effects), image dependence."""
from __future__ import annotations

import torch

from .core import measure_receptive_field, make_input


def sweep_centers(out_hw, period: int = 1, margin: int = 0, anchor=None):
    """All (y, x) offsets inside one `period` x `period` block near `anchor` (default grid middle,
    rounded down to a multiple of `period`). Use period = block size on the OUTPUT grid to expose
    block-alignment dependence."""
    H, W = out_hw
    ay, ax = anchor if anchor is not None else (H // 2, W // 2)
    ay -= ay % period
    ax -= ax % period
    assert margin <= ay and margin <= ax and ay + period <= H - margin and ax + period <= W - margin, \
        "grid too small for this margin/period"
    return [(ay + dy, ax + dx) for dy in range(period) for dx in range(period)]


def grid_centers(out_hw, step: int = 1, margin: int = 0):
    """Regular grid of output cells, for position maps."""
    H, W = out_hw
    return [(y, x) for y in range(margin, H - margin, step) for x in range(margin, W - margin, step)]


def position_map(fn, in_shape=None, *, input=None, step: int = 1, margin: int = 0, **kw):
    """RF vs position: returns dict of [gh, gw] tensors (support radius, effective radius, asymmetry).

    Use to see border effects and to spot tokens whose reach is out of line with their neighbours
    (e.g. register-like tokens in models that are meant to be local)."""
    x = make_input(in_shape, input)
    with torch.no_grad():
        h, w = fn(x).shape[-2:]
    cs = grid_centers((h, w), step, margin)
    rs = measure_receptive_field(fn, input=x, centers=cs, **kw)
    gh = len(range(margin, h - margin, step)); gw = len(range(margin, w - margin, step))
    t = lambda f: torch.tensor([f(r) for r in rs], dtype=torch.float32).view(gh, gw)
    return {"support_radius": t(lambda r: r.radius), "effective_radius": t(lambda r: r.eff_radius),
            "asym_x": t(lambda r: r.asymmetry[0]), "asym_y": t(lambda r: r.asymmetry[1]),
            "centers": cs}


def over_images(fn, images, centers=None, **kw):
    """How much does the RF depend on the input image? Returns per-image results and mean/std of the
    effective radius (support is usually input-independent for local models; effective RF is not)."""
    per = [measure_receptive_field(fn, input=im, centers=centers, **kw) for im in images]
    eff = torch.tensor([[r.eff_radius for r in rs] for rs in per])        # [n_images, n_centres]
    sup = torch.tensor([[r.radius for r in rs] for rs in per])
    return {"results": per, "eff_mean": eff.mean(), "eff_std": eff.std() if eff.numel() > 1 else torch.tensor(0.),
            "support_mean": sup.mean(), "support_std": sup.std() if sup.numel() > 1 else torch.tensor(0.)}
