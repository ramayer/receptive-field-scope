"""Per-layer receptive fields: where does the reach come from?

One forward pass with hooks. For every selected module we take the output cell at the middle of that
module's output grid and measure its receptive field back to the MODEL input. Differences between
consecutive rows are the per-layer 'hops'."""
from __future__ import annotations

import torch
import torch.nn as nn

from .core import make_input, result_from_map


def layerwise_receptive_field(model: nn.Module, in_shape=None, *, input=None, names=None, predicate=None,
                              n_proj: int = 4, eps: float = 1e-6, energy: float = 0.95, seed: int = 0):
    """Returns list of dicts: name, out_hw, left/right/up/down (input pixels), radius, eff_radius, hop.

    names:     explicit module names (as in model.named_modules()); default = direct children.
    predicate: alternative filter f(name, module) -> bool over ALL named modules.
    Only modules whose output is a 4D tensor are reported; 'hop' is the change in radius vs the
    previous reported row (assumes `names` are in forward order)."""
    named = dict(model.named_modules())
    if names is None:
        names = [n for n, m in model.named_modules() if (predicate(n, m) if predicate else (n and "." not in n))]
    captured = {}
    hooks = []
    for n in names:
        def hook(_m, _i, out, n=n):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o) and o.dim() == 4:
                captured[n] = o
        hooks.append(named[n].register_forward_hook(hook))
    x = make_input(in_shape, input, seed=seed).requires_grad_(True)
    H, W = x.shape[-2:]
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    g = torch.Generator().manual_seed(seed + 1)
    rows, prev = [], 0.0
    for n in names:
        if n not in captured:
            continue
        o = captured[n]
        h, w = o.shape[-2:]
        cy, cx = h // 2, w // 2
        vec = o[0, :, cy, cx]
        acc = torch.zeros(H, W, dtype=torch.float64)
        for _ in range(n_proj):
            proj = torch.randn(vec.shape, generator=g, dtype=vec.dtype).to(vec.device)
            (grad,) = torch.autograd.grad((vec * proj).sum(), x, retain_graph=True, allow_unused=True)
            if grad is not None:
                acc += grad[0].abs().sum(0).double().cpu()
        if acc.max() == 0:
            rows.append(dict(name=n, out_hw=(h, w), radius=0.0, note="no gradient path"))
            continue
        r = result_from_map(acc / n_proj, (cy, cx), ((cy + 0.5) * H / h, (cx + 0.5) * W / w), eps=eps, energy=energy)
        rows.append(dict(name=n, out_hw=(h, w), left=r.left, right=r.right, up=r.up, down=r.down,
                         radius=r.radius, eff_radius=r.eff_radius, hop=r.radius - prev))
        prev = r.radius
    return rows


def print_layerwise(rows):
    print(f"{'module':28s} {'out':>9s} {'left':>7s} {'right':>7s} {'radius':>7s} {'hop':>6s} {'eff':>6s}")
    for r in rows:
        if "left" not in r:
            print(f"{r['name']:28s} {str(r['out_hw']):>9s}   (no gradient path)")
            continue
        print(f"{r['name']:28s} {str(r['out_hw']):>9s} {r['left']:7.1f} {r['right']:7.1f} {r['radius']:7.1f} "
              f"{r['hop']:6.1f} {r['eff_radius']:6.1f}")
