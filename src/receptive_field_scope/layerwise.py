"""Per-layer receptive fields: where does the reach come from?

One forward pass with hooks. For every selected module we take the output cell at the middle of that
module's output grid and measure its receptive field back to the MODEL input. Differences between
consecutive rows are the per-layer 'hops'. Works for any number of spatial dims."""
from __future__ import annotations

import torch
import torch.nn as nn

from .core import make_input, result_from_map


def layerwise_receptive_field(model: nn.Module, in_shape=None, *, input=None, names=None, predicate=None,
                              layout="cf", scale=1.0, n_proj: int = 4, eps: float = 1e-6, energy: float = 0.95,
                              power: float = 1.0, seed: int = 0, call=None):
    """Returns list of dicts: name, out_shape, lo, hi, radius, eff_radius, hop (plus left/right in 1D/2D).

    names:     explicit module names (as in model.named_modules()); default = direct children.
    predicate: alternative filter f(name, module) -> bool over ALL named modules.
    layout:    'cf' (outputs are [1, C, *s]), 'cl' (outputs are [1, *s, C], e.g. transformer blocks), or a dict
               {module_name: 'cf'|'cl'} (anything unlisted is 'cf'). Outputs with other ranks are skipped.
    call:      optional f(model, x) -> anything; use for models that need a custom call (e.g. waveform [B,T]).
    'hop' is the change in radius vs the previous reported row (assumes `names` are in forward order)."""
    named = dict(model.named_modules())
    if names is None:
        names = [n for n, m in model.named_modules() if (predicate(n, m) if predicate else (n and "." not in n))]
    captured, hooks = {}, []
    for n in names:
        def hook(_m, _i, out, n=n):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                captured[n] = o
        hooks.append(named[n].register_forward_hook(hook))
    x = make_input(in_shape, input, seed=seed).requires_grad_(True)
    S = tuple(x.shape[2:])
    D = len(S)
    try:
        (call or (lambda m, t: m(t)))(model, x)
    finally:
        for h in hooks:
            h.remove()
    g = torch.Generator().manual_seed(seed + 1)
    rows, prev = [], 0.0
    for n in names:
        o = captured.get(n)
        if o is None or o.dim() != 2 + D or o.shape[0] != 1:
            continue
        if (layout.get(n, "cf") if isinstance(layout, dict) else layout) == "cl":
            o = o.movedim(-1, 1)
        s = tuple(o.shape[2:])
        c = tuple(v // 2 for v in s)
        vec = o[(0, slice(None), *c)]
        acc = torch.zeros(S, dtype=torch.float64)
        for _ in range(n_proj):
            proj = torch.randn(vec.shape, generator=g, dtype=vec.dtype).to(vec.device)
            (grad,) = torch.autograd.grad((vec * proj).sum(), x, retain_graph=True, allow_unused=True)
            if grad is not None:
                acc += grad[0].abs().sum(0).double().cpu()
        if acc.max() == 0:
            rows.append(dict(name=n, out_shape=s, radius=0.0, note="no gradient path"))
            continue
        ref = tuple((c[i] + 0.5) * S[i] / s[i] for i in range(D))
        r = result_from_map(acc / n_proj, c, ref, scale=scale, eps=eps, energy=energy, power=power)
        row = dict(name=n, out_shape=s, lo=r.lo, hi=r.hi, radius=r.radius, eff_radius=r.eff_radius, hop=r.radius - prev)
        row.update(left=r.left, right=r.right)
        rows.append(row)
        prev = r.radius
    return rows


def print_layerwise(rows):
    print(f"{'module':28s} {'out':>12s} {'lo':>9s} {'hi':>9s} {'radius':>8s} {'hop':>7s} {'eff':>7s}")
    for r in rows:
        if "lo" not in r:
            print(f"{r['name']:28s} {str(r['out_shape']):>12s}   (no gradient path)")
            continue
        lo = ",".join(f"{v:.0f}" for v in r["lo"]); hi = ",".join(f"{v:.0f}" for v in r["hi"])
        print(f"{r['name']:28s} {str(r['out_shape']):>12s} {lo:>9s} {hi:>9s} {r['radius']:8.1f} {r['hop']:7.1f} {r['eff_radius']:7.1f}")
