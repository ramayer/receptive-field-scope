"""Gradient-based receptive-field measurement for any tensor->tensor callable.

Pick one output position, backprop a random projection of its channels to the input and look at
|grad| over the input grid.

  support    where |grad| > eps * max. For strictly local models (convs, windowed attention) the
             gradient is EXACTLY zero outside the structural support, so this is a tiling-safe bound.
  effective  where the gradient *energy* sits (Luo et al. 2016). Meaningful even for models that
             leak everywhere (global attention, GroupNorm).

Conventions
  * Distances are in INPUT-grid cells times `scale` (use scale=patch_size for a decoder whose input
    is the token grid, to report pixels).
  * Distances run from the output cell's centre to the OUTER EDGE of the farthest cell, so
    left + right == support width.
  * `centers` are (y, x) cells on the OUTPUT grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch


@dataclass
class RFResult:
    center: tuple
    ref: tuple                      # centre of that output cell, in input coordinates
    left: float
    right: float
    up: float
    down: float
    eff_left: float
    eff_right: float
    eff_up: float
    eff_down: float
    eff_radius: float               # radial radius holding `energy` of the mass
    touches_border: bool            # support reaches the input edge -> numbers are lower bounds
    is_global: bool                 # support spans >= global_frac of the input on either axis
    grad_map: torch.Tensor | None = field(default=None, repr=False)   # [H, W] sensitivity map

    @property
    def radius(self) -> float:      # worst-case one-sided support reach
        return max(self.left, self.right, self.up, self.down)

    @property
    def width(self) -> tuple:       # (horizontal, vertical) support width
        return self.left + self.right, self.up + self.down

    @property
    def asymmetry(self) -> tuple:   # (|left-right|, |up-down|): nonzero => alignment dependence
        return abs(self.left - self.right), abs(self.up - self.down)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("grad_map", None)
        d["radius"] = self.radius
        return d


def _extent(mask_1d):
    idx = torch.nonzero(mask_1d).flatten()
    return int(idx.min()), int(idx.max()) + 1          # [lo, hi) in cells


def _central_interval(profile, energy):
    c = torch.cumsum(profile, 0) / profile.sum()
    tail = (1 - energy) / 2
    lo = int((c < tail).sum())
    hi = min(int((c < 1 - tail).sum()) + 1, len(profile))
    return lo, hi


def result_from_map(m, center, ref, *, scale=1.0, eps=1e-6, energy=0.95, global_frac=0.9, keep_map=False):
    """Turn a non-negative [H, W] sensitivity map into an RFResult (shared by gradient/perturbation)."""
    m = m.double().cpu()
    H, W = m.shape
    peak = m.max()
    if peak == 0:
        raise RuntimeError(f"zero sensitivity at centre {center}: dead paths (zero-init gates? use activate_gates)")
    ry, rx = ref
    sup = m > eps * peak
    x0, x1 = _extent(sup.any(0))
    y0, y1 = _extent(sup.any(1))
    e = m / m.sum()
    ex0, ex1 = _central_interval(e.sum(0), energy)
    ey0, ey1 = _central_interval(e.sum(1), energy)
    yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    dist = ((yy - ry) ** 2 + (xx - rx) ** 2).sqrt().flatten()
    order = torch.argsort(dist)
    cum = torch.cumsum(e.flatten()[order], 0)
    eff_r = float(dist[order][min(int((cum < energy).sum()), len(cum) - 1)])
    return RFResult(
        center=tuple(center), ref=(float(ry), float(rx)),
        left=(rx - x0) * scale, right=(x1 - rx) * scale, up=(ry - y0) * scale, down=(y1 - ry) * scale,
        eff_left=(rx - ex0) * scale, eff_right=(ex1 - rx) * scale,
        eff_up=(ry - ey0) * scale, eff_down=(ey1 - ry) * scale,
        eff_radius=eff_r * scale,
        touches_border=bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H),
        is_global=bool((x1 - x0) >= global_frac * W or (y1 - y0) >= global_frac * H),
        grad_map=m.float() if keep_map else None,
    )


def make_input(in_shape=None, input=None, *, seed=0, dtype=torch.float32, device=None):
    """Return a [1,C,H,W] tensor: `input` if given (batch dim added if missing), else random normal."""
    if input is not None:
        x = input.detach().clone()
        return x[None] if x.dim() == 3 else x
    shape = tuple(int(v) for v in in_shape)
    shape = shape if len(shape) == 4 else (1, *shape)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype)
    return x.to(device) if device is not None else x


def measure_receptive_field(
    fn,
    in_shape=None,
    centers=None,
    *,
    input=None,
    n_proj: int = 4,
    eps: float = 1e-6,
    energy: float = 0.95,
    scale: float = 1.0,
    global_frac: float = 0.9,
    keep_maps: bool = False,
    seed: int = 0,
):
    """Measure the receptive field of `fn` ([1,C,H,W] -> [1,C',h,w]) at each output-grid centre.

    Give `in_shape` (random-normal probe; fine for structural support of local models) or `input`
    (a real image: use this for trained, data-dependent models such as attention).
    `fn` should be in eval() mode. Returns list[RFResult] (one per centre; default: grid middle).
    """
    x = make_input(in_shape, input, seed=seed)
    x.requires_grad_(True)
    H, W = x.shape[-2:]
    y = fn(x)
    assert y.dim() == 4 and y.shape[0] == 1, f"fn must return [1,C,h,w], got {tuple(y.shape)}"
    h, w = y.shape[-2:]
    sy, sx = H / h, W / w
    centers = centers or [(h // 2, w // 2)]
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for (cy, cx) in centers:
        vec = y[0, :, cy, cx]
        acc = torch.zeros(H, W, dtype=torch.float64)
        for _ in range(n_proj):
            proj = torch.randn(vec.shape, generator=g, dtype=vec.dtype).to(vec.device)
            (grad,) = torch.autograd.grad((vec * proj).sum(), x, retain_graph=True)
            acc += grad[0].abs().sum(0).double().cpu()
        out.append(result_from_map(acc / n_proj, (cy, cx), ((cy + 0.5) * sy, (cx + 0.5) * sx),
                                   scale=scale, eps=eps, energy=energy, global_frac=global_frac,
                                   keep_map=keep_maps))
    return out


def summarize(results, label: str = ""):
    """Print worst case / spread over the probed centres."""
    rows = [("support radius", lambda r: r.radius), ("left", lambda r: r.left), ("right", lambda r: r.right),
            ("up", lambda r: r.up), ("down", lambda r: r.down), ("effective radius", lambda r: r.eff_radius)]
    print(f"== {label} ({len(results)} centres) ==")
    for name, f in rows:
        v = [f(r) for r in results]
        print(f"  {name:17s} min {min(v):8.1f}   max {max(v):8.1f}")
    if any(r.touches_border for r in results):
        print("  ! some supports touch the input border -> lower bounds")
    if any(r.is_global for r in results):
        print("  ! global support on some centres")
