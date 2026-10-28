"""Gradient-based receptive-field measurement for any tensor->tensor callable, in any number of
spatial dimensions (1D audio waveforms, 2D images/spectrograms, 3D video).

Pick one output position, backprop a random projection of its channels to the input and look at
|grad| over the input grid.

  support    where |grad| > eps * max. For strictly local models (convs, windowed attention) the
             gradient is EXACTLY zero outside the structural support, so this is a tiling-safe bound.
  effective  where the gradient mass sits (Luo et al. 2016); `power` picks L1 magnitude (1) or squared
             energy (2). Meaningful even for models that leak everywhere (global attention, GroupNorm).

Conventions
  * Shapes are channels-first: input [1, C, *S], output [1, C', *s] with len(S) == len(s) == D.
    Models that return [B, T, C] (HuBERT hidden states, ViT tokens) need `channels_last(fn)` (see adapters).
  * Distances are in INPUT-grid cells times `scale` (scale may be a float or one float per axis; e.g.
    scale=1000/16000 reports a 16 kHz waveform in milliseconds, scale=patch_size reports a decoder whose
    input is the token grid in pixels).
  * Distances run from the output cell's centre to the OUTER EDGE of the farthest cell, so
    lo + hi == support width.
  * `centers` are cells on the OUTPUT grid: a tuple (y, x) in 2D, (t,) or a bare int in 1D.
  * Axis order is the tensor's spatial order: 2D = (y, x), so lo/hi[0] are up/down and lo/hi[1] are left/right.
    For D == 1, left/right alias lo[0]/hi[0] (earlier/later in time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import torch


@dataclass
class RFResult:
    center: tuple                   # output-grid cell
    ref: tuple                      # centre of that output cell, in input coordinates (cells)
    lo: tuple                       # support extent toward smaller indices, per spatial axis (x scale)
    hi: tuple                       # support extent toward larger indices, per spatial axis
    eff_lo: tuple                   # central-`energy` interval per axis
    eff_hi: tuple
    eff_radius: float               # radial (L2) radius holding `energy` of the mass
    touches_border: bool            # support reaches the input edge on some axis -> lower bounds
    is_global: bool                 # support spans >= global_frac of the input on some axis
    grad_map: torch.Tensor | None = field(default=None, repr=False)   # [*S] sensitivity map

    # ---- 2D / 1D convenience views (axes are (y, x) in 2D; time in 1D) ----
    @property
    def left(self) -> float:        # toward smaller x (or earlier time)
        return self.lo[-1]

    @property
    def right(self) -> float:
        return self.hi[-1]

    @property
    def up(self) -> float:
        return self.lo[-2]

    @property
    def down(self) -> float:
        return self.hi[-2]

    @property
    def eff_left(self) -> float:
        return self.eff_lo[-1]

    @property
    def eff_right(self) -> float:
        return self.eff_hi[-1]

    @property
    def eff_up(self) -> float:
        return self.eff_lo[-2]

    @property
    def eff_down(self) -> float:
        return self.eff_hi[-2]

    @property
    def radius(self) -> float:      # worst-case one-sided support reach over all axes
        return max(max(self.lo), max(self.hi))

    @property
    def width(self) -> tuple:       # support width per axis, last axis (x / time) first: (x, y, ...)
        return tuple(l + h for l, h in zip(self.lo[::-1], self.hi[::-1]))

    @property
    def asymmetry(self) -> tuple:   # |lo-hi| per axis, last axis first: nonzero => alignment dependence
        return tuple(abs(l - h) for l, h in zip(self.lo[::-1], self.hi[::-1]))

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "grad_map"}
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


def _as_tuple(v, d):
    return tuple(float(a) for a in v) if hasattr(v, "__len__") else (float(v),) * d


def result_from_map(m, center, ref, *, scale=1.0, eps=1e-6, energy=0.95, power=1.0, global_frac=0.9, keep_map=False):
    """Turn a non-negative [*S] sensitivity map into an RFResult (shared by gradient/perturbation).

    power: effective-RF mass is m**power. 1 = L1 gradient magnitude (Luo et al. style; diffuse tails add up);
    2 = squared gradient / 'energy' (suppresses diffuse tails). The support is unaffected by `power`."""
    m = m.double().cpu()
    shape = tuple(m.shape)
    D = len(shape)
    scale = _as_tuple(scale, D)
    ref = tuple(float(r) for r in ref)
    peak = m.max()
    if peak == 0:
        raise RuntimeError(f"zero sensitivity at centre {tuple(center)}: dead paths (zero-init gates? use activate_gates)")
    sup = m > eps * peak
    e = m ** power
    e = e / e.sum()
    lo, hi, elo, ehi, touches, glob = [], [], [], [], False, False
    for ax in range(D):
        other = tuple(a for a in range(D) if a != ax)
        a0, a1 = _extent(sup.any(dim=other) if other else sup)
        b0, b1 = _central_interval(e.sum(dim=other) if other else e, energy)
        lo.append((ref[ax] - a0) * scale[ax]); hi.append((a1 - ref[ax]) * scale[ax])
        elo.append((ref[ax] - b0) * scale[ax]); ehi.append((b1 - ref[ax]) * scale[ax])
        touches |= bool(a0 == 0 or a1 == shape[ax])
        glob |= bool((a1 - a0) >= global_frac * shape[ax])
    grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) + 0.5 - ref[i] for i, n in enumerate(shape)],
                           indexing="ij")
    dist = sum((g * scale[i]) ** 2 for i, g in enumerate(grids)).sqrt().flatten()
    order = torch.argsort(dist)
    cum = torch.cumsum(e.flatten()[order], 0)
    eff_r = float(dist[order][min(int((cum < energy).sum()), len(cum) - 1)])
    return RFResult(center=tuple(center), ref=ref, lo=tuple(lo), hi=tuple(hi), eff_lo=tuple(elo), eff_hi=tuple(ehi),
                    eff_radius=eff_r, touches_border=touches, is_global=glob,
                    grad_map=m.float() if keep_map else None)


def make_input(in_shape=None, input=None, *, seed=0, dtype=torch.float32, device=None):
    """Return a [1,C,*S] tensor: `input` if given (batch dim added if it has C,*S only), else random normal."""
    if input is not None:
        x = input.detach().clone()
        return x
    shape = tuple(int(v) for v in in_shape)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype)
    return x.to(device) if device is not None else x


def norm_centers(centers, out_shape):
    """Default to the middle of the output grid; ints become 1-tuples."""
    if centers is None:
        return [tuple(n // 2 for n in out_shape)]
    return [(c,) if isinstance(c, int) else tuple(int(v) for v in c) for c in centers]


def measure_receptive_field(
    fn,
    in_shape=None,
    centers=None,
    *,
    input=None,
    n_proj: int = 4,
    eps: float = 1e-6,
    energy: float = 0.95,
    power: float = 1.0,
    scale=1.0,
    global_frac: float = 0.9,
    keep_maps: bool = False,
    seed: int = 0,
):
    """Measure the receptive field of `fn` ([1,C,*S] -> [1,C',*s]) at each output-grid centre.

    Give `in_shape` (random-normal probe; fine for structural support of local models) or `input`
    (a real image / clip: use this for trained, data-dependent models such as attention). `input` must
    already have the batch dimension: [1, C, *S].
    `fn` should be in eval() mode. Returns list[RFResult] (one per centre; default: grid middle).
    """
    x = make_input(in_shape, input, seed=seed)
    x.requires_grad_(True)
    S = tuple(x.shape[2:])
    D = len(S)
    y = fn(x)
    assert y.dim() == 2 + D and y.shape[0] == 1, \
        f"fn must return [1,C,{','.join('s' + str(i) for i in range(D))}], got {tuple(y.shape)} " \
        f"(channels-last outputs: wrap with channels_last(fn))"
    s = tuple(y.shape[2:])
    ratio = tuple(S[i] / s[i] for i in range(D))
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for c in norm_centers(centers, s):
        vec = y[(0, slice(None), *c)]
        acc = torch.zeros(S, dtype=torch.float64)
        for _ in range(n_proj):
            proj = torch.randn(vec.shape, generator=g, dtype=vec.dtype).to(vec.device)
            (grad,) = torch.autograd.grad((vec * proj).sum(), x, retain_graph=True)
            acc += grad[0].abs().sum(0).double().cpu()
        ref = tuple((c[i] + 0.5) * ratio[i] for i in range(D))
        out.append(result_from_map(acc / n_proj, c, ref, scale=scale, eps=eps, energy=energy,
                                   power=power, global_frac=global_frac, keep_map=keep_maps))
    return out


def summarize(results, label: str = ""):
    """Print worst case / spread over the probed centres (per axis: lo/hi)."""
    D = len(results[0].lo)
    names = {1: ["time"], 2: ["y", "x"], 3: ["z", "y", "x"]}.get(D, [f"ax{i}" for i in range(D)])
    print(f"== {label} ({len(results)} centres) ==")
    rows = [("support radius", lambda r: r.radius)]
    for i, n in enumerate(names):
        rows += [(f"{n} lo", lambda r, i=i: r.lo[i]), (f"{n} hi", lambda r, i=i: r.hi[i])]
    rows.append(("effective radius", lambda r: r.eff_radius))
    for name, f in rows:
        v = [f(r) for r in results]
        print(f"  {name:17s} min {min(v):9.1f}   max {max(v):9.1f}")
    if any(r.touches_border for r in results):
        print("  ! some supports touch the input border -> lower bounds")
    if any(r.is_global for r in results):
        print("  ! global support on some centres")
