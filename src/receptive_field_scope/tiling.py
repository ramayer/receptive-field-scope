"""Does a given tile margin actually reproduce the full-input output? (the practical RF question)
Works for 2D images and 1D audio (chunked inference) and higher."""
from __future__ import annotations

from itertools import product

import torch


def _tup(v, d):
    return tuple(v) if hasattr(v, "__len__") else (v,) * d


@torch.no_grad()
def tiled_apply(fn, x, tile, margin, ratio=None):
    """Apply `fn` tile-by-tile with `margin` input cells of context, then stitch.

    x: [1, C, *S]. tile/margin: int or one per axis. ratio: output cells per input cell (per axis or scalar;
    default: inferred from a full run -- costs one extra forward). tile*ratio and margin*ratio must be whole;
    each S must be a multiple of tile. Windows are clipped at the border, so borders see the same padding as
    a full run. To compare against a block-aligned model keep tile/margin multiples of the block size."""
    S = tuple(x.shape[2:])
    D = len(S)
    tile, margin = _tup(tile, D), _tup(margin, D)
    if ratio is None:
        ratio = tuple(a / b for a, b in zip(fn(x).shape[2:], S))
    ratio = _tup(ratio, D)
    assert all(n % t == 0 for n, t in zip(S, tile)), "input size must be a multiple of tile"
    assert all(float(t * r).is_integer() and float(m * r).is_integer() for t, m, r in zip(tile, margin, ratio)), \
        "tile and margin must map to whole output cells"
    out = None
    for starts in product(*[range(0, n, t) for n, t in zip(S, tile)]):
        w0 = [max(0, a - m) for a, m in zip(starts, margin)]
        w1 = [min(n, a + t + m) for a, t, m, n in zip(starts, tile, margin, S)]
        o = fn(x[(slice(None), slice(None), *[slice(a, b) for a, b in zip(w0, w1)])])
        if out is None:
            out = torch.zeros(o.shape[0], o.shape[1], *[int(n * r) for n, r in zip(S, ratio)],
                              dtype=o.dtype, device=o.device)
        src = [slice(int((a - b) * r), int((a - b) * r) + int(t * r)) for a, b, t, r in zip(starts, w0, tile, ratio)]
        dst = [slice(int(a * r), int(a * r) + int(t * r)) for a, t, r in zip(starts, tile, ratio)]
        out[(slice(None), slice(None), *dst)] = o[(slice(None), slice(None), *src)]
    return out


@torch.no_grad()
def margin_sweep(fn, inputs, tile, margins, *, atol: float = 1e-4, discrete: bool = False):
    """For each margin: fraction of output elements that differ from the full run, and max abs error.

    discrete=True compares exactly (e.g. quantized bits); otherwise elements differ if |a-b| > atol.
    Returns list of dicts, one per margin, averaged over inputs. `margins` entries may be ints or per-axis tuples."""
    rows = []
    fulls = [fn(im) for im in inputs]
    ratio = tuple(a / b for a, b in zip(fulls[0].shape[2:], inputs[0].shape[2:]))
    for m in margins:
        frac, mx = [], []
        for im, full in zip(inputs, fulls):
            t = tiled_apply(fn, im, tile, m, ratio)
            diff = (t != full) if discrete else ((t - full).abs() > atol)
            frac.append(diff.float().mean().item())
            mx.append((t - full).abs().max().item())
        rows.append(dict(margin=m, frac_differing=sum(frac) / len(frac), max_abs_err=max(mx)))
    return rows


def print_margin_sweep(rows):
    print(f"{'margin':>10s} {'frac differing':>15s} {'max |err|':>12s}")
    for r in rows:
        print(f"{str(r['margin']):>10s} {r['frac_differing']:15.6f} {r['max_abs_err']:12.3e}")
