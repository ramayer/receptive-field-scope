"""Does a given tile margin actually reproduce the full-image output? (the practical RF question)"""
from __future__ import annotations

import torch


@torch.no_grad()
def tiled_apply(fn, image, tile: int, margin: int, ratio: float):
    """Apply `fn` tile-by-tile with `margin` input pixels of context, then stitch.

    ratio = output cells per input pixel (1/8 for a patch-8 encoder, 8 for a patch-8 decoder).
    tile and margin must map to whole output cells (tile*ratio, margin*ratio integers); image dims must be
    multiples of tile. Windows are clipped at the image border, so borders see the same padding as a
    full-image run. To compare against a block-aligned model keep tile/margin multiples of the block size."""
    _, _, H, W = image.shape
    assert H % tile == 0 and W % tile == 0, "image size must be a multiple of tile"
    assert float(tile * ratio).is_integer() and float(margin * ratio).is_integer()
    out = None
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            y0, y1 = max(0, ty - margin), min(H, ty + tile + margin)
            x0, x1 = max(0, tx - margin), min(W, tx + tile + margin)
            o = fn(image[:, :, y0:y1, x0:x1])
            if out is None:
                out = torch.zeros(o.shape[0], o.shape[1], int(H * ratio), int(W * ratio), dtype=o.dtype, device=o.device)
            cy0, cx0 = int((ty - y0) * ratio), int((tx - x0) * ratio)
            ch, cw = int(tile * ratio), int(tile * ratio)
            out[:, :, int(ty * ratio):int(ty * ratio) + ch, int(tx * ratio):int(tx * ratio) + cw] = \
                o[:, :, cy0:cy0 + ch, cx0:cx0 + cw]
    return out


@torch.no_grad()
def margin_sweep(fn, images, tile: int, margins, *, atol: float = 1e-4, discrete: bool = False):
    """For each margin: fraction of output elements that differ from the full-image run, and max abs error.

    discrete=True compares exactly (e.g. quantized bits); otherwise elements differ if |a-b| > atol.
    Returns list of dicts, one per margin, averaged over images."""
    rows = []
    fulls = [fn(im) for im in images]
    ratio = fulls[0].shape[-1] / images[0].shape[-1]
    for m in margins:
        frac, mx = [], []
        for im, full in zip(images, fulls):
            t = tiled_apply(fn, im, tile, m, ratio)
            diff = (t != full) if discrete else ((t - full).abs() > atol)
            frac.append(diff.float().mean().item())
            mx.append((t - full).abs().max().item())
        rows.append(dict(margin=m, frac_differing=sum(frac) / len(frac), max_abs_err=max(mx)))
    return rows


def print_margin_sweep(rows):
    print(f"{'margin(px)':>10s} {'frac differing':>15s} {'max |err|':>12s}")
    for r in rows:
        print(f"{r['margin']:10d} {r['frac_differing']:15.6f} {r['max_abs_err']:12.3e}")
