"""Matplotlib helpers (matplotlib is imported lazily; install the `examples` extra).

All helpers need results measured with `keep_maps=True` (so `result.grad_map` exists) and `scale=1`
(distances in input cells), because they draw in input coordinates."""
from __future__ import annotations

import numpy as np
import torch


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _log_norm(m, floor_db: float = -6.0):
    """Map a non-negative sensitivity map to [0, 1] on a log10 scale spanning `floor_db` decades below the peak."""
    m = np.asarray(m, dtype=np.float64)
    r = np.log10(np.maximum(m / m.max(), 10.0 ** floor_db))
    return (r - floor_db) / (-floor_db)


def _to_hwc(image):
    a = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def overlay_heatmap(image, result, ax=None, *, log=True, floor_db=-6.0, alpha=0.65, cmap="inferno",
                    show_support=True, show_effective=True, show_center=True, title=None):
    """Overlay a receptive-field sensitivity map on an image (2D).

    The heat is |d output / d input| on a log scale by default (the useful dynamic range is many decades).
    Dashed box = support (where gradient is nonzero above eps); solid circle = radius holding the
    `energy` fraction of the gradient mass (effective receptive field); cross = the probed output cell's centre."""
    plt = _plt()
    from matplotlib.patches import Circle, Rectangle
    if result.grad_map is None:
        raise ValueError("result has no grad_map; call measure_receptive_field(..., keep_maps=True)")
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    img = _to_hwc(image)
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
    m = result.grad_map.numpy()
    heat = _log_norm(m, floor_db) if log else m / m.max()
    ax.imshow(heat, cmap=cmap, alpha=alpha, vmin=0, vmax=1, interpolation="nearest")
    ry, rx = result.ref
    if show_support:
        ax.add_patch(Rectangle((rx - result.lo[-1], ry - result.lo[-2]), result.lo[-1] + result.hi[-1],
                               result.lo[-2] + result.hi[-2], fill=False, ls="--", lw=1.5, ec="cyan"))
    if show_effective:
        ax.add_patch(Circle((rx, ry), result.eff_radius, fill=False, lw=2, ec="white"))
    if show_center:
        ax.plot([rx], [ry], "+", color="white", ms=12, mew=2)
    ax.set_xlim(0, m.shape[1]); ax.set_ylim(m.shape[0], 0)
    ax.set_title(title or f"support {result.width[0]:.0f}x{result.width[1]:.0f}px (dashed), "
                          f"effective radius {result.eff_radius:.0f}px (circle)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_waveform_rf(waveform, result, sample_rate: int, ax=None, *, floor_db=-6.0, show_support=True,
                     show_effective=True, xlim=None):
    """Plot a waveform (1D) with the receptive field of one frame shaded behind it (time in seconds).

    Shading = log-scaled sensitivity. Dashed lines = support; solid lines = the central `energy` interval.
    xlim=(t0, t1) in seconds zooms in (a 25 ms receptive field is a thin sliver of a 1 s clip)."""
    plt = _plt()
    if result.grad_map is None:
        raise ValueError("result has no grad_map; call measure_receptive_field(..., keep_maps=True)")
    ax = ax or plt.subplots(figsize=(9, 2.8))[1]
    w = waveform.detach().cpu().numpy().reshape(-1) if torch.is_tensor(waveform) else np.asarray(waveform).reshape(-1)
    t = np.arange(len(w)) / sample_rate
    heat = _log_norm(result.grad_map.numpy(), floor_db)
    ax.imshow(heat[None], aspect="auto", cmap="magma", alpha=0.9, extent=[0, len(w) / sample_rate, w.min(), w.max()])
    ax.plot(t, w, lw=0.7, color="white")
    r = result.ref[0]
    if show_support:
        for v in (r - result.lo[0], r + result.hi[0]):
            ax.axvline(v / sample_rate, ls="--", color="cyan", lw=1.2)
    if show_effective:
        for v in (r - result.eff_lo[0], r + result.eff_hi[0]):
            ax.axvline(v / sample_rate, color="white", lw=1.8)
    ax.axvline(r / sample_rate, color="red", lw=0.8)
    ax.set_xlabel("time (s)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(f"support {sum(result.width) * 1000 / sample_rate:.0f} ms (dashed), effective "
                 f"{(result.eff_lo[0] + result.eff_hi[0]) * 1000 / sample_rate:.0f} ms (solid)", fontsize=9)
    return ax


def plot_position_map(pm, key="effective_radius", ax=None, *, title=None):
    """Heatmap of a position_map() quantity over the token grid (2D grids only)."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    im = ax.imshow(pm[key].numpy(), cmap="viridis")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title or key)
    return ax
