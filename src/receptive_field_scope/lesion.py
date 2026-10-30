"""Latent lesion studies: damage some latent tokens, decode, and measure (a) how well the damaged region is
recovered and (b) how far the damage spreads. Decoder-side counterpart of the encoder receptive-field tools.

Typical use (CompressAI-style, latents [N, C, h, w], decode(latents) -> images [N, 3, H, W]):

    y = model.g_a(images)
    mask = make_mask(y.shape[-2:], "lattice", step=2)           # True = lesioned token
    out = lesion_latents(model.g_s, y, mask, fill="neg", quantize=torch.round)
    prof = damage_profile(out["baseline"], out["lesioned"], mask, original=images)

Why not just zero? A zero latent is not "missing": it is an actual value that some decoders turn into a flat
patch and others into texture. Fills are therefore a family (see FILLS); results that agree across fills are
properties of the decoder, results that vary with the fill are properties of the fill.

All distances between cells are Chebyshev (square rings) in TOKEN cells; multiply by the decoder's cell size
(pixels per token) for pixels.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

FILLS = ("zero", "mean", "neg", "neg0", "noise", "shuffle", "other", "neighbor")


# ----------------------------------------------------------------------------------------------- masks
def make_mask(grid_hw, pattern: str = "lattice", *, step: int = 2, offset=(0, 0), size=(1, 1), at=None,
              frac: float = 0.25, seed: int = 0):
    """Boolean [h, w] token mask, True = lesioned. Patterns: 'lattice' (every `step`-th token in both axes, the
    `y[:, :, ::2, ::2] = 0` experiment), 'block' / 'single' (`size` cells at `at`, default centred), 'random' (`frac`)."""
    h, w = grid_hw
    m = torch.zeros(h, w, dtype=torch.bool)
    if pattern == "lattice":
        m[offset[0]::step, offset[1]::step] = True
    elif pattern in ("block", "single"):
        sz = (1, 1) if pattern == "single" else size
        y0, x0 = at if at is not None else (h // 2 - sz[0] // 2, w // 2 - sz[1] // 2)
        m[y0:y0 + sz[0], x0:x0 + sz[1]] = True
    elif pattern == "random":
        g = torch.Generator().manual_seed(seed)
        m = torch.rand(h, w, generator=g) < frac
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return m


def upsample_mask(mask, out_hw):
    """Token mask [h, w] -> pixel mask [H, W] (nearest)."""
    return F.interpolate(mask[None, None].float(), size=tuple(out_hw), mode="nearest")[0, 0] > 0.5


# ------------------------------------------------------------------------------------------------ fills
def apply_fill(y, mask, fill: str = "neg", *, center=None, other=None, seed: int = 0, stats=None):
    """Return a copy of latents `y` [N, C, h, w] with the tokens where `mask` [h, w] is True replaced.

      zero      0                                    (often NOT neutral: decoders turn it into flat OR textured patches)
      mean      per-channel mean of the image's latents (or `center`)
      neg       2*c - y  : the latent reflected through the centre c (default per-channel mean): same magnitude
                statistics, opposite content. `center="zero"` or fill='neg0' gives the literal -y.
      noise     c + per-channel std * N(0,1)         (marginals matched, content random)
      shuffle   random tokens from the un-lesioned set (marginals matched exactly, content from the wrong place)
      other     latents of `other` [N, C, h, w] at the same positions (another image's content)
      neighbor  mean of un-lesioned 3x3 neighbours (latent-space interpolation baseline)
    center: None (per-channel mean), "zero", or a tensor broadcastable to [N, C, 1, 1] (e.g. the hyperprior means
    of a mean-scale model).
    stats:  latents [M, C, h, w] from which the per-channel mean/std used by 'mean', 'neg' (default centre) and 'noise'
            are computed. Default None = THIS image's own latents, which is a global image statistic: a 'noise' fill drawn
            from it inherits the image's palette. Pass latents of another image (or a dataset) to remove that confound."""
    assert fill in FILLS, f"fill must be one of {FILLS}"
    N, C, h, w = y.shape
    mask = mask.to(y.device)
    ref = y if stats is None else stats.to(y.device)
    mu = ref.mean((0, 2, 3), keepdim=True) if stats is not None else y.mean((2, 3), keepdim=True)
    sd = ref.std((0, 2, 3), keepdim=True) if stats is not None else y.std((2, 3), keepdim=True)
    c = mu if center is None else (torch.zeros_like(mu) if isinstance(center, str) and center == "zero" else center)
    m4 = mask[None, None].expand(N, C, h, w)
    g = torch.Generator().manual_seed(seed)
    if fill == "zero":
        new = torch.zeros_like(y)
    elif fill == "mean":
        new = c.expand_as(y)
    elif fill == "neg":
        new = 2 * c - y
    elif fill == "neg0":
        new = -y
    elif fill == "noise":
        new = c + sd * torch.randn(y.shape, generator=g).to(y.device)
    elif fill == "shuffle":
        keep = (~mask).flatten().nonzero().flatten()
        K = int(mask.sum())
        pick = keep[torch.randint(len(keep), (K,), generator=g)]
        flat = y.flatten(2)
        new = flat.clone()
        new[:, :, mask.flatten()] = flat[:, :, pick.to(y.device)]
        new = new.view_as(y)
    elif fill == "other":
        assert other is not None, "fill='other' needs other=[N,C,h,w]"
        new = other
    else:                                                    # neighbor
        keepf = (~mask).float()[None, None]
        num = F.avg_pool2d(y * keepf, 3, 1, 1) * 9
        den = F.avg_pool2d(keepf, 3, 1, 1) * 9
        new = torch.where(den > 0.5, num / den.clamp_min(1e-6), c.expand_as(y))
    return torch.where(m4, new, y)


def lesion_latents(decode, y, mask, fill: str = "neg", *, quantize=None, center=None, other=None, seed: int = 0, stats=None):
    """Decode the un-lesioned and lesioned latents. quantize (e.g. torch.round) is applied to BOTH, after the fill.
    Compare against `baseline` (quantized, un-lesioned), not the original image: the quantization floor is not damage."""
    q = quantize or (lambda t: t)
    y2 = apply_fill(y, mask, fill, center=center, other=other, seed=seed, stats=stats)
    with torch.no_grad():
        return {"baseline": decode(q(y)), "lesioned": decode(q(y2)), "y_lesioned": y2}


# ------------------------------------------------------------------------------------------- distances
def cheb_distance(target, maxd: int):
    """Chebyshev distance (in cells) from every cell to the nearest True cell of `target` [h, w]; 0 on the target;
    cells farther than `maxd` get maxd + 1."""
    t = target.bool()
    d = torch.full(t.shape, maxd + 1, dtype=torch.long)
    d[t] = 0
    cur = t.clone()
    for k in range(1, maxd + 1):
        dil = F.max_pool2d(cur[None, None].float(), 3, 1, 1)[0, 0] > 0
        d[dil & ~cur] = k
        cur = dil
    return d


def _pix(d_cells, out_hw):
    return F.interpolate(d_cells[None, None].float(), size=tuple(out_hw), mode="nearest")[0, 0].long()


def psnr(a, b, peak: float = 1.0):
    return float(10 * torch.log10(peak ** 2 / (a - b).pow(2).mean().clamp_min(1e-12)))


def damage_profile(baseline, lesioned, mask, *, original=None, maxd: int = 8):
    """Squared pixel change (lesioned vs baseline) as a function of distance from the lesion.

    NOTE: needs a mask with room for several distances/depths. The dense lattice (::2) gives distance 1 and depth 1 only
    (every cell is adjacent to the other kind), so use a block hole (size >= 2*maxdepth) instead; a warning is raised otherwise.
    outside[d]: pixels whose cell is d >= 1 cells (Chebyshev) from the nearest lesioned token  -> collateral damage
    inside[k]:  pixels in lesioned cells that are k >= 1 cells deep (distance to the nearest un-lesioned token)
                                                                                              -> how well holes heal
    With `original`, also reports error versus the original image (baseline floor and lesioned).
    Returns dict of 1D tensors (index = distance - 1) plus scalars: 'far_fraction_2' = share of the total
    outside-damage beyond 2 cells."""
    baseline, lesioned = baseline.detach(), lesioned.detach()
    original = original.detach() if original is not None else None
    N, _, H, W = baseline.shape
    h, w = mask.shape
    dmg = (lesioned - baseline).pow(2).mean(1).mean(0)                      # [H, W]
    d_out = _pix(cheb_distance(mask, maxd), (H, W))                         # 0 on lesion
    d_in = _pix(cheb_distance(~mask, maxd), (H, W))                         # 0 on un-lesioned
    out = {}
    out["outside_dist_cells"] = torch.arange(1, maxd + 1)
    out["outside_damage"] = torch.stack([dmg[d_out == k].mean() if (d_out == k).any() else torch.tensor(float("nan"))
                                         for k in range(1, maxd + 1)])
    out["inside_depth_cells"] = torch.arange(1, maxd + 1)
    out["inside_damage"] = torch.stack([dmg[d_in == k].mean() if (d_in == k).any() else torch.tensor(float("nan"))
                                        for k in range(1, maxd + 1)])
    n_out, n_in = int(torch.isfinite(out["outside_damage"]).sum()), int(torch.isfinite(out["inside_damage"]).sum())
    if n_out < 2 or n_in < 2:
        parts = []
        if n_out < 2:
            parts.append(f"outside-distance has {n_out} populated level(s)")
        if n_in < 2:
            parts.append(f"inside-depth has {n_in} populated level(s)")
        warnings.warn(
            "degenerate mask geometry: " + " and ".join(parts) + ". A dense lattice (::2) puts every cell exactly 1 cell from a "
            "lesion and a 1-wide hole has depth 1 only; use a block hole (make_mask(..., 'block', size=(8, 8))) for distance AND depth "
            "profiles, or a sparse lattice (step >= 6) for distance only.", stacklevel=2)
    outside_all = dmg[d_out >= 1]
    far = dmg[d_out >= 3]
    out["far_fraction_2"] = float(far.sum() / outside_all.sum().clamp_min(1e-20)) if outside_all.numel() else 0.0
    out["damage_inside_mean"] = float(dmg[d_out == 0].mean()) if (d_out == 0).any() else float("nan")
    if original is not None:
        lm = (d_out == 0)
        e_b = (baseline - original).pow(2).mean(1).mean(0)
        e_l = (lesioned - original).pow(2).mean(1).mean(0)
        out["err_inside_baseline"] = float(e_b[lm].mean())
        out["err_inside_lesioned"] = float(e_l[lm].mean())
        out["err_outside_baseline"] = float(e_b[~lm].mean())
        out["err_outside_lesioned"] = float(e_l[~lm].mean())
    return out


def hole_sweep(decode, y, sizes=(1, 2, 3, 4, 6), *, fill: str = "neg", quantize=None, original=None, center=None,
               other=None, seed: int = 0, maxd: int = 6):
    """Heal a square hole of growing size (tokens per side, centred): inside damage/error, and the damage in the
    1-cell ring around it. Answers 'how large a hole can the decoder interpolate?'"""
    rows = []
    for s in sizes:
        mask = make_mask(y.shape[-2:], "block", size=(s, s))
        o = lesion_latents(decode, y, mask, fill, quantize=quantize, center=center, other=other, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # small holes legitimately populate few depth levels
            p = damage_profile(o["baseline"], o["lesioned"], mask, original=original, maxd=maxd)
        rows.append(dict(size=s, inside_damage=p["damage_inside_mean"], ring1_damage=float(p["outside_damage"][0]),
                         ring3_damage=float(p["outside_damage"][2]),
                         err_inside_lesioned=p.get("err_inside_lesioned"), err_inside_baseline=p.get("err_inside_baseline")))
    return rows


@torch.no_grad()
def lesion_importance_map(decode, y, *, fill: str = "neg", near: int = 2, step: int = 1, batch: int = 16, quantize=None,
                          center=None, seed: int = 0):
    """Lesion each token alone (every `step`-th, all channels) and measure the damage it causes (1 image, y [1,C,h,w]).

    Returns {'near_map','far_map','total_map'} over the (strided) token grid: mean squared pixel change within
    `near` cells (Chebyshev) of the token, outside that window, and overall. `far_map` is the register detector: a
    token (often a trivial uniform-background token) whose single-token lesion changes distant pixels carries global
    information. Cost: (h/step)*(w/step) decodes."""
    assert y.shape[0] == 1
    q = quantize or (lambda t: t)
    base = decode(q(y))
    H, W = base.shape[-2:]
    h, w = y.shape[-2:]
    ch, cw = H // h, W // w
    cells = [(i, j) for i in range(0, h, step) for j in range(0, w, step)]
    near_v, far_v = torch.zeros(len(cells)), torch.zeros(len(cells))
    for s in range(0, len(cells), batch):
        chunk = cells[s:s + batch]
        ys = []
        for (i, j) in chunk:
            m = torch.zeros(h, w, dtype=torch.bool)
            m[i, j] = True
            ys.append(apply_fill(y, m, fill, center=center, seed=seed))
        dm = (decode(q(torch.cat(ys))) - base).pow(2).mean(1)                # [B, H, W]
        for k, (i, j) in enumerate(chunk):
            win = torch.zeros(H, W, dtype=torch.bool)
            win[max(0, (i - near) * ch):(i + near + 1) * ch, max(0, (j - near) * cw):(j + near + 1) * cw] = True
            near_v[s + k] = dm[k][win].mean()
            far_v[s + k] = dm[k][~win].mean() if (~win).any() else 0.0
    gh, gw = len(range(0, h, step)), len(range(0, w, step))
    return {"near_map": near_v.view(gh, gw), "far_map": far_v.view(gh, gw), "total_map": (near_v + far_v).view(gh, gw)}


# --------------------------------------------------------------------------------- texture of a filled hole
def harmonic_fill(img, pixel_mask, iters: int = 400):
    """Smooth pixel-space baseline: fill masked pixels by Laplace (harmonic) diffusion from the unmasked ones.
    Introduces no new texture, only smooth continuation; compare a decoder's filled hole against it.
    img [C, H, W] or [N, C, H, W]; pixel_mask [H, W] bool (True = fill)."""
    x = img.clone()
    single = x.dim() == 3
    if single:
        x = x[None]
    m = pixel_mask[None, None].to(x.device)
    k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)[None, None] / 4
    C = x.shape[1]
    x[m.expand_as(x)] = x[(~m).expand_as(x)].mean()
    for _ in range(iters):
        nb = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.repeat(C, 1, 1, 1), groups=C)
        x = torch.where(m, nb, x)
    return x[0] if single else x


def radial_power(patch, bins: int = 16):
    """Radially averaged power spectrum of a patch [C, H, W] / [H, W] (grayscale, Hann-windowed).
    Returns (freq in cycles/pixel, 0..0.5; power)."""
    p = patch.float()
    if p.dim() == 3:
        p = p.mean(0)
    H, W = p.shape
    win = torch.hann_window(H, periodic=False)[:, None] * torch.hann_window(W, periodic=False)[None]
    f = torch.fft.fftshift(torch.fft.fft2((p - p.mean()) * win)).abs().pow(2)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H))[:, None].expand(H, W)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W))[None].expand(H, W)
    r = (fx ** 2 + fy ** 2).sqrt().flatten()
    edges = torch.linspace(0, 0.5, bins + 1)
    idx = torch.bucketize(r, edges[1:-1])
    power = torch.zeros(bins).index_add_(0, idx, f.flatten()) / torch.bincount(idx, minlength=bins).clamp_min(1)
    return (edges[:-1] + edges[1:]) / 2, power


def highfreq_ratio(patch, cutoff: float = 0.15):
    """Fraction of (non-DC) spectral power above `cutoff` cycles/pixel. Texture (hair, fur) > smooth continuation."""
    fr, p = radial_power(patch)
    return float(p[fr >= cutoff].sum() / p.sum().clamp_min(1e-20))


# ------------------------------------------------------------------------- where did a filled colour come from?
def cell_means(img, cell: int):
    """Mean colour per token cell: img [N, C, H, W] -> [N, C, H/cell, W/cell]."""
    return F.avg_pool2d(img.detach(), cell)


def fill_color_attribution(baseline, lesioned, mask, *, ring: int = 1):
    """For every lesioned token cell compare the colour the decoder produced against candidate sources:
        filled   mean colour of the lesioned decode in the cell
        true     what the un-lesioned decode had there
        local    mean colour of the un-lesioned decode in the intact cells within `ring` cells (the immediate surround)
        global   mean colour of the whole un-lesioned decode
    Returns dict with per-cell tensors [K, C] and distances; 'closer_to_local' is the fraction of cells whose filled colour is
    nearer the local surround than the global mean. If the filled colours follow the local surround, a LOCAL mechanism (the
    decoder footprints of neighbouring latents) is enough; if they follow the global mean regardless of surround, look for
    global state (in the decoder, or, as with a 'noise' fill, in the statistics used to build the fill)."""
    H, W = baseline.shape[-2:]
    h, w = mask.shape
    cell = H // h
    cm_b, cm_l = cell_means(baseline, cell)[0], cell_means(lesioned, cell)[0]            # [C, h, w]
    intact = (~mask).float()[None, None]
    k = 2 * ring + 1
    num = F.avg_pool2d(cm_b[None] * intact, k, 1, ring) * k * k
    den = F.avg_pool2d(intact, k, 1, ring) * k * k
    local = (num / den.clamp_min(1e-6))[0]
    glob = baseline.detach().mean((0, 2, 3))
    sel = mask
    f, t, l = cm_l[:, sel].T, cm_b[:, sel].T, local[:, sel].T
    g = glob[None].expand_as(f)
    d = lambda a, b: (a - b).norm(dim=1)
    ok = den[0, 0][sel] > 0.5
    return {"filled": f, "true": t, "local": l, "global": g,
            "dist_to_true": d(f, t), "dist_to_local": d(f, l), "dist_to_global": d(f, g),
            "closer_to_local": float((d(f, l)[ok] < d(f, g)[ok]).float().mean()) if ok.any() else float("nan")}


def palette_distance(colors, palette):
    """Distance from each colour in `colors` [K, C] to the nearest colour in `palette` [P, C]
    (e.g. palette = cell means of the un-lesioned decode). Small = filled colours come from the image's own palette."""
    return torch.cdist(colors, palette).min(1).values
