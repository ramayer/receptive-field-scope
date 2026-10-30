"""Cut tests: remove or replace part of the input and ask how far into the retained region the output changes.

Answers two practical questions about tiling and about hidden global state:
  * "How much padding do I need beyond a tile edge / corner?"  -> the distance (token cells) beyond which the error is ~0
    ('margin_needed').
  * "If I chop off a corner or band, did I remove a register (a token carrying non-local information)?"  -> error that persists
    far from the cut.

Two modes isolate the cause:
  crop     the band is removed (smaller input): content is gone AND the border moved (zero padding / absolute positions change).
  replace  the band keeps its place but its content is replaced (mean / noise / another input): geometry and padding are unchanged,
           only content changes.
  far error under replace            -> content-borne long-range dependence (register-like);
  far error under crop, not replace  -> border / position effect (padding, absolute positional encodings).

Works for any fn [1, C, H, W] -> [1, C', h, w]: ENCODERS (pass the image) and DECODERS (pass the latents). All cuts and distances are in
TOKEN cells: the grid with fewer elements (the output grid of an encoder, the input grid of a decoder). Each token cell must correspond to a
whole number of pixels."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _edge_distance(h, w):
    i = torch.arange(h)[:, None]
    j = torch.arange(w)[None]
    return torch.minimum(torch.minimum(i, j), torch.minimum(h - 1 - i, w - 1 - j))


def _up(t, k):
    return t.repeat_interleave(k, 0).repeat_interleave(k, 1) if k > 1 else t


@torch.no_grad()
def cut_profile(fn, x, cut=None, *, mode: str = "crop", replace_mask=None, replacement="mean", other=None, maxd: int = 16, seed: int = 0):
    """Relative output change in the retained region versus distance (token cells) from the cut.

    cut: (top, bottom, left, right) token cells removed from each side (e.g. (0, 0, 0, k) = a right band; (k, 0, k, 0) = the top and
         left bands). Alternatively replace_mask [gh, gw] bool on the token grid (True = replace), mode='replace'.
    replacement: 'mean' (per-channel mean of x), 'zero', 'noise', or 'other' (content from `other`, same shape as x).
    Returns dict: dist (0..maxd), rel_error (mean squared change / mean squared magnitude of the full output; nan where no pixels),
    count (pixels per bin), mode. For 'replace', distance 0 is the replaced region itself and is reported as nan."""
    from .lesion import cheb_distance
    y_full = fn(x)
    down = y_full.shape[-2] <= x.shape[-2]
    if down:
        k_in, k_out = x.shape[-2] // y_full.shape[-2], 1
        gh, gw = y_full.shape[-2:]
    else:
        k_in, k_out = 1, y_full.shape[-2] // x.shape[-2]
        gh, gw = x.shape[-2:]
    assert (x.shape[-2] == gh * k_in and x.shape[-1] == gw * k_in and y_full.shape[-2] == gh * k_out and y_full.shape[-1] == gw * k_out), \
        "each token cell must map to a whole number of pixels"
    scale = y_full.pow(2).mean().clamp_min(1e-20)
    ds = torch.arange(0, maxd + 1)
    if mode == "crop":
        assert cut is not None and replace_mask is None
        t, b, l, r = cut
        y_cut = fn(x[..., t * k_in:x.shape[-2] - b * k_in, l * k_in:x.shape[-1] - r * k_in])
        ref = y_full[..., t * k_out:(gh - b) * k_out, l * k_out:(gw - r) * k_out]
        err = (y_cut - ref).pow(2).mean(1)[0]
        dist = _up(_edge_distance(gh - t - b, gw - l - r), k_out)
        valid = torch.ones_like(err, dtype=torch.bool)
    elif mode == "replace":
        if replace_mask is None:
            t, b, l, r = cut
            replace_mask = torch.zeros(gh, gw, dtype=torch.bool)
            replace_mask[:t], replace_mask[gh - b:], replace_mask[:, :l], replace_mask[:, gw - r:] = True, True, True, True
        reg_px = F.interpolate(replace_mask[None, None].float(), size=x.shape[-2:], mode="nearest")[0, 0] > 0.5
        if replacement == "mean":
            new = x.mean((2, 3), keepdim=True).expand_as(x)
        elif replacement == "zero":
            new = torch.zeros_like(x)
        elif replacement == "noise":
            g = torch.Generator().manual_seed(seed)
            new = x.mean((2, 3), keepdim=True) + x.std((2, 3), keepdim=True) * torch.randn(x.shape, generator=g).to(x.device)
        elif replacement == "other":
            assert other is not None
            new = other
        else:
            raise ValueError(replacement)
        y_cut = fn(torch.where(reg_px[None, None].expand_as(x), new, x))
        err = (y_cut - y_full).pow(2).mean(1)[0]
        dist = _up(cheb_distance(replace_mask, maxd), k_out)
        valid = _up((~replace_mask), k_out)
    else:
        raise ValueError(mode)
    rel, cnt = [], []
    for d in ds:
        m = valid & (dist == d)
        cnt.append(int(m.sum()))
        rel.append(float(err[m].mean() / scale) if m.any() else float("nan"))
    return {"dist": ds, "rel_error": torch.tensor(rel), "count": torch.tensor(cnt), "mode": mode}


def margin_needed(profile, rel_tol: float = 1e-4):
    """Smallest distance d (token cells) such that EVERY populated distance >= d has relative error <= rel_tol; None if the error
    never settles within the profile (non-local dependence, or a profile too short to see the reach). Distances are measured from the
    first retained cell (0 = adjacent to the cut), so pad at least d + 1 cells."""
    e, d = profile["rel_error"], profile["dist"]
    ok = torch.isfinite(e)
    for k in range(len(d)):
        tail = e[k:][ok[k:]]
        if len(tail) and (tail <= rel_tol).all():
            return int(d[k])
    return None
