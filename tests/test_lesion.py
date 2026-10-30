import pytest
import torch
import torch.nn as nn

from receptive_field_scope.lesion import (FILLS, apply_fill, fill_color_attribution, palette_distance, cheb_distance, damage_profile, harmonic_fill, highfreq_ratio,
                                          hole_sweep, lesion_importance_map, lesion_latents, make_mask, psnr, radial_power)

torch.manual_seed(0)
CELL = 4


class LocalDec(nn.Module):
    """Each token paints its own 4x4 cell (kernel = stride), then two 3x3 convs: pixel reach 2 -> 1 cell."""
    def __init__(self, C=8, glob=0.0, register=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(C, 8, CELL, stride=CELL)
        self.c1, self.c2 = nn.Conv2d(8, 8, 3, padding=1), nn.Conv2d(8, 3, 3, padding=1)
        self.glob, self.register = glob, register

    def forward(self, z):
        x = self.up(z)
        if self.glob:                                      # global mean broadcast (e.g. a global norm / attention)
            x = x + self.glob * z.mean((1, 2, 3), keepdim=True)
        if self.register:                                  # token (0,0)'s first channel leaks everywhere
            x = x + self.register * z[:, :1, :1, :1]
        return self.c2(torch.relu(self.c1(torch.relu(x))))


def test_masks_and_distance():
    m = make_mask((8, 8), "lattice", step=2)
    assert m.sum() == 16 and m[0, 0] and not m[0, 1]
    assert make_mask((8, 8), "block", size=(2, 2)).sum() == 4
    d = cheb_distance(make_mask((9, 9), "single"), 4)
    assert d[4, 4] == 0 and d[4, 5] == 1 and d[2, 2] == 2 and d[0, 0] == 4


def test_fills():
    y = torch.randn(2, 6, 8, 8) + 3
    m = make_mask((8, 8), "block", size=(2, 2))
    mu = y.mean((2, 3), keepdim=True)
    assert torch.equal(apply_fill(y, m, "neg0")[:, :, 3:5, 3:5], -y[:, :, 3:5, 3:5])
    assert torch.allclose(apply_fill(y, m, "neg")[:, :, 3:5, 3:5], (2 * mu - y)[:, :, 3:5, 3:5])
    assert torch.equal(apply_fill(y, m, "zero")[:, :, 3:5, 3:5], torch.zeros(2, 6, 2, 2))
    for f in FILLS:
        if f == "other":
            out = apply_fill(y, m, f, other=torch.zeros_like(y))
        else:
            out = apply_fill(y, m, f)
        assert torch.equal(out[:, :, ~m], y[:, :, ~m])          # un-lesioned tokens untouched
    sh = apply_fill(y, m, "shuffle")
    kept = {tuple(v.tolist()) for v in y[0].flatten(1)[:, ~m.flatten()].T}
    assert all(tuple(v.tolist()) in kept for v in sh[0][:, m].T)  # shuffled values come from un-lesioned tokens


def test_local_decoder_damage_is_contained():
    dec = LocalDec().eval()
    y = torch.randn(1, 8, 16, 16)
    mask = make_mask((16, 16), "single")
    o = lesion_latents(dec, y, mask, "neg")
    p = damage_profile(o["baseline"], o["lesioned"], mask, maxd=6)
    assert p["outside_damage"][0] > 0                          # 1 cell away: the 3x3 convs reach it
    assert (p["outside_damage"][1:] == 0).all()                # >=2 cells away: exactly zero
    assert p["damage_inside_mean"] > 0 and p["far_fraction_2"] == 0


def test_global_decoder_damage_spreads():
    dec = LocalDec(glob=2.0).eval()
    y = torch.randn(1, 8, 16, 16)
    mask = make_mask((16, 16), "single")
    o = lesion_latents(dec, y, mask, "neg")
    p = damage_profile(o["baseline"], o["lesioned"], mask, maxd=6)
    assert (p["outside_damage"][2:] > 0).all() and p["far_fraction_2"] > 0   # local decoder: exactly 0


def test_importance_map_finds_register_token():
    dec = LocalDec(register=5.0).eval()
    y = torch.randn(1, 8, 8, 8)
    r = lesion_importance_map(dec, y, fill="neg", near=1, step=1, batch=8)
    far = r["far_map"]
    assert far.shape == (8, 8) and far.argmax() == 0           # token (0, 0) is the register
    assert far[0, 0] > 10 * far.flatten()[1:].max()


def test_quantize_hook_and_hole_sweep():
    dec = LocalDec().eval()
    y = torch.randn(1, 8, 16, 16) * 3
    with torch.no_grad():
        orig = dec(torch.round(y))
    rows = hole_sweep(dec, y, sizes=(1, 2, 4), quantize=torch.round, original=orig)
    assert [r["size"] for r in rows] == [1, 2, 4] and rows[0]["ring1_damage"] > 0


def test_harmonic_fill_and_spectrum():
    yy, xx = torch.meshgrid(torch.arange(32.), torch.arange(32.), indexing="ij")
    img = torch.stack([xx / 32, yy / 32, (xx + yy) / 64])
    pm = torch.zeros(32, 32, dtype=torch.bool)
    pm[12:20, 12:20] = True
    out = harmonic_fill(img, pm, iters=800)
    assert torch.equal(out[:, ~pm], img[:, ~pm])
    assert (out[:, pm] - img[:, pm]).abs().max() < 0.03        # planar gradient is harmonic: recovered closely
    noise = torch.randn(3, 64, 64)
    smooth = torch.nn.functional.avg_pool2d(noise[None], 9, 1, 4)[0]
    assert highfreq_ratio(noise) > 3 * highfreq_ratio(smooth)
    f, p = radial_power(noise)
    assert f.shape == p.shape == (16,)
    assert psnr(img, img + 0.1) == pytest.approx(20.0, abs=0.01)


def test_stats_param_controls_noise_fill_palette():
    a = torch.randn(1, 6, 16, 16) + 5          # "orange" image latents: channel means ~ +5
    b = torch.randn(1, 6, 16, 16) - 5          # another image: channel means ~ -5
    m = make_mask((16, 16), "lattice", step=2)
    own = apply_fill(a, m, "noise")[:, :, m]
    foreign = apply_fill(a, m, "noise", stats=b)[:, :, m]
    assert own.mean() > 4 and foreign.mean() < -4          # default inherits the image's own statistics; stats= removes that
    assert torch.equal(apply_fill(a, m, "zero", stats=b)[:, :, m], torch.zeros(1, 6, int(m.sum())))


def test_color_attribution_local_vs_global():
    dec = LocalDec().eval()
    y = torch.randn(1, 8, 16, 16)
    m = make_mask((16, 16), "lattice", step=2)
    o = lesion_latents(dec, y, m, "neighbor")
    r = fill_color_attribution(o["baseline"], o["lesioned"], m)
    assert r["filled"].shape[0] == int(m.sum()) and 0.0 <= r["closer_to_local"] <= 1.0
    pal = torch.nn.functional.avg_pool2d(o["baseline"].detach(), CELL)[0].flatten(1).T
    assert palette_distance(r["filled"], pal).shape == (int(m.sum()),)


def test_dense_lattice_profile_is_degenerate_and_warns_block_is_not():
    dec = LocalDec().eval()
    y = torch.randn(1, 8, 16, 16)
    lat = make_mask((16, 16), "lattice", step=2)
    o = lesion_latents(dec, y, lat, "neg")
    with pytest.warns(UserWarning, match="degenerate mask geometry"):
        p = damage_profile(o["baseline"], o["lesioned"], lat, maxd=4)
    assert torch.isfinite(p["outside_damage"]).sum() == 1 and torch.isfinite(p["inside_damage"]).sum() == 1
    blk = make_mask((16, 16), "block", size=(8, 8))
    o = lesion_latents(dec, y, blk, "neg")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = damage_profile(o["baseline"], o["lesioned"], blk, maxd=4)
    assert torch.isfinite(p["outside_damage"]).sum() == 4 and torch.isfinite(p["inside_damage"]).sum() == 4
