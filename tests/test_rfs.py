import pytest
import torch

from receptive_field_scope import (activate_gates, layerwise_receptive_field, margin_sweep, measure_receptive_field,
                                   over_images, perturbation_receptive_field, position_map, sweep_centers)
from toys import BlockedMix, ConvTokenizer, GlobalMix

torch.manual_seed(0)


def test_conv_tokenizer_exact_width():
    m = ConvTokenizer(5).eval()
    r = measure_receptive_field(m, (1, 3, 512, 512))[0]
    assert r.left + r.right == 8 + 2 * 8 * 5 + 2 == 90
    assert not r.touches_border and not r.is_global


def test_in_shape_may_be_a_tensor():                      # ints required internally; tensors caused tensor-valued fields
    r = measure_receptive_field(ConvTokenizer(2).eval(), torch.tensor((1, 3, 256, 256)))[0]
    assert isinstance(r.left, float) and isinstance(r.ref[0], float)


def test_global_flagged():
    r = measure_receptive_field(GlobalMix().eval(), (1, 3, 64, 64))[0]
    assert r.is_global and r.eff_radius > 10


def test_gate_closed_then_open():
    m = BlockedMix().eval()
    closed = measure_receptive_field(m, (1, 8, 32, 32))[0]
    assert closed.width == (1.0, 1.0)
    with activate_gates(m, 1.0):
        opened = measure_receptive_field(m, (1, 8, 32, 32), centers=sweep_centers((32, 32), period=4))
    assert max(r.radius for r in opened) == 4.5            # (b - 1 + h) cells + half-cell convention
    assert m.gate.item() == 0.0                            # restored


def test_block_alignment_asymmetry():
    m = BlockedMix(4, 1).eval()
    with activate_gates(m):
        rs = measure_receptive_field(m, (1, 8, 32, 32), centers=sweep_centers((32, 32), period=4))
    lefts = sorted({r.left for r in rs})
    assert lefts == [1.5, 2.5, 3.5, 4.5]                  # reach depends on offset inside the block
    assert any(r.asymmetry[0] > 0 for r in rs)


def test_real_input_accepted():
    m = ConvTokenizer(2).eval()
    img = torch.rand(1, 3, 256, 256)
    r = measure_receptive_field(m, input=img)[0]
    assert r.left + r.right == 8 + 2 * 8 * 2 + 2


def test_perturbation_agrees_with_gradient_on_conv():
    m = ConvTokenizer(3).eval()
    g = measure_receptive_field(m, (1, 3, 256, 256))[0]
    p = perturbation_receptive_field(m, (1, 3, 256, 256), patch=1, step=1, window=64, rel_tol=1e-5)
    assert abs((p.left + p.right) - (g.left + g.right)) <= 2     # perturbation sampling is a bit coarser


def test_layerwise_hops():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.Conv2d(8, 8, 3, padding=1),
                            torch.nn.Conv2d(8, 8, 3, padding=1)).eval()
    rows = layerwise_receptive_field(m, (1, 3, 64, 64))
    assert [r["radius"] for r in rows] == [1.5, 2.5, 3.5]
    assert [r["hop"] for r in rows[1:]] == [1.0, 1.0]


def test_position_map_shows_border_effect():
    m = ConvTokenizer(2).eval()
    pm = position_map(m, (1, 3, 128, 128), step=2)
    assert pm["support_radius"].shape == (8, 8)
    assert pm["asym_x"][0, 0] > 0 and pm["asym_x"][4, 4] == 0        # clipped by the image border vs interior


def test_over_images_runs():
    out = over_images(ConvTokenizer(1).eval(), [torch.rand(1, 3, 128, 128) for _ in range(2)])
    assert out["support_std"] < 1e-6                                  # conv support is image-independent


def test_margin_sweep_matches_support():
    m = ConvTokenizer(5).eval()                    # one-sided support ~ 45px -> margin 48 (multiple of 8) suffices
    imgs = [torch.randn(1, 3, 256, 256)]
    rows = {r["margin"]: r for r in margin_sweep(m, imgs, tile=64, margins=[0, 16, 48])}
    assert rows[0]["frac_differing"] > 0
    assert rows[48]["frac_differing"] == 0
