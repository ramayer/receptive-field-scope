import matplotlib
matplotlib.use("Agg")
import torch

from receptive_field_scope import measure_receptive_field, position_map
from receptive_field_scope.plotting import overlay_heatmap, plot_position_map, plot_waveform_rf
from toys import ConvTokenizer


def test_overlay_heatmap_smoke():
    img = torch.rand(1, 3, 128, 128)
    r = measure_receptive_field(ConvTokenizer(2).eval(), input=img, keep_maps=True)[0]
    ax = overlay_heatmap(img, r)
    assert ax is not None


def test_waveform_plot_smoke():
    import torch.nn as nn
    m = nn.Sequential(nn.Conv1d(1, 4, 10, stride=5), nn.Conv1d(4, 4, 3, stride=2)).eval()
    w = torch.randn(1, 1, 4000)
    r = measure_receptive_field(m, input=w, keep_maps=True, eps=1e-12)[0]
    assert plot_waveform_rf(w, r, 16000, xlim=(0.1, 0.15)) is not None


def test_position_map_plot_smoke():
    pm = position_map(ConvTokenizer(1).eval(), (1, 3, 64, 64), step=2)
    assert plot_position_map(pm) is not None


def test_overlay_requires_maps():
    import pytest
    r = measure_receptive_field(ConvTokenizer(1).eval(), (1, 3, 64, 64))[0]
    with pytest.raises(ValueError):
        overlay_heatmap(torch.rand(1, 3, 64, 64), r)
