"""1D (audio), 3D and HuggingFace HuBERT checks."""
import pytest
import torch
import torch.nn as nn

from receptive_field_scope import (channels_last, hf_audio_layer, layerwise_receptive_field, margin_sweep,
                                   measure_receptive_field, over_clips, sweep_centers)

torch.manual_seed(0)
KERNELS, STRIDES = (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)


def wav2vec2_extractor(width=16, group_norm=False):
    layers, cin = [], 1
    for i, (k, s) in enumerate(zip(KERNELS, STRIDES)):
        layers.append(nn.Conv1d(cin, width, k, stride=s, bias=False))
        if i == 0 and group_norm:
            layers.append(nn.GroupNorm(width, width))       # per-channel norm over the WHOLE time axis (wav2vec2-base)
        layers.append(nn.GELU())
        cin = width
    return nn.Sequential(*layers).eval()


def test_wav2vec2_conv_extractor_is_400_samples_stride_320():
    m = wav2vec2_extractor()
    r = measure_receptive_field(m, (1, 1, 16000), eps=1e-12)[0]
    assert r.width == (400.0,)                              # 25 ms at 16 kHz
    assert r.lo[0] + r.hi[0] == 400 and not r.is_global
    ms = measure_receptive_field(m, (1, 1, 16000), scale=1000 / 16000, eps=1e-12)[0]
    assert abs(ms.width[0] - 25.0) < 1e-9                   # scale reports milliseconds


def test_group_norm_over_time_makes_support_global():
    m = wav2vec2_extractor(group_norm=True)
    r1 = measure_receptive_field(m, (1, 1, 16000), power=1)[0]
    r2 = measure_receptive_field(m, (1, 1, 16000), power=2)[0]
    assert r1.is_global and r2.is_global                    # support: the whole clip, via the normalisation statistics
    assert r2.eff_radius < 0.05 * 16000 < r1.eff_radius     # squared energy is local; L1 mass is dominated by diffuse tails


def test_layerwise_audio_hops_match_theory():
    rows = layerwise_receptive_field(wav2vec2_extractor(), (1, 1, 16000), eps=1e-12,
                                     predicate=lambda n, m: isinstance(m, nn.Conv1d))
    widths = [r["lo"][0] + r["hi"][0] for r in rows]
    assert widths == [10, 20, 40, 80, 160, 240, 400]         # standard RF recursion


def test_sweep_centers_1d_and_tuple_centers():
    cs = sweep_centers((49,), period=4)
    assert len(cs) == 4 and all(len(c) == 1 for c in cs)
    r = measure_receptive_field(wav2vec2_extractor(), (1, 1, 16000), centers=[10, (11,)], eps=1e-12)
    assert len(r) == 2


def test_over_clips_runs():
    out = over_clips(wav2vec2_extractor(), [torch.randn(1, 1, 8000) for _ in range(2)], eps=1e-12)
    assert out["support_std"] < 1e-6


def test_3d_conv_has_three_axes():
    m = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.Conv3d(4, 4, 3, padding=1)).eval()
    r = measure_receptive_field(m, (1, 1, 16, 16, 16))[0]
    assert len(r.lo) == 3 and r.width == (5.0, 5.0, 5.0)


def test_1d_tiling_with_padded_strided_conv():
    m = nn.Sequential(nn.Conv1d(1, 8, 9, stride=4, padding=4), nn.GELU(), nn.Conv1d(8, 8, 9, padding=4)).eval()
    x = torch.randn(1, 1, 1024)
    r = measure_receptive_field(m, input=x, eps=1e-12)[0]
    need = int(max(r.lo[0], r.hi[0]))                        # one-sided reach in samples
    margin = ((need + 3) // 4) * 4                           # multiple of the stride
    rows = {row["margin"]: row for row in margin_sweep(m, [x], tile=128, margins=[0, margin])}
    assert rows[0]["frac_differing"] > 0 and rows[margin]["frac_differing"] == 0


def test_channels_last_adapter():
    f = channels_last(lambda x: x.movedim(1, -1))            # fn returns [1, T, C]
    r = measure_receptive_field(f, (1, 3, 64))[0]
    assert r.width == (1.0,)


transformers = pytest.importorskip("transformers")


def _tiny_hubert(feat_extract_norm):
    cfg = transformers.HubertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                    intermediate_size=128, conv_dim=(32,) * 7, feat_extract_norm=feat_extract_norm,
                                    num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    return transformers.HubertModel(cfg).eval()


def test_hubert_extractor_and_global_attention():
    m = _tiny_hubert("layer")                                # 'layer' norm = per-frame: no whole-clip normalisation
    ex = measure_receptive_field(hf_audio_layer(m, extractor_only=True), (1, 1, 16000), eps=1e-12)[0]
    assert ex.width == (400.0,)
    top = measure_receptive_field(hf_audio_layer(m), (1, 1, 16000))[0]
    assert top.is_global                                     # self-attention sees the whole clip
    assert top.eff_radius < 16000 / 2


def test_hubert_group_norm_variant_is_global_already_in_extractor():
    m = _tiny_hubert("group")                                # wav2vec2-base style
    ex = measure_receptive_field(hf_audio_layer(m, extractor_only=True), (1, 1, 16000))[0]
    assert ex.is_global
