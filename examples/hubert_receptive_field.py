"""Audio example: receptive fields of a HuBERT/wav2vec2-style model (Hugging Face), RANDOMLY initialised.

Structural support of the conv feature extractor does not depend on weights, so random init is enough for that.
The effective receptive field of the transformer layers DOES depend on trained weights and on the audio:
for real analysis load your checkpoint (HuBERT, or an AVES checkpoint converted to HubertModel) and pass real clips.

    pip install transformers
    python examples/hubert_receptive_field.py
"""
import torch
import transformers

from receptive_field_scope import (hf_audio_layer, layerwise_receptive_field, measure_receptive_field, print_layerwise,
                                   sweep_centers, summarize)

torch.manual_seed(0)
SR = 16000
MS = 1000 / SR                                                     # scale: samples -> milliseconds

cfg = transformers.HubertConfig(feat_extract_norm="layer")        # 'layer': per-frame norm. HuBERT-base/wav2vec2-base use 'group'
model = transformers.HubertModel(cfg).eval()
shape = (1, 1, SR)                                                 # one second of mono audio

print("--- conv feature extractor (support in ms; alignment sweep over 4 consecutive frames) ---")
ex = hf_audio_layer(model, extractor_only=True)
summarize(measure_receptive_field(ex, shape, centers=sweep_centers((49,), period=4), eps=1e-12, scale=MS), "extractor")

print("\n--- per conv layer (samples): lo+hi per layer = 10,20,40,80,160,240,400 (standard RF recursion) ---")
print_layerwise(layerwise_receptive_field(
    model.feature_extractor, shape, eps=1e-12,
    predicate=lambda n, m: isinstance(m, torch.nn.Conv1d),
    call=lambda m, x: m(x[:, 0])))

print("\n--- full model: support is the whole clip (global self-attention); effective radius (ms), L1 vs squared ---")
for layer in (None, 6):
    f = hf_audio_layer(model, layer=layer)
    for power in (1, 2):
        r = measure_receptive_field(f, shape, power=power, scale=MS)[0]
        print(f"layer={layer!s:5s} power={power}  global={r.is_global}  effective radius {r.eff_radius:7.1f} ms")

print("\n--- 'group' norm variant (HuBERT-base style): the extractor itself becomes global ---")
g = transformers.HubertModel(transformers.HubertConfig(feat_extract_norm="group")).eval()
r = measure_receptive_field(hf_audio_layer(g, extractor_only=True), shape, power=2, scale=MS)[0]
print(f"support is_global={r.is_global}; squared-energy effective radius {r.eff_radius:.1f} ms")
