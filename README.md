# receptive_field_scope

Measure the receptive field of **any** PyTorch model (any `tensor -> tensor` callable) in **1D (audio), 2D (images, spectrograms) or 3D (video)**, with a focus on
tokenizers/autoencoders, windowed or haloed attention, strided/patchified stems, speech encoders (HuBERT/wav2vec2 style), and tiled/chunked inference.
"scope" as in the *scope* of the receptive field, and as in a telescope you point at it.

Status: **v0.4 draft** (written to bootstrap a project; see [Roadmap](#roadmap)). Tests: 38 pass (`pytest`; ~1.5 min, `-m "not slow"` takes ~10 s).

## Why

"How far does one latent token / one audio frame see?" is a surprisingly slippery question:

* Formula-based counts break the moment a model has blocked attention, zero-init residual gates, odd stems, or
  non-truncating leakage. (This tool exists because a hand-derived formula was wrong twice.)
* The **structural support** (what *can* influence an output) and the **effective receptive field** (what *does*,
  by gradient mass) can differ by 3x or more, and the answer to "does the receptive field matter?" is usually in the gap between them.
* Strided and blocked models are **alignment-dependent**: reach differs by which cell inside a block/stride you probe,
  and is often asymmetric left/right.
* What you usually want in the end is practical: *what tile/chunk margin reproduces the full-input result?*

## Install

```bash
pip install -e .                 # needs torch
pip install -e ".[dev]"          # + pytest
pip install -e ".[examples]"     # + compressai, torchvision, matplotlib, transformers, jupyter
pytest                           # or: pytest -m "not slow"
```

## Quickstart

```python
import torch
from receptive_field_scope import (measure_receptive_field, sweep_centers, summarize, activate_gates, channels_last,
                                   layerwise_receptive_field, print_layerwise, margin_sweep, print_margin_sweep)
from receptive_field_scope.plotting import overlay_heatmap, plot_waveform_rf

enc = my_encoder.eval()                                    # any callable [1,C,*S] -> [1,C',*s]

# one centre, random-noise probe (fine for structural support of local models)
r = measure_receptive_field(enc, (1, 3, 512, 512))[0]
print(r.left, r.right, r.up, r.down, r.eff_radius, r.touches_border, r.is_global)

# a REAL image (needed for trained, data-dependent attention) + every offset inside one 4x4-token block
rs = measure_receptive_field(enc, input=image, centers=sweep_centers((64, 64), period=4), keep_maps=True)
summarize(rs, "encoder")
overlay_heatmap(image, rs[0])                              # heatmap over the image, support box + effective circle

# models with zero-init residual gates: open them to see the structural worst case
with activate_gates(model):
    rs = measure_receptive_field(model.encoder, (1, 3, 512, 512), centers=sweep_centers((64, 64), period=4))

# decoder: input is the token grid; report in pixels
measure_receptive_field(model.decoder, (1, 32, 64, 64), scale=patch_size)

# where does the reach come from?  per-layer hops
print_layerwise(layerwise_receptive_field(enc, (1, 3, 512, 512)))

# the practical question: which tile margin reproduces the full-image output?
print_margin_sweep(margin_sweep(enc, [image], tile=64, margins=[0, 16, 32, 64, 96]))
```

### Audio (1D)

```python
from transformers import HubertModel
from receptive_field_scope import hf_audio_layer

model = HubertModel.from_pretrained(...).eval()            # or an AVES checkpoint converted to HubertModel
wave = torch.randn(1, 1, 16000)                            # [1, 1, samples]; use real clips for effective RFs
ms = 1000 / 16000                                          # scale: samples -> milliseconds

measure_receptive_field(hf_audio_layer(model, extractor_only=True), input=wave, eps=1e-12, scale=ms)   # conv extractor: 25 ms
measure_receptive_field(hf_audio_layer(model, layer=6), input=wave, power=2, scale=ms)                 # transformer layer 6
```

Models whose forward returns `[B, T, C]` (most transformer stacks) need `channels_last(fn)`; models taking `[B, T]` waveforms need
`waveform_model(fn)`. `hf_audio_layer` does both for Hugging Face wav2vec2/HuBERT-style models. For torchaudio/fairseq-format models
(AVES is, as far as we know, distributed in a torchaudio-compatible format) write the same few-line wrapper yourself; this has not been tested here.

## What is measured

| quantity | meaning | use it for |
|---|---|---|
| `lo`, `hi` (per axis); `left/right/up/down` views | **support**: farthest input cell with `|grad| > eps * max`, from the output cell's centre to the outer cell edge | tile/chunk margins; strictly local models have *exactly zero* gradient outside it |
| `radius`, `width`, `asymmetry` | max one-sided reach; `lo+hi` per axis (x/time first); `|lo-hi|` per axis | worst-case reach; alignment (block/stride) dependence |
| `eff_lo/eff_hi`, `eff_radius` | **effective**: central interval / radial radius holding `energy` (default 95%) of the gradient mass (Luo et al. 2016) | "does the long-range path matter?"; models that leak everywhere |
| `touches_border` | support reached the input edge | numbers are lower bounds, use a bigger input |
| `is_global` | support spans >= 90% of the input on some axis | global attention / GroupNorm / pooling |

Conventions: distances are in **input-grid cells x `scale`** (`scale` may be one number or one per axis), from the **centre of the output cell** to the
**outer edge of the farthest cell**, so `lo + hi` is the support width. `centers` are output-grid cells (`(y, x)` in 2D, `(t,)` or an int in 1D);
`sweep_centers(out_shape, period)` enumerates every offset in one block.

Gradients use several random channel projections of the output vector (so LayerNorm-style cancellations of a plain channel sum cannot hide paths)
and a random-normal input unless you pass `input=`.

### L1 mass vs squared energy (`power`)

The effective radius is computed from `|grad|**power`. `power=1` (default; Luo et al.'s magnitude) lets **diffuse tails add up**; `power=2` (squared, "energy")
suppresses them. For local models the two agree; for models with weak global leakage they do not. Example from the tests: a wav2vec2-style conv extractor with
GroupNorm over time (global support) has an effective radius of ~5250 samples at `power=1` but ~270 samples (17 ms) at `power=2`. For leaky models report both.

### Perturbation-based variant

`perturbation_receptive_field(fn, ...)` adds noise to one input patch at a time and watches one output cell. It needs no gradients, so it works through
quantizers, argmax and black-box models; it is coarser and slower (use `window=` to restrict the probed area).

## Lesion studies (decoder side)

`receptive_field_scope.lesion` damages latent tokens, decodes, and asks (a) how well the hole heals and (b) how far the damage spreads.
It needs only `decode(latents)`, so it works for any tokenizer/codec.

```python
from receptive_field_scope import make_mask, lesion_latents, damage_profile, hole_sweep, lesion_importance_map

y = model.g_a(images)                                       # CompressAI-style latents [N, C, h, w]
mask = make_mask(y.shape[-2:], "lattice", step=2)           # the y[:, :, ::2, ::2] = fill experiment (True = lesioned)
out = lesion_latents(model.g_s, y, mask, fill="neg", quantize=torch.round)
prof = damage_profile(out["baseline"], out["lesioned"], mask, original=images)    # damage vs distance, inside and outside
```

* **Fills are a family, not zero.** A zero latent is not "missing": some decoders turn it into a flat patch, others into texture. Available:
  `zero`, `mean`, `neg` (reflect through the channel centre, `2c - y`: same magnitude, opposite content; `center=` accepts e.g. hyperprior means),
  `neg0` (literal `-y`), `noise`, `shuffle` (real latents from the wrong place), `other` (another image), `neighbor` (latent interpolation).
  Results that agree across fills are properties of the decoder; results that vary are properties of the fill.
* **Fill statistics are a confound.** `mean`, `neg` (default centre) and `noise` use the per-channel mean/std of *this image's* latents, i.e. a global image statistic, so a
  noise-filled hole can inherit the image's palette without any global state in the decoder. Pass `stats=` (latents of another image or a pooled set) to control for it, and use
  `fill_color_attribution` (does the filled colour follow the local surround, the global mean, or the true content?) and `palette_distance` to quantify palette effects.
* **Compare against the un-lesioned *quantized* decode** (`baseline`), not the original: the quantization floor is not damage.
* **`damage_profile`**: collateral damage versus distance (a strictly local decoder drops to *exactly zero* beyond its reach) and healing versus depth inside the hole.
* **`hole_sweep`**: how large a hole can be interpolated.
* **`lesion_importance_map`**: lesion every token alone and measure damage *far* away. A token that changes distant pixels carries global information
  (the decoder-side register detector; uniform-background tokens are the suspects).
* **Texture vs smooth continuation**: `harmonic_fill` (Laplace diffusion, no new texture) and `radial_power` / `highfreq_ratio` let you check whether a filled hole carries
  high-frequency content from the neighbouring latents or is just a smooth continuation. Decoders also ring: look at the crops as well as the spectrum.

## Cut tests: tiling padding and "did I chop off a register?"

Lesion profiles measure damage around a hole. For tiling the practical questions are different, and `cut_profile` answers them: remove or replace a band or corner of the
input and measure how far into the *retained* region the output changes (encoders: pass the image; decoders: pass the latents; distances in token cells).

```python
from receptive_field_scope import cut_profile, margin_needed
p_crop = cut_profile(model.encoder, image, (0, 0, 0, 6), mode="crop")        # right band of 6 tokens removed
p_repl = cut_profile(model.encoder, image, mode="replace", replace_mask=corner_mask, replacement="noise")
margin_needed(p_crop, rel_tol=1e-6)      # tokens of context to keep beyond the tile edge (None = the error never settles)
```

* **`crop`**: content removed *and* the border moved (zero padding, absolute positions). **`replace`**: the band keeps its place and only its content changes.
* Far error under `replace` = content-borne long-range dependence (register-like). Far error under `crop` but not `replace` = a border/position effect.
* `margin_needed` returning `None` means the output never settled within the profile: either a non-local dependence or a profile too short to see the reach.

## Demos

* `notebooks/receptive_field_demo.ipynb` (executed, outputs included): heatmaps of the receptive field overlaid on an image for a local-conv, a blocked-mixing
  (haloed-attention-like) and a global-attention toy tokenizer; block-alignment asymmetry; image dependence; per-layer hops; tile-margin sweep; a waveform.
* `notebooks/lesion_demo.ipynb` (executed with a TOY autoencoder because pretrained CompressAI weights could not be downloaded in the sandbox; it uses real CompressAI weights automatically when they can be fetched): lattice lesions with 7 fills, damage-vs-distance (block hole), hole sweep, tiling-margin / register cut tests, texture-vs-smooth test, register hunt.
* `examples/compressai_encoder.py`: off-the-shelf encoder (CompressAI `g_a`, random init).
* `examples/hubert_receptive_field.py`: Hugging Face HuBERT (random init): conv extractor = 400 samples/25 ms at stride 320; global support above it.

CompressAI `g_a` (random init), alignment sweep, input pixels:

```
left 38 right 23 up 38 down 23     <- strided 5x5 convs: asymmetric reach, same on every offset
per layer: radius 3 -> 8 -> 18 -> 38
margin(px)  frac differing
         0        0.806742
        16        0.338704
        32        0.000000       <- margin >= support radius - half a cell (38 - 8 = 30) reproduces the full encode
```

HuBERT-style conv extractor (16 kHz): per-layer support widths 10, 20, 40, 80, 160, 240, **400 samples**; stride 320. The tests check this against a real
`transformers.HubertModel`, and also show that the **`group` feature-extractor norm variant (wav2vec2-base/HuBERT-base style) makes the conv extractor globally dependent
on the clip** (weakly: tiny at `power=2`), while the `layer` norm variant is exactly 400 samples. That matters for chunked inference: chunk outputs will not
exactly match full-clip outputs for `group`-norm models even with generous margins.

## Caveats (read these)

* The default probe is **random noise**. For trained models with data-dependent attention, pass real inputs (`input=`) and compare several (`over_images` / `over_clips`);
  effective radius can differ a lot between inputs.
* The support threshold `eps` is relative to the peak gradient. For strictly local models it hardly matters (gradient is exactly zero outside the support, so
  `eps=1e-12` is safe); for models with soft tails it does. Report the `eps` and `power` you used.
* Zero-init gates (LayerScale/ReZero) make a fresh model look conv-only. Use `activate_gates` to see the structural worst case, and measure trained models
  without it to see what training actually opened.
* **Unpadded ("valid") convolutions** (e.g. the wav2vec2 extractor) do not tile the input: the output grid has a non-integer ratio to the input, so the
  reference point used for `lo`/`hi` is proportional, and `lo`/`hi` can look asymmetric even though `lo + hi` (the width) is exact. Tiling helpers
  (`tiled_apply`) require whole-number ratios, so use them with padded models; chunk valid-conv models by hand.
* Models must be in `eval()` mode (BatchNorm batch statistics otherwise mix samples).
* Gradient influence is not the same as importance for the loss: a path with 3% of the gradient mass can still carry information the decoder needs.
* Data-dependent structure (deformable ops, dynamic routing) has input-dependent support; the tool reports what it saw.
* The audio support for models with a global first-layer normalisation is "global"; look at the effective radius at both `power`s.
* Tests take ~2 minutes on CPU (perturbation and position-map tests are marked `slow`).

## Related tools (not evaluated in depth)

Found by a quick search while drafting; check them before assuming overlap or gaps:
[fornaxai/receptivefield](https://github.com/fornaxai/receptivefield) (gradient-based, TF/PyTorch),
[rfa-toolbox](https://pypi.org/project/rfa-toolbox/1.2.0/) (graph-based analysis for conv nets),
[Fangyh09/pytorch-receptive-field](https://github.com/Fangyh09/pytorch-receptive-field),
[jotaf98/easy-receptive-fields-pytorch](https://github.com/jotaf98/easy-receptive-fields-pytorch)
(analytical conv-layer RF calculators). First task of the next session should be to read their READMEs and state
plainly what this project adds (candidates: attention/blocked/haloed models, per-direction asymmetry and alignment
sweeps, decoders, audio, gate handling, effective-vs-support and L1-vs-L2 mass, perturbation variant, per-layer hops, tiling-margin validation).

## Roadmap

Done so far: lesion studies and cut tests. Near-term, in rough priority order (motivation: large-image tiled generation / infinite zoom with VAR/GRN-style generators: the decoder's reach and the *stationarity* of the latent distribution matter more than the encoder's):

* **Paired-context probe** (encoder side): paste identical pixels (e.g. a saturated-white sclera vs a white border) into different contexts and measure the latent difference versus *depth inside the region* (beyond the structural support a strictly local model must give exactly zero), plus a linear probe for context class.
* **Far-field maps** for encoders: gradient mass outside the token's near region as a map and a leakage curve L(m) = mass beyond half-width m; **context-swap** maps (replace everything outside the near region, measure relative latent change per token); **constant-image probe** (latent structure of a flat white/black image: position/border/global-state signal).
* **Stationarity checks**: translation equivariance (shift by whole patches -> latents shift identically) and a border-distance curve (latent change vs distance to the crop border, separating zero-padding position cues from long-range content influence).
* Lesion study on real checkpoints (bmshj2018, mbt2018, cheng2020-attn) and on MAE-on vs MAE-off tokenizers.

Longer-term:

* Effective RF **across datasets** (distribution of effective radius over many images/clips, per token position, per content class).
* **Gradient vs perturbation vs attention-rollout** side-by-side, and where they disagree.
* **Position maps** (`position_map`, `plot_position_map` exist): effective radius / influence per token position to expose border effects and
  register-like tokens (tokens whose reach or latent norm is out of line with neighbours; cf. Darcet et al., *Vision Transformers Need Registers*, arXiv 2309.16588).
* Inverse view: **influence of each latent token on the whole output**.
* Layer-by-layer **effective** RF growth for transformer stacks; compare with theoretical RF formulas (ship formula validators, e.g. a measured-vs-formula unit test).
* Valid-conv-aware reference points and chunking helpers (speech models), including a margin recommender for wav2vec2-style chunked inference.
* Spectrogram-input audio models (AST/BEATs-style: 2D with time and frequency axes), verified on real checkpoints (HuBERT, AVES, others), with real speech/animal-sound clips.
* More plotting (radius-vs-offset, per-layer growth), MLflow logging helpers.
* Benchmarks of popular tokenizers/autoencoders/speech encoders as a table in this README (trained checkpoints, not random init).

## Layout

```
src/receptive_field_scope/
  core.py        measure_receptive_field, RFResult, summarize  (N-d)
  gates.py       activate_gates
  adapters.py    channels_last, waveform_model, hf_audio_layer
  layerwise.py   layerwise_receptive_field (per-layer hops via forward hooks)
  perturb.py     perturbation_receptive_field
  sweeps.py      sweep_centers, grid_centers, position_map, over_images / over_clips
  tiling.py      tiled_apply, margin_sweep  (N-d)
  cut.py         cut_profile, margin_needed (crop/replace tests: tiling margins and register checks)
  lesion.py      make_mask, apply_fill, lesion_latents, damage_profile, hole_sweep, lesion_importance_map, harmonic_fill, radial_power
  plotting.py    overlay_heatmap, plot_waveform_rf, plot_position_map  (matplotlib, imported lazily)
tests/           pytest; exact-answer toy models (tests/toys.py), 1D/3D, HuBERT checks (tests/test_audio_nd.py)
examples/        compressai_encoder.py, hubert_receptive_field.py, tiny_autoencoder.py
notebooks/       receptive_field_demo.ipynb, lesion_demo.ipynb
```

## Provenance

Extracted from work on hierarchical-binary-quantization tokenizers (narrow-receptive-field autoencoder with haloed
local attention), where measurement found that a hand-derived RF formula undercounted blocked attention reach
(`attn_block - 1 + halo` tokens, not `halo`) and mis-assumed layer structure.

## ⚖️ License -- likely no traditional license or copyright applies

This project was created almost entirely by large language models, based on research papers.  
The human contribution was minimal and mostly evaluative: initiating the idea, requesting papers, and describing qualitative behavior of the generated images.

As a result, there may be **no copyright ownership** in the traditional legal sense — most jurisdictions require meaningful human authorship to grant copyright.

Accordingly:

- ⚠️ No copyright is claimed.
- ✅ To the fullest extent allowed by law, this work is **dedicated to the public domain**.
- 🌍 Where a legal dedication is required, this repository is released under [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/).

You are free to **use, modify, copy, and redistribute** this project for any purpose, without restriction.
