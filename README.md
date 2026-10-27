# receptive_field_scope

Measure the receptive field of **any** PyTorch encoder/decoder (any `tensor -> tensor` callable), with a focus on
tokenizers/autoencoders, windowed or haloed attention, strided/patchified stems, and tiled inference.

Status: **v0.1 draft** (written to bootstrap a project; see [Roadmap](#roadmap)).

## Why

"How far does one latent token see?" is a surprisingly slippery question:

* Formula-based counts break the moment a model has blocked attention, zero-init residual gates, odd stems, or
  non-truncating leakage. (The tool in this repo exists because a hand-derived formula was wrong twice.)
* The **structural support** (what *can* influence an output) and the **effective receptive field** (what *does*,
  by gradient energy) can differ by 3x or more, and the answer to "does the receptive field matter?" is usually in
  the gap between them.
* Strided and blocked models are **alignment-dependent**: reach differs by which cell inside a block/stride you probe,
  and is often asymmetric left/right.
* What you usually want in the end is practical: *what tile margin reproduces the full-image result?*

## Install

```bash
pip install -e .            # needs torch
pip install -e ".[dev]"     # + pytest
pip install -e ".[examples]"  # + compressai, torchvision, matplotlib
pytest
```

## Quickstart

```python
import torch
from receptive_field_scope import (measure_receptive_field, sweep_centers, summarize, activate_gates,
                                   layerwise_receptive_field, print_layerwise, margin_sweep, print_margin_sweep)

enc = my_encoder.eval()                                    # any callable [1,C,H,W] -> [1,C',h,w]

# one centre, random-noise probe (fine for structural support of local models)
r = measure_receptive_field(enc, (1, 3, 512, 512))[0]
print(r.left, r.right, r.up, r.down, r.eff_radius, r.touches_border, r.is_global)

# sweep every offset inside one 4x4-token block, with a REAL image (use this for trained, data-dependent attention)
rs = measure_receptive_field(enc, input=image, centers=sweep_centers((64, 64), period=4))
summarize(rs, "encoder")

# models with zero-init residual gates: open them to see the structural worst case
with activate_gates(model):
    rs = measure_receptive_field(model.encoder, (1, 3, 512, 512), centers=sweep_centers((64, 64), period=4))

# decoder: input is the token grid; report in pixels
measure_receptive_field(model.decoder, (1, 32, 64, 64), scale=patch_size)

# where does the reach come from?  per-layer hops
print_layerwise(layerwise_receptive_field(enc, (1, 3, 512, 512)))

# the practical question: which tile margin reproduces full-image output?
print_margin_sweep(margin_sweep(enc, [image], tile=64, margins=[0, 16, 32, 64, 96], discrete=False))
```

## What is measured

| quantity | meaning | use it for |
|---|---|---|
| `left/right/up/down` | **support**: farthest input cell with `|grad| > eps * max`, from the output cell's centre to the outer cell edge | tile margins; strictly local models have *exactly zero* gradient outside it |
| `radius` | `max(left,right,up,down)` | worst-case one-sided reach |
| `eff_left/...`, `eff_radius` | **effective**: central interval / disk holding `energy` (default 95%) of the gradient mass (Luo et al. 2016) | "does the long-range path matter?"; models that leak everywhere |
| `asymmetry` | `|left-right|`, `|up-down|` | alignment (block/stride) dependence |
| `touches_border` | support reached the input edge | numbers are lower bounds, use a bigger input |
| `is_global` | support spans >= 90% of the input | global attention / GroupNorm / pooling |

Conventions: distances are in **input-grid cells x `scale`**, from the **centre of the output cell** to the **outer edge of
the farthest cell**, so `left + right` is the support width. `centers` are output-grid cells; `sweep_centers(out_hw, period)`
enumerates every offset in one block.

Gradients use several random channel projections of the output vector (so LayerNorm-style cancellations of a plain
channel sum cannot hide paths) and a random-normal input unless you pass `input=`.

### Perturbation-based variant

`perturbation_receptive_field(fn, ...)` adds noise to one input patch at a time and watches one output cell. It needs no
gradients, so it works through quantizers, argmax and black-box models; it is coarser and slower
(use `window=` to restrict the probed area).

## Example: an off-the-shelf encoder (CompressAI `g_a`, random init)

`python examples/compressai_encoder.py` (output from this repo's first run):

```
== g_a (4 centres) ==                  # alignment sweep, support in input pixels
  left   38.0   right 23.0   up 38.0   down 23.0     <- strided 5x5 convs: asymmetric reach, same on every offset
per layer: radius 3 -> 8 -> 18 -> 38 (hops 3, 5, 10, 20 as stride doubles)
margin(px)  frac differing
         0        0.806742
        16        0.338704
        32        0.000000       <- margin >= support radius - half a cell (38 - 8 = 30) reproduces the full encode
```

Practical rule of thumb used by `tiled_apply`/`margin_sweep`: the needed margin is about
`support radius - half an output cell`, rounded up to a multiple of whatever alignment the model needs (block size x
stride). Always confirm with `margin_sweep` on real images.

## Caveats (read these)

* The default probe is **random noise**. For trained models with data-dependent attention, pass real images
  (`input=`) and compare several (`over_images`); effective radius can differ a lot between images.
* The support threshold `eps` is relative to the peak gradient. For strictly local models it hardly matters (gradient is
  exactly zero outside the support); for models with soft tails it does. Report the `eps` you used.
* Zero-init gates (LayerScale/ReZero) make a fresh model look conv-only. Use `activate_gates` to see the structural
  worst case, and measure trained models without it to see what training actually opened.
* Models must be in `eval()` mode (BatchNorm batch statistics otherwise mix samples).
* Gradient influence is not the same as importance for the loss: a path with 3% of the gradient energy can still carry
  information the decoder needs (global colour, tone).
* Data-dependent structure (deformable ops, dynamic routing) has input-dependent support; the tool reports what it saw.
* Tests take ~1.5 minutes on CPU (perturbation and position-map tests dominate).

## Related tools (not evaluated in depth)

Found by a quick search while drafting; check them before assuming overlap or gaps:
[fornaxai/receptivefield](https://github.com/fornaxai/receptivefield) (gradient-based, TF/PyTorch),
[rfa-toolbox](https://pypi.org/project/rfa-toolbox/1.2.0/) (graph-based analysis for conv nets),
[Fangyh09/pytorch-receptive-field](https://github.com/Fangyh09/pytorch-receptive-field),
[jotaf98/easy-receptive-fields-pytorch](https://github.com/jotaf98/easy-receptive-fields-pytorch)
(analytical conv-layer RF calculators). First task of the next session should be to read their READMEs and state
plainly what this project adds (candidates: attention/blocked/haloed models, per-direction asymmetry and alignment
sweeps, decoders, gate handling, effective-vs-support, perturbation variant, per-layer hops, tiling-margin
validation).

## Roadmap

Richer analyses that this scaffold is meant to grow into:

* Effective RF **across datasets** (distribution of eff. radius over many images, per token position, per class of content).
* **Gradient vs perturbation vs attention-rollout** side-by-side, and where they disagree.
* **Position maps**: effective radius / influence per token position to expose border effects and
  register-like tokens (tokens whose reach or latent norm is out of line with neighbours; cf. Darcet et al.,
  *Vision Transformers Need Registers*, arXiv 2309.16588).
* Inverse view: **influence of each latent token on the whole output** (how much does token t matter to the reconstruction).
* Layer-by-layer **effective** RF growth, and comparison against theoretical RF formulas (ship formula validators).
* Tiling: seam-error maps, latent-space blending recommendations, margin recommender per model.
* Plotting helpers (matplotlib): gradient maps, radius-vs-offset, position maps; MLflow logging helpers.
* Benchmarks of popular tokenizers/autoencoders (CompressAI family, SD-style VAEs, VQGAN-like, others) as a table in this README.

## Layout

```
src/receptive_field_scope/
  core.py        measure_receptive_field, RFResult, summarize
  gates.py       activate_gates
  layerwise.py   layerwise_receptive_field (per-layer hops via forward hooks)
  perturb.py     perturbation_receptive_field
  sweeps.py      sweep_centers, grid_centers, position_map, over_images
  tiling.py      tiled_apply, margin_sweep
tests/           pytest, with exact-answer toy models (tests/toys.py)
examples/        compressai_encoder.py
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
