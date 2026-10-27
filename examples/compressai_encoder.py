"""Off-the-shelf example: CompressAI's factorized-prior analysis transform (g_a), RANDOMLY initialised
(no pretrained weights needed for structural support; effective radius of trained weights will differ --
load a checkpoint and rerun with real images to see that).

    pip install compressai
    python examples/compressai_encoder.py
"""
import torch
from compressai.models import FactorizedPrior

from receptive_field_scope import (layerwise_receptive_field, margin_sweep, measure_receptive_field,
                                   print_layerwise, print_margin_sweep, summarize, sweep_centers)

torch.manual_seed(0)
enc = FactorizedPrior(128, 192).g_a.eval()          # 4 stride-2 5x5 convs with GDN: total stride 16

print("--- whole encoder, alignment sweep over one 16px token period (probe: random noise) ---")
rs = measure_receptive_field(enc, (1, 3, 512, 512), centers=sweep_centers((32, 32), period=2))
summarize(rs, "g_a")

print("\n--- per layer: where the reach comes from ---")
print_layerwise(layerwise_receptive_field(enc, (1, 3, 512, 512)))

print("\n--- tiling: fraction of latent elements that differ from a full-image encode ---")
images = [torch.rand(1, 3, 256, 256)]
print_margin_sweep(margin_sweep(enc, images, tile=64, margins=[0, 16, 32, 64, 96, 128]))
