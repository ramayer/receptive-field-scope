"""receptive_field_scope: measure receptive fields of arbitrary PyTorch encoders/decoders (images, audio, video)."""
from .core import RFResult, measure_receptive_field, summarize, result_from_map
from .gates import activate_gates
from .adapters import channels_last, waveform_model, hf_audio_layer
from .layerwise import layerwise_receptive_field, print_layerwise
from .perturb import perturbation_receptive_field
from .sweeps import sweep_centers, grid_centers, position_map, over_images, over_clips
from .tiling import tiled_apply, margin_sweep, print_margin_sweep
from .cut import cut_profile, margin_needed
from .lesion import (make_mask, apply_fill, lesion_latents, damage_profile, hole_sweep, lesion_importance_map,
                     harmonic_fill, radial_power, highfreq_ratio)

__all__ = [
    "RFResult", "measure_receptive_field", "summarize", "result_from_map", "activate_gates",
    "channels_last", "waveform_model", "hf_audio_layer",
    "layerwise_receptive_field", "print_layerwise", "perturbation_receptive_field",
    "sweep_centers", "grid_centers", "position_map", "over_images", "over_clips",
    "tiled_apply", "margin_sweep", "print_margin_sweep",
    "make_mask", "apply_fill", "lesion_latents", "damage_profile", "hole_sweep", "lesion_importance_map",
    "harmonic_fill", "radial_power", "highfreq_ratio", "cut_profile", "margin_needed",
]
# plotting helpers live in receptive_field_scope.plotting (matplotlib imported lazily)
__version__ = "0.4.0"
