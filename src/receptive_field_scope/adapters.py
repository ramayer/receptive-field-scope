"""Adapters so models with other layouts fit the `[1, C, *S] -> [1, C', *s]` callable contract."""
from __future__ import annotations

import torch


def channels_last(fn):
    """Wrap a callable that returns [1, *s, C'] (e.g. transformer hidden states [1, T, C]) so it returns [1, C', *s]."""
    def wrapped(x):
        y = fn(x)
        return y.movedim(-1, 1)
    return wrapped


def waveform_model(fn, *, takes_channel_dim: bool = False):
    """Wrap a model taking a waveform [B, T] so it accepts [1, 1, T]. Combine with channels_last if needed."""
    def wrapped(x):
        return fn(x if takes_channel_dim else x[:, 0])
    return wrapped


def hf_audio_layer(model, layer: int | None = None, *, extractor_only: bool = False):
    """Callable [1, 1, T] -> [1, C, frames] for a Hugging Face wav2vec2/HuBERT-style model (includes AVES
    checkpoints loaded as HubertModel).

      extractor_only=True  -> the convolutional feature extractor output (before the transformer).
      layer=None           -> last_hidden_state; layer=k -> hidden_states[k] (0 = after projection + positional conv).
    The model must be in eval() mode. Padding masks are not used (single clip)."""
    def fn(x):
        wav = x[:, 0]
        if extractor_only:
            return model.feature_extractor(wav)                           # [1, C, frames]
        out = model(wav, output_hidden_states=layer is not None)
        h = out.last_hidden_state if layer is None else out.hidden_states[layer]
        return h.movedim(-1, 1)                                           # [1, frames, C] -> [1, C, frames]
    return fn
