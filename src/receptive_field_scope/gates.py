"""Helpers for models with zero-init residual gates (LayerScale / ReZero style)."""
from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def activate_gates(model: torch.nn.Module, value: float = 1.0, match: str = "gate"):
    """Temporarily set every parameter whose name contains `match` to `value`.

    A fresh model with zero-init gates has no gradient path through gated branches, so it measures as
    'conv path only'. Open the gates to see the structural worst case that training could grow into."""
    params = dict(model.named_parameters())
    saved = {n: p.detach().clone() for n, p in params.items() if match in n}
    if not saved:
        raise ValueError(f"no parameter name contains {match!r}")
    with torch.no_grad():
        for n in saved:
            params[n].fill_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            for n, v in saved.items():
                params[n].copy_(v)
