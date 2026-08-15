"""
Safe checkpoint loading for the PyTorch models.

`torch.load(..., weights_only=False)` runs arbitrary pickle during load, so a
checkpoint file is effectively executable. These checkpoints only ever contain a
state dict plus plain metadata, so they can be loaded with `weights_only=True`.

One wrinkle: checkpoints written before this change stored training history as
numpy scalars, which the restricted unpickler rejects. Rather than falling back
to unrestricted loading, the numpy scalar reconstructors are explicitly allow-listed
— that permits exactly those types and nothing else.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Reconstructors needed to read numpy scalars out of legacy checkpoints.
# numpy 2.x exposes these under the private `_core`; `core` is a deprecated shim.
_multiarray = getattr(np, "_core", None) or np.core
_NUMPY_SAFE_GLOBALS = [
    _multiarray.multiarray.scalar,
    np.dtype,
]
for _name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType"):
    _dtype_cls = getattr(np.dtypes, _name, None)
    if _dtype_cls is not None:
        _NUMPY_SAFE_GLOBALS.append(_dtype_cls)


def safe_torch_load(path: str, map_location=None) -> dict:
    """Load a checkpoint without granting it arbitrary code execution."""
    with torch.serialization.safe_globals(_NUMPY_SAFE_GLOBALS):
        return torch.load(path, map_location=map_location, weights_only=True)


def to_plain(value):
    """
    Convert numpy scalars/arrays to built-in types so new checkpoints need no
    allow-list at all.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value
