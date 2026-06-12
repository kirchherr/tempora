"""Reconstruction metrics for finite temporal arrays."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reconstruction_mse(target: npt.ArrayLike, reconstruction: npt.ArrayLike) -> float:
    """Return mean squared reconstruction error for same-shaped arrays."""

    target_array = np.asarray(target, dtype=np.float64)
    reconstruction_array = np.asarray(reconstruction, dtype=np.float64)
    if target_array.shape != reconstruction_array.shape:
        raise ValueError("target and reconstruction must have identical shapes.")
    if target_array.size == 0:
        raise ValueError("arrays must not be empty.")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target must contain only finite values.")
    if not np.all(np.isfinite(reconstruction_array)):
        raise ValueError("reconstruction must contain only finite values.")
    return float(np.mean(np.square(target_array - reconstruction_array)))
