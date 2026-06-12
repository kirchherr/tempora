"""Small array helpers for synthetic temporal datasets."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from tempora.data.types import FloatArray, TemporalDataset

IntArray = npt.NDArray[np.int64]


def window_observations(
    dataset: TemporalDataset,
    *,
    window_size: int,
    stride: int = 1,
) -> FloatArray:
    """Return rolling observation windows with shape `(n_windows, window, dim)`."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if window_size > dataset.n_steps:
        raise ValueError("window_size cannot exceed dataset length.")

    starts = range(0, dataset.n_steps - window_size + 1, stride)
    windows = [dataset.observations[start : start + window_size] for start in starts]
    return cast(FloatArray, np.stack(windows, axis=0).astype(np.float64))


def train_test_split_indices(
    dataset: TemporalDataset,
    *,
    train_fraction: float = 0.8,
) -> tuple[IntArray, IntArray]:
    """Create deterministic contiguous train/test indices for time series."""

    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and between 0 and 1.")
    split = int(round(dataset.n_steps * train_fraction))
    split = min(max(split, 1), dataset.n_steps - 1)
    train_indices = np.arange(0, split, dtype=np.int64)
    test_indices = np.arange(split, dataset.n_steps, dtype=np.int64)
    return train_indices, test_indices
