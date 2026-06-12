"""Invariance and robustness scores for finite temporal representations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tempora.data import (
    TemporalDataset,
    add_gaussian_noise,
    mask_missing_segment,
    time_warp,
)

FloatArray = npt.NDArray[np.float64]


def representation_distance(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    eps: float = 1e-12,
) -> float:
    """Return normalized RMSE between two same-shaped representations."""

    first_array = _validate_representation(first, name="first")
    second_array = _validate_representation(second, name="second")
    if first_array.shape != second_array.shape:
        raise ValueError("representations must have identical shapes.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    rmse = np.sqrt(np.mean(np.square(first_array - second_array)))
    scale = (
        np.sqrt(np.mean(np.square(first_array - np.mean(first_array, axis=0)))) + eps
    )
    return float(rmse / scale)


def representation_similarity_score(
    first: npt.ArrayLike, second: npt.ArrayLike
) -> float:
    """Return `1 / (1 + normalized_distance)` for two representations."""

    distance = representation_distance(first, second)
    return float(1.0 / (1.0 + distance))


def time_warp_invariance_score(
    dataset: TemporalDataset,
    *,
    duration_scale: float = 1.25,
    exponent: float = 1.2,
) -> float:
    """Score invariance to controlled time-grid warping."""

    warped = time_warp(dataset, duration_scale=duration_scale, exponent=exponent)
    return representation_similarity_score(dataset.observations, warped.observations)


def noise_robustness_score(
    dataset: TemporalDataset,
    *,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> float:
    """Score robustness to finite Gaussian observation noise."""

    noisy = add_gaussian_noise(dataset, noise_std=noise_std, seed=seed)
    return representation_similarity_score(dataset.observations, noisy.observations)


def missing_segment_robustness_score(
    dataset: TemporalDataset,
    *,
    fraction: float = 0.1,
    seed: int | None = None,
    mask_value: float = 0.0,
) -> float:
    """Score robustness to a finite contiguous missing-observation segment."""

    masked = mask_missing_segment(
        dataset,
        fraction=fraction,
        seed=seed,
        mask_value=mask_value,
    )
    return representation_similarity_score(dataset.observations, masked.observations)


def _validate_representation(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_steps, dim).")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array
