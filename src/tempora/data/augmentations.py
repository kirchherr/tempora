"""Finite perturbations for synthetic temporal datasets."""

from __future__ import annotations

import numpy as np

from tempora.data.types import TemporalDataset, with_metadata


def add_gaussian_noise(
    dataset: TemporalDataset,
    *,
    noise_std: float,
    seed: int | None = None,
) -> TemporalDataset:
    """Add finite Gaussian noise to observations while keeping clean states."""

    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    observations = dataset.observations.copy()
    if noise_std > 0.0:
        observations += rng.normal(0.0, noise_std, size=observations.shape)
    metadata = with_metadata(
        {
            **dataset.metadata,
            "augmentation": "gaussian_noise",
            "source_system": dataset.metadata.get("system"),
            "noise_std": float(noise_std),
            "augmentation_seed": seed,
        },
        times=dataset.times,
        observations=observations,
        clean_states=dataset.clean_states,
    )
    return TemporalDataset(
        times=dataset.times.copy(),
        observations=observations,
        clean_states=dataset.clean_states.copy(),
        metadata=metadata,
    )


def mask_missing_segment(
    dataset: TemporalDataset,
    *,
    fraction: float,
    seed: int | None = None,
    mask_value: float = 0.0,
) -> TemporalDataset:
    """Replace one contiguous observation segment with a finite mask value."""

    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be finite and between 0 and 1.")
    if not np.isfinite(mask_value):
        raise ValueError("mask_value must be finite.")

    n_steps = dataset.n_steps
    length = max(1, int(round(n_steps * fraction)))
    if length >= n_steps:
        length = n_steps - 1
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n_steps - length + 1))
    stop = start + length

    observations = dataset.observations.copy()
    observations[start:stop, :] = float(mask_value)
    metadata = with_metadata(
        {
            **dataset.metadata,
            "augmentation": "missing_segment_mask",
            "source_system": dataset.metadata.get("system"),
            "augmentation_seed": seed,
            "missing_start": start,
            "missing_stop": stop,
            "missing_fraction": float(fraction),
            "mask_value": float(mask_value),
        },
        times=dataset.times,
        observations=observations,
        clean_states=dataset.clean_states,
    )
    return TemporalDataset(
        times=dataset.times.copy(),
        observations=observations,
        clean_states=dataset.clean_states.copy(),
        metadata=metadata,
    )
