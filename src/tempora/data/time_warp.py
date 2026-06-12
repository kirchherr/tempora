"""Time-warp augmentation for temporal datasets."""

from __future__ import annotations

import numpy as np

from tempora.data.types import TemporalDataset, with_metadata


def time_warp(
    dataset: TemporalDataset,
    *,
    duration_scale: float = 1.0,
    exponent: float = 1.0,
) -> TemporalDataset:
    """Warp sampling times while preserving the sampled trajectory values.

    The operation changes the time grid only. This is useful for controlled
    invariance tests where the trajectory class and sampled states should remain
    fixed while the temporal parametrization changes.
    """

    if not np.isfinite(duration_scale) or duration_scale <= 0.0:
        raise ValueError("duration_scale must be finite and positive.")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("exponent must be finite and positive.")

    start = float(dataset.times[0])
    duration = float(dataset.times[-1] - dataset.times[0]) * duration_scale
    normalized = (dataset.times - dataset.times[0]) / (
        dataset.times[-1] - dataset.times[0]
    )
    warped_times = start + duration * np.power(normalized, exponent)
    metadata = with_metadata(
        {
            **dataset.metadata,
            "augmentation": "time_warp",
            "source_system": dataset.metadata.get("system"),
            "duration_scale": float(duration_scale),
            "exponent": float(exponent),
        },
        times=warped_times,
        observations=dataset.observations,
        clean_states=dataset.clean_states,
    )
    return TemporalDataset(
        times=warped_times.astype(np.float64),
        observations=dataset.observations.copy(),
        clean_states=dataset.clean_states.copy(),
        metadata=metadata,
    )
