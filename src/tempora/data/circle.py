"""Circle trajectory generator."""

from __future__ import annotations

import numpy as np

from tempora.data.types import TemporalDataset, make_time_grid, with_metadata


def generate_circle(
    *,
    n_steps: int = 256,
    duration: float = 2.0 * np.pi,
    radius: float = 1.0,
    angular_velocity: float = 1.0,
    phase: float | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> TemporalDataset:
    """Generate a deterministic noisy observation of a 2D circle trajectory."""

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive.")
    if not np.isfinite(angular_velocity):
        raise ValueError("angular_velocity must be finite.")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    resolved_phase = (
        float(rng.uniform(0.0, 2.0 * np.pi)) if phase is None else float(phase)
    )
    times = make_time_grid(n_steps=n_steps, duration=duration)
    theta = angular_velocity * times + resolved_phase
    clean_states = np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta))
    ).astype(np.float64)
    observations = clean_states.copy()
    if noise_std > 0.0:
        observations += rng.normal(0.0, noise_std, size=observations.shape)

    metadata = with_metadata(
        {
            "system": "circle",
            "seed": seed,
            "radius": float(radius),
            "angular_velocity": float(angular_velocity),
            "phase": resolved_phase,
            "noise_std": float(noise_std),
        },
        times=times,
        observations=observations,
        clean_states=clean_states,
    )
    return TemporalDataset(
        times=times,
        observations=observations,
        clean_states=clean_states,
        metadata=metadata,
    )
