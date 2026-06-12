"""Torus trajectory generator."""

from __future__ import annotations

import numpy as np

from tempora.data.types import TemporalDataset, make_time_grid, with_metadata


def generate_torus(
    *,
    n_steps: int = 256,
    duration: float = 2.0 * np.pi,
    major_radius: float = 2.0,
    minor_radius: float = 0.5,
    major_angular_velocity: float = 1.0,
    minor_angular_velocity: float = 3.0,
    phase_major: float | None = None,
    phase_minor: float | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> TemporalDataset:
    """Generate a 3D curve constrained to the surface of a torus."""

    if not np.isfinite(major_radius) or major_radius <= 0.0:
        raise ValueError("major_radius must be finite and positive.")
    if not np.isfinite(minor_radius) or minor_radius <= 0.0:
        raise ValueError("minor_radius must be finite and positive.")
    if minor_radius >= major_radius:
        raise ValueError("minor_radius must be smaller than major_radius.")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    resolved_phase_major = (
        float(rng.uniform(0.0, 2.0 * np.pi))
        if phase_major is None
        else float(phase_major)
    )
    resolved_phase_minor = (
        float(rng.uniform(0.0, 2.0 * np.pi))
        if phase_minor is None
        else float(phase_minor)
    )
    times = make_time_grid(n_steps=n_steps, duration=duration)
    u = major_angular_velocity * times + resolved_phase_major
    v = minor_angular_velocity * times + resolved_phase_minor
    radial = major_radius + minor_radius * np.cos(v)
    clean_states = np.column_stack(
        (
            radial * np.cos(u),
            radial * np.sin(u),
            minor_radius * np.sin(v),
        )
    ).astype(np.float64)
    observations = clean_states.copy()
    if noise_std > 0.0:
        observations += rng.normal(0.0, noise_std, size=observations.shape)

    metadata = with_metadata(
        {
            "system": "torus",
            "seed": seed,
            "major_radius": float(major_radius),
            "minor_radius": float(minor_radius),
            "major_angular_velocity": float(major_angular_velocity),
            "minor_angular_velocity": float(minor_angular_velocity),
            "phase_major": resolved_phase_major,
            "phase_minor": resolved_phase_minor,
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
