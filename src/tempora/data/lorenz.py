"""Lorenz system trajectory generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from tempora.data.types import (
    FloatArray,
    TemporalDataset,
    make_time_grid,
    with_metadata,
)


def generate_lorenz(
    *,
    n_steps: int = 512,
    duration: float = 20.0,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial_state: Sequence[float] | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> TemporalDataset:
    """Generate a finite Lorenz trajectory using deterministic ODE integration."""

    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    times = make_time_grid(n_steps=n_steps, duration=duration)
    y0 = _initial_state(initial_state=initial_state, rng=rng)

    solution = solve_ivp(
        lambda _time, state: _lorenz_rhs(
            state=state,
            sigma=float(sigma),
            rho=float(rho),
            beta=float(beta),
        ),
        t_span=(float(times[0]), float(times[-1])),
        y0=y0,
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Lorenz integration failed: {solution.message}")
    clean_states = np.asarray(solution.y.T, dtype=np.float64)
    observations = clean_states.copy()
    if noise_std > 0.0:
        observations += rng.normal(0.0, noise_std, size=observations.shape)

    metadata = with_metadata(
        {
            "system": "lorenz",
            "seed": seed,
            "sigma": float(sigma),
            "rho": float(rho),
            "beta": float(beta),
            "initial_state": tuple(float(value) for value in y0),
            "noise_std": float(noise_std),
            "integrator": "scipy.solve_ivp",
            "rtol": 1e-8,
            "atol": 1e-10,
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


def _lorenz_rhs(
    *,
    state: FloatArray,
    sigma: float,
    rho: float,
    beta: float,
) -> FloatArray:
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        dtype=np.float64,
    )


def _initial_state(
    *,
    initial_state: Sequence[float] | None,
    rng: np.random.Generator,
) -> FloatArray:
    if initial_state is None:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64) + rng.normal(
            0.0, 0.01, size=3
        )
    array = np.asarray(initial_state, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("initial_state must be a finite 3-vector.")
    return array
