"""Rossler system trajectory generator."""

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


def generate_rossler(
    *,
    n_steps: int = 512,
    duration: float = 40.0,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    initial_state: Sequence[float] | None = None,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> TemporalDataset:
    """Generate a finite Rossler trajectory using deterministic ODE integration."""

    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    times = make_time_grid(n_steps=n_steps, duration=duration)
    y0 = _initial_state(initial_state=initial_state, rng=rng)

    solution = solve_ivp(
        lambda _time, state: _rossler_rhs(
            state=state,
            a=float(a),
            b=float(b),
            c=float(c),
        ),
        t_span=(float(times[0]), float(times[-1])),
        y0=y0,
        t_eval=times,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Rossler integration failed: {solution.message}")
    clean_states = np.asarray(solution.y.T, dtype=np.float64)
    observations = clean_states.copy()
    if noise_std > 0.0:
        observations += rng.normal(0.0, noise_std, size=observations.shape)

    metadata = with_metadata(
        {
            "system": "rossler",
            "seed": seed,
            "a": float(a),
            "b": float(b),
            "c": float(c),
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


def _rossler_rhs(
    *,
    state: FloatArray,
    a: float,
    b: float,
    c: float,
) -> FloatArray:
    x, y, z = state
    return np.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ],
        dtype=np.float64,
    )


def _initial_state(
    *,
    initial_state: Sequence[float] | None,
    rng: np.random.Generator,
) -> FloatArray:
    if initial_state is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64) + rng.normal(
            0.0, 0.01, size=3
        )
    array = np.asarray(initial_state, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("initial_state must be a finite 3-vector.")
    return array
