"""Shared datatypes and validation for synthetic temporal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TemporalDataset:
    """Finite temporal trajectory with clean states and observed values.

    Shape conventions:
    - `times`: `(n_steps,)`
    - `observations`: `(n_steps, observation_dim)`
    - `clean_states`: `(n_steps, state_dim)`

    `observations` may include finite perturbations such as Gaussian noise or
    zero-filled missing segments. `clean_states` should remain the unperturbed
    system trajectory used for controlled experiments.
    """

    times: FloatArray
    observations: FloatArray
    clean_states: FloatArray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        validate_temporal_dataset(self)

    @property
    def n_steps(self) -> int:
        return int(self.times.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.clean_states.shape[1])


def as_float_array(values: npt.ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Convert an array-like value to finite float64 with a required rank."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        msg = f"{name} must have {ndim} dimensions, got shape {array.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return array


def validate_temporal_dataset(dataset: TemporalDataset) -> None:
    """Validate TEMPORA's common finite trajectory invariants."""

    times = as_float_array(dataset.times, name="times", ndim=1)
    observations = as_float_array(dataset.observations, name="observations", ndim=2)
    clean_states = as_float_array(dataset.clean_states, name="clean_states", ndim=2)

    n_steps = times.shape[0]
    if n_steps < 2:
        raise ValueError("times must contain at least two samples.")
    if observations.shape[0] != n_steps:
        raise ValueError("observations and times must have matching first axes.")
    if clean_states.shape[0] != n_steps:
        raise ValueError("clean_states and times must have matching first axes.")
    if observations.shape[1] < 1:
        raise ValueError("observations must have at least one feature.")
    if clean_states.shape[1] < 1:
        raise ValueError("clean_states must have at least one feature.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    if not isinstance(dataset.metadata, dict):
        raise TypeError("metadata must be a dictionary.")


def make_time_grid(*, n_steps: int, duration: float) -> FloatArray:
    """Create a strictly increasing finite time grid."""

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive.")
    return np.linspace(0.0, float(duration), int(n_steps), dtype=np.float64)


def with_metadata(
    metadata: dict[str, Any],
    *,
    times: FloatArray,
    observations: FloatArray,
    clean_states: FloatArray,
) -> dict[str, Any]:
    """Attach standard shape metadata to a dataset-specific metadata mapping."""

    return {
        **metadata,
        "n_steps": int(times.shape[0]),
        "duration": float(times[-1] - times[0]),
        "time_shape": tuple(int(value) for value in times.shape),
        "observation_shape": tuple(int(value) for value in observations.shape),
        "clean_state_shape": tuple(int(value) for value in clean_states.shape),
    }
