"""Simple reservoir-computing baseline."""

from __future__ import annotations

import numpy as np

from tempora.baselines.common import baseline_metrics, validate_prediction
from tempora.baselines.protocol import BaselineMetrics
from tempora.data import TemporalDataset
from tempora.data.types import FloatArray


class ReservoirBaseline:
    """Fixed random reservoir with ridge readout."""

    name = "reservoir"

    def __init__(
        self,
        *,
        reservoir_dim: int = 16,
        spectral_scale: float = 0.8,
        ridge: float = 1e-4,
    ) -> None:
        if reservoir_dim < 1:
            raise ValueError("reservoir_dim must be positive.")
        if spectral_scale <= 0.0:
            raise ValueError("spectral_scale must be positive.")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive.")
        self.reservoir_dim = int(reservoir_dim)
        self.spectral_scale = float(spectral_scale)
        self.ridge = float(ridge)
        self._recurrent: FloatArray | None = None
        self._input: FloatArray | None = None
        self._readout: FloatArray | None = None

    def fit(self, dataset: TemporalDataset, seed: int) -> BaselineMetrics:
        rng = np.random.default_rng(seed)
        recurrent = rng.normal(0.0, 1.0, size=(self.reservoir_dim, self.reservoir_dim))
        norm = float(np.linalg.norm(recurrent, ord=2))
        recurrent = self.spectral_scale * recurrent / max(norm, 1e-12)
        input_weight = rng.normal(
            0.0,
            1.0 / np.sqrt(dataset.observation_dim),
            size=(self.reservoir_dim, dataset.observation_dim),
        )
        self._recurrent = recurrent.astype(np.float64)
        self._input = input_weight.astype(np.float64)
        states = self.encode(dataset)
        design = _with_bias(states)
        targets = dataset.observations
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self._readout = np.linalg.solve(gram, design.T @ targets).astype(np.float64)
        prediction = self.predict(dataset)
        return baseline_metrics(dataset, prediction, fit_epochs=1)

    def encode(self, dataset: TemporalDataset) -> FloatArray:
        if self._recurrent is None or self._input is None:
            raise RuntimeError("reservoir must be fit before calling encode.")
        states = np.zeros((dataset.n_steps, self.reservoir_dim), dtype=np.float64)
        state = np.zeros(self.reservoir_dim, dtype=np.float64)
        for index, observation in enumerate(dataset.observations):
            state = np.tanh(self._recurrent @ state + self._input @ observation)
            states[index] = state
        return states

    def predict(self, dataset: TemporalDataset) -> FloatArray:
        if self._readout is None:
            raise RuntimeError("reservoir must be fit before calling predict.")
        states = self.encode(dataset)
        prediction = _with_bias(states) @ self._readout
        prediction = prediction.astype(np.float64)
        validate_prediction(dataset, prediction)
        return prediction


def _with_bias(states: FloatArray) -> FloatArray:
    return np.column_stack((states, np.ones(states.shape[0], dtype=np.float64))).astype(
        np.float64
    )
