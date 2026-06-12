"""Unconstrained Neural-ODE-style baseline."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from torch import nn

from tempora.baselines.common import (
    baseline_metrics,
    ensure_fitted,
    observations_tensor,
    set_reproducible_seed,
)
from tempora.baselines.protocol import BaselineMetrics
from tempora.data import TemporalDataset
from tempora.data.types import FloatArray


class VanillaNeuralODEBaseline:
    """Unconstrained continuous-time baseline without contraction projection."""

    name = "vanilla_neural_ode"

    def __init__(self, *, epochs: int = 50, learning_rate: float = 0.03) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._model: _VanillaODEModel | None = None

    def fit(self, dataset: TemporalDataset, seed: int) -> BaselineMetrics:
        set_reproducible_seed(seed)
        model = _VanillaODEModel(dataset.observation_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        observations = observations_tensor(dataset)
        times = torch.as_tensor(dataset.times, dtype=torch.float32)
        initial_state = observations[:, 0, :]

        for _ in range(self.epochs):
            optimizer.zero_grad()
            states = model(observations, times=times, initial_state=initial_state)
            loss = torch.mean((states[:, 1:, :] - observations[:, 1:, :]).square())
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        self._model = model
        return baseline_metrics(dataset, self.predict(dataset), fit_epochs=self.epochs)

    def encode(self, dataset: TemporalDataset) -> FloatArray:
        model = cast(_VanillaODEModel, ensure_fitted(self._model, name=self.name))
        with torch.no_grad():
            states = _predict_states(model, dataset)
        return np.asarray(states.squeeze(0).numpy(), dtype=np.float64)

    def predict(self, dataset: TemporalDataset) -> FloatArray:
        return self.encode(dataset)


class _VanillaODEModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.recurrent_weight = nn.Parameter(torch.empty(dim, dim))
        self.input_weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.recurrent_weight)
        nn.init.xavier_uniform_(self.input_weight)

    def dynamics(self, state: torch.Tensor, input_value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            state @ self.recurrent_weight.transpose(0, 1)
            + input_value @ self.input_weight.transpose(0, 1)
            + self.bias
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        times: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        states = [initial_state]
        state = initial_state
        for step in range(int(inputs.shape[1] - 1)):
            dt = times[step + 1] - times[step]
            state = _rk4_step(
                dynamics=self.dynamics,
                state=state,
                input_value=inputs[:, step, :],
                dt=dt,
            )
            states.append(state)
        return torch.stack(states, dim=1)


def _predict_states(model: _VanillaODEModel, dataset: TemporalDataset) -> torch.Tensor:
    observations = observations_tensor(dataset)
    times = torch.as_tensor(dataset.times, dtype=torch.float32)
    return cast(
        torch.Tensor,
        model(observations, times=times, initial_state=observations[:, 0, :]),
    )


def _rk4_step(
    *,
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    input_value: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    half_dt = 0.5 * dt
    k1 = dynamics(state, input_value)
    k2 = dynamics(state + half_dt * k1, input_value)
    k3 = dynamics(state + half_dt * k2, input_value)
    k4 = dynamics(state + dt * k3, input_value)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
