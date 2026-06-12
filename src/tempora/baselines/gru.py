"""GRU baseline with the shared TEMPORA temporal interface."""

from __future__ import annotations

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


class GRUBaseline:
    """Small deterministic GRU next-step baseline."""

    name = "gru"

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        epochs: int = 40,
        learning_rate: float = 0.03,
    ) -> None:
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._network: _GRUNetwork | None = None

    def fit(self, dataset: TemporalDataset, seed: int) -> BaselineMetrics:
        set_reproducible_seed(seed)
        network = _GRUNetwork(dataset.observation_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        observations = observations_tensor(dataset)
        inputs = observations[:, :-1, :]
        targets = observations[:, 1:, :]

        for _ in range(self.epochs):
            optimizer.zero_grad()
            predictions, _states = network(inputs)
            loss = torch.mean((predictions - targets).square())
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        self._network = network
        return baseline_metrics(
            dataset,
            self.predict(dataset),
            fit_epochs=self.epochs,
        )

    def encode(self, dataset: TemporalDataset) -> FloatArray:
        network = cast(_GRUNetwork, ensure_fitted(self._network, name=self.name))
        with torch.no_grad():
            observations = observations_tensor(dataset)
            _predictions, states = network(observations)
        return np.asarray(states.squeeze(0).numpy(), dtype=np.float64)

    def predict(self, dataset: TemporalDataset) -> FloatArray:
        network = cast(_GRUNetwork, ensure_fitted(self._network, name=self.name))
        with torch.no_grad():
            observations = observations_tensor(dataset)
            predictions, _states = network(observations[:, :-1, :])
            full = observations.clone()
            full[:, 1:, :] = predictions
        return np.asarray(full.squeeze(0).numpy(), dtype=np.float64)


class _GRUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states, _hidden = self.gru(inputs)
        predictions = self.readout(states)
        return predictions, states
