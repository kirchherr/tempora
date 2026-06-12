"""Minimal training loops for early TEMPORA smoke experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tempora.data import TemporalDataset
from tempora.models import ContractiveCTRNN
from tempora.models.plasticity import apply_projected_oja_update_
from tempora.training.callbacks import TrainingRecord, records_to_metrics
from tempora.training.losses import next_step_prediction_loss


@dataclass(frozen=True)
class TrainingResult:
    """Result of a small deterministic training run."""

    records: list[TrainingRecord]
    metrics: dict[str, object]


def train_circle_next_step(
    model: ContractiveCTRNN,
    dataset: TemporalDataset,
    *,
    epochs: int = 40,
    learning_rate: float = 0.03,
    plasticity_learning_rate: float = 1e-3,
    plasticity_stabilization: float = 1.0,
    homeostatic_decay: float = 1e-4,
) -> TrainingResult:
    """Train a tiny next-step observation smoke task on a circle-like dataset.

    This helper is deliberately narrow: it assumes `latent_dim == observation_dim`
    and uses the latent states directly as predictions. It exists to verify the
    mechanics of projected optimization and plasticity, not to report benchmark
    performance.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if model.latent_dim != dataset.observation_dim:
        raise ValueError("model.latent_dim must equal dataset.observation_dim.")
    if model.input_dim != dataset.observation_dim:
        raise ValueError("model.input_dim must equal dataset.observation_dim.")

    observations = torch.as_tensor(
        dataset.observations,
        dtype=torch.float32,
    ).unsqueeze(0)
    times = torch.as_tensor(dataset.times, dtype=torch.float32)
    initial_state = observations[:, 0, :]

    optimizer = torch.optim.Adam(
        [
            model.recurrent_weight,
            model.input_weight,
            model.bias,
        ],
        lr=learning_rate,
    )
    model.project_recurrent_weight_()

    records: list[TrainingRecord] = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        states = model(observations, times=times, initial_state=initial_state)
        loss = next_step_prediction_loss(states, observations)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        margin = model.project_recurrent_weight_()

        plasticity_margin_after: float | None = None
        if plasticity_learning_rate > 0.0:
            plasticity_log = apply_projected_oja_update_(
                model,
                states.detach(),
                learning_rate=plasticity_learning_rate,
                stabilization=plasticity_stabilization,
                homeostatic_decay=homeostatic_decay,
            )
            plasticity_margin_after = plasticity_log.margin_after
            margin = model.sufficient_contraction_margin()

        records.append(
            TrainingRecord(
                epoch=epoch,
                loss=float(loss.detach().item()),
                contraction_margin=float(margin.detach().item()),
                recurrent_weight_norm=float(
                    torch.linalg.matrix_norm(model.recurrent_weight.detach()).item()
                ),
                plasticity_margin_after=plasticity_margin_after,
            )
        )

    metrics = records_to_metrics(records)
    if not np.isfinite([record.loss for record in records]).all():
        raise RuntimeError("non-finite training loss encountered.")
    return TrainingResult(records=records, metrics=metrics)
