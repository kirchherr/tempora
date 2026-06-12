"""Small callback-style records for TEMPORA training diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingRecord:
    """JSON-serializable metrics for one training epoch."""

    epoch: int
    loss: float
    contraction_margin: float
    recurrent_weight_norm: float
    plasticity_margin_after: float | None = None

    def to_metrics(self) -> dict[str, float | int | None]:
        return asdict(self)


def records_to_metrics(records: list[TrainingRecord]) -> dict[str, Any]:
    """Convert training records into a JSON-serializable metrics payload."""

    if not records:
        raise ValueError("records must not be empty.")
    losses = [record.loss for record in records]
    margins = [record.contraction_margin for record in records]
    return {
        "epochs": len(records),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "contraction_margin_min": min(margins),
        "contraction_margin_final": margins[-1],
        "records": [record.to_metrics() for record in records],
    }
