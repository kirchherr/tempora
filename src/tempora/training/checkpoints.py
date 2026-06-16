"""Checkpoint helpers for TEMPORA training and benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

from tempora.models import ContractiveCTRNN


@dataclass(frozen=True)
class ContractiveCTRNNCheckpoint:
    """Loaded checkpoint payload with a reconstructed contractive CTRNN."""

    model: ContractiveCTRNN
    metadata: dict[str, Any]


class ContractiveCTRNNKwargs(TypedDict):
    """Constructor kwargs needed to reconstruct a contractive CTRNN."""

    input_dim: int
    latent_dim: int
    margin: float
    lipschitz: float
    use_torchdiffeq: bool


def save_contractive_ctrnn_checkpoint(
    model: ContractiveCTRNN,
    path: Path,
    *,
    dataset_name: str,
    seed: int,
    config: Mapping[str, object],
    training_metrics: Mapping[str, object],
) -> Path:
    """Save a reconstructable `ContractiveCTRNN` checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "ContractiveCTRNN",
            "model_kwargs": {
                "input_dim": model.input_dim,
                "latent_dim": model.latent_dim,
                "margin": model.margin,
                "lipschitz": model.lipschitz,
                "use_torchdiffeq": model.use_torchdiffeq,
            },
            "dataset": dataset_name,
            "seed": int(seed),
            "config": dict(config),
            "training_metrics": dict(training_metrics),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def load_contractive_ctrnn_checkpoint(
    path: Path,
    *,
    map_location: str = "cpu",
) -> ContractiveCTRNNCheckpoint:
    """Load a `ContractiveCTRNN` checkpoint and reconstruct its model."""

    raw = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must contain a mapping payload.")
    if raw.get("model_class") != "ContractiveCTRNN":
        raise ValueError("checkpoint model_class must be 'ContractiveCTRNN'.")

    model_kwargs = _validated_model_kwargs(raw.get("model_kwargs"))
    state_dict = raw.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain a state_dict mapping.")

    model = ContractiveCTRNN(**model_kwargs)
    model.load_state_dict(cast(Mapping[str, Any], state_dict))
    metadata = {key: value for key, value in raw.items() if key != "state_dict"}
    return ContractiveCTRNNCheckpoint(model=model, metadata=metadata)


def _validated_model_kwargs(value: object) -> ContractiveCTRNNKwargs:
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain model_kwargs.")
    input_dim = _required_int(value, "input_dim")
    latent_dim = _required_int(value, "latent_dim")
    margin = _required_float(value, "margin")
    lipschitz = _required_float(value, "lipschitz")
    use_torchdiffeq = _required_bool(value, "use_torchdiffeq")
    return {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "margin": margin,
        "lipschitz": lipschitz,
        "use_torchdiffeq": use_torchdiffeq,
    }


def _required_int(mapping: dict[Any, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"checkpoint model_kwargs.{key} must be an integer.")
    return value


def _required_float(mapping: dict[Any, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"checkpoint model_kwargs.{key} must be numeric.")
    return float(value)


def _required_bool(mapping: dict[Any, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"checkpoint model_kwargs.{key} must be boolean.")
    return value
