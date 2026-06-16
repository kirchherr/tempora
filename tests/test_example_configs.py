from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tempora.data import (
    TemporalDataset,
    generate_circle,
    generate_lorenz,
    generate_rossler,
    generate_torus,
)
from tempora.models import ContractiveCTRNN

ROOT = Path(".")
DATASET_CONFIGS: dict[str, Callable[..., TemporalDataset]] = {
    "circle": generate_circle,
    "torus": generate_torus,
    "lorenz": generate_lorenz,
    "rossler": generate_rossler,
}


def test_example_dataset_configs_generate_finite_datasets() -> None:
    for config_path in sorted((ROOT / "configs").glob("synth_*.yaml")):
        config = _read_yaml(config_path)
        dataset_name = str(config.pop("dataset"))
        generator = DATASET_CONFIGS[dataset_name]

        dataset = generator(**config)

        assert dataset.metadata["system"] == dataset_name
        assert dataset.n_steps == int(config["n_steps"])
        assert np.isfinite(dataset.observations).all()
        assert np.isfinite(dataset.clean_states).all()
        assert dataset.observations.shape == dataset.clean_states.shape


def test_contractive_ctrnn_example_config_builds_contractive_model() -> None:
    config = _read_yaml(ROOT / "configs/contractive_ctrnn.yaml")
    assert config.pop("model") == "contractive_ctrnn"

    model = ContractiveCTRNN(**config)
    margin = model.project_recurrent_weight_()

    assert model.input_dim == 2
    assert model.latent_dim == 2
    assert float(margin.detach().item()) > model.margin


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must contain a mapping"
    return dict(payload)
