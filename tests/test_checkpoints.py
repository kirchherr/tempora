from pathlib import Path

import pytest
import torch

from tempora.models import ContractiveCTRNN
from tempora.training import (
    load_contractive_ctrnn_checkpoint,
    save_contractive_ctrnn_checkpoint,
)


def test_contractive_ctrnn_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = ContractiveCTRNN(input_dim=2, latent_dim=3, damping_init=1.3, margin=0.1)
    model.project_recurrent_weight_()
    path = tmp_path / "model.pt"

    saved_path = save_contractive_ctrnn_checkpoint(
        model,
        path,
        dataset_name="circle",
        seed=17,
        config={"run_id": "unit", "datasets": ["circle"]},
        training_metrics={"epochs": 1, "final_loss": 0.25},
    )
    loaded = load_contractive_ctrnn_checkpoint(saved_path)

    assert loaded.model.input_dim == 2
    assert loaded.model.latent_dim == 3
    assert loaded.metadata["dataset"] == "circle"
    assert loaded.metadata["seed"] == 17
    assert loaded.metadata["config"]["run_id"] == "unit"
    assert loaded.metadata["training_metrics"]["final_loss"] == 0.25
    for name, tensor in model.state_dict().items():
        assert torch.allclose(loaded.model.state_dict()[name], tensor)


def test_contractive_ctrnn_checkpoint_rejects_wrong_model_class(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_class": "OtherModel"}, path)

    with pytest.raises(ValueError, match="model_class"):
        load_contractive_ctrnn_checkpoint(path)
