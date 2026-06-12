import json

import pytest
import torch

from tempora.models import ContractiveCTRNN, apply_projected_oja_update_, oja_delta


def test_repeated_projected_oja_updates_preserve_contraction_margin() -> None:
    torch.manual_seed(201)
    model = ContractiveCTRNN(input_dim=2, latent_dim=4, damping_init=1.3, margin=0.1)
    with torch.no_grad():
        model.recurrent_weight.mul_(50.0)
    model.project_recurrent_weight_()
    initial_norm = torch.linalg.matrix_norm(model.recurrent_weight.detach())

    logs = []
    for _ in range(25):
        activations = torch.randn(3, 8, 4) * 0.2
        log = apply_projected_oja_update_(
            model,
            activations,
            learning_rate=0.05,
            stabilization=0.5,
            homeostatic_decay=1e-3,
        )
        logs.append(log)

    assert all(log.margin_after > 0.09 for log in logs)
    assert (
        torch.linalg.matrix_norm(model.recurrent_weight.detach()) <= initial_norm + 1e-5
    )


def test_projected_oja_update_logs_json_serializable_metrics() -> None:
    torch.manual_seed(202)
    model = ContractiveCTRNN(input_dim=1, latent_dim=2, damping_init=1.0, margin=0.05)
    model.project_recurrent_weight_()
    activations = torch.randn(5, 2) * 0.1

    log = apply_projected_oja_update_(model, activations, learning_rate=0.01)

    assert log.margin_after > 0.04
    json.dumps(log.to_metrics())


def test_oja_delta_rejects_wrong_activation_dimension() -> None:
    weights = torch.eye(3)
    activations = torch.randn(4, 2)

    with pytest.raises(ValueError, match="activation dimension"):
        oja_delta(weights, activations, learning_rate=0.1)
