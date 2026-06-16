import json

import torch

from tempora.data import generate_circle
from tempora.models import ContractiveCTRNN
from tempora.training import train_circle_next_step


def test_circle_smoke_training_decreases_loss_and_preserves_margin() -> None:
    torch.manual_seed(301)
    dataset = generate_circle(
        n_steps=36,
        duration=1.2,
        phase=0.0,
        noise_std=0.0,
        seed=302,
    )
    model = ContractiveCTRNN(
        input_dim=2,
        latent_dim=2,
        damping_init=1.4,
        margin=0.1,
        recurrent_scale=0.02,
    )

    result = train_circle_next_step(
        model,
        dataset,
        epochs=35,
        learning_rate=0.04,
        plasticity_learning_rate=5e-4,
    )

    assert result.metrics["loss_decreased"] is True
    assert result.metrics["final_loss"] < result.metrics["initial_loss"]
    assert result.metrics["contraction_margin_min"] > 0.09
    assert result.last_plasticity_log is not None
    assert result.last_plasticity_log.margin_after > model.margin
    plasticity_last_update = result.metrics["plasticity_last_update"]
    assert isinstance(plasticity_last_update, dict)
    assert plasticity_last_update["margin_after"] > model.margin
    assert len(result.records) == 35
    json.dumps(result.metrics)
