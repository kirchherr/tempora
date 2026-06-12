import pytest
import torch

from tempora.models.projections import (
    contraction_margin,
    project_recurrent_weights,
    spectral_norm,
)


def test_projection_reduces_spectral_norm_when_constraint_is_violated() -> None:
    damping = torch.tensor([1.0, 1.2], dtype=torch.float32)
    weights = torch.tensor([[3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    before_norm = spectral_norm(weights)
    before_margin = contraction_margin(damping, weights)
    projected = project_recurrent_weights(weights, damping, margin=0.1)
    after_norm = spectral_norm(projected)
    after_margin = contraction_margin(damping, projected)

    assert before_margin < 0.0
    assert after_norm < before_norm
    assert after_margin > 0.09


def test_projection_leaves_already_valid_weights_unchanged() -> None:
    damping = torch.tensor([2.0, 2.2], dtype=torch.float32)
    weights = torch.tensor([[0.1, 0.0], [0.0, 0.2]], dtype=torch.float32)

    projected = project_recurrent_weights(weights, damping, margin=0.1)

    torch.testing.assert_close(projected, weights)


def test_projection_rejects_margin_larger_than_damping() -> None:
    damping = torch.tensor([0.05, 0.1], dtype=torch.float32)
    weights = torch.eye(2)

    with pytest.raises(ValueError, match="margin"):
        project_recurrent_weights(weights, damping, margin=0.1)
