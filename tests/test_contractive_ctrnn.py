import torch

from tempora.metrics import model_contraction_margin, symmetric_jacobian_max_eigenvalue
from tempora.models import ContractiveCTRNN


def test_model_projection_restores_positive_contraction_margin() -> None:
    torch.manual_seed(101)
    model = ContractiveCTRNN(input_dim=2, latent_dim=4, damping_init=1.2, margin=0.1)
    with torch.no_grad():
        model.recurrent_weight.mul_(100.0)

    assert model.sufficient_contraction_margin() < 0.0

    margin = model.project_recurrent_weight_()

    assert margin > 0.09
    assert model_contraction_margin(model) > 0.09


def test_ctrnn_forward_is_finite_and_has_expected_shape() -> None:
    torch.manual_seed(102)
    model = ContractiveCTRNN(input_dim=2, latent_dim=3, damping_init=1.5)
    model.project_recurrent_weight_()
    inputs = torch.randn(5, 7, 2) * 0.1
    times = torch.linspace(0.0, 0.6, 7)

    states = model(inputs, times=times)

    assert states.shape == (5, 7, 3)
    assert torch.isfinite(states).all()


def test_ctrnn_backward_computes_finite_gradients() -> None:
    torch.manual_seed(103)
    model = ContractiveCTRNN(input_dim=2, latent_dim=3, damping_init=1.5)
    model.project_recurrent_weight_()
    inputs = torch.randn(4, 6, 2) * 0.1
    states = model(inputs, times=torch.linspace(0.0, 0.5, 6))

    loss = states.square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_torchdiffeq_integration_path_is_finite() -> None:
    torch.manual_seed(104)
    model = ContractiveCTRNN(
        input_dim=1,
        latent_dim=2,
        damping_init=1.2,
        use_torchdiffeq=True,
    )
    model.project_recurrent_weight_()
    inputs = torch.zeros(2, 4, 1)

    states = model(inputs, times=torch.linspace(0.0, 0.3, 4))

    assert states.shape == (2, 4, 2)
    assert torch.isfinite(states).all()


def test_symmetric_jacobian_diagnostic_is_negative_for_projected_model() -> None:
    torch.manual_seed(105)
    model = ContractiveCTRNN(input_dim=1, latent_dim=3, damping_init=2.0, margin=0.2)
    model.project_recurrent_weight_()
    states = torch.randn(8, 3) * 0.05
    inputs = torch.zeros(8, 1)

    max_eigenvalue = symmetric_jacobian_max_eigenvalue(
        model.dynamics,
        states,
        inputs,
    )

    assert max_eigenvalue < -0.1
