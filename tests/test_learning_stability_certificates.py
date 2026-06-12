import json

import pytest
import torch

from tempora.models import ContractiveCTRNN, apply_projected_oja_update_
from tempora.proof import (
    CertificateViolation,
    certify_projected_update_stability,
    require_certificate,
)


def test_projected_update_stability_certificate_is_json_serializable() -> None:
    torch.manual_seed(901)
    model = ContractiveCTRNN(input_dim=2, latent_dim=3, damping_init=1.3, margin=0.1)
    model.project_recurrent_weight_()
    activations = torch.randn(4, 6, 3) * 0.1

    log = apply_projected_oja_update_(
        model,
        activations,
        learning_rate=0.05,
        stabilization=0.5,
        homeostatic_decay=1e-3,
    )
    certificate = certify_projected_update_stability(
        log,
        required_margin=model.margin,
    )

    assert certificate.theorem == "theorem_02_projected_learning_stability"
    assert certificate.is_certified is True
    assert certificate.margin_after > model.margin
    assert certificate.assumptions
    json.dumps(certificate.to_jsonable())
    assert require_certificate(certificate) is certificate


def test_projected_update_certificate_reports_failed_margin() -> None:
    torch.manual_seed(902)
    model = ContractiveCTRNN(input_dim=1, latent_dim=2, damping_init=1.0, margin=0.05)
    model.project_recurrent_weight_()
    log = apply_projected_oja_update_(
        model,
        torch.randn(3, 2) * 0.1,
        learning_rate=0.01,
    )

    certificate = certify_projected_update_stability(
        log,
        required_margin=log.margin_after + 1.0,
    )

    assert certificate.is_certified is False
    with pytest.raises(CertificateViolation, match="not certified"):
        require_certificate(certificate)


def test_projected_update_certificate_rejects_negative_required_margin() -> None:
    torch.manual_seed(903)
    model = ContractiveCTRNN(input_dim=1, latent_dim=2)
    model.project_recurrent_weight_()
    log = apply_projected_oja_update_(
        model,
        torch.randn(3, 2) * 0.1,
        learning_rate=0.01,
    )

    with pytest.raises(ValueError, match="required_margin"):
        certify_projected_update_stability(log, required_margin=-0.1)
