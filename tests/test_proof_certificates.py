import json

import pytest
import torch

from tempora.models import ContractiveCTRNN
from tempora.proof import (
    CertificateViolation,
    certify_model_contraction,
    certify_sufficient_contraction,
    require_certificate,
)


def test_sufficient_contraction_certificate_is_json_serializable() -> None:
    damping = torch.tensor([1.2, 1.4])
    recurrent = torch.eye(2) * 0.05

    certificate = certify_sufficient_contraction(
        damping=damping,
        recurrent_weights=recurrent,
        required_margin=0.1,
    )

    assert certificate.is_certified is True
    assert certificate.contraction_margin > 0.1
    assert certificate.theorem == "theorem_01_sufficient_contraction"
    assert certificate.assumptions
    json.dumps(certificate.to_jsonable())
    assert require_certificate(certificate) is certificate


def test_contraction_certificate_reports_uncertified_condition() -> None:
    damping = torch.tensor([0.2, 0.2])
    recurrent = torch.eye(2) * 0.3

    certificate = certify_sufficient_contraction(
        damping=damping,
        recurrent_weights=recurrent,
        required_margin=0.0,
    )

    assert certificate.is_certified is False
    assert certificate.contraction_margin < 0.0
    with pytest.raises(CertificateViolation, match="not certified"):
        require_certificate(certificate)


def test_model_contraction_certificate_uses_model_margin() -> None:
    model = ContractiveCTRNN(
        input_dim=2,
        latent_dim=2,
        damping_init=1.4,
        recurrent_scale=0.02,
        margin=0.1,
    )
    model.project_recurrent_weight_()

    certificate = certify_model_contraction(model)

    assert certificate.required_margin == pytest.approx(model.margin)
    assert certificate.is_certified is True


def test_contraction_certificate_rejects_negative_required_margin() -> None:
    with pytest.raises(ValueError, match="required_margin"):
        certify_sufficient_contraction(
            damping=torch.tensor([1.0]),
            recurrent_weights=torch.zeros(1, 1),
            required_margin=-0.1,
        )
