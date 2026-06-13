import json

import numpy as np
import pytest

from tempora.data import add_gaussian_noise, generate_circle
from tempora.experiments import evaluate_topology_pair
from tempora.proof import (
    CertificateViolation,
    certify_topology_comparison,
    require_certificate,
)


def test_topology_certificate_accepts_small_empirical_distance() -> None:
    dataset = generate_circle(n_steps=80, phase=0.0, seed=1001)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    latent_points = dataset.clean_states @ rotation
    metrics = evaluate_topology_pair(dataset.clean_states, latent_points, maxdim=1)

    certificate = certify_topology_comparison(
        metrics,
        homology_dim=1,
        max_distance=1e-5,
    )

    assert certificate.theorem == "theorem_03_empirical_persistence_diagram_comparison"
    assert certificate.is_certified is True
    assert certificate.distance <= 1e-5
    assert certificate.input_n_points == 80
    assert certificate.latent_n_points == 80
    assert certificate.assumptions
    json.dumps(certificate.to_jsonable())
    assert require_certificate(certificate) is certificate


def test_topology_certificate_reports_threshold_failure() -> None:
    dataset = generate_circle(n_steps=80, phase=0.0, seed=1002)
    noisy = add_gaussian_noise(dataset, noise_std=0.15, seed=1003)
    metrics = evaluate_topology_pair(dataset.clean_states, noisy.observations, maxdim=1)

    certificate = certify_topology_comparison(
        metrics,
        homology_dim=1,
        max_distance=0.0,
    )

    assert certificate.is_certified is False
    with pytest.raises(CertificateViolation, match="not certified"):
        require_certificate(certificate)


def test_topology_certificate_validates_threshold_and_metric_payload() -> None:
    with pytest.raises(ValueError, match="max_distance"):
        certify_topology_comparison({}, max_distance=-1.0)

    with pytest.raises(ValueError, match="bottleneck_h1"):
        certify_topology_comparison(
            {
                "metric": "bottleneck",
                "input_n_points": 4,
                "latent_n_points": 4,
            },
            homology_dim=1,
            max_distance=1.0,
        )
