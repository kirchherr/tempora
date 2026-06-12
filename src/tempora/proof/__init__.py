"""Machine-readable proof-adjacent checks for TEMPORA."""

from tempora.proof.assumptions import (
    CONTRACTION_ASSUMPTIONS,
    LEARNING_STABILITY_ASSUMPTIONS,
)
from tempora.proof.certificates import (
    ContractionCertificate,
    LearningStabilityCertificate,
    certify_model_contraction,
    certify_projected_update_stability,
    certify_sufficient_contraction,
)
from tempora.proof.theorem_checks import CertificateViolation, require_certificate

__all__ = [
    "CONTRACTION_ASSUMPTIONS",
    "LEARNING_STABILITY_ASSUMPTIONS",
    "CertificateViolation",
    "ContractionCertificate",
    "LearningStabilityCertificate",
    "certify_model_contraction",
    "certify_projected_update_stability",
    "certify_sufficient_contraction",
    "require_certificate",
]
