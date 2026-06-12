"""Machine-readable proof-adjacent checks for TEMPORA."""

from tempora.proof.assumptions import CONTRACTION_ASSUMPTIONS
from tempora.proof.certificates import (
    ContractionCertificate,
    certify_model_contraction,
    certify_sufficient_contraction,
)
from tempora.proof.theorem_checks import CertificateViolation, require_certificate

__all__ = [
    "CONTRACTION_ASSUMPTIONS",
    "CertificateViolation",
    "ContractionCertificate",
    "certify_model_contraction",
    "certify_sufficient_contraction",
    "require_certificate",
]
