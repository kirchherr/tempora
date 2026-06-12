"""Small theorem-check helpers for TEMPORA certificates."""

from __future__ import annotations

from tempora.proof.certificates import (
    ContractionCertificate,
    LearningStabilityCertificate,
)


class CertificateViolation(RuntimeError):
    """Raised when a proof-adjacent certificate does not meet its threshold."""


Certificate = ContractionCertificate | LearningStabilityCertificate


def require_certificate(certificate: Certificate) -> Certificate:
    """Return a certificate only if its numerical condition is certified."""

    if not certificate.is_certified:
        raise CertificateViolation(
            "certificate is not certified: "
            f"{_certificate_margin(certificate):.6g} <= "
            f"{certificate.required_margin:.6g}"
        )
    return certificate


def _certificate_margin(certificate: Certificate) -> float:
    if isinstance(certificate, ContractionCertificate):
        return certificate.contraction_margin
    return certificate.margin_after
