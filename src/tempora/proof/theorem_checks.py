"""Small theorem-check helpers for TEMPORA certificates."""

from __future__ import annotations

from tempora.proof.certificates import (
    ContractionCertificate,
    LearningStabilityCertificate,
    TopologyComparisonCertificate,
)


class CertificateViolation(RuntimeError):
    """Raised when a proof-adjacent certificate does not meet its threshold."""


Certificate = (
    ContractionCertificate
    | LearningStabilityCertificate
    | TopologyComparisonCertificate
)


def require_certificate(certificate: Certificate) -> Certificate:
    """Return a certificate only if its numerical condition is certified."""

    if not certificate.is_certified:
        raise CertificateViolation(_failure_message(certificate))
    return certificate


def _failure_message(certificate: Certificate) -> str:
    if isinstance(certificate, TopologyComparisonCertificate):
        return (
            "certificate is not certified: "
            f"{certificate.distance:.6g} > {certificate.max_distance:.6g}"
        )
    return (
        "certificate is not certified: "
        f"{_certificate_margin(certificate):.6g} <= "
        f"{certificate.required_margin:.6g}"
    )


def _certificate_margin(
    certificate: ContractionCertificate | LearningStabilityCertificate,
) -> float:
    if isinstance(certificate, ContractionCertificate):
        return certificate.contraction_margin
    return certificate.margin_after
