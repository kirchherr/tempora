"""Small theorem-check helpers for TEMPORA certificates."""

from __future__ import annotations

from tempora.proof.certificates import ContractionCertificate


class CertificateViolation(RuntimeError):
    """Raised when a proof-adjacent certificate does not meet its threshold."""


def require_certificate(certificate: ContractionCertificate) -> ContractionCertificate:
    """Return a certificate only if its numerical condition is certified."""

    if not certificate.is_certified:
        raise CertificateViolation(
            "certificate is not certified: "
            f"{certificate.contraction_margin:.6g} <= "
            f"{certificate.required_margin:.6g}"
        )
    return certificate
