"""Build checksum manifests for TEMPORA evidence review artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from tempora.experiments.evidence_index import load_json_object, validate_evidence_index
from tempora.experiments.evidence_report import validate_evidence_report


def build_evidence_bundle_manifest(
    index_path: Path,
    report_path: Path,
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """Build a deterministic manifest for generated evidence review artifacts."""

    resolved_index_path = _resolve_path(index_path, base_dir=base_dir)
    resolved_report_path = _resolve_path(report_path, base_dir=base_dir)
    index = load_json_object(resolved_index_path)
    report = resolved_report_path.read_text(encoding="utf-8")
    validate_evidence_index(index)
    validate_evidence_report(report, index)

    runs = [_ensure_mapping(run, "runs[]") for run in _required_list(index, "runs")]
    manifest = {
        "schema": "tempora.evidence_bundle.v1",
        "evidence_index": _file_record(index_path, resolved_index_path),
        "evidence_report": _file_record(report_path, resolved_report_path),
        "run_count": _required_int(index, "run_count"),
        "dataset_count": _required_int(index, "dataset_count"),
        "git_commits": _string_list(index, "git_commits"),
        "all_certificate_gates_passed": (
            index.get("all_certificate_gates_passed") is True
        ),
        "run_ids": [_required_string(run, "run_id") for run in runs],
        "limitations": [
            "This manifest records hashes for generated review artifacts.",
            "It does not prove general temporal semantic preservation.",
            "It does not validate artifacts outside the referenced index and report.",
        ],
    }
    validate_evidence_bundle_manifest(manifest)
    return manifest


def validate_evidence_bundle_manifest(manifest: dict[str, Any]) -> None:
    """Validate the internal shape of an evidence bundle manifest."""

    if manifest.get("schema") != "tempora.evidence_bundle.v1":
        raise ValueError("evidence bundle schema must be tempora.evidence_bundle.v1.")
    _validate_file_record(_required_mapping(manifest, "evidence_index"))
    _validate_file_record(_required_mapping(manifest, "evidence_report"))

    run_count = _required_int(manifest, "run_count")
    run_ids = _string_list(manifest, "run_ids")
    if run_count != len(run_ids):
        raise ValueError("evidence bundle run_count does not match run_ids length.")

    _required_int(manifest, "dataset_count")
    _string_list(manifest, "git_commits")
    if not isinstance(manifest.get("all_certificate_gates_passed"), bool):
        raise ValueError("all_certificate_gates_passed must be a boolean.")
    _string_list(manifest, "limitations")


def validate_evidence_bundle_files(
    manifest: dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    """Validate that bundle file records still match local evidence artifacts."""

    validate_evidence_bundle_manifest(manifest)
    for key in ("evidence_index", "evidence_report"):
        record = _required_mapping(manifest, key)
        _validate_file_record_matches(record, base_dir=base_dir)


def validate_evidence_bundle_artifacts(
    manifest: dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    """Validate a bundle manifest against local index and report artifacts."""

    validate_evidence_bundle_files(manifest, base_dir=base_dir)
    index_record = _required_mapping(manifest, "evidence_index")
    report_record = _required_mapping(manifest, "evidence_report")
    index_path = _resolve_path(
        Path(_required_string(index_record, "path")),
        base_dir=base_dir,
    )
    report_path = _resolve_path(
        Path(_required_string(report_record, "path")),
        base_dir=base_dir,
    )
    index = load_json_object(index_path)
    report = report_path.read_text(encoding="utf-8")
    validate_evidence_index(index)
    validate_evidence_report(report, index)
    _validate_bundle_summary_matches_index(manifest, index)


def write_evidence_bundle_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write an evidence bundle manifest JSON artifact."""

    validate_evidence_bundle_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _file_record(display_path: Path, resolved_path: Path) -> dict[str, Any]:
    return {
        "path": str(display_path),
        "bytes": resolved_path.stat().st_size,
        "sha256": _sha256_file(resolved_path),
    }


def _validate_file_record(record: dict[str, Any]) -> None:
    _required_string(record, "path")
    byte_count = _required_int(record, "bytes")
    if byte_count < 0:
        raise ValueError("file record bytes must be non-negative.")
    sha256 = _required_string(record, "sha256")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError("file record sha256 must be a lowercase SHA256 digest.")


def _validate_file_record_matches(
    record: dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    display_path = Path(_required_string(record, "path"))
    resolved_path = _resolve_path(display_path, base_dir=base_dir)
    expected_bytes = _required_int(record, "bytes")
    if resolved_path.stat().st_size != expected_bytes:
        raise ValueError(f"{display_path} byte count does not match manifest.")
    expected_sha256 = _required_string(record, "sha256")
    if _sha256_file(resolved_path) != expected_sha256:
        raise ValueError(f"{display_path} sha256 does not match manifest.")


def _validate_bundle_summary_matches_index(
    manifest: dict[str, Any],
    index: dict[str, Any],
) -> None:
    runs = [_ensure_mapping(run, "runs[]") for run in _required_list(index, "runs")]
    expected = {
        "run_count": _required_int(index, "run_count"),
        "dataset_count": _required_int(index, "dataset_count"),
        "git_commits": _string_list(index, "git_commits"),
        "all_certificate_gates_passed": (
            index.get("all_certificate_gates_passed") is True
        ),
        "run_ids": [_required_string(run, "run_id") for run in runs],
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"evidence bundle {key} does not match evidence index.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON list.")
    return value


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return _ensure_mapping(payload.get(key), key)


def _ensure_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(dict[str, Any], value)


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    values = _required_list(payload, key)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{key} must contain only non-empty strings.")
    return cast(list[str], values)
