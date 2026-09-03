"""Render human-readable reports from TEMPORA evidence indexes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from tempora.experiments.evidence_index import validate_evidence_index


def render_evidence_report(index: dict[str, Any]) -> str:
    """Render a conservative Markdown report from an evidence index.

    The report summarizes generated benchmark artifacts for review. It does not
    add new certificate claims and should be read together with the referenced
    metrics, reports, figures, checkpoints, and artifact manifests.
    """

    validate_evidence_index(index)

    lines = [
        "# TEMPORA Evidence Index Report",
        "",
        "This report summarizes generated benchmark evidence for review.",
        "It does not prove general temporal semantic preservation.",
        "",
        "## Summary",
        "",
        f"- schema: {_code(_required_string(index, 'schema'))}",
        f"- runs: {_required_int(index, 'run_count')}",
        f"- datasets: {_required_int(index, 'dataset_count')}",
        "- all certificate gates passed: "
        f"{_yes_no(index.get('all_certificate_gates_passed') is True)}",
        f"- git commits: {_format_string_list(_string_list(index, 'git_commits'))}",
        "",
        "## Runs",
        "",
    ]

    for run_value in _required_list(index, "runs"):
        run = _ensure_mapping(run_value, "runs[]")
        lines.extend(_render_run(run))

    limitations = _optional_string_list(index, "limitations")
    if limitations:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in limitations)
        lines.append("")

    lines.extend(
        [
            "## Open Review Points",
            "",
            "- Inspect referenced metrics, reports, figures, checkpoints, and "
            "artifact manifests before making research claims.",
            "- Rebuild this report after rerunning benchmarks or changing indexed "
            "artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def expected_evidence_report_text(index: dict[str, Any]) -> str:
    """Return the exact on-disk Markdown text expected for an evidence index."""

    return render_evidence_report(index).rstrip() + "\n"


def validate_evidence_report(report: str, index: dict[str, Any]) -> None:
    """Validate that a Markdown evidence report matches its evidence index."""

    expected = expected_evidence_report_text(index)
    if report != expected:
        raise ValueError("evidence report does not match the evidence index.")


def write_evidence_report(report: str, path: Path) -> Path:
    """Write a Markdown evidence report artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.rstrip() + "\n", encoding="utf-8")
    return path


def _render_run(run: dict[str, Any]) -> list[str]:
    run_id = _required_string(run, "run_id")
    summary = _required_mapping(run, "certificate_summary")
    gate = _required_mapping(run, "certificate_gate")

    lines = [
        f"### {run_id}",
        "",
        f"- seed: {_required_int(run, 'seed')}",
        f"- git commit: {_format_optional_string(run.get('git_commit'))}",
        f"- metrics: {_code(_required_string(run, 'metrics_path'))}",
        f"- datasets: {_format_string_list(_string_list(run, 'datasets'))}",
        f"- dataset count: {_required_int(run, 'dataset_count')}",
        f"- artifact manifest: {_format_optional_string(run.get('artifact_manifest'))}",
        f"- artifact count: {_format_optional_int(run.get('artifact_count'))}",
        "- certificate types: "
        f"{_format_string_list(_string_list(run, 'certificate_types'))}",
        "- certificate summary all certified: "
        f"{_yes_no(summary.get('all_certified') is True)}",
        "- certificate summary failure count: "
        f"{_required_int(summary, 'failure_count')}",
        f"- certificate gate passed: {_yes_no(gate.get('passed') is True)}",
        "- required certificates: "
        f"{_format_string_list(_string_list(gate, 'required_certificates'))}",
        f"- certificate gate failure count: {_required_int(gate, 'failure_count')}",
        "",
    ]

    lines.extend(
        _render_failure_section(
            "Certificate Gate Failures",
            _mapping_list(gate, "failures"),
        )
    )
    lines.extend(
        _render_failure_section(
            "Certificate Summary Failures",
            _mapping_list(summary, "failures"),
        )
    )
    return lines


def _render_failure_section(
    title: str,
    failures: list[dict[str, Any]],
) -> list[str]:
    lines = [f"#### {title}", ""]
    if not failures:
        lines.extend(["No failure records.", ""])
        return lines

    for position, failure in enumerate(failures, start=1):
        lines.append(f"{position}. {_format_failure(failure)}")
    lines.append("")
    return lines


def _format_failure(failure: dict[str, Any]) -> str:
    ordered_keys = (
        "dataset",
        "certificate",
        "theorem",
        "metric",
        "homology_dim",
        "distance",
        "max_distance",
        "value",
        "threshold",
    )
    parts: list[str] = []
    for key in ordered_keys:
        if key in failure:
            parts.append(f"{key}={_format_failure_value(failure[key])}")

    extra_keys = sorted(key for key in failure if key not in ordered_keys)
    for key in extra_keys:
        parts.append(f"{key}={_format_failure_value(failure[key])}")

    if parts:
        return ", ".join(parts)
    return json.dumps(failure, sort_keys=True)


def _format_failure_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, str):
        return _code(value)
    return json.dumps(value, sort_keys=True)


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


def _optional_string_list(payload: dict[str, Any], key: str) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON list when present.")
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{key} must contain only non-empty strings.")
    return cast(list[str], values)


def _mapping_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = _required_list(payload, key)
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{key} must contain only JSON objects.")
    return [cast(dict[str, Any], value) for value in values]


def _format_string_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(_code(value) for value in values)


def _format_optional_string(value: Any) -> str:
    if isinstance(value, str) and value:
        return _code(value)
    if value is None:
        return "none"
    raise ValueError("optional string fields must be strings or null.")


def _format_optional_int(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError("optional integer fields must be integers or null.")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _code(value: str) -> str:
    if "`" in value:
        return value
    return f"`{value}`"
