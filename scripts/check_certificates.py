from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import (
    evaluate_certificate_gate,
    summarize_benchmark_certificates,
)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a benchmark metrics payload from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be a JSON object.")
    return cast(dict[str, Any], payload)


def resolve_certificate_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return stored certificate gate data or compute it from metrics."""

    stored_gate = metrics.get("certificate_gate")
    if isinstance(stored_gate, dict):
        return cast(dict[str, Any], stored_gate)

    datasets = metrics.get("datasets")
    config = metrics.get("config")
    if not isinstance(datasets, dict) or not isinstance(config, dict):
        raise ValueError("metrics payload must include datasets and config.")

    summary_payload = metrics.get("certificate_summary")
    certificate_summary = (
        cast(dict[str, Any], summary_payload)
        if isinstance(summary_payload, dict)
        else summarize_benchmark_certificates(cast(dict[str, Any], datasets))
    )
    required = tuple(str(item) for item in config.get("required_certificates", []))
    return evaluate_certificate_gate(
        cast(dict[str, Any], datasets),
        certificate_summary,
        required_certificates=required,
    )


def render_gate_status(gate: dict[str, Any]) -> str:
    """Render one-line certificate gate status for CLI output."""

    required = gate.get("required_certificates", [])
    required_text = ", ".join(str(item) for item in required) if required else "none"
    return (
        f"certificate_gate passed={gate.get('passed', False)} required={required_text}"
    )


def render_gate_failures(gate: dict[str, Any]) -> list[str]:
    """Render certificate gate failures for CLI output."""

    failures = gate.get("failures", [])
    if not isinstance(failures, list):
        return ["certificate_gate failure payload is not a list."]
    return [_render_failure(cast(dict[str, Any], failure)) for failure in failures]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check TEMPORA certificate gate.")
    parser.add_argument("metrics", help="Path to benchmark metrics.json.")
    args = parser.parse_args(argv)

    gate = resolve_certificate_gate(load_metrics(Path(args.metrics)))
    print(render_gate_status(gate))
    if gate.get("passed") is True:
        return 0
    for failure in render_gate_failures(gate):
        print(failure, file=sys.stderr)
    return 1


def _render_failure(failure: dict[str, Any]) -> str:
    dataset = failure.get("dataset")
    certificate = failure.get("certificate", "unknown")
    if dataset is not None:
        return f"{dataset}/{certificate}: certificate gate failure"
    reason = failure.get("reason", "unknown")
    observed = failure.get("observed_datasets", "unknown")
    expected = failure.get("expected_datasets", "unknown")
    return (
        f"{certificate}: {reason}; observed_datasets={observed}; "
        f"expected_datasets={expected}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
