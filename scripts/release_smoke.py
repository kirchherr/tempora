from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import (
    SyntheticBenchmarkResult,
    load_benchmark_config,
    render_benchmark_report,
    run_synthetic_benchmark,
    validate_benchmark_metrics,
)

try:
    from scripts.check_certificates import render_gate_failures, render_gate_status
    from scripts.validate_metrics import collect_artifact_paths, validate_artifact_paths
except ModuleNotFoundError:  # pragma: no cover - used when invoked as a script path.
    from check_certificates import render_gate_failures, render_gate_status
    from validate_metrics import collect_artifact_paths, validate_artifact_paths


def run_release_smoke(config_path: Path) -> SyntheticBenchmarkResult:
    """Run the v0.1 release smoke path and validate generated artifacts."""

    config = load_benchmark_config(config_path)
    result = run_synthetic_benchmark(config)
    validate_benchmark_metrics(result.metrics)
    validate_artifact_paths(result.metrics, base_dir=Path.cwd())
    result.report_path.write_text(
        render_benchmark_report(result.metrics),
        encoding="utf-8",
    )
    gate = cast(dict[str, Any], result.metrics["certificate_gate"])
    print(render_gate_status(gate))
    if gate.get("passed") is not True:
        failures = "\n".join(render_gate_failures(gate))
        raise RuntimeError(f"certificate gate failed:\n{failures}")
    write_artifact_manifest(
        result.metrics,
        result.output_dir / "artifact_manifest.json",
        base_dir=Path.cwd(),
    )
    return result


def build_artifact_manifest(
    metrics: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """Build a checksum manifest for generated benchmark artifacts."""

    entries = []
    for artifact_path in collect_artifact_paths(metrics):
        candidate = (
            artifact_path if artifact_path.is_absolute() else base_dir / artifact_path
        )
        entries.append(
            {
                "path": str(artifact_path),
                "bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    entries.sort(key=lambda item: str(item["path"]))
    return {
        "run_id": metrics["run_id"],
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def write_artifact_manifest(
    metrics: dict[str, Any],
    path: Path,
    *,
    base_dir: Path,
) -> Path:
    """Write an artifact checksum manifest next to release smoke outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(metrics, base_dir=base_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TEMPORA v0.1 release smoke.")
    parser.add_argument(
        "--config",
        default="configs/benchmark_smoke.yaml",
        help="Path to benchmark YAML config.",
    )
    args = parser.parse_args(argv)

    try:
        result = run_release_smoke(Path(args.config))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release smoke failed: {exc}", file=sys.stderr)
        return 1
    print(f"release smoke metrics: {result.metrics_path}")
    print(f"release smoke report: {result.report_path}")
    print(f"release smoke manifest: {result.output_dir / 'artifact_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
