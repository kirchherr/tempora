from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tempora.experiments.evidence_bundle import (
    build_evidence_bundle_manifest,
    write_evidence_bundle_manifest,
)
from tempora.experiments.evidence_index import (
    build_evidence_index,
    write_evidence_index,
)
from tempora.experiments.evidence_report import (
    render_evidence_report,
    write_evidence_report,
)

try:
    from scripts.build_evidence_index import build_records
    from scripts.release_smoke import run_release_smoke
    from scripts.validate_evidence_bundle import validate_evidence_bundle_file
    from scripts.validate_evidence_index import validate_evidence_index_file
    from scripts.validate_evidence_report import validate_evidence_report_file
except ModuleNotFoundError:  # pragma: no cover - used when invoked as a script path.
    from build_evidence_index import build_records
    from release_smoke import run_release_smoke
    from validate_evidence_bundle import validate_evidence_bundle_file
    from validate_evidence_index import validate_evidence_index_file
    from validate_evidence_report import validate_evidence_report_file


@dataclass(frozen=True)
class ReleaseEvidenceResult:
    """Paths produced by the complete release evidence runner."""

    metrics_path: Path
    benchmark_report_path: Path
    artifact_manifest_path: Path
    evidence_index_path: Path
    evidence_report_path: Path
    evidence_bundle_path: Path


def run_release_evidence(
    config_path: Path,
    *,
    evidence_index_path: Path,
    evidence_report_path: Path,
    evidence_bundle_path: Path,
) -> ReleaseEvidenceResult:
    """Run and validate the full CI-small release evidence path."""

    base_dir = Path.cwd()
    smoke_result = run_release_smoke(config_path)

    records = build_records(
        (smoke_result.metrics_path,),
        require_manifest=True,
        check_files=True,
    )
    index = build_evidence_index(records)
    write_evidence_index(index, evidence_index_path)
    validate_evidence_index_file(
        evidence_index_path,
        require_gates_passed=True,
        require_manifests=True,
        require_git_commits=True,
        check_files=True,
        base_dir=base_dir,
    )

    report = render_evidence_report(index)
    write_evidence_report(report, evidence_report_path)
    validate_evidence_report_file(evidence_index_path, evidence_report_path)

    bundle = build_evidence_bundle_manifest(
        evidence_index_path,
        evidence_report_path,
        base_dir=base_dir,
    )
    write_evidence_bundle_manifest(bundle, evidence_bundle_path)
    validate_evidence_bundle_file(evidence_bundle_path, base_dir=base_dir)

    return ReleaseEvidenceResult(
        metrics_path=smoke_result.metrics_path,
        benchmark_report_path=smoke_result.report_path,
        artifact_manifest_path=smoke_result.output_dir / "artifact_manifest.json",
        evidence_index_path=evidence_index_path,
        evidence_report_path=evidence_report_path,
        evidence_bundle_path=evidence_bundle_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate the TEMPORA release evidence path."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_smoke.yaml",
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--index-output",
        default="outputs/evidence_index.json",
        help="Path to write evidence_index.json.",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/evidence_report.md",
        help="Path to write evidence_report.md.",
    )
    parser.add_argument(
        "--bundle-output",
        default="outputs/evidence_bundle.json",
        help="Path to write evidence_bundle.json.",
    )
    args = parser.parse_args(argv)

    try:
        result = run_release_evidence(
            Path(args.config),
            evidence_index_path=Path(args.index_output),
            evidence_report_path=Path(args.report_output),
            evidence_bundle_path=Path(args.bundle_output),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release evidence failed: {exc}", file=sys.stderr)
        return 1

    print(f"release evidence metrics: {result.metrics_path}")
    print(f"release evidence benchmark report: {result.benchmark_report_path}")
    print(f"release evidence artifact manifest: {result.artifact_manifest_path}")
    print(f"release evidence index: {result.evidence_index_path}")
    print(f"release evidence report: {result.evidence_report_path}")
    print(f"release evidence bundle: {result.evidence_bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
