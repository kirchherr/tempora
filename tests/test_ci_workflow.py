from pathlib import Path

import yaml

ROOT = Path(".")


def test_ci_uses_read_only_permissions() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )

    assert workflow["permissions"] == {"contents": "read"}


def test_ci_runs_standard_checks_and_smoke_benchmark() -> None:
    workflow_text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    required_commands = (
        "python -m pytest",
        "python -m ruff check .",
        "python -m ruff format --check .",
        "python -m mypy src",
        "python scripts/train_synth.py --config configs/benchmark_smoke.yaml",
        "python scripts/make_report.py outputs/benchmark_smoke/metrics.json",
        "python scripts/check_certificates.py outputs/benchmark_smoke/metrics.json",
        "test -f outputs/benchmark_smoke/config.yaml",
        "test -f outputs/benchmark_smoke/metrics.json",
        "test -f outputs/benchmark_smoke/report.md",
    )
    for command in required_commands:
        assert command in workflow_text
