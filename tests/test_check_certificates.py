import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/check_certificates.py")


def test_check_certificates_accepts_passing_stored_gate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "certificate_gate": {
                    "passed": True,
                    "required_certificates": ["contraction"],
                    "failures": [],
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(metrics_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "passed=True" in completed.stdout
    assert "required=contraction" in completed.stdout


def test_check_certificates_rejects_failing_stored_gate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "certificate_gate": {
                    "passed": False,
                    "required_certificates": ["learning_stability"],
                    "failures": [
                        {
                            "certificate": "learning_stability",
                            "reason": "missing",
                            "expected_datasets": 1,
                            "observed_datasets": 0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(metrics_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "passed=False" in completed.stdout
    assert "learning_stability: missing" in completed.stderr


def test_check_certificates_recomputes_missing_gate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "config": {"required_certificates": ["contraction"]},
                "datasets": {
                    "circle": {
                        "certificates": {
                            "contraction": {
                                "theorem": "theorem_01_sufficient_contraction",
                                "contraction_margin": 0.2,
                                "required_margin": 0.1,
                                "is_certified": True,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(metrics_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "passed=True" in completed.stdout
