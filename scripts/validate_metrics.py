from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import validate_benchmark_metrics


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a benchmark metrics payload from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be a JSON object.")
    return cast(dict[str, Any], payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate TEMPORA metrics.json.")
    parser.add_argument("metrics", help="Path to benchmark metrics.json.")
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    try:
        validate_benchmark_metrics(load_metrics(metrics_path))
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"metrics schema invalid: {exc}", file=sys.stderr)
        return 1
    print(f"metrics schema valid: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
