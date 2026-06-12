from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import render_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TEMPORA benchmark report.")
    parser.add_argument("metrics", help="Path to metrics.json.")
    parser.add_argument("--output", default=None, help="Optional output report path.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    report = render_benchmark_report(metrics)
    output_path = (
        Path(args.output)
        if args.output is not None
        else metrics_path.with_name("report.md")
    )
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
