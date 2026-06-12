from __future__ import annotations

import argparse
from pathlib import Path

from tempora.experiments.run_synthetic import (
    load_benchmark_config,
    run_synthetic_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TEMPORA synthetic smoke benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_smoke.yaml",
        help="Path to benchmark YAML config.",
    )
    args = parser.parse_args()
    config = load_benchmark_config(Path(args.config))
    result = run_synthetic_benchmark(config)
    print(result.config_path)
    print(result.metrics_path)
    print(result.report_path)
    for checkpoint_path in result.checkpoint_paths:
        print(checkpoint_path)


if __name__ == "__main__":
    main()
