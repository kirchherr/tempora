import json
from pathlib import Path

import yaml

from tempora.experiments import load_benchmark_config, run_synthetic_benchmark
from tempora.experiments.run_synthetic import render_benchmark_report


def test_synthetic_smoke_benchmark_writes_metrics_figures_and_report(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "ci_smoke",
                "seed": 801,
                "output_root": str(tmp_path / "outputs"),
                "datasets": ["circle", "torus", "lorenz", "rossler"],
                "n_steps": 20,
                "epochs": 2,
                "learning_rate": 0.03,
                "plasticity_learning_rate": 0.0005,
                "baseline_epochs": 2,
            }
        ),
        encoding="utf-8",
    )
    config = load_benchmark_config(config_path)

    result = run_synthetic_benchmark(config)

    assert result.metrics_path.exists()
    assert result.report_path.exists()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "ci_smoke"
    assert set(payload["datasets"]) == {"circle", "torus", "lorenz", "rossler"}
    assert payload["config"]["epochs"] == 2
    assert payload["dependency_versions"]["torch"] is not None
    for _dataset_name, dataset_metrics in payload["datasets"].items():
        assert dataset_metrics["prediction_mse"] >= 0.0
        assert dataset_metrics["contraction_margin_final"] > 0.0
        assert "gru" in dataset_metrics["baselines"]
        assert "vanilla_neural_ode" in dataset_metrics["baselines"]
        assert Path(dataset_metrics["figures"]["input_trajectory"]).exists()
        assert Path(dataset_metrics["figures"]["latent_trajectory"]).exists()
    assert "## Claims" in result.report_path.read_text(encoding="utf-8")


def test_render_benchmark_report_separates_claims_evidence_and_open_points() -> None:
    metrics = {
        "run_id": "example",
        "config": {"epochs": 1},
        "datasets": {
            "circle": {
                "prediction_mse": 0.1,
                "reconstruction_mse": 0.2,
                "contraction_margin_final": 0.3,
                "largest_lyapunov_estimate": -0.1,
                "tda_bottleneck_h1": 0.05,
                "time_warp_invariance_score": 1.0,
                "baselines": {
                    "gru": {
                        "prediction_mse": 0.4,
                        "reconstruction_mse": 0.5,
                        "fit_epochs": 1.0,
                    }
                },
            }
        },
    }

    report = render_benchmark_report(metrics)

    assert "## Claims" in report
    assert "## Evidence" in report
    assert "## Open Points" in report
    assert "TEMPORA Contractive CTRNN" in report
