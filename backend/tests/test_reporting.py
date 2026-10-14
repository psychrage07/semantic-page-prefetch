import json
from pathlib import Path

from pagewise.reporting import build_markdown_report, build_consolidated_report, build_dashboard_snapshot


def test_build_suite_markdown_report(tmp_path: Path):
    source = tmp_path / "suite_summary.json"
    source.write_text(json.dumps({
        "mode": "latency-injected",
        "queries_per_run": 10,
        "seeds": [1],
        "warning": "Keep separate from final claims.",
        "strategies": [
            {"strategy": "none", "trials": 1, "median_of_run_p50_ms": 2.0},
            {"strategy": "model_a", "trials": 1, "median_of_run_p50_ms": 1.5, "mean_dropped_prediction_jobs": 3},
        ],
        "paired_query_comparisons": [
            {"strategy": "model_a", "baseline": "none", "matched_queries": 10, "mean_delta_ms": -0.5, "median_delta_ms": -0.4, "improved_query_pct": 70.0}
        ],
    }), encoding="utf-8")

    result = build_markdown_report(source)
    output = Path(result["output"])
    text = output.read_text(encoding="utf-8")
    assert result["report_type"] == "suite_summary"
    assert "Paired Deltas vs none" in text
    assert "model_a" in text


def test_build_training_sweep_markdown_report(tmp_path: Path):
    source = tmp_path / "training_sweep_report.json"
    source.write_text(json.dumps({
        "training_seeds": [1, 2],
        "fixed_split_seed": 7,
        "epochs": 2,
        "note": "coarse",
        "aggregate": {
            "model_b": {
                "top1_accuracy_pct": {"runs": 2, "mean": 10.0},
                "top5_accuracy_pct": {"runs": 2, "mean": 20.0},
            }
        },
        "delta_vs_model_b": {
            "model_b_retrained_no_object": {
                "top1_accuracy_pct": {"runs": 2, "mean_delta": -1.0},
                "top5_accuracy_pct": {"runs": 2, "mean_delta": -2.0},
            }
        },
    }), encoding="utf-8")

    result = build_markdown_report(source, tmp_path / "report.md")
    text = Path(result["output"]).read_text(encoding="utf-8")
    assert result["report_type"] == "training_sweep"
    assert "Deltas vs Model B" in text
    assert "model_b_retrained_no_object" in text


def test_build_consolidated_report(tmp_path: Path):
    """Consolidated report combines predictor metrics and suite summary."""
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "predictor_metrics.json").write_text(json.dumps({
        "test_executions": 20, "top_k": 5,
        "note": "Held-out test.",
        "policies": {
            "model_a": {"top1_accuracy_pct": 15.0, "top5_accuracy_pct": 40.0, "candidate_recall_pct": 80.0, "evaluated_next_page_predictions": 100},
            "model_b": {"top1_accuracy_pct": 25.0, "top5_accuracy_pct": 60.0, "candidate_recall_pct": 80.0, "evaluated_next_page_predictions": 100},
            "model_b_retrained_no_object": {"top1_accuracy_pct": 20.0, "top5_accuracy_pct": 50.0, "candidate_recall_pct": 80.0, "evaluated_next_page_predictions": 100},
            "markov1": {"top1_accuracy_pct": 45.0, "top5_accuracy_pct": 70.0, "evaluated_next_page_predictions": 100},
        },
    }), encoding="utf-8")

    suite_dir = tmp_path / "suite-tuned-all"
    suite_dir.mkdir()
    (suite_dir / "suite_summary.json").write_text(json.dumps({
        "mode": "latency-injected", "queries_per_run": 50, "seeds": [1, 2],
        "strategies": [
            {"strategy": "none", "trials": 2, "median_of_run_p50_ms": 5.0},
            {"strategy": "model_a", "trials": 2, "median_of_run_p50_ms": 4.5, "mean_inference_ms_total": 100.0},
        ],
        "paired_query_comparisons": [
            {"strategy": "model_a", "baseline": "none", "matched_queries": 100, "mean_delta_ms": -0.3,
             "delta_ci95_low_ms": -0.5, "delta_ci95_high_ms": -0.1, "improved_query_pct": 65.0},
        ],
    }), encoding="utf-8")
    result = build_consolidated_report(tmp_path)
    text = Path(result["output"]).read_text(encoding="utf-8")
    assert "Consolidated Research Report" in text
    assert "Offline Prediction Accuracy" in text
    assert "Runtime:" in text
    assert "suite-tuned-all" in text
    assert "Semantic Ablation Deltas" in text
    assert "model_b_retrained_no_object" in text
    assert len(result["sources"]) == 2


def test_build_dashboard_snapshot(tmp_path: Path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "predictor_metrics.json").write_text(json.dumps({
        "dataset": "observed VFS page-request sequences",
        "note": "held out",
        "test_executions": 4,
        "top_k": 5,
        "policies": {
            "model_a": {"evaluated_next_page_predictions": 10, "top1_accuracy_pct": 20.0, "top5_accuracy_pct": 40.0, "candidate_recall_pct": 70.0},
            "model_b": {"evaluated_next_page_predictions": 10, "top1_accuracy_pct": 30.0, "top5_accuracy_pct": 50.0, "candidate_recall_pct": 70.0},
        },
    }), encoding="utf-8")

    suite_dir = tmp_path / "suite-high-delay"
    suite_dir.mkdir()
    (suite_dir / "suite_summary.json").write_text(json.dumps({
        "mode": "latency-injected",
        "queries_per_run": 12,
        "seeds": [1],
        "strategies": [
            {"strategy": "none", "median_of_run_p50_ms": 10.0},
            {"strategy": "model_b", "median_of_run_p50_ms": 9.0},
        ],
        "paired_query_comparisons": [
            {"strategy": "model_b", "baseline": "none", "matched_queries": 12, "mean_delta_ms": -0.7, "improved_query_pct": 60.0},
        ],
    }), encoding="utf-8")
    run_dir = suite_dir / "runtime-model-b"
    run_dir.mkdir()
    (run_dir / "query_results.csv").write_text(
        "query_id,template_id,workload_family,elapsed_ms,plan_class\n"
        "q-1,t-1,point,8.5,index_seek\n",
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text(json.dumps({
        "seq": 3,
        "query_id": "q-1",
        "template_id": "t-1",
        "workload_family": "point",
        "plan_class": "index_seek",
        "first_page": 42,
        "duration_us": 125.0,
        "cache_status": "miss",
        "source": "demand",
    }) + "\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text(json.dumps({
        "prediction_id": 9,
        "query_id": "q-1",
        "policy": "model_b",
        "selected": [{"page": 42, "score": 0.75, "status": "prefetched"}],
    }) + "\n", encoding="utf-8")
    (suite_dir / "suite_runs.json").write_text(json.dumps({
        "runs": [{
            "strategy": "model_b",
            "trial": 1,
            "seed": 1,
            "output_dir": str(run_dir),
            "query_results_path": str(run_dir / "query_results.csv"),
        }],
    }), encoding="utf-8")

    final_dir = tmp_path / "final-latency-0p2ms"
    final_dir.mkdir()
    (final_dir / "suite_summary.json").write_text(json.dumps({
        "mode": "latency-injected",
        "queries_per_run": 12,
        "seeds": [901],
        "strategies": [
            {"strategy": "none", "median_of_run_p50_ms": 10.0},
            {"strategy": "model_b", "median_of_run_p50_ms": 9.0},
        ],
        "paired_query_comparisons": [
            {"strategy": "model_b", "baseline": "none", "matched_queries": 12, "mean_delta_ms": -0.7, "improved_query_pct": 60.0},
        ],
    }), encoding="utf-8")
    (final_dir / "suite_runs.json").write_text(json.dumps({
        "runs": [{
            "strategy": "model_b",
            "trial": 1,
            "seed": 901,
            "output_dir": str(run_dir),
            "query_results_path": str(run_dir / "query_results.csv"),
        }],
    }), encoding="utf-8")

    sweep_dir = tmp_path / "ablation-sweep"
    sweep_dir.mkdir()
    (sweep_dir / "training_sweep_report.json").write_text(json.dumps({
        "training_seeds": [7],
        "fixed_split_seed": 42,
        "epochs": 1,
        "aggregate": {
            "model_b": {
                "top1_accuracy_pct": {"runs": 1, "mean": 31.0, "bootstrap_mean_ci95_low": 31.0, "bootstrap_mean_ci95_high": 31.0},
                "top5_accuracy_pct": {"runs": 1, "mean": 68.0},
            },
        },
    }), encoding="utf-8")

    snapshot = build_dashboard_snapshot(tmp_path)
    assert snapshot["schema_version"] == 1
    assert snapshot["prediction"]["policies"][0]["policy"] == "model_b"
    assert snapshot["runtime_suites"][0]["name"] == "final-latency-0p2ms"
    assert snapshot["runtime_suites"][0]["best_mean_delta_vs_none"]["strategy"] == "model_b"
    assert snapshot["measured_samples"]["suite"] == "final-latency-0p2ms"
    latency_point = snapshot["runtime_suites"][0]["query_latency_series"][0]["points"][0]
    assert latency_point["elapsed_ms"] == 8.5
    assert snapshot["measured_samples"]["trace_events"][0]["page"] == 42
    assert snapshot["measured_samples"]["predictions"][0]["selected"][0]["score"] == 0.75
    assert snapshot["multi_seed_training"]["aggregate"][0]["top1_mean_pct"] == 31.0
    assert len(snapshot["sources"]) == 4
