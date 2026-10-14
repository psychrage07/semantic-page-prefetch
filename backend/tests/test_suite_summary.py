import csv
import json
from pathlib import Path

import pagewise.suite as suite_module
from pagewise.suite import recover_suite, run_runtime_sensitivity_sweep, summarize_suite


def test_suite_summary_aggregates_runs_without_fabricating_latency():
    report = {
        "mode": "latency-injected",
        "database_path": "example.sqlite",
        "queries_per_run": 10,
        "seeds": [1, 2],
        "warning": "Injected latency is not hardware latency.",
        "runs": [
            {"strategy": "model_a", "query_latency_ms": {"median": 10.0, "p95": 18.0, "total": 100.0}, "runtime_stats": {"cache_hits": 5, "demand_requests": 10, "useful_prefetches": 2, "wasted_prefetches": 1, "unused_prefetches_still_resident": 1, "inference_ms_total": 3.0}},
            {"strategy": "model_a", "query_latency_ms": {"median": 12.0, "p95": 20.0, "total": 120.0}, "runtime_stats": {"cache_hits": 6, "demand_requests": 10, "useful_prefetches": 3, "wasted_prefetches": 2, "unused_prefetches_still_resident": 0, "inference_ms_total": 4.0, "prediction_requests": 7, "queued_prediction_jobs": 5, "dropped_jobs": 2}},
        ],
    }
    summary = summarize_suite(report)
    row = summary["strategies"][0]
    assert row["strategy"] == "model_a"
    assert row["trials"] == 2
    assert row["median_of_run_p50_ms"] == 11.0
    assert row["mean_cache_hit_pct"] == 55.0
    assert row["mean_prediction_requests"] == 7.0
    assert row["mean_queued_prediction_jobs"] == 5.0
    assert row["mean_dropped_prediction_jobs"] == 2.0
    assert summary["mode"] == "latency-injected"
    assert "not hardware latency" in summary["warning"]


def _write_query_results(path: Path, rows: list[tuple[str, float]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "elapsed_ms"])
        writer.writeheader()
        for query_id, elapsed_ms in rows:
            writer.writerow({"query_id": query_id, "elapsed_ms": elapsed_ms})
    return str(path)


def test_suite_summary_reports_paired_query_deltas(tmp_path: Path):
    stock_path = _write_query_results(tmp_path / "stock" / "query_results.csv", [("Q1", 10.0), ("Q2", 20.0)])
    model_path = _write_query_results(tmp_path / "model_b" / "query_results.csv", [("Q1", 8.0), ("Q2", 25.0)])
    report = {
        "mode": "real-storage-no-injected-delay",
        "database_path": "example.sqlite",
        "queries_per_run": 2,
        "seeds": [1],
        "strategies": ["stock", "model_b"],
        "warning": "VFS reads may be OS-cache reads.",
        "runs": [
            {"strategy": "stock", "trial": 1, "seed": 1, "query_results_path": stock_path, "query_latency_ms": {"median": 15.0, "p95": 20.0, "total": 30.0}},
            {"strategy": "model_b", "trial": 1, "seed": 1, "query_results_path": model_path, "query_latency_ms": {"median": 16.5, "p95": 25.0, "total": 33.0}},
        ],
    }
    summary = summarize_suite(report)
    paired = summary["paired_query_comparisons"]
    assert len(paired) == 1
    assert paired[0]["strategy"] == "model_b"
    assert paired[0]["baseline"] == "stock"
    assert paired[0]["matched_queries"] == 2
    assert paired[0]["mean_delta_ms"] == 1.5
    assert paired[0]["median_delta_ms"] == 1.5
    assert paired[0]["improved_query_pct"] == 50.0


def test_runtime_sensitivity_sweep_ranks_mocked_suite_results(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run_suite(database_path, model_dir, output_dir, **kwargs):
        calls.append(kwargs)
        cache_pages = kwargs["cache_pages"]
        mean_delta = -0.2 if cache_pages == 64 else 0.1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "output_dir": str(output_dir),
            "runs": 2,
            "summary_path": str(output_dir / "suite_summary.json"),
            "summary": {
                "strategies": [
                    {"strategy": "none", "median_of_run_p50_ms": 2.0},
                    {
                        "strategy": "model_b",
                        "median_of_run_p50_ms": 1.8,
                        "median_total_query_ms": 50.0,
                        "mean_prediction_requests": 10.0,
                    },
                ],
                "paired_query_comparisons": [
                    {
                        "strategy": "model_b",
                        "baseline": "none",
                        "mean_delta_ms": mean_delta,
                        "median_delta_ms": mean_delta,
                        "delta_ci95_low_ms": mean_delta - 0.1,
                        "delta_ci95_high_ms": mean_delta + 0.1,
                        "improved_query_pct": 60.0,
                    }
                ],
            },
        }

    monkeypatch.setattr(suite_module, "run_suite", fake_run_suite)
    report = run_runtime_sensitivity_sweep(
        "db.sqlite", "models", tmp_path,
        seeds=[1], strategies=["none", "model_b"],
        cache_pages_values=[128, 64], degree_values=[1],
        queue_size_values=[4], predict_on_values=["all"],
        predict_every_values=[1], baseline="none",
    )

    assert len(calls) == 2
    assert report["ranking"][0]["cache_pages"] == 64
    assert report["ranking"][0]["mean_delta_ms"] == -0.2
    assert (tmp_path / "runtime_sensitivity_report.json").exists()
    assert (tmp_path / "runtime_sensitivity_ranking.csv").exists()


def test_recover_suite_rebuilds_manifest_from_completed_runs(tmp_path: Path):
    run_dir = tmp_path / "trial-01" / "seed-7" / "hybrid"
    run_dir.mkdir(parents=True)
    _write_query_results(run_dir / "query_results.csv", [("Q1", 4.0)])
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "database_path": "example.sqlite", "query_count": 1,
        "mode": "latency-injected", "miss_delay_ms_injected": 0.2,
        "physical_io_note": "emulated", "query_latency_ms": {"median": 4.0, "p95": 4.0, "total": 4.0},
    }), encoding="utf-8")
    (run_dir / "runtime_stats.json").write_text(json.dumps({
        "worker_shutdown_clean": True, "hybrid_markov_jobs": 1, "hybrid_semantic_jobs": 0,
    }), encoding="utf-8")

    result = recover_suite(tmp_path)

    assert result["runs"] == 1
    assert (tmp_path / "suite_runs.json").exists()
    assert result["summary"]["strategies"][0]["strategy"] == "hybrid"
    assert result["summary"]["strategies"][0]["all_worker_shutdown_clean"] is True
