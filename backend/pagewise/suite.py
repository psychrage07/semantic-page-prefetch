from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .runner import trace_workload

VALID_STRATEGIES = ("stock", "none", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid", "null", "oracle_next", "oracle_miss")


def recover_suite(output_dir: str | Path) -> dict[str, Any]:
    """Rebuild suite artifacts from completed run directories after interruption."""
    out = Path(output_dir).resolve()
    run_records: list[dict[str, Any]] = []
    for metadata_path in sorted(out.glob("trial-*/seed-*/*/run_metadata.json")):
        run_path = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        strategy = run_path.name
        if strategy not in VALID_STRATEGIES:
            continue
        trial_name = run_path.parent.parent.name
        seed_name = run_path.parent.name
        try:
            trial = int(trial_name.removeprefix("trial-"))
            seed = int(seed_name.removeprefix("seed-"))
        except ValueError:
            continue
        runtime_path = run_path / "runtime_stats.json"
        results_path = run_path / "query_results.csv"
        if not results_path.exists() or (strategy != "stock" and not runtime_path.exists()):
            continue
        run_records.append({
            "strategy": strategy, "trial": trial, "seed": seed,
            "output_dir": str(run_path), "query_count": metadata.get("query_count"),
            "query_results_path": str(results_path),
            "query_latency_ms": metadata.get("query_latency_ms", {}),
            "runtime_stats": json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None,
            "mode": metadata.get("mode"),
            "miss_delay_ms": metadata.get("miss_delay_ms_injected", 0.0),
            "storage_model": metadata.get("storage_model", "independent"),
            "drop_stale_jobs": metadata.get("drop_stale_jobs", False),
            "physical_io_note": metadata.get("physical_io_note"),
        })
    if not run_records:
        raise ValueError(f"No completed run_metadata.json files found under {out}")
    run_records.sort(key=lambda row: (row["trial"], row["seed"], row["strategy"]))
    first = run_records[0]
    strategies = [name for name in VALID_STRATEGIES if any(row["strategy"] == name for row in run_records)]
    report = {
        "schema_version": 1,
        "database_path": json.loads((Path(first["output_dir"]) / "run_metadata.json").read_text(encoding="utf-8")).get("database_path"),
        "queries_per_run": first.get("query_count"),
        "seeds": sorted({row["seed"] for row in run_records}),
        "strategies": strategies,
        "mode": first.get("mode"),
        "miss_delay_ms": first.get("miss_delay_ms"),
        "storage_model": first.get("storage_model", "independent"),
        "drop_stale_jobs": first.get("drop_stale_jobs", False),
        "warning": "Recovered from completed per-run artifacts. VFS reads may be satisfied by the OS cache; latency-injected runs do not establish physical I/O performance.",
        "runs": run_records,
    }
    (out / "suite_runs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = summarize_suite(report)
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["strategies"]:
        with (out / "suite_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["strategies"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["strategies"])
    return {"output_dir": str(out), "runs": len(run_records), "summary_path": str(out / "suite_summary.json"), "summary": summary}


def run_suite(database_path: str | Path, model_dir: str | Path, output_dir: str | Path, *,
              queries: int = 250, seeds: list[int] | None = None, strategies: list[str] | None = None,
              cache_kib: int = 4096, cache_pages: int = 256, degree: int = 2,
              threshold: float = 0.0, miss_delay_ms: float = 0.0,
              queue_size: int = 64, predict_on: str = "all", predict_every: int = 1,
              max_score_candidates: int = 0,
              storage_model: str = "independent",
              drop_stale_jobs: bool = False, diagnostics_enabled: bool = True,
              prefetch_workers: int = 1) -> dict[str, Any]:
    """Run comparable strategies on identical generated query parameters per seed.

    Strategy order rotates across repetitions to reduce simple order bias, but
    this is not a substitute for a documented OS cache reset between runs.
    """
    seeds = seeds or [101, 202, 303]
    strategies = strategies or list(VALID_STRATEGIES)
    invalid = set(strategies) - set(VALID_STRATEGIES)
    if invalid:
        raise ValueError(f"Unsupported strategies {sorted(invalid)}. Valid choices: {VALID_STRATEGIES}")
    if len(seeds) < 1:
        raise ValueError("At least one seed is required")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir).resolve()
    run_records = []

    for repeat_idx, seed in enumerate(seeds):
        oracle_trace_path = None
        if "oracle_next" in strategies or "oracle" in strategies:
            reference = out / "oracle-reference" / f"seed-{seed}"
            trace_workload(database_path, reference, query_count=queries, seed=seed,
                           cache_kib=cache_kib, mixed=True)
            oracle_trace_path = str(reference / "trace.jsonl")
        oracle_miss_path = None
        if "oracle_miss" in strategies:
            # A no-prefetch run with the same cache records, per query, the ordered
            # demand page stream and which pages missed. Delay is irrelevant to which
            # pages are demanded, so the reference runs with no injected delay.
            reference = out / "oracle-miss-reference" / f"seed-{seed}"
            ref_cfg = {
                "strategy": "none", "candidate_index_path": str(model_dir / "candidate_index.json"),
                "model_path": None, "semantic": False, "cache_pages": cache_pages, "degree": degree,
                "confidence_threshold": threshold, "queue_size": queue_size, "miss_delay_ms": 0.0,
                "predict_on": predict_on, "predict_every": predict_every,
                "max_score_candidates": max_score_candidates, "storage_model": storage_model,
                "drop_stale_jobs": drop_stale_jobs, "diagnostics_enabled": diagnostics_enabled,
                "record_demand_reference": True, "prefetch_workers": 1,
            }
            trace_workload(database_path, reference, query_count=queries, seed=seed,
                           cache_kib=cache_kib, mixed=True, prefetch_config=ref_cfg)
            oracle_miss_path = str(reference / "demand_reference.json")
        # Rotate the strategy ordering for each repeated workload seed.
        rotate = repeat_idx % len(strategies)
        ordered = strategies[rotate:] + strategies[:rotate]
        for strategy in ordered:
            run_path = out / f"trial-{repeat_idx + 1:02d}" / f"seed-{seed}" / strategy
            cfg = None
            if strategy != "stock":
                model_file = model_dir / ("model_a.pt" if strategy == "model_a" else "model_b.pt" if strategy in {"model_b", "hybrid"} else "")
                cfg = {
                    "strategy": strategy,
                    "candidate_index_path": str(model_dir / "candidate_index.json"),
                    "model_path": str(model_file) if strategy in {"model_a", "model_b", "hybrid"} else None,
                    "semantic": strategy in {"model_b", "hybrid"},
                    "cache_pages": cache_pages,
                    "degree": degree,
                    "confidence_threshold": threshold,
                    "queue_size": queue_size,
                    "miss_delay_ms": miss_delay_ms,
                    "predict_on": predict_on,
                    "predict_every": predict_every,
                    "max_score_candidates": max_score_candidates,
                    "storage_model": storage_model,
                    "drop_stale_jobs": drop_stale_jobs,
                    "diagnostics_enabled": diagnostics_enabled,
                    "oracle_trace_path": oracle_trace_path if strategy in {"oracle", "oracle_next"} else None,
                    "oracle_miss_path": oracle_miss_path if strategy == "oracle_miss" else None,
                    "prefetch_workers": int(prefetch_workers),
                }
            result = trace_workload(database_path, run_path, query_count=queries, seed=seed,
                                    cache_kib=cache_kib, mixed=True, prefetch_config=cfg,
                                    miss_delay_ms=miss_delay_ms if strategy == "stock" else 0.0)
            meta = json.loads((run_path / "run_metadata.json").read_text(encoding="utf-8"))
            run_records.append({
                "strategy": strategy, "trial": repeat_idx + 1, "seed": seed,
                "output_dir": str(run_path), "query_count": meta["query_count"],
                "query_results_path": str(run_path / "query_results.csv"),
                "query_latency_ms": meta["query_latency_ms"],
                "runtime_stats": json.loads((run_path / "runtime_stats.json").read_text(encoding="utf-8")) if (run_path / "runtime_stats.json").exists() else None,
                "mode": "latency-injected" if miss_delay_ms > 0 else "real-storage-no-injected-delay",
                "miss_delay_ms": miss_delay_ms,
                "storage_model": storage_model,
                "drop_stale_jobs": drop_stale_jobs,
                "physical_io_note": meta["physical_io_note"],
            })
            print(f"Completed trial={repeat_idx + 1} seed={seed} strategy={strategy}: p50={meta['query_latency_ms']['median']} ms")

    run_report = {
        "schema_version": 1,
        "database_path": str(Path(database_path).resolve()),
        "queries_per_run": queries,
        "seeds": seeds,
        "strategies": strategies,
        "sqlite_cache_kib": cache_kib,
        "custom_cache_pages": cache_pages,
        "prefetch_degree": degree,
        "confidence_threshold": threshold,
        "queue_size": queue_size,
        "predict_on": predict_on,
        "predict_every": predict_every,
        "max_score_candidates": max_score_candidates,
        "miss_delay_ms": miss_delay_ms,
        "storage_model": storage_model,
        "drop_stale_jobs": drop_stale_jobs,
        "mode": "latency-injected" if miss_delay_ms > 0 else "real-storage-no-injected-delay",
        "warning": "VFS reads may be satisfied by the OS cache. This report does not prove every miss caused physical disk I/O. Stock SQLite has no additional custom page cache, so it is a practical baseline rather than an identical-cache ablation.",
        "runs": run_records,
    }
    (out / "suite_runs.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    summary = summarize_suite(run_report)
    (out / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["strategies"]:
        with (out / "suite_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["strategies"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["strategies"])
    return {"output_dir": str(out), "runs": len(run_records), "summary_path": str(out / "suite_summary.json"), "summary": summary}


def run_runtime_sensitivity_sweep(database_path: str | Path, model_dir: str | Path, output_dir: str | Path, *,
                                  queries: int = 150, seeds: list[int] | None = None,
                                  strategies: list[str] | None = None, cache_kib: int = 4096,
                                  cache_pages_values: list[int] | None = None,
                                  degree_values: list[int] | None = None,
                                  queue_size_values: list[int] | None = None,
                                  predict_on_values: list[str] | None = None,
                                  predict_every_values: list[int] | None = None,
                                  threshold: float = 0.0, miss_delay_ms: float = 0.0,
                                  baseline: str = "none") -> dict[str, Any]:
    """Run a small grid of runtime configurations and rank paired latency deltas.

    This is intentionally a thin wrapper around run_suite so every combination
    keeps the same per-run artifacts and summary format as the normal benchmark.
    """
    seeds = seeds or [101, 202]
    strategies = strategies or ["none", "model_a", "model_b"]
    cache_pages_values = cache_pages_values or [256]
    degree_values = degree_values or [2]
    queue_size_values = queue_size_values or [64]
    predict_on_values = predict_on_values or ["all"]
    predict_every_values = predict_every_values or [1]
    if baseline not in strategies and baseline != "stock":
        raise ValueError(f"Baseline {baseline!r} must be included in strategies or be 'stock'")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    combo_idx = 0
    for cache_pages in cache_pages_values:
        for degree in degree_values:
            for queue_size in queue_size_values:
                for predict_on in predict_on_values:
                    for predict_every in predict_every_values:
                        combo_idx += 1
                        config = {
                            "cache_pages": int(cache_pages),
                            "degree": int(degree),
                            "queue_size": int(queue_size),
                            "predict_on": str(predict_on),
                            "predict_every": int(predict_every),
                        }
                        label = (
                            f"cfg-{combo_idx:03d}-cache{config['cache_pages']}-degree{config['degree']}"
                            f"-queue{config['queue_size']}-on{config['predict_on']}-every{config['predict_every']}"
                        )
                        suite_result = run_suite(
                            database_path, model_dir, out / label, queries=queries, seeds=seeds,
                            strategies=strategies, cache_kib=cache_kib, cache_pages=config["cache_pages"],
                            degree=config["degree"], threshold=threshold, miss_delay_ms=miss_delay_ms,
                            queue_size=config["queue_size"], predict_on=config["predict_on"],
                            predict_every=config["predict_every"],
                        )
                        summary = suite_result["summary"]
                        runs.append({
                            "config_id": label,
                            "config": config,
                            "suite_output_dir": suite_result["output_dir"],
                            "summary_path": suite_result["summary_path"],
                        })
                        strategy_rows = {row["strategy"]: row for row in summary.get("strategies", [])}
                        paired_rows = {
                            (row["strategy"], row["baseline"]): row
                            for row in summary.get("paired_query_comparisons", [])
                        }
                        for strategy in strategies:
                            if strategy == baseline:
                                continue
                            paired = paired_rows.get((strategy, baseline))
                            strategy_row = strategy_rows.get(strategy, {})
                            ranking_rows.append({
                                "config_id": label,
                                **config,
                                "strategy": strategy,
                                "baseline": baseline,
                                "mean_delta_ms": paired.get("mean_delta_ms") if paired else None,
                                "median_delta_ms": paired.get("median_delta_ms") if paired else None,
                                "delta_ci95_low_ms": paired.get("delta_ci95_low_ms") if paired else None,
                                "delta_ci95_high_ms": paired.get("delta_ci95_high_ms") if paired else None,
                                "improved_query_pct": paired.get("improved_query_pct") if paired else None,
                                "median_of_run_p50_ms": strategy_row.get("median_of_run_p50_ms"),
                                "median_total_query_ms": strategy_row.get("median_total_query_ms"),
                                "mean_cache_hit_pct": strategy_row.get("mean_cache_hit_pct"),
                                "mean_useful_prefetches": strategy_row.get("mean_useful_prefetches"),
                                "mean_wasted_prefetches_evicted": strategy_row.get("mean_wasted_prefetches_evicted"),
                                "mean_inference_ms_total": strategy_row.get("mean_inference_ms_total"),
                                "mean_prediction_requests": strategy_row.get("mean_prediction_requests"),
                                "mean_queued_prediction_jobs": strategy_row.get("mean_queued_prediction_jobs"),
                                "mean_dropped_prediction_jobs": strategy_row.get("mean_dropped_prediction_jobs"),
                            })
    ranking_rows.sort(key=lambda row: (
        float("inf") if row.get("mean_delta_ms") is None else float(row["mean_delta_ms"]),
        float("inf") if row.get("median_total_query_ms") is None else float(row["median_total_query_ms"]),
    ))
    report = {
        "schema_version": 1,
        "database_path": str(Path(database_path).resolve()),
        "model_dir": str(Path(model_dir).resolve()),
        "queries_per_run": queries,
        "seeds": seeds,
        "strategies": strategies,
        "baseline": baseline,
        "cache_kib": cache_kib,
        "threshold": threshold,
        "miss_delay_ms": miss_delay_ms,
        "mode": "latency-injected" if miss_delay_ms > 0 else "real-storage-no-injected-delay",
        "warning": "Sensitivity results are only as strong as the underlying run-suite repetitions; keep development sweeps separate from final experiments.",
        "runs": runs,
        "ranking": ranking_rows,
    }
    (out / "runtime_sensitivity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ranking_rows:
        with (out / "runtime_sensitivity_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ranking_rows)
    return report


def _bootstrap_median_ci(values: list[float], *, seed: int = 11, iterations: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.median(draw))
    samples.sort()
    return round(samples[int(0.025 * (iterations - 1))], 6), round(samples[int(0.975 * (iterations - 1))], 6)


def _bootstrap_mean_ci(values: list[float], *, seed: int = 19, iterations: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return round(values[0], 6), round(values[0], 6)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.mean(draw))
    samples.sort()
    return round(samples[int(0.025 * (iterations - 1))], 6), round(samples[int(0.975 * (iterations - 1))], 6)


def _query_latencies(run: dict[str, Any]) -> dict[str, float]:
    path_value = run.get("query_results_path")
    if path_value is None and run.get("output_dir"):
        path_value = str(Path(run["output_dir"]) / "query_results.csv")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["query_id"]: float(row["elapsed_ms"])
            for row in csv.DictReader(handle)
            if row.get("query_id") and row.get("elapsed_ms") not in (None, "")
        }


def _paired_query_comparisons(report: dict[str, Any]) -> list[dict[str, Any]]:
    by_trial_seed_strategy: dict[tuple[int, int, str], dict[str, Any]] = {}
    for run in report.get("runs", []):
        if "trial" in run and "seed" in run and "strategy" in run:
            by_trial_seed_strategy[(int(run["trial"]), int(run["seed"]), str(run["strategy"]))] = run

    rows = []
    strategies = list(report.get("strategies") or VALID_STRATEGIES)
    for baseline in ("stock", "none"):
        if not any(key[2] == baseline for key in by_trial_seed_strategy):
            continue
        for strategy in strategies:
            if strategy == baseline:
                continue
            deltas: list[float] = []
            matched_queries = 0
            matched_runs = 0
            for trial, seed, current_strategy in list(by_trial_seed_strategy):
                if current_strategy != strategy:
                    continue
                base_run = by_trial_seed_strategy.get((trial, seed, baseline))
                if base_run is None:
                    continue
                base_latencies = _query_latencies(base_run)
                strategy_latencies = _query_latencies(by_trial_seed_strategy[(trial, seed, strategy)])
                shared = sorted(set(base_latencies) & set(strategy_latencies))
                if not shared:
                    continue
                matched_runs += 1
                matched_queries += len(shared)
                deltas.extend(strategy_latencies[query_id] - base_latencies[query_id] for query_id in shared)
            if not deltas:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(deltas, seed=31 + len(strategy) + len(baseline))
            rows.append({
                "strategy": strategy,
                "baseline": baseline,
                "matched_runs": matched_runs,
                "matched_queries": matched_queries,
                "mean_delta_ms": round(statistics.mean(deltas), 6),
                "median_delta_ms": round(statistics.median(deltas), 6),
                "delta_ci95_low_ms": ci_low,
                "delta_ci95_high_ms": ci_high,
                "improved_query_pct": round(100.0 * sum(1 for value in deltas if value < 0) / len(deltas), 3),
                "unit": "paired query execution; delta = strategy latency - baseline latency",
            })
    order = {name: idx for idx, name in enumerate(VALID_STRATEGIES)}
    rows.sort(key=lambda row: (row["baseline"], order.get(row["strategy"], 999)))
    return rows


def summarize_suite(report: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("runs", []):
        grouped.setdefault(row["strategy"], []).append(row)
    rows = []
    for strategy, runs in grouped.items():
        p50s = [float(run["query_latency_ms"]["median"]) for run in runs if run.get("query_latency_ms", {}).get("median") is not None]
        p95s = [float(run["query_latency_ms"]["p95"]) for run in runs if run.get("query_latency_ms", {}).get("p95") is not None]
        totals = [float(run["query_latency_ms"]["total"]) for run in runs if run.get("query_latency_ms", {}).get("total") is not None]
        ci_low, ci_high = _bootstrap_median_ci(p50s)
        runtime = [run.get("runtime_stats") for run in runs if run.get("runtime_stats") is not None]
        def mean_stat(key: str) -> float | None:
            vals = [float(r[key]) for r in runtime if r.get(key) is not None]
            return round(statistics.mean(vals), 3) if vals else None
        rows.append({
            "strategy": strategy,
            "trials": len(runs),
            "median_of_run_p50_ms": round(statistics.median(p50s), 6) if p50s else None,
            "p50_run_median_ci95_low_ms": ci_low,
            "p50_run_median_ci95_high_ms": ci_high,
            "median_of_run_p95_ms": round(statistics.median(p95s), 6) if p95s else None,
            "median_total_query_ms": round(statistics.median(totals), 6) if totals else None,
            "mean_cache_hit_pct": round(100 * mean_stat("cache_hits") / max(1, mean_stat("demand_requests") or 0), 3) if runtime and mean_stat("demand_requests") else None,
            "mean_useful_prefetches": mean_stat("useful_prefetches"),
            "mean_wasted_prefetches_evicted": mean_stat("wasted_prefetches"),
            "mean_unused_prefetches_still_resident": mean_stat("unused_prefetches_still_resident"),
            "mean_useful_prefetch_pct": mean_stat("useful_prefetch_pct"),
            "mean_wasted_prefetch_evicted_pct": mean_stat("wasted_prefetch_evicted_pct"),
            "mean_unused_prefetch_resident_pct": mean_stat("unused_prefetch_resident_pct"),
            "mean_wasted_prefetch_total_pct": mean_stat("wasted_prefetch_total_pct"),
            "mean_prefetch_hit_coverage_pct": mean_stat("prefetch_hit_coverage_pct"),
            "mean_inference_ms_total": mean_stat("inference_ms_total"),
            "mean_inference_ms_per_job": mean_stat("inference_ms_per_job"),
            "mean_queue_wait_ms_mean": mean_stat("queue_wait_ms_mean"),
            "mean_queue_wait_ms_max": mean_stat("queue_wait_ms_max"),
            "mean_prefetch_read_ms_total": mean_stat("prefetch_read_ms_total"),
            "mean_prefetch_read_ms_mean": mean_stat("prefetch_read_ms_mean"),
            "mean_stale_jobs_detected": mean_stat("stale_jobs_detected"),
            "mean_stale_jobs_dropped": mean_stat("stale_jobs_dropped"),
            "mean_demand_misses_while_prefetch_in_flight": mean_stat("demand_misses_while_prefetch_in_flight"),
            "mean_demand_use_lead_time_ms": mean_stat("demand_use_lead_time_ms_mean"),
            "mean_prediction_observations": mean_stat("prediction_observations"),
            "mean_skipped_prediction_observations": mean_stat("skipped_prediction_observations"),
            "mean_prediction_requests": mean_stat("prediction_requests"),
            "mean_queued_prediction_jobs": mean_stat("queued_prediction_jobs"),
            "mean_completed_prediction_jobs": mean_stat("completed_jobs"),
            "mean_dropped_prediction_jobs": mean_stat("dropped_jobs"),
            "mean_coalesced_prediction_jobs": mean_stat("coalesced_jobs"),
            "mean_candidates_pruned_cached_or_pending": mean_stat("candidates_pruned_cached_or_pending"),
            "mean_candidates_pre_ranked_out": mean_stat("candidates_pre_ranked_out"),
            "mean_worker_drain_ms": mean_stat("worker_drain_ms"),
            "mean_worker_shutdown_ms": mean_stat("worker_shutdown_ms"),
            "all_worker_shutdown_clean": all(bool(item.get("worker_shutdown_clean")) for item in runtime) if runtime else None,
            "mean_hybrid_markov_jobs": mean_stat("hybrid_markov_jobs"),
            "mean_hybrid_semantic_jobs": mean_stat("hybrid_semantic_jobs"),
            "storage_model": report.get("storage_model", "independent"),
            "drop_stale_jobs": report.get("drop_stale_jobs", False),
            "latency_mode": report.get("mode"),
        })
    # Keep output order stable and useful to a presenter.
    order = {name: idx for idx, name in enumerate(VALID_STRATEGIES)}
    rows.sort(key=lambda row: order.get(row["strategy"], 999))
    return {
        "schema_version": 1,
        "mode": report.get("mode"),
        "miss_delay_ms": report.get("miss_delay_ms"),
        "storage_model": report.get("storage_model", "independent"),
        "drop_stale_jobs": report.get("drop_stale_jobs", False),
        "database_path": report.get("database_path"),
        "queries_per_run": report.get("queries_per_run"),
        "seeds": report.get("seeds"),
        "warning": report.get("warning"),
        "method_note": "Strategy rows bootstrap independent run-level p50 values; paired_query_comparisons bootstrap matched per-query latency deltas across paired workload seeds. Useful and wasted prefetch rates share the consistent denominator of total prefetched pages.",
        "strategies": rows,
        "paired_query_comparisons": _paired_query_comparisons(report),
    }
