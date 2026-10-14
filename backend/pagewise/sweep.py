from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import evaluate_predictors
from .models import train_models


def _bootstrap_mean_ci(values: list[float], seed: int = 101, iterations: int = 4000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return round(values[0], 4), round(values[0], 4)
    rng = random.Random(seed)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(iterations)]
    means.sort()
    return round(means[int(0.025 * (iterations - 1))], 4), round(means[int(0.975 * (iterations - 1))], 4)


def train_evaluate_sweep(sequences_path: str | Path, page_map_path: str | Path, output_dir: str | Path, *,
                         seeds: list[int] | None = None, split_seed: int = 7, epochs: int = 15,
                         max_candidates: int = 32, hidden: int = 128, top_k: int = 5,
                         semantic_ablations: dict[str, list[str]] | None = None) -> dict[str, Any]:
    seeds = seeds or [7, 17, 27]
    if not seeds:
        raise ValueError("At least one model seed is required")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = out / f"seed-{seed}"
        training = train_models(sequences_path, page_map_path, seed_dir, seed=seed, split_seed=split_seed,
                                epochs=epochs, max_candidates=max_candidates, hidden=hidden,
                                semantic_ablations=semantic_ablations)
        evaluation = evaluate_predictors(seed_dir / "test_sequences.jsonl", page_map_path,
                                         seed_dir / "candidate_index.json", seed_dir / "model_a.pt",
                                         seed_dir / "model_b.pt", seed_dir / "evaluation", top_k=top_k,
                                         semantic_ablations=semantic_ablations,
                                         retrained_ablation_dir=seed_dir if semantic_ablations else None)
        row = {
            "seed": seed, "split_seed": split_seed,
            "train_executions": training["counts"]["train_executions"],
            "validation_executions": training["counts"]["validation_executions"],
            "test_executions": training["counts"]["test_executions"],
            "policies": evaluation["policies"],
            "model_a": evaluation["policies"].get("model_a", {}),
            "model_b": evaluation["policies"].get("model_b", {}),
            "evaluation_path": str(seed_dir / "evaluation" / "predictor_metrics.json"),
            "training_report_path": str(seed_dir / "training_report.json"),
        }
        run_rows.append(row)
        print(f"Trained/evaluated seed {seed}: Model A top-1={row['model_a'].get('top1_accuracy_pct')}%, Model B top-1={row['model_b'].get('top1_accuracy_pct')}%")

    aggregate: dict[str, Any] = {}
    policy_names = sorted({policy for row in run_rows for policy in row.get("policies", {})})
    for model_name in policy_names:
        aggregate[model_name] = {}
        for metric in ("top1_accuracy_pct", "top5_accuracy_pct", "candidate_recall_pct"):
            vals = [float(row["policies"][model_name][metric]) for row in run_rows if row.get("policies", {}).get(model_name, {}).get(metric) is not None]
            lo, hi = _bootstrap_mean_ci(vals, seed=100 + len(metric) + len(model_name))
            aggregate[model_name][metric] = {
                "runs": len(vals), "mean": round(statistics.mean(vals), 4) if vals else None,
                "std_sample": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0 if vals else None,
                "bootstrap_mean_ci95_low": lo, "bootstrap_mean_ci95_high": hi,
            }
    delta_vs_model_b: dict[str, Any] = {}
    for policy in policy_names:
        if policy == "model_b":
            continue
        delta_vs_model_b[policy] = {}
        for metric in ("top1_accuracy_pct", f"top{top_k}_accuracy_pct"):
            deltas = []
            for row in run_rows:
                baseline = row.get("policies", {}).get("model_b", {}).get(metric)
                current = row.get("policies", {}).get(policy, {}).get(metric)
                if baseline is not None and current is not None:
                    deltas.append(float(current) - float(baseline))
            lo, hi = _bootstrap_mean_ci(deltas, seed=700 + len(metric) + len(policy))
            delta_vs_model_b[policy][metric] = {
                "runs": len(deltas),
                "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
                "bootstrap_mean_delta_ci95_low": lo,
                "bootstrap_mean_delta_ci95_high": hi,
            }
    report = {
        "schema_version": 1,
        "training_seeds": seeds,
        "fixed_split_seed": split_seed,
        "epochs": epochs,
        "candidate_count": max_candidates,
        "hidden_units": hidden,
        "semantic_ablations": semantic_ablations or {},
        "note": "Training-seed intervals describe initialization/training variation on the fixed split. They are not confidence intervals for all future database workloads.",
        "aggregate": aggregate,
        "delta_vs_model_b": delta_vs_model_b,
        "runs": run_rows,
    }
    (out / "training_sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
