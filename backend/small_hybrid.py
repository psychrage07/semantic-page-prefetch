"""Train the selected neural-Markov architecture for one isolated small demo."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import copy
from collections import Counter

import numpy as np
import torch
from torch import nn

from pagewise.dataset import load_jsonl
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex, MLPScorer, page_map_digest
from pagewise.offline import load_cases
from pagewise.ranking import (RankerConfig, build_groups, evaluate_ranker, logits,
                              fit_parameter_scales, fit_temperature)
from pagewise.suite import run_suite


def train_small_ranker(train: dict, val: dict, config: RankerConfig, seed: int = 101):
    """Select the best trained epoch on validation, retaining the Markov baseline separately."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLPScorer(train["x"].shape[-1], hidden=config.hidden)
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)
    baseline = evaluate_ranker(model, val, config)["top1_macro_pct"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ids = np.flatnonzero(train["target"] >= 0)
    if not len(ids):
        raise ValueError("No eligible training transitions")
    frequencies = Counter(train["query_id"][i] for i in ids)
    weights = np.asarray([1 / frequencies.get(q, 1) for q in train["query_id"]], np.float32)
    weights *= len(ids) / len(frequencies)
    best, best_accuracy, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        shuffled = rng.permutation(ids)
        for start in range(0, len(shuffled), 128):
            batch = shuffled[start:start + 128]
            x = torch.from_numpy(train["x"][batch])
            prior = torch.from_numpy(train["prior"][batch])
            valid = torch.from_numpy(train["pages"][batch] != -1)
            target = torch.from_numpy(train["target"][batch])
            loss = (nn.functional.cross_entropy(logits(model, x, prior, valid, config), target,
                                               reduction="none") * torch.from_numpy(weights[batch])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        accuracy = evaluate_ranker(model, val, config)["top1_macro_pct"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_top1_pct": accuracy})
        if accuracy > best_accuracy + 1e-6:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"config": asdict(config), "seed": seed, "best_epoch": best_epoch,
                   "best_validation_macro_top1_pct": best_accuracy,
                   "untrained_markov_validation_macro_top1_pct": baseline,
                   "eligible_training_groups": len(ids), "history": history,
                   "selection_metric": "validation macro Top-1 among trained epochs; epoch zero reported separately"}


def run(root: Path) -> None:
    root = root.resolve()
    models, trace = root / "models", root / "trace"
    output = root / "selected-hybrid"
    output.mkdir(parents=True, exist_ok=False)
    train = load_jsonl(models / "train_sequences.jsonl")
    val = load_jsonl(models / "validation_sequences.jsonl")
    cases = load_cases(trace / "workload.jsonl")
    page_map = json.loads((trace / "page_map.json").read_text(encoding="utf-8"))
    db = root / "pagewise-demo.sqlite"
    index = CandidateIndex.build(train, 128, strategy="layout", page_map=page_map,
                                 layout_database_path=str(db))
    index.page_map_sha256 = page_map_digest(page_map)
    counts = ML.build_training_transition_counts(train, max_candidates=index.max_candidates)
    scales = fit_parameter_scales(train, cases)
    config = RankerConfig("hybrid_semantics", variant="b2", objective="listwise",
                          hidden=64, epochs=16, prior_weight=1.0)
    training = build_groups(train, index, page_map, counts, config, cases, scales, max_per_query=16)
    validation = build_groups(val, index, page_map, counts, config, cases, scales, max_per_query=16)
    print("Training selected neural-Markov architecture on this small database", flush=True)
    model, fit = train_small_ranker(training, validation, config, seed=101)
    payload = {"config": asdict(config), "state_dict": model.state_dict(),
               "input_dim": training["x"].shape[-1], "normalization_max_page": index.max_page,
               "counts": counts, "parameter_scales": scales, "fit": fit,
               "calibration": fit_temperature(model, validation, config)}
    torch.save(payload, output / "model_ranker.pt")
    (output / "candidate_index.json").write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
    (output / "page_map.json").write_text(json.dumps(page_map), encoding="utf-8")
    # Open held-out examples only after the training checkpoint is fixed.
    test = load_jsonl(models / "test_sequences.jsonl")
    test_groups = build_groups(test, index, page_map, counts, config, cases, scales)
    scores = evaluate_ranker(model, test_groups, config)
    report = {"architecture": "hybrid_semantics", "training_seed": 101,
              "best_epoch": fit["best_epoch"], "neural_residual_trained": fit["best_epoch"] > 0,
              "checkpoint_sha256": hashlib.sha256((output / "model_ranker.pt").read_bytes()).hexdigest(),
              "test": scores, "fit": fit,
              "note": "New weights and candidate index for this database; the original research checkpoint is unchanged."}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected checkpoint epoch={fit['best_epoch']}; held-out Top-1={scores['top1_micro_pct']:.2f}%", flush=True)
    suite = run_suite(db, output, output / "suite", queries=30, seeds=[101, 202],
                      strategies=["model_ranker"], cache_kib=128, cache_pages=128, degree=2)
    # Verify identical generated SQL and parameters against the existing suite.
    for trial, seed in enumerate((101, 202), 1):
        relative = Path(f"trial-{trial:02d}") / f"seed-{seed}"
        original = root / "suite-local" / relative / "none" / "workload.jsonl"
        current = output / "suite" / relative / "model_ranker" / "workload.jsonl"
        if original.read_bytes() != current.read_bytes():
            raise ValueError("Selected hybrid workload differs from the original suite")
        stats = json.loads((current.parent / "runtime_stats.json").read_text(encoding="utf-8"))
        if stats.get("prefetch_read_errors") or not stats.get("worker_shutdown_clean"):
            raise ValueError("Selected hybrid worker did not finish cleanly")
    dashboard_path = root / "dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    if scores["transitions"] != dashboard["prediction"]["policies"][0]["evaluated_next_page_predictions"]:
        raise ValueError("Held-out transition counts do not match the other predictors")
    dashboard["prediction"]["policies"].append({"policy": "model_ranker",
        "evaluated_next_page_predictions": scores["transitions"],
        "top1_accuracy_pct": scores["top1_micro_pct"], "top5_accuracy_pct": scores["top5_micro_pct"],
        "candidate_recall_pct": scores["candidate_recall_pct"]})
    dashboard["selected_hybrid"] = {"best_epoch": fit["best_epoch"],
        "neural_residual_trained": fit["best_epoch"] > 0, "checkpoint_sha256": report["checkpoint_sha256"],
        "note": report["note"], "runtime_order": "Executed after the original suite; OS cache was not reset."}
    dashboard["runtime_suites"][0]["strategies"].extend(suite["summary"]["strategies"])
    # Evaluate the routing policy on the identical held-out transitions, then
    # measure it using the existing asynchronous worker and matching SQL seeds.
    from workspace import context_configuration
    from selective_context import context_action
    from selective_prefetch import install_runtime
    from pagewise import runner
    from live_evaluation import metrics
    context_config = context_configuration("small")
    context_rows = []
    for seq in test:
        for position in range(1, len(seq["pages"])):
            history = seq["pages"][max(0, position - 8):position]
            action, _ = context_action(index, history, seq["workload_family"], context_config)
            first = index.successors.get(history[-1], [])
            second = index.second_order.get(f"{history[-2]}:{history[-1]}", []) if len(history) > 1 else []
            predicted = second if action == "markov2" else first if action == "markov1" else []
            target = seq["pages"][position]
            context_rows.append({"query_id": seq["query_id"], "family": seq["workload_family"],
                "top1": bool(predicted and predicted[0] == target), "top5": target in predicted[:5],
                "candidate_hit": target in predicted, "skipped": not predicted})
    context_scores = metrics(context_rows)
    print("Measuring context-filtered policy (Markov-2, Markov-1 or skip; neural fallback disabled)", flush=True)
    original_runtime = runner.PredictiveRuntime
    try:
        install_runtime(runner, "context", context_config)
        context_suite = run_suite(db, output, output / "context-suite", queries=30, seeds=[101, 202],
            strategies=["model_ranker"], cache_kib=128, cache_pages=128, degree=1,
            predict_on="miss", predict_every=4, drop_stale_jobs=True)
    finally:
        runner.PredictiveRuntime = original_runtime
    for trial, seed in enumerate((101, 202), 1):
        relative = Path(f"trial-{trial:02d}") / f"seed-{seed}"
        current = output / "context-suite" / relative / "model_ranker"
        reference = root / "suite-local" / relative / "none/workload.jsonl"
        stats = json.loads((current / "runtime_stats.json").read_text(encoding="utf-8"))
        if (current / "workload.jsonl").read_bytes() != reference.read_bytes() or stats.get("prefetch_read_errors") or not stats.get("worker_shutdown_clean"):
            raise ValueError("Context comparison workload or worker verification failed")
    dashboard["prediction"]["policies"].append({"policy": "context",
        "evaluated_next_page_predictions": context_scores["transitions"],
        "top1_accuracy_pct": context_scores["top1_micro_pct"], "top5_accuracy_pct": context_scores["top5_micro_pct"],
        "candidate_recall_pct": context_scores["candidate_recall_pct"]})
    for row in context_suite["summary"]["strategies"]:
        dashboard["runtime_suites"][0]["strategies"].append({**row, "strategy": "context"})
    dashboard["context_policy"] = {"scope": context_config["scope"],
        "skipped_transitions": sum(row["skipped"] for row in context_rows),
        "runtime_settings": {"degree": 1, "predict_on": "miss", "predict_every": 4, "max_jobs_per_query": 8},
        "runtime_order": "Measured after the hybrid; OS cache not reset."}
    (output / "context-evaluation.json").write_text(json.dumps({"config": context_config,
        "metrics": context_scores, "runtime_summary": context_suite["summary"]}, indent=2), encoding="utf-8")
    dashboard_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print("Selected hybrid and context-policy results saved to dashboard.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
