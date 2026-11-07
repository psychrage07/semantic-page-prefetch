"""Train and evaluate P4 using the existing idle executor and semantic MLP pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import torch

from idle_experiment import BACKEND, DEFAULT_ROOT, execute, fingerprint, make_cases, workload_pool, write_json, query_quality
from idle_markov import QueryMarkov
from idle_neural import IdleNeuralHybrid, early_miss_records
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex
from pagewise.runledger import source_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_models(training, validation, metadata, model_dir, out, database_identity, epochs=16):
    training, validation = early_miss_records(training), early_miss_records(validation)
    prior = QueryMarkov(2, "template").fit(training)
    prior.save(out / "early-miss-prior.json")
    index = CandidateIndex.build(training, max_candidates=64)
    counts = ML.build_training_transition_counts(training, max_candidates=64)
    contexts = {row["template_id"]: {"workload_family": row["family"], "plan_class": row["plan_class"]} for row in training}
    models, fits = {}, {}
    for policy, no_semantics in (("P4-neural-semantics", False), ("P4-no-semantics", True)):
        model = IdleNeuralHybrid(prior, index, counts, metadata, contexts, no_semantics=no_semantics)
        started = time.perf_counter_ns()
        fit = model.fit(training, validation, seed=101, epochs=epochs)
        fit["training_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
        model.save(out / policy, fit=fit, database_identity=database_identity,
                   metadata_path=model_dir / "page_map.json")
        # Reload the serialized weights; runtime never evaluates an unsaved
        # training object. Checksum is recorded before opening test queries.
        models[policy] = IdleNeuralHybrid.load(out / policy, metadata, database_identity)
        fits[policy] = {**fit, "config": asdict(model.config), "checkpoint_sha256": sha(out / policy / "idle_neural.pt")}
        print(f"Trained {policy}: epoch {fit['selected_epoch']} selected from {fit['epochs_run']}; validation early-miss recall {fit['validation_selected']['early_miss_recall_macro_pct']:.2f}%", flush=True)
    return models, fits, prior


def run(args):
    torch.set_num_threads(1)
    out, database, model_dir = args.out.resolve(), args.database.resolve(), args.model_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before, original_source, original_checkpoint = fingerprint(database), source_hash(), sha(model_dir / "model_ranker.pt")
    extension_names = ["idle_neural.py", "idle_neural_experiment.py", "idle_runtime.py", "idle_markov.py", "idle_experiment.py"]
    extensions = {name: sha(BACKEND / name) for name in extension_names}
    train_pool, val_pool = workload_pool(database, 1101), workload_pool(database, 1201)
    training, validation, train_sql = [], [], set()
    for seed in (1101, 1102, 1103):
        cases = make_cases(train_pool, 36, seed, prefix=f"T{seed}-")
        train_sql.update((case.sql, tuple(case.params)) for case in cases)
        _, _, records = execute(database, model_dir, out / f"training-{seed}", cases, policy="P0", cache_pages=32)
        training.extend(records)
    validation_sql = set()
    for seed in (1201, 1202):
        cases = make_cases(val_pool, 24, seed, prefix=f"V{seed}-")
        validation_sql.update((case.sql, tuple(case.params)) for case in cases)
        _, _, records = execute(database, model_dir, out / f"validation-{seed}", cases, policy="P0", cache_pages=32)
        validation.extend(records)
    metadata = json.loads((model_dir / "page_map.json").read_text(encoding="utf-8"))
    original_index = json.loads((model_dir / "candidate_index.json").read_text(encoding="utf-8"))
    if original_index.get("static_metadata_path"):
        from pagewise.static_metadata import StaticMetadata
        metadata = StaticMetadata(original_index["static_metadata_path"])
    models, fits, miss_prior = fit_models(training, validation, metadata, model_dir, out, before, args.epochs)
    models["P3-page-set"] = QueryMarkov(2, "template").fit(training)
    models["P3-early-miss"] = miss_prior
    policies = ["P0", "P1", "P2", "P3-page-set", "P3-early-miss", "P4-neural-semantics", "P4-no-semantics"]
    # Settings and the focused replication condition are fixed before any P4
    # evaluation results exist. No winner is picked from the test matrix.
    conditions = [(gap, delay, workload, args.seed) for gap, delay, workload in itertools.product(
        (0, 50), (0, .2), ("repeated", "novel"))]
    conditions += [(50, .2, "repeated", seed) for seed in (9001, 9002, 9003)]
    protocol = {"schema_version": 2, "phase": "P4 query-level semantic neural-Markov idle experiment",
        "database": str(database), "model_dir": str(model_dir), "database_identity": before,
        "frozen_source_sha256": original_source, "original_checkpoint_sha256": original_checkpoint,
        "extension_source_sha256": extensions, "training_queries": len(training), "validation_queries": len(validation),
        "training_seeds": [1101, 1102, 1103], "validation_parameter_pool_seed": 1201,
        "validation_query_overlap_count": len(validation_sql & train_sql), "fits": fits,
        "conditions": conditions, "queries_per_trial": args.queries, "policies": policies,
        "primary_condition": {"gap_ms": 50, "artificial_delay_ms": .2, "workload": "repeated", "cache_pages": 32,
                              "replication_seeds": [9001, 9002, 9003]},
        "P0": "Same bounded cache, no prefetch", "P1": "Existing within-query Markov-2",
        "P2": "Existing frozen matching neural-Markov checkpoint; no retraining",
        "P3-page-set": "Idle query-template Markov-2 trained on full next-query page sets",
        "P3-early-miss": "Idle Markov-2 trained on the same first-16-distinct-miss targets as P4",
        "P4": "Newly trained adaptation: existing B2 semantic features and width-64 MLP plus query-level Markov prior",
        "P4-no-semantics": "Retrained control with semantic blocks zeroed, otherwise same targets/candidates/architecture",
        "admission": "All idle policies: exclude cached/pending pages before ranking, at most 16 insert candidates, empirical training frequency >=0.25, 25% speculative quota, hot-page protection and cancellation at query arrival",
        "confidence": "Training conditional early-miss/page-set frequency, not an uncalibrated neural probability",
        "information": "Only completed query templates and last eight observed pages; unknown next SQL, template, parameters and pages are never passed to the online predictor",
        "weights": "P4 is a separately trained query-level experimental variant; original selected weights are unchanged",
        "scope": "Controlled familiar-value and query-level novel-template sessions; deliberate repeated-value overlap. No OS cache reset; nonzero delay is artificial",
        "selection": "Validation early-miss recall at fixed budget among genuinely trained epochs; no evaluation results used to choose weights/settings"}
    write_json(out / "protocol.json", protocol)
    results = []
    for condition_id, (gap, delay, workload, seed) in enumerate(conditions):
        cases = make_cases(train_pool, args.queries, seed, workload, prefix="E")
        order = list(policies); random.Random(seed + condition_id).shuffle(order)
        baseline_records, expected, reports = None, None, []
        for policy in order:
            report, rows, records = execute(database, model_dir, out / f"condition-{condition_id:02d}" / policy,
                cases, policy=policy, cache_pages=32, gap_ms=gap, delay_ms=delay, query_model=models.get(policy))
            values = [(row["query_id"], row["rows"], row["result_sha256"]) for row in rows]
            if expected is not None and expected != values:
                raise ValueError("P4 SQL results differ across policies")
            expected = values
            if policy == "P0": baseline_records = records
            report.update({"condition_id": condition_id, "workload": workload, "seed": seed,
                           "output": str(out / f"condition-{condition_id:02d}" / policy)})
            reports.append(report)
            print(f"{condition_id+1}/{len(conditions)} seed={seed} {workload} gap={gap} delay={delay} {policy}: mean={report['mean_ms']:.3f}ms useful={report['runtime']['useful_prefetches']}/{report['runtime']['pages_prefetched']}", flush=True)
        for report in reports:
            if report["policy"] in models:
                report["prediction_quality"] = query_quality(models[report["policy"]], baseline_records, 16, .25)
        results.extend(reports)
        write_json(out / "results.json", {"protocol": protocol, "results": results})
    if before != fingerprint(database) or original_source != source_hash() or original_checkpoint != sha(model_dir / "model_ranker.pt"):
        raise RuntimeError("Original frozen artifacts changed")
    if extensions != {name: sha(BACKEND / name) for name in extension_names}:
        raise RuntimeError("Experiment source changed during evaluation")
    for policy, fit in fits.items():
        if fit["checkpoint_sha256"] != sha(out / policy / "idle_neural.pt"):
            raise RuntimeError("P4 weights changed during evaluation")
    write_json(out / "verification.json", {"database_unchanged": True, "frozen_source_unchanged": True,
        "original_checkpoint_unchanged": True, "p4_checkpoints_unchanged": True, "experiment_source_unchanged": True,
        "sql_results_identical": True, "workers_clean": True, "trials": len(results)})
    print(f"Verified P4 experiment: {len(results)} trials, {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_ROOT / "pagewise-demo.sqlite")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ROOT / "selected-hybrid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    if args.queries < 2 or args.epochs < 1: parser.error("Use at least two queries and one epoch")
    run(args)
