"""Paired runtime pilot and frozen-policy confirmation with measured worker cost.

All delays/channels are explicitly artificial storage scenarios. No physical
device speedup is inferred. Run the pilot before confirmatory runtime seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import numpy as np
import torch
from pagewise.runner import trace_workload
from pagewise.runledger import record_run, verify_ledger
from pagewise.statistics import declare_family


POLICIES = {
    "ranker": {"degree": 1, "confidence_threshold": 0.0, "predict_every": 1},
    "ranker_confident": {"degree": 1, "confidence_threshold": 0.7, "predict_every": 1},
    "ranker_stride4": {"degree": 1, "confidence_threshold": 0.7, "predict_every": 4},
}


def read_latencies(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["query_id"]: float(row["elapsed_ms"]) for row in csv.DictReader(handle)}


def run(database: Path, frozen: Path, out: Path, phase: str, policy_path: Path | None = None, queries=60):
    if out.exists():
        raise ValueError("Output exists; preserve the existing attempts and choose a new directory")
    out.mkdir(parents=True)
    model = frozen / "params"
    if phase == "pilot":
        seeds = [107, 108]
        scenarios = [("k_channel", 4, 0.2)]
        policies = dict(none={}, markov2={"degree": 1}, null={}, **POLICIES)
    else:
        if policy_path is None:
            raise ValueError("Confirmation requires the pilot's frozen policy.json")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy["checkpoint_sha256"] != hashlib.sha256((model / "model_ranker.pt").read_bytes()).hexdigest():
            raise ValueError("Runtime checkpoint changed after policy selection")
        seeds = [921, 922, 923, 924, 925]
        scenarios = [("shared", 1, 0.0), ("shared", 1, 0.2), ("k_channel", 4, 0.2), ("k_channel", 16, 0.2)]
        policies = {"none": {}, "markov2": {"degree": 1}, "selected_ranker": policy["settings"]}
    report = {"phase": phase, "evidence_class": "real APSW queries; injected delays are simulated storage, not physical I/O",
              "queries_per_seed": queries, "seeds": seeds, "scenarios": [], "runs": []}
    family = []
    for storage, channels, delay in scenarios:
        label = f"k{channels}-delay{delay}"
        paired = {name: {} for name in policies if name != "none"}
        collected = {name: [] for name in policies}
        for seed in seeds:
            order = list(policies)
            random.Random(seed * 101 + channels).shuffle(order)
            results = {}
            for name in order:
                strategy = "model_ranker" if name not in {"none", "null", "markov2"} else name
                target = out / label / f"seed-{seed}" / name
                config = {"strategy": strategy, "candidate_index_path": str(model / "candidate_index.json"),
                          "model_path": str(model / "model_ranker.pt") if strategy == "model_ranker" else None,
                          "cache_pages": 512, "degree": 1, "confidence_threshold": 0.0,
                          "queue_size": 64, "predict_on": "miss", "predict_every": 1,
                          "miss_delay_ms": delay, "storage_model": storage, "storage_channels": channels,
                          "drop_stale_jobs": True, "diagnostics_enabled": False,
                          "record_demand_reference": name == "none", "prefetch_workers": 1,
                          **policies[name]}
                started = time.monotonic()
                try:
                    trace_workload(database, target, query_count=queries, seed=seed, cache_kib=4096, prefetch_config=config)
                    stats = json.loads((target / "runtime_stats.json").read_text(encoding="utf-8"))
                    if stats.get("prefetch_read_errors") or not stats.get("worker_shutdown_clean"):
                        raise RuntimeError("Worker errors or unclean shutdown; run is invalid")
                    values = read_latencies(target / "query_results.csv")
                    results[name] = values
                    collected[name].extend(values.values())
                    row = {"scenario": label, "seed": seed, "policy": name, "configuration": config,
                           "mean_ms": float(np.mean(list(values.values()))), "p95_ms": float(np.percentile(list(values.values()), 95)),
                           "runtime_stats": stats, "directory": str(target)}
                    report["runs"].append(row)
                    record_run(out / "run_log.jsonl", {"phase": phase, "scenario": label, "policy": name, **config},
                               seed=seed, status="completed", metrics={"mean_ms": row["mean_ms"], "p95_ms": row["p95_ms"]},
                               outputs=[str(target / "query_results.csv"), str(target / "runtime_stats.json")])
                    print(f"{phase}/{label}/seed{seed}/{name}: mean={row['mean_ms']:.3f}ms p95={row['p95_ms']:.3f}ms "
                          f"{time.monotonic()-started:.1f}s", flush=True)
                except Exception as exc:
                    record_run(out / "run_log.jsonl", {"phase": phase, "scenario": label, "policy": name, **config},
                               seed=seed, status="failed", reason=repr(exc))
                    raise
                (out / "runtime.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            for name in paired:
                if results[name].keys() != results["none"].keys():
                    raise ValueError("Unmatched query identities")
                paired[name][seed] = [results[name][q] - results["none"][q] for q in results["none"]]
        summary = {name: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95)), "queries": len(v)}
                   for name, v in collected.items()}
        for name, sessions in paired.items():
            summary[name]["queries_improved_pct"] = float(100 * np.mean([d < 0 for ds in sessions.values() for d in ds]))
            summary[name]["queries_regressed_pct"] = float(100 * np.mean([d > 0 for ds in sessions.values() for d in ds]))
            family.append((f"{label}:{name}_vs_none", sessions))
        report["scenarios"].append({"label": label, "summary": summary})
    directions = {name: "less" for name, _ in family}
    report["comparisons"] = declare_family(family, favorable_directions=directions, iterations=5000,
                                           replication_unit="paired query executions clustered by workload seed")
    report["ledger_integrity"] = verify_ledger(out / "run_log.jsonl")
    if phase == "pilot":
        values = report["scenarios"][0]["summary"]
        winner = min(POLICIES, key=lambda n: values[n]["mean_ms"])
        policy = {"selected": winner, "settings": POLICIES[winner], "development_seeds": seeds,
                  "selection": "lowest paired mean query latency among the three declared neural policies",
                  "enable_recommended_on_pilot_mean": values[winner]["mean_ms"] < values["none"]["mean_ms"],
                  "checkpoint_sha256": hashlib.sha256((model / "model_ranker.pt").read_bytes()).hexdigest()}
        (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (out / "runtime.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RUNTIME COMPLETE", phase, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.db.resolve(), args.frozen.resolve(), args.out, args.phase, args.policy, args.queries)


if __name__ == "__main__":
    main()
