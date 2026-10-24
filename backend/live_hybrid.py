"""Live execution of the existing selected checkpoint; no fitting or tuning.

Kept outside the frozen predictor-source directories. Each browser attempt
writes a separate result directory and uses a newly generated workload seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics

BACKEND = Path(__file__).resolve().parent
FROZEN = BACKEND / "runs/neural-improvement-20261020/frozen-final"
DATABASE = BACKEND / "runs/large-branch-20261015/pagewise-large.sqlite"
POLICY = BACKEND / "runs/neural-improvement-20261020/runtime-pilot-optimized/policy.json"
POLICIES = {"none": "No prefetch", "markov2": "Markov-2", "model_ranker": "Selected neural hybrid"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assets() -> dict:
    paths = [DATABASE, FROZEN / "freeze.json", FROZEN / "params/model_ranker.pt",
             FROZEN / "params/candidate_index.json", FROZEN / "params/page_map.json",
             FROZEN / "static-metadata.npz", POLICY]
    missing = [str(path.relative_to(BACKEND)) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("The trained-model test needs these local research artifacts: " + ", ".join(missing))
    declaration = json.loads((FROZEN / "freeze.json").read_text(encoding="utf-8"))
    selected = declaration["splits"]["params"]["models"]["selected"]
    checksum = sha(FROZEN / "params/model_ranker.pt")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if checksum != selected["sha256"] or checksum != policy["checkpoint_sha256"]:
        raise ValueError("The selected checkpoint does not match the frozen declaration and policy")
    from pagewise.runledger import source_hash
    if source_hash() != declaration["source_sha256"]:
        raise ValueError("The predictor source differs from the frozen study; this test requires its matching source")
    from pagewise.static_metadata import StaticMetadata, snapshot_identity
    metadata = StaticMetadata(FROZEN / "static-metadata.npz")
    if metadata.snapshot_identity != snapshot_identity(DATABASE):
        raise ValueError("The database does not match the frozen metadata snapshot")
    return {"available": True, "model": "Selected neural hybrid", "variant": selected["validation_selected_variant"],
            "checkpoint_sha256": checksum, "source_sha256": declaration["source_sha256"],
            "database_bytes": DATABASE.stat().st_size, "best_epoch": selected["best_epoch"],
            "settings": policy["settings"], "training": "Existing frozen weights; no retraining"}


def order(seed: int) -> list[str]:
    policies = list(POLICIES)
    random.Random(seed).shuffle(policies)
    return policies


def verify(out: Path, queries: int, seed: int):
    if (out / "live-plan.json").exists():
        raise ValueError("A live plan already exists; choose a new run directory")
    evidence = assets()
    model = out / "model"
    model.mkdir(parents=True, exist_ok=True)
    # Rebase only a run-local copy of the index; frozen artifacts stay immutable.
    index = json.loads((FROZEN / "params/candidate_index.json").read_text(encoding="utf-8"))
    index["layout_database_path"] = str(DATABASE)
    index["static_metadata_path"] = str(FROZEN / "static-metadata.npz")
    (model / "candidate_index.json").write_text(json.dumps(index), encoding="utf-8")
    shutil.copyfile(FROZEN / "params/page_map.json", model / "page_map.json")
    plan = {"schema_version": 1, **evidence, "queries_per_seed": queries, "seeds": [seed, seed + 1],
            "run_order": {str(value): order(value) for value in [seed, seed + 1]},
            "scope": "Fresh live workload; values may overlap training. Demonstration, not a new held-out confirmation.",
            "diagnostics_enabled": True, "miss_delay_ms": 0.0}
    (out / "live-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Verified frozen selected model, predictor source and database snapshot.", flush=True)
    print(f"Checkpoint {evidence['checkpoint_sha256']}", flush=True)


def run_policy(out: Path, policy: str, seed: int, queries: int):
    import torch
    torch.set_num_threads(1)
    from pagewise.runner import trace_workload
    plan = json.loads((out / "live-plan.json").read_text(encoding="utf-8"))
    if seed not in plan["seeds"] or queries != plan["queries_per_seed"] or policy not in POLICIES:
        raise ValueError("Runtime request does not match the live plan")
    checkpoint = FROZEN / "params/model_ranker.pt"
    if sha(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during the live test")
    target = out / f"seed-{seed}" / policy
    if target.exists():
        raise ValueError("This policy output already exists; preserve the previous result")
    config = {"strategy": policy, "candidate_index_path": str(out / "model/candidate_index.json"),
              "model_path": str(checkpoint) if policy == "model_ranker" else None,
              "cache_pages": 512, "degree": 1, "confidence_threshold": 0.0,
              "queue_size": 64, "predict_on": "miss", "predict_every": 1,
              "miss_delay_ms": 0.0, "storage_model": "shared", "storage_channels": 1,
              "drop_stale_jobs": True, "diagnostics_enabled": True, "prefetch_workers": 1}
    if policy == "model_ranker":
        config.update(plan["settings"])
    print(f"Running {POLICIES[policy]}: {queries} actual queries, workload seed {seed}", flush=True)
    trace_workload(DATABASE, target, query_count=queries, seed=seed, cache_kib=512, prefetch_config=config)
    stats = json.loads((target / "runtime_stats.json").read_text(encoding="utf-8"))
    if stats.get("prefetch_read_errors", 0):
        raise ValueError(f"{POLICIES[policy]} reported prediction/read errors; inspect runtime_stats.json")
    if policy == "model_ranker" and not stats.get("completed_jobs", 0):
        raise ValueError("The selected model scored no prediction jobs; increase the query count")
    print(f"Completed {POLICIES[policy]}; scored jobs={stats.get('completed_jobs', 0)}, "
          f"pages prefetched={stats.get('pages_prefetched', 0)}", flush=True)


def summarize(out: Path):
    plan = json.loads((out / "live-plan.json").read_text(encoding="utf-8"))
    if sha(FROZEN / "params/model_ranker.pt") != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during the comparison")
    rows = []
    samples = []
    workload_ids: dict[int, str] = {}
    for policy, label in POLICIES.items():
        times, jobs, prefetched, useful = [], 0, 0, 0
        demand, hits, unused, inference_ms = 0, 0, 0, 0.0
        for seed in plan["seeds"]:
            directory = out / f"seed-{seed}" / policy
            fingerprint = sha(directory / "workload.jsonl")
            if seed in workload_ids and workload_ids[seed] != fingerprint:
                raise ValueError("Policy workloads differ; the timing comparison is not paired")
            workload_ids[seed] = fingerprint
            with (directory / "query_results.csv").open(newline="", encoding="utf-8") as handle:
                measured = [float(row["elapsed_ms"]) for row in csv.DictReader(handle)]
            if len(measured) != plan["queries_per_seed"]:
                raise ValueError("Incomplete policy query timings")
            times.extend(measured)
            stats = json.loads((directory / "runtime_stats.json").read_text(encoding="utf-8"))
            jobs += stats.get("completed_jobs", 0)
            prefetched += stats.get("pages_prefetched", 0)
            useful += stats.get("useful_prefetches", 0)
            demand += stats.get("demand_requests", 0)
            hits += stats.get("cache_hits", 0)
            unused += stats.get("total_wasted_prefetches", 0)
            inference_ms += stats.get("inference_ms_total", 0.0)
            if policy == "model_ranker" and len(samples) < 3:
                with (directory / "predictions.jsonl").open(encoding="utf-8") as handle:
                    for line in handle:
                        event = json.loads(line)
                        if event.get("ranked_candidates"):
                            samples.append({"query_id": event.get("query_id"), "seed": seed,
                                            "history": event.get("history", [])[-4:],
                                            "candidates": event["ranked_candidates"][:3]})
                            if len(samples) >= 3:
                                break
        rows.append({"policy": policy, "label": label, "queries": len(times), "mean_ms": statistics.mean(times),
                     "median_ms": statistics.median(times), "scored_jobs": jobs,
                     "pages_prefetched": prefetched, "useful_pages": useful,
                     "cache_hit_pct": hits / demand * 100 if demand else None,
                     "unused_pages": unused, "inference_ms_per_job": inference_ms / jobs if jobs else None})
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Evaluation and runtime checkpoints differ")
    result = {"schema_version": 1, "artifact_type": "trained_hybrid_live", **plan, "evaluation": evaluation,
              "rows": rows, "prediction_samples": samples, "workloads_paired": True,
              "accuracy_note": "Fresh prediction accuracy is evaluated separately from runtime. Workload values may overlap training; this does not replace the saved held-out study."}
    (out / "hybrid-live.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "run", "evaluate", "summarize"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--policy", choices=list(POLICIES))
    args = parser.parse_args()
    if not 1 <= args.queries <= 100:
        raise ValueError("Query count must be between 1 and 100 per policy/seed")
    if args.stage == "verify":
        verify(args.out, args.queries, args.seed)
    elif args.stage == "run":
        run_policy(args.out, args.policy, args.seed, args.queries)
    elif args.stage == "evaluate":
        from live_evaluation import evaluate
        evaluate(args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
