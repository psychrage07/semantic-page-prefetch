"""Matched selective-prefetch experiments, isolated from the frozen study.

The previous queue-optimization branch and its H1 data were unavailable. This
study uses the existing frozen checkpoint and fresh generated SQLite workloads;
it does not treat the pasted H1 figures as verified results.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DB = ROOT / "runs/large-branch-20261015/pagewise-large.sqlite"
FROZEN = ROOT / "runs/neural-improvement-20261020/frozen-final"
OUTPUT = ROOT / "runs/selective-prefetch-20261027"
DEV = [1101, 1102]
TEST = [4401, 4402, 4403, 4404, 4405]
QUERIES = 30
SCENARIOS = {"zero": 0.0, "controlled-delay": 0.2}
BASELINES = ["none", "markov2", "model_ranker"]
ABLATIONS = ["context", "deadline", "adaptive", "combined"]
SOURCE_FILES = ["selective_context.py", "selective_deadline.py", "selective_adaptive.py", "selective_prefetch.py", "queue_experiments.py", "calibrate_selective.py"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def source_digests():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def peak_rss_bytes():
    """Return process peak working set on Windows; None if unavailable."""
    if os.name != "nt":
        try:
            import resource
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if sys.platform != "darwin" else 1)
        except (ImportError, OSError):
            return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
                    ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.peak_working_set)


def initialize():
    from live_hybrid import assets
    evidence = assets()
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve existing experiment: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    protocol = {
        "evidence_class": "new experiment; missing queue-optimization H1 results are not treated as verified",
        "database": str(DB), "database_bytes": DB.stat().st_size,
        "checkpoint_sha256": evidence["checkpoint_sha256"], "frozen_source_sha256": evidence["source_sha256"],
        "development_seeds": DEV, "heldout_seeds": TEST, "queries_per_seed": QUERIES,
        "seed_audit": "No matching seeds in 561 available run_metadata files; absent prior queue artifacts cannot be audited.",
        "scenarios": SCENARIOS, "storage_model": "shared, one simulated service channel; requested delay, not physical device latency",
        "baselines": BASELINES, "ablations": ABLATIONS,
        "workload": "existing six-template synthetic Pagewise database workload, generated per seed",
        "pairing": "identical seed and query parameters, policy order randomized per seed/scenario",
        "cache": "512 Pagewise pages; 4096 KiB SQLite pager; OS cache not reset",
        "diagnostics": "enabled equally for all policies", "prefetch_workers": 1,
        "frozen_settings": {"degree": 1, "confidence_threshold": 0.7, "predict_on": "miss", "predict_every": 4, "queue_size": 64, "drop_stale_jobs": True},
        "selection": "Development seeds only. Fallback neural requires at least 40 transitions, 10 queries, and a positive paired 95% query-bootstrap lower bound versus Markov-1. Recursive traversal and indexed_lookup disabled unless new adequate development evidence supports them; no threshold tuned on held-out data.",
        "deadline": "Per-family median observed demand gap on zero-delay development traces; do not claim physical deadline guarantees.",
        "adaptive": "Simple net-benefit estimate from development costs and delayed online counters; no future accesses.",
        "test_rule": "freeze policy and code hashes before running held-out seeds; no re-tuning on test",
        "accuracy_scope": "offline on same no-prefetch VFS transitions; scheduling can skip some runtime predictions",
        "uncertainty": "paired hierarchical bootstrap clusters by seed and queries within seed; five seed clusters; exploratory by-family results",
        "success": "held-out paired mean latency vs none with 95% interval entirely below zero; compare current hybrid and Markov-2 too",
        "limitations": "OS cache is not controlled. Non-preemptible I/O already in progress can still delay demand; this is not a physical-storage benchmark."
    }
    write(OUTPUT / "protocol.json", protocol)
    print(f"Initialized independent experiment: {OUTPUT}", flush=True)


def worker(spec):
    # Each trial is a separate process so peak RSS and CPU cost are meaningful.
    import torch
    torch.set_num_threads(1)
    from pagewise import runner
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    config = dict(spec["config"])
    if spec["policy"] in ABLATIONS:
        from selective_prefetch import install_runtime
        settings = dict(spec["policy_config"])
        settings.update(settings["by_scenario"][spec["scenario"]])
        install_runtime(runner, spec["policy"], settings)
        config["strategy"] = "model_ranker"
    runner.trace_workload(DB, spec["output"], query_count=QUERIES, seed=spec["seed"],
                          cache_kib=4096, prefetch_config=config)
    out = Path(spec["output"])
    stats = json.loads((out / "runtime_stats.json").read_text(encoding="utf-8"))
    if stats.get("prefetch_read_errors") or not stats.get("worker_shutdown_clean") or not stats.get("worker_idle_before_close"):
        raise RuntimeError("Prefetch worker reported errors or did not shut down cleanly")
    write(out / "resource_usage.json", {"cpu_seconds": time.process_time() - cpu_start,
          "wall_seconds": time.perf_counter() - wall_start, "peak_process_rss_bytes": peak_rss_bytes(),
          "cpu_scope": "process CPU after importing dependencies; includes model initialization, SQL, trace, workers and shutdown",
          "rss_scope": "peak process working set since process start"})


def run_phase(phase, selected_policy=None):
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    if phase == "baseline":
        seeds, policies, policy_config = DEV, BASELINES, None
    elif phase == "development":
        if selected_policy not in ABLATIONS:
            raise ValueError("Development requires one explicitly named ablation")
        seeds, policies = DEV, [selected_policy]
        policy_config = json.loads((OUTPUT / "policy-development.json").read_text(encoding="utf-8"))
        if selected_policy == "combined":
            for scenario in SCENARIOS:
                for seed in DEV:
                    for policy in ("context", "deadline", "adaptive"):
                        if not (OUTPUT / phase / scenario / f"seed-{seed}" / policy / "resource_usage.json").is_file():
                            raise ValueError("Complete each individual ablation before the combined policy")
    elif phase == "heldout":
        seeds, policies = TEST, BASELINES + ABLATIONS
        policy_config = json.loads((OUTPUT / "policy-frozen.json").read_text(encoding="utf-8"))
        assert policy_config["source_sha256"] == source_digests(), "Selective source changed after freeze"
        assert policy_config["checkpoint_sha256"] == protocol["checkpoint_sha256"] == digest(FROZEN / "params/model_ranker.pt")
        from pagewise.runledger import source_hash
        assert source_hash() == protocol["frozen_source_sha256"], "Frozen predictor source changed"
    else:
        raise ValueError(phase)
    model = FROZEN / "params"
    for scenario, delay in SCENARIOS.items():
        for seed in seeds:
            order = list(policies)
            random.Random(seed * 997 + (1 if delay else 0)).shuffle(order)
            for policy in order:
                out = OUTPUT / phase / scenario / f"seed-{seed}" / policy
                if out.exists():
                    raise FileExistsError(f"Preserve prior attempt: {out}")
                out.mkdir(parents=True)
                cfg = {"strategy": policy, "candidate_index_path": str(model / "candidate_index.json"),
                       "model_path": str(model / "model_ranker.pt") if policy == "model_ranker" or policy in ABLATIONS else None,
                       "cache_pages": 512, "degree": 1, "confidence_threshold": 0.7,
                       "queue_size": 64, "predict_on": "miss", "predict_every": 4,
                       "miss_delay_ms": delay, "storage_model": "shared", "storage_channels": 1,
                       "drop_stale_jobs": True, "diagnostics_enabled": True, "prefetch_workers": 1,
                       "record_demand_reference": True}
                spec = {"seed": seed, "policy": policy, "scenario": scenario, "output": str(out),
                        "config": cfg, "policy_config": policy_config, "phase": phase}
                write(out / "trial.json", spec)
                start = time.monotonic()
                with (out / "worker.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, __file__, "worker", "--spec", str(out / "trial.json")],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Trial failed ({result.returncode}); preserved log: {out / 'worker.log'}")
                with (out / "query_results.csv").open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if len(rows) != QUERIES:
                    raise RuntimeError(f"Incomplete trial: {out}")
                timings = [float(r["elapsed_ms"]) for r in rows]
                print(f"{phase}/{scenario}/{seed}/{policy}: {len(rows)} queries, "
                      f"mean={sum(timings)/len(timings):.3f}ms, wall={time.monotonic()-start:.1f}s", flush=True)
    print(f"Completed {phase} {selected_policy or ''}", flush=True)


def freeze():
    destination = OUTPUT / "policy-frozen.json"
    if destination.exists():
        raise FileExistsError("Frozen configuration already exists")
    for scenario in SCENARIOS:
        for seed in DEV:
            for policy in ABLATIONS:
                if not (OUTPUT / "development" / scenario / f"seed-{seed}" / policy / "resource_usage.json").is_file():
                    raise ValueError("All development ablations must complete before freezing")
    policy = json.loads((OUTPUT / "policy-development.json").read_text(encoding="utf-8"))
    policy["source_sha256"] = source_digests()
    policy["freeze_note"] = "Development policy frozen before held-out evaluation; no changes based on test results."
    policy["development_offline_sha256"] = digest(OUTPUT / "development-offline-transitions.csv")
    write(destination, policy)
    print(f"Frozen policy {digest(destination)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["initialize", "worker", "baseline", "development", "freeze", "heldout"])
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--policy", choices=ABLATIONS)
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize()
    elif args.stage == "worker":
        worker(json.loads(args.spec.read_text(encoding="utf-8")))
    elif args.stage == "freeze":
        freeze()
    else:
        run_phase(args.stage, args.policy)
