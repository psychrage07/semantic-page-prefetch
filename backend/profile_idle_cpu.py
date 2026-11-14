"""Fresh-process CPU/thread diagnosis using existing saved P4 weights and real SQL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

BACKEND = Path(__file__).resolve().parent


def child(args):
    import time
    import torch
    from idle_experiment import DEFAULT_ROOT, execute, fingerprint, make_cases, workload_pool, write_json
    from idle_neural import IdleNeuralHybrid
    torch.set_num_threads(1)
    if args.mode == "single-passive":
        torch.set_num_interop_threads(1)
    database, directory = DEFAULT_ROOT / "pagewise-demo.sqlite", DEFAULT_ROOT / "selected-hybrid"
    model = IdleNeuralHybrid.load(BACKEND / "runs/idle-neural-20261106/small-pilot-001/P4-neural-semantics",
        json.loads((directory / "page_map.json").read_text()), fingerprint(database))
    pool = workload_pool(database)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    if args.warm_p2:
        execute(database, directory, out / "p2-warmup", make_cases(pool, 6, 9501), policy="P2")
    trials = []
    for repeat in range(2):
        report, _, _ = execute(database, directory, out / f"p4-{repeat}", make_cases(pool, 24, 9501 + repeat),
            policy="P4-neural-semantics", query_model=model, gap_ms=50, delay_ms=.2, cache_pages=32)
        proposals = json.loads((out / f"p4-{repeat}/idle_proposals.json").read_text())
        report["neural_jobs"] = sum(p.get("neural_scored", False) for p in proposals)
        report["stage_totals_ms"] = {key: sum(p.get("stage_ns", {}).get(key, 0) for p in proposals) / 1e6
            for key in ("candidate_ns", "feature_ns", "forward_and_sort_ns")}
        trials.append(report)
    start, cpu = time.perf_counter_ns(), time.process_time_ns()
    time.sleep(.5)
    result = {"mode": args.mode, "warm_p2": args.warm_p2, "torch": torch.__version__,
        "intra_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_WAIT_POLICY", "KMP_BLOCKTIME")},
        "post_workload_sleep_wall_ms": (time.perf_counter_ns() - start)/1e6,
        "post_workload_sleep_cpu_ms": (time.process_time_ns() - cpu)/1e6, "trials": trials,
        "torch_build": torch.__config__.parallel_info()}
    write_json(out / "diagnostic.json", result)
    print(json.dumps({"mode": args.mode, "warm_p2": args.warm_p2,
        "trial_cpu_ms": [r["process_cpu_ms"] for r in trials],
        "trial_query_mean_ms": [r["mean_ms"] for r in trials],
        "stages_ms": [r["stage_totals_ms"] for r in trials], "sleep_cpu_ms": result["post_workload_sleep_cpu_ms"]}), flush=True)


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    commands = []
    for mode in ("default", "passive", "single-passive"):
        for warm in (False, True):
            env = dict(os.environ)
            if mode != "default":
                env.update(OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
            if mode == "single-passive":
                env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--mode", mode,
                       "--out", str(out / f"{mode}-warm{int(warm)}")]
            if warm: command.append("--warm-p2")
            completed = subprocess.run(command, cwd=BACKEND, env=env, text=True, capture_output=True)
            print(completed.stdout, end="", flush=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            commands.append({"command": command, "stdout": completed.stdout, "stderr": completed.stderr})
    (out / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--warm-p2", action="store_true")
    args = parser.parse_args()
    child(args) if args.child else run(args)
