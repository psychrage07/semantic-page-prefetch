"""Isolated real-APSW query-boundary experiments. Run from backend/ with --help."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time

import torch

from pagewise.metadata import read_page_size
from pagewise.runtime import PredictiveRuntime
from pagewise.tracing import TraceRecorder, TraceVFS, apsw, set_query_context, reset_query_context
from pagewise.workloads import generate_workload, explain_query_plan, classify_plan
from idle_markov import QueryMarkov
from idle_runtime import IdleRuntime, IdleStorageDevice, ProbeCache

BACKEND = Path(__file__).resolve().parent
DEFAULT_ROOT = BACKEND / "runs/local-lab/13d383a70d3c"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def percentile(values, fraction=0.95):
    return sorted(values)[max(0, min(len(values) - 1, int(len(values) * fraction + 0.999999) - 1))]


def fingerprint(path):
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(4096)
    return {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            "header_sha256": hashlib.sha256(header).hexdigest()}


def workload_pool(database, seed=1101):
    grouped = {}
    for case in generate_workload(database, count=240, seed=seed, mixed=False):
        group = grouped.setdefault(case.template_id, [])
        if len(group) < 4:
            group.append(case)
    return grouped


def make_cases(pool, count, seed, condition="repeated", prefix="Q"):
    # Explicit session pattern, not claimed to be an observed human workload.
    # Repeated conditions intentionally reuse familiar parameter values.
    pattern = ["customer_orders", "order_items_join", "product_sales", "customer_orders", "graph_walk", "order_items_join"]
    if condition == "novel":
        pattern = ["created_range", "customer_profile", "product_sales", "created_range", "graph_walk", "customer_profile"]
    cases = []
    for i in range(count):
        template = pattern[i % len(pattern)]
        variant = (seed + i // len(pattern)) % 4
        cases.append(replace(pool[template][variant], query_id=f"{prefix}{i:05d}"))
    return cases


class StageProbe:
    """Trial-local timings around the existing candidate/features/MLP path."""
    def __init__(self, runtime):
        self.runtime, self.timings = runtime, Counter()
        self.hooks = []

    def __enter__(self):
        import pagewise.ranking as ranking
        self.module = ranking
        self.original_features = ranking.group_features
        self.original_candidates = self.runtime.index.candidates
        def candidates(*args, **kwargs):
            start = time.perf_counter_ns()
            value = self.original_candidates(*args, **kwargs)
            self.timings["candidate_ns"] += time.perf_counter_ns() - start
            self.timings["candidate_calls"] += 1
            self.timings["candidates_proposed"] += len(value)
            return value
        def features(*args, **kwargs):
            start = time.perf_counter_ns()
            value = self.original_features(*args, **kwargs)
            self.timings["feature_ns"] += time.perf_counter_ns() - start
            self.timings["feature_calls"] += 1
            return value
        self.runtime.index.candidates = candidates
        ranking.group_features = features
        def before(*_):
            self.forward_started = time.perf_counter_ns()
        def after(*_):
            self.timings["neural_forward_ns"] += time.perf_counter_ns() - self.forward_started
            self.timings["neural_forward_calls"] += 1
        if self.runtime.ranker is not None:
            self.hooks = [self.runtime.ranker.model.register_forward_pre_hook(before),
                          self.runtime.ranker.model.register_forward_hook(after)]
        return self

    def __exit__(self, *_):
        self.runtime.index.candidates = self.original_candidates
        self.module.group_features = self.original_features
        for hook in self.hooks:
            hook.remove()


def execute(database, model_dir, out, cases, *, policy="P0", cache_pages=32, gap_ms=0,
            delay_ms=0, query_model=None, max_pages=16, min_confidence=0.25):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    page_size = read_page_size(database)
    device = IdleStorageDevice(delay_ms=delay_ms, enabled=delay_ms > 0)
    strategy = {"P0": "none", "P1": "markov2", "P2": "model_ranker"}.get(policy, "none")
    args = (database, page_size, model_dir / "page_map.json", model_dir / "candidate_index.json")
    kwargs = dict(strategy=strategy, model_path=model_dir / "model_ranker.pt" if policy == "P2" else None,
                  cache_pages=cache_pages, degree=1, confidence_threshold=0.7 if policy == "P2" else 0,
                  queue_size=64, predict_on="miss", predict_every=4 if policy == "P2" else 1,
                  storage_device=device, storage_model="shared", drop_stale_jobs=True,
                  miss_delay_ms=delay_ms,
                  diagnostics_enabled=True, record_demand_reference=True,
                  prediction_log_path=out / "predictions.jsonl")
    initialization_started, initialization_cpu = time.perf_counter_ns(), time.process_time_ns()
    runtime = (IdleRuntime(*args, query_model=query_model, max_pages=max_pages, min_confidence=min_confidence, **kwargs)
               if query_model else PredictiveRuntime(*args, **kwargs))
    if not query_model:
        runtime.cache = ProbeCache(cache_pages)
    initialization_ms = (time.perf_counter_ns() - initialization_started) / 1e6
    initialization_cpu_ms = (time.process_time_ns() - initialization_cpu) / 1e6
    recorder = TraceRecorder(out / "trace.jsonl", database, page_size)
    vfs = TraceVFS(recorder, runtime=runtime, storage_device=device)
    conn, rows, records = None, [], []
    stage_probe = StageProbe(runtime) if policy == "P2" else None
    if stage_probe:
        stage_probe.__enter__()
    workload_started = cpu_started = 0
    try:
        conn = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY, vfs=vfs.name)
        conn.execute("PRAGMA mmap_size=0")
        conn.execute("PRAGMA cache_size=-128")
        conn.execute("PRAGMA query_only=ON")
        plans = {case.template_id: explain_query_plan(database, case) for case in cases}
        write_json(out / "workload.json", [case.serializable() for case in cases])
        workload_started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
        previous_end = None
        for number, case in enumerate(cases):
            if previous_end is not None:
                # This is a controlled think-time condition, not a measured
                # claim about real users. The predictor never receives it.
                remaining = previous_end + int(gap_ms * 1e6) - time.perf_counter_ns()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
            arrival = time.perf_counter_ns()
            runtime.set_active_query(case.query_id)  # Cancel idle admission before SQL.
            token = set_query_context({"query_id": case.query_id, "template_id": case.template_id,
                "workload_family": case.family, "plan_class": classify_plan(plans[case.template_id]),
                "plan_details": plans[case.template_id], "params": list(case.params), "source": "demand"})
            digest, count = hashlib.sha256(), 0
            try:
                for row in conn.execute(case.sql, case.params):
                    count += 1
                    digest.update(json.dumps(row, default=str, separators=(",", ":")).encode())
                ended = time.perf_counter_ns()
            finally:
                reset_query_context(token)
                runtime.set_active_query(None)
            demanded = [entry[0] for entry in runtime._demand_reference.get(case.query_id, [])]
            row = {"query_id": case.query_id, "template_id": case.template_id, "rows": count,
                   "result_sha256": digest.hexdigest(), "arrival_ns": arrival, "end_ns": ended,
                   "elapsed_ms": (ended - arrival) / 1e6,
                   "gap_observed_ms": (arrival - previous_end) / 1e6 if previous_end is not None else None,
                   "configured_gap_ms": gap_ms,
                   "arrival_lateness_ms": max(0, (arrival - previous_end) / 1e6 - gap_ms) if previous_end is not None else 0}
            rows.append(row)
            records.append({"session_id": out.name, "query_id": case.query_id,
                            "template_id": case.template_id, "pages": demanded,
                            "family": case.family, "plan_class": classify_plan(plans[case.template_id]),
                            "miss_pages": [entry[0] for entry in runtime._demand_reference.get(case.query_id, []) if entry[1]]})
            previous_end = ended
            conn.execute("PRAGMA shrink_memory")
            if query_model:
                runtime.query_completed(case.query_id, case.template_id, demanded, final=number == len(cases) - 1)
        workload_ended, cpu_ended = time.perf_counter_ns(), time.process_time_ns()
    finally:
        if conn is not None:
            conn.close()
        drain_started, shutdown_cpu_started = time.perf_counter_ns(), time.process_time_ns()
        clean = runtime.close(timeout=10)
        shutdown_ms = (time.perf_counter_ns() - drain_started) / 1e6
        shutdown_cpu_ms = (time.process_time_ns() - shutdown_cpu_started) / 1e6
        recorder.close()
        if stage_probe:
            stage_probe.__exit__()
    stats = runtime.snapshot()
    if not clean or stats.get("prefetch_read_errors") or stats.get("worker_errors"):
        raise RuntimeError(f"Unclean or failed prefetch worker: {out}")
    write_json(out / "runtime_stats.json", stats)
    write_json(out / "query_records.json", records)
    if query_model:
        write_json(out / "idle_events.json", runtime.idle_events)
        write_json(out / "idle_proposals.json", runtime.proposals)
    write_json(out / "prefetch_uses.json", runtime.cache._demand_use_events)
    with (out / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    demand_bytes = 0
    for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("query_id"):
            demand_bytes += event["underlying_bytes"]
    gaps = [row["gap_observed_ms"] for row in rows if row["gap_observed_ms"] is not None]
    values = [row["elapsed_ms"] for row in rows]
    timely_uses = 0
    if query_model:
        arrivals = {row["query_id"]: row["arrival_ns"] for row in rows}
        insertions = {(e["page"], e["after_query"]): e for e in runtime.idle_events if e["event"] == "cache_inserted"}
        for use in runtime.cache._demand_use_events:
            inserted = insertions.get((use["page"], use["prefetch_query_id"]))
            if inserted and inserted["time_ns"] <= arrivals.get(use["demand_query_id"], -1):
                timely_uses += 1
    report = {"policy": policy, "queries": len(rows), "mean_ms": statistics.mean(values),
              "p95_ms": percentile(values), "workload_completion_ms": (workload_ended - workload_started) / 1e6,
              "process_cpu_ms": (cpu_ended - cpu_started) / 1e6, "initialization_ms": initialization_ms,
              "initialization_cpu_ms": initialization_cpu_ms, "shutdown_cpu_ms": shutdown_cpu_ms,
              "worker_shutdown_ms": shutdown_ms, "demand_read_bytes": demand_bytes,
              "useful_idle_pages_inserted_before_query_arrival": timely_uses if query_model else None,
              "cache_pages": cache_pages, "configured_gap_ms": gap_ms, "injected_delay_ms": delay_ms,
              "observed_gap_mean_ms": statistics.mean(gaps) if gaps else None,
              "gap_windows_at_least_ms": {str(k): sum(value >= k for value in gaps) for k in (25, 50, 100)},
              "stage_profile": dict(stage_probe.timings) if stage_probe else {}, "runtime": stats}
    write_json(out / "summary.json", report)
    return report, rows, records


def query_quality(model, records, max_pages, confidence):
    history, matched, targets, macro = [], 0, 0, []
    for row in records:
        if history:
            predicted = {item["page"] for item in model.predict(history, max_pages=max_pages, min_confidence=confidence)["pages"]}
            actual = set(row["pages"])
            hits = len(actual & predicted)
            matched, targets = matched + hits, targets + len(actual)
            if actual:
                macro.append(100 * hits / len(actual))
        history.append(model.state(row["template_id"], row["pages"]))
        history = history[-2:]
    return {"next_query_page_recall_micro_pct": 100 * matched / targets if targets else None,
            "next_query_page_recall_macro_pct": statistics.mean(macro) if macro else None,
            "next_query_unique_page_targets": targets,
            "scope": "Offline top-budget proposals on the same P0 demand trace; separate from actual admission and timeliness."}


def run(args):
    torch.set_num_threads(1)
    out, database, model_dir = args.out.resolve(), args.database.resolve(), args.model_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = fingerprint(database)
    from pagewise.runledger import source_hash
    source_before = source_hash()
    pool = workload_pool(database)
    training_records = []
    for seed in (1101, 1102, 1103):
        _, _, records = execute(database, model_dir, out / f"training-{seed}",
            make_cases(pool, 36, seed, prefix=f"T{seed}-"), policy="P0", cache_pages=32)
        training_records.extend(records)
    models = {}
    for order, signal in itertools.product((1, 2), ("template", "footprint")):
        model = QueryMarkov(order, signal).fit(training_records)
        key = f"P3-M{order}-{signal}"
        model.save(out / f"{key}.json")
        models[key] = model
    conditions = itertools.product((0, 50) if args.quick else (0, 25, 50, 100),
        (32,) if args.quick else (16, 128), (0, 0.2), ("repeated", "novel"))
    specs = list(conditions)
    protocol = {"schema_version": 1, "phase": "isolated idle-time Markov baseline",
        "database": str(database), "model_dir": str(model_dir), "database_fingerprint_before": before,
        "checkpoint_sha256": hashlib.sha256((model_dir / "model_ranker.pt").read_bytes()).hexdigest(),
        "source_sha256": source_before, "training_seeds": [1101, 1102, 1103], "evaluation_seed": args.seed,
        "extension_source_sha256": {name: hashlib.sha256((BACKEND / name).read_bytes()).hexdigest()
            for name in ("idle_markov.py", "idle_runtime.py", "idle_experiment.py")},
        "queries_per_trial": args.queries, "conditions": specs, "max_pages": args.max_pages,
        "min_training_page_frequency": args.confidence, "query_model_training_queries": len(training_records),
        "primary_idle_policy": "P3-M2-template", "other_idle_policies": "predeclared ablations, not selected using test outcomes",
        "policies": ["P0", "P1", "P2", *models],
        "P0": "SQLite with the same bounded Pagewise cache and prefetch disabled; isolates cache size from prefetching",
        "P1": "Existing within-query Markov-2, degree 1, every miss",
        "P2": "Existing matching neural-Markov checkpoint, degree 1, every fourth miss, threshold 0.7; no retraining",
        "P3": "Query-level Markov trained only on completed training queries; no next-query SQL, template, parameters or pages supplied online",
        "gaps": "Controlled completion-relative think times. Not evidence that these windows occur in a real user workload.",
        "cache": "128 KiB SQLite pager released between queries; Pagewise cache persists across queries; same capacity per condition",
        "storage": "No OS cache reset; nonzero delay is artificial, shared service, not physical disk latency",
        "admission": "At most 16 proposed pages by default, 25% unused speculative cache quota, protect pages demanded twice, expire unused idle pages after the next query",
        "cancellation": "No new idle reads admitted after arrival; actual OS read already admitted cannot be forcibly preempted",
        "uncertainty": "Single-seed exploratory pilot; no significance or general-speedup claim"}
    write_json(out / "protocol.json", protocol)
    results = []
    for condition_id, (gap, cache, delay, workload) in enumerate(specs):
        cases = make_cases(pool, args.queries, args.seed, workload, prefix="E")
        policies = ["P0", "P1", "P2", *models]
        random.Random(args.seed + condition_id).shuffle(policies)
        reference, condition_results = None, []
        expected = None
        for policy in policies:
            target = out / f"condition-{condition_id:02d}" / policy
            report, rows, records = execute(database, model_dir, target, cases, policy=policy,
                cache_pages=cache, gap_ms=gap, delay_ms=delay, query_model=models.get(policy),
                max_pages=args.max_pages, min_confidence=args.confidence)
            hashes = [(row["query_id"], row["rows"], row["result_sha256"]) for row in rows]
            if expected is not None and hashes != expected:
                raise ValueError("SQL result mismatch across policies")
            expected = hashes
            if policy == "P0":
                reference = records
            report.update({"condition_id": condition_id, "workload": workload, "seed": args.seed, "output": str(target)})
            condition_results.append(report)
            print(f"{condition_id + 1}/{len(specs)} {workload} gap={gap} cache={cache} delay={delay} {policy}: mean={report['mean_ms']:.3f}ms useful={report['runtime']['useful_prefetches']}", flush=True)
        for report in condition_results:
            if report["policy"] in models:
                report["prediction_quality"] = query_quality(models[report["policy"]], reference, args.max_pages, args.confidence)
        results.extend(condition_results)
        write_json(out / "results.json", {"protocol": protocol, "results": results})
    extension_after = {name: hashlib.sha256((BACKEND / name).read_bytes()).hexdigest()
                       for name in protocol["extension_source_sha256"]}
    if (before != fingerprint(database) or source_before != source_hash()
            or extension_after != protocol["extension_source_sha256"]
            or hashlib.sha256((model_dir / "model_ranker.pt").read_bytes()).hexdigest() != protocol["checkpoint_sha256"]):
        raise RuntimeError("Protected database or frozen source changed")
    write_json(out / "verification.json", {"database_unchanged": True, "frozen_source_unchanged": True,
        "sql_results_identical": True, "trials": len(results), "workers_clean": True,
        "extension_source_unchanged": True, "checkpoint_unchanged": True})
    print(f"Verified {len(results)} real trials; results: {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_ROOT / "pagewise-demo.sqlite")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ROOT / "selected-hybrid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=18)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--max-pages", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.queries < 2:
        parser.error("At least two queries are needed to evaluate query boundaries")
    run(args)
