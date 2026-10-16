from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

from .metadata import build_page_map, read_page_size
from .tracing import TraceRecorder, TraceVFS, apsw, reset_query_context, set_query_context
from .workloads import classify_plan, explain_query_plan, generate_workload
from .runtime import PredictiveRuntime, SimulatedStorageDevice


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, int((percentile / 100.0) * len(sorted_values) + 0.999999) - 1))
    return round(float(sorted_values[idx]), 6)


def trace_workload(database_path: str | Path, output_dir: str | Path, *, query_count: int = 500,
                   seed: int = 1234, cache_kib: int = 4096, mixed: bool = True,
                   prefetch_config: dict[str, Any] | None = None, miss_delay_ms: float = 0.0,
                   sparse_page_map: bool = False) -> dict[str, Any]:
    """Trace actual SQLite VFS read requests for a reproducible read-only workload."""
    if apsw is None:
        raise RuntimeError("APSW is not installed. Run: python -m pip install -r backend/requirements.txt")
    db_path = Path(database_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    page_size = read_page_size(db_path)
    if sparse_page_map and prefetch_config is not None:
        raise ValueError("sparse_page_map is only supported when collecting a training trace")
    trained_page_map = (Path(prefetch_config["candidate_index_path"]).parent / "page_map.json"
                        if prefetch_config is not None else None)
    if trained_page_map is not None and trained_page_map.exists():
        page_map = json.loads(trained_page_map.read_text(encoding="utf-8"))
    else:
        page_map = {} if sparse_page_map else build_page_map(db_path)
    if not sparse_page_map:
        (out / "page_map.json").write_text(json.dumps({str(k): v for k, v in page_map.items()}, indent=2), encoding="utf-8")
    cases = generate_workload(db_path, query_count, seed, mixed=mixed)

    # EXPLAIN is prepared with a separate standard connection before registering
    # the trace VFS, so plan exploration does not contaminate the xRead trace.
    plans: dict[str, list[str]] = {}
    for case in cases:
        if case.template_id not in plans:
            plans[case.template_id] = explain_query_plan(db_path, case)
    (out / "plans.json").write_text(json.dumps({k: {"details": v, "plan_class": classify_plan(v)} for k, v in plans.items()}, indent=2), encoding="utf-8")
    with (out / "workload.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.serializable()) + "\n")

    diagnostics_enabled = bool((prefetch_config or {}).get("diagnostics_enabled", True))
    recorder = TraceRecorder(out / "trace.jsonl", db_path, page_size, enabled=diagnostics_enabled)
    run_started_ns = time.perf_counter_ns()
    runtime = None
    storage_model = (prefetch_config or {}).get("storage_model", "independent")
    drop_stale_jobs = bool((prefetch_config or {}).get("drop_stale_jobs", False))
    oracle_trace_path = (prefetch_config or {}).get("oracle_trace_path")
    effective_delay = float((prefetch_config or {}).get("miss_delay_ms", miss_delay_ms) or 0.0)
    storage_device = SimulatedStorageDevice(effective_delay, enabled=(storage_model == "shared"))

    if prefetch_config is not None:
        runtime = PredictiveRuntime(
            db_path, page_size, out / "page_map.json", prefetch_config["candidate_index_path"],
            prefetch_config.get("model_path"), semantic=bool(prefetch_config.get("semantic", False)),
            strategy=prefetch_config.get("strategy", "none"), cache_pages=int(prefetch_config.get("cache_pages", 256)),
            degree=int(prefetch_config.get("degree", 2)), confidence_threshold=float(prefetch_config.get("confidence_threshold", 0.5)),
            queue_size=int(prefetch_config.get("queue_size", 64)),
            miss_delay_ms=float(prefetch_config.get("miss_delay_ms", 0.0)),
            predict_on=str(prefetch_config.get("predict_on", "all")),
            predict_every=int(prefetch_config.get("predict_every", 1)),
            max_score_candidates=int(prefetch_config.get("max_score_candidates", 0)),
            prediction_log_path=out / "predictions.jsonl",
            storage_model=storage_model,
            storage_device=storage_device,
            drop_stale_jobs=drop_stale_jobs,
            oracle_trace_path=oracle_trace_path,
            diagnostics_enabled=diagnostics_enabled,
            oracle_miss_path=prefetch_config.get("oracle_miss_path"),
            record_demand_reference=bool(prefetch_config.get("record_demand_reference", False)),
            prefetch_workers=int(prefetch_config.get("prefetch_workers", 1)),
        )
    # Stock/no-custom-cache mode can optionally inject one delay per main-file
    # xRead call. Cache-based modes inject the same configured delay per cache miss.
    stock_read_delay = miss_delay_ms if runtime is None else 0.0
    vfs = TraceVFS(recorder, runtime=runtime, read_delay_ms=stock_read_delay, storage_device=storage_device)
    query_results: list[dict[str, Any]] = []
    conn = None
    try:
        conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY, vfs=vfs.name)
        mmap_result = list(conn.execute("PRAGMA mmap_size=0"))
        cache_result = list(conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}"))
        conn.execute("PRAGMA query_only=ON")
        mmap_size_effective = int(mmap_result[0][0]) if mmap_result else None
        cache_size_effective = int(cache_result[0][0]) if cache_result else None
        if mmap_size_effective not in (0, None):
            raise RuntimeError(f"Could not disable SQLite memory-mapped I/O; effective mmap_size={mmap_size_effective}")
        for idx, case in enumerate(cases, start=1):
            plan_details = plans[case.template_id]
            context = {
                "query_id": case.query_id,
                "template_id": case.template_id,
                "workload_family": case.family,
                "plan_class": classify_plan(plan_details),
                "plan_details": plan_details,
                "source": "demand",
            }
            token = set_query_context(context)
            if runtime is not None:
                runtime.set_active_query(case.query_id)
            start = time.perf_counter_ns()
            row_count = 0
            first_rows: list[list[Any]] = []
            try:
                cursor = conn.cursor()
                cursor.execute(case.sql, case.params)
                for row in cursor:
                    row_count += 1
                    if len(first_rows) < 2:
                        first_rows.append(list(row))
                elapsed = (time.perf_counter_ns() - start) / 1e6
            finally:
                reset_query_context(token)
                if runtime is not None:
                    runtime.set_active_query(None)
            query_results.append({
                "query_id": case.query_id, "template_id": case.template_id,
                "workload_family": case.family, "elapsed_ms": round(elapsed, 6),
                "result_rows": row_count, "first_rows_preview": first_rows,
                "plan_class": classify_plan(plan_details), "seed": seed,
            })
            # Explicitly release SQLite's page cache between query executions for
            # a controlled experiment. This does not clear the OS page cache.
            conn.execute("PRAGMA shrink_memory")
            if idx % 100 == 0:
                print(f"Executed {idx}/{len(cases)} queries")
    finally:
        if conn is not None:
            conn.close()
        if runtime is not None:
            drain_started_ns = time.perf_counter_ns()
            idle_before_close = runtime.wait_idle(timeout=30.0)
            worker_drain_ms = (time.perf_counter_ns() - drain_started_ns) / 1e6
            shutdown_started_ns = time.perf_counter_ns()
            worker_shutdown_clean = runtime.close(timeout=30.0)
            worker_shutdown_ms = (time.perf_counter_ns() - shutdown_started_ns) / 1e6
            runtime_stats = runtime.snapshot()
            runtime_stats.update({
                "worker_idle_before_close": idle_before_close,
                "worker_drain_ms": round(worker_drain_ms, 3),
                "worker_shutdown_ms": round(worker_shutdown_ms, 3),
                "worker_shutdown_clean": worker_shutdown_clean,
            })
            (out / "runtime_stats.json").write_text(json.dumps(runtime_stats, indent=2), encoding="utf-8")
            if bool((prefetch_config or {}).get("record_demand_reference", False)):
                (out / "demand_reference.json").write_text(
                    json.dumps(runtime.demand_reference(), separators=(",", ":")), encoding="utf-8")
            if runtime.cache._demand_use_events:
                with (out / "prefetch_uses.jsonl").open("w", encoding="utf-8") as handle:
                    for ev in runtime.cache._demand_use_events:
                        handle.write(json.dumps(ev, ensure_ascii=False) + "\n")
        recorder.close()

    if sparse_page_map:
        observed_pages: set[int] = set()
        for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            first_page, last_page = row.get("first_page"), row.get("last_page")
            if isinstance(first_page, int) and isinstance(last_page, int):
                observed_pages.update(range(first_page, last_page + 1))
        page_map = build_page_map(db_path, observed_pages)
        (out / "page_map.json").write_text(
            json.dumps({str(k): v for k, v in page_map.items()}, indent=2), encoding="utf-8")

    result_csv = out / "query_results.csv"
    columns = ["query_id", "template_id", "workload_family", "elapsed_ms", "result_rows", "plan_class", "seed"]
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in query_results:
            writer.writerow({key: row[key] for key in columns})
    latencies = [float(row["elapsed_ms"]) for row in query_results]
    (out / "run_metadata.json").write_text(json.dumps({
        "database_path": str(db_path), "page_size": page_size,
        "query_count": len(cases), "seed": seed, "cache_kib_requested": cache_kib,
        "sqlite_cache_size_pragma": cache_size_effective, "mmap_size_effective_bytes": mmap_size_effective,
        "stdlib_sqlite_version": sqlite3.sqlite_version,
        "apsw_sqlite_version": apsw.sqlitelibversion() if apsw is not None else None,
        "page_map_pages": len(page_map),
        "page_map_scope": "observed trace pages" if sparse_page_map else "complete dbstat map",
        "trace_kind": "APSW VFS xRead requests; not a complete logical reference trace",
        "physical_io_note": "A VFS xRead can be satisfied by the OS page cache; this trace does not prove physical disk I/O.",
        "prefetch_config": prefetch_config,
        "storage_model": storage_model,
        "drop_stale_jobs": drop_stale_jobs,
        "oracle_trace_path": str(oracle_trace_path) if oracle_trace_path else None,
        "oracle_miss_path": (prefetch_config or {}).get("oracle_miss_path"),
        "record_demand_reference": bool((prefetch_config or {}).get("record_demand_reference", False)),
        "diagnostics_enabled": diagnostics_enabled,
        "miss_delay_ms_injected": (prefetch_config or {}).get("miss_delay_ms", miss_delay_ms),
        "mode": "latency-injected" if ((prefetch_config or {}).get("miss_delay_ms", miss_delay_ms) or 0) > 0 else "real-storage-no-injected-delay",
        "query_latency_ms": {
            "median": round(float(statistics.median(latencies)), 6) if latencies else None, "p95": _percentile(latencies, 95),
            "total": round(sum(latencies), 6), "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "end_to_end_wall_ms": round((time.perf_counter_ns() - run_started_ns) / 1e6, 3),
        "results": query_results,
    }, indent=2), encoding="utf-8")
    return {
        "database_path": str(db_path), "page_size": page_size, "page_map_pages": len(page_map),
        "query_count": len(cases), "trace_path": str(out / "trace.jsonl"),
        "results_path": str(result_csv), "output_dir": str(out),
        "runtime_stats_path": str(out / "runtime_stats.json") if prefetch_config is not None else None,
    }
