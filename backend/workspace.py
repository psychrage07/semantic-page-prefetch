"""Interactive SQL adapter for the existing Pagewise VFS, models and runtime.

All new outputs live in unique local-lab run directories. Database connections
are read-only; the frozen predictor source and research artifacts are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
import threading
import time

BACKEND = Path(__file__).resolve().parent
LAB = BACKEND / "runs/local-lab"
LABELS = {"none": "No prefetch", "sequential": "Sequential", "markov1": "Markov-1", "markov2": "Markov-2",
          "model_a": "Model A · history", "model_b": "Model B · semantics", "hybrid": "Standard hybrid",
          "model_ranker": "Selected neural hybrid"}


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    # Windows readers briefly deny replacement while a polling request holds
    # the old file open. Keep atomic publication and retry that short sharing
    # conflict instead of failing a successfully executed query.
    for attempt in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.01)


def resolve_database(identifier: str) -> dict:
    if identifier == "research":
        from live_hybrid import DATABASE, FROZEN, assets
        evidence = assets()
        return {"id": identifier, "scope": "research", "label": "8.62 GB research database", "db": DATABASE,
                "models": FROZEN / "params", "page_map": FROZEN / "params/page_map.json", "evidence": evidence,
                "policies": ["none", "sequential", "markov1", "markov2", "model_ranker"]}
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{12}", identifier):
        raise ValueError("Choose a generated database from the workspace catalog")
    root = (LAB / identifier).resolve()
    if LAB.resolve() not in root.parents:
        raise ValueError("Database must stay inside the local demo directory")
    db, models, page_map = root / "pagewise-demo.sqlite", root / "models", root / "trace/page_map.json"
    if not all(path.is_file() for path in [db, page_map, models / "candidate_index.json", models / "model_b.pt"]):
        raise ValueError("Complete a small demo build before querying its database with trained models")
    return {"id": identifier, "scope": "small", "label": f"Small demo · {identifier}", "db": db, "models": models,
            "page_map": page_map, "policies": ["none", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid"]}


def catalog() -> dict:
    from pagewise.workloads import TEMPLATES
    from live_hybrid import DATABASE, FROZEN
    databases = []
    for path in sorted(LAB.glob("*/pagewise-demo.sqlite"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            item = resolve_database(path.parent.name)
        except ValueError:
            continue
        databases.append({key: item[key] for key in ("id", "label", "scope", "policies")} | {"bytes": path.stat().st_size})
    if DATABASE.is_file() and (FROZEN / "params/model_ranker.pt").is_file():
        databases.append({"id": "research", "label": "8.62 GB research study · existing frozen model", "scope": "research",
                          "bytes": DATABASE.stat().st_size, "policies": ["none", "sequential", "markov1", "markov2", "model_ranker"]})
    params = {"customer_orders": [1, 25], "order_items_join": [1, 50], "product_sales": [1, 100],
              "created_range": [0, 2147483647, 100], "graph_walk": [1, 3, 100], "customer_profile": [1]}
    return {"databases": databases, "policies": LABELS,
            "templates": [{"id": name, **spec, "params": params[name]} for name, spec in TEMPLATES.items()]}


def cell(value):
    if isinstance(value, bytes):
        return {"blob_bytes": len(value), "hex_preview": value[:64].hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"sqlite_real": str(value)}
    return value


def preview_cell(value):
    if isinstance(value, str) and len(value) > 500:
        return {"text_chars": len(value), "text_preview": value[:500]}
    return cell(value)


def schema(identifier: str, table: str | None = None) -> dict:
    item = resolve_database(identifier)
    conn = sqlite3.connect(f"file:{item['db'].as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        tables = []
        for name, kind, sql in conn.execute("SELECT name,type,sql FROM sqlite_schema WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [{"name": row[1], "type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
                       for row in conn.execute(f"PRAGMA table_info({quoted})")]
            indexes = [{"name": row[1], "unique": bool(row[2])} for row in conn.execute(f"PRAGMA index_list({quoted})")]
            tables.append({"name": name, "type": kind, "sql": sql, "columns": columns, "indexes": indexes})
        result = {"database": identifier, "scope": item["scope"], "path": str(item["db"].relative_to(BACKEND)),
                  "bytes": item["db"].stat().st_size, "page_size": conn.execute("PRAGMA page_size").fetchone()[0],
                  "page_count": conn.execute("PRAGMA page_count").fetchone()[0], "tables": tables}
        if table is not None:
            if table not in {row["name"] for row in tables}:
                raise ValueError("Unknown table in this database")
            quoted = '"' + table.replace('"', '""') + '"'
            cursor = conn.execute(f"SELECT * FROM {quoted} LIMIT 25")
            result["preview"] = {"table": table, "columns": [col[0] for col in cursor.description],
                                 "rows": [[preview_cell(value) for value in row] for row in cursor]}
        return result
    finally:
        conn.close()


def authorizer(action, arg1, arg2, database, trigger):
    import apsw
    allowed = {apsw.SQLITE_SELECT, apsw.SQLITE_READ, apsw.SQLITE_FUNCTION, apsw.SQLITE_RECURSIVE}
    if action == apsw.SQLITE_FUNCTION and str(arg2 or arg1).lower() in {"load_extension", "readfile", "writefile"}:
        return apsw.SQLITE_DENY
    return apsw.SQLITE_OK if action in allowed else apsw.SQLITE_DENY


def check_query(conn, sql: str, params: list) -> list[str]:
    import apsw.ext
    conn.set_authorizer(authorizer)
    info = apsw.ext.query_info(conn, sql, params)
    if info.query_remaining and info.query_remaining.strip():
        raise ValueError("Run one SQL statement at a time")
    if not info.is_readonly or not info.description:
        raise ValueError("The SQL editor accepts read-only queries that return rows")
    return [column[0] for column in info.description]


def validate_request(payload: dict) -> dict:
    item = resolve_database(payload.get("database", ""))
    sql = payload.get("sql", "")
    params = payload.get("params", [])
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 20000:
        raise ValueError("Enter a SQL query of up to 20,000 characters")
    if not isinstance(params, list) or len(params) > 100 or any(not isinstance(v, (str, int, float, type(None))) for v in params):
        raise ValueError("SQL parameters must be a JSON array of scalar values")
    repeats, limit = int(payload.get("repeats", 1)), int(payload.get("row_limit", 1000))
    if not 1 <= repeats <= 20 or not 1 <= limit <= 5000:
        raise ValueError("Use 1–20 repetitions and a result limit of 1–5000 rows")
    policies = payload.get("policies", ["model_ranker" if item["scope"] == "research" else "model_b"])
    if not isinstance(policies, list) or not policies or len(policies) != len(set(policies)) or any(policy not in item["policies"] for policy in policies):
        raise ValueError("Choose supported policies for the selected database/model")
    import apsw
    conn = apsw.Connection(str(item["db"]), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        check_query(conn, sql, params)
    except apsw.Error as exc:
        raise ValueError(f"SQL could not be prepared: {exc}") from exc
    finally:
        conn.close()
    return {"database": item["id"], "scope": item["scope"], "sql": sql, "params": params, "repeats": repeats,
            "row_limit": limit, "policies": policies, "order_seed": int(payload.get("order_seed", time.time_ns() % 2147483647))}


def tail_jsonl(path: Path, limit: int = 500) -> list:
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        offset = max(0, path.stat().st_size - 512000)
        handle.seek(offset)
        if offset:
            handle.readline()
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    result = []
    for line in lines[-limit:]:
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # A concurrent writer may not yet have finished its final line.
    return result


def run_view(root: Path, policy: str | None = None) -> dict:
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    policy = policy or state.get("policy")
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    if policy is not None and policy not in request["policies"]:
        raise ValueError("Policy is not part of this run")
    directory = root / policy if policy else root
    result_path = root / "workspace-result.json"
    artifacts = [{"name": str(path.relative_to(root)).replace('\\', '/'), "path": str(path.relative_to(BACKEND)), "bytes": path.stat().st_size}
                 for path in sorted(root.rglob("*")) if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}]
    return {"state": state, "policy": policy, "request": request,
            "trace": tail_jsonl(directory / "trace.jsonl"), "predictions": tail_jsonl(directory / "predictions.jsonl", 50),
            "uses": tail_jsonl(directory / "prefetch_uses.jsonl", 50), "artifacts": artifacts,
            "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None}


def execute_policy(item: dict, request: dict, out: Path, policy: str, state):
    import apsw
    from pagewise.metadata import read_page_size
    from pagewise.runtime import PredictiveRuntime
    from pagewise.tracing import TraceRecorder, TraceVFS, set_query_context, reset_query_context
    from pagewise.workloads import classify_plan, TEMPLATES

    directory = out / policy
    directory.mkdir(exist_ok=False)
    page_size = read_page_size(item["db"])
    model_dir = out / "model"
    model = model_dir / ("model_a.pt" if policy == "model_a" else "model_b.pt")
    if item["scope"] == "research":
        model = item["models"] / "model_ranker.pt"
    settings = {"cache_pages": 512 if item["scope"] == "research" else 128, "degree": 1 if item["scope"] == "research" else 2,
                "confidence_threshold": 0.0, "predict_on": "miss", "predict_every": 1}
    if policy == "model_ranker":
        settings.update(item["evidence"]["settings"])
    runtime = PredictiveRuntime(item["db"], page_size, model_dir / "page_map.json", model_dir / "candidate_index.json",
        model if policy in {"model_a", "model_b", "hybrid", "model_ranker"} else None,
        semantic=policy in {"model_b", "hybrid"}, strategy=policy, prediction_log_path=directory / "predictions.jsonl",
        storage_model="shared", drop_stale_jobs=True, diagnostics_enabled=True, prefetch_workers=1, **settings)
    recorder = TraceRecorder(directory / "trace.jsonl", item["db"], page_size)
    vfs = TraceVFS(recorder, runtime=runtime)
    conn = None
    stop = threading.Event()
    error = None
    measurements = []
    preview, columns = [], []
    def monitor():
        while not stop.wait(0.25):
            with recorder._lock:
                if recorder._file and not recorder._closed:
                    recorder._file.flush()
            state.update({"stats": runtime.snapshot(), "trace_events": recorder._seq, "snapshot_at": time.time()})
            write_json(out / "state.json", dict(state))
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        conn = apsw.Connection(str(item["db"]), flags=apsw.SQLITE_OPEN_READONLY, vfs=vfs.name)
        conn.execute("PRAGMA mmap_size=0")
        conn.execute("PRAGMA cache_size=-128")
        conn.execute("PRAGMA query_only=ON")
        conn.limit(apsw.SQLITE_LIMIT_LENGTH, 1_000_000)
        columns = check_query(conn, request["sql"], request["params"])
        plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + request["sql"], request["params"])]
        template = next((name for name, spec in TEMPLATES.items() if spec["sql"].strip() == request["sql"].strip()), "interactive_sql")
        context = {"template_id": template, "workload_family": TEMPLATES.get(template, {}).get("family", "mixed"),
                   "plan_class": classify_plan(plan), "plan_details": plan, "params": request["params"], "source": "demand"}
        write_json(directory / "plan.json", {**context, "sql": request["sql"]})
        print(f"[{policy}] EXPLAIN QUERY PLAN: " + " | ".join(plan), flush=True)
        for repeat in range(request["repeats"]):
            query_id = f"SQL{repeat + 1:04d}"
            token = set_query_context({**context, "query_id": query_id})
            runtime.set_active_query(query_id)
            started = time.perf_counter()
            conn.set_progress_handler(lambda: time.perf_counter() - started > 10, 1000)
            rows = []
            digest = hashlib.sha256()
            truncated = False
            result_bytes = 0
            cursor = conn.cursor()
            try:
                for row in cursor.execute(request["sql"], request["params"]):
                    if len(rows) == request["row_limit"]:
                        truncated = True
                        break
                    converted = [cell(value) for value in row]
                    result_bytes += sum(len(value) if isinstance(value, bytes) else len(str(value).encode("utf-8"))
                                        for value in row)
                    if result_bytes > 8_000_000:
                        truncated = True
                        break
                    canonical = [{"blob_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} if isinstance(value, bytes) else cell(value) for value in row]
                    digest.update(json.dumps(canonical, ensure_ascii=False, default=str).encode("utf-8"))
                    rows.append(converted)
                elapsed = (time.perf_counter() - started) * 1000
            finally:
                cursor.close()
                conn.set_progress_handler(None)
                reset_query_context(token)
                runtime.set_active_query(None)
            if repeat == 0:
                preview = [[preview_cell(value) for value in row] for row in rows[:100]]
                print(f"[{policy}] Columns: {columns}; first rows: {json.dumps(rows[:3], ensure_ascii=False, default=str)[:1500]}", flush=True)
                with (directory / "query-results.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle); writer.writerow(columns); writer.writerows(rows)
            measurement = {"query_id": query_id, "elapsed_ms": elapsed, "rows": len(rows), "truncated": truncated,
                           "result_sha256": digest.hexdigest()}
            measurements.append(measurement)
            state.update({"repeat": repeat + 1, "elapsed_ms": elapsed, "rows": len(rows)})
            print(f"[{policy}] {query_id}: {len(rows)} rows{' (capped)' if truncated else ''}, {elapsed:.3f} ms", flush=True)
            conn.set_authorizer(None)
            conn.execute("PRAGMA shrink_memory")
            conn.set_authorizer(authorizer)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if conn is not None:
            conn.close()
        idle = runtime.wait_idle(timeout=10)
        clean = runtime.close(timeout=10)
        stop.set(); worker.join(timeout=2)
        recorder.close()
        vfs.unregister()
        stats = runtime.snapshot()
        stats.update({"worker_idle_before_close": idle, "worker_shutdown_clean": clean})
        write_json(directory / "runtime_stats.json", stats)
        with (directory / "prefetch_uses.jsonl").open("w", encoding="utf-8") as handle:
            for event in runtime.cache._demand_use_events:
                handle.write(json.dumps(event) + "\n")
        state.update({"stats": stats, "trace_events": recorder._seq, "error": error, "snapshot_at": time.time()})
        write_json(out / "state.json", dict(state))
    if stats.get("prefetch_read_errors") or not clean:
        raise RuntimeError("Runtime reported a prefetch error or unclean worker shutdown; inspect artifacts")
    print(f"[{policy}] Runtime: scored={stats['completed_jobs']}, prefetched={stats['pages_prefetched']}, "
          f"useful={stats['useful_prefetches']}, cache_hits={stats['cache_hits']}, "
          f"cache_misses={stats['cache_misses']}, clean_shutdown={clean}", flush=True)
    with (directory / "query-timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0])); writer.writeheader(); writer.writerows(measurements)
    result = {"policy": policy, "label": LABELS[policy], "columns": columns, "preview": preview, "measurements": measurements,
              "mean_ms": statistics.mean(row["elapsed_ms"] for row in measurements),
              "median_ms": statistics.median(row["elapsed_ms"] for row in measurements), "runtime": stats, "settings": settings,
              "plan": plan, "template": template, "family": context["workload_family"]}
    write_json(directory / "query-result.json", result)
    return result


def run(out: Path):
    import shutil
    import torch
    torch.set_num_threads(1)
    request = json.loads((out / "request.json").read_text(encoding="utf-8"))
    item = resolve_database(request["database"])
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=False)
    index = json.loads((item["models"] / "candidate_index.json").read_text(encoding="utf-8"))
    index["layout_database_path"] = str(item["db"])
    if item["scope"] == "research":
        from live_hybrid import FROZEN
        index["static_metadata_path"] = str(FROZEN / "static-metadata.npz")
    write_json(model_dir / "candidate_index.json", index)
    shutil.copyfile(item["page_map"], model_dir / "page_map.json")
    if item["scope"] == "small":
        for name in ["model_a.pt", "model_b.pt"]:
            shutil.copyfile(item["models"] / name, model_dir / name)
    before = {"bytes": item["db"].stat().st_size, "mtime_ns": item["db"].stat().st_mtime_ns}
    order = list(request["policies"])
    random.Random(request["order_seed"]).shuffle(order)
    state = {"status": "running", "database": request["database"], "scope": item["scope"], "order": order}
    results = []
    print(f"Database: {item['db']} ({before['bytes']} bytes) · scope={item['scope']}", flush=True)
    print(f"Existing Pagewise TraceVFS + PredictiveRuntime; real APSW SQL; order={order}", flush=True)
    print(f"SQL: {request['sql']}\nParameters: {request['params']}", flush=True)
    try:
        for position, policy in enumerate(order):
            state.update({"policy": policy, "policy_number": position + 1, "repeat": 0, "stats": {}, "trace_events": 0})
            write_json(out / "state.json", state)
            print(f"Starting {LABELS[policy]} ({position + 1}/{len(order)})", flush=True)
            results.append(execute_policy(item, request, out, policy, state))
        after = {"bytes": item["db"].stat().st_size, "mtime_ns": item["db"].stat().st_mtime_ns}
        if before != after:
            raise RuntimeError("Database identity changed during this read-only run")
        state["phase"] = "evaluating_accuracy"
        write_json(out / "state.json", state)
        from workspace_evaluation import evaluate
        evaluation = evaluate(out, request, item, order)
        hashes = {row["result_sha256"] for result in results for row in result["measurements"]}
        sql_ordered = bool(re.search(r"\border\s+by\b", request["sql"], re.I))
        # SQLite does not promise row order without ORDER BY; disclose rather
        # than claim mismatches are necessarily wrong query results.
        result = {"artifact_type": "pagewise_workspace", "schema_version": 1, "request": request, "database": before,
                  "database_unchanged": True, "results": results, "order": order, "evaluation": evaluation,
                  "identical_row_order_and_values": len(hashes) == 1, "has_order_by": sql_ordered,
                  "checkpoint_sha256": item.get("evidence", {}).get("checkpoint_sha256"),
                  "notes": ["VFS xRead events are not proof of physical disk I/O.",
                            "No injected delay; OS cache is not reset. Diagnostic monitoring adds overhead.",
                            "Each policy uses a fresh Pagewise cache; SQLite cache is released between repetitions.",
                            "Timings cover execution and row fetching up to the configured result cap; model loading and worker drain are separate.",
                            "Custom SQL is not a held-out accuracy evaluation. Model scores rank candidates, not correctness percentages."]}
        write_json(out / "workspace-result.json", result)
        with (out / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["policy", "repetitions", "mean_ms", "median_ms", "scope", "database"])
            for row in results:
                writer.writerow([row["policy"], len(row["measurements"]), row["mean_ms"], row["median_ms"], item["scope"], item["id"]])
        print(f"Saved measured SQL rows, timings, VFS traces, predictions, runtime counters and benchmark CSV/JSON in {out}", flush=True)
        state["status"] = "completed"
    except Exception as exc:
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        write_json(out / "state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
