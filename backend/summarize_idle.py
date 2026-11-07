"""Audit and summarize isolated idle-prefetch trials without selecting policies."""
import argparse
import csv
import json
from pathlib import Path


def fixed(value):
    return "n/a" if value is None else f"{value:.3f}"


def summarize(paths, out):
    out.mkdir(parents=True, exist_ok=False)
    flat, studies = [], []
    for path in paths:
        from pagewise.metadata import read_page_size
        if (path / "EXCLUDED.json").exists():
            raise ValueError("Excluded attempt cannot be summarized")
        data = json.loads((path / "results.json").read_text(encoding="utf-8"))
        page_size = read_page_size(data["protocol"]["database"])
        verification = json.loads((path / "verification.json").read_text(encoding="utf-8"))
        if not all(verification.get(k) for k in ("database_unchanged", "frozen_source_unchanged", "workers_clean", "sql_results_identical")):
            raise ValueError("Failed experiment verification")
        studies.append((path, data))
        for result in data["results"]:
            runtime, output = result["runtime"], Path(result["output"])
            is_idle = result["policy"].startswith("P3")
            if result["injected_delay_ms"] > 0 and runtime["storage_device"]["demand_ios"] <= 0:
                raise ValueError("Delay did not reach demand reads")
            reads = runtime.get("idle", {}).get("prefetch_reads", 0)
            if is_idle:
                idle_open = False
                for event in json.loads((output / "idle_events.json").read_text(encoding="utf-8")):
                    if event["event"] == "idle_open": idle_open = True
                    elif event["event"] == "arrival": idle_open = False
                    elif event["event"] == "read_admitted" and not idle_open:
                        raise ValueError("Idle read admitted while foreground active")
            else:
                log = output / "predictions.jsonl"
                if log.exists():
                    for line in log.read_text(encoding="utf-8").splitlines():
                        event = json.loads(line)
                        reads += sum("page_read_done_ns" in item for item in event.get("selected", []))
            useful, prefetched = runtime["useful_prefetches"], runtime["pages_prefetched"]
            if useful + runtime["total_wasted_prefetches"] != prefetched:
                raise ValueError("Prefetch usefulness/waste denominator mismatch")
            quality = result.get("prediction_quality", {})
            stage = result.get("stage_profile", {})
            row = {"study": path.name, "condition_id": result["condition_id"], "policy": result["policy"],
                "workload": result["workload"], "cache_pages": result["cache_pages"],
                "gap_ms": result["configured_gap_ms"], "artificial_delay_ms": result["injected_delay_ms"],
                "queries": result["queries"], "mean_ms": result["mean_ms"], "p95_ms": result["p95_ms"],
                "workload_completion_ms": result["workload_completion_ms"], "cpu_workload_ms": result["process_cpu_ms"],
                "initialization_cpu_ms": result["initialization_cpu_ms"], "shutdown_cpu_ms": result["shutdown_cpu_ms"],
                "pages_inserted_by_prefetch": prefetched, "prefetch_file_reads": reads, "useful_pages": useful,
                "useful_pct": 100 * useful / prefetched if prefetched else None,
                "wasted_insertions": runtime["total_wasted_prefetches"], "wasted_reads": max(0, reads - useful),
                "pollution_misses": runtime.get("pollution_misses", 0),
                "demand_pages_displaced": runtime.get("prefetch_displaced_demand_pages", 0),
                "demand_vfs_bytes": result["demand_read_bytes"], "speculative_file_bytes": reads * page_size,
                "mean_observed_gap_ms": result["observed_gap_mean_ms"],
                "useful_pages_inserted_before_arrival": result.get("useful_idle_pages_inserted_before_query_arrival"),
                "query_set_recall_pct": quality.get("next_query_page_recall_micro_pct"),
                "candidate_ms_per_call": stage.get("candidate_ns", 0) / max(1, stage.get("candidate_calls", 0)) / 1e6 if stage else None,
                "feature_ms_per_call": stage.get("feature_ns", 0) / max(1, stage.get("feature_calls", 0)) / 1e6 if stage else None,
                "neural_forward_ms_per_call": stage.get("neural_forward_ns", 0) / max(1, stage.get("neural_forward_calls", 0)) / 1e6 if stage else None,
                "queue_wait_ms_per_job": runtime.get("idle", {}).get("idle_queue_wait_ns", 0) / max(1, runtime.get("idle", {}).get("prediction_jobs", 0)) / 1e6 if is_idle else runtime.get("queue_wait_ms_mean"),
                "output": str(output)}
            flat.append(row)
    with (out / "all_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    write("all_trials.json", flat)
    idle_rows = [row for row in flat if row["policy"].startswith("P3")]
    primary_rows = [row for row in flat if row["policy"] == "P3-M2-template"]
    slower = sum(row["mean_ms"] > next(control["mean_ms"] for control in flat
        if control["study"] == row["study"] and control["condition_id"] == row["condition_id"] and control["policy"] == "P0") for row in primary_rows)
    lines = ["# Idle-time Markov: controlled exploratory pilot", "",
        "This is a separate experiment. Existing models, source, databases and research exports were preserved.",
        "P0 is a cache-matched no-prefetch control; P1/P2 are the existing within-query predictors. P3 predicts the next query's page set.",
        "P3-M2-template was declared the primary idle baseline before evaluation; the other Markov configurations are ablations.",
        "Each study has one workload seed and short trials. Differences are observations, not statistical or physical-device speedup claims.",
        "P4 (a separately trained query-level neural residual) has not been implemented or evaluated in this phase.", "",
        f"Across all {len(flat)} accepted trials, idle policies inserted {sum(r['pages_inserted_by_prefetch'] for r in idle_rows)} pages; {sum(r['useful_pages'] for r in idle_rows)} were useful and {sum(r['useful_pages_inserted_before_arrival'] for r in idle_rows)} useful insertions completed before the next query arrived.",
        f"Requested zero-gap idle trials produced {sum(r['useful_pages'] for r in idle_rows if r['gap_ms'] == 0)} useful pages. The primary P3 mean was higher than P0 in {slower} of {len(primary_rows)} conditions. Across all idle policies, the shadow-cache probe recorded {sum(r['pollution_misses'] for r in idle_rows)} pollution misses. These outcomes must remain visible alongside favorable observations.", "",
        "## Predeclared focus: 50 ms gap and 0.2 ms artificial delay", "",
        "All other conditions and all ablations remain in all_trials.csv. None are selected or hidden according to whether they win.", "",
        "| Study / workload / cache | P0 mean ms | P1 mean ms | P2 mean ms | P3 mean ms | P3 p95 ms | P3 useful / inserted | P3 pollution misses | P3 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path, data in studies:
        grouped = {}
        for row in flat:
            if row["study"] == path.name and row["gap_ms"] == 50 and row["artificial_delay_ms"] == .2:
                grouped.setdefault((row["workload"], row["cache_pages"]), {})[row["policy"]] = row
        for (workload, cache), policies in sorted(grouped.items()):
            p = policies["P3-M2-template"]
            lines.append(f"| {path.name} / {workload} / {cache} | " + " | ".join(fixed(policies[k]["mean_ms"]) for k in ("P0", "P1", "P2", "P3-M2-template")) +
                         f" | {fixed(p['p95_ms'])} | {p['useful_pages']} / {p['pages_inserted_by_prefetch']} | {p['pollution_misses']} | {fixed(p['query_set_recall_pct'])}% |")
    lines.extend(["", "## Boundaries and interpretation", "",
        "- Gaps are controlled completion-relative think times, not naturally observed human idle windows. Actual arrival/end timestamps and scheduling lateness are saved per query.",
        "- Query-level training uses completed training queries only. No future SQL/template/parameters/pages are supplied to the online predictor. The repeated condition deliberately reuses familiar parameters.",
        "- Novel templates are absent from query-level Markov training, but may have appeared in the older frozen P2 training corpus. This is not a universal unseen-template test.",
        "- Recall is offline page-set proposal recall on the common P0 VFS stream, with the configured page budget. It is separate from cache admission, usefulness and timeliness.",
        "- The cache persists across queries; the SQLite pager is released between queries. Prefetch quota is 25% of cache capacity; pages demanded at least twice are protected from speculative replacement.",
        "- Old unused idle insertions expire after the next query. Lower-ranked proposals cannot churn newly prefetched pages. Actual waste includes unused evictions and unused resident insertions.",
        "- Pollution is a measured shadow-LRU miss proxy on the same observed request stream, not a claim that every differing miss is causally attributable to one eviction.",
        "- Idle read admission stops at query arrival. Cancellable artificial service yields to demand. A real OS read already admitted cannot be forcibly preempted from Python.",
        "- OS cache is not reset. VFS bytes and software service delays are not physical disk I/O measurements. Stage profiling adds diagnostic overhead.",
        "- Workload completion includes think time. CPU fields separate query-workload time, initialization, and shutdown. Short p95 estimates and one-seed results need replication.", ""])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audited {len(flat)} trials; {out / 'RESULTS.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.studies, args.out)
