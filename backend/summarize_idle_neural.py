"""Audit all P4 trials and report the predeclared repeated-seed comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from pagewise.metadata import read_page_size


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(studies, out):
    out.mkdir(parents=True, exist_ok=False)
    rows, protocols, paired = [], {}, []
    for study in studies:
        if (study / "EXCLUDED.json").exists():
            raise ValueError("Excluded study")
        data, verified = read(study / "results.json"), read(study / "verification.json")
        if any(value is not True for key, value in verified.items() if key != "trials"):
            raise ValueError("Failed study verification")
        protocols[study.name] = data["protocol"]
        page_size = read_page_size(data["protocol"]["database"])
        for result in data["results"]:
            runtime, output = result["runtime"], Path(result["output"])
            idle = result["policy"].startswith(("P3", "P4"))
            if runtime["worker_alive"] or runtime.get("worker_errors") or runtime["prefetch_read_errors"]:
                raise ValueError("Worker failed or remained alive")
            if result["injected_delay_ms"] and not runtime["storage_device"]["demand_ios"]:
                raise ValueError("Artificial service omitted from demand reads")
            useful, inserted = runtime["useful_prefetches"], runtime["pages_prefetched"]
            if useful + runtime["total_wasted_prefetches"] != inserted:
                raise ValueError("Cache accounting mismatch")
            proposals, reads = [], 0
            if idle:
                proposals = read(output / "idle_proposals.json")
                active = False
                for event in read(output / "idle_events.json"):
                    if event["event"] == "idle_open": active = True
                    elif event["event"] == "arrival": active = False
                    elif event["event"] in ("read_admitted", "cache_inserted") and not active:
                        raise ValueError("Speculative read/admission outside idle period")
                reads = runtime.get("idle", {}).get("prefetch_reads", 0)
                if useful != result["useful_idle_pages_inserted_before_query_arrival"]:
                    raise ValueError("An allegedly useful idle page arrived late")
            elif (output / "predictions.jsonl").exists():
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                    reads += sum("page_read_done_ns" in item for item in json.loads(line).get("selected", []))
            scored = [p for p in proposals if p.get("neural_scored")]
            stage = {key: sum(p.get("stage_ns", {}).get(key, 0) for p in scored) / max(1, len(scored)) / 1e6
                     for key in ("candidate_ns", "feature_ns", "forward_and_sort_ns")}
            row = {
                "study": study.name, "condition_id": result["condition_id"], "seed": result["seed"],
                "policy": result["policy"], "workload": result["workload"], "cache_pages": result["cache_pages"],
                "gap_ms": result["configured_gap_ms"], "artificial_delay_ms": result["injected_delay_ms"],
                "queries": result["queries"], "mean_ms": result["mean_ms"], "p95_ms": result["p95_ms"],
                "workload_completion_ms": result["workload_completion_ms"], "cpu_workload_ms": result["process_cpu_ms"],
                "initialization_cpu_ms": result["initialization_cpu_ms"], "shutdown_cpu_ms": result["shutdown_cpu_ms"],
                "inserted": inserted, "file_reads": reads, "useful": useful,
                "useful_pct": 100 * useful / inserted if inserted else None,
                "wasted_insertions": runtime["total_wasted_prefetches"], "unused_reads": max(0, reads - useful),
                "pollution_misses": runtime.get("pollution_misses", 0),
                "demand_pages_displaced": runtime.get("prefetch_displaced_demand_pages", 0),
                "timely_useful": result["useful_idle_pages_inserted_before_query_arrival"],
                "demand_vfs_bytes": result["demand_read_bytes"], "prefetch_file_bytes": reads * page_size,
                "page_set_proposal_recall_pct": result.get("prediction_quality", {}).get("next_query_page_recall_micro_pct"),
                "idle_prediction_jobs": len(proposals), "neural_scored_jobs": len(scored),
                "candidate_ms_per_neural_job": stage["candidate_ns"] if scored else None,
                "feature_ms_per_neural_job": stage["feature_ns"] if scored else None,
                "forward_sort_ms_per_neural_job": stage["forward_and_sort_ns"] if scored else None,
                "observed_torch_threads": ",".join(map(str, sorted({p["torch_threads_observed"] for p in scored}))),
                "queue_ms_per_idle_job": runtime.get("idle", {}).get("idle_queue_wait_ns", 0) / max(1, len(proposals)) / 1e6 if idle else None,
                "output": str(output),
            }
            rows.append(row)
        # Pair every evaluation query, retaining each run as a separate unit.
        baseline = {r["condition_id"]: r for r in rows if r["study"] == study.name and r["policy"] == "P0"}
        for row in [r for r in rows if r["study"] == study.name and r["policy"] != "P0"]:
            with (Path(row["output"]) / "queries.csv").open(newline="", encoding="utf-8") as handle:
                queries = list(csv.DictReader(handle))
            with (Path(baseline[row["condition_id"]]["output"]) / "queries.csv").open(newline="", encoding="utf-8") as handle:
                control = list(csv.DictReader(handle))
            for q, c in zip(queries, control, strict=True):
                if (q["query_id"], q["rows"], q["result_sha256"]) != (c["query_id"], c["rows"], c["result_sha256"]):
                    raise ValueError("SQL values do not match")
                paired.append({"study": study.name, "condition_id": row["condition_id"], "seed": row["seed"],
                               "policy": row["policy"], "query_id": q["query_id"],
                               "delta_vs_P0_ms": float(q["elapsed_ms"]) - float(c["elapsed_ms"])})
    def write_json(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    def write_csv(name, obj):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(obj[0])); writer.writeheader(); writer.writerows(obj)
    write_json("all_trials.json", rows); write_csv("all_trials.csv", rows)
    write_csv("paired_queries.csv", paired)
    primary = [r for r in rows if r["seed"] in (9001, 9002, 9003)]
    aggregate = []
    lines = ["# P4: idle semantic neural-Markov pilot", "",
        "P4 is a new query-level adaptation of the existing B2 semantic feature pipeline and width-64 MLP, plus an empirical query-level Markov prior. It uses separate newly trained weights. The original selected checkpoint, database and core sources were preserved.", "",
        f"{len(rows)} matched trials and {sum(r['queries'] for r in rows)} evaluation query executions. All policies and conditions remain in all_trials.csv; paired per-query differences remain in paired_queries.csv.", "",
        "## Predeclared replication condition", "",
        "Familiar query values, 50 ms controlled idle gaps, 0.2 ms artificial shared storage service, 32-page Pagewise cache. Three seeds (9001-9003) select familiar parameter variants. Means below average the three run means; p95 averages three short-run p95 estimates. Useful/inserted counts are pooled. These are small controlled pilots, not proof of physical-device or general speedup.", "",
        "| Database | Policy | Mean ms | Mean of run p95 ms | Workload wall ms | CPU ms | Useful / inserted | Useful % | Pollution misses | Neural jobs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for study in protocols:
        for policy in protocols[study]["policies"]:
            selected = [r for r in primary if r["study"] == study and r["policy"] == policy]
            useful, inserted = sum(r["useful"] for r in selected), sum(r["inserted"] for r in selected)
            item = {"study": study, "policy": policy, "seeds": [r["seed"] for r in selected],
                "mean_ms": statistics.mean(r["mean_ms"] for r in selected),
                "p95_run_average_ms": statistics.mean(r["p95_ms"] for r in selected),
                "workload_wall_ms": statistics.mean(r["workload_completion_ms"] for r in selected),
                "cpu_ms": statistics.mean(r["cpu_workload_ms"] for r in selected),
                "run_means_ms": [r["mean_ms"] for r in selected], "useful": useful, "inserted": inserted,
                "useful_pct": 100 * useful / inserted if inserted else None,
                "pollution_misses": sum(r["pollution_misses"] for r in selected),
                "neural_jobs": sum(r["neural_scored_jobs"] for r in selected),
                "timely_useful": sum(r["timely_useful"] or 0 for r in selected)}
            aggregate.append(item)
            pct = f"{item['useful_pct']:.1f}%" if inserted else "n/a"
            lines.append(f"| {study} | {policy} | {item['mean_ms']:.3f} | {item['p95_run_average_ms']:.3f} | {item['workload_wall_ms']:.2f} | {item['cpu_ms']:.2f} | {useful}/{inserted} | {pct} | {item['pollution_misses']} | {item['neural_jobs']} |")
    write_json("primary_comparison.json", aggregate)
    lines += ["", "## Every condition, including unsuccessful cases", "",
        "| Database | Seed | Workload | Gap ms | Artificial delay ms | P4 mean ms | Change vs P0 | Change vs P2 | Change vs matched-target P3 | P4 useful / inserted |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in [r for r in rows if r["policy"] == "P4-neural-semantics"]:
        controls = {c["policy"]: c for c in rows if c["study"] == row["study"] and c["condition_id"] == row["condition_id"]}
        changes = [100 * (row["mean_ms"] / controls[p]["mean_ms"] - 1) for p in ("P0", "P2", "P3-early-miss")]
        lines.append(f"| {row['study']} | {row['seed']} | {row['workload']} | {row['gap_ms']} | {row['artificial_delay_ms']} | {row['mean_ms']:.3f} | {changes[0]:+.1f}% | {changes[1]:+.1f}% | {changes[2]:+.1f}% | {row['useful']}/{row['inserted']} |")
    lines += ["", "## Training and interpretation", "",
        "- P0: same bounded cache without prefetch. P1: current within-query Markov-2. P2: original matching neural-Markov checkpoint. P3-page-set: idle template Markov over full next-query pages. P3-early-miss: idle Markov using exactly P4's target definition. P4-no-semantics: separately retrained neural control with semantic blocks zeroed.",
        "- Training: 108 completed queries; validation: 48 queries using a different parameter pool. Targets are the first 16 distinct demand misses of each next training query. The query-level Markov prior and all neural training features use only training records. Protocols disclose any coincident validation SQL/parameter overlap.",
        "- Only completed templates and completed-query page history reach the online predictor. The next query's SQL, template, parameters, pages and actual arrival time are unavailable. Cached/pending pages are filtered before scoring for all idle policies.",
        "- Ranking combines the existing semantic MLP score with log empirical query-level Markov frequency. The 0.25 admission threshold applies to empirical training frequency, not a calibrated neural probability. P4 reranks up to 64 candidates for a 16-page proposal budget; unused insertions occupy at most eight cache slots.",
        "- Selected epochs are genuinely trained epochs chosen by validation early-miss recall. Zero-residual Markov validation scores are reported separately; no test-driven winner selection or substitution of Markov weights is used.",
        "- Offline page-set proposal recall uses the common P0 observed page sequence. It is not Top-1 accuracy or actual useful-prefetch rate. Runtime usefulness means a prefetched insertion was subsequently demanded; useful plus unused/wasted insertions equals all insertions.",
        "- All useful P3/P4 insertions were audited against actual query-arrival timestamps. No new read or cache insertion was admitted while foreground work was active. Already-admitted physical OS reads and an ongoing neural forward pass cannot be forcibly preempted.",
        "- The SQLite pager is released between queries, but OS cache is not reset. Artificial service, VFS bytes and file-read counters are not physical-device latency or I/O measurements. Idle windows are imposed, not measured from real user arrivals.",
        "- Pollution is an actual miss where a same-capacity demand-only shadow LRU hits. CPU time includes background threads and may exceed wall time. Large CPU consumption remains a concern; no causal threading diagnosis or CPU-efficiency improvement is claimed.",
        "- In the captured pilot runtime_stats.json, inherited worker_shutdown_clean can remain its default false for idle workers. Actual close return values were checked by the driver, and the independent audit requires worker_alive=false with zero worker/read errors. Raw artifacts are preserved.",
        "- Benefits from idle scheduling and target selection must not be attributed to semantics alone. Compare P4 against P3-early-miss and P4-no-semantics before making a semantic-benefit claim.", ""]
    for study, protocol in protocols.items():
        lines.append(f"### {study}")
        lines.append("")
        for policy, fit in protocol["fits"].items():
            lines.append(f"- {policy}: {fit['epochs_run']} epochs run, selected epoch {fit['selected_epoch']}; validation early-miss recall {fit['validation_selected']['early_miss_recall_macro_pct']:.2f}% (zero residual {fit['validation_zero_residual']['early_miss_recall_macro_pct']:.2f}%); residual weight norm {fit['selected_residual_weight_norm']:.6f}; checkpoint SHA-256 `{fit['checkpoint_sha256']}`.")
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json("audit.json", {"trials": len(rows), "query_executions": sum(r["queries"] for r in rows),
        "sql_results_identical": True, "idle_admission_only": True, "useful_pages_timely": True,
        "accounting_consistent": True, "workers_stopped": True,
        "zero_gap_idle_insertions": sum(r["inserted"] for r in rows if r["gap_ms"] == 0 and r["policy"].startswith(("P3", "P4")))})
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit(args.studies, args.out)
