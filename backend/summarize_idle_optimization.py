"""Audit every isolated optimization trial, retaining all conditions and controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(studies, out):
    import numpy as np
    from pagewise.metadata import read_page_size
    out.mkdir(parents=True,exist_ok=False)
    flat, all_data, paired = [], {}, []
    for study in studies:
        if (study/"EXCLUDED.json").exists():raise ValueError("Excluded development attempt")
        data, verified=read(study/"results.json"),read(study/"verification.json")
        if not all(verified.get(k) for k in ("sql_results_identical","protected_artifacts_unchanged","experiment_source_unchanged")):
            raise ValueError("Failed study verification")
        if verified["trials"]!=len(data["results"]):raise ValueError("Trial count mismatch")
        all_data[data["protocol"]["plan"]["dataset"]]=data
        page_size = read_page_size(data["protocol"]["plan"]["database"])
        for result in data["results"]:
            rt=result["runtime"];target=Path(result["output"])
            if rt["worker_alive"] or not rt["worker_shutdown_clean"] or rt["prefetch_read_errors"] or rt.get("worker_errors"):
                raise ValueError("Worker failed")
            inserted,useful=rt["pages_prefetched"],rt["useful_prefetches"]
            if useful+rt["total_wasted_prefetches"]!=inserted:raise ValueError("Incorrect cache accounting")
            idle=result["policy"] not in ("P0","P2")
            if idle:
                active=False
                for event in read(target/"idle_events.json"):
                    if event["event"]=="idle_open":active=True
                    elif event["event"]=="arrival":active=False
                    elif event["event"] in ("read_admitted","cache_inserted") and not active:raise ValueError("Non-idle speculative admission")
                if useful!=result["useful_idle_pages_inserted_before_query_arrival"]:raise ValueError("Late useful insertion")
            if result["injected_delay_ms"] and rt["storage_device"]["demand_ios"]<=0:raise ValueError("Demand delay not wired")
            row={"dataset":result["dataset"],"condition":result["condition"],"seed":result["seed"],"policy":result["policy"],
                 "label":result["label"],"queries":result["queries"],"cache_pages":result["cache_pages"],
                 "gap_ms":result["configured_gap_ms"],"artificial_delay_ms":result["injected_delay_ms"],
                 "mean_ms":result["mean_ms"],"run_p95_ms":result["p95_ms"],"cpu_ms":result["process_cpu_ms"],
                 "workload_wall_ms":result["workload_completion_ms"],"initialization_cpu_ms":result["initialization_cpu_ms"],
                 "useful":useful,"prefetched":inserted,"useful_pct":100*useful/inserted if inserted else None,
                 "wasted_insertions":rt["total_wasted_prefetches"],"pollution_misses":rt.get("pollution_misses",0),
                 "timely_useful":result["useful_idle_pages_inserted_before_query_arrival"],
                 "demand_read_bytes":result["demand_read_bytes"],"idle_read_bytes":rt.get("idle",{}).get("prefetch_reads",0)*page_size if idle else None,
                 "neural_calls":result.get("stage_profile",{}).get("neural_forward_calls",0) if result["policy"]=="P2" else result["neural_jobs"],
                 "idle_jobs":result["idle_jobs"],"feature_cache_hits":(result["feature_cache"] or {}).get("feature_hits",0),
                 "feature_cache_entries":(result["feature_cache"] or {}).get("feature_entries",0),
                 "candidate_total_ms":result["neural_stage_totals_ms"]["candidate_ns"],
                 "features_total_ms":result["neural_stage_totals_ms"]["feature_ns"],
                 "forward_sort_total_ms":result["neural_stage_totals_ms"]["forward_and_sort_ns"],"output":str(target)}
            if row["feature_cache_entries"]>4096:raise ValueError("Unbounded feature cache")
            flat.append(row)
        for row in [r for r in flat if r["dataset"]==data["protocol"]["plan"]["dataset"] and r["policy"]!="P0"]:
            control=next(r for r in flat if r["dataset"]==row["dataset"] and r["condition"]==row["condition"] and r["seed"]==row["seed"] and r["policy"]=="P0")
            with (Path(row["output"])/"queries.csv").open(newline="",encoding="utf-8") as f:qs=list(csv.DictReader(f))
            with (Path(control["output"])/"queries.csv").open(newline="",encoding="utf-8") as f:cs=list(csv.DictReader(f))
            for q,c in zip(qs,cs,strict=True):
                if (q["query_id"],q["rows"],q["result_sha256"])!=(c["query_id"],c["rows"],c["result_sha256"]):raise ValueError("SQL mismatch")
                paired.append({"dataset":row["dataset"],"condition":row["condition"],"seed":row["seed"],"policy":row["policy"],
                    "query_id":q["query_id"],"delta_vs_P0_ms":float(q["elapsed_ms"])-float(c["elapsed_ms"])})
    def write_json(name,value): (out/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
    def write_csv(name,value):
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(value[0]));w.writeheader();w.writerows(value)
    write_json("all_trials.json",flat);write_csv("all_trials.csv",flat);write_csv("paired_queries.csv",paired)
    grouped={}
    for row in flat:grouped.setdefault((row["dataset"],row["condition"],row["policy"]),[]).append(row)
    aggregate=[]
    for (dataset,condition,policy),rows in grouped.items():
        u,p=sum(r["useful"] for r in rows),sum(r["prefetched"] for r in rows)
        aggregate.append({"dataset":dataset,"condition":condition,"policy":policy,"label":rows[0]["label"],"seeds":len(rows),
            "mean_ms":statistics.mean(r["mean_ms"] for r in rows),"mean_run_p95_ms":statistics.mean(r["run_p95_ms"] for r in rows),
            "mean_cpu_ms":statistics.mean(r["cpu_ms"] for r in rows),"mean_workload_wall_ms":statistics.mean(r["workload_wall_ms"] for r in rows),
            "useful":u,"prefetched":p,"useful_pct":100*u/p if p else None,"pollution_misses":sum(r["pollution_misses"] for r in rows),
            "neural_calls":sum(r["neural_calls"] for r in rows),"idle_jobs":sum(r["idle_jobs"] for r in rows)})
    write_json("aggregate.json",aggregate)
    comparisons=[];rng=np.random.default_rng(1701)
    for dataset in all_data:
        for condition in ("familiar-delayed","fresh-no-delay"):
            for comparator in ("P4-original","P3","P0"):
                treatment=sorted([r for r in flat if (r["dataset"],r["condition"],r["policy"])==(dataset,condition,"P4-cached")],key=lambda r:r["seed"])
                control=sorted([r for r in flat if (r["dataset"],r["condition"],r["policy"])==(dataset,condition,comparator)],key=lambda r:r["seed"])
                if [r["seed"] for r in treatment]!=[r["seed"] for r in control]:raise ValueError("Unpaired seeds")
                delta=np.array([a["mean_ms"]-b["mean_ms"] for a,b in zip(treatment,control)])
                boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
                comparisons.append({"dataset":dataset,"condition":condition,"policy":"P4-cached","comparator":comparator,
                    "paired_seeds":len(delta),"mean_delta_ms":float(delta.mean()),"bootstrap_seed_ci95_ms":np.quantile(boot,[.025,.975]).tolist(),
                    "note":"Exploratory paired run-mean bootstrap over five seeds; small-sample interval, no multiple-comparison adjustment"})
    write_json("paired_seed_comparisons.json",comparisons)
    lines=["# Idle neural optimization: complete measured evaluation","",
        "Same P4 checkpoints and trained semantic MLP as the earlier pilot. Optimizations are passive native-thread waiting and bounded reuse of immutable semantic/page/history features. A separately labeled selective gate may use Markov alone; it is not substituted for the full neural policy.","",
        f"Accepted: {len(flat)} fresh-process trials, {sum(r['queries'] for r in flat)} evaluation query executions. Every comparison has matching SQL row-stream hashes. All conditions and controls remain in CSV/JSON.","",
        "Each policy runs sequentially in a new process with randomized order. Original P4 and original within-query P2 retain inherited native-library waiting settings; other policies use OMP_WAIT_POLICY=PASSIVE and KMP_BLOCKTIME=0 before import. This isolates an execution-policy change, not new neural training.","",
        "## Five-seed comparisons","",
        "Familiar-delayed deliberately reuses familiar training parameter values, with 50 ms gaps, 0.2 ms requested artificial service and a 32-page cache. Fresh-no-delay uses SQL/parameter combinations absent from the original P4 training sessions, 50 ms gaps, no artificial service and the same cache. Each has five seeds and 24 queries per policy/seed. p95 is the mean of the five short-run p95 estimates, not a pooled percentile.","",
        "| Database | Condition | Policy | Mean ms | Mean run p95 ms | Process CPU ms | Useful / inserted | Neural calls |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(aggregate,key=lambda r:(r["dataset"],r["condition"],r["policy"])):
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['label']} | {r['mean_ms']:.3f} | {r['mean_run_p95_ms']:.3f} | {r['mean_cpu_ms']:.2f} | {r['useful']}/{r['prefetched']} | {r['neural_calls']} |")
    lines += ["","## Selection, correctness and boundaries","",
        "- Cached full-neural inference matched original rankings and logits within 1e-6 on 184 validation/exclusion cases across the two databases before evaluation. Caches are bounded and cannot modify weights or learn evaluation successors.",
        "- The selective gate is chosen only from pre-existing validation records: minimum support 3; margin screened at 0.1/0.25/0.5; minimize neural calls while losing at most 0.5 percentage points of macro early-miss recall at budget 8. Both datasets selected margin 0.5 without observed validation recall loss. The small validation set used neural scoring in 36/46 cases; large used 46/46.",
        "- At runtime the selective proposal budget is the smaller of 16 and 25% of cache capacity. The gate skips neural ordering when all candidates fit, or the empirical prior clearly separates the cutoff. The full cached-neural policy retains the original 16-page proposal budget and always scores available candidates.",
        "- Fresh parameter workloads explicitly exclude identical SQL/parameter pairs from P4 training. Their query templates and deterministic session pattern remain familiar; this is not an arbitrary real-user workload or a fully independent database.",
        "- Additional single-seed cases cover variable controlled gaps (0/5/25/50/100 ms), query-level unfamiliar templates with a 128-page cache, and zero gaps with a 16-page cache. The predictor is never told future SQL, parameters or arrival times.",
        "- Workers stop cleanly; idle admission stays between query completion and new arrival. All useful idle insertions completed before their demand query arrived. An already-admitted OS read or running neural pass is not forcibly preempted.",
        "- OS cache is not reset. Artificial delay is not a measured physical-device service time. Low CPU counters have Windows process-time quantization; CPU includes all process threads. Profiling and metadata preparation outside query workload are reported separately where available.",
        "- CPU savings do not by themselves establish lower SQL latency or superior semantic prediction. Cached features preserve model predictions; selective gate results remain an explicit ablation. Do not choose a winning test condition to claim general speedup.",
        "- Full paired per-query differences and five-seed bootstrap intervals are exported. Five seeds and short runs support an expanded development evaluation, not a final population-level speedup claim.","",
        "Native waiting references: [LLVM OpenMP runtime](https://openmp.llvm.org/design/Runtimes.html), [PyTorch threading environment](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).",""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    write_json("audit.json",{"trials":len(flat),"query_executions":sum(r["queries"] for r in flat),"sql_results_identical":True,
        "workers_clean":True,"useful_insertions_timely":True,"idle_admission_only":True,"accounting_consistent":True})
    print(json.dumps([r for r in aggregate if r["condition"] in ("familiar-delayed","fresh-no-delay")],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.studies,args.out)
