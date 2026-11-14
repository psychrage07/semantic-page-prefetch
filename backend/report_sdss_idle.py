"""Audit SDSS cross-dataset trials and report latency, CPU and frozen-weight transfer."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from sdss_idle_experiment import LABELS, apply_frozen_neural_weights, read, records, sha, source_guard, tensor_hash


def quality(model, sessions, budget=8):
    recalls=[];hits=targets=top1=top1_total=neural=groups=0
    for session in sessions:
        history=[]
        for row in session:
            if history:
                groups+=1
                prediction=model.predict(history,max_pages=budget)
                neural+=bool(prediction.get("neural_scored"))
                actual=list(dict.fromkeys(row["miss_pages"]))[:16]
                proposed=[p["page"] for p in prediction["pages"]]
                matched=len(set(actual)&set(proposed))
                hits+=matched;targets+=len(actual)
                if actual:
                    recalls.append(matched/len(actual));top1_total+=1
                    top1+=bool(proposed and proposed[0]==actual[0])
            history.append(model.state(row["template_id"],row["pages"]));history=history[-2:]
    return {"query_boundaries":groups,"first_miss_eligible_queries":top1_total,"early_miss_targets":targets,"neural_calls":neural,
        "first_miss_top1_pct":100*top1/top1_total if top1_total else None,
        "early_miss_recall_macro_at8_pct":100*statistics.mean(recalls) if recalls else None,
        "early_miss_recall_micro_at8_pct":100*hits/targets if targets else None,
        "scope":"Common P0 no-delay test traces; first-miss Top-1 predicts the next query's first demand miss, not the original within-query next-page task. Recall covers first 16 distinct next-query misses at proposal budget 8; absent candidates count as errors."}


def report(args):
    import numpy as np
    import torch
    from idle_experiment import fingerprint
    from idle_markov import QueryMarkov
    from idle_neural import IdleNeuralHybrid, early_miss_records
    from idle_optimized import OptimizedIdleHybrid
    torch.set_num_threads(1)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    study=read(args.evaluation/"results.json");verification=read(args.evaluation/"verification.json")
    plan=study["protocol"]["plan"];training_root=args.plan.resolve().parent
    if any(value is not True for key,value in verification.items() if key not in ("trials","query_executions")):
        raise ValueError("Study verification failed")
    flat=[];paired=[]
    for trial in study["results"]:
        rt=trial["runtime"];path=Path(trial["output"])
        if rt["worker_alive"] or not rt["worker_shutdown_clean"] or rt.get("worker_errors") or rt["prefetch_read_errors"]:
            raise ValueError("Worker failure")
        useful,prefetched=rt["useful_prefetches"],rt["pages_prefetched"]
        if useful+rt["total_wasted_prefetches"]!=prefetched:raise ValueError("Cache accounting error")
        if trial["is_idle"]:
            active=False
            for event in read(path/"idle_events.json"):
                if event["event"]=="idle_open":active=True
                elif event["event"]=="arrival":active=False
                elif event["event"] in ("read_admitted","cache_inserted") and not active:raise ValueError("Speculative work admitted during foreground query")
            if trial["useful_idle_pages_inserted_before_query_arrival"]!=useful:raise ValueError("A useful idle insertion arrived late")
        if trial["injected_delay_ms"] and rt["storage_device"]["demand_ios"]<=0:raise ValueError("Demand service delay missing")
        flat.append({"condition":trial["condition"],"seed":trial["seed"],"policy":trial["policy"],"label":trial["label"],
            "queries":trial["queries"],"cache_pages":trial["cache_pages"],"gap_ms":trial["configured_gap_ms"],
            "artificial_delay_ms":trial["injected_delay_ms"],"mean_query_ms":trial["mean_ms"],"run_p95_ms":trial["p95_ms"],
            "cpu_ms":trial["process_cpu_ms"],"workload_wall_ms":trial["workload_completion_ms"],
            "useful":useful,"prefetched":prefetched,"useful_pct":100*useful/prefetched if prefetched else None,
            "wasted":rt["total_wasted_prefetches"],"pollution_misses":rt.get("pollution_misses",0),
            "timely_useful":trial["useful_idle_pages_inserted_before_query_arrival"],"neural_calls":trial["neural_calls"],
            "idle_jobs":trial["idle_jobs"],"demand_vfs_bytes":trial["demand_read_bytes"],
            "neural_state_sha256":trial["executed_neural_state_sha256"],"output":trial["output"]})
    if len(flat)!=verification["trials"]:raise ValueError("Trial count mismatch")
    for trial in flat:
        baseline=next(r for r in flat if r["condition"]==trial["condition"] and r["seed"]==trial["seed"] and r["policy"]=="P0")
        def queries(row):
            with (Path(row["output"])/"queries.csv").open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
        for q,b in zip(queries(trial),queries(baseline),strict=True):
            if (q["query_id"],q["rows"],q["result_sha256"])!=(b["query_id"],b["rows"],b["result_sha256"]):raise ValueError("SQL mismatch")
            if trial["policy"]!="P0":paired.append({"condition":trial["condition"],"seed":trial["seed"],"policy":trial["policy"],"query_id":q["query_id"],"delta_vs_P0_ms":float(q["elapsed_ms"])-float(b["elapsed_ms"])})
    def write_json(name,value):(out/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
    def write_csv(name,value):
        with (out/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(value[0]));writer.writeheader();writer.writerows(value)
    write_json("all_trials.json",flat);write_csv("all_trials.csv",flat);write_csv("paired_queries.csv",paired)
    aggregates=[]
    for condition in sorted({r["condition"] for r in flat}):
        for policy in LABELS:
            rows=[r for r in flat if (r["condition"],r["policy"])==(condition,policy)]
            useful,prefetched=sum(r["useful"] for r in rows),sum(r["prefetched"] for r in rows)
            aggregates.append({"condition":condition,"policy":policy,"label":LABELS[policy],"seeds":len(rows),
                "mean_query_ms":statistics.mean(r["mean_query_ms"] for r in rows),
                "mean_run_p95_ms":statistics.mean(r["run_p95_ms"] for r in rows),
                "mean_cpu_ms":statistics.mean(r["cpu_ms"] for r in rows),
                "mean_workload_wall_ms":statistics.mean(r["workload_wall_ms"] for r in rows),
                "useful":useful,"prefetched":prefetched,"useful_pct":100*useful/prefetched if prefetched else None,
                "neural_calls":sum(r["neural_calls"] for r in rows),"pollution_misses":sum(r["pollution_misses"] for r in rows)})
    write_json("aggregate.json",aggregates);write_csv("aggregate.csv",aggregates)
    rng=np.random.default_rng(20261114);comparisons=[]
    for condition in ("fresh-no-delay","fresh-delayed"):
        for treatment,control in (("P2-passive","P2"),("P4-cached","P4-original"),("P4-cached","P3-early"),
                                  ("P4-transfer-small","P4-cached"),("P4-transfer-large","P4-cached"),("P4-cached","P0")):
            a=sorted([r for r in flat if (r["condition"],r["policy"])==(condition,treatment)],key=lambda r:r["seed"])
            b=sorted([r for r in flat if (r["condition"],r["policy"])==(condition,control)],key=lambda r:r["seed"])
            if [r["seed"] for r in a]!=[r["seed"] for r in b]:raise ValueError("Seed pairing failed")
            deltas=np.asarray([x["mean_query_ms"]-y["mean_query_ms"] for x,y in zip(a,b)])
            samples=rng.choice(deltas,size=(10000,len(deltas)),replace=True).mean(axis=1)
            comparisons.append({"condition":condition,"treatment":treatment,"control":control,"paired_seeds":len(a),
                "mean_delta_ms":float(deltas.mean()),"bootstrap_seed_ci95_ms":np.quantile(samples,[.025,.975]).tolist(),
                "scope":"Exploratory paired run-mean bootstrap across five seeds, no multiple-comparison adjustment"})
    write_json("paired_seed_comparisons.json",comparisons)
    # Offline transfer comparison on common no-prefetch observations, after all
    # runtime settings and checkpoints have already been frozen and evaluated.
    sessions=[read(Path(r["output"])/"query_records.json") for r in flat if r["condition"]=="fresh-no-delay" and r["policy"]=="P0"]
    training=[]
    for seed in (101,102,103):training.extend(read(training_root/f"training-{seed}/query_records.json"))
    metadata=read(Path(plan["runtime_model_dir"])/"page_map.json")
    model=lambda folder="P4-neural-semantics":IdleNeuralHybrid.load(training_root/folder,metadata,fingerprint(plan["database"]))
    models={"P3-full":QueryMarkov(2).fit(training),"P3-early":QueryMarkov(2).fit(early_miss_records(training)),
            "P4-cached":OptimizedIdleHybrid(model()),"P4-no-semantics":model("P4-no-semantics")}
    for dataset in ("small","large"):
        transferred=model();source=plan["transfers"][dataset]
        apply_frozen_neural_weights(transferred,source["checkpoint"],source["sha256"])
        models[f"P4-transfer-{dataset}"]=OptimizedIdleHybrid(transferred)
    accuracy={name:quality(value,sessions) for name,value in models.items()}
    write_json("query_level_quality.json",accuracy)
    guard=read(training_root/"main-project-before.json")
    if any(sha(name)!=checksum for name,checksum in guard.items()):raise ValueError("Original main project artifacts changed")
    if source_guard()!=read(args.evaluation/"original-sdss-before.json"):raise ValueError("Original SDSS experiment changed")
    transfer_checks={}
    for dataset in ("small","large"):
        selected=[r for r in flat if r["policy"]==f"P4-transfer-{dataset}"]
        if len({r["neural_state_sha256"] for r in selected})!=1:raise ValueError("Transfer neural weights varied between trials")
        if sum(r["neural_calls"] for r in selected)<=0:raise ValueError("Transfer arm never executed neural scoring")
        source_protocol=read(Path(plan["transfers"][dataset]["checkpoint"]).parent.parent/"protocol.json")
        if source_protocol["frozen_source_sha256"]!=plan["core_source_sha256"]:
            raise ValueError("Source and target core feature implementations differ")
        if tensor_hash(models[f"P4-transfer-{dataset}"].base.model)!=selected[0]["neural_state_sha256"]:
            raise ValueError("Executed transfer tensors differ from frozen source tensors")
        transfer_checks[dataset]={"source_checkpoint_sha256":plan["transfers"][dataset]["sha256"],
            "executed_state_sha256":selected[0]["neural_state_sha256"],"actual_runtime_neural_calls":sum(r["neural_calls"] for r in selected),
            "source_weights_unchanged":True,"matching_core_feature_implementation":True,
            "executed_tensors_match_frozen_source":True,"target_neural_fitting_for_transfer":False,"target_statistics_fitted":True}
    write_json("transfer-verification.json",transfer_checks)
    audit={**verification,"idle_admission_only":True,"useful_insertions_timely":True,"workers_clean":True,
        "original_frontend_and_research_artifacts_unchanged":True,"cross_dataset_neural_execution_verified":True,
        "zero_shot_transfer_claim":False,"cached_validation_parity_cases":plan["cached_parity_cases"]}
    write_json("audit.json",audit)
    lines=["# SDSS cross-dataset verification of idle neural optimization","",
        "Dataset: the existing real SDSS DR7 subset linked to SeLeP, with 49,999 photometric objects and 50,000 spectra in a 15,790,080-byte SQLite database. SQL templates are generated local adaptations, not SeLeP's original logs. This is not a SeLeP reproduction or a comparison with its published performance.","",
        f"Completed {len(flat)} matched fresh-process trials and {sum(r['queries'] for r in flat)} evaluation query executions. Separately retraced 600 training and 200 validation queries. New test combinations exclude every original training, validation and test signature and use the original held-out SHA256 partitions.","",
        "## Two distinct questions","",
        "1. Architecture/execution reproducibility: fit separate SDSS idle-neural weights with the same fixed P4 architecture and admission rules, then compare original and optimized execution on matching held-out queries.",
        "2. Frozen neural transfer: execute the small-database or 8.62 GB P4 neural state_dict unchanged on SDSS features. Candidate generation, Markov statistics and feature normalization come from SDSS training records and static metadata. There is no SDSS neural fitting in these arms. This is transfer with target-side statistical adaptation, not zero-shot transfer.","",
        "The original SDSS within-query neural checkpoint is also tested with inherited and passive waiting settings. It is a different task/model from the newly trained next-query idle predictor.","",
        "## Mean query latency and CPU","",
        "Five seeds per main condition, 60 queries per policy/seed, 50 ms imposed gaps, 32-page cache. Fresh-no-delay has no artificial service delay; fresh-delayed adds 0.2 ms requested artificial shared service. Every trial reads the complete database to verify SHA256 before timing, so these are warm-OS-cache tests. p95 is the average of run p95 estimates, not a pooled percentile.","",
        "| Condition | Policy | Mean query ms | Mean run p95 ms | Process CPU ms | Useful / inserted | Neural calls |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['condition']} | {row['label']} | {row['mean_query_ms']:.3f} | {row['mean_run_p95_ms']:.3f} | {row['mean_cpu_ms']:.2f} | {row['useful']}/{row['prefetched']} | {row['neural_calls']} |")
    lines += ["","Single-seed variable-gap (0/5/25/50/100 ms) and zero-gap/128-page-cache results are included above and in all_trials.csv. No condition is selected or excluded because it wins or loses.","",
        "## Query-level prediction check","",
        "These metrics use the same five P0 no-delay page streams. First-miss Top-1 means the first proposal equals the next query's first custom-cache demand miss. Recall@8 covers its first 16 distinct misses. These are not the earlier within-query 35.32% SDSS Top-1 score or the original 47.74% research score.","",
        "| Predictor | First-miss Top-1 % | Macro early-miss recall@8 % | Micro early-miss recall@8 % | Evaluated boundaries |",
        "|---|---:|---:|---:|---:|"]
    for name,value in accuracy.items():
        lines.append(f"| {LABELS[name]} | {value['first_miss_top1_pct']:.3f} | {value['early_miss_recall_macro_at8_pct']:.3f} | {value['early_miss_recall_micro_at8_pct']:.3f} | {value['query_boundaries']} |")
    lines += ["","## Interpretation and limits","",
        "- Neural weights were trained on SDSS only for the explicitly SDSS-trained arms; transferred source tensors were checksum-verified, loaded strictly, marked non-trainable and checked unchanged after every runtime trial. Logs and hashes prove neural execution occurred; execution alone does not establish prediction superiority.",
        "- Transfer is one-way from the two synthetic-database P4 models to SDSS, with target-side statistics. Reverse transfer and fully zero-shot prediction without target training traces were not tested.",
        "- Selected SDSS semantic and no-semantics models each ran six epochs and selected epoch one. The selected semantic residual is nonzero, but validation early-miss recall at budget 16 equaled the zero-residual Markov reference. Keep that limitation when attributing gains.",
        "- Candidate and Markov counts use SDSS training only. Gate margin 0.5 is selected from validation only. Cached score/ranking parity was checked on 396 validation/exclusion cases before test queries were generated.",
        "- Query order is shuffled, not the deterministic repeating e-commerce session pattern used in earlier experiments. Idle gaps are imposed rather than measured from human users. Database size, layout, working set, cache and query mix differ from the 8.62 GB study; compare matched policies within this dataset.",
        "- This is a bounded, nonrandom DR7 subset ordered by spectrum ID, not SeLeP's complete PostgreSQL database or a representative astronomy workload. Overlapping ranges can access the same objects/pages despite disjoint SQL parameter tuples.",
        "- CPU includes process threads, and low counters have Windows time granularity. VFS reads and imposed delay do not prove physical disk I/O. Bootstrap intervals across five seeds are exploratory and do not correct for multiple comparisons.",
        "- Original SDSS source/data/models/results, main-project dashboard exports and original research weights passed before/after checks. New training, weights, traces, CSV/JSON and reports are isolated in this sibling experiment folder.","",
        "Files: all_trials.csv/json, aggregate.csv/json, paired_queries.csv, paired_seed_comparisons.json, query_level_quality.json, transfer-verification.json and audit.json. Raw SQL workloads, VFS traces, worker logs and weights remain in the adjacent training/evaluation folders.",""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"audit":audit,"main_comparisons":[r for r in aggregates if r["condition"] in ("fresh-no-delay","fresh-delayed")],"quality":accuracy},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    report(parser.parse_args())
