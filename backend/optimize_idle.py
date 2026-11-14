"""Calibrate P4 optimizations on validation only, then compare in fresh processes.

The module imports no numerical libraries until a child process has configured
its native waiting policy. Original checkpoints and research exports are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

BACKEND = Path(__file__).resolve().parent
PILOTS = BACKEND / "runs/idle-neural-20261106"
LABELS = {"P0": "No prefetch", "P2": "Original neural hybrid", "P3": "Idle Markov",
          "P4-original": "Original idle neural", "P4-passive": "Idle neural with passive waiting",
          "P4-cached": "Optimized idle neural", "P4-selective": "Selective idle neural"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    import torch
    torch.set_num_threads(1)
    return torch


def load_assets(dataset):
    from idle_experiment import DEFAULT_ROOT, fingerprint
    from idle_neural import IdleNeuralHybrid
    protocol = read(PILOTS / f"{dataset}-pilot-001/protocol.json")
    database, directory = Path(protocol["database"]), Path(protocol["model_dir"])
    metadata = read(directory / "page_map.json")
    original_index = read(directory / "candidate_index.json")
    if original_index.get("static_metadata_path"):
        from pagewise.static_metadata import StaticMetadata
        metadata = StaticMetadata(original_index["static_metadata_path"])
    weights = PILOTS / f"{dataset}-pilot-001/P4-neural-semantics"
    model = IdleNeuralHybrid.load(weights, metadata, fingerprint(database))
    return database, directory, model, weights


def offline(model, records, budget=8):
    history, session, hits, targets, recalls, routes = [], None, 0, 0, [], Counter()
    for row in records:
        if row["session_id"] != session:
            history, session = [], row["session_id"]
        if history:
            proposal = model.predict(history, max_pages=budget)
            routes["neural" if proposal.get("neural_scored") else "markov_or_empty"] += 1
            target = set(list(dict.fromkeys(row["miss_pages"]))[:16])
            matched = len(target & {p["page"] for p in proposal["pages"]})
            hits, targets = hits + matched, targets + len(target)
            if target: recalls.append(matched / len(target))
        history.append(model.state(row["template_id"], row["pages"])); history = history[-2:]
    return {"macro_early_miss_recall_pct": statistics.mean(recalls)*100 if recalls else 0,
            "micro_early_miss_recall_pct": 100*hits/targets if targets else 0, "targets": targets, "routes": dict(routes)}


def prepare(args):
    configure()
    from idle_optimized import OptimizedIdleHybrid
    from pagewise.runledger import source_hash
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    database, directory, model, weights = load_assets(args.dataset)
    records = []
    for seed in (1201, 1202):
        records.extend(read(PILOTS / f"{args.dataset}-pilot-001/validation-{seed}/query_records.json"))
    # Check candidate filtering and exact neural scoring on the pre-existing
    # validation histories before any new evaluation queries are generated.
    optimized = OptimizedIdleHybrid(model)
    histories, session, checked = [], None, 0
    for row in records:
        if row["session_id"] != session: histories, session = [], row["session_id"]
        if histories:
            for exclude in ((), tuple(histories[-1]["pages"][-4:])):
                old, new = model.predict(histories, exclude_pages=exclude), optimized.predict(histories, exclude_pages=exclude)
                if [p["page"] for p in old["pages"]] != [p["page"] for p in new["pages"]]:
                    raise ValueError("Cached inference changed validation rankings")
                if any(abs(a["neural_markov_score"]-b["neural_markov_score"]) > 1e-6 for a,b in zip(old["pages"],new["pages"])):
                    raise ValueError("Cached inference changed neural scores")
                checked += 1
        histories.append(model.state(row["template_id"],row["pages"])); histories=histories[-2:]
    baseline = offline(model, records)
    screened = []
    for margin in (.1, .25, .5):
        control = OptimizedIdleHybrid(model, selective=True, admission_budget=8, margin=margin)
        result = offline(control, records)
        screened.append({"margin": margin, "min_support": 3, "validation": result})
    eligible = [r for r in screened if r["validation"]["macro_early_miss_recall_pct"] >= baseline["macro_early_miss_recall_pct"]-.5]
    selected = min(eligible, key=lambda r:(r["validation"]["routes"].get("neural",0), -r["validation"]["macro_early_miss_recall_pct"], -r["margin"])) if eligible else None
    plan = {"schema_version":1, "dataset":args.dataset, "database":str(database), "model_dir":str(directory),
        "weights":str(weights), "checkpoint_sha256":sha(weights/"idle_neural.pt"), "frozen_source_sha256":source_hash(),
        "cached_parity_cases":checked, "validation_neural_at8":baseline, "screened_gates":screened,
        "selected_gate":selected, "gate_enabled":selected is not None,
        "selection":"Minimize neural calls subject to validation macro early-miss recall within 0.5 percentage points of the unchanged neural model at budget 8; no evaluation results used",
        "candidate_limit":64, "maximum_proposal_budget":16, "speculative_fraction":.25,
        "waiting_policy":{"OMP_WAIT_POLICY":"PASSIVE","KMP_BLOCKTIME":"0"},
        "scope":"Same saved P4 weights; optimize execution and candidate feature reuse, not training or original study accuracy"}
    write(out/"optimization-plan.json",plan)
    print(json.dumps({"dataset":args.dataset,"parity_cases":checked,"selected_gate":selected,"baseline":baseline}),flush=True)


def child(args):
    configure()
    from idle_experiment import execute, fingerprint, make_cases, workload_pool
    from idle_markov import QueryMarkov
    from idle_optimized import OptimizedIdleHybrid
    from pagewise.runledger import source_hash
    from pagewise.workloads import QueryCase
    plan, spec = read(args.plan), read(args.spec)
    database, directory, model, weights = load_assets(plan["dataset"])
    before, core, weight_sha = fingerprint(database), source_hash(), sha(weights/"idle_neural.pt")
    if core != plan["frozen_source_sha256"] or weight_sha != plan["checkpoint_sha256"]:
        raise ValueError("Frozen model/source changed after optimization selection")
    cases = [QueryCase(**row) for row in spec["cases"]]
    query_model, policy = None, args.policy
    if policy == "P3":
        records=[]
        for seed in (1101,1102,1103): records.extend(read(PILOTS/f"{plan['dataset']}-pilot-001/training-{seed}/query_records.json"))
        query_model=QueryMarkov(2).fit(records)
    elif policy.startswith("P4"):
        query_model=model
        if policy in ("P4-cached","P4-selective"):
            gate=plan["selected_gate"] or {"min_support":3,"margin":.25}
            query_model=OptimizedIdleHybrid(model, selective=policy=="P4-selective" and plan["gate_enabled"],
                admission_budget=min(16,max(1,spec["cache_pages"]//4)) if policy=="P4-selective" else None,
                min_support=gate["min_support"],margin=gate["margin"])
    report, rows, _=execute(database,directory,args.out,cases,policy=policy,query_model=query_model,
        cache_pages=spec["cache_pages"],gap_ms=spec["gap_ms"],delay_ms=spec["delay_ms"],gap_schedule=spec.get("gap_schedule"))
    proposals=read(args.out/"idle_proposals.json") if query_model else []
    report.update({"label":LABELS[policy],"condition":spec["condition"],"seed":spec["seed"],"dataset":plan["dataset"],
        "environment":{k:os.environ.get(k) for k in ("OMP_WAIT_POLICY","KMP_BLOCKTIME")},
        "neural_jobs":sum(p.get("neural_scored",False) for p in proposals),"idle_jobs":len(proposals),
        "routes":dict(Counter(p.get("route","unchanged_neural" if p.get("neural_scored") else "markov_or_empty") for p in proposals)),
        "feature_cache":query_model.snapshot() if isinstance(query_model,OptimizedIdleHybrid) else None,
        "neural_stage_totals_ms":{k:sum(p.get("stage_ns",{}).get(k,0) for p in proposals)/1e6 for k in ("candidate_ns","feature_ns","forward_and_sort_ns")},
        "output":str(args.out.resolve())})
    if before!=fingerprint(database) or core!=source_hash() or weight_sha!=sha(weights/"idle_neural.pt"):
        raise ValueError("Protected database/source/weights changed")
    report["protected_artifacts_unchanged"]=True
    write(args.out/"comparison.json",report)
    print(f"{policy}: mean {report['mean_ms']:.3f} ms, CPU {report['process_cpu_ms']:.1f} ms, useful {report['runtime']['useful_prefetches']}/{report['runtime']['pages_prefetched']}, neural jobs {report['neural_jobs']}",flush=True)


def launch(policy, plan, spec, out):
    env={**os.environ,"PYTHONIOENCODING":"utf-8"}
    if policy not in ("P4-original","P2"):
        env.update(OMP_WAIT_POLICY="PASSIVE",KMP_BLOCKTIME="0")
    command=[sys.executable,"-u",str(Path(__file__).resolve()),"child","--policy",policy,"--plan",str(plan),"--spec",str(spec),"--out",str(out)]
    completed=subprocess.run(command,cwd=BACKEND,env=env,text=True,capture_output=True)
    print(completed.stdout,end="",flush=True)
    if completed.returncode: raise RuntimeError(completed.stderr)
    return read(out/"comparison.json")


def fresh_pool(database, seed, forbidden):
    from pagewise.workloads import generate_workload
    pool={}
    for case in generate_workload(database,count=1200,seed=seed,mixed=False):
        if (case.sql,tuple(case.params)) in forbidden: continue
        values=pool.setdefault(case.template_id,[])
        if len(values)<4 and all(tuple(c.params)!=tuple(case.params) for c in values): values.append(case)
    if any(len(values)<4 for values in pool.values()): raise ValueError("Not enough fresh parameter variants")
    return pool


def run(args):
    # Numerical libraries in this parent only construct SQL workloads; policy
    # inference always runs sequentially in a fresh child process.
    configure()
    from idle_experiment import make_cases, workload_pool
    plan=read(args.plan); database=Path(plan["database"])
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    forbidden=set()
    for seed in (1101,1102,1103):
        for row in read(PILOTS/f"{plan['dataset']}-pilot-001/training-{seed}/workload.json"):
            forbidden.add((row["sql"],tuple(row["params"])))
    conditions=[("familiar-delayed",seed,args.queries,50,.2,32,"repeated",False) for seed in range(11001,11006)]
    conditions += [("fresh-no-delay",seed,args.queries,50,0,32,"repeated",True) for seed in range(12001,12006)]
    conditions += [("fresh-variable-gaps",12006,args.queries+12,0,0,32,"repeated",True),
                   ("novel-templates-large-cache",12007,args.queries,25,0,128,"novel",True),
                   ("fresh-zero-gap-small-cache",12008,args.queries+12,0,0,16,"repeated",True)]
    policies=list(LABELS)
    protocol={"plan":plan,"conditions":conditions,"policies":policies,"queries":args.queries,
        "extension_source_sha256":{name:sha(BACKEND/name) for name in ("idle_optimized.py","optimize_idle.py","idle_experiment.py","idle_runtime.py","idle_neural.py","idle_markov.py")},
        "process_isolation":"Each policy/condition in a fresh subprocess, sequential execution, randomized policy order",
        "notes":"Familiar values intentionally overlap training. Fresh conditions exclude identical training SQL/parameter pairs. Gaps are controlled, not naturally observed. OS cache is not reset. Nonzero service delay is artificial."}
    write(out/"protocol.json",protocol)
    results=[]
    for number,(name,seed,count,gap,delay,cache,workload,fresh) in enumerate(conditions):
        pool=fresh_pool(database,seed+40000,forbidden) if fresh else workload_pool(database)
        cases=make_cases(pool,count,seed,workload)
        schedule=[(0,5,25,50,100)[i%5] for i in range(count-1)] if name=="fresh-variable-gaps" else None
        spec={"condition":name,"seed":seed,"cache_pages":cache,"gap_ms":gap,"delay_ms":delay,
              "gap_schedule":schedule,"cases":[case.serializable() for case in cases],
              "training_value_overlap":sum((case.sql,tuple(case.params)) in forbidden for case in cases)}
        if fresh and spec["training_value_overlap"]: raise ValueError("Fresh evaluation overlaps training values")
        target=out/f"condition-{number:02d}";target.mkdir()
        write(target/"spec.json",spec)
        order=policies.copy();random.Random(seed).shuffle(order)
        print(f"{number+1}/{len(conditions)} {plan['dataset']} {name} seed={seed}",flush=True)
        expected=None
        for policy in order:
            report=launch(policy,args.plan.resolve(),target/"spec.json",target/policy)
            with (target/policy/"queries.csv").open(newline="",encoding="utf-8") as handle:
                values=[(r["query_id"],r["rows"],r["result_sha256"]) for r in csv.DictReader(handle)]
            if expected is not None and values!=expected: raise ValueError("SQL results differ")
            expected=values;results.append(report)
            write(out/"results.json",{"protocol":protocol,"results":results})
    if any(sha(BACKEND/name)!=checksum for name,checksum in protocol["extension_source_sha256"].items()):
        raise ValueError("Experiment implementation changed during evaluation")
    write(out/"verification.json",{"trials":len(results),"sql_results_identical":True,"protected_artifacts_unchanged":True,"experiment_source_unchanged":True})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","child","run"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--dataset",choices=("small","large"),default="small")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--policy",choices=tuple(LABELS))
    parser.add_argument("--queries",type=int,default=24)
    args=parser.parse_args()
    if args.action in ("child","run") and args.plan is None: parser.error("--plan is required")
    if args.action=="child" and (args.spec is None or args.policy is None): parser.error("--spec and --policy are required")
    if args.queries<2: parser.error("At least two queries are required")
    {"prepare":prepare,"child":child,"run":run}[args.action](args)
