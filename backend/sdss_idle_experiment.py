"""Isolated SDSS cross-dataset verification of P4 execution and frozen-weight transfer."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

BACKEND = Path(__file__).resolve().parent
SOURCE = BACKEND.parent.parent / "paper-dataset-experiments/selep-sdss-dr7-20261027"
DEFAULT_OUT = SOURCE.parent / "selep-sdss-idle-20261114"
LABELS = {"P0":"No prefetch", "P1":"Within-query Markov-2", "P2":"Original SDSS neural hybrid",
    "P2-passive":"Original SDSS neural, passive waiting", "P3-full":"Idle Markov, full page sets",
    "P3-early":"Idle Markov, matched early-miss targets", "P4-original":"New SDSS idle neural, inherited waiting",
    "P4-cached":"Optimized SDSS idle semantic neural", "P4-selective":"Selective SDSS idle neural",
    "P4-no-semantics":"Retrained SDSS idle neural without semantics",
    "P4-transfer-small":"Frozen small-database neural weights + SDSS statistics",
    "P4-transfer-large":"Frozen 8.62 GB neural weights + SDSS statistics"}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): digest.update(block)
    return digest.hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def source_guard():
    manifest = read(SOURCE/"artifacts.sha256.json")
    verified = {}
    for name, declared in manifest.items():
        path = SOURCE / name.replace("\\", "/")
        actual = sha(path)
        if actual != declared["sha256"]: raise ValueError(f"Original SDSS artifact changed: {name}")
        verified[name] = actual
    return verified


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def configure():
    import torch
    torch.set_num_threads(1)
    return torch


def apply_frozen_neural_weights(base, checkpoint, expected_sha256):
    import torch
    if sha(checkpoint) != expected_sha256:
        raise ValueError("Source transfer checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["input_dim"] != base.model.net[0].in_features or payload["config"]["hidden"] != base.config.hidden:
        raise ValueError("Transfer architecture is incompatible")
    base.model.load_state_dict(payload["state_dict"])
    base.model.eval().requires_grad_(False)
    return tensor_hash(base.model)


def select_gate(model, validation):
    from idle_optimized import OptimizedIdleHybrid
    from optimize_idle import offline
    reference = offline(model, validation, 8)
    choices = []
    for margin in (.1, .25, .5):
        candidate = OptimizedIdleHybrid(model, selective=True, admission_budget=8, margin=margin)
        choices.append({"margin":margin, "min_support":3, "validation":offline(candidate, validation, 8)})
    eligible = [r for r in choices if r["validation"]["macro_early_miss_recall_pct"] >= reference["macro_early_miss_recall_pct"]-.5]
    best = min(eligible, key=lambda r:(r["validation"]["routes"].get("neural", 0), -r["validation"]["macro_early_miss_recall_pct"], -r["margin"])) if eligible else None
    return {"reference":reference, "screened":choices, "selected":best,
            "rule":"Validation only: minimize neural calls subject to <=0.5 percentage-point macro early-miss recall loss at budget 8"}


def prepare(args):
    os.environ.update(OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
    torch = configure()
    from idle_experiment import execute, fingerprint
    from idle_neural_experiment import fit_models
    from pagewise.runledger import source_hash
    from pagewise.workloads import QueryCase
    from sdss_idle_workload import signature
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    original = source_guard()
    write(out/"original-sdss-before.json", original)
    database = SOURCE/"data/sdss-dr7.sqlite"
    runtime_dir = out/"runtime-model";runtime_dir.mkdir()
    for source, dest in (("candidate_index.json","candidate_index.json"),("page_map.json","page_map.json"),("hybrid_semantics.pt","model_ranker.pt")):
        shutil.copyfile(SOURCE/"models"/source, runtime_dir/dest)
    training, validation = [], []
    for split, seeds, destination in (("training",(101,102,103),training),("validation",(201,202),validation)):
        workloads = records(SOURCE/f"traces/{split}-workload.jsonl")
        for seed in seeds:
            cases = [QueryCase(**row) for row in workloads if row["query_id"].startswith(f"{seed}-")]
            print(f"Retracing {split} seed {seed}: {len(cases)} actual SDSS SQL queries", flush=True)
            _, _, rows = execute(database,runtime_dir,out/f"{split}-{seed}",cases,policy="P0",cache_pages=32)
            destination.extend(rows)
    metadata = read(runtime_dir/"page_map.json")
    models, fits, _ = fit_models(training,validation,metadata,runtime_dir,out,fingerprint(database),16)
    # Record the complete existing training/validation/test signatures. New
    # evaluation excludes all of them, not merely the newly fitted subset.
    seen = set()
    for split in ("training","validation","test"):
        seen.update(signature(r["template_id"],r["params"]) for r in records(SOURCE/f"traces/{split}-workload.jsonl"))
    gate = select_gate(models["P4-neural-semantics"],validation)
    transfers = {}
    for dataset in ("small","large"):
        path=BACKEND/f"runs/idle-neural-20261106/{dataset}-pilot-001/P4-neural-semantics/idle_neural.pt"
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload["input_dim"]!=models["P4-neural-semantics"].model.net[0].in_features or payload["config"]["hidden"]!=64:
            raise ValueError("Transfer architecture is incompatible")
        transfers[dataset]={"checkpoint":str(path),"sha256":sha(path),"source_database":payload["database_identity"],
            "scope":"Freeze neural state_dict; use only SDSS-training-derived candidate/Markov statistics and SDSS static metadata. No SDSS neural fitting for this transfer arm; not zero-shot."}
    validation_parity=0
    from idle_optimized import OptimizedIdleHybrid
    cached=OptimizedIdleHybrid(models["P4-neural-semantics"])
    history=[];session=None
    for row in validation:
        if row["session_id"]!=session:history=[];session=row["session_id"]
        if history:
            for excluded in ((), tuple(history[-1]["pages"][-4:])):
                a=models["P4-neural-semantics"].predict(history,exclude_pages=excluded)
                b=cached.predict(history,exclude_pages=excluded)
                if [p["page"] for p in a["pages"]]!=[p["page"] for p in b["pages"]]:raise ValueError("SDSS cache changed ranking")
                if any(abs(p["neural_markov_score"]-q["neural_markov_score"])>1e-6 for p,q in zip(a["pages"],b["pages"])):raise ValueError("SDSS cache changed score")
                validation_parity+=1
        history.append(cached.state(row["template_id"],row["pages"]));history=history[-2:]
    plan={"schema_version":1,"dataset":"sdss","database":str(database),"database_identity":fingerprint(database),
        "database_sha256":sha(database),"runtime_model_dir":str(runtime_dir),"source_experiment":str(SOURCE),
        "data_scope":"Real SDSS DR7 ordered subset: 49,999 PhotoObj and 50,000 SpecObj rows, 15,790,080 bytes. Generated SQLite query templates; not original SeLeP logs or a SeLeP reproduction.",
        "core_source_sha256":source_hash(),"training_queries":len(training),"validation_queries":len(validation),
        "fits":fits,"gate":gate,"transfers":transfers,"cached_parity_cases":validation_parity,
        "original_sdss_neural_sha256":sha(runtime_dir/"model_ranker.pt"),
        "excluded_evaluation_signatures":sorted(seen),"policies":LABELS,
        "selection":"Fixed inherited P4 architecture/objective/admission; new SDSS neural fitting and gate selection use only training/validation queries. No test-driven tuning.",
        "independence":"Dataset/architecture reproducibility and frozen-neural transfer with target statistics are separate questions. All source and target checkpoints remain separate."}
    write(out/"plan.json",plan)
    print(json.dumps({"trained":fits,"gate":gate,"parity_cases":validation_parity},indent=2),flush=True)
    if source_guard()!=original:raise ValueError("Original SDSS experiment changed")


def child(args):
    torch = configure()
    from idle_experiment import execute, fingerprint
    from idle_neural import IdleNeuralHybrid, early_miss_records
    from idle_markov import QueryMarkov
    from idle_optimized import OptimizedIdleHybrid
    from pagewise.runledger import source_hash
    from pagewise.workloads import QueryCase
    plan=read(args.plan);root=args.plan.resolve().parent;spec=read(args.spec)
    database=Path(plan["database"]);runtime_dir=Path(plan["runtime_model_dir"])
    core=source_hash()
    if core!=plan["core_source_sha256"] or sha(database)!=plan["database_sha256"] or sha(runtime_dir/"model_ranker.pt")!=plan["original_sdss_neural_sha256"]:
        raise ValueError("Protected source/database/original model changed")
    weights=root/("P4-no-semantics" if args.policy=="P4-no-semantics" else "P4-neural-semantics")
    base=IdleNeuralHybrid.load(weights,read(runtime_dir/"page_map.json"),fingerprint(database))
    checkpoint_before=sha(weights/"idle_neural.pt");state_before=tensor_hash(base.model)
    query_model=None;transfer=None
    if args.policy.startswith("P3"):
        training=[]
        for seed in (101,102,103):training.extend(read(root/f"training-{seed}/query_records.json"))
        query_model=QueryMarkov(2).fit(early_miss_records(training) if args.policy=="P3-early" else training)
    elif args.policy.startswith("P4"):
        if args.policy.startswith("P4-transfer-"):
            source=args.policy.rsplit("-",1)[-1];transfer=plan["transfers"][source]
            state_before=apply_frozen_neural_weights(base,transfer["checkpoint"],transfer["sha256"])
        query_model=base
        if args.policy not in ("P4-original","P4-no-semantics"):
            gate=plan["gate"]["selected"] or {"margin":.25,"min_support":3}
            query_model=OptimizedIdleHybrid(base,selective=args.policy=="P4-selective" and plan["gate"]["selected"] is not None,
                admission_budget=min(16,max(1,spec["cache_pages"]//4)) if args.policy=="P4-selective" else None,
                min_support=gate["min_support"],margin=gate["margin"])
    execution_policy="P2" if args.policy=="P2-passive" else args.policy
    cases=[QueryCase(**row) for row in spec["cases"]]
    report, rows, _=execute(database,runtime_dir,args.out,cases,policy=execution_policy,query_model=query_model,
        cache_pages=spec["cache_pages"],gap_ms=spec["gap_ms"],delay_ms=spec["delay_ms"],gap_schedule=spec.get("gap_schedule"))
    proposals=read(args.out/"idle_proposals.json") if query_model else []
    report.update({"policy":args.policy,"label":LABELS[args.policy],"condition":spec["condition"],"seed":spec["seed"],
        "dataset":"sdss","output":str(args.out.resolve()),"is_idle":query_model is not None,
        "neural_calls":sum(p.get("neural_scored",False) for p in proposals) if query_model else report["stage_profile"].get("neural_forward_calls",0),
        "idle_jobs":len(proposals),"routes":dict(Counter(p.get("route","original_neural" if p.get("neural_scored") else "markov_or_empty") for p in proposals)),
        "cache":query_model.snapshot() if isinstance(query_model,OptimizedIdleHybrid) else None,
        "neural_stage_totals_ms":{k:sum(p.get("stage_ns",{}).get(k,0) for p in proposals)/1e6 for k in ("candidate_ns","feature_ns","forward_and_sort_ns")},
        "native_environment":{k:os.environ.get(k) for k in ("OMP_WAIT_POLICY","KMP_BLOCKTIME")},
        "transfer":transfer,"executed_neural_state_sha256":state_before})
    if sha(database)!=plan["database_sha256"] or source_hash()!=core or sha(weights/"idle_neural.pt")!=checkpoint_before or tensor_hash(base.model)!=state_before or sha(runtime_dir/"model_ranker.pt")!=plan["original_sdss_neural_sha256"]:
        raise ValueError("Database/source/checkpoint or active neural weights changed")
    if transfer and sha(transfer["checkpoint"])!=transfer["sha256"]:raise ValueError("Source neural checkpoint changed")
    report["protected_artifacts_unchanged"]=True
    write(args.out/"comparison.json",report)
    print(f"{args.policy}: mean {report['mean_ms']:.3f} ms; CPU {report['process_cpu_ms']:.1f} ms; useful {report['runtime']['useful_prefetches']}/{report['runtime']['pages_prefetched']}; neural calls {report['neural_calls']}",flush=True)


def run(args):
    from sdss_idle_workload import make_cases, signature, bucket
    plan=read(args.plan);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    original=source_guard();write(out/"original-sdss-before.json",original)
    sources=["sdss_idle_experiment.py","sdss_idle_workload.py","idle_neural.py","idle_neural_experiment.py","optimize_idle.py","idle_markov.py","idle_optimized.py","idle_runtime.py","idle_experiment.py"]
    snapshot=out/"implementation";snapshot.mkdir()
    source_hashes={name:sha(BACKEND/name) for name in sources}
    for name in sources:shutil.copyfile(BACKEND/name,snapshot/name)
    for folder in ("pagewise", "scripts"):
        (snapshot/folder).mkdir()
        for path in (BACKEND/folder).glob("*.py"):
            shutil.copyfile(path,snapshot/folder/path.name)
    seen=set(plan["excluded_evaluation_signatures"]);case_sets={}
    for seed in range(22001,22008):
        case_sets[seed]=make_cases(plan["database"],seed,args.queries,"test",seen)
    conditions=[("fresh-no-delay",seed,50,0,32) for seed in range(22001,22006)]
    conditions += [("fresh-delayed",seed,50,.2,32) for seed in range(22001,22006)]
    conditions += [("fresh-variable-gaps",22006,0,0,32),("fresh-zero-gap-large-cache",22007,0,0,128)]
    protocol={"plan":plan,"conditions":conditions,"queries_per_policy":args.queries,"source_sha256":source_hashes,
        "scope":"New held-out SDSS parameter combinations and fresh process for every policy. Source neural weights in transfer arms are frozen; only candidate/Markov statistics and static metadata are target-specific. Not zero-shot transfer.",
        "evaluation":"Five seeds per no-delay/artificial-delay condition. Same queries paired across service conditions. Additional variable-gap and zero-gap/larger-cache stress cases. Every child reads the full database for SHA256 before timing: warm-OS-cache evaluation, not a cold physical-disk benchmark.",
        "waiting":"P2 and P4-original inherit native waiting settings; all other arms set PASSIVE and KMP_BLOCKTIME=0 before numerical imports."}
    write(out/"protocol.json",protocol)
    results=[]
    for number,(condition,seed,gap,delay,cache) in enumerate(conditions):
        cases=case_sets[seed]
        if any(bucket(signature(c.template_id,c.params)) not in (8,9) or signature(c.template_id,c.params) in plan["excluded_evaluation_signatures"] for c in cases):raise ValueError("Held-out parameter split violated")
        target=out/f"condition-{number:02d}";target.mkdir()
        spec={"condition":condition,"seed":seed,"cache_pages":cache,"gap_ms":gap,"delay_ms":delay,
            "gap_schedule":[(0,5,25,50,100)[i%5] for i in range(len(cases)-1)] if condition=="fresh-variable-gaps" else None,
            "cases":[c.serializable() for c in cases]}
        write(target/"spec.json",spec)
        order=list(LABELS);random.Random(seed+number).shuffle(order);expected=None
        print(f"{number+1}/{len(conditions)} SDSS {condition} seed={seed}",flush=True)
        for policy in order:
            env={**os.environ,"PYTHONIOENCODING":"utf-8"}
            if policy not in ("P2","P4-original"):env.update(OMP_WAIT_POLICY="PASSIVE",KMP_BLOCKTIME="0")
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"child","--plan",str(args.plan.resolve()),"--spec",str(target/"spec.json"),"--policy",policy,"--out",str(target/policy)]
            completed=subprocess.run(cmd,cwd=BACKEND,env=env,capture_output=True,text=True)
            print(completed.stdout,end="",flush=True)
            if completed.returncode:raise RuntimeError(completed.stderr)
            report=read(target/policy/"comparison.json")
            with (target/policy/"queries.csv").open(newline="",encoding="utf-8") as f:
                values=[(r["query_id"],r["rows"],r["result_sha256"]) for r in csv.DictReader(f)]
            if expected is not None and values!=expected:raise ValueError("SQL results differ across policies")
            expected=values;results.append(report)
            write(out/"results.json",{"protocol":protocol,"results":results})
    if original!=source_guard() or source_hashes!={name:sha(BACKEND/name) for name in sources}:raise ValueError("Original experiment or implementation changed")
    write(out/"verification.json",{"trials":len(results),"query_executions":sum(r["queries"] for r in results),
        "original_sdss_artifacts_unchanged":True,"experiment_source_unchanged":True,"sql_results_identical":True,
        "fresh_disjoint_test_parameters":True,"weights_unchanged_during_evaluation":True})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","child"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--policy",choices=tuple(LABELS))
    parser.add_argument("--queries",type=int,default=60)
    args=parser.parse_args()
    if args.action in ("run","child") and args.plan is None:parser.error("--plan is required")
    if args.action=="child" and (args.spec is None or args.policy is None):parser.error("--spec and --policy are required")
    if args.queries<6:parser.error("Use at least six queries to cover all templates")
    {"prepare":prepare,"run":run,"child":child}[args.action](args)
