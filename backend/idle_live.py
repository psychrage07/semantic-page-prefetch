"""Local-only real SQL demonstration of the separately trained idle hybrid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from optimize_idle import BACKEND, PILOTS, LABELS, launch, read, sha, write

ROOT = BACKEND / "runs/idle-optimization-20261114"


def assets(dataset="small"):
    if dataset not in ("small", "large"):
        raise ValueError("Choose the small or large database")
    plan_path = ROOT / f"{dataset}-selection-001/optimization-plan.json"
    if not plan_path.is_file():
        return {"available":False,"reason":"Prepare the local idle optimization plan and matching P4 checkpoint first."}
    plan=read(plan_path)
    files=[Path(plan["database"]),Path(plan["model_dir"])/"page_map.json",
           Path(plan["model_dir"])/"candidate_index.json",Path(plan["model_dir"])/"model_ranker.pt",
           Path(plan["weights"])/"idle_neural.pt",Path(plan["weights"])/"query_prior.json"]
    if any(not path.is_file() for path in files):
        return {"available":False,"reason":"This demonstration requires its matching local database, original model and idle-model artifacts."}
    if sha(Path(plan["weights"])/"idle_neural.pt")!=plan["checkpoint_sha256"]:
        return {"available":False,"reason":"Idle neural checkpoint checksum mismatch."}
    return {"available":True,"dataset":dataset,"plan":str(plan_path),"checkpoint_sha256":plan["checkpoint_sha256"],
            "model":"Optimized idle semantic neural hybrid","training":"Saved query-level P4 weights; no retraining in this run"}


def run(args):
    evidence=assets(args.dataset)
    if not evidence["available"]: raise ValueError(evidence["reason"])
    from idle_experiment import make_cases, workload_pool
    plan_path=Path(evidence["plan"]);plan=read(plan_path)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/"idle-result.json").exists(): raise ValueError("Choose a new result directory")
    database=Path(plan["database"])
    cases=make_cases(workload_pool(database,args.seed),args.queries,args.seed)
    spec={"condition":"live-no-injected-delay","seed":args.seed,"cache_pages":32,"gap_ms":50,
          "delay_ms":0,"gap_schedule":None,"cases":[c.serializable() for c in cases]}
    write(out/"spec.json",spec)
    policies=["P0","P2","P3","P4-original","P4-cached","P4-selective"]
    order=policies.copy();random.Random(args.seed).shuffle(order)
    print("Loading saved semantic neural weights and matching local database.",flush=True)
    print("Running matched SQL workloads with 50 ms controlled pauses and no added storage delay.",flush=True)
    results=[];expected=None
    import csv
    for number,policy in enumerate(order,1):
        print(f"Policy {number}/{len(order)} started.",flush=True)
        row=launch(policy,plan_path,out/"spec.json",out/policy)
        with (out/policy/"queries.csv").open(newline="",encoding="utf-8") as handle:
            hashes=[(r["query_id"],r["rows"],r["result_sha256"]) for r in csv.DictReader(handle)]
        if expected is not None and hashes!=expected:raise ValueError("SQL results differ between policies")
        expected=hashes
        row["neural_scoring_calls"]=row.get("stage_profile",{}).get("neural_forward_calls",0) if policy=="P2" else row["neural_jobs"]
        results.append(row)
    result={"schema_version":1,"artifact_type":"idle_hybrid_live","dataset":args.dataset,"queries_per_policy":args.queries,
        "seed":args.seed,"gap_ms":50,"injected_delay_ms":0,"checkpoint_sha256":plan["checkpoint_sha256"],
        "rows":results,"sql_results_identical":True,"scope":"Fresh local runtime demonstration with saved query-level weights and controlled pauses; OS cache not reset; separate from original held-out study.",
        "gate":plan["selected_gate"],"complete_artifacts":str(out)}
    write(out/"idle-result.json",result)
    print("Experiment complete. SQL values matched; actual results and traces saved.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--dataset",choices=("small","large"),default="small")
    parser.add_argument("--queries",type=int,default=20)
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    if not 10<=args.queries<=100:parser.error("Choose 10 to 100 queries per policy")
    run(args)
