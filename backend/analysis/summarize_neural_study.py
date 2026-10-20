"""Postprocess measured artifacts without changing frozen predictor sources.

Run from backend: python analysis/summarize_neural_study.py simulator|report
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from pagewise.sim import Costs
from pagewise.v21 import diagnose_sim_cells, run_headroom_v21
from pagewise.runledger import source_hash, verify_ledger

STUDY = BACKEND / "runs/neural-improvement-20261020"
LABELS = {
    "selected": "Selected neural hybrid", "a_bce": "Model A / binary loss",
    "b_bce": "Model B / binary loss", "b2_bce": "B2 / binary loss",
    "b_listwise": "Model B / listwise loss", "b_structured_listwise": "Model B / structured listwise",
    "hybrid_no_semantics": "Hybrid without semantics", "markov1": "Markov-1",
    "markov2": "Markov-2", "markov_ctx": "Contextual Markov",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def simulator(study):
    original = read(study / "calibration/calibration.json")
    corrected = copy.deepcopy(original)
    fit = original["costs"]["fit_configs"]
    x = np.asarray([[original["entries"][str(c)]["cache_hits"],
                     original["entries"][str(c)]["cache_misses"]] for c in fit], dtype=float)
    y = np.asarray([original["entries"][str(c)]["total_ms"] * 1000 for c in fit])
    candidates = [np.zeros(2), np.array([0., max(0., x[:, 1] @ y / (x[:, 1] @ x[:, 1]))]),
                  np.array([max(0., x[:, 0] @ y / (x[:, 0] @ x[:, 0])), 0.])]
    unconstrained = np.linalg.lstsq(x, y, rcond=None)[0]
    if np.all(unconstrained >= 0):
        candidates.append(unconstrained)
    fitted = min(candidates, key=lambda b: float(np.square(y - x @ b).sum()))
    corrected["raw_unconstrained_costs"] = original["costs"]
    corrected["costs"].update(query_overhead_us=0., cpu_hit_us=float(fitted[0]), cpu_miss_us=float(fitted[1]),
                              residual_sum_sq_us2=float(np.square(y - x @ fitted).sum()))
    corrected["constraint_amendment"] = {
        "reason": "Raw least-squares calibration produced a physically invalid negative hit cost.",
        "method": "Exact two-variable nonnegative least squares by active-set enumeration; fit caches 32,128,512 only.",
        "identifiability": "Read and query counts are identical across runs: three costs are not separately identifiable. Query overhead fixed to zero; CPU costs are effective aggregate costs.",
        "held_out_caches": [64, 256], "original_measurements_retained": True,
    }
    output = study / "simulator"
    if output.exists():
        raise ValueError("Simulator outputs already exist; preserve attempts")
    write(output / "calibration-nonnegative.json", corrected)
    runs = [study / "calibration" / f"{s}-c{c}" for s in ("none", "null") for c in (32, 64, 128, 256, 512)]
    runs.append(study / "calibration/null-w4-c32")
    diagnosis = diagnose_sim_cells(output / "calibration-nonnegative.json", runs, out_dir=output / "validation")
    for row in diagnosis["cells"]:
        row["role"] = "held-out cache configuration" if row.get("cache_pages") in (64, 256) else "calibration configuration"
    diagnosis["scope"] = "Zero-delay none/null validation only. Concurrent device service and exact neural predictions are not validated by these cells."
    write(output / "validation/simulator_cell_diagnosis.json", diagnosis)
    costs = Costs(**{k: v for k, v in corrected["costs"].items() if k in Costs.__dataclass_fields__})
    delays = [corrected["sleep_measured_ms"][str(d)] for d in (0.1, 0.2, 0.5, 1.0)]
    report = run_headroom_v21(study / "calibration/ref-c32/demand_reference.json", output / "headroom",
                             costs=costs, caches=[32, 64, 128, 256, 512], delays=delays, channels=[1, 4, 16],
                             degree=2, extra_metadata={
                                 "requested_delay_ms": [0.1, 0.2, 0.5, 1.0], "achieved_delay_ms": delays,
                                 "validation_scope": diagnosis["scope"],
                                 "calibration_amendment": corrected["constraint_amendment"],
                                 "confirmation_status": "exploratory timing-model sensitivity; not confirmed physical-device headroom",
                             })
    print(json.dumps({"decision": report["decision"], "cells": len(report["cells"]),
                      "reopen": len(report["reopen_configurations"]),
                      "validation_cells": len(diagnosis["cells"]), "validation_failures": len(diagnosis["cells_missing_target"])}), flush=True)


def report(study):
    final = read(study / "confirmation/evaluation.json")
    amended = read(study / "amended/evaluation.json")
    runtime = read(study / "runtime-confirmation/runtime.json")
    declaration = read(study / "frozen-final/freeze.json")
    diagnosis = read(study / "simulator/validation/simulator_cell_diagnosis.json")
    headroom = read(study / "simulator/headroom/headroom_v21.json")
    if source_hash() != declaration["source_sha256"]:
        raise ValueError("Predictor sources differ from frozen source")
    assert final["mode"] == "confirmatory" and final["seeds"] == declaration["confirmation_seeds"]
    assert len(runtime["runs"]) == 60 and len(headroom["cells"]) == 60
    for split, spec in declaration["splits"].items():
        for model in spec["models"].values():
            assert hashlib.sha256((study / "frozen-final" / split / model["checkpoint"]).read_bytes()).hexdigest() == model["sha256"]
    checks = {name: verify_ledger(study / name / "run_log.jsonl") for name in ("confirmation", "amended", "runtime-confirmation")}
    if not all(check["ok"] for check in checks.values()):
        raise ValueError("Broken experiment ledger")
    screen = read(study / "assembled-screen/screen.json")
    development = {split: {name: {"validation_macro_top1_pct": row["validation"]["top1_macro_pct"],
                                  "config": row["fit"]["config"], "best_epoch": row["fit"]["best_epoch"]}
                           for name, row in result["models"].items()} for split, result in screen["splits"].items()}
    telemetry = []
    for scenario in runtime["scenarios"]:
        stats = [r["runtime_stats"] for r in runtime["runs"] if r["scenario"] == scenario["label"] and r["policy"] == "selected_ranker"]
        total = lambda key: sum(row.get(key, 0) or 0 for row in stats)
        pages, useful, requests = total("pages_prefetched"), total("useful_prefetches"), total("demand_requests")
        completed, stale = total("completed_jobs"), total("stale_jobs_dropped")
        actual_scored = completed - stale
        telemetry.append({"scenario": scenario["label"], "pages_prefetched": pages, "useful_pages": useful,
                          "unused_pages": total("total_wasted_prefetches"),
                          "useful_prefetch_pct": 100 * useful / pages if pages else None,
                          "useful_pages_per_demand_request_pct": 100 * useful / requests if requests else None,
                          "actual_scored_jobs": actual_scored, "stale_jobs_dropped": stale,
                          "queue_jobs_dropped": total("dropped_jobs"),
                          "inference_ms_per_scored_job": total("inference_ms_total") / actual_scored if actual_scored else None,
                          "queue_wait_ms_per_dequeued_job": total("queue_wait_ms_total") / completed if completed else None,
                          "demand_misses_while_prefetch_in_flight": total("demand_misses_while_prefetch_in_flight"),
                          "foreground_device_wait_ms": sum(r.get("storage_device", {}).get("demand_wait_ms_total", 0) for r in stats)})
    payload = {"schema_version": 1, "study": "Pagewise neural improvement / 2026-10-20", "evidence": "fresh confirmatory workloads",
               "database_bytes": 8615698432, "seeds": final["seeds"], "labels": LABELS,
               "source_sha256": declaration["source_sha256"], "freeze_sha256": final["freeze_sha256"],
               "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "models": {s: v["selected_model"] for s, v in declaration["splits"].items()},
               "splits": {s: v["summary"] for s, v in final["splits"].items()},
               "comparisons": final["family"]["members"], "runtime": runtime["scenarios"],
               "development": development, "runtime_telemetry": telemetry,
               "runtime_comparisons": runtime["comparisons"]["members"],
               "simulator": {"decision": headroom["decision"], "cells": len(headroom["cells"]),
                             "reopened_cells": len(headroom["reopen_configurations"]),
                             "validation_cells": len(diagnosis["cells"]), "validation_failures": len(diagnosis["cells_missing_target"]),
                             "scope": diagnosis["scope"], "bound": headroom["headroom_bound"]},
               "scope": ["Macro Top-1 weights query executions equally; micro scores weight page transitions.",
                         "Same candidate generator and training partitions for neural controls; native Markov controls also reported.",
                         "Five independent workload seeds; inference is conditional on this fixed database and workload generator.",
                         "Historical v2 scores use a different corpus and are not a direct before/after comparison.",
                         "Latency uses real APSW query execution with artificial service delay; it does not measure physical storage performance.",
                         "Unseen-template outcomes and unsupported comparisons are retained."], "integrity": checks}
    out = study / "final"
    write(out / "results.json", payload)
    write(BACKEND.parent / "public/pagewise-neural-improvement.json", payload)
    param = payload["splits"]["params"]
    latency_outcome = ("The frozen neural policy was slower than no prefetching in every measured runtime scenario. Prediction improved; query latency did not."
                       if all(s["summary"]["selected_ranker"]["mean_ms"] > s["summary"]["none"]["mean_ms"] for s in runtime["scenarios"])
                       else "Runtime effects vary by scenario; see paired uncertainty before claiming a query speedup.")
    lines = ["# Pagewise neural improvement results", "",
             f"On fresh unseen parameters, semantic Model B improved from **{param['b_bce']['top1_macro_pct']:.2f}% to {param['b_structured_listwise']['top1_macro_pct']:.2f}%** macro Top-1 with structured features and listwise ranking. The selected neural–Markov hybrid reached **{param['selected']['top1_macro_pct']:.2f}%**, versus B2 **{param['b2_bce']['top1_macro_pct']:.2f}%** and native Markov-2 **{param['markov2']['top1_macro_pct']:.2f}%**. These are matched new-study results, not a comparison with the historical v2 corpus.",
             "", latency_outcome, "", "Fresh confirmation on the fixed 8.62 GB database; all eligible transitions evaluated without sampling.",
             f"Workload seeds: {final['seeds']}. Weights, candidates, calibration and comparison family were frozen before these traces were collected.",
             "", "## Prediction accuracy", "", "Macro Top-1 is the mean query accuracy. Candidate failures count as errors.", "",
             "| Split | Predictor | Macro Top-1 % | Micro Top-1 % | Micro Top-5 % | Candidate recall % |",
             "|---|---|---:|---:|---:|---:|"]
    for split, rows in payload["splits"].items():
        for name, row in rows.items():
            recall = f"{row['candidate_recall_pct']:.3f}" if "candidate_recall_pct" in row else "n/a"
            lines.append(f"| {split} | {LABELS.get(name,name)} | {row['top1_macro_pct']:.3f} | {row['top1_micro_pct']:.3f} | {row['top5_micro_pct']:.3f} | {recall} |")
    lines += ["", "![Fresh prediction accuracy on a shared percentage scale](figures/neural-improvement-prediction.png)",
              "", "## Paired effects", "", "Positive percentage-point effects favor the named candidate. 95% hierarchical-bootstrap intervals resample independent workload seeds and paired queries within each seed. Holm corrects the entire frozen comparison family.", "",
              "| Comparison | Delta pp | 95% CI | Holm p | Favorable support |", "|---|---:|---|---:|---|"]
    for name, row in payload["comparisons"].items():
        lines.append(f"| {name} | {row['mean']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['holm_adjusted_p']:.5f} | {row['holm_decision']} |")
    lines += ["", "## Development selection", "", "These scores are validation-only, sampled at most 16 transitions per query, and are not directly comparable to the full fresh-test scores. B3-based hybrids were screened but not promoted; their final confirmatory score is unavailable.", "",
              "| Split | Trial | Variant | Loss | Validation macro Top-1 % | Best epoch |", "|---|---|---|---|---:|---:|"]
    for split, rows in development.items():
        for name, row in rows.items():
            config = row["config"]
            lines.append(f"| {split} | {name} | {config['variant']} | {config['objective']} | {row['validation_macro_top1_pct']:.3f} | {row['best_epoch']} |")
    lines += ["", "## Runtime", "", "The policy was selected on development seeds 107–108, then frozen. Confirmation runs 60 queries per seed per policy, cache 512 pages, one prefetch worker. Settings: degree 1, confidence threshold 0.7, predict every fourth miss. Negative latency effects favor prefetching.",
              "The delay labels are requested artificial service times; the 0.2 ms sleep calibration achieved 0.5234 ms on this Windows host.", "",
              "| Scenario | Policy | Mean ms | p95 ms | Queries |", "|---|---|---:|---:|---:|"]
    for scenario in runtime["scenarios"]:
        for name, row in scenario["summary"].items():
            lines.append(f"| {scenario['label']} | {name} | {row['mean_ms']:.3f} | {row['p95_ms']:.3f} | {row['queries']} |")
    lines += ["", "| Runtime comparison | Delta ms | 95% CI | Holm p | Favorable support |", "|---|---:|---|---:|---|"]
    for name, row in payload["runtime_comparisons"].items():
        lines.append(f"| {name} | {row['mean']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['holm_adjusted_p']:.5f} | {row['holm_decision']} |")
    lines += ["", "![Measured runtime by declared scenario](figures/neural-improvement-runtime.png)",
              "", "### Runtime diagnosis", "", "Useful rate divides useful prefetched pages by pages prefetched. Demand coverage below divides useful prefetched pages by all demand requests, not reference misses. Unused pages include evicted and still-resident unused prefetches. Scoring time includes candidate generation/features/ranking and divides by actually scored jobs, excluding stale jobs dropped before scoring. Foreground query latency also includes request scheduling and contention.", "",
              "| Scenario | Prefetched | Useful | Useful % | Demand coverage % | Scoring ms/job | Queue wait ms/job | Stale drops |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in telemetry:
        fmt = lambda k: f"{row[k]:.3f}" if row[k] is not None else "n/a"
        lines.append(f"| {row['scenario']} | {row['pages_prefetched']} | {row['useful_pages']} | {fmt('useful_prefetch_pct')} | {fmt('useful_pages_per_demand_request_pct')} | {fmt('inference_ms_per_scored_job')} | {fmt('queue_wait_ms_per_dequeued_job')} | {row['stale_jobs_dropped']} |")
    lines += ["", "The JSON additionally retains queue drops, misses while prefetch was in flight and foreground device waiting. These are diagnostic counters, not a complete causal attribution of every latency regression. Exact timing-failure counts require an aligned reference-miss opportunity denominator; no offline timely-coverage claim is made."]
    lines += ["", "## Generalization and errors", "", "Held-out templates are an independent stress test. An epoch-zero selected residual checkpoint is a Markov fallback and provides no learned neural benefit.",
              "Candidate recall limits every candidate ranker. The following rows pool all selected-model queries by template, including candidate misses.", "",
              "| Split | Template | Queries | Transitions | Candidate misses % | Macro Top-1 % |", "|---|---|---:|---:|---:|---:|"]
    for split, result in final["splits"].items():
        grouped = {}
        for seed in result["by_seed"].values():
            for row in seed["selected_error_breakdown"]:
                grouped.setdefault(row["template_id"], []).append(row)
        for template, rows in sorted(grouped.items()):
            n = sum(r["transitions"] for r in rows)
            misses = 100 * sum(r["candidate_missing"] for r in rows) / n
            lines.append(f"| {split} | {template} | {len(rows)} | {n} | {misses:.3f} | {np.mean([r['top1_accuracy_pct'] for r in rows]):.3f} |")
    lines += ["", "## Simulator and evidence limits", "", f"The 60-cell grid returned **{headroom['decision']}**, with {len(headroom['reopen_configurations'])} reopened eligible concurrent cells.",
              "This is exploratory virtual-time sensitivity, with a lower bound on headroom in a restricted action family {none, oracle-miss degree 1..2}. It is not a global scheduling bound or physical-device result.",
              f"Zero-delay none/null validation: {len(diagnosis['cells'])} cells, {len(diagnosis['cells_missing_target'])} outside the 15% error target. Caches 64 and 256 were held out from CPU calibration.",
              "The raw fit produced a negative hit cost. Original measurements are preserved; a separately recorded nonnegative fit fixes per-query overhead at zero because the three raw costs are not identifiable. Concurrent device timing and exact neural replay remain unvalidated, so simulator results do not overturn measured runtime findings.",
              "Historical frozen manifest directories were absent locally (0 of 2 present); only the manifest file was available. Historical artifacts cannot be certified from that file alone.",
              "", "## Reproduction and integrity", "", f"Frozen declaration SHA-256: `{final['freeze_sha256']}`. Predictor source SHA-256: `{declaration['source_sha256']}`.",
              "The frozen source archive and checkpoints are in `backend/runs/neural-improvement-20261020/frozen-final/`. Complete prediction tables are in that study's `confirmation/`; reused historical seeds 911–915 are explicitly labeled amended evidence in `amended/`.",
              "Run from backend using `.venv/Scripts/python.exe`:", "", "```text",
              "scripts/evaluate_neural.py evaluate --frozen runs/neural-improvement-20261020/frozen-final --study runs/neural-improvement-20261020 --seeds 921,922,923,924,925 --out runs/neural-improvement-20261020/reproduction-confirmation --mode confirmatory",
              "scripts/benchmark_neural_runtime.py --db runs/large-branch-20261015/pagewise-large.sqlite --frozen runs/neural-improvement-20261020/frozen-final --out runs/neural-improvement-20261020/reproduction-runtime --phase confirmatory --policy runs/neural-improvement-20261020/runtime-pilot-optimized/policy.json --queries 60",
              "analysis/summarize_neural_study.py report", "```", "", "The collected trace and model artifacts are local ignored research outputs. This report and the compact dashboard export are tracked; a fresh clone needs the local database and frozen artifacts to reproduce this exact run.", "",
              "Validation: 106 backend tests passed before confirmation. Final frontend and integrity checks are recorded in NEURAL_IMPROVEMENT_PLAN.md."]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    (BACKEND / "NEURAL_IMPROVEMENT_RESULTS.md").write_text(text, encoding="utf-8")
    plots(payload, out)
    print(json.dumps({"report": str(out / "report.md"), "public_export": str(BACKEND.parent / "public/pagewise-neural-improvement.json"), "integrity": checks}), flush=True)


def plots(payload, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["b_bce", "b_structured_listwise", "b2_bce", "selected", "markov2"]
    (out / "figures").mkdir(exist_ok=True)
    (BACKEND / "figures").mkdir(exist_ok=True)
    upper = max(rows[n]["top1_macro_pct"] for rows in payload["splits"].values() for n in names) * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, (split, rows) in zip(axes, payload["splits"].items()):
        bars = ax.barh([LABELS[n] for n in names], [rows[n]["top1_macro_pct"] for n in names], color=["#91a7b8", "#2a887a", "#91a7b8", "#d09535", "#577b96"])
        ax.bar_label(bars, labels=[f"{rows[n]['top1_macro_pct']:.2f}%" for n in names], padding=4, fontsize=9)
        ax.set(xlabel="Macro Top-1 accuracy (%)", title=f"Held-out {split}", xlim=(0, upper))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Pagewise: fresh confirmation, fixed 8.62 GB database")
    fig.savefig(out / "prediction.png", dpi=180)
    shutil.copyfile(out / "prediction.png", out / "figures/neural-improvement-prediction.png")
    shutil.copyfile(out / "prediction.png", BACKEND / "figures/neural-improvement-prediction.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, policy in enumerate(("none", "markov2", "selected_ranker")):
        ax.bar(np.arange(4) + (i - 1) * .24, [s["summary"][policy]["mean_ms"] for s in payload["runtime"]], width=.24, label=policy)
    ax.set(xticks=np.arange(4), xticklabels=[s["label"] for s in payload["runtime"]], ylabel="Mean query latency (ms)", title="Measured APSW queries; service delays are artificial")
    ax.legend()
    fig.savefig(out / "runtime.png", dpi=180)
    shutil.copyfile(out / "runtime.png", out / "figures/neural-improvement-runtime.png")
    shutil.copyfile(out / "runtime.png", BACKEND / "figures/neural-improvement-runtime.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("simulator", "report"))
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    {"simulator": simulator, "report": report}[args.stage](args.study.resolve())
