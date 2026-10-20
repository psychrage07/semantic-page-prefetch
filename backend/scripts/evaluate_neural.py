"""Freeze validation-selected rankers, then evaluate independent held-out workloads.

Freeze before testing:
  python scripts/evaluate_neural.py freeze --screen ... --training-trace ... --out ...
Evaluate without retraining or changing settings:
  python scripts/evaluate_neural.py evaluate --frozen ... --study ... --seeds 107,108,109,110 --out ... --mode development
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import numpy as np
import torch

from pagewise.dataset import load_jsonl
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex
from pagewise.offline import load_cases, _param_bucket
from pagewise.ranking import RankerConfig, NeuralPredictor, build_groups, evaluate_ranker, fit_temperature
from pagewise.runledger import record_run, source_hash, verify_ledger
from pagewise.statistics import declare_family

BASELINES = ("b_bce", "b2_bce", "hybrid_no_semantics")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(screen: Path, training_trace: Path, out: Path):
    if out.exists():
        raise ValueError("Freeze directory already exists; preserve it and use a new path")
    out.mkdir(parents=True)
    report = json.loads((screen / "screen.json").read_text(encoding="utf-8"))
    seqs = load_jsonl(training_trace / "sequences.jsonl")
    cases = load_cases(training_trace / "workload.jsonl")
    declaration = {"source_sha256": source_hash(), "screen_sha256": sha(screen / "screen.json"),
                   "training_trace_sha256": sha(training_trace / "sequences.jsonl"),
                   "selection": "validation macro Top-1; test labels not used", "splits": {},
                   "comparison_family": [], "comparison_specs": [], "confirmation_seeds": [921, 922, 923, 924, 925]}
    for split, result in report["splits"].items():
        root, target = screen / split, out / split
        target.mkdir()
        partition = json.loads((root / "partition.json").read_text(encoding="utf-8"))
        validation_ids = set(partition["validation"])
        validation = [s for s in seqs if str(s["query_id"]) in validation_ids]
        training_ids = set(partition["train"])
        training = [s for s in seqs if str(s["query_id"]) in training_ids]
        contextual = ML.build_training_transition_counts(training)["contextual"]
        index = CandidateIndex.from_json(json.loads((root / "candidate_index.json").read_text(encoding="utf-8")))
        shutil.copyfile(root / "page_map.json", target / "page_map.json")
        shutil.copyfile(root / "partition.json", target / "partition.json")
        if index.static_metadata_path:
            metadata_target = out / "static-metadata.npz"
            if not metadata_target.exists():
                shutil.copyfile(index.static_metadata_path, metadata_target)
            index.static_metadata_path = str(metadata_target.resolve())
        (target / "candidate_index.json").write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
        page_map = json.loads((target / "page_map.json").read_text(encoding="utf-8"))
        names = dict(selected=result["selected_model"], **{n: n for n in BASELINES})
        # Give binary-loss baselines their best validation-selected negative
        # sampling recipe; do not inflate gains against an artificially weak control.
        for baseline in ("b_bce", "b2_bce"):
            alternatives = [n for n in (baseline, baseline + "_sampled") if n in result["models"]
                            and result["models"][n]["fit"]["best_epoch"] > 0]
            if alternatives:
                names[baseline] = max(alternatives, key=lambda n: result["models"][n]["validation"]["top1_macro_pct"])
        for name in ("a_bce", "b_listwise", "b_structured_listwise"):
            if name in result["models"]:
                names[name] = name
        declared = {"selected_model": result["selected_model"], "partition": partition["info"], "models": {}}
        declaration["splits"][split] = declared
        calibration_groups = {}
        for name, original in names.items():
            payload = torch.load(root / f"{original}.pt", map_location="cpu", weights_only=True)
            payload["counts"]["contextual"] = contextual
            predictor = NeuralPredictor(root / f"{original}.pt", index, page_map)
            key = (predictor.config.variant, predictor.config.structural, predictor.config.no_semantics)
            if key not in calibration_groups:
                calibration_groups[key] = build_groups(validation, index, predictor.page_map, predictor.counts,
                                                        predictor.config, cases, predictor.parameter_scales, max_per_query=16)
            groups = calibration_groups[key]
            payload["calibration"] = fit_temperature(predictor.model, groups, predictor.config)
            destination = target / ("model_ranker.pt" if name == "selected" else f"{name}.pt")
            torch.save(payload, destination)
            declared["models"][name] = {"checkpoint": destination.name, "sha256": sha(destination),
                                        "validation_selected_variant": original,
                                        "config": payload["config"], "calibration": payload["calibration"],
                                        "best_epoch": payload["fit"]["best_epoch"]}
            del groups
        for baseline in (*BASELINES, "markov2"):
            label = f"{split}:selected_vs_{baseline}"
            declaration["comparison_family"].append(label)
            declaration["comparison_specs"].append(dict(label=label, split=split, candidate="selected", baseline=baseline))
        for candidate, baseline in (("b_listwise", "b_bce"), ("b_structured_listwise", "b_bce"),
                                    ("b_structured_listwise", "b_listwise")):
            if candidate in names and baseline in names:
                label = f"{split}:{candidate}_vs_{baseline}"
                declaration["comparison_family"].append(label)
                declaration["comparison_specs"].append(dict(label=label, split=split, candidate=candidate, baseline=baseline))
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("pagewise", "scripts", "tests"):
            for path in sorted((BACKEND / directory).glob("*.py")):
                archive.write(path, path.relative_to(BACKEND))
    declaration["source_archive_sha256"] = sha(out / "source.zip")
    (out / "freeze.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    print(json.dumps({"frozen": str(out), "models": {s: r["selected_model"] for s, r in declaration["splits"].items()}}), flush=True)


def held_out(sequences, cases, split, declaration):
    if split == "templates":
        return [s for s in sequences if s.get("template_id") in declaration["test_templates"]]
    return [s for s in sequences if _param_bucket(s, cases) in declaration["test_buckets"]]


def evaluate_seed(frozen: Path, split: str, declaration: dict, trace: Path) -> dict:
    index = CandidateIndex.from_json(json.loads((frozen / split / "candidate_index.json").read_text(encoding="utf-8")))
    page_map = json.loads((frozen / split / "page_map.json").read_text(encoding="utf-8"))
    cases = load_cases(trace / "workload.jsonl")
    seqs = held_out(load_jsonl(trace / "sequences.jsonl"), cases, split, declaration["partition"])
    if not seqs:
        raise ValueError(f"No held-out queries in {trace}/{split}")
    predictors = {name: NeuralPredictor(frozen / split / spec["checkpoint"], index, page_map)
                  for name, spec in declaration["models"].items()}
    per_query = defaultdict(dict)
    totals = defaultdict(lambda: defaultdict(float))
    selected_details = []
    for start in range(0, len(seqs), 3):
        chunk = seqs[start:start + 3]
        feature_cache = {}
        for name, predictor in predictors.items():
            config = predictor.config
            key = (config.variant, config.structural, config.no_semantics)
            # Chunked construction keeps full, unsampled evaluation within RAM.
            if key not in feature_cache:
                if sum(group["x"].nbytes for group in feature_cache.values()) > 512_000_000:
                    feature_cache.clear()
                data = None
                feature_cache[key] = build_groups(chunk, index, predictor.page_map, predictor.counts, config,
                                                  cases, predictor.parameter_scales, max_per_query=None)
            data = feature_cache[key]
            measured = evaluate_ranker(predictor.model, data, config)
            n = measured["transitions"]
            per_query[name].update(measured["per_query_top1_pct"])
            totals[name]["transitions"] += n
            for metric in ("top1_micro_pct", "top5_micro_pct", "candidate_recall_pct"):
                totals[name][metric] += measured[metric] * n
            totals[name]["reference_miss_correct"] += (measured["reference_miss_top1_pct"] or 0) * measured["reference_miss_transitions"] / 100
            totals[name]["reference_miss_transitions"] += measured["reference_miss_transitions"]
            totals[name]["forward_us"] += measured["forward_us_per_prediction_batched"] * n
            if name == "selected":
                qids = np.asarray(data["query_id"])
                for baseline in ("markov1", "markov2", "markov_ctx"):
                    correct = data[f"{baseline}_top1"] == data["target_page"]
                    top5 = np.any(data[f"{baseline}_top5"] == data["target_page"][:, None], axis=1)
                    for q in dict.fromkeys(data["query_id"]):
                        per_query[baseline][q] = float(correct[qids == q].mean() * 100)
                    totals[baseline]["transitions"] += n
                    totals[baseline]["top1_micro_pct"] += correct.sum() * 100
                    totals[baseline]["top5_micro_pct"] += top5.sum() * 100
                for q, score in measured["per_query_top1_pct"].items():
                    mask = qids == q
                    selected_details.append({"query_id": q, "template_id": next(s.get("template_id") for s in chunk if str(s["query_id"]) == q),
                                             "transitions": int(mask.sum()), "candidate_missing": int((data["target"][mask] < 0).sum()),
                                             "top1_accuracy_pct": score})
        del feature_cache, data
    output = {}
    for name, queries in per_query.items():
        n = totals[name]["transitions"]
        row = {"top1_macro_pct": float(np.mean(list(queries.values()))),
               "top1_micro_pct": totals[name]["top1_micro_pct"] / n,
               "top5_micro_pct": totals[name]["top5_micro_pct"] / n,
               "transitions": int(n), "queries": len(queries), "per_query_top1_pct": queries}
        if name in predictors:
            recall = totals[name]["candidate_recall_pct"] / n
            row.update(candidate_recall_pct=recall, conditional_top1_pct=row["top1_micro_pct"] * 100 / recall if recall else None,
                       reference_miss_top1_pct=100 * totals[name]["reference_miss_correct"] / max(1, totals[name]["reference_miss_transitions"]),
                       forward_us_per_prediction_batched=totals[name]["forward_us"] / n)
        output[name] = row
    return {"models": output, "selected_error_breakdown": selected_details, "full_unsampled_evaluation": True}


def evaluate(frozen: Path, study: Path, seeds: list[int], out: Path, mode: str):
    declaration = json.loads((frozen / "freeze.json").read_text(encoding="utf-8"))
    if source_hash() != declaration["source_sha256"]:
        raise ValueError("Source changed after freezing; restore the frozen source before confirmation")
    if mode == "confirmatory" and seeds != declaration["confirmation_seeds"]:
        raise ValueError("Confirmatory seeds differ from the frozen declaration")
    if out.exists():
        raise ValueError("Evaluation output exists; preserve it and use a new directory")
    out.mkdir(parents=True)
    for split, spec in declaration["splits"].items():
        for model in spec["models"].values():
            if sha(frozen / split / model["checkpoint"]) != model["sha256"]:
                raise ValueError("Frozen checkpoint changed")
    report = {"mode": mode, "evidence_class": "offline held-out VFS page prediction, no latency claim",
              "freeze_sha256": sha(frozen / "freeze.json"), "source_sha256": source_hash(),
              "seeds": seeds, "splits": {}, "full_unsampled_evaluation": True}
    comparisons = []
    for split, spec in declaration["splits"].items():
        rows = {}
        for seed in seeds:
            started = time.monotonic()
            trace = study / f"trace-seed{seed}"
            result = evaluate_seed(frozen, split, spec, trace)
            rows[str(seed)] = result
            record_run(out / "run_log.jsonl", {"stage": "held_out", "split": split, "mode": mode,
                                               "freeze_sha256": report["freeze_sha256"], "trace_sha256": sha(trace / "sequences.jsonl")},
                       seed=seed, status="completed", metrics={n: {k: v for k, v in m.items() if k != "per_query_top1_pct"}
                                                               for n, m in result["models"].items()})
            print(f"{mode}/{split}/seed{seed}: selected={result['models']['selected']['top1_macro_pct']:.3f}% "
                  f"B2={result['models']['b2_bce']['top1_macro_pct']:.3f}% "
                  f"Markov2={result['models']['markov2']['top1_macro_pct']:.3f}% {time.monotonic()-started:.1f}s", flush=True)
            report["splits"][split] = {"by_seed": rows}
            (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = {}
        for name in next(iter(rows.values()))["models"]:
            values = [r["models"][name] for r in rows.values()]
            queries = [v for row in values for v in row["per_query_top1_pct"].values()]
            transitions = sum(row["transitions"] for row in values)
            summary[name] = {"top1_macro_pct": float(np.mean(queries)), "queries": len(queries), "transitions": transitions}
            for key in ("top1_micro_pct", "top5_micro_pct", "candidate_recall_pct"):
                if all(key in row for row in values):
                    summary[name][key] = sum(row[key] * row["transitions"] for row in values) / transitions
        for comparison in [c for c in declaration["comparison_specs"] if c["split"] == split]:
            candidate, baseline = comparison["candidate"], comparison["baseline"]
            sessions = {}
            for seed, result in rows.items():
                a = result["models"][candidate]["per_query_top1_pct"]
                b = result["models"][baseline]["per_query_top1_pct"]
                if a.keys() != b.keys():
                    raise ValueError("Unmatched paired queries")
                sessions[int(seed)] = [a[q] - b[q] for q in a]
            comparisons.append((comparison["label"], sessions))
        report["splits"][split]["summary"] = summary
    report["family"] = declare_family(comparisons, iterations=10_000, replication_unit="independently generated workload seed; paired query executions within seed")
    report["ledger_integrity"] = verify_ledger(out / "run_log.jsonl")
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(report, out / "report.md")
    return report


def write_report(report: dict, destination: Path):
    lines = ["# Pagewise neural improvement evaluation", "", f"Evidence: {report['mode']}; {report['evidence_class']}.",
             "", f"Workload seeds: {report['seeds']}. All eligible transitions evaluated without sampling.",
             "Top-1 macro gives every query equal weight; Top-1/Top-5 micro count transitions.",
             "", "| Split | Predictor | Top-1 macro % | Top-1 micro % | Top-5 micro % |", "|---|---|---:|---:|---:|"]
    for split, result in report["splits"].items():
        for name, metrics in result["summary"].items():
            lines.append(f"| {split} | {name} | {metrics['top1_macro_pct']:.3f} | {metrics['top1_micro_pct']:.3f} | {metrics['top5_micro_pct']:.3f} |")
    lines += ["", "## Paired comparisons", "", f"{len(report['family']['members'])} comparisons were declared before testing. Positive effects favor the named candidate.",
              "Clusters are independently generated workload seeds, not model initialization seeds.", "",
              "| Comparison | Delta pp | 95% CI | Holm p | Supported after correction |", "|---|---:|---|---:|---|"]
    for label, result in report["family"]["members"].items():
        lines.append(f"| {label} | {result['mean']:.3f} | [{result['ci95_low']:.3f}, {result['ci95_high']:.3f}] | {result['holm_adjusted_p']:.5f} | {result['holm_decision']} |")
    lines += ["", "## Scope", "", "The models and calibration were frozen before evaluating these seeds. No query-latency improvement follows from these offline results.",
              "B-tree proposals parse the complete child list from a static page; metadata comes from the entire database, independent of trace membership.",
              "VFS reads do not establish physical storage I/O. Batched forward time excludes feature construction and scheduling; use runtime telemetry for end-to-end cost.",
              "Historical v2 results use a different training corpus and remain separate.",
              "", f"Frozen declaration SHA-256: `{report['freeze_sha256']}`."]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--screen", required=True, type=Path)
    f.add_argument("--training-trace", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("--frozen", required=True, type=Path)
    e.add_argument("--study", required=True, type=Path)
    e.add_argument("--seeds", required=True)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--mode", choices=("development", "amended", "confirmatory"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "freeze":
        freeze(args.screen, args.training_trace, args.out)
    else:
        evaluate(args.frozen, args.study, [int(s) for s in args.seeds.split(",")], args.out, args.mode)


if __name__ == "__main__":
    main()
