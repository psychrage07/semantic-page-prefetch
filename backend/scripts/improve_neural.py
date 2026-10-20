"""Reproducible validation-only neural screening on existing or freshly traced data.

Example (from backend):
  .venv/Scripts/python scripts/improve_neural.py --trace-dir runs/.../trace-seed101 --out runs/.../screen

Screening never evaluates the test partition. A selected configuration is
frozen with weights and training-only normalization before later evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np
import torch

from pagewise.dataset import load_jsonl
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex, page_map_digest
from pagewise.offline import build_splits, load_cases
from pagewise.ranking import (RankerConfig, build_groups, evaluate_ranker, fit_parameter_scales,
                              train_ranker)
from pagewise.runledger import record_run, source_hash


CONFIGS = [
    RankerConfig("a_bce", variant="a", objective="bce"),
    RankerConfig("b_bce", variant="b", objective="bce"),
    RankerConfig("b_bce_sampled", variant="b", objective="sampled_bce"),
    RankerConfig("b_listwise", variant="b"),
    RankerConfig("b_structured_listwise", variant="b", structural=True),
    RankerConfig("b2_bce", objective="bce"),
    RankerConfig("b2_bce_sampled", objective="sampled_bce"),
    RankerConfig("b3_bce", variant="b3", objective="bce"),
    RankerConfig("b2_listwise"),
    RankerConfig("b3_listwise", variant="b3"),
    RankerConfig("b2_pairwise", objective="pairwise"),
    RankerConfig("hybrid_no_semantics", prior_weight=1.0, no_semantics=True),
    RankerConfig("hybrid_semantics", prior_weight=1.0),
    RankerConfig("hybrid_structured", variant="b3", prior_weight=1.0, structural=True),
    RankerConfig("hybrid_miss_only", variant="b3", prior_weight=1.0, structural=True, miss_only=True),
    RankerConfig("hybrid_structured_h128", variant="b3", prior_weight=1.0, structural=True, hidden=128),
    RankerConfig("hybrid_structured_prior025", variant="b3", prior_weight=0.25, structural=True),
    RankerConfig("hybrid_structured_lr003", variant="b3", prior_weight=1.0, structural=True, learning_rate=0.003),
]


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != "per_query_top1_pct"}


def candidate_metrics(sequences, index):
    per_query, total, correct = [], 0, 0
    for seq in sequences:
        pages = seq["pages"]
        values = [pages[i] in index.candidates(pages[max(0, i - 8):i]) for i in range(1, len(pages))]
        if values:
            per_query.append(float(np.mean(values) * 100))
            total += len(values); correct += sum(values)
    return {"recall_micro_pct": 100 * correct / max(1, total),
            "recall_macro_pct": float(np.mean(per_query)), "transitions": total}


def screen(trace_dir: Path, out: Path, *, seed: int = 101, epochs: int = 20,
           max_per_query: int = 48, splits=("params", "templates"), names=None,
           generators=("legacy", "markov2_first", "balanced"), sizes=(16, 32, 64, 128), database="",
           train_workload_seeds=None, validation_workload_seeds=None, static_metadata_path=""):
    import dataclasses
    out.mkdir(parents=True, exist_ok=True)
    sequences = load_jsonl(trace_dir / "sequences.jsonl")
    cases = load_cases(trace_dir / "workload.jsonl")
    page_map = json.loads((trace_dir / "page_map.json").read_text(encoding="utf-8"))
    results = {"evidence_class": "development validation; not confirmatory", "seed": seed,
               "source_sha256": source_hash(),
               "trace_sha256": hashlib.sha256((trace_dir / "sequences.jsonl").read_bytes()).hexdigest(),
               "test_partition_evaluated": False, "splits": {}}
    configs = [dataclasses.replace(c, epochs=epochs) for c in CONFIGS if names is None or c.name in names]
    for split in splits:
        train, val, test, info = build_splits(sequences, cases, kind=split, seed=7)
        if train_workload_seeds:
            train = [s for s in train if s.get("workload_seed") in train_workload_seeds]
        if validation_workload_seeds:
            val = [s for s in val if s.get("workload_seed") in validation_workload_seeds]
        if not train or not val:
            raise ValueError(f"Empty training/validation partition for {split}")
        proposals = []
        for strategy in generators:
            for size in sizes:
                index = CandidateIndex.build(train, size, strategy=strategy, page_map=page_map, layout_database_path=str(database))
                proposals.append({"strategy": strategy, "size": size, **candidate_metrics(val, index)})
        # Prefer inexpensive generators within half a point of the best recall.
        best_recall = max(p["recall_macro_pct"] for p in proposals)
        eligible = [p for p in proposals if p["recall_macro_pct"] >= best_recall - 0.5]
        chosen = min(eligible, key=lambda p: (p["size"], -p["recall_macro_pct"]))
        index = CandidateIndex.build(train, chosen["size"], strategy=chosen["strategy"], page_map=page_map, layout_database_path=str(database))
        index.page_map_sha256 = page_map_digest(page_map)
        index.static_metadata_path = str(static_metadata_path)
        counts = ML.build_training_transition_counts(train, max_candidates=index.max_candidates)
        scales = fit_parameter_scales(train, cases)
        root = out / split
        root.mkdir(exist_ok=True)
        (root / "candidate_index.json").write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
        (root / "page_map.json").write_text(json.dumps(page_map), encoding="utf-8")
        (root / "partition.json").write_text(json.dumps({"info": info,
            "train": [s["query_id"] for s in train], "validation": [s["query_id"] for s in val],
            "test": [s["query_id"] for s in test]}, indent=2), encoding="utf-8")
        if static_metadata_path:
            from pagewise.static_metadata import StaticMetadata
            feature_metadata = StaticMetadata(static_metadata_path)
        else:
            feature_metadata = page_map
        selected = {"candidate_screen": proposals, "selected_generator": chosen, "partition": info,
                    "train_queries": len(train), "validation_queries": len(val), "reserved_test_queries": len(test),
                    "models": {}, "selection_metric": "validation macro Top-1"}
        results["splits"][split] = selected
        print(f"{split}: {len(train)} train / {len(val)} val; candidates={chosen}", flush=True)
        cache = {}
        for config in configs:
            started = time.monotonic()
            print(f"START {split}/{config.name}", flush=True)
            key = (config.variant, config.structural, config.no_semantics)
            # Share feature arrays across objectives but bound memory to one schema.
            if key not in cache:
                training = validation = None
                cache.clear()
                cache[key] = (build_groups(train, index, feature_metadata, counts, config, cases, scales,
                                          max_per_query=max_per_query),
                              build_groups(val, index, feature_metadata, counts, config, cases, scales,
                                           max_per_query=max_per_query))
            training, validation = cache[key]
            try:
                model, fit = train_ranker(training, validation, config, seed=seed)
                measured = evaluate_ranker(model, validation, config)
                payload = {"config": asdict(config), "state_dict": model.state_dict(),
                           "input_dim": training["x"].shape[-1], "normalization_max_page": index.max_page,
                           "counts": counts, "parameter_scales": scales, "fit": fit,
                           "source_sha256": results["source_sha256"]}
                torch.save(payload, root / f"{config.name}.pt")
                row = {"validation": compact(measured), "fit": fit, "seconds": time.monotonic() - started}
                selected["models"][config.name] = row
                record_run(out / "run_log.jsonl", {"split": split, **asdict(config)}, status="completed", seed=seed,
                           outputs=[str(root / f"{config.name}.pt")], metrics=compact(measured))
                print(f"DONE {split}/{config.name}: Top1={measured['top1_macro_pct']:.3f}% "
                      f"Top5={measured['top5_micro_pct']:.3f}% best_epoch={fit['best_epoch']} "
                      f"seconds={row['seconds']:.1f}", flush=True)
            except Exception as exc:
                record_run(out / "run_log.jsonl", {"split": split, **asdict(config)}, status="failed", seed=seed,
                           reason=repr(exc))
                raise
            (out / "screen.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        # The choice is frozen before any test partition is evaluated.
        winner = max(selected["models"], key=lambda n: selected["models"][n]["validation"]["top1_macro_pct"])
        selected["selected_model"] = winner
        selected["frozen_checkpoint_sha256"] = hashlib.sha256((root / f"{winner}.pt").read_bytes()).hexdigest()
        print(f"FROZEN {split}: {winner}", flush=True)
        (out / "screen.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-per-query", type=int, default=48)
    parser.add_argument("--splits", default="params,templates")
    parser.add_argument("--models", default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--generators", default="legacy,markov2_first,balanced")
    parser.add_argument("--candidate-sizes", default="16,32,64,128")
    parser.add_argument("--train-workload-seeds", default=None)
    parser.add_argument("--validation-workload-seeds", default=None)
    parser.add_argument("--static-metadata", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    screen(args.trace_dir, args.out, seed=args.seed, epochs=args.epochs, max_per_query=args.max_per_query,
           splits=tuple(args.splits.split(",")), names=args.models.split(",") if args.models else None,
           generators=tuple(args.generators.split(",")), sizes=tuple(int(s) for s in args.candidate_sizes.split(",")), database=args.db.resolve() if args.db else "",
           train_workload_seeds=[int(s) for s in args.train_workload_seeds.split(",")] if args.train_workload_seeds else None,
           validation_workload_seeds=[int(s) for s in args.validation_workload_seeds.split(",")] if args.validation_workload_seeds else None,
           static_metadata_path=args.static_metadata.resolve() if args.static_metadata else "")


if __name__ == "__main__":
    main()
