"""Evaluate the frozen semantic predictor on fresh baseline VFS sequences.

Offline scoring is separate from the timed runtime comparison. Every eligible
transition is retained, including targets absent from the training candidate set.
"""
from collections import defaultdict
import json
from pathlib import Path
import statistics


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No eligible next-page transitions to evaluate")
    queries = defaultdict(list)
    families = defaultdict(list)
    for row in rows:
        queries[row["query_id"]].append(row)
        families[row["family"]].append(row)
    return {
        "queries": len(queries), "transitions": len(rows),
        "top1_macro_pct": statistics.mean(statistics.mean(r["top1"] for r in group) * 100 for group in queries.values()),
        "top1_micro_pct": statistics.mean(r["top1"] for r in rows) * 100,
        "top5_micro_pct": statistics.mean(r["top5"] for r in rows) * 100,
        "candidate_recall_pct": statistics.mean(r["candidate_hit"] for r in rows) * 100,
        "by_family": {family: {"transitions": len(group), "top1_micro_pct": statistics.mean(r["top1"] for r in group) * 100}
                      for family, group in families.items()},
    }


def evaluate(out: Path):
    import torch
    from pagewise.dataset import build_sequences
    from pagewise.models import CandidateIndex, HISTORY_LENGTH
    from pagewise.ranking import NeuralPredictor
    from live_hybrid import FROZEN, sha

    torch.set_num_threads(1)
    plan = json.loads((out / "live-plan.json").read_text(encoding="utf-8"))
    checkpoint = FROZEN / "params/model_ranker.pt"
    if sha(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before evaluation")
    index = CandidateIndex.from_json(json.loads((out / "model/candidate_index.json").read_text(encoding="utf-8")))
    page_map = json.loads((out / "model/page_map.json").read_text(encoding="utf-8"))
    predictor = NeuralPredictor(checkpoint, index, page_map)
    if predictor.config.no_semantics:
        raise ValueError("Live evaluation requires the selected semantic checkpoint")
    policies = {"model_ranker": [], "markov1": [], "markov2": []}
    samples = []
    for seed in plan["seeds"]:
        source = out / f"seed-{seed}" / "none"
        sequences = build_sequences(source / "trace.jsonl", out / "model/page_map.json", source / "evaluation-sequences.jsonl")
        for seq in sequences:
            pages = seq["pages"]
            query_id = f"{seed}:{seq['query_id']}"
            for position in range(1, len(pages)):
                history = pages[max(0, position - HISTORY_LENGTH):position]
                context = {**seq, "position_in_query": min(position, 512) / 512.0}
                ranked = predictor.rank(history, context)
                target = pages[position]
                first = index.successors.get(history[-1], [])
                second = index.second_order.get(f"{history[-2]}:{history[-1]}", []) if len(history) >= 2 else []
                rankings = {"model_ranker": [int(page) for page, _ in ranked], "markov1": first, "markov2": second or first}
                for policy, predictions in rankings.items():
                    policies[policy].append({"query_id": query_id, "family": seq["workload_family"],
                        "top1": bool(predictions and predictions[0] == target), "top5": target in predictions[:5],
                        "candidate_hit": target in predictions})
                # Capped examples are aligned to the exact history/target, never
                # joined by query ID alone. Aggregate scores use ALL transitions.
                if position <= 4 and len(samples) < 96:
                    def metadata(page):
                        row = predictor.page_map.get(str(page), {})
                        return {key: row.get(key) for key in ("object", "page_type", "depth")}
                    samples.append({"query_id": query_id, "seed": seed, "position": position,
                        "template_id": seq["template_id"], "family": seq["workload_family"],
                        "plan_class": seq["plan_class"], "history": history, "target_page": target,
                        "target_metadata": metadata(target), "top1_correct": bool(ranked and ranked[0][0] == target),
                        "top5_correct": target in rankings["model_ranker"][:5],
                        "candidates": [{"page": int(page), "score": float(score), **metadata(page)} for page, score in ranked[:5]]})
        print(f"Evaluated actual neural predictions for workload {seed}", flush=True)
    result = {"checkpoint_sha256": plan["checkpoint_sha256"], "semantics_enabled": True,
        "source": "Fresh no-prefetch VFS sequences; all eligible next-page transitions, unsampled",
        "scope": "Fresh workload evaluation; values can overlap training. Not the held-out final study.",
        "policies": {policy: metrics(rows) for policy, rows in policies.items()}, "samples": samples}
    if sha(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({policy: {k: v for k, v in row.items() if k != "by_family"} for policy, row in result["policies"].items()}), flush=True)
