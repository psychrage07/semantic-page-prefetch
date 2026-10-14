from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_jsonl
from .models import CandidateIndex, load_model, normalize_semantic_ablation_groups, score_candidates


def evaluate_predictors(test_sequences_path: str | Path, page_map_path: str | Path,
                        candidate_index_path: str | Path, model_a_path: str | Path,
                        model_b_path: str | Path, output_dir: str | Path, *, top_k: int = 5,
                        semantic_ablations: dict[str, list[str]] | None = None,
                        retrained_ablation_dir: str | Path | None = None) -> dict[str, Any]:
    sequences = load_jsonl(test_sequences_path)
    page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    index = CandidateIndex.from_json(json.loads(Path(candidate_index_path).read_text(encoding="utf-8")))
    model_a, sem_a = load_model(model_a_path)
    model_b, sem_b = load_model(model_b_path)
    max_page = index.max_page
    semantic_ablations = semantic_ablations or {}
    normalized_ablations = {
        label: sorted(normalize_semantic_ablation_groups(groups))
        for label, groups in semantic_ablations.items()
    }
    retrained_models: dict[str, tuple[Any, bool, list[str]]] = {}
    if retrained_ablation_dir is not None:
        model_dir = Path(retrained_ablation_dir)
        for checkpoint in sorted(model_dir.glob("model_b_retrained_*.pt")):
            label = checkpoint.stem.removeprefix("model_b_retrained_")
            model, semantic = load_model(checkpoint)
            disabled_groups = sorted(normalize_semantic_ablation_groups(getattr(model, "disabled_semantic_groups", [])))
            retrained_models[label] = (model, semantic, disabled_groups)

    policy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_rows: list[dict[str, Any]] = []
    for seq in sequences:
        pages = [int(p) for p in seq.get("pages", [])]
        if len(pages) < 2:
            continue
        base_context = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"), "plan_class": seq.get("plan_class")}
        for i in range(1, len(pages)):
            history = pages[:i][-8:]
            target = pages[i]
            context = {**base_context, "position_in_query": min(i, 512) / 512.0}
            candidates = index.candidates(history)
            is_recall = target in candidates
            predictions: dict[str, list[tuple[int, float]]] = {}
            predictions["model_a"] = score_candidates(model_a, history, candidates, context, page_map, max_page, sem_a)
            predictions["model_b"] = score_candidates(model_b, history, candidates, context, page_map, max_page, sem_b)
            for label, groups in normalized_ablations.items():
                predictions[f"model_b_ablate_{label}"] = score_candidates(
                    model_b, history, candidates, context, page_map, max_page, sem_b,
                    disabled_semantic_groups=groups,
                )
            for label, (model, semantic, groups) in retrained_models.items():
                predictions[f"model_b_retrained_{label}"] = score_candidates(
                    model, history, candidates, context, page_map, max_page, semantic,
                    disabled_semantic_groups=groups,
                )

            current = history[-1]
            # Transition lists and their order are derived exclusively from the
            # training executions and serialized in candidate_index.json.
            markov1 = index.successors.get(current, [])[:top_k]
            markov2 = index.second_order.get(f"{history[-2]}:{current}", [])[:top_k] if len(history) > 1 else []
            if not markov2:
                markov2 = markov1
            predictions["markov1"] = [(p, 1.0 / rank) for rank, p in enumerate(markov1, start=1)]
            predictions["markov2"] = [(p, 1.0 / rank) for rank, p in enumerate(markov2, start=1)]
            if len(history) > 1 and index.second_order.get(f"{history[-2]}:{current}"):
                predictions["hybrid"] = [(p, 1.0 / rank) for rank, p in enumerate(markov2, start=1)]
            else:
                predictions["hybrid"] = predictions["model_b"]
            predictions["sequential"] = [(current + 1, 1.0)]
            predictions["oracle"] = [(target, 1.0)]
            predictions["no_prefetch"] = []

            for policy, ranked in predictions.items():
                # No-prefetch is a runtime policy, not a predictor; do not invent
                # a zero prediction accuracy for a method that makes no prediction.
                if policy == "no_prefetch":
                    continue
                ranked_pages = [p for p, _ in ranked]
                top1 = bool(ranked_pages and ranked_pages[0] == target)
                topk_hit = target in ranked_pages[:top_k]
                candidate_hit = is_recall if policy == "model_a" or policy.startswith("model_b") else None
                policy_rows[policy].append({
                    "query_id": seq.get("query_id"), "template_id": seq.get("template_id"),
                    "workload_family": seq.get("workload_family"), "target_page": target,
                    "top1": top1, "topk": topk_hit, "candidate_recall": candidate_hit,
                    "candidate_count": len(candidates), "history_last_page": current,
                })
                if policy == "model_a" or policy.startswith("model_b"):
                    prediction_rows.append({
                        "query_id": seq.get("query_id"), "template_id": seq.get("template_id"),
                        "workload_family": seq.get("workload_family"), "history": history,
                        "target_page": target, "policy": policy,
                        "candidate_recall": is_recall,
                        "top_predictions": [{"page": p, "score": round(float(score), 6)} for p, score in ranked[:top_k]],
                    })

    summary: dict[str, Any] = {
        "dataset": "observed VFS page-request sequences",
        "note": "Prediction metrics are measured on held-out query executions. They do not by themselves establish runtime speedup.",
        "test_executions": len(sequences), "top_k": top_k,
        "semantic_ablations": normalized_ablations,
        "retrained_semantic_ablations": {label: groups for label, (_, _, groups) in retrained_models.items()},
        "policies": {},
    }
    rows_out: list[dict[str, Any]] = []
    for policy, rows in policy_rows.items():
        n = len(rows)
        m = {
            "policy": policy,
            "evaluated_next_page_predictions": n,
            "top1_accuracy_pct": round(100 * sum(r["top1"] for r in rows) / n, 3) if n else None,
            f"top{top_k}_accuracy_pct": round(100 * sum(r["topk"] for r in rows) / n, 3) if n else None,
        }
        recalls = [r["candidate_recall"] for r in rows if r["candidate_recall"] is not None]
        if recalls:
            m["candidate_recall_pct"] = round(100 * sum(recalls) / len(recalls), 3)
        summary["policies"][policy] = m
        rows_out.append(m)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictor_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in prediction_rows) + ("\n" if prediction_rows else ""), encoding="utf-8")
    import csv
    with (out / "predictor_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else ["policy"])
        writer.writeheader()
        writer.writerows(rows_out)
    return summary
