"""Measured next-page accuracy for an interactive SQL run, after timing ends."""
import csv
import json
from pathlib import Path

MAX_PER_QUERY = 512


def positions(page_count: int, cap: int = MAX_PER_QUERY) -> list[int]:
    count = max(0, page_count - 1)
    if count <= cap:
        return list(range(1, page_count))
    return [1 + i * (count - 1) // (cap - 1) for i in range(cap)]


def evaluate(out: Path, request: dict, item: dict, order: list[str]) -> dict:
    from pagewise.dataset import build_sequences
    from pagewise.models import CandidateIndex, HISTORY_LENGTH, load_model, score_candidates
    from live_evaluation import metrics
    from workspace import write_json

    reference = "none" if "none" in order else order[0]
    model_dir = out / "model"
    sequences = build_sequences(out / reference / "trace.jsonl", model_dir / "page_map.json", out / "accuracy-sequences.jsonl")
    total = sum(max(0, len(seq["pages"]) - 1) for seq in sequences)
    result = {"available": bool(total), "reference_policy": reference,
              "eligible_transitions": total, "max_transitions_per_query": MAX_PER_QUERY,
              "sampled": any(len(seq["pages"]) - 1 > MAX_PER_QUERY for seq in sequences),
              "scope": "Accuracy on this SQL workload; values may overlap training. Separate from the held-out research study.",
              "policies": {}}
    if not total:
        result["reason"] = "No eligible next-page transitions were observed for this SQL query."
        write_json(out / "prediction-accuracy.json", result)
        return result
    index = CandidateIndex.from_json(json.loads((model_dir / "candidate_index.json").read_text(encoding="utf-8")))
    page_map = json.loads((model_dir / "page_map.json").read_text(encoding="utf-8"))
    own = "model_ranker" if "model_ranker" in item["policies"] else "model_b"
    result["primary_policy"] = own
    compared = list(dict.fromkeys([own, "markov1", "markov2", *[p for p in request["policies"] if p != "none"]]))
    models = {}
    if own == "model_ranker":
        from pagewise.ranking import NeuralPredictor
        if item["scope"] == "research":
            ranker = NeuralPredictor(item["models"] / "model_ranker.pt", index, page_map)
        else:
            ranker_dir = model_dir / "selected-hybrid"
            ranker_index = CandidateIndex.from_json(json.loads((ranker_dir / "candidate_index.json").read_text(encoding="utf-8")))
            ranker = NeuralPredictor(ranker_dir / "model_ranker.pt", ranker_index, page_map)
    if item["scope"] == "small":
        models["model_b"] = load_model(model_dir / "model_b.pt")
        if "model_a" in compared:
            models["model_a"] = load_model(model_dir / "model_a.pt")
    rows = {policy: [] for policy in compared}
    context_config = json.loads((model_dir / "context-policy.json").read_text(encoding="utf-8")) if "context" in compared else None
    if context_config:
        result["context_scope"] = context_config["scope"] + " Offline accuracy evaluates the routing decision at each sampled transition; it does not simulate worker admission."
    print(f"Evaluating prediction accuracy from actual {reference} traces; up to {MAX_PER_QUERY} evenly spaced transitions per query.", flush=True)
    for seq in sequences:
        for position in positions(len(seq["pages"])):
            history = seq["pages"][max(0, position - HISTORY_LENGTH):position]
            target = seq["pages"][position]
            context = {"template_id": seq["template_id"], "workload_family": seq["workload_family"],
                       "plan_class": seq["plan_class"], "params": request["params"],
                       "position_in_query": min(position, 512) / 512.0}
            first = index.successors.get(history[-1], [])
            second = index.second_order.get(f"{history[-2]}:{history[-1]}", []) if len(history) > 1 else []
            candidates = index.candidates(history)
            ranked = {"markov1": first, "markov2": second or first, "sequential": [history[-1] + 1]}
            if own == "model_ranker":
                ranked[own] = [int(page) for page, _ in ranker.rank(history, context)]
            if item["scope"] == "small":
                for policy, (model, semantic) in models.items():
                    ranked[policy] = [int(page) for page, _ in score_candidates(model, history, candidates, context, page_map, index.max_page, semantic)]
                ranked["hybrid"] = second if second else ranked["model_b"]
            if context_config:
                from selective_context import context_action
                context_index = ranker.index if own == "model_ranker" else index
                action, _ = context_action(context_index, history, seq["workload_family"], context_config)
                context_first = context_index.successors.get(history[-1], [])
                context_second = context_index.second_order.get(f"{history[-2]}:{history[-1]}", []) if len(history) > 1 else []
                ranked["context"] = context_second if action == "markov2" else context_first if action == "markov1" else ranked.get("model_ranker", []) if action == "neural" else []
            for policy in compared:
                predictions = ranked[policy]
                rows[policy].append({"query_id": seq["query_id"], "family": seq["workload_family"],
                    "top1": bool(predictions and predictions[0] == target), "top5": target in predictions[:5],
                    "candidate_hit": target in predictions, "skipped": not predictions})
    result["policies"] = {policy: metrics(values) for policy, values in rows.items()}
    if "context" in rows:
        result["policies"]["context"]["skipped_transitions"] = sum(row["skipped"] for row in rows["context"])
    result["scored_transitions"] = len(rows[own])
    write_json(out / "prediction-accuracy.json", result)
    with (out / "prediction-accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "queries", "scored_transitions", "top1_macro_pct", "top1_micro_pct", "top5_micro_pct", "sampled"])
        for policy, row in result["policies"].items():
            writer.writerow([policy, row["queries"], row["transitions"], row["top1_macro_pct"], row["top1_micro_pct"], row["top5_micro_pct"], result["sampled"]])
    print(f"Accuracy evaluated on {result['scored_transitions']}/{total} actual transitions. "
          f"{own} Top-1={result['policies'][own]['top1_micro_pct']:.2f}%", flush=True)
    return result
