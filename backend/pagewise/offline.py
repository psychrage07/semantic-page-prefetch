"""Offline next-page prediction study for v2 (M2).

Builds two held-out splits by *query execution*:

- ``params``: executions are grouped by a hash of their SQL parameters into
  buckets; whole buckets are held out, so test executions use parameter
  combinations absent from training (templates overlap).
- ``templates``: whole query templates are held out, so test executions use
  query templates absent from training and tuning.

All splits are by execution, never by individual reads.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_jsonl
from .models import (
    HISTORY_LENGTH,
    SEMANTIC_FEATURE_GROUPS,
    CandidateIndex,
    _fit,
    _training_pairs,
    page_map_digest,
    score_candidates,
)

POLICIES = ("markov1", "markov2", "markov_ctx", "model_a", "model_b", "model_b_table")
ALL_SEMANTIC_GROUPS = tuple(SEMANTIC_FEATURE_GROUPS)


def load_cases(workload_path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in load_jsonl(workload_path)}


def _param_bucket(sequence: dict[str, Any], cases: dict[str, dict[str, Any]], buckets: int = 6) -> int:
    case = cases.get(str(sequence.get("query_id")), {})
    params = case.get("params", sequence.get("query_id"))
    key = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def build_splits(sequences: list[dict[str, Any]], cases: dict[str, dict[str, Any]], *,
                 kind: str, seed: int = 7) -> tuple[list, list, list, dict[str, Any]]:
    """Return (train, val, test, info) for the requested split kind."""
    execs = [s for s in sequences if len(s.get("pages", [])) >= 2]
    rng = random.Random(seed)
    if kind == "templates":
        templates = sorted({str(s.get("template_id")) for s in execs})
        shuffled = list(templates)
        rng.shuffle(shuffled)
        n_test = max(1, len(shuffled) // 3)
        n_val = 1 if len(shuffled) >= 4 else 0
        test_t = set(shuffled[:n_test])
        val_t = set(shuffled[n_test:n_test + n_val])
        train = [s for s in execs if str(s.get("template_id")) not in test_t | val_t]
        val = [s for s in execs if str(s.get("template_id")) in val_t]
        test = [s for s in execs if str(s.get("template_id")) in test_t]
        return train, val, test, {"kind": kind, "test_templates": sorted(test_t), "val_templates": sorted(val_t)}
    if kind != "params":
        raise ValueError(f"Unknown split kind {kind!r}; expected 'params' or 'templates'")
    bucket_ids = list(range(6))
    rng.shuffle(bucket_ids)
    train_buckets = set(bucket_ids[:4])
    val_buckets = {bucket_ids[4]}
    test_buckets = {bucket_ids[5]}
    grouped: dict[int, list] = defaultdict(list)
    for s in execs:
        grouped[_param_bucket(s, cases)].append(s)
    train = [s for b in train_buckets for s in grouped.get(b, [])]
    val = [s for b in val_buckets for s in grouped.get(b, [])]
    test = [s for b in test_buckets for s in grouped.get(b, [])]
    # Guarantee all three splits are non-empty for small inputs.
    if not test and train:
        test = train[len(train) * 3 // 4:]
        train = train[:len(train) * 3 // 4]
    if not val and train:
        val = train[len(train) * 3 // 4:]
        train = train[:len(train) * 3 // 4]
    return train, val, test, {"kind": kind, "train_buckets": sorted(train_buckets),
                              "val_buckets": sorted(val_buckets), "test_buckets": sorted(test_buckets)}


def train_on_split(train: list, val: list, page_map: dict[str, Any], seed: int, *,
                   epochs: int = 20, hidden: int = 128, max_candidates: int = 32) -> dict[str, Any]:
    index = CandidateIndex.build(train, max_candidates=max_candidates)
    index.page_map_sha256 = page_map_digest(page_map)
    empty = lambda dim: (np.empty((0, dim), np.float32), np.empty((0,), np.float32))  # noqa: E731
    xa, ya = _training_pairs(train, index, page_map, False, random.Random(seed))
    xav, yav = _training_pairs(val, index, page_map, False, random.Random(seed + 1)) if val else empty(xa.shape[1])
    model_a, _ = _fit(xa, ya, xav, yav, seed=seed, epochs=epochs, hidden=hidden)
    xb, yb = _training_pairs(train, index, page_map, True, random.Random(seed + 2))
    xbv, ybv = _training_pairs(val, index, page_map, True, random.Random(seed + 3)) if val else empty(xb.shape[1])
    model_b, _ = _fit(xb, yb, xbv, ybv, seed=seed + 2, epochs=epochs, hidden=hidden)
    return {"index": index, "model_a": model_a, "model_b": model_b}


def _ctx_key(page: int, context: dict[str, Any], page_map: dict[str, Any]) -> tuple:
    meta = page_map.get(str(page), {}) or {}
    return (int(page), str(context.get("template_id")), str(context.get("plan_class")),
            int(meta.get("depth", 0) or 0))


def build_context_tables(train: list, trained: dict[str, Any], page_map: dict[str, Any], *,
                         top_k: int = 8) -> tuple[dict, dict]:
    """Build markov_ctx (counts) and model_b_table (Model B's ranked successors) on the same key."""
    index = trained["index"]
    model_b = trained["model_b"]
    markov_ctx: dict[tuple, Counter] = defaultdict(Counter)
    model_b_table: dict[tuple, Counter] = defaultdict(Counter)
    for seq in train:
        pages = [int(p) for p in seq["pages"]]
        base = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"),
                "plan_class": seq.get("plan_class")}
        for i in range(1, len(pages)):
            history = pages[:i][-HISTORY_LENGTH:]
            target = pages[i]
            last = pages[i - 1]
            context = {**base, "position_in_query": min(i, 512) / 512.0}
            key = _ctx_key(last, context, page_map)
            markov_ctx[key][target] += 1
            candidates = index.candidates(history)
            if candidates:
                ranked = score_candidates(model_b, history, candidates, context, page_map, index.max_page, True)
                for rank, (page, _) in enumerate(ranked[:top_k]):
                    model_b_table[key][page] += top_k - rank
    return markov_ctx, model_b_table


def _predict(history: list[int], context: dict[str, Any], key: tuple, index: CandidateIndex,
             trained: dict[str, Any], page_map: dict[str, Any], markov_ctx: dict, model_b_table: dict,
             top_k: int) -> dict[str, list[int]]:
    last = history[-1]
    markov1 = list(index.successors.get(last, []))
    markov2 = list(index.second_order.get(f"{history[-2]}:{last}", [])) if len(history) > 1 else []
    if not markov2:
        markov2 = markov1
    ctx_hit = markov_ctx.get(key)
    table_hit = model_b_table.get(key)
    ctx_pred = [p for p, _ in ctx_hit.most_common(top_k)] if ctx_hit else markov1
    tbl_pred = [p for p, _ in table_hit.most_common(top_k)] if table_hit else markov1
    candidates = index.candidates(history)
    ranked_a = score_candidates(trained["model_a"], history, candidates, context, page_map, index.max_page, False)
    ranked_b = score_candidates(trained["model_b"], history, candidates, context, page_map, index.max_page, True)
    return {
        "markov1": markov1,
        "markov2": markov2,
        "markov_ctx": ctx_pred,
        "model_a": [p for p, _ in ranked_a],
        "model_b": [p for p, _ in ranked_b],
        "model_b_table": tbl_pred,
    }


def evaluate_split(test: list, trained: dict[str, Any], page_map: dict[str, Any], markov_ctx: dict,
                   model_b_table: dict, *, top_k: int = 8) -> dict[str, Any]:
    """Per-execution accuracy per policy plus table coverage."""
    per_exec: dict[str, dict[str, list[bool]]] = {}
    coverage = {"markov_ctx": 0, "model_b_table": 0}
    transitions = 0
    index = trained["index"]
    for seq in test:
        pages = [int(p) for p in seq["pages"]]
        if len(pages) < 2:
            continue
        base = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"),
                "plan_class": seq.get("plan_class")}
        hits: dict[str, list[bool]] = {policy: [] for policy in POLICIES}
        for i in range(1, len(pages)):
            history = pages[:i][-HISTORY_LENGTH:]
            target = pages[i]
            context = {**base, "position_in_query": min(i, 512) / 512.0}
            key = _ctx_key(pages[i - 1], context, page_map)
            transitions += 1
            if key in markov_ctx:
                coverage["markov_ctx"] += 1
            if key in model_b_table:
                coverage["model_b_table"] += 1
            preds = _predict(history, context, key, index, trained, page_map, markov_ctx, model_b_table, top_k)
            for policy in POLICIES:
                ranked = preds[policy]
                hits[policy].append(bool(ranked and ranked[0] == target))
        per_exec[str(seq.get("query_id"))] = hits
    return {"per_exec": per_exec, "coverage": coverage, "transitions": transitions}


def paired_bootstrap_ci(diffs: list[float], *, seed: int = 11, iterations: int = 10_000) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    if len(diffs) == 1:
        return (round(diffs[0], 4), round(diffs[0], 4))
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    means.sort()
    return (round(means[int(0.025 * (iterations - 1))], 4), round(means[int(0.975 * (iterations - 1))], 4))


def _exec_accuracy(per_exec: dict[str, dict[str, list[bool]]], policy: str) -> dict[str, float]:
    out = {}
    for query_id, hits in per_exec.items():
        values = hits.get(policy, [])
        if values:
            out[query_id] = 100.0 * sum(values) / len(values)
    return out


def _paired_diffs(per_exec: dict[str, dict[str, list[bool]]], candidate: str, baseline: str) -> list[float]:
    cand = _exec_accuracy(per_exec, candidate)
    base = _exec_accuracy(per_exec, baseline)
    return [cand[q] - base[q] for q in sorted(set(cand) & set(base))]


def _combine_outcomes(*outcomes: str) -> str:
    if all(o == "supported" for o in outcomes):
        return "supported"
    if any(o == "not supported" for o in outcomes):
        return "not supported"
    return "inconclusive"


def outcome(ci_low: float, ci_high: float, *, greater: bool = True) -> str:
    if ci_low > 0 and greater:
        return "supported"
    if ci_high < 0 and not greater:
        return "supported"
    if ci_low <= 0 <= ci_high:
        return "inconclusive"
    return "not supported"


def run_offline_study(sequences_path: str | Path, workload_path: str | Path, page_map_path: str | Path,
                      output_dir: str | Path, *, seeds: list[int] | None = None, epochs: int = 20,
                      hidden: int = 128, max_candidates: int = 32, top_k: int = 8) -> dict[str, Any]:
    seeds = seeds or [7, 17, 27, 37, 47]
    sequences = load_jsonl(sequences_path)
    cases = load_cases(workload_path)
    page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_exec_by_split: dict[str, dict[str, dict[str, list[bool]]]] = {"params": {}, "templates": {}}
    for split_kind in ("params", "templates"):
        train, val, test, info = build_splits(sequences, cases, kind=split_kind, seed=7)
        for seed in seeds:
            trained = train_on_split(train, val, page_map, seed, epochs=epochs, hidden=hidden,
                                     max_candidates=max_candidates)
            markov_ctx, model_b_table = build_context_tables(train, trained, page_map, top_k=top_k)
            evaluation = evaluate_split(test, trained, page_map, markov_ctx, model_b_table, top_k=top_k)
            for query_id, hits in evaluation["per_exec"].items():
                per_exec_by_split[split_kind][f"{seed}:{query_id}"] = hits
            mean_top1 = {}
            for policy in POLICIES:
                accs = _exec_accuracy(evaluation["per_exec"], policy)
                mean_top1[policy] = round(statistics.mean(accs.values()), 4) if accs else None
                rows.append({
                    "split": split_kind, "seed": seed, "policy": policy,
                    "top1_accuracy_pct": mean_top1[policy],
                    "executions": len(evaluation["per_exec"]), "transitions": evaluation["transitions"],
                    "markov_ctx_coverage_pct": round(100 * evaluation["coverage"]["markov_ctx"] / max(1, evaluation["transitions"]), 3),
                    "model_b_table_coverage_pct": round(100 * evaluation["coverage"]["model_b_table"] / max(1, evaluation["transitions"]), 3),
                })
            print(f"[{split_kind}] seed {seed} done: model_a top1={mean_top1['model_a']} "
                  f"model_b top1={mean_top1['model_b']} markov_ctx={mean_top1['markov_ctx']}")

    # Aggregate per split/policy and compute paired CIs (pooled across seeds).
    aggregated: dict[str, Any] = {}
    comparisons = [
        ("H1", "model_b", "model_a"),
        ("H2a", "model_b", "markov_ctx"),
        ("H2b", "model_b", "markov2"),
    ]
    for split_kind in ("params", "templates"):
        pooled = per_exec_by_split[split_kind]
        agg = {}
        for policy in POLICIES:
            accs = _exec_accuracy(pooled, policy)
            agg[policy] = round(statistics.mean(accs.values()), 4) if accs else None
        aggregated[split_kind] = {"mean_top1_accuracy_pct": agg, "comparisons": {}}
        for label, cand, base in comparisons:
            diffs = _paired_diffs(pooled, cand, base)
            lo, hi = paired_bootstrap_ci(diffs, seed=100 + len(label) + len(split_kind))
            aggregated[split_kind]["comparisons"][label] = {
                "candidate": cand, "baseline": base, "paired_units": len(diffs),
                "mean_delta_pct": round(statistics.mean(diffs), 4) if diffs else None,
                "ci95_low": lo, "ci95_high": hi,
                "outcome": outcome(lo, hi),
            }

    hypotheses = {
        "H1": {"statement": "On held-out query parameters, Model B has higher top-1 accuracy than Model A.",
               "scope": "params split",
               "outcome": aggregated["params"]["comparisons"]["H1"]["outcome"],
               "evidence": aggregated["params"]["comparisons"]["H1"]},
        "H2": {"statement": "Model B beats markov_ctx and markov2 in top-1 accuracy.",
               "scope": "params split",
               "outcome": _combine_outcomes(aggregated["params"]["comparisons"]["H2a"]["outcome"],
                                             aggregated["params"]["comparisons"]["H2b"]["outcome"]),
               "evidence": {"vs_markov_ctx": aggregated["params"]["comparisons"]["H2a"],
                            "vs_markov2": aggregated["params"]["comparisons"]["H2b"]}},
        "H5_offline": {"statement": "H1 still holds on held-out query templates.",
                       "scope": "templates split",
                       "outcome": aggregated["templates"]["comparisons"]["H1"]["outcome"],
                       "evidence": aggregated["templates"]["comparisons"]["H1"]},
    }
    report = {
        "schema_version": 1, "seeds": seeds, "epochs": epochs, "hidden": hidden,
        "candidate_count": max_candidates, "top_k": top_k,
        "splits": {k: v for k, v in aggregated.items()},
        "hypotheses": hypotheses,
        "note": "Accuracy is on held-out query executions. Param-bucket and template splits are by execution, never by read.",
        "rows": rows,
    }
    (out / "offline_accuracy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import csv
    with (out / "offline_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return report


def run_ablation(sequences_path: str | Path, workload_path: str | Path, page_map_path: str | Path,
                 output_dir: str | Path, *, seeds: list[int] | None = None, split_kind: str = "params",
                 epochs: int = 20, hidden: int = 128, max_candidates: int = 32) -> dict[str, Any]:
    """Feature ablation for Model B: only one semantic group, and no page metadata."""
    seeds = seeds or [7, 17, 27]
    sequences = load_jsonl(sequences_path)
    cases = load_cases(workload_path)
    page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    train, val, test, _ = build_splits(sequences, cases, kind=split_kind, seed=7)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    variants: dict[str, list[str]] = {"full": []}
    for group in ALL_SEMANTIC_GROUPS:
        variants[f"only_{group}"] = [g for g in ALL_SEMANTIC_GROUPS if g != group]
    variants["no_page_metadata"] = ["page_type", "depth", "object"]

    rows = []
    for label, disabled in variants.items():
        top1s = []
        for seed in seeds:
            index = CandidateIndex.build(train, max_candidates=max_candidates)
            index.page_map_sha256 = page_map_digest(page_map)
            xb, yb = _training_pairs(train, index, page_map, True, random.Random(seed + 2),
                                     disabled_semantic_groups=disabled)
            xbv, ybv = (_training_pairs(val, index, page_map, True, random.Random(seed + 3),
                                        disabled_semantic_groups=disabled)
                        if val else (np.empty((0, xb.shape[1]), np.float32), np.empty((0,), np.float32)))
            model_b, _ = _fit(xb, yb, xbv, ybv, seed=seed + 2, epochs=epochs, hidden=hidden)
            hits = []
            for seq in test:
                pages = [int(p) for p in seq["pages"]]
                if len(pages) < 2:
                    continue
                base = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"),
                        "plan_class": seq.get("plan_class")}
                for i in range(1, len(pages)):
                    history = pages[:i][-HISTORY_LENGTH:]
                    context = {**base, "position_in_query": min(i, 512) / 512.0}
                    ranked = score_candidates(model_b, history, index.candidates(history), context,
                                              page_map, index.max_page, True, disabled_semantic_groups=disabled)
                    hits.append(bool(ranked and ranked[0][0] == pages[i]))
            top1s.append(100.0 * sum(hits) / len(hits) if hits else 0.0)
        rows.append({"variant": label, "disabled_groups": disabled,
                     "mean_top1_accuracy_pct": round(statistics.mean(top1s), 4),
                     "seeds": len(top1s)})
        print(f"[ablation] {label}: {rows[-1]['mean_top1_accuracy_pct']}%")
    report = {"schema_version": 1, "split": split_kind, "seeds": seeds, "rows": rows}
    (out / "model_b_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
