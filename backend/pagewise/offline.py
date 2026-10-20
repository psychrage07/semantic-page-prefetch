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

from . import ml_variants as ML

POLICIES = ("markov1", "markov2", "markov_ctx", "model_a", "model_b", "model_b_table")
ALL_SEMANTIC_GROUPS = tuple(SEMANTIC_FEATURE_GROUPS)
# v2.1 declared policy set for the B2/B3 study (Model B PRESERVED as a
# separate baseline; the score-bias variant is intentionally excluded from
# this family because it is a separate named variant, not a merge).
V21_POLICIES = ("markov2", "markov_ctx", "model_a", "model_b", "model_b2", "model_b3")
# Predeclared comparison family for v2.1 (frozen BEFORE evaluation; see the
# protocol v2.1 amendment). Every member is a Model-style paired comparison
# matched by workload seed and query id.
V21_FAMILY = (
    ("H1_preserved", "model_b", "model_a"),
    ("H2_B2_vs_markov2", "model_b2", "markov2"),
    ("H2_B2_vs_markov_ctx", "model_b2", "markov_ctx"),
    ("H2_B2_vs_model_b", "model_b2", "model_b"),
    ("H2_B3_vs_markov2", "model_b3", "markov2"),
    ("H2_B3_vs_markov_ctx", "model_b3", "markov_ctx"),
    ("H2_B3_vs_model_b2", "model_b3", "model_b2"),
)


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
    out = {
        "markov1": markov1,
        "markov2": markov2,
        "markov_ctx": ctx_pred,
        "model_a": [p for p, _ in ranked_a],
        "model_b": [p for p, _ in ranked_b],
        "model_b_table": tbl_pred,
    }
    # v2.1 variants: rank candidates by the calibrated-score order. Tie-breaking
    # and ordering follow the candidate-generator iteration order.
    for name, variant in (("model_b2", "b2"), ("model_b3", "b3")):
        model = trained.get(name)
        if model is None:
            out[name] = list(candidates)
            continue
        ctx = dict(context)
        if variant == "b3":
            params = context.get("params")
            ctx = {**ctx, "param_bucket": ML.param_bucket(params), "param_available": True}
        logits = ML.candidate_logits(model, history, candidates, ctx, page_map, variant,
                                     trained.get("variants_counts", {}).get(
                                         variant, trained.get("counts")))
        order = sorted(zip(list(candidates), [float(v) for v in logits]),
                       key=lambda pair: pair[1], reverse=True)
        out[name] = [p for p, _ in order]
    return out


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


def _build_ctx_tables_v21(train: list, index: CandidateIndex, page_map: dict[str, Any], *, top_k: int = 8) -> dict:
    """markov_ctx counts on the semantic context key (training executions only)."""
    markov_ctx: dict[tuple, Counter] = defaultdict(Counter)
    for seq in train:
        pages = [int(p) for p in seq["pages"]]
        base = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"),
                "plan_class": seq.get("plan_class")}
        for i in range(1, len(pages)):
            context = {**base, "position_in_query": min(i, 512) / 512.0}
            markov_ctx[_ctx_key(pages[i - 1], context, page_map)][pages[i]] += 1
    return markov_ctx


def evaluate_split_v21(test: list, trained: dict[str, Any], page_map: dict[str, Any],
                       markov_ctx: dict, *, top_k: int = 8, temperature: dict | None = None) -> dict[str, Any]:
    """v2.1 per-execution evaluation with the declared metric set.

    Identical train/validation/test executions, identical candidate generation
    (the single ``CandidateIndex`` passed via ``trained["index"]``) and
    identical tuning budgets apply to every policy. Reported separately:
    candidate-generation recall, native candidate coverage per generator,
    ranked Top-1/Top-5, timely coverage (1-step-lagged Top-K at reference-cache
    misses), and inference microseconds per prediction.
    """
    import time as _time
    from .models import HISTORY_LENGTH as HL
    index = trained["index"]
    inference_us: dict[str, list[float]] = defaultdict(list)
    per_exec: dict[str, dict[str, list[bool]]] = {}
    top5_hits: dict[str, list[bool]] = defaultdict(list)
    recall_hits: list[bool] = []
    coverage: Counter = Counter()
    timely_hits: dict[str, list[bool]] = defaultdict(list)
    # calibration stream per neural policy (normalized candidate distributions)
    calib: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    for seq in test:
        pages = [int(p) for p in seq["pages"]]
        if len(pages) < 2:
            continue
        case = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"),
                "plan_class": seq.get("plan_class"), "params": seq.get("params")}
        hits: dict[str, list[bool]] = defaultdict(list)
        prev_topk: dict[str, list[int]] = {}
        for i in range(1, len(pages)):
            history = pages[:i][-HL:]
            target = pages[i]
            context = {**case, "position_in_query": min(i, 512) / 512.0}
            candidates = index.candidates(history)
            recall_hits.append(bool(target in candidates))
            # native candidate coverage per generator
            coverage["semantic_generator"] += 1
            m2 = list(index.second_order.get(f"{pages[i-2]}:{pages[i-1]}", []) if i >= 2 else [])
            if not m2:
                m2 = list(index.successors.get(pages[i - 1], []))
            if m2:
                coverage["markov2_generator"] += 1
            key = _ctx_key(pages[i - 1], context, page_map)
            if key in markov_ctx:
                coverage["markov_ctx_generator"] += 1
            t_start = _time.perf_counter_ns()
            preds = _predict(history, context, key, index, trained, page_map,
                             markov_ctx, model_b_table={}, top_k=top_k)
            inference_us["table_policies"] = inference_us.get("table_policies", [])
            for policy in V21_POLICIES:
                ranked = preds.get(policy) or []
                hits[policy].append(bool(ranked and ranked[0] == target))
                top5_hits[policy].append(bool(ranked and target in list(ranked)[:5]))
                # timely coverage: a miss that the PREVIOUS position's Top-K
                # (the degree-K slots the runtime would actually prefetch)
                # already covered before this demand.
                if prev_topk.get(policy) is not None:
                    timely_hits[policy].append(target in prev_topk[policy][:top_k])
                prev_topk[policy] = list(ranked)[:top_k]
            _time_us = (_time.perf_counter_ns() - t_start) / max(1, 1000.0)
            inference_us["table_policies"].append(_time_us * 1000.0)
            # neural inference cost separately-timed with real model calls
            for policy in ("model_a", "model_b"):
                candidates_ = index.candidates(history)
                ctx = {**case, "position_in_query": min(i, 512) / 512.0}
                t_n = _time.perf_counter_ns()
                score_candidates(trained[policy], history, candidates_, ctx, page_map,
                                 index.max_page, semantic=(policy == "model_b"))
                inference_us[policy].append((_time.perf_counter_ns() - t_n) / 1000.0)
            for policy in ("model_b2", "model_b3"):
                candidates_ = index.candidates(history)
                ctx = {**case, "position_in_query": min(i, 512) / 512.0}
                if policy == "model_b3":
                    ctx = {**ctx, "param_bucket": ML.param_bucket(seq.get("params")), "param_available": True}
                t_n = _time.perf_counter_ns()
                logits = ML.candidate_logits(trained[policy], history, candidates_, ctx,
                                             page_map, "b2" if policy == "model_b2" else "b3",
                                             trained["counts"])
                inference_us[policy].append((_time.perf_counter_ns() - t_n) / 1000.0)
                probs = ML.softmax_normalize(logits, temperature=(temperature or {}).get(policy, 1.0))
                if len(probs) >= 2 and target in candidates_:
                    pl, cl = calib[policy]
                    pl.append([float(x) for x in probs])
                    cl.append(candidates_.index(target))
        per_exec[str(seq.get("query_id"))] = dict(hits)
    calib_report = {}
    for policy, (probs, golds) in calib.items():
        calib_report[policy] = ML.expected_calibration_error(probs, golds)
    return {
        "per_exec": per_exec,
        "candidate_recall_pct": round(100.0 * sum(recall_hits) / max(1, len(recall_hits)), 4),
        "native_coverage_pct": {
            "markov2": round(100.0 * coverage["markov2_generator"] / max(1, coverage["semantic_generator"]), 4),
            "markov_ctx": round(100.0 * coverage["markov_ctx_generator"] / max(1, coverage["semantic_generator"]), 4),
            "semantic": 100.0,
        },
        "top5_hit_count": {policy: sum(v) for policy, v in top5_hits.items()},
        "top5_once": top5_hits,
        "timely_once": timely_hits,
        "inference_us_per_prediction": {k: round(statistics.mean(v), 3) for k, v in inference_us.items() if v},
        "calibration": calib_report,
    }


def run_offline_study_v21(sequences_path: str | Path, workload_path: str | Path, page_map_path: str | Path,
                          output_dir: str | Path, *, seeds: list[int] | None = None, epochs: int = 20,
                          hidden: int = 128, max_candidates: int = 32, top_k: int = 8) -> dict[str, Any]:
    """v2.1 offline study: Model B preserved + B2/B3 + declared family + Holm.

    All variants get identical train/val/test executions, identical
    candidate-generation rules, and identical tuning budgets. Statistics use
    paired comparisons matched by workload seed and query id against the
    COMMON set of held-out transitions; the seed-clustered bootstrap and
    Holm correction come from ``pagewise.statistics`` (declared family).
    """
    from .statistics import declare_family
    seeds = seeds or [101, 102, 103, 104, 105]
    sequences = load_jsonl(sequences_path)
    cases = load_cases(workload_path)
    page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    sessions: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for split_kind in ("params", "templates"):
        train, val, test, info = build_splits(sequences, cases, kind=split_kind, seed=7)
        for seed in seeds:
            # -- identical budgets: each variant gets epochs/hidden identical;
            #    candidate index + transition counts from TRAINING only.
            trained: dict[str, Any] = {"index": CandidateIndex.build(train, max_candidates=max_candidates),
                                       "counts": None}
            trained["index"].page_map_sha256 = page_map_digest(page_map)
            ctx_tables = _build_ctx_tables_v21(train, trained["index"], page_map, top_k=top_k)
            # identical tuning budget for every variant: same epochs/hidden,
            # same candidate generation, same negative sampling recipe.
            variant_counts: dict[str, Any] = {}
            for name, variant in (("model_b2", "b2"), ("model_b3", "b3")):
                (model, index, counts), report = ML.train_variant_model(
                    train, val, page_map, variant, seed=seed, epochs=epochs, hidden=hidden,
                    max_candidates=max_candidates, workload_cases=cases)
                trained[name] = model
                variant_counts[variant] = counts
            # temperature per variant: fit on VALIDATION data only
            temps: dict[str, float] = {}
            for name, variant in (("model_b2", "b2"), ("model_b3", "b3")):
                temps[name] = ML.fit_temperature_from_validation(
                    val, trained[name], page_map, variant, variant_counts[variant],
                    trained["index"], workload_cases=cases)["temperature"]
            # model_a / model_b (original, PRESERVED) with the same budget
            trained = {
                "index": trained["index"],
                "model_a": _train_model_for_split(train, val, page_map, False, seed, epochs, hidden, max_candidates),
                "model_b": _train_model_for_split(train, val, page_map, True, seed + 2, epochs, hidden, max_candidates),
                "model_b2": trained.get("model_b2"),
                "model_b3": trained.get("model_b3"),
                "counts": variant_counts.get("b2"),
                "variants_counts": variant_counts,
            }
            evaluation = evaluate_split_v21(test, trained, page_map, ctx_tables, top_k=top_k,
                                            temperature=temps)
            for query_id, hits in evaluation["per_exec"].items():
                for family_label, cand, base in V21_FAMILY:
                    cand_hits = hits.get(cand, []); base_hits = hits.get(base, [])
                    if len(cand_hits) == len(base_hits) and cand_hits:
                        sessions.setdefault((split_kind, cand, base), {"sessions": {}})
                        unit = sessions[(split_kind, cand, base)]["sessions"].setdefault(seed, [])
                        unit.extend([100.0 * (c - b) / max(1, len(cand_hits)) for c, b in zip(cand_hits, base_hits)])
            # aggregate top-1 row per policy
            for policy in V21_POLICIES:
                accs = _exec_accuracy(evaluation["per_exec"], policy)
                top5s = evaluation["top5_once"].get(policy, [])
                timely = evaluation["timely_once"].get(policy, [])
                rows.append({
                    "split": split_kind, "seed": seed, "policy": policy,
                    "top1_accuracy_pct": round(statistics.mean(accs.values()), 4) if accs else None,
                    "top5_accuracy_pct": round(100.0 * sum(top5s) / max(1, len(top5s)), 4) if top5s else None,
                    "timely_coverage_pct": round(100.0 * sum(timely) / max(1, len(timely)), 4) if timely else None,
                    "inference_us_per_prediction": evaluation["inference_us_per_prediction"].get(policy),
                    "calibration": evaluation["calibration"].get(policy),
                    "executions": len(evaluation["per_exec"]),
                })
            rows.append({
                "split": split_kind, "seed": seed, "policy": "candidate_recall",
                "candidate_recall_pct": evaluation["candidate_recall_pct"],
                "native_coverage_pct": evaluation["native_coverage_pct"],
            })
    # declared family evaluation per split with Holm
    family_reports: dict[str, Any] = {}
    for split_kind in ("params", "templates"):
        members = []
        for cand, base in {(c, b) for _, c, b in V21_FAMILY}:
            key = (split_kind, cand, base)
            if key in sessions:
                members.append((f"{cand}_vs_{base}", sessions[key]["sessions"]))
        if members:
            family_reports[split_kind] = declare_family(members, iterations=5000)
    report = {
        "schema_version": "v2.1",
        "seeds": seeds, "epochs": epochs, "hidden": hidden, "top_k": top_k,
        "candidate_count": max_candidates,
        "rows": rows,
        "family_reports": family_reports,
        "fairness_note": ("Identical train/val/test executions, candidate generation, and "
                          "tuning budgets across model_a, model_b (preserved), model_b2, and "
                          "model_b3; transition counts and candidate indexes from training "
                          "executions only; temperature scaling fit on validation only."),
    }
    (out / "offline_study_v21.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import csv as _csv
    flat_rows = []
    for row in rows:
        base = {k: v for k, v in row.items() if not isinstance(v, dict)}
        flat_rows.append(base)
    if flat_rows:
        with (out / "offline_study_v21.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter.handle if False else _csv.DictWriter(handle, fieldnames=sorted({k for r in flat_rows for k in r}))
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)
    return report


def _train_model_for_split(train: list, val: list, page_map: dict[str, Any], semantic: bool,
                           seed: int, epochs: int, hidden: int, max_candidates: int):
    """Train one v2-era model (model_a or model_b) with the same given budget."""
    index = CandidateIndex.build(train, max_candidates=max_candidates)
    index.page_map_sha256 = page_map_digest(page_map)
    empty = lambda dim: (np.empty((0, dim), np.float32), np.empty((0,), np.float32))  # noqa: E731
    x, y = _training_pairs(train, index, page_map, semantic, random.Random(seed))
    xv, yv = _training_pairs(val, index, page_map, semantic, random.Random(seed + 1)) if val else empty(x.shape[1])
    model, _ = _fit(x, y, xv, yv, seed=seed, epochs=epochs, hidden=hidden)
    return model


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
