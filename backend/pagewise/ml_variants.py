"""v2.1 model variants and ML-fairness machinery.

Variants (declared in the protocol v2.1 amendment, never re-merged):

* ``model_a`` — history-only MLP (unchanged baseline).
* ``model_b`` — original semantic MLP, PRESERVED as a separate baseline.
* ``model_b2`` — model_b + training-derived Markov-2 successor
  probability/count input features.
* ``model_b_score_bias`` (optional variant, never merged into b2) — output-score
  bias instead of input features.
* ``b3`` — model_b2 + query-parameter features that are demonstrably available
  when the prediction is issued (the full SQL parameter set is known at query
  start, before any page prediction). Row identifiers obtained from an index
  leaf are NOT used in this implementation, because the current VFS demand
  context does not expose index child→rowid observability events; that
  extension would require a new instrumentation pass and is documented as
  unimplemented rather than silently assumed available.

Feature provenance rules (fairness):
* Candidate indexes, Markov transition counts, and priors are derived from
  TRAINING executions only.
* No feature uses future page targets, future query outcomes, or transition
  statistics computed on validation/test executions.

Calibration:
* Temperature scaling is fit on VALIDATION data only.
* Candidate scores are normalized over the competing candidate set with a
  softmax over temperature-scaled logits. Independent sigmoids are explicitly
  NOT treated as a categorical distribution (an explicit violation in v2).

Labels:
* All-read training uses the observed next page (v2 behavior).
* Miss-only training uses replay of a DECLARED, reproducible reference cache
  (the no-prefetch LRU cache of fixed declared size) over the training
  executions; transitions whose page would hit that reference cache are
  excluded from label generation. Labels are never derived from each tested
  model's own future cache outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

from .dataset import load_jsonl
from . import models as M

# Declared reference-cache replay size for miss-only label generation.
MISS_ONLY_REPLAY_CACHE_PAGES = 32
# Declared param-bucket count for B3 query-parameter features.
PARAM_BUCKETS = 8
# Markov-2 feature dimension (documented in the amendment).
MARKOV2_FEATURES = 6
PARAM_FEATURES = PARAM_BUCKETS + 1


def variant_semantic(variant: str) -> bool:
    if variant in {"a"}:
        return False
    if variant in {"b", "b2", "b3", "b_score_bias"}:
        return True
    raise ValueError(f"Unknown model variant {variant!r}")


def variant_extra_dims(variant: str) -> int:
    if variant == "b2":
        return MARKOV2_FEATURES
    if variant == "b3":
        return MARKOV2_FEATURES + PARAM_FEATURES
    return 0


def build_training_transition_counts(
    sequences: list[dict[str, Any]], *, max_candidates: int = 32
) -> dict[str, Any]:
    """Markov-1/2 transition counts from TRAINING executions only."""
    first: defaultdict[int, Counter] = defaultdict(Counter)
    second: defaultdict[str, Counter] = defaultdict(Counter)
    contextual: defaultdict[str, Counter] = defaultdict(Counter)
    first_total: Counter = Counter()
    second_total: Counter = Counter()
    for seq in sequences:
        pages = [int(p) for p in seq["pages"]]
        for i in range(1, len(pages)):
            prev1 = pages[i - 1]
            target = pages[i]
            first[prev1][target] += 1
            first_total[prev1] += 1
            context_key = f"{prev1}|{seq.get('template_id')}|{seq.get('plan_class')}"
            contextual[context_key][target] += 1
            if i >= 2:
                key = f"{pages[i - 2]}:{prev1}"
                second[key][target] += 1
                second_total[key] += 1
    successors = {p: [c for c, _ in cnt.most_common(max_candidates)]
                  for p, cnt in first.items()}
    second_order = {k: [c for c, _ in cnt.most_common(max_candidates)]
                    for k, cnt in second.items()}
    return {
        "first": {p: dict(c) for p, c in first.items()},
        "second": {k: dict(c) for k, c in second.items()},
        "first_total": dict(first_total),
        "second_total": dict(second_total),
        "successors": successors,
        "second_order": second_order,
        "contextual": {k: dict(c) for k, c in contextual.items()},
        "max_page": max((max(pages) for seq in sequences for pages in [seq["pages"]] if pages), default=1),
    }


def param_bucket(params: Any, buckets: int = PARAM_BUCKETS) -> int:
    key = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def prediction_context(sequence: dict[str, Any], workload_cases: dict | None = None) -> dict[str, Any]:
    """The same observable query context for training, validation and inference."""
    case = (workload_cases or {}).get(str(sequence.get("query_id")), {})
    params = case.get("params", sequence.get("params"))
    return {"template_id": sequence.get("template_id"),
            "workload_family": sequence.get("workload_family"),
            "plan_class": sequence.get("plan_class"), "params": params,
            "param_bucket": param_bucket(params) if params is not None else None,
            "param_available": params is not None}


def markov2_feature_block(candidate: int, prev2: int | None, prev1: int, counts: dict[str, Any]) -> list[float]:
    """Training-derived Markov-2 successor features for one candidate.

    Features (6, fixed order documented in the amendment):
      0. p(succ | 2-order context)   = count2[c] / tot2   (0 if unseen context)
      1. p(succ | 1-order context)   = count1[c] / tot1
      2. indicator: candidate in training second-order successor list
      3. rank of candidate in that list, normalized to [0,1] (1 = unseen)
      4. indicator: candidate in training first-order successor list
      5. count2[c] normalized by log(1 + tot2) (short-memory count scale)
    """
    key = f"{prev2}:{prev1}" if prev2 is not None else ""
    second = counts["second"].get(key) or {}
    total2 = counts["second_total"].get(key, 0)
    first = counts["first"].get(prev1) or {}
    total1 = counts["first_total"].get(prev1, 0)
    c2 = second.get(candidate, 0)
    c1 = first.get(candidate, 0)
    succ2 = counts["second_order"].get(key, [])
    succ1 = counts["successors"].get(prev1, [])
    rank2 = (succ2.index(candidate) + 1) / len(succ2) if candidate in succ2 else 1.0
    p2 = c2 / total2 if total2 else 0.0
    p1 = c1 / total1 if total1 else 0.0
    return [
        p2, p1,
        1.0 if candidate in succ2 else 0.0,
        rank2,
        1.0 if candidate in succ1 else 0.0,
        (math.log1p(c2) / math.log1p(total2)) if total2 else 0.0,
    ]


def markov2_feature_matrix(candidates, history, counts):
    """Same six training-derived features; build context ranks once per group."""
    import numpy as np
    prev1 = history[-1] if history else 0
    key = f"{history[-2]}:{prev1}" if len(history) > 1 else ""
    second = counts["second"].get(key, {})
    first = counts["first"].get(prev1, {})
    total2 = counts["second_total"].get(key, 0)
    total1 = counts["first_total"].get(prev1, 0)
    successors2 = counts["second_order"].get(key, [])
    ranks = {p: (i + 1) / len(successors2) for i, p in enumerate(successors2)}
    successors1 = set(counts["successors"].get(prev1, []))
    denom = math.log1p(total2) if total2 else 1.0
    return np.asarray([[second.get(p, 0) / total2 if total2 else 0.0,
                        first.get(p, 0) / total1 if total1 else 0.0,
                        float(p in ranks), ranks.get(p, 1.0), float(p in successors1),
                        math.log1p(second.get(p, 0)) / denom if total2 else 0.0]
                       for p in candidates], dtype=np.float32)


def variant_feature_vector(
    history: list[int], candidate: int, context: dict[str, Any], metadata: dict[str, Any],
    max_page: int, variant: str, counts: dict[str, Any] | None,
    disabled_semantic_groups: Any = None,
) -> Any:
    """Feature vector for any declared variant; extras appended past v2 dims."""
    base = M.feature_vector(history, candidate, context, metadata, max_page,
                            semantic=variant_semantic(variant),
                            disabled_semantic_groups=disabled_semantic_groups)
    if variant == "b2":
        extra = markov2_feature_block(candidate,
                                      history[-2] if len(history) >= 2 else None,
                                      history[-1] if history else 0, counts or {})
        return M.np.concatenate([base, M.np.asarray(extra, dtype=M.np.float32)])
    if variant == "b3":
        block = markov2_feature_block(candidate,
                                      history[-2] if len(history) >= 2 else None,
                                      history[-1] if history else 0, counts or {})
        # Availability gate: parameters are known at query start, so the gate is
        # satisfied whenever the runtime passes a param bucket; otherwise zeros.
        bucket = context.get("param_bucket")
        if bucket is None or not context.get("param_available", True):
            param_block = [0.0] * PARAM_FEATURES
        else:
            param_block = M.one_hot(int(bucket), PARAM_BUCKETS) + [1.0]
        return M.np.concatenate([base, M.np.asarray(block + param_block, dtype=M.np.float32)])
    return base


# ---------------------------------------------------------------- labels

def reference_cache_miss_replay(
    sequences: list[dict[str, Any]], cache_pages: int = MISS_ONLY_REPLAY_CACHE_PAGES
) -> dict[str, list[bool]]:
    """Declared reference-cache replay for miss-only labels.

    Deterministic no-prefetch LRU replay of ``cache_pages`` (declared constant,
    v2.1 default 32) over each training execution. Produces, per query id, one
    boolean per observed read: True when the page would MISS the reference
    cache. This replay is identical for every model and execution split, so
    training labels never come from a tested model's own future outcomes.
    """
    out: dict[str, list[bool]] = {}
    for seq in sequences:
        cache: list[int] = []
        flags: list[bool] = []
        for page in [int(p) for p in seq["pages"]]:
            if page in cache:
                cache.remove(page)
                cache.append(page)
                flags.append(False)
            else:
                if len(cache) >= cache_pages:
                    cache.pop(0)
                cache.append(page)
                flags.append(True)
        out[str(seq.get("query_id"))] = flags
    return out


def variant_training_pairs(
    train: list, index: M.CandidateIndex, metadata: dict[str, Any], variant: str,
    counts: dict[str, Any] | None, rng: Any, *, negatives_per_positive: int = 16,
    disabled_semantic_groups: Any = None, miss_replay: dict[str, list[bool]] | None = None,
    workload_cases: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    """Training pairs for a variant with explicit label-reference discipline.

    ``miss_replay`` is the declared reference-cache replay (see
    :func:`reference_cache_miss_replay`) computed on the same training
    executions; when provided, transitions flagged as reference-cache HITS are
    excluded (miss-only training). Candidate proposals never teach the model
    about pages the inference-time candidate generator cannot produce.
    """
    xs: list[Any] = []
    ys: list[float] = []
    for seq in train:
        pages = [int(p) for p in seq["pages"]]
        flags = miss_replay.get(str(seq.get("query_id"))) if miss_replay is not None else None
        base_context = prediction_context(seq, workload_cases)
        for i in range(1, len(pages)):
            if flags is not None and (i >= len(flags) or not flags[i]):
                # flags[j] marks observed read at position j; the transition
                # ending at pages[i] is read index i. Miss-only training keeps
                # only transitions whose page misses the declared reference cache.
                continue
            history, target = pages[:i][-M.HISTORY_LENGTH:], pages[i]
            context = {**base_context, "position_in_query": min(i, 512) / 512.0}
            candidates = index.candidates(history)
            if target not in candidates:
                continue
            negatives = [p for p in candidates if p != target]
            if len(negatives) > negatives_per_positive:
                negatives = rng.sample(negatives, negatives_per_positive)
            for candidate in [target] + negatives:
                xs.append(variant_feature_vector(history, candidate, context, metadata,
                                                 index.max_page, variant, counts,
                                                 disabled_semantic_groups))
                ys.append(1.0 if candidate == target else 0.0)
    if not xs:
        from .models import SEMANTIC_FEATURE_GROUPS  # noqa: F401
        semantic_dim = (len(M.PAGE_TYPES) + 1 + 32 + len(M.PLAN_CLASSES) + 16
                        + len(M.WORKLOAD_FAMILIES) + 1)
        dims = M.HISTORY_LENGTH * 2 + 3 + (semantic_dim if variant_semantic(variant) else 0) + variant_extra_dims(variant)
        empty = M.np.empty((0, dims), dtype=M.np.float32)
        return empty, M.np.empty((0,), dtype=M.np.float32)
    return M.np.stack(xs).astype(M.np.float32), M.np.asarray(ys, dtype=M.np.float32)


def train_variant_model(
    train: list, val: list, page_map: dict[str, Any], variant: str, seed: int, *,
    max_candidates: int = 32, hidden: int = 128, epochs: int = 20,
    miss_only: bool = False, negatives_per_positive: int = 16,
    workload_cases: dict[str, dict[str, Any]] | None = None,
    label_temperature: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Train one variant with the same hyperparameter budget as every other.

    ``miss_only=False`` = all-read training (every observed transition).
    ``miss_only=True``  = miss-only training under the declared reference-cache
    replay. Checkpoints record the variant and the label mode, so downstream
    evaluation can never silently mix changes.
    """
    counts = (build_training_transition_counts(train, max_candidates=max_candidates)
              if variant in {"b2", "b3"} else None)
    index = M.CandidateIndex.build(train, max_candidates=max_candidates)
    index.page_map_sha256 = M.page_map_digest(page_map)
    replay = (reference_cache_miss_replay(train) if miss_only else None)
    x_train, y_train = variant_training_pairs(train, index, page_map, variant, counts,
                                              __import__("random").Random(seed),
                                              negatives_per_positive=negatives_per_positive,
                                              miss_replay=replay, workload_cases=workload_cases)
    if val:
        x_val, y_val = variant_training_pairs(val, index, page_map, variant, counts,
                                              __import__("random").Random(seed + 1),
                                              negatives_per_positive=negatives_per_positive,
                                              miss_replay=reference_cache_miss_replay(val) if miss_only else None,
                                              workload_cases=workload_cases)
    else:
        x_val = M.np.empty((0, x_train.shape[1]), M.np.float32)
        y_val = M.np.empty((0,), M.np.float32)
    model, fit_report = M._fit(x_train, y_train, x_val, y_val, seed=seed, epochs=epochs, hidden=hidden)
    report = {
        "variant": variant, "miss_only": miss_only, "seed": seed,
        "train_pairs": int(len(x_train)), "val_pairs": int(len(x_val)),
        "extra_feature_dims": variant_extra_dims(variant),
        "feature_schema": (
            ("semantic" if variant_semantic(variant) else "history")
            + ("+markov2" if variant in ("b2", "b3") else "")
            + ("+param" if variant == "b3" else "")
            + ("|miss_only_labels" if miss_only else "|all_read_labels")
        ),
        **fit_report,
    }
    return (model, index, counts), report


# ------------------------------------------------------------- calibration

def candidate_logits(model: Any, history: list[int], candidates: list[int], context: dict[str, Any],
                     page_map: dict[str, Any], variant: str, counts: dict[str, Any] | None,
                     *, max_page: int | None = None) -> Any:
    if not candidates:
        return M.np.zeros((0,), dtype=M.np.float32)
    # Persisted training normalization must not change with the current candidates.
    scale = max_page if max_page is not None else (counts or {}).get("max_page")
    if scale is None:
        raise ValueError("candidate_logits requires the training max_page normalization")
    x = M.np.stack([
        variant_feature_vector(history, p, context, page_map, scale, variant, counts)
        for p in candidates
    ]).astype(M.np.float32)
    with M.torch.no_grad():
        logits = model(M.torch.from_numpy(x))
    return logits.cpu().numpy().astype(float)


def softmax_normalize(logits: Any, temperature: float = 1.0) -> Any:
    """Normalize candidate scores over the COMPETING candidate set.

    z_i = logit_i / T; p_i = softmax(z) over the candidate set. This is a
    normalized categorical distribution over competing candidates; independent
    sigmoid outputs are explicitly not used (v2 violation fixed here).
    """
    import numpy as np
    z = np.asarray(logits, dtype=float) / max(1e-6, temperature)
    z = z - z.max() if len(z) else z
    e = np.exp(z)
    return e / e.sum() if e.sum() > 0 else e


def fit_temperature_from_validation(
    val: list, model: Any, page_map: dict[str, Any], variant: str, counts: dict[str, Any] | None,
    index: M.CandidateIndex, *, workload_cases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Temperature scaling fit on VALIDATION data only (NLL minimization)."""
    import numpy as np
    logits_list: list[Any] = []
    target_pos: list[int] = []
    for seq in val:
        pages = [int(p) for p in seq["pages"]]
        base_context = prediction_context(seq, workload_cases)
        for i in range(1, len(pages)):
            history, target = pages[:i][-M.HISTORY_LENGTH:], pages[i]
            context = {**base_context, "position_in_query": min(i, 512) / 512.0}
            candidates = index.candidates(history)
            if target not in candidates:
                continue
            logits = candidate_logits(model, history, candidates, context, page_map, variant, counts,
                                      max_page=index.max_page)
            if len(logits) >= 2:
                logits_list.append(logits)
                target_pos.append(candidates.index(target))
    if not logits_list:
        return {"temperature": 1.0, "val_nll": float("nan"), "val_transitions": 0}
    # Golden-section-free closed search over T in [0.05, 20] on a log grid.
    best_t, best_nll = 1.0, float("inf")
    for t in list(np.exp(np.linspace(math.log(0.05), math.log(20.0), 200))):
        nll = 0.0
        n = 0
        for logits, pos in zip(logits_list, target_pos):
            p = softmax_normalize(logits, float(t))
            nll -= math.log(max(p[pos], 1e-12))
            n += 1
        if nll / n < best_nll:
            best_nll, best_t = nll / n, float(t)
    return {"temperature": round(best_t, 6), "val_nll": round(best_nll, 6),
            "val_transitions": n}


def expected_calibration_error(probs: list[float], correct: list[int], bins: int = 10) -> dict[str, float]:
    """ECE + NLL of the normalized candidate distribution.

    ``correct[i]`` is the gold index of the target inside candidate i's set.
    Confidence = max normalized probability; accuracy = 1 when argmax == gold.
    """
    if not probs:
        return {"ece": float("nan"), "nll": float("nan"), "count": 0}
    confidences = [max(p) for p in probs]
    outcomes = [1.0 if _argmax(p) == gold else 0.0 for p, gold in zip(probs, correct)]
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(c, o) for c, o in zip(confidences, outcomes)
                 if lo <= c < hi or (b == bins - 1 and c == hi)]
        if not group:
            continue
        acc = sum(o for _, o in group) / len(group)
        mean_conf = sum(c for c, _ in group) / len(group)
        ece += len(group) / n * abs(acc - mean_conf)
    nll = -statistics.mean(math.log(max(p[g], 1e-12)) for p, g in zip(probs, correct))
    return {"ece": round(ece, 6), "nll": round(nll, 6), "count": n}


def _argmax(values: list[float]) -> int:
    best, best_i = values[0], 0
    for i, v in enumerate(values):
        if v > best:
            best, best_i = v, i
    return best_i


# --------------------------------------------------------- break-even gate

def breakeven_gate(
    *,
    timely_probability: float, saved_latency_ms: float, wasted_rate: float,
    waste_cost_ms: float, inference_cost_ms: float, scheduling_cost_ms: float,
    contention_cost_ms: float = 0.0,
) -> dict[str, Any]:
    """Expected net benefit of prefetching a candidate under the calibrated model.

    net = P(timely useful) * achievable_demand_latency_saved
          - P(waste) * waste_cost - contention - inference - scheduling costs.

    Thresholds are fit on development/validation data only (see amendment);
    this function only evaluates them.
    """
    benefit = timely_probability * max(0.0, saved_latency_ms)
    cost = wasted_rate * max(0.0, waste_cost_ms) + max(0.0, inference_cost_ms) \
        + max(0.0, scheduling_cost_ms) + max(0.0, contention_cost_ms)
    net = benefit - cost
    return {
        "expected_net_benefit_ms_per_candidate": round(net, 6),
        "benefit_ms": round(benefit, 6),
        "cost_ms": round(cost, 6),
        "gate_pass": net > 0.0,
        "units": "milliseconds per predicted candidate; all inputs same unit",
        "formula": "net = P(timely_useful)*saved_latency - P(waste)*waste_cost "
                   "- contention - inference - scheduling",
    }


def summarize_inference_cost_us(start_ns: int, end_ns: int, predictions: int) -> float:
    """Mean inference wall time in microseconds per prediction."""
    if predictions <= 0:
        return 0.0
    return round((end_ns - start_ns) / max(1, predictions) / 1000.0, 3)
