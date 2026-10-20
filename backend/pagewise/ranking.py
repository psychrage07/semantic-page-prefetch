"""Development neural candidate ranking with explicit feature and sampling contracts.

All proposal counts and normalization come from training only. Groups retain
targets missing from the candidate set as errors during evaluation. Training
uses only available targets; candidate-generation failures are never hidden.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import copy
import math
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from . import ml_variants as ML
from . import models as M


@dataclass(frozen=True)
class RankerConfig:
    name: str
    variant: str = "b2"
    objective: str = "listwise"
    hidden: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    prior_weight: float = 0.0
    structural: bool = False
    no_semantics: bool = False
    miss_only: bool = False
    negatives_per_positive: int = 16


def fit_parameter_scales(sequences: list[dict], cases: dict) -> list[float]:
    scales = [1.0] * 3
    for seq in sequences:
        params = cases.get(str(seq["query_id"]), {}).get("params")
        for i, value in enumerate(params[:3] if isinstance(params, (tuple, list)) else []):
            if isinstance(value, (int, float)) and math.isfinite(value):
                scales[i] = max(scales[i], math.log1p(abs(value)))
    return scales


def extra_features(history: list[int], candidate: int, context: dict, metadata: dict,
                   max_page: int, parameter_scales: list[float]) -> list[float]:
    last = history[-1]
    prev = metadata.get(str(last), {})
    cur = metadata.get(str(candidate), {})
    delta = candidate - last
    values = [float(prev.get("object") == cur.get("object")),
              float(prev.get("page_type") == cur.get("page_type")),
              (float(cur.get("depth", 0)) - float(prev.get("depth", 0))) / 32,
              math.copysign(math.log1p(abs(delta)), delta) / math.log1p(max(1, max_page))]
    params = context.get("params")
    params = params if isinstance(params, (tuple, list)) else []
    for i in range(3):
        value = params[i] if i < len(params) else None
        available = isinstance(value, (float, int)) and math.isfinite(value)
        norm = math.copysign(math.log1p(abs(value)), value) / parameter_scales[i] if available else 0.0
        values.extend([float(np.clip(norm, -2, 2)), float(available)])
    return values


def group_features(history: list[int], candidates: list[int], context: dict, metadata: dict,
                   index: M.CandidateIndex, counts: dict, config: RankerConfig,
                   parameter_scales: list[float]) -> np.ndarray:
    disabled = tuple(M.SEMANTIC_FEATURE_GROUPS) if config.no_semantics else None
    # Broadcast candidate-independent history and context once per prediction.
    history_part = M.feature_vector(history, candidates[0], context, metadata, index.max_page, False)[:2 * M.HISTORY_LENGTH]
    pages = np.asarray(candidates, dtype=float)
    common = np.column_stack([np.broadcast_to(history_part, (len(pages), len(history_part))),
                              pages / max(1, index.max_page),
                              np.clip((pages - history[-1]) / max(1, index.max_page), -1, 1),
                              (pages == history[-1]).astype(float)]).astype(np.float32)
    blocks = [common]
    if ML.variant_semantic(config.variant):
        if config.no_semantics:
            width = len(M.PAGE_TYPES) + 1 + 32 + len(M.PLAN_CLASSES) + 16 + len(M.WORKLOAD_FAMILIES) + 1
            semantic = np.zeros((len(candidates), width), np.float32)
        elif hasattr(metadata, "semantic_matrix"):
            semantic = metadata.semantic_matrix(candidates, context)
        else:
            semantic = np.asarray([M.apply_semantic_ablation(M.semantic_page_features(p, metadata, context), disabled)
                                   for p in candidates], np.float32)
        blocks.append(semantic)
    if config.variant in {"b2", "b3"}:
        blocks.append(ML.markov2_feature_matrix(candidates, history, counts))
    if config.variant == "b3":
        bucket = context.get("param_bucket")
        param = M.one_hot(int(bucket), ML.PARAM_BUCKETS) + [1.0] if bucket is not None and context.get("param_available", True) else [0.0] * ML.PARAM_FEATURES
        blocks.append(np.broadcast_to(np.asarray(param, np.float32), (len(pages), ML.PARAM_FEATURES)))
    if config.structural:
        if hasattr(metadata, "relative_features"):
            relative = metadata.relative_features(history[-1], candidates)
            delta = np.fromiter((math.copysign(math.log1p(abs(p - history[-1])), p - history[-1])
                                 / math.log1p(max(1, index.max_page)) for p in candidates), np.float32)
            params = np.asarray(extra_features(history, candidates[0], context, metadata, index.max_page, parameter_scales)[4:], np.float32)
            extra = np.column_stack([relative, delta, np.broadcast_to(params, (len(candidates), len(params)))])
        else:
            extra = np.asarray([extra_features(history, p, context, metadata, index.max_page, parameter_scales)
                                for p in candidates], np.float32)
        blocks.append(extra)
    return np.concatenate(blocks, axis=1)


def markov_log_prior(history: list[int], candidates: list[int], counts: dict) -> np.ndarray:
    second_key = f"{history[-2]}:{history[-1]}" if len(history) > 1 else ""
    cnt = counts["second"].get(second_key) or counts["first"].get(history[-1], {})
    total = sum(cnt.values())
    # Fixed training-count backoff; never use validation/test successor counts.
    return np.asarray([math.log(max(1e-4, cnt.get(p, 0) / max(1, total))) for p in candidates], np.float32)


def build_groups(sequences: list[dict], index: M.CandidateIndex, metadata: dict, counts: dict,
                 config: RankerConfig, cases: dict, parameter_scales: list[float], *,
                 max_per_query: int | None = None, sampling_seed: int = 101) -> dict[str, Any]:
    rng = random.Random(sampling_seed)
    xs, priors, targets, pages_out, qids, miss_flags, families = [], [], [], [], [], [], []
    target_pages, markov1, markov2, markov1_top5, markov2_top5 = [], [], [], [], []
    markov_ctx, markov_ctx_top5 = [], []
    replay = ML.reference_cache_miss_replay(sequences)
    for seq in sequences:
        pages = seq["pages"]
        positions = list(range(1, len(pages)))
        if max_per_query is not None and len(positions) > max_per_query:
            positions = sorted(rng.sample(positions, max_per_query))
        base = ML.prediction_context(seq, cases)
        for i in positions:
            history = pages[max(0, i - M.HISTORY_LENGTH):i]
            candidates = index.candidates(history)
            if not candidates:
                # A non-empty training index always has a popular fallback.
                raise ValueError("Empty candidate set; cannot discard this transition")
            context = {**base, "position_in_query": min(i, 512) / 512.0}
            x = group_features(history, candidates, context, metadata, index, counts, config, parameter_scales)
            padding = index.max_candidates - len(candidates)
            xs.append(np.pad(x, ((0, padding), (0, 0))))
            priors.append(np.pad(markov_log_prior(history, candidates, counts), (0, padding), constant_values=-1e9))
            pages_out.append(candidates + [-1] * padding)
            targets.append(candidates.index(pages[i]) if pages[i] in candidates else -1)
            target_pages.append(pages[i])
            first = index.successors.get(history[-1], [])
            second = index.second_order.get(f"{history[-2]}:{history[-1]}", []) if len(history) >= 2 else []
            second = second or first
            markov1.append(first[0] if first else -1)
            markov2.append(second[0] if second else -1)
            markov1_top5.append(first[:5] + [-1] * max(0, 5 - len(first)))
            markov2_top5.append(second[:5] + [-1] * max(0, 5 - len(second)))
            context_key = f"{history[-1]}|{base.get('template_id')}|{base.get('plan_class')}"
            context_counts = counts.get("contextual", {}).get(context_key, {})
            contextual = sorted(context_counts, key=lambda p: -context_counts[p]) or first
            markov_ctx.append(contextual[0] if contextual else -1)
            markov_ctx_top5.append(contextual[:5] + [-1] * max(0, 5 - len(contextual)))
            qids.append(str(seq["query_id"]))
            miss_flags.append(replay[str(seq["query_id"])][i])
            families.append(seq.get("workload_family", "unknown"))
    if not xs:
        raise ValueError("No next-page transitions in this split")
    return {"x": np.stack(xs), "prior": np.stack(priors), "pages": np.asarray(pages_out),
            "target": np.asarray(targets, dtype=np.int64), "query_id": qids,
            "reference_miss": np.asarray(miss_flags), "family": families,
            "target_page": np.asarray(target_pages), "markov1_top1": np.asarray(markov1),
            "markov2_top1": np.asarray(markov2),
            "markov1_top5": np.asarray(markov1_top5), "markov2_top5": np.asarray(markov2_top5),
            "markov_ctx_top1": np.asarray(markov_ctx), "markov_ctx_top5": np.asarray(markov_ctx_top5),
            "sampled": max_per_query is not None, "max_per_query": max_per_query}


def _tensor_batch(data: dict, ids: np.ndarray | slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(data["x"][ids]), torch.from_numpy(data["prior"][ids]),
            torch.from_numpy(data["pages"][ids] != -1))


def logits(model: nn.Module, x: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor,
           config: RankerConfig) -> torch.Tensor:
    scores = model(x) + config.prior_weight * prior
    return scores.masked_fill(~valid, -1e9)


def evaluate_ranker(model: nn.Module, data: dict, config: RankerConfig, *, batch_size: int = 512) -> dict:
    model.eval()
    ranks, elapsed = [], []
    with torch.inference_mode():
        for start in range(0, len(data["target"]), batch_size):
            x, prior, valid = _tensor_batch(data, slice(start, start + batch_size))
            t = time.perf_counter_ns()
            scores = logits(model, x, prior, valid, config)
            elapsed.append((time.perf_counter_ns() - t) / 1000)
            ranks.append(torch.argsort(scores, dim=1, descending=True, stable=True).numpy())
    order = np.concatenate(ranks)
    target = data["target"]
    top1 = (order[:, 0] == target) & (target >= 0)
    top5 = np.any(order[:, :5] == target[:, None], axis=1) & (target >= 0)
    qids = np.asarray(data["query_id"])
    per_query = {q: float(top1[qids == q].mean() * 100) for q in dict.fromkeys(data["query_id"])}
    family = np.asarray(data["family"])
    misses = data["reference_miss"]
    return {"top1_macro_pct": float(np.mean(list(per_query.values()))),
            "top1_micro_pct": float(top1.mean() * 100), "top5_micro_pct": float(top5.mean() * 100),
            "candidate_recall_pct": float((target >= 0).mean() * 100),
            "conditional_top1_pct": float(top1[target >= 0].mean() * 100) if np.any(target >= 0) else None,
            "reference_miss_top1_pct": float(top1[misses].mean() * 100) if misses.any() else None,
            "reference_miss_transitions": int(misses.sum()),
            "forward_us_per_prediction_batched": sum(elapsed) / len(target),
            "transitions": len(target), "queries": len(per_query), "per_query_top1_pct": per_query,
            "by_family": {f: float(top1[family == f].mean() * 100) for f in sorted(set(family))},
            "sampled": data["sampled"], "max_per_query": data["max_per_query"]}


def train_ranker(train: dict, val: dict, config: RankerConfig, *, seed: int = 101) -> tuple[nn.Module, dict]:
    if config.objective not in {"bce", "sampled_bce", "listwise", "pairwise"}:
        raise ValueError("Unknown ranking objective")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    negative_rng = torch.Generator().manual_seed(seed + 7919)
    model = M.MLPScorer(train["x"].shape[-1], hidden=config.hidden)
    if config.prior_weight:
        # Start at the Markov policy and learn a neural correction.
        nn.init.zeros_(model.net[-1].weight)
        nn.init.zeros_(model.net[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    eligible = train["target"] >= 0
    if config.miss_only:
        eligible &= train["reference_miss"]
    ids = np.flatnonzero(eligible)
    if not len(ids):
        raise ValueError("No eligible training targets")
    # Give each query equal aggregate training weight, matching macro Top-1.
    from collections import Counter
    frequencies = Counter(train["query_id"][i] for i in ids)
    weights = np.asarray([1 / max(1, frequencies[q]) for q in train["query_id"]], np.float32)
    weights *= len(ids) / len(frequencies)
    baseline = evaluate_ranker(model, val, config)
    # Only the residual model has a meaningful untrained baseline (Markov).
    # A random non-residual network must never be reported as a trained model.
    best_accuracy = baseline["top1_macro_pct"] if config.prior_weight else -math.inf
    best_epoch, stale = 0, 0
    best = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        shuffled = rng.permutation(ids)
        losses = []
        for start in range(0, len(shuffled), 128):
            batch = shuffled[start:start + 128]
            x, prior, valid = _tensor_batch(train, batch)
            scores = logits(model, x, prior, valid, config)
            target = torch.from_numpy(train["target"][batch])
            if config.objective == "listwise":
                loss_per = nn.functional.cross_entropy(scores, target, reduction="none")
            elif config.objective in {"bce", "sampled_bce"}:
                labels = nn.functional.one_hot(target, num_classes=scores.shape[1]).float()
                loss_mask = valid
                if config.objective == "sampled_bce":
                    negative = valid & ~labels.bool()
                    random_order = torch.rand(scores.shape, generator=negative_rng).masked_fill(~negative, float("inf"))
                    k = min(config.negatives_per_positive, max(0, scores.shape[1] - 1))
                    selected = torch.zeros_like(valid)
                    selected.scatter_(1, random_order.topk(k, largest=False).indices, True)
                    loss_mask = (selected & negative) | labels.bool()
                loss_per = (nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none") * loss_mask).sum(1) / loss_mask.sum(1)
            else:
                positive = scores.gather(1, target[:, None])
                negative = valid & ~nn.functional.one_hot(target, num_classes=scores.shape[1]).bool()
                loss_per = (nn.functional.softplus(scores - positive) * negative).sum(1) / negative.sum(1).clamp(min=1)
            loss = (loss_per * torch.from_numpy(weights[batch])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        measured = evaluate_ranker(model, val, config)
        accuracy = measured["top1_macro_pct"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_top1_pct": accuracy})
        if accuracy > best_accuracy + 1e-6:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {"config": asdict(config), "seed": seed, "best_epoch": best_epoch,
                   "best_validation_macro_top1_pct": best_accuracy, "eligible_training_groups": len(ids),
                   "history": history, "selection_metric": "validation macro Top-1"}


def fit_temperature(model: nn.Module, validation: dict, config: RankerConfig) -> dict:
    """Validation-only conditional calibration, using the exact deployed logits."""
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation["target"]), 512):
            x, prior, valid = _tensor_batch(validation, slice(start, start + 512))
            scores.append(logits(model, x, prior, valid, config))
    scores = torch.cat(scores)
    target = torch.from_numpy(validation["target"])
    eligible = target >= 0
    if not eligible.any():
        return {"temperature": 1.0, "transitions": 0, "conditional_nll": None}
    scores, target = scores[eligible], target[eligible]
    best = min((float(nn.functional.cross_entropy(scores / float(t), target)), float(t))
               for t in np.exp(np.linspace(math.log(0.05), math.log(20), 101)))
    return {"temperature": best[1], "conditional_nll": best[0], "transitions": len(target),
            "scope": "conditional on target inclusion; validation only"}


class NeuralPredictor:
    """Identical persisted feature pipeline for offline and asynchronous runtime."""

    def __init__(self, checkpoint: str | Path, index: M.CandidateIndex, page_map: dict):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = RankerConfig(**payload["config"])
        if payload["normalization_max_page"] != index.max_page:
            raise ValueError("Ranker/index normalization mismatch")
        self.model = M.MLPScorer(payload["input_dim"], hidden=self.config.hidden)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.index, self.page_map = index, page_map
        if index.static_metadata_path:
            from .static_metadata import StaticMetadata, snapshot_identity
            self.page_map = StaticMetadata(index.static_metadata_path)
            if index.layout_database_path and self.page_map.snapshot_identity != snapshot_identity(index.layout_database_path):
                raise ValueError("Static metadata belongs to a different database snapshot")
        self.counts = payload["counts"]
        self.parameter_scales = payload["parameter_scales"]
        self.temperature = float(payload.get("calibration", {}).get("temperature", 1.0))

    def rank(self, history: list[int], context: dict) -> list[tuple[int, float]]:
        if not history:
            return []
        candidates = self.index.candidates(history, layout_children=context.get("layout_children"))
        if not candidates:
            return []
        context = {**context, **ML.prediction_context(context)}
        x = group_features(history, candidates, context, self.page_map, self.index,
                           self.counts, self.config, self.parameter_scales)
        prior = torch.from_numpy(markov_log_prior(history, candidates, self.counts))
        with torch.inference_mode():
            scores = self.model(torch.from_numpy(x)) + self.config.prior_weight * prior
            probabilities = torch.softmax(scores / self.temperature, dim=0).numpy()
        order = np.argsort(-probabilities, kind="stable")
        return [(candidates[i], float(probabilities[i])) for i in order]
