"""Bounded inference caches and opt-in selective scoring for the saved P4 model."""
from __future__ import annotations

from collections import Counter, OrderedDict
import time

import numpy as np
import torch

from pagewise import models as M
from pagewise.ranking import group_features


class BoundedMemo:
    def __init__(self, capacity):
        self.capacity, self.values, self.stats = capacity, OrderedDict(), Counter()

    def get(self, key, factory):
        if key in self.values:
            self.stats["hits"] += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.stats["misses"] += 1
        value = factory()
        self.values[key] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


class CachedSemantics:
    """Cache immutable page metadata separately from completed-query context."""
    def __init__(self, metadata, capacity=4096):
        self.metadata, self.pages, self.contexts = metadata, BoundedMemo(capacity), BoundedMemo(128)
        self.static_width = len(M.PAGE_TYPES) + 1 + 32

    def get(self, page, default=None):
        return self.metadata.get(page, default)

    def semantic_matrix(self, pages, context):
        key = tuple((name, context.get(name)) for name in ("plan_class", "template_id", "workload_family", "position_in_query"))
        tail = self.contexts.get(key, lambda: np.asarray(M.semantic_page_features(0, {}, context)[self.static_width:], np.float32))
        def static(page):
            # Use the original metadata's own vectorized representation where
            # available, preserving its exact frozen feature interpretation.
            if hasattr(self.metadata, "semantic_matrix"):
                return self.metadata.semantic_matrix([page], {})[0, :self.static_width].copy()
            return np.asarray(M.semantic_page_features(page, self.metadata, {})[:self.static_width], np.float32)
        head = np.stack([self.pages.get(int(page), lambda page=page: static(page)) for page in pages])
        return np.concatenate([head, np.broadcast_to(tail, (len(pages), len(tail)))], axis=1)


class OptimizedIdleHybrid:
    order, signal = 2, "optimized_neural_semantic_query"

    def __init__(self, base, *, selective=False, admission_budget=None, min_support=3, margin=.25):
        self.base, self.selective = base, selective
        self.admission_budget, self.min_support, self.margin = admission_budget, min_support, margin
        self.metadata = CachedSemantics(base.metadata)
        self.priors, self.features = BoundedMemo(128), BoundedMemo(4096)
        self.stats = Counter()

    def state(self, template, pages):
        return self.base.state(template, pages)

    def predict(self, history, *, max_pages=16, min_confidence=.25, exclude_pages=()):
        started = time.perf_counter_ns()
        states = tuple(row["template"] for row in history)
        proposal = self.priors.get((states, min_confidence), lambda: self.base.prior.predict(states,
            max_pages=max(1, len(self.base.prior.global_counts)), min_confidence=min_confidence))
        excluded = set(exclude_pages)
        candidates = [dict(p) for p in proposal["pages"] if p["page"] not in excluded][:self.base.candidate_limit]
        candidate_done = time.perf_counter_ns()
        budget = min(max_pages, self.admission_budget) if self.admission_budget is not None else max_pages
        if budget < 1:
            raise ValueError("Admission budget must be positive")
        common = {k: v for k, v in proposal.items() if k != "pages"}
        if not history or not candidates:
            return {**common, "pages": [], "neural_scored": False, "route": "no_candidates", "stage_ns": {}}
        # A transparent development-time rule, never a future-query hint:
        # skip neural ordering when all candidates fit, or the empirical prior
        # separates the admission boundary by the declared confidence margin.
        margin = candidates[budget-1]["confidence"] - candidates[budget]["confidence"] if len(candidates) > budget else None
        skip = self.selective and (len(candidates) <= budget or
            (proposal["support"] >= self.min_support and margin is not None and margin >= self.margin))
        if skip:
            self.stats["markov_only_jobs"] += 1
            return {**common, "pages": candidates[:budget], "neural_scored": False,
                    "route": "markov_all_candidates_fit" if len(candidates) <= budget else "markov_clear_boundary",
                    "stage_ns": {"candidate_ns": candidate_done-started}, "cache": self.snapshot()}
        latest = history[-1]
        context = {"template_id": latest["template"], "position_in_query": 0.0,
                   **self.base.contexts.get(latest["template"], {"workload_family": "unknown", "plan_class": "other"})}
        page_history = latest["pages"] or [0]
        keys = [(tuple(page_history), latest["template"], row["page"]) for row in candidates]
        ready = {key: self.features.get(key, lambda: None) for key in keys if key in self.features.values}
        missing = [(key, row["page"]) for key, row in zip(keys, candidates) if key not in self.features.values]
        if missing:
            matrix = group_features(page_history, [p for _, p in missing], context, self.metadata,
                self.base.index, self.base.counts, self.base.config, [1., 1., 1.])
            for (key, _), value in zip(missing, matrix):
                ready[key] = self.features.get(key, lambda value=value: value.copy())
        x = np.stack([ready[key] for key in keys])
        prior = np.asarray([np.log(max(1e-4, row["confidence"])) for row in candidates], np.float32)
        feature_done = time.perf_counter_ns()
        with torch.inference_mode():
            scores = self.base.model(torch.from_numpy(x)) + torch.from_numpy(prior)
            order = torch.argsort(scores, descending=True, stable=True).numpy()
        self.stats["neural_jobs"] += 1
        return {**common, "pages": [{**candidates[int(i)], "neural_markov_score": float(scores[int(i)])} for i in order[:budget]],
            "neural_scored": True, "route": "semantic_neural", "torch_threads_observed": torch.get_num_threads(),
            "stage_ns": {"candidate_ns": candidate_done-started, "feature_ns": feature_done-candidate_done,
                         "forward_and_sort_ns": time.perf_counter_ns()-feature_done}, "cache": self.snapshot()}

    def snapshot(self):
        return {**self.stats, "prior_hits": self.priors.stats["hits"], "feature_hits": self.features.stats["hits"],
                "feature_misses": self.features.stats["misses"], "feature_entries": len(self.features.values),
                "metadata_entries": len(self.metadata.pages.values)}
