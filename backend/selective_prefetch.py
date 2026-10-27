"""Opt-in selective neural/Markov prefetching for the existing VFS runtime.

This module deliberately leaves frozen pagewise/ source and legacy strategies
untouched. All decisions use observed history, training counts and a declared
development calibration; no future accesses are supplied to the controller.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from pagewise.runtime import PredictiveRuntime
from selective_adaptive import choose_action
from selective_deadline import deadline_feasible

POLICIES = {"context", "deadline", "adaptive", "combined"}


class SelectiveRuntime(PredictiveRuntime):
    """Bounded asynchronous worker with optional context, deadlines and costs."""

    def __init__(self, *args, policy: str, policy_config: dict, **kwargs):
        if policy not in POLICIES:
            raise ValueError(f"Unsupported selective policy {policy}")
        self.policy = policy
        self.policy_config = policy_config
        self.max_jobs_per_query = max(1, int(policy_config.get("max_jobs_per_query", 8)))
        self._query_attempts: dict[str, int] = {}
        self._policy_reasons: Counter[str] = Counter()
        self._demand_waiters = 0
        super().__init__(*args, **kwargs)
        self._stats.update({"policy_skipped": 0, "policy_markov2": 0, "policy_markov1": 0,
                            "policy_neural": 0, "policy_deadline_dropped": 0,
                            "policy_query_limit_dropped": 0, "policy_query_completed_dropped": 0,
                            "policy_duplicate_candidates": 0, "policy_cached_candidates": 0,
                            "policy_pending_candidates": 0, "policy_device_busy_dropped": 0,
                            "policy_pages_read": 0, "policy_late_read_discarded": 0,
                            "policy_net_decisions": 0, "policy_estimated_benefit_ms_total": 0.0,
                            "policy_estimated_cost_ms_total": 0.0, "policy_estimated_net_ms_total": 0.0,
                            "policy_error_details": []})
        device = self.storage_device
        if device is not None and device.enabled:
            original_access = device.access
            def demand_aware_access(source="demand", pages=1):
                if source != "demand":
                    return original_access(source, pages)
                with self._lock:
                    self._demand_waiters += 1
                try:
                    return original_access(source, pages)
                finally:
                    with self._lock:
                        self._demand_waiters -= 1
            device.access = demand_aware_access

    def _reason(self, reason):
        with self._lock:
            self._policy_reasons[reason] += 1

    def set_active_query(self, query_id):
        with self._lock:
            if query_id != self._active_query_id:
                self._query_attempts.clear()
            self._active_query_id = query_id

    def request_prediction(self, history, context):
        if not history or self._closed:
            return
        requested_ns = time.perf_counter_ns()
        qid = context.get("query_id")
        family = str(context.get("workload_family") or "unknown")
        if qid is None:
            self._reason("missing_query_id")
            return
        qid = str(qid)
        with self._lock:
            if self._active_query_id != qid:
                self._stats["policy_query_completed_dropped"] += 1
                self._policy_reasons["query_inactive_at_request"] += 1
                return
            n = self._query_attempts.get(qid, 0)
            if n >= self.max_jobs_per_query:
                self._stats["policy_query_limit_dropped"] += 1
                self._policy_reasons["per_query_limit"] += 1
                return
            self._query_attempts[qid] = n + 1
            queue_fraction = self._queue.qsize() / self._queue.maxsize
        cache_fraction = 0.0
        if self.policy in {"adaptive", "combined"}:
            cache_state = self.cache.snapshot()
            cache_fraction = cache_state["unused_prefetches_still_resident"] / cache_state["capacity_pages"]
        action, reason, net = choose_action(self.index, history, family, self.policy, self.policy_config,
                                            queue_fraction=queue_fraction, cache_fraction=cache_fraction)
        self._reason(reason)
        if net is not None:
            with self._lock:
                self._stats["policy_net_decisions"] += 1
                self._stats["policy_estimated_benefit_ms_total"] += net["expected_benefit_ms"]
                self._stats["policy_estimated_cost_ms_total"] += net["estimated_cost_ms"]
                self._stats["policy_estimated_net_ms_total"] += net["estimated_net_ms"]
        if action == "skip":
            with self._lock:
                self._stats["policy_skipped"] += 1
            return
        budget = self.policy_config.get("deadlines_ms", {}).get(family)
        expected = self.policy_config.get("runtime_cost_ms", {}).get(action, {})
        if self.policy in {"deadline", "combined"}:
            remaining = float(expected.get("score_ms", 0)) + float(expected.get("read_ms", 0))
            if not deadline_feasible(time.perf_counter_ns(), requested_ns, budget, remaining):
                with self._lock:
                    self._stats["policy_deadline_dropped"] += 1
                self._reason("deadline_before_enqueue")
                return
        job_context = {**context, "selective_action": action, "selective_reason": reason,
                       "selective_request_ns": requested_ns, "deadline_budget_ms": budget,
                       "estimated_net": net}
        # Parent performs bounded admission, duplicate job coalescing, and
        # captures observed B-tree children from the already-read page.
        super().request_prediction(history, job_context)

    def _expired(self, context, remaining_ms=0.0):
        if self.policy not in {"deadline", "combined"}:
            return False
        return not deadline_feasible(time.perf_counter_ns(), context["selective_request_ns"],
                                     context.get("deadline_budget_ms"), remaining_ms)

    def _rank_neural(self, history, context):
        """Filter cached/pending pages before constructing neural features."""
        import numpy as np
        import torch
        from pagewise import ml_variants as ML
        from pagewise.ranking import group_features, markov_log_prior
        candidates = self.index.candidates(history, layout_children=context.get("layout_children"))
        scoreable = []
        with self._lock:
            for page in candidates:
                if page in self._pending:
                    self._stats["policy_pending_candidates"] += 1
                elif self.cache.contains(page):
                    self._stats["policy_cached_candidates"] += 1
                else:
                    scoreable.append(page)
        if not scoreable:
            return []
        predictor = self.ranker
        local = {**context, **ML.prediction_context(context)}
        features = group_features(history, scoreable, local, predictor.page_map, self.index,
                                  predictor.counts, predictor.config, predictor.parameter_scales)
        prior = torch.from_numpy(markov_log_prior(history, scoreable, predictor.counts))
        with torch.inference_mode():
            scores = predictor.model(torch.from_numpy(features)) + predictor.config.prior_weight * prior
            probabilities = torch.softmax(scores / predictor.temperature, dim=0).numpy()
        return [(scoreable[i], float(probabilities[i])) for i in np.argsort(-probabilities, kind="stable")]

    def _rank(self, action, history, context):
        if action == "neural":
            return self._rank_neural(history, context)
        if action == "markov2":
            pages = self.index.second_order.get(f"{history[-2]}:{history[-1]}", [])
        else:
            pages = self.index.successors.get(history[-1], [])
        return [(page, 1.0 / rank) for rank, page in enumerate(pages, 1)]

    def _try_device(self):
        """Never queue a speculative simulated service behind active demand.

        An already-started operation cannot be preempted; OS reads themselves
        are outside this artificial device model and have no priority guarantee.
        """
        device = self.storage_device
        if device is None or not device.enabled:
            return True
        with self._lock:
            if self._demand_waiters:
                return False
        if not device._lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if self._demand_waiters:
                    return False
            duration = device.delay_ms / 1000.0
            time.sleep(duration)
            device.stats["prefetch_ios"] += 1
            device.stats["device_busy_ms_total"] += duration * 1000.0
            return True
        finally:
            device._lock.release()

    def _run_worker(self):
        try:
            with self.database_path.open("rb") as handle:
                while True:
                    job = self._queue.get()
                    if job is None:
                        self._queue.task_done()
                        break
                    pid, history, context, queue_ns, wall_ns, key = job
                    worker_ns = time.perf_counter_ns()
                    qid = context.get("query_id")
                    wait_ms = (worker_ns - queue_ns) / 1e6
                    action = context["selective_action"]
                    selected = []
                    ranked = []
                    inference_start = worker_ns
                    inference_done = worker_ns
                    status = "completed"
                    try:
                        with self._lock:
                            self._stats["queue_wait_ms_total"] += wait_ms
                            self._stats["queue_wait_ms_max"] = max(self._stats["queue_wait_ms_max"], wait_ms)
                            inactive = self._active_query_id != qid
                        if inactive:
                            status = "query_completed_dropped"
                            with self._lock:
                                self._stats["policy_query_completed_dropped"] += 1
                                self._stats["stale_jobs_detected"] += 1
                                self._stats["stale_jobs_dropped"] += 1
                            self._reason(status)
                            continue
                        expected = self.policy_config.get("runtime_cost_ms", {}).get(action, {})
                        if self._expired(context, float(expected.get("score_ms", 0)) + float(expected.get("read_ms", 0))):
                            status = "deadline_before_scoring"
                            with self._lock:
                                self._stats["policy_deadline_dropped"] += 1
                            self._reason(status)
                            continue
                        inference_start = time.perf_counter_ns()
                        ranked = self._rank(action, history, context)
                        inference_done = time.perf_counter_ns()
                        scored_ms = (inference_done - inference_start) / 1e6
                        with self._lock:
                            self._stats["inference_ms_total"] += scored_ms
                            self._stats["inference_ms_max"] = max(self._stats["inference_ms_max"], scored_ms)
                            self._stats[f"policy_{action}"] += 1
                        seen = set()
                        prefetched = 0
                        for page, score in ranked:
                            if prefetched >= self.degree:
                                break
                            if page in seen:
                                with self._lock:
                                    self._stats["policy_duplicate_candidates"] += 1
                                selected.append({"page": int(page), "score": float(score), "status": "duplicate"})
                                continue
                            seen.add(page)
                            if not 1 <= page <= self._database_pages:
                                selected.append({"page": int(page), "score": float(score), "status": "invalid_page"})
                                continue
                            with self._lock:
                                if self._active_query_id != qid:
                                    status = "query_completed_before_io"
                                    self._stats["policy_query_completed_dropped"] += 1
                                    break
                                if page in self._pending:
                                    self._stats["policy_pending_candidates"] += 1
                                    selected.append({"page": int(page), "score": float(score), "status": "pending"})
                                    continue
                                if self.cache.contains(page):
                                    self._stats["policy_cached_candidates"] += 1
                                    selected.append({"page": int(page), "score": float(score), "status": "cached"})
                                    continue
                                self._pending.add(page)
                            try:
                                if self._expired(context, float(expected.get("read_ms", 0))):
                                    status = "deadline_before_io"
                                    with self._lock:
                                        self._stats["policy_deadline_dropped"] += 1
                                    self._reason(status)
                                    break
                                start = time.perf_counter_ns()
                                if not self._try_device():
                                    with self._lock:
                                        self._stats["policy_device_busy_dropped"] += 1
                                    selected.append({"page": int(page), "score": float(score), "status": "device_busy"})
                                    break
                                with self._lock:
                                    inactive = self._active_query_id != qid
                                expired = self._expired(context)
                                if inactive or expired:
                                    status = "query_completed_after_service" if inactive else "deadline_after_service"
                                    with self._lock:
                                        self._stats["policy_query_completed_dropped" if inactive else "policy_deadline_dropped"] += 1
                                    self._reason(status)
                                    break
                                handle.seek((page - 1) * self.page_size)
                                data = handle.read(self.page_size)
                                done = time.perf_counter_ns()
                                read_ms = (done - start) / 1e6
                                with self._lock:
                                    self._stats["prefetch_read_ms_total"] += read_ms
                                    self._stats["prefetch_read_ms_max"] = max(self._stats["prefetch_read_ms_max"], read_ms)
                                    self._stats["policy_pages_read"] += 1
                                if len(data) != self.page_size:
                                    selected.append({"page": int(page), "score": float(score), "status": "short_read"})
                                    continue
                                with self._lock:
                                    inactive = self._active_query_id != qid
                                expired = self._expired(context)
                                if inactive or expired:
                                    status = "query_completed_after_read" if inactive else "deadline_after_read"
                                    with self._lock:
                                        self._stats["policy_late_read_discarded"] += 1
                                        self._stats["policy_query_completed_dropped" if inactive else "policy_deadline_dropped"] += 1
                                    selected.append({"page": int(page), "score": float(score), "status": status,
                                                     "page_read_start_ns": start, "page_read_done_ns": done, "page_read_ms": read_ms})
                                    self._reason(status)
                                    break
                                inserted = self.cache.put(page, data, origin="prefetch", metadata={
                                    "prediction_id": pid, "query_id": qid, "template_id": context.get("template_id"),
                                    "request_ns": context["selective_request_ns"], "worker_start_ns": worker_ns,
                                    "queue_wait_ms": wait_ms, "inference_start_ns": inference_start,
                                    "inference_done_ns": inference_done, "inference_ms": scored_ms,
                                    "page_read_start_ns": start, "page_read_done_ns": done, "page_read_ms": read_ms})
                                if inserted:
                                    prefetched += 1
                                selected.append({"page": int(page), "score": float(score), "status": "prefetched" if inserted else "filled_by_demand",
                                                 "page_read_start_ns": start, "page_read_done_ns": done, "page_read_ms": read_ms})
                            finally:
                                with self._lock:
                                    self._pending.discard(page)
                    except Exception as exc:
                        status = "worker_error"
                        with self._lock:
                            self._stats["prefetch_read_errors"] += 1
                            self._stats["policy_error_details"].append(f"{type(exc).__name__}: {exc}")
                    finally:
                        with self._lock:
                            self._stats["completed_jobs"] += 1
                            self._queued_prediction_keys.discard(key)
                        try:
                            self._write_prediction({"prediction_id": pid, "query_id": qid, "template_id": context.get("template_id"),
                                "workload_family": context.get("workload_family"), "query_page_position": context.get("query_page_position"),
                                "history": history, "policy": self.policy, "action": action, "reason": context.get("selective_reason"),
                                "status": status, "request_ns": context["selective_request_ns"], "queue_entry_ns": queue_ns,
                                "request_wall_ns": wall_ns, "worker_start_ns": worker_ns, "inference_start_ns": inference_start,
                                "inference_done_ns": inference_done, "queue_wait_ms": wait_ms,
                                "inference_ms": (inference_done - inference_start) / 1e6,
                                "deadline_budget_ms": context.get("deadline_budget_ms"), "estimated_net": context.get("estimated_net"),
                                "ranked_candidates": [{"page": int(p), "score": float(s)} for p, s in ranked[:8]], "selected": selected})
                        except OSError as exc:
                            with self._lock:
                                self._stats["prefetch_read_errors"] += 1
                                self._stats["policy_error_details"].append(f"prediction log: {exc}")
                        finally:
                            self._queue.task_done()
        except OSError as exc:
            with self._lock:
                self._stats["prefetch_read_errors"] += 1
                self._stats["policy_error_details"].append(f"{type(exc).__name__}: {exc}")

    def snapshot(self):
        result = super().snapshot()
        with self._lock:
            result["selective_policy"] = self.policy
            result["policy_reasons"] = dict(self._policy_reasons)
            result["max_jobs_per_query"] = self.max_jobs_per_query
        return result


def install_runtime(runner_module, policy, config):
    """Install the opt-in adapter only in an isolated experiment process."""
    if policy not in POLICIES:
        raise ValueError(policy)
    def factory(*args, **kwargs):
        return SelectiveRuntime(*args, policy=policy, policy_config=config, **kwargs)
    runner_module.PredictiveRuntime = factory
