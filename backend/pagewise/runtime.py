from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


class BoundedPageCache:
    """Thread-safe page cache with LRU eviction, prefetch-use accounting, and timestamped diagnostics."""

    def __init__(self, capacity_pages: int):
        if capacity_pages < 1:
            raise ValueError("capacity_pages must be >= 1")
        self.capacity = capacity_pages
        self._pages: OrderedDict[int, tuple[bytes, str, bool, dict[str, Any]]] = OrderedDict()
        self._demand_use_events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.stats = {
            "demand_requests": 0, "cache_hits": 0, "cache_misses": 0,
            "pages_prefetched": 0, "prefetch_bytes": 0, "useful_prefetches": 0,
            "wasted_prefetches": 0, "evictions": 0,
            "useful_prefetches_same_query": 0, "useful_prefetches_cross_query": 0,
            "demand_use_lead_time_ms_total": 0.0,
            "demand_use_lead_time_ms_min": -1.0,
            "demand_use_lead_time_ms_max": 0.0,
        }

    def contains(self, page: int) -> bool:
        with self._lock:
            return page in self._pages

    def peek(self, page: int) -> bytes | None:
        """Inspect already-observed bytes without changing LRU or hit statistics."""
        with self._lock:
            entry = self._pages.get(page)
            return entry[0] if entry is not None else None

    def get(self, page: int, *, demand: bool = True, query_id: str | None = None) -> bytes | None:
        with self._lock:
            entry = self._pages.get(page)
            if demand:
                self.stats["demand_requests"] += 1
            if entry is None:
                if demand:
                    self.stats["cache_misses"] += 1
                return None
            self._pages.move_to_end(page)
            data = entry[0]
            origin = entry[1]
            used = entry[2]
            meta = entry[3] if len(entry) > 3 else {}
            if demand:
                self.stats["cache_hits"] += 1
                if origin == "prefetch" and not used:
                    self.stats["useful_prefetches"] += 1
                    demand_use_ns = time.perf_counter_ns()
                    page_read_done_ns = meta.get("page_read_done_ns")
                    lead_time_ms = ((demand_use_ns - page_read_done_ns) / 1e6) if page_read_done_ns else None
                    if lead_time_ms is not None:
                        self.stats["demand_use_lead_time_ms_total"] += lead_time_ms
                        if lead_time_ms < self.stats["demand_use_lead_time_ms_min"] or self.stats["demand_use_lead_time_ms_min"] < 0:
                            self.stats["demand_use_lead_time_ms_min"] = lead_time_ms
                        if lead_time_ms > self.stats["demand_use_lead_time_ms_max"]:
                            self.stats["demand_use_lead_time_ms_max"] = lead_time_ms
                    same_query = bool(query_id is not None and query_id == meta.get("query_id"))
                    if same_query:
                        self.stats["useful_prefetches_same_query"] += 1
                    else:
                        self.stats["useful_prefetches_cross_query"] += 1
                    if len(self._demand_use_events) < 5000:
                        self._demand_use_events.append({
                            "page": int(page),
                            "demand_use_ns": demand_use_ns,
                            "prediction_id": meta.get("prediction_id"),
                            "prefetch_query_id": meta.get("query_id"),
                            "demand_query_id": query_id,
                            "same_query": same_query,
                            "lead_time_ms": round(lead_time_ms, 3) if lead_time_ms is not None else None,
                            "queue_wait_ms": meta.get("queue_wait_ms"),
                            "inference_ms": meta.get("inference_ms"),
                            "page_read_ms": meta.get("page_read_ms"),
                        })
                    self._pages[page] = (data, origin, True, meta)
            return data

    def put(self, page: int, data: bytes, *, origin: str, metadata: dict[str, Any] | None = None) -> bool:
        with self._lock:
            existing = self._pages.get(page)
            if existing is not None:
                self._pages.move_to_end(page)
                # Preserve a prior prefetch origin so a concurrent demand fill
                # doesn't erase its usefulness/waste accounting.
                return False
            while len(self._pages) >= self.capacity:
                old_page, old_entry = self._pages.popitem(last=False)
                self.stats["evictions"] += 1
                old_origin = old_entry[1]
                was_used = old_entry[2]
                if old_origin == "prefetch" and not was_used:
                    self.stats["wasted_prefetches"] += 1
            self._pages[page] = (bytes(data), origin, False, dict(metadata or {}))
            if origin == "prefetch":
                self.stats["pages_prefetched"] += 1
                self.stats["prefetch_bytes"] += len(data)
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            unused_resident = sum(1 for entry in self._pages.values() if entry[1] == "prefetch" and not entry[2])
            pages_pf = self.stats["pages_prefetched"]
            useful = self.stats["useful_prefetches"]
            wasted_evicted = self.stats["wasted_prefetches"]
            total_wasted = wasted_evicted + unused_resident
            demand_reqs = self.stats["demand_requests"]
            lead_total = self.stats["demand_use_lead_time_ms_total"]
            lead_min = self.stats["demand_use_lead_time_ms_min"] if self.stats["demand_use_lead_time_ms_min"] >= 0 else None

            return {
                **self.stats,
                "capacity_pages": self.capacity,
                "resident_pages": len(self._pages),
                "unused_prefetches_still_resident": unused_resident,
                "total_wasted_prefetches": total_wasted,
                "useful_prefetch_pct": round(100.0 * useful / pages_pf, 3) if pages_pf > 0 else 0.0,
                "wasted_prefetch_evicted_pct": round(100.0 * wasted_evicted / pages_pf, 3) if pages_pf > 0 else 0.0,
                "unused_prefetch_resident_pct": round(100.0 * unused_resident / pages_pf, 3) if pages_pf > 0 else 0.0,
                "wasted_prefetch_total_pct": round(100.0 * total_wasted / pages_pf, 3) if pages_pf > 0 else 0.0,
                "prefetch_hit_coverage_pct": round(100.0 * useful / demand_reqs, 3) if demand_reqs > 0 else 0.0,
                "cache_hit_pct": round(100.0 * self.stats["cache_hits"] / demand_reqs, 3) if demand_reqs > 0 else 0.0,
                "demand_use_lead_time_ms_mean": round(lead_total / useful, 3) if useful > 0 else None,
                "demand_use_lead_time_ms_min": round(lead_min, 3) if lead_min is not None else None,
                "demand_use_lead_time_ms_max": round(self.stats["demand_use_lead_time_ms_max"], 3) if useful > 0 else None,
                "demand_use_events_count": len(self._demand_use_events),
            }


class SimulatedStorageDevice:
    """Thread-safe simulated storage device supporting shared contention or independent delay.

    When storage_model is 'shared', both demand miss reads and prefetch reads must
    acquire the device. If a prefetch read is currently occupying the channel, a
    concurrent demand miss read waits until the prefetch finishes, accurately reflecting
    storage bus / head contention.
    """

    def __init__(self, delay_ms: float = 0.0, enabled: bool = False):
        self.delay_ms = max(0.0, float(delay_ms))
        self.enabled = bool(enabled and self.delay_ms > 0)
        self._lock = threading.Lock()
        self.stats = {
            "demand_ios": 0,
            "prefetch_ios": 0,
            "demand_wait_ms_total": 0.0,
            "demand_wait_ms_max": 0.0,
            "prefetch_wait_ms_total": 0.0,
            "prefetch_wait_ms_max": 0.0,
            "device_busy_ms_total": 0.0,
        }

    def access(self, source: str = "demand", pages: int = 1) -> float:
        """Simulate accessing the device. Returns wait time in milliseconds spent in queue."""
        if not self.enabled or self.delay_ms <= 0:
            return 0.0
        start_wait = time.perf_counter_ns()
        with self._lock:
            wait_ms = (time.perf_counter_ns() - start_wait) / 1e6
            service_duration = (self.delay_ms * max(1, pages)) / 1000.0
            time.sleep(service_duration)
            service_ms = service_duration * 1000.0
            if source == "demand":
                self.stats["demand_ios"] += 1
                self.stats["demand_wait_ms_total"] += wait_ms
                if wait_ms > self.stats["demand_wait_ms_max"]:
                    self.stats["demand_wait_ms_max"] = wait_ms
            else:
                self.stats["prefetch_ios"] += 1
                self.stats["prefetch_wait_ms_total"] += wait_ms
                if wait_ms > self.stats["prefetch_wait_ms_max"]:
                    self.stats["prefetch_wait_ms_max"] = wait_ms
            self.stats["device_busy_ms_total"] += service_ms
            return wait_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stats = dict(self.stats)
        stats["delay_ms"] = self.delay_ms
        stats["enabled"] = self.enabled
        demand_ios = max(1, stats["demand_ios"])
        prefetch_ios = max(1, stats["prefetch_ios"])
        stats["demand_wait_ms_mean"] = round(stats["demand_wait_ms_total"] / demand_ios, 3) if stats["demand_ios"] else 0.0
        stats["prefetch_wait_ms_mean"] = round(stats["prefetch_wait_ms_total"] / prefetch_ios, 3) if stats["prefetch_ios"] else 0.0
        return stats


class PredictiveRuntime:
    """Asynchronous candidate-ranking worker used by the optional VFS cache mode.

    Prefetch reads use a separate ordinary read-only file handle and therefore do
    not recursively pass through the wrapping VFS. Use only with a static database.
    """

    def __init__(self, database_path: str | Path, page_size: int, page_map_path: str | Path,
                 candidate_index_path: str | Path, model_path: str | Path | None = None, *,
                 semantic: bool = False, strategy: str = "model_a", cache_pages: int = 256, degree: int = 2,
                 confidence_threshold: float = 0.0, queue_size: int = 64,
                 miss_delay_ms: float = 0.0, prediction_log_path: str | Path | None = None,
                 predict_on: str = "all", predict_every: int = 1,
                 max_score_candidates: int = 0,
                 storage_model: str = "independent",
                 storage_channels: int | None = None,
                 storage_device: SimulatedStorageDevice | None = None,
                 drop_stale_jobs: bool = False, oracle_trace_path: str | Path | None = None,
                 diagnostics_enabled: bool = True,
                 oracle_miss_path: str | Path | None = None,
                 record_demand_reference: bool = False,
                 prefetch_workers: int = 1):
        from .models import CandidateIndex, load_model, page_map_digest

        if strategy == "oracle_optimal":
            raise ValueError("oracle_optimal is a simulator-only action-family comparison; use oracle_miss for runtime diagnostics")
        supported = {"none", "null", "oracle", "oracle_next", "oracle_miss", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid", "model_ranker"}
        if strategy not in supported:
            raise ValueError(f"Unsupported strategy: {strategy}; expected one of {sorted(supported)}")
        self.database_path = Path(database_path).resolve()
        self.page_size = int(page_size)
        self._database_pages = self.database_path.stat().st_size // self.page_size
        self.page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
        index_payload = json.loads(Path(candidate_index_path).read_text(encoding="utf-8"))
        self.index = CandidateIndex.from_json(index_payload)
        if self.index.layout_database_path:
            self.index.layout_database_path = str(self.database_path)
        expected_hash = str(index_payload.get("page_map_sha256", ""))
        actual_hash = page_map_digest(self.page_map)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("Page-map fingerprint mismatch. The model was trained with a different database layout; use the same static database snapshot.")
        self.strategy = strategy
        self.model = None
        self.ranker = None
        if strategy == "model_ranker":
            from .ranking import NeuralPredictor
            if not model_path:
                raise ValueError("model_ranker requires a checkpoint")
            self.ranker = NeuralPredictor(model_path, self.index, self.page_map)
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self._oracle_sequences: dict[str, list[int]] = {}
        self._oracle_observed_positions: dict[str, int] = {}
        if oracle_trace_path:
            trace_path = Path(oracle_trace_path).resolve()
            if not trace_path.exists():
                raise FileNotFoundError(trace_path)
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = row.get("query_id")
                first_page = row.get("first_page")
                last_page = row.get("last_page")
                if not query_id or not isinstance(first_page, int) or not isinstance(last_page, int):
                    continue
                self._oracle_sequences.setdefault(str(query_id), []).extend(range(first_page, last_page + 1))
        if strategy in {"oracle", "oracle_next"} and not self._oracle_sequences:
            raise ValueError("oracle_next requires a non-empty oracle trace")
        # oracle_miss uses a `none`-run demand reference: per query, the ordered
        # demand page stream plus whether each page missed under the same cache.
        self._oracle_miss: dict[str, list[list[Any]]] = {}
        if oracle_miss_path:
            om_path = Path(oracle_miss_path).resolve()
            if not om_path.exists():
                raise FileNotFoundError(om_path)
            payload = json.loads(om_path.read_text(encoding="utf-8"))
            for qid, seq in payload.items():
                self._oracle_miss[str(qid)] = [[int(entry[0]), bool(entry[1])] for entry in seq]
        if strategy == "oracle_miss" and not self._oracle_miss:
            raise ValueError("oracle_miss requires a non-empty demand reference")
        self.recording = bool(record_demand_reference)
        self._demand_reference: dict[str, list[list[Any]]] = {}
        self.model_path = str(Path(model_path).resolve()) if model_path else None
        self.semantic = semantic
        if strategy in {"model_a", "model_b", "hybrid"}:
            if not model_path:
                raise ValueError(f"{strategy} requires --model")
            self.model, model_semantic = load_model(model_path)
            if model_semantic != semantic:
                raise ValueError(f"Model semantic flag mismatch: model semantic={model_semantic}, configured semantic={semantic}")
            if strategy == "hybrid" and not semantic:
                raise ValueError("hybrid requires a semantic Model B checkpoint")
        self.cache = BoundedPageCache(cache_pages)
        self.degree = max(1, int(degree))
        self.confidence_threshold = float(confidence_threshold)
        self.miss_delay_ms = max(0.0, float(miss_delay_ms))
        if predict_on not in {"all", "miss", "hit"}:
            raise ValueError("predict_on must be one of: all, miss, hit")
        self.predict_on = predict_on
        self.predict_every = max(1, int(predict_every))
        self.max_score_candidates = max(0, int(max_score_candidates))
        self.storage_model = storage_model
        self.storage_channels = int(storage_channels or (1 if storage_model == "shared" else 0))
        if storage_device is None:
            if str(storage_model).startswith("k_channel") and self.miss_delay_ms > 0:
                # v2.1 abstract k-channel model: k concurrent service slots.
                from .storage import KChannelStorageDevice
                self.storage_device = KChannelStorageDevice(self.miss_delay_ms,
                                                            channels=self.storage_channels or 4,
                                                            enabled=True)
                self.storage_model = self.storage_device.storage_kind
            elif storage_model == "shared" and self.miss_delay_ms > 0:
                self.storage_device = SimulatedStorageDevice(self.miss_delay_ms, enabled=True)
            else:
                self.storage_device = None
        else:
            self.storage_device = storage_device
            self.storage_device = storage_device
        self.drop_stale_jobs = bool(drop_stale_jobs)
        self._active_query_id: str | None = None
        self._prediction_observations = 0
        self._queue: queue.Queue[tuple[int, list[int], dict[str, Any], int, int, tuple[Any, ...]] | None] = queue.Queue(maxsize=max(1, queue_size))
        self._pending: set[int] = set()
        self._queued_prediction_keys: set[tuple[Any, ...]] = set()
        self._lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._closed = False
        self._worker_started_ns = time.perf_counter_ns()
        self._worker_stopped_ns: int | None = None
        self._prediction_id = 0
        self._prediction_path = Path(prediction_log_path).resolve() if prediction_log_path and self.diagnostics_enabled else None
        if self._prediction_path:
            self._prediction_path.parent.mkdir(parents=True, exist_ok=True)
            self._prediction_path.write_text("", encoding="utf-8")
        self._stats = {
            "prediction_jobs": 0, "prediction_requests": 0, "queued_prediction_jobs": 0,
            "completed_jobs": 0, "dropped_jobs": 0, "coalesced_jobs": 0,
            "prediction_observations": 0, "skipped_prediction_observations": 0,
            "candidates_pruned_cached_or_pending": 0,
            "candidates_pre_ranked_out": 0,
            "hybrid_markov_jobs": 0, "hybrid_semantic_jobs": 0,
            "inference_ms_total": 0.0, "inference_ms_max": 0.0,
            "queue_wait_ms_total": 0.0, "queue_wait_ms_max": 0.0,
            "prefetch_read_ms_total": 0.0, "prefetch_read_ms_max": 0.0,
            "prefetch_read_errors": 0,
            "stale_jobs_detected": 0, "stale_jobs_dropped": 0,
            "demand_misses_while_prefetch_in_flight": 0,
            "oracle_alignment_misses": 0,
            "worker_shutdown_clean": False,
        }
        # Prefetch reads may run on several threads so that a parallel device
        # (independent model) can actually serve them concurrently. Under the
        # shared model the SimulatedStorageDevice lock still serializes access,
        # so the pool only removes throughput limits when the device allows it.
        self.prefetch_workers = max(1, int(prefetch_workers))
        self._workers = [
            threading.Thread(target=self._run_worker, name=f"pagewise-prefetch-{index}", daemon=True)
            for index in range(self.prefetch_workers)
        ]
        for worker in self._workers:
            worker.start()

    def set_active_query(self, query_id: str | None) -> None:
        """Inform the runtime of the currently executing query in the VFS."""
        with self._lock:
            self._active_query_id = query_id

    def is_page_pending(self, page: int) -> bool:
        """Check whether a page is currently being prefetched."""
        with self._lock:
            return page in self._pending

    def record_demand_miss_in_flight(self, page: int) -> bool:
        """Record if a demand cache miss occurred while the page was already in flight."""
        with self._lock:
            if page in self._pending:
                self._stats["demand_misses_while_prefetch_in_flight"] += 1
                return True
            return False

    def should_predict(self, access_status: str) -> bool:
        """Return whether an observed page access should schedule prediction work."""
        normalized = "hit" if access_status == "hit" else "miss"
        with self._lock:
            self._prediction_observations += 1
            self._stats["prediction_observations"] += 1
            if self.predict_on != "all" and normalized != self.predict_on:
                self._stats["skipped_prediction_observations"] += 1
                return False
            if (self._prediction_observations - 1) % self.predict_every != 0:
                self._stats["skipped_prediction_observations"] += 1
                return False
            return True

    def request_prediction(self, history: list[int], context: dict[str, Any]) -> None:
        if len(history) < 1 or self._closed:
            return
        if self.strategy == "none":
            return
        if self.strategy == "model_ranker" and self.index.strategy == "layout":
            from .layout import parse_children
            observed = self.cache.peek(history[-1])
            context = {**context, "layout_children": list(parse_children(observed, history[-1], self._database_pages)) if observed else []}
        if self.strategy in {"oracle_miss"}:
            query_id = str(context.get("query_id"))
            position = context.get("query_page_position")
            seq = self._oracle_miss.get(query_id, [])
            aligned = isinstance(position, int) and 1 <= position <= len(seq) and int(seq[position - 1][0]) == int(history[-1])
            if not aligned:
                with self._lock:
                    self._stats["oracle_alignment_misses"] += 1
                return
        if self.strategy in {"oracle", "oracle_next"}:
            query_id = str(context.get("query_id"))
            position = context.get("query_page_position")
            sequence = self._oracle_sequences.get(query_id, [])
            aligned = isinstance(position, int) and position >= 1 and position <= len(sequence)
            if aligned:
                suffix = history[-min(8, position):]
                aligned = sequence[position - len(suffix):position] == suffix
            if not aligned:
                # A single mismatched observation must not crash the query. Stop
                # prefetching for this query and record the divergence so the
                # oracle bound can be reported as unreliable when it is large.
                with self._lock:
                    self._stats["oracle_alignment_misses"] += 1
                return
            self._oracle_observed_positions[query_id] = position
        compact_history = tuple(int(page) for page in history[-8:])
        prediction_key = (
            compact_history,
            context.get("query_id"),
            context.get("template_id"),
            context.get("workload_family"),
            context.get("plan_class"),
            round(float(context.get("position_in_query", 0.0) or 0.0), 3),
        )
        request_ns = time.perf_counter_ns()
        request_wall_ns = time.time_ns()
        with self._lock:
            self._stats["prediction_requests"] += 1
            if prediction_key in self._queued_prediction_keys:
                self._stats["coalesced_jobs"] += 1
                return
            self._stats["prediction_jobs"] += 1
            self._prediction_id += 1
            prediction_id = self._prediction_id
            self._queued_prediction_keys.add(prediction_key)
            # Count the enqueue optimistically here so the hot path needs a
            # single lock acquisition; roll it back only if the queue is full.
            self._stats["queued_prediction_jobs"] += 1
        job = (prediction_id, list(compact_history), dict(context), request_ns, request_wall_ns, prediction_key)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._stats["dropped_jobs"] += 1
                self._stats["queued_prediction_jobs"] -= 1
                self._queued_prediction_keys.discard(prediction_key)

    def _run_worker(self) -> None:
        from .models import score_candidates
        try:
            with self.database_path.open("rb") as raw_file:
                while True:
                    job = self._queue.get()
                    if job is None:
                        self._queue.task_done()
                        break
                    prediction_id, history, context, request_ns, request_wall_ns, prediction_key = job
                    worker_start_ns = time.perf_counter_ns()
                    queue_wait_ms = (worker_start_ns - request_ns) / 1e6
                    job_query_id = context.get("query_id")
                    with self._lock:
                        self._queued_prediction_keys.discard(prediction_key)
                        current_active = self._active_query_id
                        self._stats["queue_wait_ms_total"] += queue_wait_ms
                        if queue_wait_ms > self._stats["queue_wait_ms_max"]:
                            self._stats["queue_wait_ms_max"] = queue_wait_ms
                        is_stale_query = bool(current_active is not None and job_query_id != current_active)
                        if is_stale_query:
                            self._stats["stale_jobs_detected"] += 1
                            if self.drop_stale_jobs:
                                self._stats["stale_jobs_dropped"] += 1
                                self._stats["completed_jobs"] += 1
                    if is_stale_query:
                        if self.drop_stale_jobs:
                            self._write_prediction({
                                "prediction_id": prediction_id, "query_id": job_query_id,
                                "template_id": context.get("template_id"), "workload_family": context.get("workload_family"),
                                "history": history, "policy": self.strategy,
                                "status": "stale_query_dropped",
                                "request_ns": request_ns, "request_wall_ns": request_wall_ns,
                                "worker_start_ns": worker_start_ns, "queue_wait_ms": round(queue_wait_ms, 3),
                                "inference_ms": 0.0, "ranked_candidates": [], "selected": [],
                            })
                            self._queue.task_done()
                            continue

                    selected: list[dict[str, Any]] = []
                    inference_start_ns = time.perf_counter_ns()
                    try:
                        if self.strategy == "null":
                            ranked = []
                        elif self.strategy in {"oracle", "oracle_next"}:
                            sequence = self._oracle_sequences.get(str(context.get("query_id")), [])
                            position = int(context.get("query_page_position") or 0)
                            next_pages = sequence[position:position + self.degree]
                            ranked = []
                            seen: set[int] = set()
                            for rank, page in enumerate(next_pages, start=1):
                                if page not in seen:
                                    ranked.append((page, 1.0 / rank))
                                    seen.add(page)
                        elif self.strategy in {"oracle_miss"}:
                            sequence = self._oracle_miss.get(str(context.get("query_id")), [])
                            position = int(context.get("query_page_position") or 0)
                            # Next `degree` pages that would miss under `none`, after the
                            # current demand position. Already-cached axes are skipped by
                            # the generic selection loop below.
                            miss_pages: list[int] = []
                            for entry in sequence[position:]:
                                if bool(entry[1]):
                                    miss_pages.append(int(entry[0]))
                                    if len(miss_pages) >= self.degree:
                                        break
                            ranked = [(page, 1.0 - 0.01 * rank) for rank, page in enumerate(miss_pages)]
                        elif self.strategy == "sequential":
                            ranked = [(history[-1] + 1, 1.0)] if history else []
                        elif self.strategy == "markov1":
                            ranked = [(p, 1.0 / rank) for rank, p in enumerate(self.index.successors.get(history[-1], []) if history else [], start=1)]
                        elif self.strategy == "markov2":
                            key = f"{history[-2]}:{history[-1]}" if len(history) >= 2 else ""
                            candidates = self.index.second_order.get(key, []) or self.index.successors.get(history[-1], []) if history else []
                            ranked = [(p, 1.0 / rank) for rank, p in enumerate(candidates, start=1)]
                        elif self.strategy == "model_ranker":
                            ranked = self.ranker.rank(history, context)
                        elif self.strategy == "hybrid" and len(history) >= 2 and self.index.second_order.get(f"{history[-2]}:{history[-1]}"):
                            candidates = self.index.second_order[f"{history[-2]}:{history[-1]}"]
                            ranked = [(p, 1.0 / rank) for rank, p in enumerate(candidates, start=1)]
                            with self._lock:
                                self._stats["hybrid_markov_jobs"] += 1
                        elif self.strategy in {"model_a", "model_b", "hybrid"}:
                            candidates = self.index.candidates(history)
                            scoreable = []
                            pruned = 0
                            with self._lock:
                                for page in candidates:
                                    if page in self._pending or self.cache.contains(page):
                                        pruned += 1
                                    else:
                                        scoreable.append(page)
                                self._stats["candidates_pruned_cached_or_pending"] += pruned
                            candidates = scoreable
                            if self.max_score_candidates > 0 and len(candidates) > self.max_score_candidates:
                                last_page = history[-1] if history else 0
                                successor_order = self.index.successors.get(last_page, [])
                                order_rank = {p: idx for idx, p in enumerate(successor_order)}
                                candidates.sort(key=lambda p: order_rank.get(p, len(successor_order)))
                                pre_ranked_out = len(candidates) - self.max_score_candidates
                                candidates = candidates[:self.max_score_candidates]
                                with self._lock:
                                    self._stats["candidates_pre_ranked_out"] += pre_ranked_out
                            ranked = score_candidates(self.model, history, candidates, context, self.page_map,
                                                      self.index.max_page, self.semantic,
                                                      disabled_semantic_groups=getattr(self.model, "disabled_semantic_groups", []))
                            if self.strategy == "hybrid":
                                with self._lock:
                                    self._stats["hybrid_semantic_jobs"] += 1
                        else:
                            ranked = []
                        inference_done_ns = time.perf_counter_ns()
                        scored_ms = (inference_done_ns - inference_start_ns) / 1e6
                        with self._lock:
                            self._stats["inference_ms_total"] += scored_ms
                            if scored_ms > self._stats["inference_ms_max"]:
                                self._stats["inference_ms_max"] = scored_ms
                            self._stats["completed_jobs"] += 1
                        prefetches = 0
                        threshold = self.confidence_threshold if self.strategy in {"model_a", "model_b", "hybrid", "model_ranker", "oracle", "oracle_next", "oracle_miss"} else 0.0
                        for page, score in ranked[:max(self.degree * 4, 8)]:
                            if score < threshold or prefetches >= self.degree:
                                selected.append({"page": int(page), "score": float(score), "status": "below_threshold_or_degree"})
                                continue
                            if page < 1:
                                selected.append({"page": int(page), "score": float(score), "status": "invalid_page"})
                                continue
                            offset = (page - 1) * self.page_size
                            with self._lock:
                                if page in self._pending or self.cache.contains(page):
                                    selected.append({"page": int(page), "score": float(score), "status": "already_cached_or_pending"})
                                    continue
                                self._pending.add(page)
                            page_read_start_ns = time.perf_counter_ns()
                            device_wait_ms = 0.0
                            try:
                                if self.storage_device is not None and self.storage_device.enabled:
                                    device_wait_ms = self.storage_device.access("prefetch", 1)
                                elif self.miss_delay_ms:
                                    time.sleep(self.miss_delay_ms / 1000.0)
                                raw_file.seek(offset)
                                data = raw_file.read(self.page_size)
                                page_read_done_ns = time.perf_counter_ns()
                                read_duration_ms = (page_read_done_ns - page_read_start_ns) / 1e6
                                with self._lock:
                                    self._stats["prefetch_read_ms_total"] += read_duration_ms
                                    if read_duration_ms > self._stats["prefetch_read_ms_max"]:
                                        self._stats["prefetch_read_ms_max"] = read_duration_ms
                                if len(data) == self.page_size:
                                    self.cache.put(page, data, origin="prefetch", metadata={
                                        "page": page,
                                        "prediction_id": prediction_id,
                                        "query_id": job_query_id,
                                        "template_id": context.get("template_id"),
                                        "request_ns": request_ns,
                                        "request_wall_ns": request_wall_ns,
                                        "worker_start_ns": worker_start_ns,
                                        "queue_wait_ms": round(queue_wait_ms, 3),
                                        "inference_start_ns": inference_start_ns,
                                        "inference_done_ns": inference_done_ns,
                                        "inference_ms": round(scored_ms, 3),
                                        "page_read_start_ns": page_read_start_ns,
                                        "page_read_done_ns": page_read_done_ns,
                                        "device_wait_ms": round(device_wait_ms, 3),
                                        "page_read_ms": round(read_duration_ms, 3),
                                    })
                                    prefetches += 1
                                    selected.append({
                                        "page": int(page), "score": float(score), "status": "prefetched", "bytes": len(data),
                                        "page_read_start_ns": page_read_start_ns, "page_read_done_ns": page_read_done_ns,
                                        "page_read_ms": round(read_duration_ms, 3), "device_wait_ms": round(device_wait_ms, 3),
                                    })
                                else:
                                    selected.append({"page": int(page), "score": float(score), "status": "short_read", "bytes": len(data)})
                            except OSError as exc:
                                with self._lock:
                                    self._stats["prefetch_read_errors"] += 1
                                selected.append({"page": int(page), "score": float(score), "status": "error", "error": str(exc)})
                            finally:
                                with self._lock:
                                    self._pending.discard(page)
                        self._write_prediction({
                            "prediction_id": prediction_id, "query_id": context.get("query_id"),
                            "template_id": context.get("template_id"), "workload_family": context.get("workload_family"),
                            "history": history, "policy": self.strategy,
                            "request_ns": request_ns, "request_wall_ns": request_wall_ns,
                            "worker_start_ns": worker_start_ns,
                            "inference_start_ns": inference_start_ns, "inference_done_ns": inference_done_ns,
                            "inference_ms": round(scored_ms, 3),
                            "queue_wait_ms": round(queue_wait_ms, 3),
                            "inference_and_prefetch_ms": round((time.perf_counter_ns() - worker_start_ns) / 1e6, 3),
                            "ranked_candidates": [{"page": int(p), "score": round(float(s), 6)} for p, s in ranked[:max(self.degree * 3, 8)]],
                            "selected": selected,
                        })
                    except Exception:
                        with self._lock:
                            self._stats["prefetch_read_errors"] += 1
                    finally:
                        self._queue.task_done()
        except OSError:
            with self._lock:
                self._stats["prefetch_read_errors"] += 1

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait for queued jobs to drain. `Queue.join` has no timeout; use a helper."""
        finished = threading.Event()
        def waiter() -> None:
            self._queue.join()
            finished.set()
        threading.Thread(target=waiter, daemon=True).start()
        return finished.wait(timeout)

    def record_query_demand(self, query_id: str | None, pages: list[int], statuses: list[str]) -> None:
        """Append one xRead's demanded pages (with hit/miss status) to the reference."""
        if not self.recording or not query_id:
            return
        seq = self._demand_reference.setdefault(str(query_id), [])
        for page, status in zip(pages, statuses):
            seq.append([int(page), status == "miss"])

    def demand_reference(self) -> dict[str, list[list[Any]]]:
        return {qid: [list(entry) for entry in seq] for qid, seq in self._demand_reference.items()}

    def _write_prediction(self, row: dict[str, Any]) -> None:
        if not self._prediction_path:
            return
        with self._event_lock:
            with self._prediction_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
        completed = max(1, stats.get("completed_jobs", 1))
        stats["inference_ms_per_job"] = round(stats["inference_ms_total"] / completed, 4)
        stats["queue_wait_ms_mean"] = round(stats["queue_wait_ms_total"] / completed, 4)
        stats.update(self.cache.snapshot())
        pages_pf = max(1, stats.get("pages_prefetched", 1))
        stats["prefetch_read_ms_mean"] = round(stats["prefetch_read_ms_total"] / pages_pf, 4)
        if self.storage_device is not None:
            stats["storage_device"] = self.storage_device.snapshot()
        stats["storage_model"] = self.storage_model
        stats["drop_stale_jobs"] = self.drop_stale_jobs
        stats["queue_length"] = self._queue.qsize()
        stats["strategy"] = self.strategy
        stats["model_path"] = self.model_path
        stats["disabled_semantic_groups"] = list(getattr(self.model, "disabled_semantic_groups", [])) if self.model is not None else []
        stats["miss_delay_ms_injected"] = self.miss_delay_ms
        stats["predict_on"] = self.predict_on
        stats["predict_every"] = self.predict_every
        stats["max_score_candidates"] = self.max_score_candidates
        stats["prediction_log_path"] = str(self._prediction_path) if self._prediction_path else None
        stats["diagnostics_enabled"] = self.diagnostics_enabled
        stats["oracle_trace_queries"] = len(self._oracle_sequences)
        stats["oracle_miss_queries"] = len(self._oracle_miss)
        stats["prefetch_workers"] = self.prefetch_workers
        stats["worker_alive"] = any(worker.is_alive() for worker in self._workers)
        stats["worker_lifetime_ms"] = round(((self._worker_stopped_ns or time.perf_counter_ns()) - self._worker_started_ns) / 1e6, 3)
        return stats

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the worker after all accepted jobs have completed.

        Closing first stops new requests. The sentinel is then queued after any
        pending work, so a saturated queue cannot strand a daemon worker.
        """
        if self._closed:
            return not any(worker.is_alive() for worker in self._workers)
        self._closed = True
        for _ in self._workers:
            try:
                self._queue.put(None, timeout=timeout)
            except queue.Full:
                return False
        for worker in self._workers:
            worker.join(timeout=timeout)
        clean = not any(worker.is_alive() for worker in self._workers)
        with self._lock:
            self._stats["worker_shutdown_clean"] = clean
            if clean:
                self._worker_stopped_ns = time.perf_counter_ns()
        return clean
