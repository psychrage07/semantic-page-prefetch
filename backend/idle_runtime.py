"""Idle-only admission over Pagewise's real VFS cache; no frozen core edits."""
from __future__ import annotations

from collections import Counter, OrderedDict
import threading
import time

from pagewise.runtime import BoundedPageCache, PredictiveRuntime, SimulatedStorageDevice


class ProbeCache(BoundedPageCache):
    """Same demand LRU plus a measured shadow LRU and optional idle admission."""
    def __init__(self, capacity_pages, *, protect=False, speculative_fraction=0.25):
        super().__init__(capacity_pages)
        self.protect = protect
        self.speculative_limit = max(1, int(capacity_pages * speculative_fraction))
        self.shadow, self.frequency = OrderedDict(), OrderedDict()
        self.probe = Counter()

    def get(self, page, *, demand=True, query_id=None):
        with self._lock:
            if demand:
                actual_hit, shadow_hit = page in self._pages, page in self.shadow
                self.probe["shadow_hits" if shadow_hit else "shadow_misses"] += 1
                self.probe["pollution_misses"] += int(shadow_hit and not actual_hit)
                self.shadow[page] = True
                self.shadow.move_to_end(page)
                if len(self.shadow) > self.capacity:
                    self.shadow.popitem(last=False)
                self.frequency[page] = self.frequency.get(page, 0) + 1
                self.frequency.move_to_end(page)
                while len(self.frequency) > 4 * self.capacity:
                    self.frequency.popitem(last=False)
            return super().get(page, demand=demand, query_id=query_id)

    def _evict(self, page, reason):
        entry = self._pages.pop(page)
        self.stats["evictions"] += 1
        if entry[1] == "prefetch" and not entry[2]:
            self.stats["wasted_prefetches"] += 1
        self.probe[reason] += 1

    def put(self, page, data, *, origin, metadata=None):
        with self._lock:
            if page not in self._pages and origin == "prefetch":
                if self.protect:
                    unused = [p for p, entry in self._pages.items() if entry[1] == "prefetch" and not entry[2]]
                    if len(unused) >= self.speculative_limit:
                        self.probe["speculative_quota_rejections"] += 1
                        return False
                    if len(self._pages) >= self.capacity:
                        # Proposals are ranked best-first. Never churn an unused
                        # page inserted earlier in this idle window for a later
                        # lower-ranked proposal. Old unused idle pages expire.
                        victim = next((p for p, e in self._pages.items()
                                       if not (e[1] == "prefetch" and not e[2]) and self.frequency.get(p, 0) < 2), None)
                        if victim is None:
                            self.probe["hot_page_protection_rejections"] += 1
                            return False
                        if self._pages[victim][1] != "prefetch" or self._pages[victim][2]:
                            self.probe["prefetch_displaced_demand_pages"] += 1
                        self._evict(victim, "prefetch_admission_evictions")
                elif len(self._pages) >= self.capacity:
                    entry = next(iter(self._pages.values()))
                    self.probe["prefetch_displaced_demand_pages"] += int(entry[1] != "prefetch" or entry[2])
            return super().put(page, data, origin=origin, metadata=metadata)

    def can_admit_prefetch(self, page):
        with self._lock:
            if page in self._pages:
                return False
            unused = sum(e[1] == "prefetch" and not e[2] for e in self._pages.values())
            if self.protect and unused >= self.speculative_limit:
                self.probe["speculative_quota_rejections"] += 1
                return False
            if self.protect and len(self._pages) >= self.capacity and not any(
                    not (e[1] == "prefetch" and not e[2]) and self.frequency.get(p, 0) < 2 for p, e in self._pages.items()):
                self.probe["hot_page_protection_rejections"] += 1
                return False
            return True

    def expire_unused_idle(self, generation):
        with self._lock:
            expired = [p for p, e in self._pages.items() if e[1] == "prefetch" and not e[2]
                       and "idle_generation" in e[3] and e[3]["idle_generation"] < generation]
            for page in expired:
                self._evict(page, "unused_idle_pages_expired")

    def snapshot(self):
        with self._lock:
            return {**super().snapshot(), **dict(self.probe), "speculative_limit_pages": self.speculative_limit,
                    "pollution_definition": "Actual demand miss while a same-capacity demand-only shadow LRU hits, on the same observed stream."}


class IdleStorageDevice(SimulatedStorageDevice):
    """Speculation never queues behind demand; artificial service is cancellable."""
    def try_prefetch(self, cancel):
        if not self.enabled:
            return not cancel.is_set()
        if not self._lock.acquire(blocking=False):
            return False
        try:
            started = time.perf_counter_ns()
            interrupted = cancel.wait(self.delay_ms / 1000)
            self.stats["device_busy_ms_total"] += (time.perf_counter_ns() - started) / 1e6
            if not interrupted:
                self.stats["prefetch_ios"] += 1
            return not interrupted
        finally:
            self._lock.release()


class IdleRuntime(PredictiveRuntime):
    def __init__(self, *args, query_model, max_pages=16, min_confidence=0.25, idle_budget_ms=100, **kwargs):
        if max_pages < 1 or idle_budget_ms <= 0:
            raise ValueError("Idle bounds must be positive")
        self.query_model = query_model
        self.max_pages, self.min_confidence, self.idle_budget_ms = max_pages, min_confidence, idle_budget_ms
        self._condition = threading.Condition(threading.RLock())
        self._cancel_idle = threading.Event()
        self._cancel_idle.set()
        self._generation, self._job, self._foreground, self._reading, self._busy = 0, None, True, False, False
        self._stopping, self._completed_history = False, []
        self.idle_stats, self.idle_events, self.proposals = Counter(), [], []
        kwargs.update(strategy="none", model_path=None, prefetch_workers=1)
        super().__init__(*args, **kwargs)
        self.cache = ProbeCache(self.cache.capacity, protect=True)

    def should_predict(self, access_status):
        with self._lock:
            self._stats["prediction_observations"] += 1
            self._stats["skipped_prediction_observations"] += 1
        return False

    def set_active_query(self, query_id):
        if query_id is not None:
            with self._condition:
                if not self._foreground:
                    self._cancel_idle.set()
                    self._foreground = True
                    self._generation += 1
                    self._job = None
                    self.idle_stats["in_flight_reads_at_arrival"] += int(self._reading)
                    self.idle_events.append({"event": "arrival", "query_id": query_id,
                                             "time_ns": time.perf_counter_ns(), "read_in_flight": self._reading})
                    self._condition.notify_all()
        super().set_active_query(query_id)

    def query_completed(self, query_id, template, pages, *, final=False):
        state = self.query_model.state(template, pages)
        with self._condition:
            self._completed_history.append(state)
            self._completed_history = self._completed_history[-2:]
            if final:
                return
            self._generation += 1
            generation = self._generation
            self.cache.expire_unused_idle(generation)
            self._foreground = False
            self._cancel_idle = threading.Event()
            now = time.perf_counter_ns()
            self._job = (generation, query_id, tuple(self._completed_history), now, self._cancel_idle)
            self.idle_events.append({"event": "idle_open", "after_query": query_id, "time_ns": now, "generation": generation})
            self.idle_stats["idle_windows"] += 1
            self._condition.notify_all()

    def _valid(self, generation, cancel, opened):
        return (not self._stopping and not self._foreground and self._generation == generation
                and not cancel.is_set() and (time.perf_counter_ns() - opened) / 1e6 < self.idle_budget_ms)

    def _run_worker(self):
        try:
            with self.database_path.open("rb", buffering=0) as file:
                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._stopping or self._job is not None)
                        if self._stopping:
                            break
                        generation, query_id, history, opened, cancel = self._job
                        self._job, self._busy = None, True
                    try:
                        with self._condition:
                            if not self._valid(generation, cancel, opened):
                                self.idle_stats["cancelled_before_prediction"] += 1
                                continue
                        start = time.perf_counter_ns()
                        with self.cache._lock:
                            excluded = tuple(self.cache._pages)
                        with self._condition:
                            excluded = (*excluded, *self._pending)
                        prediction = self.query_model.predict(history, max_pages=self.max_pages,
                            min_confidence=self.min_confidence, exclude_pages=excluded)
                        done = time.perf_counter_ns()
                        with self._condition:
                            self.idle_stats["prediction_jobs"] += 1
                            self.idle_stats["query_prediction_ns"] += done - start
                            self.idle_stats["idle_queue_wait_ns"] += start - opened
                            self.proposals.append({"after_query": query_id, "generation": generation,
                                "history": list(history), "prediction_start_ns": start, "prediction_end_ns": done, **prediction})
                        for candidate in prediction["pages"]:
                            page = candidate["page"]
                            with self._condition:
                                if not self._valid(generation, cancel, opened):
                                    self.idle_stats["cancelled_or_budget_expired"] += 1
                                    break
                                if self.cache.contains(page) or page in self._pending:
                                    self.idle_stats["cached_or_pending_skips"] += 1
                                    continue
                                if not self.cache.can_admit_prefetch(page):
                                    self.idle_stats["admission_rejected_before_io"] += 1
                                    continue
                                if not 1 <= page <= self._database_pages:
                                    self.idle_stats["invalid_page_skips"] += 1
                                    continue
                                self._pending.add(page)
                            try:
                                device = self.storage_device
                                if device and not device.try_prefetch(cancel):
                                    self.idle_stats["device_busy_or_cancelled"] += 1
                                    continue
                                with self._condition:
                                    if not self._valid(generation, cancel, opened):
                                        self.idle_stats["cancelled_before_read"] += 1
                                        break
                                    read_start = time.perf_counter_ns()
                                    self._reading = True
                                    self.idle_events.append({"event": "read_admitted", "time_ns": read_start,
                                        "generation": generation, "page": page, "after_query": query_id})
                                file.seek((page - 1) * self.page_size)
                                data = file.read(self.page_size)
                                read_done = time.perf_counter_ns()
                                with self._condition:
                                    self._reading = False
                                    self.idle_stats["prefetch_reads"] += 1
                                    self.idle_stats["prefetch_read_ns"] += read_done - read_start
                                    if not self._valid(generation, cancel, opened):
                                        self.idle_stats["late_reads_discarded"] += 1
                                        continue
                                    if len(data) != self.page_size:
                                        raise RuntimeError("Short idle-prefetch read")
                                    inserted = self.cache.put(page, data, origin="prefetch", metadata={
                                        "query_id": query_id, "idle_generation": generation,
                                        "page_read_done_ns": read_done, "confidence": candidate["confidence"]})
                                    self.idle_stats["inserted" if inserted else "admission_rejected"] += 1
                                    if inserted:
                                        self.idle_events.append({"event": "cache_inserted", "page": page,
                                            "after_query": query_id, "generation": generation,
                                            "read_done_ns": read_done, "time_ns": time.perf_counter_ns()})
                            finally:
                                with self._condition:
                                    self._reading = False
                                    self._pending.discard(page)
                    except Exception as exc:
                        self.idle_stats["worker_errors"] += 1
                        self.idle_events.append({"event": "error", "detail": repr(exc)})
                    finally:
                        with self._condition:
                            self._busy = False
                            self._condition.notify_all()
        finally:
            self._worker_stopped_ns = time.perf_counter_ns()

    def wait_idle(self, timeout=5):
        with self._condition:
            return self._condition.wait_for(lambda: not self._busy and self._job is None, timeout=timeout)

    def close(self, timeout=5):
        with self._condition:
            self._closed, self._stopping, self._job = True, True, None
            self._cancel_idle.set()
            self._condition.notify_all()
        for worker in self._workers:
            worker.join(timeout)
        clean = not any(worker.is_alive() for worker in self._workers)
        with self._lock:
            self._stats["worker_shutdown_clean"] = clean
        return clean

    def snapshot(self):
        value = super().snapshot()
        with self._condition:
            return {**value, "strategy": f"idle_markov{self.query_model.order}_{self.query_model.signal}",
                    "idle": dict(self.idle_stats), "idle_budget_ms": self.idle_budget_ms,
                    "worker_errors": self.idle_stats["worker_errors"]}
