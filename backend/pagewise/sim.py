"""Virtual-time discrete-event simulator for the Pagewise prefetch runtime (M3).

The simulator replays a recorded demand-reference (per query, the ordered demand page
stream plus which pages missed under a ``none`` run) on a single logical timeline with
no real sleeping and no threads, so it is exact and repeatable.

It does *not* re-derive prediction quality. Except for the two oracles and the
deterministic ``sequential`` strategy, every strategy's page selection is taken from a
transition table built on a deterministic prefix of the reference; the *timing* model
(queue, workers, device contention, cache) is what the simulator exists for. Real
runtime numbers remain authoritative for accuracy-sensitive claims.

Model of one demand read
------------------------
``duration = (device service time if it misses else 0) + cpu cost``

``cpu_hit_us`` / ``cpu_miss_us`` are calibrated from real ``none`` runs; the *next*
read starts as soon as the previous one returns, which reproduces the real back-to-back
demand stream.

Model of prefetching
--------------------
One job per demand read is enqueued at that read's completion time. Jobs are dequeued
FIFO by ``min(prefetch_workers)`` worker slots; each job pays a decision cost, optional
inference cost, and for every page it actually prefetches: device access (serialised by
the shared device when applicable) plus a cache-put overhead. Prefetches that finish
after the demand for that page are naturally counted as wasted, because the demand has
already inserted the page.
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
import statistics
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

INLINE_STRATEGIES = ("sequential", "markov1", "markov2", "markov_ctx", "model_b_table",
                     "oracle_next", "oracle_miss")
ALL_STRATEGIES = ("none", "null", "sequential", "markov1", "markov2", "markov_ctx",
                  "model_a", "model_b", "model_b_table", "hybrid",
                  "oracle_miss", "oracle_next")
NEURAL_STRATEGIES = ("model_a", "model_b", "model_b_table", "hybrid")


# --------------------------------------------------------------------------- reference

def load_reference(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reference = []
    for query_id, seq in payload.items():
        pages = [int(entry[0]) for entry in seq]
        misses = [bool(entry[1]) for entry in seq]
        reference.append({"query_id": str(query_id), "pages": pages, "misses": misses})
    reference.sort(key=lambda row: row["query_id"])
    return reference


def reference_stats(reference: list[dict[str, Any]]) -> dict[str, int]:
    reads = sum(len(row["pages"]) for row in reference)
    misses = sum(1 for row in reference for flag in row["misses"] if flag)
    return {"queries": len(reference), "reads": reads, "would_be_misses": misses}


# --------------------------------------------------------------------------- tables

def build_tables(reference: list[dict[str, Any]], *, train_fraction: float = 0.7,
                 max_candidates: int = 32) -> dict[str, Any]:
    """Markov successor table / second-order table / per-key context table.

    Built on a deterministic prefix of the reference (mirroring how the real
    candidate index comes from a training split), so evaluation pages are not
    automatically memorised.
    """
    successors: dict[int, list[int]] = {}
    second_order: dict[str, list[int]] = {}
    context: dict[tuple, list[int]] = {}
    cut = max(1, int(len(reference) * train_fraction))
    train = reference[:cut]
    succ_counter: defaultdict[int, Counter] = defaultdict(Counter)
    sec_counter: defaultdict[str, Counter] = defaultdict(Counter)
    ctx_counter: defaultdict[tuple, Counter] = defaultdict(Counter)
    max_page = 1
    for row in train:
        pages = row["pages"]
        misses = row["misses"]
        for index, page in enumerate(pages):
            max_page = max(max_page, page)
            if index == 0:
                continue
            source = pages[index - 1]
            succ_counter[source][page] += 1
            if index >= 2:
                sec_counter[f"{pages[index - 2]}:{source}"][page] += 1
            if not misses[index]:
                # Only a would-be miss is worth a speculative fetch; this is the same
                # signal the oracle_miss reference carries.
                key = (source, "", "", 0)
                ctx_counter[key][page] += 1
    successors = {page: [p for p, _ in counter.most_common(max_candidates)] for page, counter in succ_counter.items()}
    second_order = {key: [p for p, _ in counter.most_common(max_candidates)] for key, counter in sec_counter.items()}
    context = {key: [p for p, _ in counter.most_common(max_candidates)] for key, counter in ctx_counter.items()}
    return {"successors": successors, "second_order": second_order, "context": context,
            "max_page": max_page, "train_queries": cut}


def _select_pages(strategy: str, position: int, row: dict[str, Any], degree: int,
                  tables: dict[str, Any]) -> list[int]:
    pages = row["pages"]
    misses = row["misses"]
    current = pages[position]
    if strategy == "sequential":
        # Matches the runtime: `sequential` issues exactly one speculative page
        # (current + 1) regardless of the configured degree.
        return [current + 1] if current + 1 <= tables["max_page"] else []
    if strategy == "oracle_next":
        return pages[position + 1:position + 1 + degree]
    if strategy == "oracle_miss":
        picked: list[int] = []
        for index in range(position + 1, len(pages)):
            if misses[index]:
                picked.append(pages[index])
                if len(picked) >= degree:
                    break
        return picked
    successors = tables["successors"].get(current, [])
    second = tables["second_order"].get(f"{pages[position - 1]}:{current}", []) if position >= 1 else []
    base = second or successors
    if strategy == "markov1":
        return successors[:degree]
    if strategy == "markov2":
        return base[:degree]
    if strategy == "markov_ctx":
        return tables["context"].get((current, "", "", 0), successors[:degree])[:degree]
    if strategy in NEURAL_STRATEGIES:
        # The simulator only models timing: neural strategies pick pages from the
        # cached semantic table and are charged the measured inference cost.
        return tables["context"].get((current, "", "", 0), successors[:degree])[:degree]
    return []


# --------------------------------------------------------------------------- cache

class _Cache:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.entries: OrderedDict[int, list] = OrderedDict()
        self.stats = {"demand_reads": 0, "hits": 0, "misses": 0, "pages_prefetched": 0,
                      "useful_prefetches": 0, "wasted_evicted": 0, "superseded": 0}

    def contains(self, page: int) -> bool:
        return page in self.entries

    def demand(self, page: int) -> bool:
        """Consume `page` for a demand read; returns True on a hit."""
        self.stats["demand_reads"] += 1
        entry = self.entries.get(page)
        if entry is not None:
            self.stats["hits"] += 1
            self.entries.move_to_end(page)
            entry[1] = True
            if entry[0] == "prefetch" and not entry[2]:
                self.stats["useful_prefetches"] += 1
                entry[2] = True
            return True
        self.stats["misses"] += 1
        self._evict_for_insert()
        self.entries[page] = ["demand", True, True]
        return False

    def put_prefetch(self, page: int) -> bool:
        if page in self.entries:
            self.stats["superseded"] += 1
            return False
        self._evict_for_insert()
        self.entries[page] = ["prefetch", False, False]
        self.stats["pages_prefetched"] += 1
        return True

    def _evict_for_insert(self) -> None:
        while len(self.entries) >= self.capacity:
            victim, entry = self.entries.popitem(last=False)
            if entry[0] == "prefetch" and not entry[2]:
                self.stats["wasted_evicted"] += 1


# --------------------------------------------------------------------------- device

class _Device:
    """Single serialized storage channel with per-side achieved holds.

    The real SimulatedStorageDevice holds its lock during the whole ``time.sleep``,
    so contention serializes on the *achieved* sleep duration (not the configured
    label its busy counter reports). The demand side and the prefetch side are
    calibrated separately: the demand hold comes from the anchored no-prefetch run,
    the prefetch hold from the run's own page-read telemetry (sleep + fs read + put-
    time contention).
    """

    def __init__(self, delay_ms: float, shared: bool, delay_prefetch_ms: float | None = None):
        self.delay_ms = max(0.0, float(delay_ms))
        self.delay_prefetch = (max(0.0, float(delay_prefetch_ms))
                               if delay_prefetch_ms is not None else self.delay_ms)
        self.shared = bool(shared)
        self.free_at = 0.0
        self.stats = {"demand_ios": 0, "prefetch_ios": 0, "busy_ms": 0.0}

    def access(self, source: str, pages: int, ready: float) -> tuple[float, float]:
        # The simulation timeline is in milliseconds, so the device service time is
        # expressed in the same unit.
        hold = self.delay_ms if source == "demand" else self.delay_prefetch
        delay = hold * max(1, pages)
        start = ready
        if self.shared:
            start = max(ready, self.free_at)
            self.free_at = start + delay
        end = start + delay
        self.stats["busy_ms"] += delay
        self.stats["demand_ios" if source == "demand" else "prefetch_ios"] += 1
        return start, end


# --------------------------------------------------------------------------- simulator

@dataclass
class Costs:
    # Per-query fixed cost (lock bookkeeping, PRAGMA shrink_memory, workload setup),
    # charged once per query to the demand thread.
    query_overhead_us: float = 0.0
    cpu_hit_us: float = 0.0
    cpu_miss_us: float = 0.0
    # Charged to the *demand* thread per enqueued job. GIL contention between the
    # demand thread and the prefetch workers is not a queueing delay, so it is
    # calibrated directly from real runs (null total - none total) / reads.
    enqueue_us: float = 0.0
    decision_us: float = 0.0
    queue_handoff_us: float = 0.0
    prefetch_overhead_us: float = 0.0
    inference_us: float = 0.0


@dataclass
class SimResult:
    strategy: str
    storage_model: str
    total_ms: float = 0.0
    per_query_ms: list[float] = field(default_factory=list)
    reads: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


def simulate(reference: list[dict[str, Any]], *, strategy: str, cache_pages: int,
             degree: int = 2, workers: int = 1, delay_ms: float = 0.0,
             delay_prefetch_ms: float | None = None,
             storage_model: str = "shared", costs: Costs, tables: dict[str, Any] | None = None,
             queue_size: int = 64, drop_stale_jobs: bool = True) -> SimResult:
    """Run the discrete-event co-simulation and return the resulting metrics.

    Event-driven co-simulation of the demand thread and the prefetch workers on one
    chronological event queue, mirroring the real runtime (tracing.py/runtime.py):

    * a demand read evaluates the cache *at its arrival instant*; prefetch puts that
      complete later (even during the miss's own device hold) cannot cover it;
    * one prefetch job per read is enqueued at the read's completion; the demand
      thread pays enqueue_us first, exactly the calibrated null-vs-none gap;
    * workers are FIFO slots; each job selects its pages at dequeue time (as the
      runtime does), skipping pages already cached or in-flight (_pending);
    * the storage device is one FIFO channel, held for the *achieved* per-side hold
      (the runtime holds its lock during time.sleep; the demand hold is the anchored
      no-prefetch run's per-miss wall, the prefetch hold the run's own page-read
      telemetry);
    * with the independent model, accesses never queue.
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    tables = tables or build_tables(reference)
    cache = _Cache(cache_pages)
    device = _Device(delay_ms, shared=(storage_model == "shared"),
                     delay_prefetch_ms=delay_prefetch_ms)

    cpu_hit_ms = costs.cpu_hit_us / 1000.0
    cpu_miss_ms = costs.cpu_miss_us / 1000.0
    decision_ms = costs.decision_us / 1000.0
    pf_overhead_ms = costs.prefetch_overhead_us / 1000.0
    inference_ms = costs.inference_us / 1000.0
    enqueue_ms = costs.enqueue_us / 1000.0
    query_overhead_ms = costs.query_overhead_us / 1000.0

    n_workers = max(1, int(workers))
    worker_free = [0.0] * n_workers          # completion time of each slot's last job
    jobs: list[list[float]] = []             # FIFO of [entry_time, query_index, position]
    queued_keys: set[tuple[int, int]] = set()
    pending: set[int] = set()
    in_flight: dict[int, tuple] = {}         # slot -> (job, candidates, next_index)
    slot_busy_until: dict[int, float] = {}   # slot -> scheduled end of its current stage
    lead_times: list[float] = []
    queue_waits: list[float] = []
    demand_waits: list[float] = []
    prefetch_waits: list[float] = []
    per_query: list[float] = []
    prefetch_done: dict[int, float] = {}     # page -> wall time its prefetch put happened
    active_query = [0]

    seq_counter = itertools.count()
    events: list[tuple] = []
    current_time = [0.0]

    def push(time_ms: float, kind: str, payload: Any) -> None:
        heapq.heappush(events, (max(0.0, time_ms), next(seq_counter), kind, payload))

    def device_request(request_ms: float, source: str, payload: Any) -> None:
        hold = device.delay_ms if source == "demand" else device.delay_prefetch
        if device.shared:
            start = max(request_ms, device.free_at)
            device.free_at = start + hold
        else:
            start = request_ms
        (demand_waits if source == "demand" else prefetch_waits).append(max(0.0, start - request_ms))
        device.stats["demand_ios" if source == "demand" else "prefetch_ios"] += 1
        device.stats["busy_ms"] += hold
        push(start + hold, "dev_done", (source, payload))

    def read_arrival(arrival_ms: float, q_index: int, position: int) -> None:
        t = arrival_ms + (query_overhead_ms if position == 0 else 0.0)
        if position == 0:
            active_query[0] = q_index
        page = reference[q_index]["pages"][position]
        hit = cache.demand(page)
        if hit:
            prior = prefetch_done.get(page)
            if prior is not None:
                lead_times.append(max(0.0, t - prior))
            read_end(t + cpu_hit_ms, q_index, position)
        else:
            device_request(t, "demand", (q_index, position, t))

    def read_end(end_ms: float, q_index: int, position: int) -> None:
        # The demand timeline continues with the (optionally paid) enqueue step, so
        # the sweep after the last read of a query closes the query's total there.
        next_arrival = end_ms
        if strategy != "none":
            next_arrival += enqueue_ms
            if len(jobs) >= queue_size:
                cache.stats["dropped_jobs"] = cache.stats.get("dropped_jobs", 0) + 1
            else:
                jobs.append([next_arrival, q_index, position])
                queued_keys.add((q_index, position))
                slot = worker_free.index(min(worker_free))
                free_at = slot_busy_until.get(slot, worker_free[slot]) if slot in in_flight else worker_free[slot]
                start = max(next_arrival, free_at)
                push(start, "dequeue", slot)
        if position + 1 >= len(reference[q_index]["pages"]):
            per_query.append(next_arrival)
        # Advance to the next demand read: one page later, rolling into the next
        # query (with its fixed per-query overhead charged at its first arrival).
        position += 1
        if position >= len(reference[q_index]["pages"]):
            q_index += 1
            position = 0
            # the per-query overhead is charged once at the new query's arrival
        if q_index < len(reference):
            push(next_arrival, "next_read", (q_index, position))

    def on_dequeue(slot: int) -> None:
        now = current_time[0]
        if slot in in_flight:
            scheduled = slot_busy_until.get(slot, now)
            if scheduled > now:
                # One worker processes one job at a time: retry when the current
                # stage release fires.
                push(max(scheduled, now), "dequeue", slot)
                return
        if not jobs:
            return
        entry, q_index, position = jobs.pop(0)
        queued_keys.discard((q_index, position))
        queue_waits.append(max(0.0, now - entry))
        if drop_stale_jobs and q_index != active_query[0]:
            cache.stats["stale_jobs_dropped"] = cache.stats.get("stale_jobs_dropped", 0) + 1
            release(slot, now)
            return
        if strategy not in NEURAL_STRATEGIES:
            inference_ms_job = 0.0
        else:
            inference_ms_job = inference_ms
        candidates = _select_pages(strategy, position, reference[q_index], degree, tables)
        in_flight[slot] = ([entry, q_index, position], candidates, 0)
        worker_stage(now + decision_ms + inference_ms_job, slot)

    def worker_stage(work_ms: float, slot: int) -> None:
        if slot not in in_flight:
            return
        slot_busy_until[slot] = work_ms
        job, candidates, index = in_flight[slot]
        i = index
        while i < len(candidates):
            page = candidates[i]
            if page in cache.entries or page in pending:
                i += 1
                continue
            in_flight[slot] = (job, candidates, i)
            pending.add(page)
            device_request(work_ms, "prefetch", (slot, page))
            return
        release(slot, work_ms)

    def release(slot: int, time_ms: float) -> None:
        in_flight.pop(slot, None)
        worker_free[slot] = time_ms
        slot_busy_until[slot] = time_ms
        if jobs:
            push(max(time_ms, jobs[0][0]), "dequeue", slot)

    def on_prefetch_done(done_ms: float, payload: tuple) -> None:
        slot, page = payload
        pending.discard(page)
        if cache.put_prefetch(page):
            prefetch_done[page] = done_ms
        else:
            cache.stats["superseded"] = cache.stats.get("superseded", 0) + 1
        worker_stage(done_ms + pf_overhead_ms, slot)

    def on_demand_done(done_ms: float, payload: tuple) -> None:
        q_index, position, arrival_ms = payload
        page = reference[q_index]["pages"][position]
        prior = prefetch_done.get(page)
        if prior is not None:
            lead_times.append(max(0.0, arrival_ms - prior))
        read_end(done_ms + cpu_miss_ms, q_index, position)

    push(0.0, "next_read", (0, 0))

    while events:
        time_ms, _, kind, payload = heapq.heappop(events)
        current_time[0] = time_ms
        if kind == "next_read":
            read_arrival(time_ms, *payload)
        elif kind == "dev_done":
            if payload[0] == "demand":
                on_demand_done(time_ms, payload[1])
            else:
                on_prefetch_done(time_ms, payload[1])
        elif kind == "dequeue":
            on_dequeue(payload)
        else:
            raise RuntimeError(f"unknown event kind {kind!r}")

    total = per_query[-1] if per_query else 0.0
    durations: list[float] = []
    previous = 0.0
    for value in per_query:
        durations.append(value - previous)
        previous = value

    stats = {
        **cache.stats,
        "device_demand_ios": device.stats["demand_ios"],
        "device_prefetch_ios": device.stats["prefetch_ios"],
        "device_busy_ms": round(device.stats["busy_ms"], 4),
        "strategy": strategy,
        "cache_pages": cache_pages,
        "degree": degree,
        "workers": n_workers,
        "delay_ms": delay_ms,
        "delay_prefetch_ms": round(device.delay_prefetch, 4),
        "storage_model": storage_model,
        "pages_prefetched": cache.stats["pages_prefetched"],
        "useful_prefetches": cache.stats["useful_prefetches"],
        "useful_prefetch_pct": round(100.0 * cache.stats["useful_prefetches"] / cache.stats["pages_prefetched"], 3) if cache.stats["pages_prefetched"] else 0.0,
        "wasted_evicted": cache.stats["wasted_evicted"],
        "superseded": cache.stats["superseded"],
        "stale_jobs_dropped": cache.stats.get("stale_jobs_dropped", 0),
        "dropped_jobs": cache.stats.get("dropped_jobs", 0),
        "queue_length_final": len(jobs) + len(in_flight),
        "queue_wait_ms_mean": round(statistics.mean(queue_waits), 4) if queue_waits else 0.0,
        "queue_wait_ms_p95": _percentile(queue_waits, 95),
        "device_demand_wait_ms_mean": round(statistics.mean(demand_waits), 4) if demand_waits else 0.0,
        "device_demand_wait_ms_total": round(sum(demand_waits), 4),
        "device_prefetch_wait_ms_mean": round(statistics.mean(prefetch_waits), 4) if prefetch_waits else 0.0,
        "device_prefetch_wait_ms_total": round(sum(prefetch_waits), 4),
        "lead_time_ms_mean": round(statistics.mean(lead_times), 4) if lead_times else None,
        "lead_time_ms_p95": _percentile(lead_times, 95),
        "cache_hit_pct": round(100.0 * cache.stats["hits"] / max(1, cache.stats["demand_reads"]), 3),
    }
    return SimResult(strategy=strategy, storage_model=storage_model, total_ms=round(total, 4),
                     per_query_ms=[round(value, 6) for value in durations],
                     reads=reference_stats(reference)["reads"], stats=stats)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return round(ordered[index], 4)


def latency_summary(results: list[SimResult]) -> dict[str, Any]:
    rows = []
    for result in results:
        stats = dict(result.stats)
        stats.update({
            "strategy": result.strategy,
            "storage_model": result.storage_model,
            "total_ms": result.total_ms,
            "p50_ms": _percentile(result.per_query_ms, 50),
            "p95_ms": _percentile(result.per_query_ms, 95),
        })
        rows.append(stats)
    rows.sort(key=lambda row: row["total_ms"])
    return {"schema_version": 1, "results": rows}
