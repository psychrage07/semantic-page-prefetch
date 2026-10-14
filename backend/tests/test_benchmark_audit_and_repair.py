import json
import threading
import time
from pathlib import Path

from pagewise.runtime import BoundedPageCache, PredictiveRuntime, SimulatedStorageDevice


def test_consistent_prefetch_denominators():
    """Verify that useful and wasted prefetch rates share the consistent denominator (pages_prefetched)."""
    cache = BoundedPageCache(capacity_pages=3)
    # Prefetch 3 pages: 1, 2, 3
    cache.put(1, b"page1", origin="prefetch", metadata={"page_read_done_ns": time.perf_counter_ns()})
    cache.put(2, b"page2", origin="prefetch", metadata={"page_read_done_ns": time.perf_counter_ns()})
    cache.put(3, b"page3", origin="prefetch", metadata={"page_read_done_ns": time.perf_counter_ns()})

    # Demand hits page 1 -> useful
    assert cache.get(1, demand=True, query_id="Q1") == b"page1"

    # Demand misses on page 4, evicting page 2 (which was unused) -> wasted_evicted
    cache.put(4, b"page4", origin="demand")

    # Demand misses on page 5, evicting page 3 (which was unused) -> wasted_evicted
    cache.put(5, b"page5", origin="demand")

    # Prefetch page 6
    cache.put(6, b"page6", origin="prefetch", metadata={"page_read_done_ns": time.perf_counter_ns()})
    # Page 6 remains resident and unused at snapshot

    snap = cache.snapshot()
    # Total prefetched: 4 pages (1, 2, 3, 6)
    assert snap["pages_prefetched"] == 4
    # Useful: 1 (page 1)
    assert snap["useful_prefetches"] == 1
    # Wasted evicted: 2 (pages 2, 3)
    assert snap["wasted_prefetches"] == 2
    # Unused resident: 1 (page 6)
    assert snap["unused_prefetches_still_resident"] == 1
    # Total wasted = 2 + 1 = 3
    assert snap["total_wasted_prefetches"] == 3

    # Denominators are consistent: 1/4 = 25%, 3/4 = 75%
    assert snap["useful_prefetch_pct"] == 25.0
    assert snap["wasted_prefetch_evicted_pct"] == 50.0
    assert snap["unused_prefetch_resident_pct"] == 25.0
    assert snap["wasted_prefetch_total_pct"] == 75.0
    assert round(snap["useful_prefetch_pct"] + snap["wasted_prefetch_total_pct"], 1) == 100.0


def test_timestamped_diagnostics_and_demand_use_tracking(tmp_path: Path):
    """Verify end-to-end timestamped diagnostics connecting prefetch to demand use."""
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 8)
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({str(p): {"page_type": "leaf", "depth": 0, "object": "t"} for p in range(1, 9)}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {"1": [2]}, "second_order": {}, "popular": [1, 2],
        "max_page": 8, "max_candidates": 2,
    }), encoding="utf-8")
    prediction_log = tmp_path / "predictions.jsonl"

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="markov1", cache_pages=4, degree=1, queue_size=4,
        prediction_log_path=prediction_log,
    )
    try:
        runtime.set_active_query("Q1")
        runtime.request_prediction([1], {"query_id": "Q1", "template_id": "T1"})
        assert runtime.wait_idle(timeout=5.0)

        # Main query thread demand-hits page 2
        data = runtime.cache.get(2, demand=True, query_id="Q1")
        assert data == b"aaaa"

        stats = runtime.snapshot()
        assert stats["useful_prefetches"] == 1
        assert stats["useful_prefetches_same_query"] == 1
        assert stats["useful_prefetches_cross_query"] == 0
        assert stats["demand_use_lead_time_ms_mean"] is not None
        assert stats["demand_use_lead_time_ms_mean"] >= 0.0

        events = runtime.cache._demand_use_events
        assert len(events) == 1
        ev = events[0]
        assert ev["page"] == 2
        assert ev["prefetch_query_id"] == "Q1"
        assert ev["demand_query_id"] == "Q1"
        assert ev["same_query"] is True
        assert ev["lead_time_ms"] >= 0.0

        # Check prediction log has full timestamped fields
        assert prediction_log.exists()
        lines = [json.loads(l) for l in prediction_log.read_text(encoding="utf-8").strip().splitlines() if l]
        assert len(lines) == 1
        log_entry = lines[0]
        assert "request_ns" in log_entry
        assert "worker_start_ns" in log_entry
        assert "inference_start_ns" in log_entry
        assert "inference_done_ns" in log_entry
        assert "queue_wait_ms" in log_entry
        assert len(log_entry["selected"]) == 1
        assert log_entry["selected"][0]["status"] == "prefetched"
        assert "page_read_start_ns" in log_entry["selected"][0]
        assert "page_read_done_ns" in log_entry["selected"][0]
    finally:
        runtime.close()


def test_shared_storage_device_contention():
    """Verify that the shared storage device model serializes concurrent access and tracks wait time."""
    device = SimulatedStorageDevice(delay_ms=25.0, enabled=True)
    events = []

    def prefetch_io():
        events.append(("prefetch_start", time.perf_counter_ns()))
        wait = device.access("prefetch", pages=1)
        events.append(("prefetch_done", time.perf_counter_ns(), wait))

    def demand_io():
        # Small delay to ensure prefetch starts first
        time.sleep(0.005)
        events.append(("demand_start", time.perf_counter_ns()))
        wait = device.access("demand", pages=1)
        events.append(("demand_done", time.perf_counter_ns(), wait))

    t1 = threading.Thread(target=prefetch_io)
    t2 = threading.Thread(target=demand_io)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = device.snapshot()
    assert snap["enabled"] is True
    assert snap["demand_ios"] == 1
    assert snap["prefetch_ios"] == 1
    assert snap["device_busy_ms_total"] >= 45.0
    # Demand had to wait behind prefetch
    assert snap["demand_wait_ms_total"] > 10.0


def test_stale_job_detection_and_dropping(tmp_path: Path):
    """Verify that queued prediction jobs for past queries are identified and dropped when configured."""
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 8)
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({str(p): {"page_type": "leaf", "depth": 0, "object": "t"} for p in range(1, 9)}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {"1": [2]}, "second_order": {}, "popular": [1, 2],
        "max_page": 8, "max_candidates": 2,
    }), encoding="utf-8")

    # Runtime with drop_stale_jobs enabled
    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="markov1", cache_pages=4, degree=1, queue_size=4,
        drop_stale_jobs=True,
    )
    try:
        # Simulate query Q1 requesting prediction
        runtime.set_active_query("Q1")
        runtime.request_prediction([1], {"query_id": "Q1"})
        # Query Q1 immediately finishes, Q2 begins before worker processes job
        runtime.set_active_query("Q2")
        assert runtime.wait_idle(timeout=5.0)

        stats = runtime.snapshot()
        assert stats["stale_jobs_detected"] == 1
        assert stats["stale_jobs_dropped"] == 1
        # No prefetch was performed for the stale job
        assert stats["pages_prefetched"] == 0
    finally:
        runtime.close()


def test_in_flight_prefetch_miss_accounting(tmp_path: Path):
    """Verify tracking when demand misses while the requested page is in-flight in prefetch."""
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 4)
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({"1": {"page_type": "leaf", "depth": 0, "object": "t"}}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {}, "second_order": {}, "popular": [1], "max_page": 4, "max_candidates": 1,
    }), encoding="utf-8")

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=2, degree=1, queue_size=2,
    )
    try:
        # Manually mark page 3 as in-flight
        with runtime._lock:
            runtime._pending.add(3)

        assert runtime.is_page_pending(3) is True
        assert runtime.is_page_pending(4) is False

        # Demand misses on page 3
        hit = runtime.record_demand_miss_in_flight(3)
        assert hit is True

        # Demand misses on page 4
        miss = runtime.record_demand_miss_in_flight(4)
        assert miss is False

        stats = runtime.snapshot()
        assert stats["demand_misses_while_prefetch_in_flight"] == 1
    finally:
        with runtime._lock:
            runtime._pending.discard(3)
        runtime.close()
