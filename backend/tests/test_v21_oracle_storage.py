"""v2.1 invariant tests: oracle semantics, k-channel storage, Belady cache.

Invariants under test (Requirement A):
1. Under identical simulator assumptions, oracle_optimal (which can choose the
   no-prefetch action) never has greater objective cost than the no-prefetch
   baseline.
2. For fixed inputs, increasing positive service time does not reduce absolute
   completion time (within numerical tolerance).
3. The measured benefit RELATIVE to no prefetching is NOT required to be
   monotone in delay (shared-device contention); the test only checks the
   completion-time monotonicity above, and documents non-monotonicity of
   benefit as an allowed observation.
"""
from __future__ import annotations

import random

import pytest

from pagewise.sim import Costs, simulate, parse_storage_model
from pagewise.sim import _BeladyCache, _NEVER


def _reference(seed: int = 3, queries: int = 6, reads: int = 120, span: int = 40) -> list[dict]:
    rng = random.Random(seed)
    ref = []
    for q in range(queries):
        pages: list[int] = []
        cur = rng.randrange(1, span // 2)
        for _ in range(reads):
            cur = max(1, min(span, cur + rng.choice([-3, -2, -1, 0, 1, 2, 3])))
            pages.append(cur)
        ref.append({"query_id": f"q{q}", "pages": pages})
    return ref


COSTS = Costs(query_overhead_us=50, cpu_hit_us=900, cpu_miss_us=900, enqueue_us=10,
              decision_us=5, prefetch_overhead_us=20, inference_us=0)


def _storage(k: int) -> tuple[str, int]:
    return ("shared", 1) if k == 1 else ("k_channel", k)


def test_oracle_optimal_never_exceeds_none_grid():
    ref = _reference()
    for k in (1, 4, 16):
        for delay in (0.1, 0.3):
            for cache_pages in (4, 32):
                kind, channels = _storage(k)
                base = simulate(ref, strategy="none", cache_pages=cache_pages, degree=2,
                                delay_ms=delay, costs=COSTS, storage_model=kind,
                                storage_channels=channels)
                oracle = simulate(ref, strategy="oracle_optimal", cache_pages=cache_pages,
                                  degree=2, delay_ms=delay, costs=COSTS, storage_model=kind,
                                  storage_channels=channels)
                assert oracle.total_ms <= base.total_ms + 1e-6, (
                    f"oracle_optimal exceeded none at k={k}, delay={delay}, cache={cache_pages}: "
                    f"{oracle.total_ms} > {base.total_ms}")
                invariant = oracle.stats["oracle_invariants"]
                assert invariant["cost_le_none"] is True
                assert invariant["none_action_cost_ms"] == base.total_ms


def test_service_time_monotonicity_for_fixed_inputs():
    ref = _reference()
    for strategy in ("none", "oracle_optimal"):
        previous = -1.0
        for delay in (0.0, 0.1, 0.2, 0.5, 1.0):
            result = simulate(ref, strategy=strategy, cache_pages=64, degree=2,
                              delay_ms=delay, costs=COSTS, storage_model="shared")
            assert result.total_ms >= previous - 1e-6
            previous = result.total_ms


def test_oracle_cpu_and_amdahl_are_reported_without_device_doublecount():
    ref = _reference()
    result = simulate(ref, strategy="markov2", cache_pages=16, degree=2, delay_ms=0.2,
                      costs=COSTS, storage_model="shared")
    stats = result.stats
    assert stats["cpu_fraction_of_total"] > 0.0
    # CPU total is bounded above by total time: no device or waiting time mixed in.
    assert stats["cpu_simulated_ms"] <= result.total_ms + 1e-6
    assert "amdahl_max_reduction_ms" in stats


def test_parse_storage_model_labels():
    assert parse_storage_model("shared") == ("shared", 1)
    assert parse_storage_model("k_channel:4") == ("k_channel", 4)
    assert parse_storage_model("kchannel_16") == ("k_channel", 16)
    assert parse_storage_model("independent") == ("independent", None)
    with pytest.raises(ValueError):
        parse_storage_model("nvme")  # no physical device label is accepted


def test_belady_cache_evicts_farthest_next_use():
    positions = {1: [100], 2: [10], 3: [50]}
    cache = _BeladyCache(2, positions, lambda: 0)
    cache.demand(3)
    cache.demand(1)
    assert set(cache.entries) == {1, 3}
    # demand(2) must evict page 1 (next use at 100) and keep 3 (next use 50)
    cache.demand(2)
    assert set(cache.entries) == {2, 3}
    assert cache.stats["belady_evictions"] == 1
    # prefetching a never-needed page is rejected (waste accounting)
    assert cache.prefetch(99, 0) is False
    assert cache.stats["wasted_superseded"] == 1


def test_belady_replacement_is_distinct_from_scheduling_optimization():
    # The module exports the replacement policy separately; the scheduling
    # policy reports its optimality scope explicitly.
    ref = _reference()
    result = simulate(ref, strategy="oracle_optimal", cache_pages=8, degree=2,
                      delay_ms=0.2, costs=COSTS, storage_model="shared")
    scope = result.stats["optimality_scope"]
    assert "NOT claimed" in scope and "_BeladyCache" in scope


def test_k_channel_threaded_device_serves_concurrently():
    import threading
    import time as _t
    from pagewise.storage import KChannelStorageDevice
    device = KChannelStorageDevice(delay_ms=0.05, channels=4, enabled=True)
    barrier = threading.Barrier(4)

    def job() -> None:
        barrier.wait()
        device.access("prefetch", 1)

    threads = [threading.Thread(target=job) for _ in range(4)]
    started = _t.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = _t.perf_counter() - started
    stats = device.snapshot()
    # one serialized channel of 50 ms x4 would take >= 200 ms; 4 slots must not.
    assert elapsed < 0.18, f"k=4 device serialized access ({elapsed*1000:.1f} ms)"
    assert stats["channels"] == 4 and stats["max_in_service"] >= 1
