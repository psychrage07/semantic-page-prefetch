import random

import pytest

from pagewise.sim import ALL_STRATEGIES, Costs, build_tables, load_reference, simulate


def _reference(queries: int = 20, length: int = 8, pages: int = 40) -> list[dict]:
    rng = random.Random(0)
    rows = []
    for index in range(queries):
        events = [rng.randint(1, pages) for _ in range(length)]
        rows.append({
            "query_id": f"Q{index:04d}",
            "pages": events,
            # In a synthetic trace we mark every page a miss so the device matters.
            "misses": [True] * length,
        })
    return rows


def _costs(**overrides) -> Costs:
    base = {"cpu_hit_us": 100.0, "cpu_miss_us": 100.0, "enqueue_us": 0.0,
            "decision_us": 0.0, "queue_handoff_us": 0.0,
            "prefetch_overhead_us": 0.0, "inference_us": 0.0, "query_overhead_us": 0.0}
    base.update(overrides)
    return Costs(**base)


def test_every_strategy_is_supported():
    for strategy in ("none", "null", "sequential", "markov1", "markov2", "markov_ctx",
                     "model_a", "model_b", "model_b_table", "hybrid",
                     "oracle_miss", "oracle_next"):
        assert strategy in ALL_STRATEGIES


def test_none_total_scales_with_reads_and_device_delay():
    reference = _reference()
    costs = _costs()
    reads = len(reference) * 8
    zero = simulate(reference, strategy="none", cache_pages=8, degree=4, workers=4,
                    delay_ms=0.0, storage_model="independent", costs=costs)
    # Equal hit/miss cpu cost: the model is exactly reads * cpu.
    assert zero.total_ms == pytest.approx(reads * 0.1, abs=1e-6)
    with_delay = simulate(reference, strategy="none", cache_pages=8, degree=4, workers=4,
                          delay_ms=1.0, storage_model="independent", costs=costs)
    # Each read that misses pays the device service time on top of the cpu cost.
    misses = with_delay.stats["misses"]
    assert with_delay.total_ms == pytest.approx(reads * 0.1 + misses * 1.0, abs=1e-6)
    assert with_delay.total_ms > zero.total_ms


def test_per_query_durations_reconstruct_the_total():
    reference = _reference()
    result = simulate(reference, strategy="none", cache_pages=8, degree=4, workers=4,
                      delay_ms=0.5, storage_model="shared", costs=_costs())
    assert len(result.per_query_ms) == len(reference)
    assert sum(result.per_query_ms) == pytest.approx(result.total_ms, abs=1e-6)


def test_simulation_is_deterministic():
    reference = _reference()
    kwargs = {"strategy": "oracle_miss", "cache_pages": 8, "degree": 4, "workers": 4,
              "delay_ms": 0.5, "storage_model": "shared", "costs": _costs()}
    first = simulate(reference, **kwargs)
    second = simulate(reference, **kwargs)
    assert first.total_ms == second.total_ms
    assert first.stats == second.stats


def test_null_pays_a_per_read_enqueue_cost():
    reference = _reference()
    kwargs = {"strategy": "null", "cache_pages": 8, "degree": 4, "workers": 4,
              "delay_ms": 0.5, "storage_model": "shared"}
    plain = simulate(reference, costs=_costs(), **kwargs)
    charged = simulate(reference, costs=_costs(enqueue_us=50.0), **kwargs)
    # 160 reads in the trace, so the enqueue cost must show up in the total.
    assert charged.total_ms > plain.total_ms
    assert charged.total_ms == pytest.approx(plain.total_ms + 50.0 * 160 / 1000.0, abs=1e-6)


def test_oracle_miss_beats_none_under_independent_but_not_shared():
    reference = _reference(queries=30)
    costs = _costs(enqueue_us=30.0, decision_us=60.0)
    independent = simulate(reference, strategy="oracle_miss", cache_pages=4, degree=4,
                           workers=4, delay_ms=2.0, storage_model="independent", costs=costs)
    baseline = simulate(reference, strategy="none", cache_pages=4, degree=4, workers=4,
                        delay_ms=2.0, storage_model="independent", costs=costs)
    assert independent.total_ms < baseline.total_ms
    assert independent.stats["useful_prefetches"] > 0

    shared_oracle = simulate(reference, strategy="oracle_miss", cache_pages=4, degree=4,
                             workers=4, delay_ms=2.0, storage_model="shared", costs=costs)
    shared_baseline = simulate(reference, strategy="none", cache_pages=4, degree=4, workers=4,
                               delay_ms=2.0, storage_model="shared", costs=costs)
    # A serialised device spends its bandwidth on speculation that also competes with
    # demand; this is the expected direction, not a simulator bug.
    assert shared_oracle.total_ms >= shared_baseline.total_ms


def test_known_strategy_never_selects_pages():
    reference = _reference()
    result = simulate(reference, strategy="none", cache_pages=8, degree=4, workers=4,
                      delay_ms=1.0, storage_model="shared", costs=_costs())
    assert result.stats["pages_prefetched"] == 0
    assert result.stats["useful_prefetches"] == 0


def test_reference_loader_round_trips(tmp_path):
    payload = {"Q1": [[1, True], [2, False], [3, True]]}
    path = tmp_path / "demand_reference.json"
    path.write_text("_" + "", encoding="utf-8") if False else path.write_text(
        "{\"Q1\": [[1, true], [2, false], [3, true]]}", encoding="utf-8")
    rows = load_reference(path)
    assert rows[0]["query_id"] == "Q1"
    assert rows[0]["pages"] == [1, 2, 3]
    assert rows[0]["misses"] == [True, False, True]
    assert build_tables(rows)["max_page"] == 3
