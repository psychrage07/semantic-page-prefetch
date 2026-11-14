import json
import sqlite3
import threading

import pytest

from idle_markov import QueryMarkov
from idle_runtime import IdleRuntime, IdleStorageDevice, ProbeCache
from pagewise.metadata import build_page_map, read_page_size
from pagewise.models import CandidateIndex


def record(template, pages, session="train"):
    return {"template_id": template, "pages": pages, "session_id": session}


def test_query_markov_uses_completed_history_and_respects_session_boundaries(tmp_path):
    model = QueryMarkov(2).fit([record("A", [1]), record("B", [2, 2]), record("C", [3]),
                               record("A", [9], "other")])
    assert model.predict(["A", "B"])["order_used"] == 2
    assert model.predict(["A", "B"])["pages"] == [{"page": 3, "confidence": 1.0}]
    assert model.predict(["C"])["order_used"] == 0  # Never learn across sessions.
    before = json.dumps(model.counts, sort_keys=True)
    assert model.predict(["A"])["pages"] == [{"page": 2, "confidence": 1.0}]
    assert json.dumps(model.counts, sort_keys=True) == before
    path = tmp_path / "model.json"
    model.save(path)
    assert QueryMarkov.load(path).predict(["A", "B"]) == model.predict(["A", "B"])


def test_footprint_state_is_a_set_and_order_one_does_not_use_older_queries():
    model = QueryMarkov(1, "footprint").fit([record("A", [4, 3, 3]), record("B", [7])])
    assert model.state("different template", [3, 4]) == model.state("A", [4, 3, 3])
    state = model.state("A", [3, 4])
    assert model.predict(["irrelevant", state]) == model.predict([state])


def test_speculation_protects_hot_pages_and_tracks_pollution():
    cache = ProbeCache(2, protect=True)
    for page in (1, 2):
        cache.get(page)
        cache.put(page, b"data", origin="demand")
        cache.get(page)
    assert not cache.can_admit_prefetch(3)
    assert not cache.put(3, b"data", origin="prefetch")
    assert cache.contains(1) and cache.contains(2)
    unprotected = ProbeCache(2)
    for page in (1, 2):
        unprotected.get(page)
        unprotected.put(page, b"data", origin="demand")
    unprotected.put(3, b"data", origin="prefetch")
    assert unprotected.get(1) is None
    assert unprotected.snapshot()["pollution_misses"] == 1
    assert unprotected.snapshot()["prefetch_displaced_demand_pages"] == 1


def test_unused_prefetch_expiry_preserves_accounting_and_used_pages():
    cache = ProbeCache(8, protect=True)
    for page in (3, 4):
        assert cache.put(page, b"data", origin="prefetch", metadata={"idle_generation": 1})
    cache.get(3, query_id="next")
    cache.expire_unused_idle(2)
    stats = cache.snapshot()
    assert cache.contains(3) and not cache.contains(4)
    assert stats["useful_prefetches"] + stats["total_wasted_prefetches"] == stats["pages_prefetched"] == 2


def test_idle_proposals_do_not_replace_each_other_when_demand_pages_are_hot():
    cache = ProbeCache(4, protect=True, speculative_fraction=.5)
    for page in (1, 2, 3):
        cache.get(page); cache.put(page, b"data", origin="demand"); cache.get(page)
    assert cache.put(4, b"best prediction", origin="prefetch", metadata={"idle_generation": 1})
    assert not cache.can_admit_prefetch(5)
    assert not cache.put(5, b"lower prediction", origin="prefetch", metadata={"idle_generation": 1})
    assert cache.contains(4) and cache.snapshot()["pages_prefetched"] == 1


@pytest.fixture
def idle_files(tmp_path):
    database = tmp_path / "test.sqlite"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE sample(x BLOB)")
    conn.executemany("INSERT INTO sample VALUES (?)", [(b"x" * 3000,)] * 24)
    conn.commit(); conn.close()
    metadata = build_page_map(database)
    (tmp_path / "page_map.json").write_text(json.dumps(metadata), encoding="utf-8")
    index = CandidateIndex(successors={}, second_order={}, popular=[2, 3],
                           max_page=database.stat().st_size // read_page_size(database), max_candidates=8)
    (tmp_path / "candidate_index.json").write_text(json.dumps(index.to_json()), encoding="utf-8")
    return database, tmp_path


def make_runtime(files, model, device=None):
    db, directory = files
    return IdleRuntime(db, read_page_size(db), directory / "page_map.json", directory / "candidate_index.json",
                       query_model=model, cache_pages=16, storage_device=device or IdleStorageDevice(),
                       record_demand_reference=True)


def test_idle_read_arrives_before_next_demand_and_no_read_is_admitted_during_query(idle_files):
    model = QueryMarkov().fit([record("A", [2]), record("B", [3])])
    runtime = make_runtime(idle_files, model)
    try:
        runtime.set_active_query("Q1")
        runtime.query_completed("Q1", "A", [2])
        assert runtime.wait_idle(2)
        runtime.set_active_query("Q2")
        assert runtime.cache.get(3, query_id="Q2") is not None
        assert runtime.cache.snapshot()["useful_prefetches_cross_query"] == 1
        events = runtime.idle_events
        arrival = next(e["time_ns"] for e in events if e["event"] == "arrival")
        assert all(e["time_ns"] < arrival for e in events if e["event"] == "read_admitted")
        assert not runtime.should_predict("miss")
    finally:
        assert runtime.close()
        assert runtime.snapshot()["worker_shutdown_clean"]


def test_query_arrival_does_not_wait_for_prediction_and_cancels_its_result(idle_files):
    entered, release = threading.Event(), threading.Event()
    class SlowModel(QueryMarkov):
        def predict(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return {"pages": [{"page": 3, "confidence": 1}], "order_used": 1, "support": 1}
    runtime = make_runtime(idle_files, SlowModel())
    try:
        runtime.set_active_query("Q1")
        runtime.query_completed("Q1", "A", [2])
        assert entered.wait(2)
        arrival = threading.Thread(target=runtime.set_active_query, args=("Q2",))
        arrival.start(); arrival.join(.5)
        assert not arrival.is_alive()
        release.set()
        assert runtime.wait_idle(2)
        assert not runtime.cache.contains(3)
        assert not any(e["event"] == "read_admitted" for e in runtime.idle_events)
    finally:
        release.set()
        assert runtime.close()


def test_new_demand_cancels_simulated_prefetch_service_before_file_read(idle_files):
    entered = threading.Event()
    class Device(IdleStorageDevice):
        def try_prefetch(self, cancel):
            entered.set()
            return super().try_prefetch(cancel)
    device = Device(delay_ms=2000, enabled=True)
    model = QueryMarkov().fit([record("A", [2]), record("B", [3])])
    runtime = make_runtime(idle_files, model, device)
    try:
        runtime.set_active_query("Q1")
        runtime.query_completed("Q1", "A", [2])
        assert entered.wait(2)
        runtime.set_active_query("Q2")
        assert runtime.wait_idle(1)
        assert runtime.idle_stats["prefetch_reads"] == 0
    finally:
        assert runtime.close()


def test_real_sql_delay_condition_routes_demand_through_shared_device(idle_files):
    from idle_experiment import execute
    from pagewise.workloads import QueryCase
    database, models = idle_files
    cases = [QueryCase(f"Q{i}", "scan", "scan", "SELECT length(x) FROM sample LIMIT 10", ()) for i in range(2)]
    report, rows, records = execute(database, models, models / "delayed-query", cases, delay_ms=.2, policy="P0")
    assert report["runtime"]["storage_device"]["enabled"]
    assert report["runtime"]["storage_device"]["demand_ios"] > 0
    assert report["runtime"]["storage_device"]["device_busy_ms_total"] > 0
    assert report["demand_read_bytes"] > 0
    assert all(row["rows"] == 10 for row in rows)


def test_real_sql_records_each_requested_gap_without_passing_future_arrivals_to_predictor(idle_files):
    from idle_experiment import execute
    from pagewise.workloads import QueryCase
    database, models = idle_files
    cases = [QueryCase(f"V{i}", "scan", "scan", "SELECT length(x) FROM sample LIMIT 1", ()) for i in range(3)]
    model = QueryMarkov().fit([record("scan", [2]), record("scan", [3])])
    report, rows, _ = execute(database, models, models / "varying-gaps", cases, policy="P3",
                              query_model=model, gap_schedule=[0, 5])
    assert [row["configured_gap_ms"] for row in rows[1:]] == [0, 5]
    assert report["gap_schedule_ms"] == [0, 5]
    assert rows[2]["gap_observed_ms"] >= 5
    assert report["runtime"]["worker_shutdown_clean"]
    with pytest.raises(ValueError, match="one nonnegative gap"):
        execute(database, models, models / "invalid-gaps", cases, gap_schedule=[-1])
