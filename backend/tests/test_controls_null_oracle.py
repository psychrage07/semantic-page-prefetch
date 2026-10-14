import json
from pathlib import Path

import pytest

from pagewise.runtime import PredictiveRuntime
from pagewise.suite import VALID_STRATEGIES
from pagewise.tracing import TraceRecorder


def _make_runtime(tmp_path: Path, strategy: str, **kwargs) -> PredictiveRuntime:
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 8)  # 8 pages of 4 bytes
    page_map = tmp_path / "page_map.json"
    page_map.write_text(
        json.dumps({str(page): {"page_type": "leaf", "depth": 0, "object": "t"} for page in range(1, 9)}),
        encoding="utf-8",
    )
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(
        json.dumps({"successors": {}, "second_order": {}, "popular": list(range(1, 9)),
                    "max_page": 8, "max_candidates": 8}),
        encoding="utf-8",
    )
    return PredictiveRuntime(
        database, 4, page_map, candidate_index, strategy=strategy,
        cache_pages=kwargs.pop("cache_pages", 8), degree=kwargs.pop("degree", 1),
        queue_size=kwargs.pop("queue_size", 8), **kwargs,
    )


def test_control_strategies_are_registered():
    assert "null" in VALID_STRATEGIES
    assert "oracle_next" in VALID_STRATEGIES
    assert "oracle_miss" in VALID_STRATEGIES
    # Every v1 strategy must remain in the roster.
    for name in ("none", "stock", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid"):
        assert name in VALID_STRATEGIES


def test_null_goes_through_the_full_path_but_never_prefetches(tmp_path: Path):
    runtime = _make_runtime(tmp_path, "null")
    try:
        for position in range(1, 6):
            runtime.request_prediction([position], {"query_id": "Q1", "query_page_position": position})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    # The full runtime path ran: hooks observed pages, jobs were queued and completed.
    assert stats["prediction_requests"] == 5
    assert stats["queued_prediction_jobs"] == 5
    assert stats["completed_jobs"] == 5
    # ...but no page was ever fetched speculatively.
    assert stats["pages_prefetched"] == 0
    assert stats["useful_prefetches"] == 0
    assert stats["prefetch_bytes"] == 0


def test_none_short_circuits_before_the_queue(tmp_path: Path):
    runtime = _make_runtime(tmp_path, "none")
    try:
        for position in range(1, 6):
            # should_predict is still invoked by the VFS for `none`, but no job is queued.
            runtime.should_predict("miss")
            runtime.request_prediction([position], {"query_id": "Q1", "query_page_position": position})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["prediction_requests"] == 0
    assert stats["queued_prediction_jobs"] == 0
    assert stats["pages_prefetched"] == 0


def test_oracle_prefetches_recorded_next_pages(tmp_path: Path):
    trace = tmp_path / "oracle_trace.jsonl"
    with trace.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "Q1", "first_page": 1, "last_page": 1}) + "\n")
        handle.write(json.dumps({"query_id": "Q1", "first_page": 2, "last_page": 2}) + "\n")

    runtime = _make_runtime(tmp_path, "oracle", oracle_trace_path=trace)
    try:
        runtime.request_prediction([1], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
        assert runtime.cache.contains(2)
    finally:
        runtime.close()

    assert stats["pages_prefetched"] == 1
    assert stats["oracle_alignment_misses"] == 0


def test_oracle_requires_a_trace(tmp_path: Path):
    with pytest.raises(ValueError):
        _make_runtime(tmp_path, "oracle")


def test_oracle_divergence_is_counted_and_does_not_crash(tmp_path: Path):
    trace = tmp_path / "oracle_trace.jsonl"
    with trace.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "Q1", "first_page": 1, "last_page": 1}) + "\n")
        handle.write(json.dumps({"query_id": "Q1", "first_page": 2, "last_page": 2}) + "\n")

    runtime = _make_runtime(tmp_path, "oracle", oracle_trace_path=trace)
    try:
        # Observed history disagrees with the recorded sequence at position 1.
        runtime.request_prediction([5], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["oracle_alignment_misses"] == 1
    assert stats["pages_prefetched"] == 0


def test_diagnostics_can_be_disabled(tmp_path: Path):
    runtime = _make_runtime(
        tmp_path, "null",
        prediction_log_path=tmp_path / "predictions.jsonl",
        diagnostics_enabled=False,
    )
    try:
        runtime.request_prediction([1], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["diagnostics_enabled"] is False
    assert stats["prediction_log_path"] is None
    assert not (tmp_path / "predictions.jsonl").exists()


def test_diagnostics_enabled_writes_prediction_log(tmp_path: Path):
    runtime = _make_runtime(
        tmp_path, "null",
        prediction_log_path=tmp_path / "predictions.jsonl",
        diagnostics_enabled=True,
    )
    try:
        runtime.request_prediction([1], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
    finally:
        runtime.close()

    log = tmp_path / "predictions.jsonl"
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["policy"] == "null"


def test_oracle_miss_prefetches_next_miss_pages(tmp_path: Path):
    ref = tmp_path / "demand_reference.json"
    ref.write_text(
        json.dumps({"Q1": [[1, False], [2, True], [3, True], [4, False], [5, True]]}),
        encoding="utf-8",
    )
    runtime = _make_runtime(tmp_path, "oracle_miss", oracle_miss_path=ref, degree=2)
    try:
        runtime.request_prediction([1], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
        assert runtime.cache.contains(2)
        assert runtime.cache.contains(3)
    finally:
        runtime.close()

    # The first two would-be-miss pages after position 1 are 2 and 3.
    assert stats["pages_prefetched"] == 2
    assert stats["oracle_alignment_misses"] == 0


def test_oracle_miss_requires_a_reference(tmp_path: Path):
    with pytest.raises(ValueError):
        _make_runtime(tmp_path, "oracle_miss")


def test_oracle_miss_alignment_mismatch_is_counted(tmp_path: Path):
    ref = tmp_path / "demand_reference.json"
    ref.write_text(json.dumps({"Q1": [[1, False], [2, True], [3, True]]}), encoding="utf-8")
    runtime = _make_runtime(tmp_path, "oracle_miss", oracle_miss_path=ref, degree=2)
    try:
        runtime.request_prediction([9], {"query_id": "Q1", "query_page_position": 1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["oracle_alignment_misses"] == 1
    assert stats["pages_prefetched"] == 0


def test_demand_reference_records_hit_miss_status(tmp_path: Path):
    runtime = _make_runtime(tmp_path, "none", record_demand_reference=True)
    try:
        runtime.record_query_demand("Q1", [1, 2, 3], ["miss", "hit", "miss"])
        ref = runtime.demand_reference()
    finally:
        runtime.close()
    assert ref == {"Q1": [[1, True], [2, False], [3, True]]}


def test_trace_recorder_can_be_disabled(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(output, tmp_path / "db.sqlite", 4096, enabled=False)
    recorder.record_read(filename=str(tmp_path / "db.sqlite"), offset=0, amount=4096, duration_ns=1, result="success")
    recorder.close()
    assert not output.exists()
