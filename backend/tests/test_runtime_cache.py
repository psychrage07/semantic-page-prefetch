import json
from pathlib import Path

from pagewise.runtime import BoundedPageCache, PredictiveRuntime


def test_lru_cache_and_prefetch_usefulness():
    cache = BoundedPageCache(2)
    cache.put(1, b"a", origin="demand")
    cache.put(2, b"b", origin="prefetch")
    assert cache.get(1) == b"a"
    # Touching page 1 makes page 2 the least recently used.
    cache.put(3, b"c", origin="demand")
    stats = cache.snapshot()
    assert stats["evictions"] == 1
    assert stats["wasted_prefetches"] == 1
    cache.put(4, b"d", origin="prefetch")
    assert cache.get(4) == b"d"
    stats = cache.snapshot()
    assert stats["useful_prefetches"] == 1
    assert stats["cache_hits"] == 2


def test_cache_capacity_is_enforced():
    cache = BoundedPageCache(3)
    for page in range(1, 10):
        cache.put(page, bytes([page]), origin="demand")
    assert cache.snapshot()["resident_pages"] == 3
    assert cache.snapshot()["evictions"] == 6


def test_predictive_runtime_reports_queue_accounting(tmp_path: Path):
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaabbbbcccc")
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({"1": {"page_type": "leaf", "depth": 0, "object": "t"}}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {},
        "second_order": {},
        "popular": [1, 2, 3],
        "max_page": 3,
        "max_candidates": 3,
    }), encoding="utf-8")

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=2, degree=1, queue_size=4,
    )
    try:
        runtime.request_prediction([1], {"query_id": "Q1", "position_in_query": 0.1})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["prediction_requests"] == 1
    assert stats["queued_prediction_jobs"] == 1
    assert stats["completed_jobs"] == 1
    assert stats["dropped_jobs"] == 0
    assert "coalesced_jobs" in stats
    assert "candidates_pruned_cached_or_pending" in stats


def test_predictive_runtime_closes_cleanly_after_a_full_queue(tmp_path: Path):
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 4)
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({"1": {"page_type": "leaf", "depth": 0, "object": "t"}}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {}, "second_order": {}, "popular": [1, 2, 3],
        "max_page": 4, "max_candidates": 3,
    }), encoding="utf-8")

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=2, degree=1, queue_size=1, miss_delay_ms=25,
    )
    runtime.request_prediction([1], {"query_id": "Q1"})
    runtime.request_prediction([2], {"query_id": "Q2"})
    assert runtime.close(timeout=5.0)
    stats = runtime.snapshot()
    assert stats["worker_shutdown_clean"] is True
    assert stats["worker_alive"] is False


def test_predictive_runtime_can_filter_and_stride_prediction_observations(tmp_path: Path):
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaabbbbcccc")
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({"1": {"page_type": "leaf", "depth": 0, "object": "t"}}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {},
        "second_order": {},
        "popular": [1, 2, 3],
        "max_page": 3,
        "max_candidates": 3,
    }), encoding="utf-8")

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=2, degree=1, queue_size=4,
        predict_on="miss", predict_every=2,
    )
    try:
        assert runtime.should_predict("hit") is False
        assert runtime.should_predict("miss") is False
        assert runtime.should_predict("miss") is True
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["prediction_observations"] == 3
    assert stats["skipped_prediction_observations"] == 2
    assert stats["predict_on"] == "miss"
    assert stats["predict_every"] == 2


def test_hybrid_uses_markov_for_known_second_order_context(tmp_path: Path):
    import torch

    from pagewise.models import MLPScorer

    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 4)
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({str(page): {"page_type": "leaf", "depth": 0, "object": "t"} for page in range(1, 5)}), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {"2": [3]}, "second_order": {"1:2": [3]}, "popular": [1, 2, 3],
        "max_page": 4, "max_candidates": 3,
    }), encoding="utf-8")
    checkpoint = tmp_path / "model_b.pt"
    model = MLPScorer(86, hidden=8)
    torch.save({"state_dict": model.state_dict(), "input_dim": 86, "hidden": 8, "semantic": True}, checkpoint)

    runtime = PredictiveRuntime(
        database, 4, page_map, candidate_index, checkpoint,
        strategy="hybrid", semantic=True, cache_pages=2, degree=1, queue_size=2,
    )
    try:
        runtime.request_prediction([1, 2], {"query_id": "Q1"})
        assert runtime.wait_idle(timeout=5.0)
        stats = runtime.snapshot()
    finally:
        runtime.close()

    assert stats["hybrid_markov_jobs"] == 1
    assert stats["hybrid_semantic_jobs"] == 0


def test_max_score_candidates_limits_neural_scoring(tmp_path: Path):
    """max_score_candidates pre-ranks candidates by Markov prior and only scores top N."""
    database = tmp_path / "pages.bin"
    database.write_bytes(b"aaaa" * 10)  # 10 pages
    page_map = tmp_path / "page_map.json"
    page_map.write_text(json.dumps({
        str(i): {"page_type": "leaf", "depth": 0, "object": f"t{i}"}
        for i in range(1, 11)
    }), encoding="utf-8")
    candidate_index = tmp_path / "candidate_index.json"
    candidate_index.write_text(json.dumps({
        "successors": {"1": [2, 3, 4, 5, 6]},
        "second_order": {},
        "popular": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_page": 10,
        "max_candidates": 8,
    }), encoding="utf-8")

    # Without max_score_candidates
    runtime_full = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=5, degree=1, queue_size=4,
        max_score_candidates=0,
    )
    try:
        stats_full = runtime_full.snapshot()
    finally:
        runtime_full.close()
    assert stats_full["max_score_candidates"] == 0

    # With max_score_candidates
    runtime_capped = PredictiveRuntime(
        database, 4, page_map, candidate_index,
        strategy="sequential", cache_pages=5, degree=1, queue_size=4,
        max_score_candidates=3,
    )
    try:
        stats_capped = runtime_capped.snapshot()
    finally:
        runtime_capped.close()
    assert stats_capped["max_score_candidates"] == 3
    assert "candidates_pre_ranked_out" in stats_capped
    assert "inference_ms_per_job" in stats_capped
