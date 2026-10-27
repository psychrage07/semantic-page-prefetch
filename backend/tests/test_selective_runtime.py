import json
import threading
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex, MLPScorer, page_map_digest
from pagewise.ranking import RankerConfig, group_features
from pagewise.runtime import SimulatedStorageDevice
from selective_prefetch import SelectiveRuntime


def make_assets(tmp_path, real=False):
    db = tmp_path / "example.sqlite"
    if real:
        from pagewise.database import generate_database
        from pagewise.metadata import build_page_map
        generate_database(db, customers=80, products=30, orders=300, graph_nodes=100, edges_per_node=2, seed=5)
        page_map = {str(k): v for k, v in build_page_map(db).items()}
    else:
        db.write_bytes(b"x" * 4096 * 16)
        page_map = {str(i): {"page": i, "object": "test", "page_type": "leaf", "depth": 1} for i in range(1, 17)}
    index = CandidateIndex({2: [3, 4], 3: [4]}, {"1:2": [3, 4]}, [3, 4, 5], 16, 8, strategy="markov2_first")
    index.page_map_sha256 = page_map_digest(page_map)
    tmp_path.joinpath("page_map.json").write_text(json.dumps(page_map))
    tmp_path.joinpath("candidate_index.json").write_text(json.dumps(index.to_json()))
    counts = ML.build_training_transition_counts([{"query_id": "train", "pages": [1, 2, 3, 4, 1, 2, 3, 4]}])
    config = RankerConfig("test", prior_weight=1)
    x = group_features([1, 2], index.candidates([1, 2]), {}, page_map, index, counts, config, [1, 1, 1])
    model = MLPScorer(x.shape[-1], hidden=config.hidden)
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "input_dim": x.shape[-1],
                "normalization_max_page": index.max_page, "counts": counts,
                "parameter_scales": [1, 1, 1]}, tmp_path / "model.pt")
    return db, tmp_path / "page_map.json", tmp_path / "candidate_index.json", tmp_path / "model.pt"


def make_runtime(tmp_path, **kwargs):
    db, pages, index, model = make_assets(tmp_path)
    policy = kwargs.pop("policy", "context")
    policy_config = kwargs.pop("policy_config", {"max_jobs_per_query": 8, "neural_fallback_families": []})
    return SelectiveRuntime(db, 4096, pages, index, model, strategy="model_ranker", semantic=True,
                            cache_pages=4, queue_size=kwargs.pop("queue_size", 4),
                            policy=policy, policy_config=policy_config,
                            prediction_log_path=tmp_path / "predictions.jsonl", **kwargs)


def context(position=1):
    return {"query_id": "q1", "template_id": "lookup", "workload_family": "point_lookup",
            "plan_class": "search", "position_in_query": position / 512, "query_page_position": position}


def test_deadline_rejects_jobs_that_cannot_finish(tmp_path):
    runtime = make_runtime(tmp_path, policy="deadline", policy_config={"max_jobs_per_query": 8,
        "deadlines_ms": {"point_lookup": 1.0}, "runtime_cost_ms": {"markov2": {"score_ms": 5.0, "read_ms": 0.0}}})
    try:
        runtime.set_active_query("q1")
        runtime.request_prediction([1, 2], context())
        assert runtime.wait_idle(2)
        snap = runtime.snapshot()
        assert snap["policy_deadline_dropped"] == 1
        assert snap["queued_prediction_jobs"] == 0
    finally:
        assert runtime.close()


def test_completed_query_cancels_inflight_scoring_before_io(tmp_path):
    runtime = make_runtime(tmp_path)
    started, release = threading.Event(), threading.Event()
    def rank(*args):
        started.set()
        assert release.wait(3)
        return [(3, 1.0)]
    runtime._rank = rank
    try:
        runtime.set_active_query("q1")
        runtime.request_prediction([1, 2], context())
        assert started.wait(2)
        runtime.set_active_query(None)
        release.set()
        assert runtime.wait_idle(3)
        snap = runtime.snapshot()
        assert snap["policy_query_completed_dropped"] >= 1
        assert snap["pages_prefetched"] == 0
        assert snap["prefetch_read_errors"] == 0
    finally:
        release.set()
        assert runtime.close()


def test_queue_is_bounded_and_coalesces_inflight_duplicate_jobs(tmp_path):
    runtime = make_runtime(tmp_path, queue_size=1)
    started, release = threading.Event(), threading.Event()
    def rank(*args):
        started.set()
        assert release.wait(3)
        return []
    runtime._rank = rank
    try:
        runtime.set_active_query("q1")
        runtime.request_prediction([1, 2], context(1))
        assert started.wait(2)
        runtime.request_prediction([1, 2], context(1))
        runtime.request_prediction([1, 2], context(2))
        runtime.request_prediction([1, 2], context(3))
        snap = runtime.snapshot()
        assert snap["coalesced_jobs"] == 1
        assert snap["dropped_jobs"] == 1
        assert snap["queue_length"] <= 1
    finally:
        release.set()
        assert runtime.wait_idle(3)
        assert runtime.close()


def test_per_query_limit_is_enforced_before_queue_admission(tmp_path):
    runtime = make_runtime(tmp_path, policy_config={"max_jobs_per_query": 1, "neural_fallback_families": []})
    started, release = threading.Event(), threading.Event()
    def rank(*args):
        started.set()
        assert release.wait(3)
        return []
    runtime._rank = rank
    try:
        runtime.set_active_query("q1")
        runtime.request_prediction([1, 2], context(1))
        assert started.wait(2)
        runtime.request_prediction([1, 2], context(2))
        assert runtime.snapshot()["policy_query_limit_dropped"] == 1
    finally:
        release.set()
        assert runtime.wait_idle(3)
        assert runtime.close()


def test_cached_pending_and_duplicate_candidates_are_not_read(tmp_path):
    runtime = make_runtime(tmp_path, degree=1)
    runtime._rank = lambda *args: [(3, 1.0), (3, 0.9), (4, 0.8), (5, 0.7)]
    try:
        runtime.cache.put(3, b"x" * 4096, origin="demand")
        runtime._pending.add(4)
        runtime.set_active_query("q1")
        runtime.request_prediction([1, 2], context())
        assert runtime.wait_idle(3)
        snap = runtime.snapshot()
        assert snap["policy_cached_candidates"] == 1
        assert snap["policy_duplicate_candidates"] == 1
        assert snap["policy_pending_candidates"] == 1
        assert snap["pages_prefetched"] == 1
        assert runtime.cache.contains(5)
    finally:
        assert runtime.close()


def test_device_does_not_queue_speculation_behind_busy_demand(tmp_path):
    device = SimulatedStorageDevice(1.0, enabled=True)
    runtime = make_runtime(tmp_path, storage_device=device)
    try:
        device._lock.acquire()
        try:
            assert runtime._try_device() is False
        finally:
            device._lock.release()
        with runtime._lock:
            runtime._demand_waiters = 1
        assert runtime._try_device() is False
        assert device.stats["prefetch_ios"] == 0
    finally:
        assert runtime.close()


def test_real_vfs_runs_opt_in_context_policy(tmp_path, monkeypatch):
    from pagewise import runner
    db, pages, index, model = make_assets(tmp_path, real=True)
    monkeypatch.setattr(runner, "PredictiveRuntime", lambda *a, **kw: SelectiveRuntime(*a, policy="context",
        policy_config={"max_jobs_per_query": 8, "neural_fallback_families": []}, **kw))
    out = tmp_path / "run"
    runner.trace_workload(db, out, query_count=12, seed=105, cache_kib=64,
        prefetch_config={"strategy": "model_ranker", "candidate_index_path": str(index), "model_path": str(model),
                         "cache_pages": 8, "degree": 1, "queue_size": 8, "drop_stale_jobs": True})
    stats = json.loads((out / "runtime_stats.json").read_text())
    assert stats["worker_shutdown_clean"] and stats["prefetch_read_errors"] == 0
    assert stats["prediction_observations"] > 0
    assert stats["selective_policy"] == "context"
