import json
from pathlib import Path

import pytest

apsw = pytest.importorskip("apsw")

from pagewise.database import generate_database
from pagewise.runner import trace_workload


def test_vfs_collects_query_context_and_valid_page_ranges(tmp_path: Path):
    db = tmp_path / "vfs.sqlite"
    generate_database(db, customers=80, products=30, orders=300, graph_nodes=100, edges_per_node=2, seed=5)
    out = tmp_path / "trace"
    result = trace_workload(db, out, query_count=30, seed=55, cache_kib=64, mixed=True)
    assert Path(result["trace_path"]).exists()
    events = [json.loads(line) for line in Path(result["trace_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert events
    correlated = [e for e in events if e.get("query_id")]
    assert correlated, "Expected some VFS read requests to be associated with a running query"
    for event in correlated:
        assert event["first_page"] >= 1
        assert event["last_page"] >= event["first_page"]
        assert event["page_size"] == 4096


def test_vfs_prefetch_runtime_can_schedule_only_after_misses(tmp_path: Path):
    from pagewise.models import CandidateIndex

    db = tmp_path / "vfs-prefetch.sqlite"
    result = generate_database(db, customers=80, products=30, orders=300, graph_nodes=100, edges_per_node=2, seed=6)
    out = tmp_path / "prefetch"
    candidate_index = tmp_path / "candidate_index.json"
    index = CandidateIndex(successors={}, second_order={}, popular=[1, 2, 3], max_page=result["page_count"], max_candidates=3)
    candidate_index.write_text(json.dumps(index.to_json()), encoding="utf-8")

    trace_workload(db, out, query_count=20, seed=56, cache_kib=64, mixed=True, prefetch_config={
        "strategy": "sequential",
        "candidate_index_path": str(candidate_index),
        "cache_pages": 8,
        "degree": 1,
        "queue_size": 8,
        "predict_on": "miss",
        "predict_every": 2,
    })
    stats = json.loads((out / "runtime_stats.json").read_text(encoding="utf-8"))
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert stats["predict_on"] == "miss"
    assert stats["predict_every"] == 2
    assert stats["prediction_observations"] > stats["prediction_requests"]
    assert stats["skipped_prediction_observations"] > 0
    assert stats["worker_shutdown_clean"] is True
    assert stats["worker_alive"] is False
    assert metadata["end_to_end_wall_ms"] > 0
