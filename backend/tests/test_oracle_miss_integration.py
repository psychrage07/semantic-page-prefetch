"""Harness sanity test for the M1 oracle fix.

Under a parallel (independent) device with a large lookahead, a perfect oracle
must clearly beat `none`. If it does not, the runtime pipeline is the defect.
"""
import json
from pathlib import Path

import pytest

apsw = pytest.importorskip("apsw")

from pagewise.database import generate_database
from pagewise.models import CandidateIndex
from pagewise.suite import run_suite


def test_oracle_miss_beats_none_under_independent(tmp_path: Path):
    db = tmp_path / "oracle.sqlite"
    generated = generate_database(db, customers=400, products=120, orders=3000,
                                  graph_nodes=700, edges_per_node=2, seed=5)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    index = CandidateIndex(successors={}, second_order={}, popular=[1, 2, 3],
                           max_page=generated["page_count"], max_candidates=3)
    (model_dir / "candidate_index.json").write_text(json.dumps(index.to_json()), encoding="utf-8")

    out = tmp_path / "suite"
    run_suite(
        db, model_dir, out, queries=40, seeds=[101],
        strategies=["none", "null", "oracle_miss"],
        cache_pages=8, degree=4, miss_delay_ms=1.0,
        storage_model="independent", drop_stale_jobs=True,
        diagnostics_enabled=False, prefetch_workers=4,
    )
    summary = json.loads((out / "suite_summary.json").read_text(encoding="utf-8"))
    by = {row["strategy"]: row for row in summary["strategies"]}
    paired = {row["strategy"]: row for row in summary["paired_query_comparisons"] if row["baseline"] == "none"}

    # (c) the oracle's prefetches should mostly be used.
    assert by["oracle_miss"]["mean_useful_prefetch_pct"] >= 50.0
    # (a) with a parallel device and lookahead, the oracle must beat none.
    assert paired["oracle_miss"]["mean_delta_ms"] < 0
