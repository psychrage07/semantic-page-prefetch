import sqlite3
from pathlib import Path

from pagewise.database import generate_database
from pagewise.metadata import build_page_map, read_page_size
from pagewise.workloads import TEMPLATES, classify_plan, explain_query_plan, generate_workload


def test_generate_database_and_dbstat(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    meta = generate_database(db, customers=40, products=20, orders=150, avg_items_per_order=2,
                             graph_nodes=50, edges_per_node=2, seed=3)
    assert db.exists()
    assert meta["page_size"] == 4096
    assert meta["table_rows"]["customers"] == 40
    assert meta["table_rows"]["orders"] == 150
    assert read_page_size(db) == 4096
    mapping = build_page_map(db)
    assert mapping
    assert all("page_type" in row and "object" in row for row in mapping.values())
    sparse = build_page_map(db, {1, 2})
    assert sparse
    assert set(sparse).issubset({1, 2})


def test_generated_workloads_execute_without_error(tmp_path: Path):
    db = tmp_path / "workloads.sqlite"
    generate_database(db, customers=80, products=30, orders=250, graph_nodes=100, edges_per_node=2, seed=5)
    cases = generate_workload(db, count=60, seed=44)
    assert len(cases) == 60
    assert set(case.template_id for case in cases).issubset(TEMPLATES)
    with sqlite3.connect(db) as con:
        for case in cases:
            rows = con.execute(case.sql, case.params).fetchmany(3)
            assert isinstance(rows, list)
            details = explain_query_plan(db, case)
            assert details
            assert classify_plan(details) in {"unknown", "search_index", "search_other", "scan", "join", "recursive"}


def test_generator_rejects_graph_without_possible_edge(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError, match="graph_nodes must be >= 2"):
        generate_database(tmp_path / "invalid.sqlite", customers=2, products=2, orders=2, graph_nodes=1)


def test_generator_can_expand_relevant_rows_for_large_database_branch(tmp_path: Path):
    db = tmp_path / "large-branch-fixture.sqlite"
    meta = generate_database(db, customers=20, products=10, orders=40,
                             graph_nodes=20, edges_per_node=1,
                             item_payload_bytes=900, seed=9)
    assert meta["item_payload_bytes"] == 900
    with sqlite3.connect(db) as con:
        lengths = con.execute("SELECT DISTINCT length(payload) FROM order_items").fetchall()
    assert lengths == [(900,)]
