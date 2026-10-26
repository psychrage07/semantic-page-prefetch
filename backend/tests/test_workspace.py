import csv
import hashlib
import json
from pathlib import Path

import pytest

import workspace
from pagewise.database import generate_database
from pagewise.dataset import build_sequences
from pagewise.models import train_models
from pagewise.runner import trace_workload


def test_snapshot_publication_retries_windows_reader_conflict(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []
    def temporary_conflict(path, target):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("Reader is holding the old snapshot")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", temporary_conflict)
    workspace.write_json(tmp_path / "state.json", {"trace_events": 128})
    assert json.loads((tmp_path / "state.json").read_text()) == {"trace_events": 128}
    assert len(attempts) == 2


def test_special_sql_values_and_long_previews_are_json_safe():
    assert workspace.cell(b"abc") == {"blob_bytes": 3, "hex_preview": "616263"}
    assert workspace.cell(float("inf")) == {"sqlite_real": "inf"}
    assert workspace.preview_cell("x" * 1000) == {"text_chars": 1000, "text_preview": "x" * 500}
    json.dumps(workspace.cell(float("inf")), allow_nan=False)


@pytest.fixture(scope="module")
def generated_demo(tmp_path_factory):
    lab = tmp_path_factory.mktemp("interactive-lab")
    root = lab / "123456abcdef"
    root.mkdir()
    generate_database(root / "pagewise-demo.sqlite", customers=80, products=30, orders=300,
                      graph_nodes=100, edges_per_node=2, seed=6)
    trace_workload(root / "pagewise-demo.sqlite", root / "trace", query_count=35, seed=55, cache_kib=64)
    build_sequences(root / "trace/trace.jsonl", root / "trace/page_map.json", root / "trace/sequences.jsonl")
    train_models(root / "trace/sequences.jsonl", root / "trace/page_map.json", root / "models", epochs=1, max_candidates=8, hidden=16)
    return lab, root


@pytest.mark.parametrize("sql", ["DELETE FROM orders", "CREATE TEMP TABLE bad(x)", "ATTACH ':memory:' AS extra",
    "PRAGMA writable_schema=ON", "SELECT 1; SELECT 2", "SELECT load_extension('bad')"])
def test_editor_rejects_writes_attachments_and_multiple_statements(generated_demo, monkeypatch, sql):
    lab, root = generated_demo
    monkeypatch.setattr(workspace, "LAB", lab)
    with pytest.raises(ValueError):
        workspace.validate_request({"database": root.name, "sql": sql, "policies": ["none"]})


def test_editor_only_accepts_catalog_databases_and_matching_models(generated_demo, monkeypatch):
    lab, root = generated_demo
    monkeypatch.setattr(workspace, "LAB", lab)
    with pytest.raises(ValueError, match="catalog"):
        workspace.resolve_database("../../research.sqlite")
    with pytest.raises(ValueError, match="supported policies"):
        workspace.validate_request({"database": root.name, "sql": "SELECT 1", "policies": ["model_ranker"]})


def test_actual_sql_benchmark_preserves_database_and_writes_artifacts(generated_demo, tmp_path, monkeypatch):
    lab, root = generated_demo
    monkeypatch.setattr(workspace, "LAB", lab)
    db = root / "pagewise-demo.sqlite"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    request = workspace.validate_request({"database": root.name,
        "sql": "SELECT order_id, customer_id FROM orders ORDER BY order_id LIMIT 120",
        "policies": ["none", "markov2", "model_b"], "repeats": 3})
    out = tmp_path / "execution"
    out.mkdir()
    workspace.write_json(out / "request.json", request)
    workspace.run(out)
    result = json.loads((out / "workspace-result.json").read_text(encoding="utf-8"))
    assert result["database_unchanged"] and result["identical_row_order_and_values"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert {row["policy"] for row in result["results"]} == {"none", "markov2", "model_b"}
    accuracy = result["evaluation"]
    assert accuracy["available"]
    assert accuracy["reference_policy"] == "none"
    assert set(accuracy["policies"]) == {"model_b", "markov1", "markov2"}
    assert accuracy["scored_transitions"] > 0
    assert accuracy["scored_transitions"] <= accuracy["eligible_transitions"]
    for metric in accuracy["policies"].values():
        assert metric["queries"] == 3
        assert metric["transitions"] == accuracy["scored_transitions"]
        assert 0 <= metric["top1_micro_pct"] <= metric["top5_micro_pct"] <= 100
    assert (out / "prediction-accuracy.json").is_file()
    assert (out / "prediction-accuracy.csv").is_file()
    for row in result["results"]:
        assert len(row["measurements"]) == 3
        assert all(event["rows"] == 120 for event in row["measurements"])
        assert row["mean_ms"] > 0
        directory = out / row["policy"]
        trace = workspace.tail_jsonl(directory / "trace.jsonl")
        assert any(event.get("query_id") == "SQL0001" for event in trace)
        assert row["runtime"]["worker_shutdown_clean"]
        assert row["runtime"]["prefetch_read_errors"] == 0
        assert (directory / "query-results.csv").is_file()
        assert (directory / "query-timings.csv").is_file()
    with (out / "benchmark.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_result_cap_is_explicit_and_real_values_are_exported(generated_demo, tmp_path, monkeypatch):
    lab, root = generated_demo
    monkeypatch.setattr(workspace, "LAB", lab)
    request = workspace.validate_request({"database": root.name, "sql": "SELECT order_id FROM orders ORDER BY order_id",
                                          "policies": ["none"], "row_limit": 3})
    workspace.write_json(tmp_path / "request.json", request)
    workspace.run(tmp_path)
    result = json.loads((tmp_path / "workspace-result.json").read_text(encoding="utf-8"))["results"][0]
    assert result["preview"] == [[1], [2], [3]]
    assert result["measurements"][0]["truncated"] is True
    assert result["measurements"][0]["rows"] == 3


def test_accuracy_sampling_is_shared_reproducible_and_covers_query_span():
    from workspace_evaluation import positions
    assert positions(1) == []
    assert positions(5) == [1, 2, 3, 4]
    sampled = positions(2001)
    assert len(sampled) == len(set(sampled)) == 512
    assert sampled == sorted(sampled) == positions(2001)
    assert sampled[0] == 1 and sampled[-1] == 2000


def test_constant_query_does_not_invent_prediction_accuracy(generated_demo, tmp_path, monkeypatch):
    lab, root = generated_demo
    monkeypatch.setattr(workspace, "LAB", lab)
    request = workspace.validate_request({"database": root.name, "sql": "SELECT 42", "policies": ["none"]})
    workspace.write_json(tmp_path / "request.json", request)
    workspace.run(tmp_path)
    result = json.loads((tmp_path / "workspace-result.json").read_text(encoding="utf-8"))
    assert result["results"][0]["preview"] == [[42]]
    assert result["evaluation"]["available"] is False
    assert result["evaluation"]["policies"] == {}
