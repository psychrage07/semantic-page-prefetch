import json
from pathlib import Path

import numpy as np

from pagewise.dataset import build_sequences
from pagewise.evaluate import evaluate_predictors
from pagewise.models import CandidateIndex, feature_vector, normalize_semantic_ablation_groups, train_models
from pagewise.sweep import train_evaluate_sweep


def test_build_sequences_expands_page_ranges_and_skips_uncorrelated(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    page_map = tmp_path / "page_map.json"
    out = tmp_path / "sequences.jsonl"
    page_map.write_text(json.dumps({"2": {"page": 2, "object": "orders", "page_type": "leaf", "depth": 2}}))
    events = [
        {"seq": 1, "query_id": None, "file_type": "main_db", "result": "success", "first_page": 1, "last_page": 1},
        {"seq": 2, "query_id": "Q1", "template_id": "t", "workload_family": "indexed_lookup", "plan_class": "search_index", "file_type": "main_db", "result": "success", "first_page": 2, "last_page": 3},
        {"seq": 3, "query_id": "Q1", "template_id": "t", "workload_family": "indexed_lookup", "plan_class": "search_index", "file_type": "main_db", "result": "success", "first_page": 2, "last_page": 2},
        {"seq": 4, "query_id": "Q2", "file_type": "main_db", "result": "error", "first_page": 8, "last_page": 8},
    ]
    trace.write_text("\n".join(json.dumps(x) for x in events) + "\n")
    sequences = build_sequences(trace, page_map, out)
    assert len(sequences) == 1
    assert sequences[0]["query_id"] == "Q1"
    assert sequences[0]["pages"] == [2, 3, 2]
    assert out.exists()


def _write_sequences(path: Path, n: int = 60):
    sequences = []
    for i in range(n):
        start = 1 + (i % 5)
        # Repeated next-page patterns with a small parameter-dependent variation.
        pages = [start, start + 10, start + 20, start + 30, start + 20, start + 40, start + 50]
        if i % 3 == 0:
            pages[4] = start + 31
        sequences.append({
            "query_id": f"Q{i:04d}", "template_id": "point" if i % 2 else "join",
            "workload_family": "indexed_lookup" if i % 2 else "indexed_join",
            "plan_class": "search_index", "pages": pages,
        })
    path.write_text("\n".join(json.dumps(x) for x in sequences) + "\n")


def test_feature_dimensions_and_candidate_index():
    metadata = {"12": {"object": "idx_orders", "page_type": "leaf", "depth": 2}}
    context = {"template_id": "customer_orders", "workload_family": "indexed_lookup", "plan_class": "search_index"}
    base = feature_vector([1, 12], 22, context, metadata, 100, False)
    semantic = feature_vector([1, 12], 22, context, metadata, 100, True)
    assert base.shape == (19,)
    assert semantic.shape[0] > base.shape[0]
    assert np.isfinite(semantic).all()
    idx = CandidateIndex.build([{"pages": [1, 2, 3, 2, 3, 4]}], max_candidates=8)
    assert 2 in idx.successors[1]


def test_train_and_evaluate_tiny_fixture_sequences(tmp_path: Path):
    seq_path = tmp_path / "sequences.jsonl"
    page_map = tmp_path / "page_map.json"
    out = tmp_path / "models"
    results = tmp_path / "evaluation"
    _write_sequences(seq_path)
    metadata = {
        str(p): {"page": p, "object": "orders_idx" if p % 2 else "orders", "page_type": "leaf" if p % 3 else "interior", "depth": p % 4}
        for p in range(1, 100)
    }
    page_map.write_text(json.dumps(metadata))
    report = train_models(
        seq_path, page_map, out, seed=13, epochs=3, max_candidates=12, hidden=32,
        semantic_ablations={"no_object": ["object"]},
    )
    assert (out / "model_a.pt").exists()
    assert (out / "model_b.pt").exists()
    assert (out / "model_b_retrained_no_object.pt").exists()
    assert report["models"]["model_a"]["input_dim"] == 19
    assert report["models"]["model_b"]["input_dim"] > 19
    assert report["models"]["model_b_retrained_no_object"]["disabled_semantic_groups"] == ["object"]
    evaluated = evaluate_predictors(
        out / "test_sequences.jsonl", page_map, out / "candidate_index.json",
        out / "model_a.pt", out / "model_b.pt", results,
        semantic_ablations={"no_object": ["object"], "no_query": ["query_context"]},
        retrained_ablation_dir=out,
    )
    assert "model_a" in evaluated["policies"]
    assert "model_b" in evaluated["policies"]
    assert "model_b_ablate_no_object" in evaluated["policies"]
    assert "model_b_retrained_no_object" in evaluated["policies"]
    assert evaluated["semantic_ablations"]["no_query"] == ["plan", "position", "template", "workload"]
    assert evaluated["retrained_semantic_ablations"]["no_object"] == ["object"]
    assert (results / "predictor_metrics.csv").exists()


def test_semantic_features_include_online_query_position():
    metadata = {"12": {"object": "idx_orders", "page_type": "leaf", "depth": 2}}
    base_context = {"template_id": "customer_orders", "workload_family": "indexed_lookup", "plan_class": "search_index"}
    early = feature_vector([1, 12], 22, {**base_context, "position_in_query": 0.1}, metadata, 100, True)
    late = feature_vector([1, 12], 22, {**base_context, "position_in_query": 0.9}, metadata, 100, True)
    assert early.shape == late.shape
    assert early[-1] == 0.1
    assert late[-1] == 0.9
    assert not np.array_equal(early, late)


def test_semantic_feature_ablation_zeroes_named_groups():
    metadata = {"12": {"object": "idx_orders", "page_type": "leaf", "depth": 2}}
    context = {"template_id": "customer_orders", "workload_family": "indexed_lookup", "plan_class": "search_index", "position_in_query": 0.5}
    full = feature_vector([1, 12], 22, context, metadata, 100, True)
    no_object = feature_vector([1, 12], 22, context, metadata, 100, True, disabled_semantic_groups=["object"])
    no_query = feature_vector([1, 12], 22, context, metadata, 100, True, disabled_semantic_groups=["query_context"])
    assert normalize_semantic_ablation_groups(["query_context"]) == {"plan", "template", "workload", "position"}
    assert np.array_equal(full[:19], no_object[:19])
    assert np.count_nonzero(no_object[25:57]) == 0
    assert np.count_nonzero(no_query[57:]) == 0


def test_train_sweep_aggregates_retrained_ablation_deltas(tmp_path: Path):
    seq_path = tmp_path / "sequences.jsonl"
    page_map = tmp_path / "page_map.json"
    out = tmp_path / "sweep"
    _write_sequences(seq_path)
    page_map.write_text(json.dumps({
        str(p): {"page": p, "object": "orders_idx" if p % 2 else "orders", "page_type": "leaf", "depth": p % 4}
        for p in range(1, 100)
    }))

    report = train_evaluate_sweep(
        seq_path, page_map, out, seeds=[3], split_seed=5, epochs=2,
        max_candidates=8, hidden=16, semantic_ablations={"no_object": ["object"]},
    )

    assert "model_b_retrained_no_object" in report["aggregate"]
    assert "model_b_retrained_no_object" in report["delta_vs_model_b"]
    assert report["delta_vs_model_b"]["model_b_retrained_no_object"]["top1_accuracy_pct"]["runs"] == 1
    assert (out / "seed-3" / "model_b_retrained_no_object.pt").exists()
    assert (out / "training_sweep_report.json").exists()
