from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex, feature_vector
from pagewise.ranking import RankerConfig, build_groups, evaluate_ranker, train_ranker, group_features, NeuralPredictor
from pagewise.statistics import declare_family, holm_correction, holm_from_bootstrap
from pagewise.v21 import run_headroom_v21


def test_prediction_normalization_does_not_depend_on_competing_candidates():
    class Recorder(torch.nn.Module):
        def forward(self, x):
            self.inputs = x.numpy().copy()
            return x[:, 0]
    model = Recorder()
    counts = ML.build_training_transition_counts([{"pages": [1, 2, 1000]}])
    ML.candidate_logits(model, [1, 2], [3, 4], {}, {}, "b2", counts)
    observed = model.inputs[0]
    expected = ML.variant_feature_vector([1, 2], 3, {}, {}, 1000, "b2", counts)
    np.testing.assert_array_equal(observed, expected)
    ML.candidate_logits(model, [1, 2], [3, 900], {}, {}, "b2", counts)
    np.testing.assert_array_equal(model.inputs[0], observed)


def test_query_parameters_have_identical_provenance_and_missing_gate():
    seq = {"query_id": "q", "template_id": "t", "pages": [1, 2]}
    case = {"q": {"params": [10, 20]}}
    ctx = ML.prediction_context(seq, case)
    assert ctx["param_bucket"] == ML.param_bucket([10, 20])
    assert ctx["param_available"] is True
    assert ML.prediction_context({**seq, "params": [10, 20]}) == ctx
    missing = ML.prediction_context(seq)
    assert missing["param_bucket"] is None and missing["param_available"] is False
    assert ML.param_bucket({"a": 1, "b": 2}) == ML.param_bucket({"b": 2, "a": 1})


def test_second_order_candidate_survives_first_order_saturation_and_round_trip():
    index = CandidateIndex({2: [10, 11]}, {"1:2": [99]}, [10, 11], 100, 2,
                           strategy="markov2_first")
    assert index.candidates([1, 2])[0] == 99
    restored = CandidateIndex.from_json(index.to_json())
    assert restored.candidates([1, 2]) == index.candidates([1, 2])
    # Legacy behavior stays reproducible for existing checkpoints.
    index.strategy = "legacy"
    assert 99 not in index.candidates([1, 2])


def test_statistical_sign_symmetry_valid_pvalues_and_monotone_holm():
    sessions = {1: [1, 2, 1], 2: [0.5, 1, 2], 3: [-0.2, 1, 0.8]}
    negative = {k: [-x for x in xs] for k, xs in sessions.items()}
    a = declare_family([("comparison", sessions)], iterations=500)
    b = declare_family([("comparison", negative)], iterations=500)
    p1 = a["members"]["comparison"]["holm_p_unadjusted"]
    p2 = b["members"]["comparison"]["holm_p_unadjusted"]
    assert p1 == p2 and 0 <= p1 <= 1
    assert not b["members"]["comparison"]["holm_decision"]
    family = declare_family([("a", sessions), ("b", negative), ("c", {1: [0, 1, -1]})], iterations=500)
    rows = sorted(family["members"].values(), key=lambda r: r["holm_step_rank"])
    assert [r["holm_adjusted_p"] for r in rows] == sorted(r["holm_adjusted_p"] for r in rows)
    with pytest.raises(ValueError):
        holm_correction([1.2])
    with pytest.raises(ValueError):
        holm_from_bootstrap([{"ci95_low": 1, "ci95_high": 2}])


def test_absent_concurrent_headroom_cells_are_inconclusive(tmp_path):
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"q": [[p, True] for p in range(1, 41)]}))
    report = run_headroom_v21(ref, tmp_path / "out", caches=[4], delays=[0.1], channels=[1])
    assert report["decision"] == "INCONCLUSIVE"
    assert report["cells"][0]["oracle_invariants"]["chosen_action"] in {"no_prefetch", "prefetch_next_misses_k"}


def test_group_metrics_count_candidate_failures_and_neural_training_learns():
    torch.set_num_threads(1)
    train = [{"query_id": f"q{i}", "pages": [1, 2, 3, 1, 2, 3]} for i in range(8)]
    index = CandidateIndex.build(train, 4, strategy="markov2_first")
    counts = ML.build_training_transition_counts(train)
    config = RankerConfig("test", epochs=5, prior_weight=1)
    data = build_groups(train, index, {}, counts, config, {}, [1, 1, 1])
    model, fit = train_ranker(data, data, config)
    measured = evaluate_ranker(model, data, config)
    assert measured["top1_macro_pct"] == 100
    assert all(torch.isfinite(p).all() for p in model.parameters())
    test = [{"query_id": "unseen", "pages": [1, 999]}]
    missing = build_groups(test, index, {}, counts, config, {}, [1, 1, 1])
    measured = evaluate_ranker(model, missing, config)
    assert measured["candidate_recall_pct"] == 0
    assert measured["top1_micro_pct"] == measured["top5_micro_pct"] == 0


def test_runtime_does_not_silently_alias_simulator_oracle():
    from pagewise.runtime import PredictiveRuntime
    with pytest.raises(ValueError, match="simulator-only"):
        PredictiveRuntime("unused", 4096, "unused", "unused", strategy="oracle_optimal")


@pytest.mark.parametrize("variant", ["a", "b", "b2", "b3"])
def test_batched_features_match_training_contract_exactly(variant):
    seqs = [{"pages": [1, 4, 8, 4, 13]}]
    index = CandidateIndex.build(seqs, 8)
    counts = ML.build_training_transition_counts(seqs)
    config = RankerConfig("parity", variant=variant)
    metadata = {"4": {"page_type": "leaf", "depth": 1, "object": "orders"}}
    context = {"template_id": "lookup", "param_bucket": 2, "param_available": True, "position_in_query": 3 / 512}
    history, candidates = [1, 4], [2, 4, 8, 13]
    expected = np.stack([ML.variant_feature_vector(history, p, context, metadata, index.max_page, variant, counts) for p in candidates])
    actual = group_features(history, candidates, context, metadata, index, counts, config, [1, 1, 1])
    np.testing.assert_array_equal(actual, expected)


def test_layout_children_include_unobserved_pages_and_page_one_header(tmp_path):
    from pagewise.layout import parse_children, children
    # Synthetic valid interior header; pointer list includes children which
    # never occur in the training stream or sparse page metadata.
    page = bytearray(4096)
    page[0] = 0x05
    page[3:5] = (1).to_bytes(2, "big")
    page[8:12] = (4).to_bytes(4, "big")
    page[12:14] = (4000).to_bytes(2, "big")
    page[4000:4004] = (3).to_bytes(4, "big")
    assert parse_children(page, 2, 4) == (3, 4)
    raw = bytearray(4096 * 4)
    raw[:16] = b"SQLite format 3\x00"
    raw[16:18] = (4096).to_bytes(2, "big")
    raw[4096:8192] = page
    path = tmp_path / "layout.sqlite"
    path.write_bytes(raw)
    index = CandidateIndex.build([{"pages": [1, 2, 1]}], 4, strategy="layout", layout_database_path=str(path))
    assert 3 in index.candidates([2]) and 4 in index.candidates([2])
    assert children(str(path), 2) == (3, 4)
    assert parse_children(raw[:4096], 1, 4) == ()


def test_deployed_ranker_matches_offline_and_checkpoint_roundtrip(tmp_path):
    from dataclasses import asdict
    from pagewise.ranking import logits, _tensor_batch
    seqs = [{"query_id": "q", "pages": [1, 2, 3, 1, 2, 3]}]
    index = CandidateIndex.build(seqs, 4)
    counts = ML.build_training_transition_counts(seqs)
    config = RankerConfig("roundtrip", prior_weight=1, epochs=2)
    data = build_groups(seqs, index, {}, counts, config, {}, [1, 1, 1])
    model, _ = train_ranker(data, data, config)
    checkpoint = tmp_path / "ranker.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "input_dim": data["x"].shape[-1],
                "normalization_max_page": index.max_page, "counts": counts, "parameter_scales": [1, 1, 1]}, checkpoint)
    deployed = NeuralPredictor(checkpoint, index, {})
    ranked = deployed.rank([1], {"position_in_query": 1 / 512})
    x, prior, valid = _tensor_batch(data, slice(0, 1))
    with torch.inference_mode():
        probabilities = torch.softmax(logits(model, x, prior, valid, config), dim=1)[0].numpy()
    expected = {p: probabilities[i] for i, p in enumerate(data["pages"][0]) if p != -1}
    for page, score in ranked:
        assert score == pytest.approx(expected[page], abs=1e-7)


def test_compact_metadata_matches_whole_database_and_checks_snapshot(tmp_path):
    import apsw
    from pagewise.static_metadata import build_static_metadata, StaticMetadata
    database = tmp_path / "static.sqlite"
    con = apsw.Connection(str(database))
    con.execute("CREATE TABLE example(id INTEGER PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO example VALUES (?,?)", [(i, "x" * 100) for i in range(300)])
    expected = list(con.execute("SELECT pageno,name,pagetype,path FROM dbstat"))
    con.close()
    target = tmp_path / "metadata.npz"
    build_static_metadata(database, target)
    compact = StaticMetadata(target)
    for page, name, kind, path in expected:
        row = compact.get(str(page), {})
        assert row["object"] == name and row["page_type"] == kind
        assert row["depth"] == len([p for p in path.split("/") if p])
    assert compact.get(99999, {}) == {}
    # Mutating the database changes its header and invalidates cached metadata.
    con = apsw.Connection(str(database))
    con.execute("CREATE TABLE another(value TEXT)")
    con.close()
    with pytest.raises(ValueError, match="different snapshot"):
        build_static_metadata(database, target)


def test_full_evaluator_keeps_macro_micro_and_candidate_failure_denominators(tmp_path):
    from dataclasses import asdict
    from scripts.evaluate_neural import evaluate_seed
    from pagewise.models import MLPScorer
    training = [{"query_id": "train", "pages": [1, 2, 3, 1, 2, 3]}]
    index = CandidateIndex.build(training, 4)
    counts = ML.build_training_transition_counts(training)
    config = RankerConfig("evaluation", prior_weight=1)
    groups = build_groups(training, index, {}, counts, config, {}, [1, 1, 1])
    model = MLPScorer(groups["x"].shape[-1], hidden=config.hidden)
    torch.nn.init.zeros_(model.net[-1].weight)
    torch.nn.init.zeros_(model.net[-1].bias)
    root = tmp_path / "frozen" / "params"
    root.mkdir(parents=True)
    (root / "candidate_index.json").write_text(json.dumps(index.to_json()))
    (root / "page_map.json").write_text("{}")
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "input_dim": groups["x"].shape[-1],
                "normalization_max_page": index.max_page, "counts": counts, "parameter_scales": [1, 1, 1]}, root / "selected.pt")
    trace = tmp_path / "trace"
    trace.mkdir()
    sequences = [{"query_id": "a", "pages": [1, 2, 3]}, {"query_id": "b", "pages": [1, 999]}]
    (trace / "sequences.jsonl").write_text("\n".join(json.dumps(s) for s in sequences))
    (trace / "workload.jsonl").write_text("\n".join(json.dumps({"query_id": s["query_id"], "params": [1]}) for s in sequences))
    result = evaluate_seed(root.parent, "params", {"partition": {"test_buckets": list(range(6))},
                           "models": {"selected": {"checkpoint": "selected.pt"}}}, trace)
    selected = result["models"]["selected"]
    assert selected["top1_macro_pct"] == 50
    assert selected["top1_micro_pct"] == pytest.approx(200 / 3)
    assert selected["candidate_recall_pct"] == pytest.approx(200 / 3)
    assert result["selected_error_breakdown"][1]["candidate_missing"] == 1


def test_layout_override_and_cache_inspection_do_not_add_io_or_change_lru():
    from pagewise.runtime import BoundedPageCache
    cache = BoundedPageCache(2)
    cache.put(1, b"one", origin="demand")
    cache.put(2, b"two", origin="demand")
    assert cache.peek(1) == b"one"
    cache.put(3, b"three", origin="demand")
    assert cache.peek(1) is None and cache.contains(2)
    assert cache.stats["demand_requests"] == 0
    index = CandidateIndex({}, {}, [1], 10, 4, strategy="layout", layout_database_path="does-not-exist")
    assert index.candidates([2], layout_children=[3, 4]) == [3, 4, 1]


def test_real_vfs_can_execute_new_ranker_without_worker_errors(tmp_path):
    from dataclasses import asdict
    from pagewise.database import generate_database
    from pagewise.metadata import build_page_map
    from pagewise.models import MLPScorer, page_map_digest
    from pagewise.runner import trace_workload
    database = tmp_path / "runtime.sqlite"
    generate_database(database, customers=80, products=30, orders=300, graph_nodes=100, edges_per_node=2, seed=5)
    page_map = {str(k): v for k, v in build_page_map(database).items()}
    sequences = [{"pages": [1, 2, 3, 1, 2, 3]}]
    index = CandidateIndex.build(sequences, 16, strategy="layout", layout_database_path=str(database))
    index.page_map_sha256 = page_map_digest(page_map)
    config = RankerConfig("runtime", prior_weight=1)
    counts = ML.build_training_transition_counts(sequences)
    candidates = index.candidates([1])
    x = group_features([1], candidates, {}, page_map, index, counts, config, [1, 1, 1])
    model = MLPScorer(x.shape[-1], hidden=config.hidden)
    root = tmp_path / "model"
    root.mkdir()
    (root / "candidate_index.json").write_text(json.dumps(index.to_json()))
    (root / "page_map.json").write_text(json.dumps(page_map))
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "input_dim": x.shape[-1],
                "normalization_max_page": index.max_page, "counts": counts, "parameter_scales": [1, 1, 1]}, root / "model_ranker.pt")
    target = tmp_path / "run"
    trace_workload(database, target, query_count=12, seed=105, cache_kib=64,
                   prefetch_config={"strategy": "model_ranker", "candidate_index_path": str(root / "candidate_index.json"),
                                    "model_path": str(root / "model_ranker.pt"), "cache_pages": 8,
                                    "degree": 1, "queue_size": 8, "drop_stale_jobs": True})
    stats = json.loads((target / "runtime_stats.json").read_text())
    assert stats["worker_shutdown_clean"] is True
    assert stats["prefetch_read_errors"] == 0
    assert stats["completed_jobs"] > 0


def test_accelerated_static_features_are_bitwise_identical(tmp_path):
    import apsw
    from pagewise.static_metadata import build_static_metadata, StaticMetadata
    from pagewise.ranking import extra_features
    database = tmp_path / "parity.sqlite"
    con = apsw.Connection(str(database))
    con.execute("CREATE TABLE example(id INTEGER PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO example VALUES (?,?)", [(i, "x" * 100) for i in range(300)])
    con.close()
    build_static_metadata(database, tmp_path / "metadata.npz")
    metadata = StaticMetadata(tmp_path / "metadata.npz")
    sequences = [{"pages": [1, 2, 3, 2, 4, 3, 1, 5, 3]}]
    index = CandidateIndex.build(sequences, 32)
    counts = ML.build_training_transition_counts(sequences)
    candidates = list(range(1, 20))
    context = {"template_id": "lookup", "workload_family": "point_lookup", "plan_class": "search_index",
               "position_in_query": 5 / 512, "params": [120, 30], "param_bucket": 3, "param_available": True}
    scales = [10, 4, 1]
    for history in ([1], [2, 3], [999, 4]):
        for variant in ("a", "b", "b2", "b3"):
            for structural in (False, True):
                config = RankerConfig("parity", variant=variant, structural=structural)
                expected = []
                for p in candidates:
                    row = ML.variant_feature_vector(history, p, context, metadata, index.max_page, variant, counts)
                    if structural:
                        row = np.concatenate([row, np.asarray(extra_features(history, p, context, metadata, index.max_page, scales), np.float32)])
                    expected.append(row)
                actual = group_features(history, candidates, context, metadata, index, counts, config, scales)
                np.testing.assert_array_equal(actual, np.stack(expected))
