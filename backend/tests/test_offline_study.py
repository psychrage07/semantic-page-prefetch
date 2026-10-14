import json
import random
from pathlib import Path

from pagewise.offline import (
    POLICIES,
    _combine_outcomes,
    _ctx_key,
    build_context_tables,
    build_splits,
    evaluate_split,
    load_cases,
    outcome,
    paired_bootstrap_ci,
    run_offline_study,
    train_on_split,
)


def _synthetic(executions: int = 60, pages: int = 40, templates: int = 4) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    rng = random.Random(0)
    sequences = []
    cases = {}
    template_ids = [f"tpl{index}" for index in range(templates)]
    for index in range(executions):
        template = template_ids[index % templates]
        query_id = f"Q{index:03d}"
        events = [rng.randint(1, pages) for _ in range(6)]
        sequences.append({
            "query_id": query_id, "template_id": template,
            "workload_family": "point_lookup", "plan_class": "search_index",
            "pages": events,
        })
        cases[query_id] = {"query_id": query_id, "template_id": template,
                           "params": [rng.randint(1, 1000), rng.randint(5, 50)]}
    page_map = {str(page): {"page_type": "leaf", "depth": page % 3, "object": f"t{page % 5}"}
                for page in range(1, pages + 1)}
    return sequences, cases, page_map


def test_parameter_split_holds_out_whole_buckets():
    sequences, cases, _ = _synthetic()
    train, val, test, info = build_splits(sequences, cases, kind="params", seed=7)
    assert train and val and test
    # Held-out parameter buckets must be disjoint from the training buckets.
    assert not set(info["test_buckets"]) & set(info["train_buckets"])
    # The split is by execution, so no execution can appear twice.
    ids = [s["query_id"] for s in train + val + test]
    assert len(ids) == len(set(ids))


def test_template_split_excludes_test_templates_from_training():
    sequences, cases, _ = _synthetic()
    train, val, test, info = build_splits(sequences, cases, kind="templates", seed=7)
    train_templates = {str(s["template_id"]) for s in train}
    test_templates = {str(s["template_id"]) for s in test}
    assert test_templates == set(info["test_templates"])
    assert not train_templates & test_templates


def test_paired_bootstrap_ci_detects_direction():
    positive = [1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5, 1.0, 2.0]
    negative = [-v for v in positive]
    lo, hi = paired_bootstrap_ci(positive, seed=3)
    assert lo > 0 and hi > 0
    lo, hi = paired_bootstrap_ci(negative, seed=3)
    assert lo < 0 and hi < 0


def test_outcome_mapping():
    assert outcome(1.0, 2.0) == "supported"
    assert outcome(-1.0, 0.0) == "inconclusive"
    assert outcome(-2.0, -0.5, greater=False) == "supported"
    assert outcome(-2.0, -0.5, greater=True) == "not supported"
    assert _combine_outcomes("supported", "supported") == "supported"
    assert _combine_outcomes("supported", "inconclusive") == "inconclusive"
    assert _combine_outcomes("inconclusive", "not supported") == "not supported"


def test_context_key_uses_page_depth_from_metadata():
    page_map = {"5": {"page_type": "leaf", "depth": 2, "object": "orders"}}
    context = {"template_id": "t1", "plan_class": "search_index"}
    assert _ctx_key(5, context, page_map) == (5, "t1", "search_index", 2)


def test_offline_study_writes_results_for_all_policies(tmp_path: Path):
    sequences, cases, page_map = _synthetic()
    sequences_path = tmp_path / "sequences.jsonl"
    with sequences_path.open("w", encoding="utf-8") as handle:
        for row in sequences:
            handle.write(json.dumps(row) + "\n")
    workload_path = tmp_path / "workload.jsonl"
    with workload_path.open("w", encoding="utf-8") as handle:
        for row in cases.values():
            handle.write(json.dumps(row) + "\n")
    page_map_path = tmp_path / "page_map.json"
    page_map_path.write_text(json.dumps(page_map), encoding="utf-8")

    report = run_offline_study(sequences_path, workload_path, page_map_path, tmp_path,
                               seeds=[7, 17], epochs=2, hidden=8, max_candidates=8, top_k=8)
    assert (tmp_path / "offline_accuracy.json").exists()
    assert (tmp_path / "offline_accuracy.csv").exists()
    for split in ("params", "templates"):
        assert set(report["splits"][split]["mean_top1_accuracy_pct"]) == set(POLICIES)
    assert report["hypotheses"]["H1"]["outcome"] in {"supported", "not supported", "inconclusive"}
    assert report["hypotheses"]["H2"]["outcome"] in {"supported", "not supported", "inconclusive"}


def test_context_tables_and_evaluation_cover_all_policies(tmp_path: Path):
    sequences, cases, page_map = _synthetic()
    train, val, test, _ = build_splits(sequences, cases, kind="params", seed=7)
    trained = train_on_split(train, val, page_map, 7, epochs=2, hidden=8, max_candidates=8)
    markov_ctx, model_b_table = build_context_tables(train, trained, page_map, top_k=5)
    assert markov_ctx, "expected the context witness table to observe training transitions"
    assert model_b_table, "expected Model B's table to observe training transitions"
    evaluation = evaluate_split(test, trained, page_map, markov_ctx, model_b_table, top_k=8)
    assert evaluation["transitions"] > 0
    for execution_hits in evaluation["per_exec"].values():
        assert set(execution_hits) == set(POLICIES)
