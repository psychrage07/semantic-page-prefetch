import csv
import json

import pytest

import summarize_selective as summary


def test_offline_accuracy_counts_skipped_and_missing_candidates_as_errors(tmp_path):
    path = tmp_path / "transitions.csv"
    rows = [
        {"seed": "1", "query_id": "q1", "family": "point_lookup", "exact_context": "1", "markov1_available": "1",
         "candidate_hit": "0", "markov1_candidate_hit": "1", "markov2_candidate_hit": "1", "neural_top1": "0", "neural_top5": "0",
         "markov1_top1": "0", "markov1_top5": "1", "markov2_top1": "1", "markov2_top5": "1"},
        {"seed": "1", "query_id": "q2", "family": "point_lookup", "exact_context": "0", "markov1_available": "0",
         "candidate_hit": "1", "markov1_candidate_hit": "0", "markov2_candidate_hit": "0", "neural_top1": "1", "neural_top5": "1",
         "markov1_top1": "0", "markov1_top5": "0", "markov2_top1": "0", "markov2_top5": "0"},
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    config = {"neural_fallback_families": [], "deadlines_ms": {"point_lookup": 1},
              "by_scenario": {"zero": {"runtime_cost_ms": {"markov2": {"score_ms": 0.1, "read_ms": 0.1}}, "estimates": {}}}}
    result = summary.offline_summary(path, config, "zero")["policies"]
    assert result["context"]["top1_macro_pct"] == 50
    assert result["context"]["top5_micro_pct"] == 50
    assert result["context"]["skipped_transitions"] == 1
    assert result["model_ranker"]["candidate_recall_pct"] == 50
    assert result["adaptive"]["top1_micro_pct"] == 0
    assert result["adaptive"]["skipped_transitions"] == 2


@pytest.mark.parametrize("policy", ["context", "markov2"])
def test_actual_job_scoring_uses_aligned_demand_and_excludes_ambiguous_history(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(summary, "OUTPUT", tmp_path)
    monkeypatch.setattr(summary, "TEST", [11])
    root = tmp_path / "heldout/zero/seed-11" / policy
    root.mkdir(parents=True)
    pages = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    (root / "demand_reference.json").write_text(json.dumps({"q1": [[page, True] for page in pages]}))
    (root / "workload.jsonl").write_text(json.dumps({"query_id": "q1", "family": "point_lookup"}) + "\n")
    events = [
        {"query_id": "q1", "history": [1, 2, 3, 4, 5, 6, 7, 8], "ranked_candidates": [{"page": 1}]},
        {"query_id": "q1", "history": [2, 3, 4, 5, 6, 7, 8, 1], "ranked_candidates": [{"page": 4}, {"page": 2}]},
    ]
    if policy == "context":
        events[0]["query_page_position"] = 8
        events[1]["query_page_position"] = 9
    (root / "predictions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    result = summary.runtime_prediction_accuracy("zero", policy)
    if policy == "context":
        assert result["scored_aligned_jobs"] == 2
        assert result["top1_micro_pct"] == 50
        assert result["top5_micro_pct"] == 100
    else:
        assert result["scored_aligned_jobs"] == 1
        assert result["alignment"]["ambiguous_legacy_history"] == 1
        assert result["top1_micro_pct"] == 0
        assert result["top5_micro_pct"] == 100
