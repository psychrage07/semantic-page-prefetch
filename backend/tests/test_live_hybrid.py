import json

import pytest

import live_hybrid
import local_lab


def test_trained_hybrid_plan_runs_each_policy_on_paired_seeds_without_training():
    steps, labels = local_lab.hybrid_steps(local_lab.LAB / "test-hybrid-plan", 20, 123456)
    assert len(steps) == len(labels) == 9
    assert steps[0][1] == "verify"
    assert steps[-1][1] == "summarize"
    assert steps[-2][1] == "evaluate"
    pairs = [(step[step.index("--seed") + 1], step[step.index("--policy") + 1]) for step in steps[1:-2]]
    assert set(pairs) == {(str(seed), policy) for seed in [123456, 123457] for policy in ["none", "markov2", "model_ranker"]}
    assert not any("train" in step for step in steps)


def test_missing_frozen_artifacts_are_reported_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(live_hybrid, "BACKEND", tmp_path)
    monkeypatch.setattr(live_hybrid, "FROZEN", tmp_path / "frozen")
    monkeypatch.setattr(live_hybrid, "DATABASE", tmp_path / "database.sqlite")
    monkeypatch.setattr(live_hybrid, "POLICY", tmp_path / "policy.json")
    with pytest.raises(ValueError, match="needs these local research artifacts"):
        live_hybrid.assets()


def test_changed_checkpoint_is_rejected_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(live_hybrid, "BACKEND", tmp_path)
    monkeypatch.setattr(live_hybrid, "FROZEN", tmp_path / "frozen")
    monkeypatch.setattr(live_hybrid, "DATABASE", tmp_path / "database.sqlite")
    monkeypatch.setattr(live_hybrid, "POLICY", tmp_path / "policy.json")
    (tmp_path / "frozen/params").mkdir(parents=True)
    for name in ["model_ranker.pt", "candidate_index.json", "page_map.json"]:
        (tmp_path / "frozen/params" / name).write_text("changed", encoding="utf-8")
    (tmp_path / "frozen/static-metadata.npz").write_text("placeholder", encoding="utf-8")
    (tmp_path / "database.sqlite").write_text("placeholder", encoding="utf-8")
    (tmp_path / "frozen/freeze.json").write_text(json.dumps({"splits": {"params": {"models": {"selected": {"sha256": "expected"}}}}}), encoding="utf-8")
    (tmp_path / "policy.json").write_text(json.dumps({"checkpoint_sha256": "expected"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        live_hybrid.assets()


def test_live_metrics_keep_candidate_misses_and_distinguish_query_weighting():
    from live_evaluation import metrics
    rows = [
        {"query_id": "seed1:q1", "family": "lookup", "top1": True, "top5": True, "candidate_hit": True},
        {"query_id": "seed2:q1", "family": "join", "top1": False, "top5": False, "candidate_hit": False},
        {"query_id": "seed2:q1", "family": "join", "top1": False, "top5": True, "candidate_hit": True},
        {"query_id": "seed2:q1", "family": "join", "top1": False, "top5": False, "candidate_hit": False},
    ]
    result = metrics(rows)
    assert result["queries"] == 2
    assert result["transitions"] == 4
    assert result["top1_macro_pct"] == 50
    assert result["top1_micro_pct"] == 25
    assert result["top5_micro_pct"] == 50
    assert result["candidate_recall_pct"] == 50
    assert result["by_family"]["join"]["top1_micro_pct"] == 0


def test_latest_selected_result_ignores_old_runtime_only_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(local_lab, "JOBS", {
        "old": {"id": "old", "preset": "trained-hybrid", "status": "completed", "root": str(tmp_path), "started_at": "2026-10-24"}
    })
    monkeypatch.setattr(local_lab, "safe_path", lambda path: tmp_path / "hybrid-live.json")
    path = tmp_path / "hybrid-live.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert local_lab.latest_selected_result()["available"] is False
    path.write_text(json.dumps({"evaluation": {"semantics_enabled": True}}), encoding="utf-8")
    result = local_lab.latest_selected_result()
    assert result["available"] is True
    assert result["job_id"] == "old"
    assert result["result"]["evaluation"]["semantics_enabled"] is True
