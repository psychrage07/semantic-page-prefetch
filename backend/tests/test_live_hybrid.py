import json

import pytest

import live_hybrid
import local_lab


def test_trained_hybrid_plan_runs_each_policy_on_paired_seeds_without_training():
    steps, labels = local_lab.hybrid_steps(local_lab.LAB / "test-hybrid-plan", 20, 123456)
    assert len(steps) == len(labels) == 8
    assert steps[0][1] == "verify"
    assert steps[-1][1] == "summarize"
    pairs = [(step[step.index("--seed") + 1], step[step.index("--policy") + 1]) for step in steps[1:-1]]
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
