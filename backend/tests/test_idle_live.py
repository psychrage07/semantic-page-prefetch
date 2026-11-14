from pathlib import Path

import pytest

import idle_live
import local_lab


def test_idle_live_requires_known_dataset_and_existing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(idle_live, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="small or large"):
        idle_live.assets("../large")
    assert not idle_live.assets("small")["available"]


def test_idle_job_validates_count_and_builds_the_real_helper_command(monkeypatch, tmp_path):
    monkeypatch.setattr(local_lab, "LAB", tmp_path / "lab")
    monkeypatch.setattr(local_lab, "BACKEND", tmp_path)
    monkeypatch.setattr(local_lab, "JOBS", {})
    monkeypatch.setattr(idle_live, "assets", lambda dataset: {"available": True})
    monkeypatch.setattr(local_lab.threading.Thread, "start", lambda self: None)
    with pytest.raises(ValueError, match="between 10 and 100"):
        local_lab.start_job({"preset": "idle-hybrid", "queries": 101})
    result = local_lab.start_job({"preset": "idle-hybrid", "dataset": "large", "queries": 10})
    job = local_lab.JOBS[result["id"]]
    step = job["steps"][0]
    assert step[0] == "local:idle-hybrid"
    assert step[step.index("--dataset") + 1] == "large"
    assert step[step.index("--queries") + 1] == "10"
    assert result["result_file"] == "idle-result.json"
    assert result["preset"] == "idle-hybrid"
