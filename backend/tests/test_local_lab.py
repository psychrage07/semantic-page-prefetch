"""Local dashboard bridge covers the real CLI and keeps file requests local."""
from pathlib import Path

import pytest

import local_lab


def test_all_cli_commands_and_research_scripts_are_exposed():
    assert len([name for name in local_lab.COMMANDS if not name.startswith("script:")]) >= 22
    assert "script:evaluate_neural:freeze" in local_lab.COMMANDS
    assert "script:run_sim_studies:headroom" in local_lab.COMMANDS
    assert "script:verify_frozen" in local_lab.COMMANDS


def test_required_arguments_and_path_boundaries():
    with pytest.raises(ValueError, match="required"):
        local_lab.build_argv("trace", {})
    with pytest.raises(ValueError, match="Paths must stay"):
        local_lab.build_argv("trace", {"db": "../../outside.sqlite"})
    with pytest.raises(ValueError, match="Unknown options"):
        local_lab.build_argv("check-env", {"arbitrary": "value"})
    with pytest.raises(ValueError, match="choice"):
        local_lab.build_argv("run-prefetch", {
            "db": "runs/example.sqlite", "model-dir": "runs/models", "out": "runs/runtime", "strategy": "invented"
        })
    assert local_lab.build_argv("trace", {"db": "runs/example.sqlite"})[:3] == [
        "trace", "--db", str((local_lab.BACKEND / "runs" / "example.sqlite").resolve())
    ]


def test_demo_pipeline_is_seven_real_cli_steps():
    root = local_lab.LAB / "test-plan"
    steps = local_lab.demo_steps(root, 12)
    assert [step[0] for step in steps] == [
        "generate-db", "trace", "build-sequences", "train", "evaluate", "run-suite", "dashboard-export"
    ]
    assert steps[-1][-1] == str(root / "dashboard.json")
    assert all(Path(part).is_absolute() for step in steps for part in step if part.startswith(str(local_lab.BACKEND)))
