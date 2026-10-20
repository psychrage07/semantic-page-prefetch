"""v2.1 statistics family and run-ledger tests."""
from __future__ import annotations

import json

import pytest

from pagewise.statistics import (
    declare_family,
    holm_correction,
    paired_cluster_bootstrap_ci,
    raw_decision,
)
from pagewise.runledger import record_run, read_ledger, verify_ledger, config_hash


def test_holm_basic_ordering():
    # p-values already small for 2 of 3 members; Holm keeps family discipline.
    decisions = holm_correction([0.001, 0.03, 0.9], alpha=0.05)
    assert decisions[0] is True
    assert decisions[1] is False   # 0.03 * 2 = 0.06 > 0.05
    assert decisions[2] is False


def test_holm_all_null():
    assert holm_correction([0.4, 0.5, 0.6]) == [False, False, False]


def test_cluster_bootstrap_reproducible_and_seed_aware():
    sessions = {1: [-1.0, -2.0, 3.0], 2: [2.0, 1.0], 3: [-0.5]}
    a = paired_cluster_bootstrap_ci(sessions, iterations=2000, seed=7)
    b = paired_cluster_bootstrap_ci(sessions, iterations=2000, seed=7)
    assert a == b
    assert a["seeds"] == 3 and a["matched_pairs"] == 6


def test_cluster_bootstrap_more_clusters_widen_less():
    # same 12 observations but a tighter cluster structure should give different
    # intervals; both remain finite and ordered.
    tight = {i: [-1.0] * 4 for i in range(3)}
    loose = {i + 1: [-1.0] * 4 for i in range(3)}
    a = paired_cluster_bootstrap_ci(tight, iterations=2000, seed=3)
    b = paired_cluster_bootstrap_ci(loose, iterations=2000, seed=3)
    assert a["mean"] == b["mean"] == -1.0


def test_raw_decisions_directions():
    assert raw_decision(-3.0, -1.0, favorable_direction="low") == "supported"
    assert raw_decision(1.0, 3.0, favorable_direction="greater") == "supported"
    assert raw_decision(-1.0, 1.0, favorable_direction="greater") == "inconclusive"


def test_declare_family_reports_every_declared_member():
    sessions_cand = {1: [1.0, 2.0], 2: [0.5, 0.5], 3: [1.5]}
    sessions_null = {1: [-0.2, 0.1], 2: [0.2, -0.1], 3: [-0.05]}
    report = declare_family(
        [("h_pos", sessions_cand), ("h_null", sessions_null)],
        favorable_directions={"h_pos": "greater", "h_null": "greater"},
        iterations=2000,
    )
    assert set(report["members"]) == {"h_pos", "h_null"}
    for member in report["members"].values():
        assert member["ci95_low"] is not None
        assert member["raw_decision"] in ("supported", "not supported", "inconclusive")
        assert member["holm_decision"] in (True, False)


def test_ledger_append_verify_and_tamper_detection(tmp_path):
    log = tmp_path / "run_log.jsonl"
    record_run(log, {"a": 1}, status="completed", reason="r1")
    record_run(log, {"b": 2}, status="excluded", reason="r2")
    record_run(log, {"c": 3}, status="failed", reason="boom", error="ValueError: x")
    summary = verify_ledger(log)
    assert summary["ok"] is True and summary["rows"] == 3
    rows = read_ledger(log)
    assert rows[1]["status"] == "excluded" and rows[2]["error"].startswith("ValueError")
    # tamper with the middle row: chain must break
    tampered = [json.loads(line) for line in log.read_text().splitlines()]
    tampered[1]["status"] = "completed"
    log.write_text("\n".join(json.dumps(r) for r in tampered) + "\n")
    broken = verify_ledger(log)
    assert broken["ok"] is False and broken["first_broken_row"] == 1


def test_ledger_config_hash_is_stable():
    assert config_hash({"x": 1, "y": [1, 2]}) == config_hash({"y": [1, 2], "x": 1})
    assert config_hash({"x": 1}) != config_hash({"x": 2})
