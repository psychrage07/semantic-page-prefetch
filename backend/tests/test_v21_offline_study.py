"""v2.1 offline study integration test (synthetic-scale, correctness of the
declared-family pipeline rather than of any research result)."""
from __future__ import annotations

import json
import random

from pagewise.offline import run_offline_study_v21


def test_offline_study_v21_produces_declared_family_and_holm(tmp_path):
    rng = random.Random(1)
    seqs, workload = [], []
    for q in range(30):
        pages: list[int] = []
        cur = rng.randrange(1, 15)
        for _ in range(20):
            cur = max(1, min(22, cur + rng.choice([-2, -1, 0, 1, 2])))
            pages.append(cur)
        tid = f"t{q % 4}"
        seqs.append({"query_id": f"q{q}", "pages": pages, "template_id": tid,
                     "workload_family": "f0", "plan_class": "scan"})
        workload.append({"query_id": f"q{q}", "params": {"a": q % 4}, "template_id": tid})
    seq_path = tmp_path / "seq.jsonl"
    wl_path = tmp_path / "wl.jsonl"
    pm_path = tmp_path / "pm.json"
    seq_path.write_text("\n".join(json.dumps(s) for s in seqs))
    wl_path.write_text("\n".join(json.dumps(w) for w in workload))
    page_map = {str(p): {"page_type": "leaf", "depth": 1, "object": "orders"} for p in range(1, 23)}
    pm_path.write_text(json.dumps(page_map))

    report = run_offline_study_v21(seq_path, wl_path, pm_path, tmp_path / "out",
                                   seeds=[101, 102], epochs=3)
    assert report["schema_version"] == "v2.1"
    for split in ("params", "templates"):
        fam = report["family_reports"][split]
        assert fam["members"]
        for member in fam["members"].values():
            assert {"raw_decision", "holm_decision", "ci95_low", "ci95_high",
                    "seeds", "matched_pairs"} <= set(member)
    # candidates from training only + candidate recall + coverage reported
    rows = report["rows"]
    recalls = [r["candidate_recall_pct"] for r in rows if r.get("policy") == "candidate_recall"]
    assert recalls and recalls[0] == 100.0
    coverages = [r["native_coverage_pct"] for r in rows if r.get("policy") == "candidate_recall"]
    assert coverages and {"markov2", "markov_ctx", "semantic"} <= set(coverages[0])
    # Model B preserved as baseline: comparison member exists
    labels = list(report["family_reports"]["params"]["members"])
    assert any("model_b_vs_model_a" in label for label in labels)
    assert any("model_b2" in label for label in labels)
    assert any("model_b3" in label for label in labels)
    # JSON + CSV artifacts written
    assert (tmp_path / "out" / "offline_study_v21.json").exists()
    assert (tmp_path / "out" / "offline_study_v21.csv").exists()
