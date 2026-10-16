import json

from scripts.hypothesis_report import build_hypothesis_report


def test_no_headroom_report_maps_runtime_hypotheses(tmp_path):
    offline = {"hypotheses": {
        "H1": {"outcome": "supported"},
        "H2": {"outcome": "not supported"},
        "H5_offline": {"outcome": "not supported"},
    }}
    headroom = {"decision": "NO_HEADROOM", "best_primary": {"headroom_pct": -5.0}}
    suite = {
        "mode": "latency-injected", "storage_model": "shared",
        "strategies": [
            {"strategy": "none", "median_total_query_ms": 100.0},
            {"strategy": "oracle_miss", "median_total_query_ms": 130.0},
        ],
        "paired_query_comparisons": [{
            "strategy": "oracle_miss", "baseline": "none",
            "delta_ci95_low_ms": 1.0, "delta_ci95_high_ms": 4.0,
        }],
    }
    paths = []
    for name, payload in (("offline", offline), ("headroom", headroom), ("suite", suite)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    report = build_hypothesis_report(*paths)
    assert report["hypotheses"]["H1"]["outcome"] == "supported"
    assert report["hypotheses"]["H3"]["outcome"] == "not supported"
    assert report["hypotheses"]["H4"]["outcome"] == "not supported"
    assert report["hypotheses"]["H5"]["outcome"] == "not supported"
    assert report["runtime"]["holm_correction"]["status"] == "not applicable"
