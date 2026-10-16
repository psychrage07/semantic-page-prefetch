"""Build the final H1-H5 report from frozen Pagewise result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_hypothesis_report(offline_path: str | Path, headroom_path: str | Path,
                            oracle_suite_path: str | Path) -> dict[str, Any]:
    offline = _load(offline_path)
    headroom = _load(headroom_path)
    oracle_suite = _load(oracle_suite_path)

    offline_h = offline["hypotheses"]
    decision = str(headroom.get("decision"))
    best = headroom.get("best_primary")
    oracle_pair = next(
        row for row in oracle_suite["paired_query_comparisons"]
        if row["strategy"] == "oracle_miss" and row["baseline"] == "none"
    )
    strategy_rows = {row["strategy"]: row for row in oracle_suite["strategies"]}
    no_headroom = decision == "NO_HEADROOM" and float(oracle_pair["delta_ci95_low_ms"]) > 0

    h3_outcome = "not supported" if no_headroom else "inconclusive"
    h4_outcome = "not supported" if no_headroom else "inconclusive"
    h5_offline_outcome = offline_h["H5_offline"]["outcome"]
    if h5_offline_outcome == "supported" and h3_outcome == "supported":
        h5_outcome = "supported"
    elif "not supported" in {h5_offline_outcome, h3_outcome}:
        h5_outcome = "not supported"
    else:
        h5_outcome = "inconclusive"

    outcomes = {
        "H1": offline_h["H1"],
        "H2": offline_h["H2"],
        "H3": {
            "statement": "The best Model B variant has total time below none.",
            "outcome": h3_outcome,
            "scope": "shared-device NO_HEADROOM branch",
            "evidence": {"headroom_decision": decision, "best_simulated_cell": best,
                         "final_oracle_vs_none": oracle_pair},
        },
        "H4": {
            "statement": "The best Model B variant has lower total time than the declared simpler baselines.",
            "outcome": h4_outcome,
            "scope": "shared-device NO_HEADROOM branch",
            "evidence": {"headroom_decision": decision,
                         "reason": "The future-aware oracle control did not beat none, so neural runtime tuning was stopped by the frozen branch rule."},
        },
        "H5": {
            "statement": "H1 and H3 still hold on held-out query templates.",
            "outcome": h5_outcome,
            "scope": "held-out templates plus shared-device runtime branch",
            "evidence": {"offline": offline_h["H5_offline"], "runtime": h3_outcome},
        },
    }

    if no_headroom:
        headline = ("Under the tested shared-device conditions, the future-aware oracle control "
                    "was slower than no prefetching, so the protocol found no usable prefetch headroom.")
    elif outcomes["H1"]["outcome"] == "supported" and outcomes["H2"]["outcome"] != "supported":
        headline = "Semantic context helps, but a cheap context-keyed table captures it, so a neural network is not required."
    else:
        headline = "The tested evidence does not support a runtime benefit for semantic page prefetching."

    return {
        "schema_version": 1,
        "headline": headline,
        "hypotheses": outcomes,
        "runtime": {
            "mode": oracle_suite.get("mode"),
            "storage_model": oracle_suite.get("storage_model"),
            "none_total_query_ms": strategy_rows["none"]["median_total_query_ms"],
            "oracle_miss_total_query_ms": strategy_rows["oracle_miss"]["median_total_query_ms"],
            "holm_correction": {
                "status": "not applicable",
                "reason": "The preregistered NO_HEADROOM branch stopped H3/H4 model comparisons before a family of neural runtime tests was formed.",
            },
        },
        "limitations": [
            "The oracle is a future-aware experimental control, not a mathematical upper bound for every possible prefetcher.",
            "Injected delays are measured host timer delays and are not hardware storage measurements.",
            "VFS reads may be served by the operating-system page cache.",
        ],
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_hypotheses.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Pagewise final hypothesis report", "", report["headline"], "",
             "| Hypothesis | Outcome |", "|---|---|"]
    for name in ("H1", "H2", "H3", "H4", "H5"):
        lines.append(f"| {name} | {report['hypotheses'][name]['outcome']} |")
    lines.extend(["", "## Runtime branch", "",
                  f"Storage model: `{report['runtime']['storage_model']}`.",
                  f"No-prefetch total query time: {report['runtime']['none_total_query_ms']:.4f} ms.",
                  f"Oracle-miss total query time: {report['runtime']['oracle_miss_total_query_ms']:.4f} ms.",
                  "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (out / "final_hypotheses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", required=True)
    parser.add_argument("--headroom", required=True)
    parser.add_argument("--oracle-suite", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = build_hypothesis_report(args.offline, args.headroom, args.oracle_suite)
    write_report(report, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
