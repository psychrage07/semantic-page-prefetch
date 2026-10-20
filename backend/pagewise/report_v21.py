"""v2.1 consolidated research report generator (appended reporting layer).

Mechanically generates the v2.1 research report, final summary and dashboard
snapshot tables from VERIFIED experiment artifacts under a v2.1 study
directory only. It never reads or edits frozen v2 result directories and never
modifies FROZEN_MANIFEST.json. Claims are separated into supported / not
supported / inconclusive, and offline-prediction, simulator, latency-injected
runtime, and physical-device evidence classes are kept distinct.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .runledger import summarize_ledger

NEURAL_POLICIES = ("markov2", "markov_ctx", "model_a", "model_b", "model_b2", "model_b3")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_error": f"{path.name} is not valid JSON"}


def build_v21_report(study_dir: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Generate `research_report_v21.md` (and a final summary block) mechanically."""
    study = Path(study_dir).resolve()
    out_path = Path(out_path) if out_path else study / "research_report_v21.md"
    headroom = (_load_json(study / "headroom_v21.json") or _load_json(study / "headroom" / "headroom_v21.json"))
    offline = _load_json(study / "offline_study_v21.json")
    ledger_summary = summarize_ledger(study / "run_log.jsonl") if (study / "run_log.jsonl").exists() else None
    suites = sorted(study.glob("suite*/suite_summary.json"))

    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    w("# Pagewise v2.1 consolidated report (amendment)")
    w()
    w(f"Generated mechanically from verified artifacts in `{study.name}` on "
      + time.strftime("%Y-%m-%d", time.gmtime()) + ". No number in this report is hand-written.")
    w()
    w("## Changes vs v2 (Requirement D identification)")
    w()
    w("- Oracle definition change: v2 headroom used `oracle_miss` (future-aware diagnostic, "
      "sometimes slower than no prefetching). v2.1 measures headroom with `oracle_optimal`, the "
      "declared offline policy over the action family {no-prefetch} ∪ {prefetch-next-misses(K)}, "
      "a certified lower bound. `oracle_miss` remains a diagnostic column, preserved as measured.")
    w("- Storage model change: v2 `shared` (k=1 serialized channel) is joined by the abstract "
      "`k_channel` model evaluated at k = 1/4/16. All concurrency results are simulated "
      "scenarios, not physical-device measurements; no HDD/NVMe label is claimed.")
    w()
    supported: list[str] = []
    not_supported: list[str] = []
    inconclusive: list[str] = []

    if headroom is None:
        w("## Headroom decision")
        w()
        w("`headroom_v21.json` not found under this study directory — no headroom claim generated.")
        w()
    else:
        w("## Headroom decision (oracle_optimal, k-channel grid)")
        w()
        completed = [c for c in headroom.get("cells", []) if c.get("status") == "completed"]
        failed = [c for c in headroom.get("cells", []) if c.get("status") != "completed"]
        w(f"- Decision: **{headroom.get('decision')}**")
        w(f"- {headroom.get('decision_note', '')}")
        w(f"- Working set: {headroom.get('working_set_pages')} pages; threshold "
          f"{headroom.get('threshold_pct')}%")
        w(f"- Completed cells: {len(completed)}; failed cells retained: {len(failed)}")
        if completed:
            w()
            w("| cache | delay_ms | k | none_ms | oracle_optimal_ms | oracle_miss_ms (diagnostic) | headroom % | eligible |")
            w("|---|---|---|---|---|---|---|---|")
            for c in completed:
                w(f"| {c['cache_pages']} | {c['delay_ms']} | {c['k']} | {c['none_total_ms']} | "
                  f"{c['oracle_optimal_total_ms']} | {c.get('oracle_miss_diagnostic_total_ms')} | "
                  f"{c.get('headroom_pct')} | {c.get('eligible')} |")
        if failed:
            w()
            w("- Failed cells (recorded, never hidden):")
            for f in failed:
                w(f"  - cache={f.get('cache_pages')} delay={f.get('delay_ms')} k={f.get('k')}: "
                  f"{str(f.get('error'))[:160]}")
        decision = headroom.get("decision")
        if decision == "REOPEN_NO_HEADROOM":
            supported.append("NO_HEADROOM reopened: at least one eligible k>=4 configuration shows "
                             ">= threshold oracle_optimal headroom.")
        elif decision == "NO_HEADROOM":
            inconclusive.append("No >=10% headroom demonstrated within the declared action family by "
                                 "oracle_optimal in every eligible k>=4 cell of the tested grid "
                                 "(a lower bound; practical-policy benefit is not ruled out).")
        else:
            inconclusive.append("Headroom decision inconclusive: no eligible cells in the grid.")
        w()

    if offline is None:
        w("## Offline model family (B2/B3)")
        w()
        w("`offline_study_v21.json` not found — no offline claim generated.")
        w()
    else:
        w("## Offline model family (all-read and miss-only training, temperature-calibrated)")
        w()
        rows = [r for r in offline.get("rows", []) if r.get("policy") in NEURAL_POLICIES]
        recalls = [r for r in offline.get("rows", []) if r.get("policy") == "candidate_recall"]
        if recalls:
            cov = recalls[0].get("native_coverage_pct", {})
            w(f"- Candidate-generation recall (common held-out transitions): **{recalls[0].get('candidate_recall_pct')}%**")
            w(f"- Native candidate coverage: markov2 {cov.get('markov2')}% / "
              f"markov_ctx {cov.get('markov_ctx')}% / semantic {cov.get('semantic')}%")
        if rows:
            w()
            w("| split | policy | top-1 % | top-5 % | timely coverage % | inference µs/pred |")
            w("|---|---|---|---|---|---|")
            for r in rows:
                w(f"| {r['split']} | {r['policy']} | {r.get('top1_accuracy_pct')} | "
                  f"{r.get('top5_accuracy_pct')} | {r.get('timely_coverage_pct')} | "
                  f"{r.get('inference_us_per_prediction')} |")
        w()
        w("### Declared family decisions (Holm-corrected, clustered bootstrap)")
        w()
        for split, fam in offline.get("family_reports", {}).items():
            w(f"Split: **{split}**")
            w()
            w("| member | mean delta (pp) | raw 95% CI | raw decision | Holm decision | seeds | matched pairs |")
            w("|---|---|---|---|---|---|---|")
            for label, m in fam.get("members", {}).items():
                holm = "supported" if m.get("holm_decision") else ("inconclusive" if m.get("raw_decision") == "inconclusive" else "not supported")
                w(f"| {label} | {m.get('mean')} | [{m.get('ci95_low')}, {m.get('ci95_high')}] | "
                  f"{m.get('raw_decision')} | {holm} | {m.get('seeds')} | {m.get('matched_pairs')} |")
                sentence = f"{label} ({split}): {holm}"
                {"supported": supported, "not supported": not_supported,
                 "inconclusive": inconclusive}.get(holm, inconclusive).append(sentence)
            w()
        calib = [r.get("calibration") for r in offline.get("rows", []) if r.get("calibration")]
        if calib:
            w("Calibration (validation-fit temperature; softmax over competing candidates): "
              + json.dumps([c for c in calib if c][:2]))
            w()
        if offline.get("fairness_note"):
            w(offline["fairness_note"])
            w()

    if suites:
        w("## v2.1 runtime suites present in this study directory")
        w()
        for path in suites:
            summary = _load_json(path) or {}
            w(f"- `{path.parent.name}`: strategies "
              + ", ".join(r.get("strategy", "?") for r in summary.get("strategies", [])[:12]))
        w()

    w("## Claim classes")
    w()
    for name, class_list in (("Supported", supported), ("Not supported", not_supported),
                             ("Inconclusive", inconclusive)):
        w(f"### {name}")
        w()
        if class_list:
            for item in class_list:
                w(f"- {item}")
        else:
            w("- (none generated from the artifacts in this directory)")
        w()

    if ledger_summary:
        w("## Run ledger integrity")
        w()
        w(f"- Recorded configuration attempts: **{ledger_summary['total']}** "
          f"(statuses: {ledger_summary['statuses']})")
        integrity = ledger_summary.get("integrity", {})
        marker = "OK" if integrity.get("ok") else f"BROKEN at row {integrity.get('first_broken_row')}"
        w(f"- Hash-chain integrity: {marker}")
        w()

    w("## Evidence-class boundary (unchanged from v2)")
    w()
    w("VFS `xRead` counts and injected delays are never claims about physical disk I/O. "
      "No physical HDD/NVMe result may be cited from any artifact here; measured device "
      "calibration requires a dedicated hardware section.")
    w()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": str(out_path), "lines": len(lines), "suites_seen": [str(s) for s in suites],
            "ledger_summary": ledger_summary}


def build_v21_dashboard(study_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Compact JSON snapshot for the React dashboard, from v2.1 artifacts only.

    Written to a NEW file (default `public/pagewise-dashboard-v2.1.json`); the
    frozen v2 dashboard snapshot is never modified.
    """
    study = Path(study_dir).resolve()
    headroom = (_load_json(study / "headroom_v21.json") or _load_json(study / "headroom" / "headroom_v21.json"))
    offline = _load_json(study / "offline_study_v21.json")
    report = {"schema_version": "v2.1", "generated_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if headroom:
        report["headroom_v21"] = {
            "decision": headroom.get("decision"),
            "decision_note": headroom.get("decision_note"),
            "reopen_configurations": headroom.get("reopen_configurations", []),
            "cells": headroom.get("cells", []),
        }
    if offline:
        rows = [r for r in offline.get("rows", []) if r.get("policy") in NEURAL_POLICIES]
        report["offline_v21"] = {
            "rows": rows,
            "candidate_recall_pct": [r for r in offline.get("rows", [])
                                     if r.get("policy") == "candidate_recall"],
            "family_reports": offline.get("family_reports", {}),
        }
    ledger = summarize_ledger(study / "run_log.jsonl") if (study / "run_log.jsonl").exists() else None
    if ledger:
        report["run_ledger"] = ledger
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"dashboard_path": str(path), "sections": sorted(report.keys())}
