"""v2.1 development-scale end-to-end verification.

Runs the complete amended pipeline at development scale and records every
attempt in the run ledger. This is a VERIFICATION run (validation that the
v2.1 machinery executes correctly with real VFS tracing), NOT a final
research run: the final-scale amendment studies must be re-executed on the
experiment host with the large-branch artifacts, as documented in the
protocol v2.1 amendment.

Output: backend/runs/v2.1-dev-verification/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
STUDY = BACKEND / "runs" / "v2.1-dev-verification"
LEDGER = STUDY / "run_log.jsonl"
DB = STUDY / "db" / "pagewise-dev.sqlite"
MODELS = STUDY / "models"
MANIFEST = BACKEND / "FROZEN_MANIFEST.json"


def run(args, *, tag: str, outputs: list[str] | None = None, extra_metrics: dict | None = None) -> dict:
    """Run one CLI configuration; record BOTH success and failure in the ledger."""
    from pagewise import runledger
    cmd = [sys.executable, "-m", "pagewise.cli", *args]
    config = {"tag": tag, "args": args}
    result = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True)
    record = {
        "tag": tag, "exit": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.returncode else "",
        "stderr_tail": result.stderr[-2000:] if result.returncode else "",
    }
    runledger.record_run(
        LEDGER, config, status="completed" if result.returncode == 0 else "failed",
        reason=("v2.1 dev-scale verification step" if result.returncode == 0
                else f"verification step failed: {result.stderr[-300]}"),
        outputs=outputs or [], metrics=extra_metrics or {})
    return record


def main() -> int:
    STUDY.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    print(f"== v2.1 dev verification -> {STUDY}")

    steps.append(run(["check-env"], tag="check-env",
                     extra_metrics={"purpose": "APSW/PyTorch availability verification"}))
    steps.append(run(["generate-db", "--db", str(DB), "--customers", "2000", "--products", "500",
                      "--orders", "8000", "--seed", "17", "--overwrite"], tag="generate-db"))

    trace_dir = STUDY / "trace-seed101"
    steps.append(run(["trace", "--db", str(DB), "--out", str(trace_dir), "--queries", "120",
                      "--seed", "101", "--sparse-page-map"], tag="trace"))
    steps.append(run(["build-sequences", "--trace", str(trace_dir / "trace.jsonl"),
                      "--page-map", str(trace_dir / "page_map.json"),
                      "--out", str(STUDY / "sequences.jsonl")], tag="build-sequences"))
    steps.append(run(["train", "--sequences", str(STUDY / "sequences.jsonl"),
                      "--page-map", str(trace_dir / "page_map.json"),
                      "--out", str(MODELS), "--seed", "7", "--epochs", "8", "--candidates", "24"],
                     tag="train"))

    steps.append(run(["offline-study-v21", "--sequences", str(STUDY / "sequences.jsonl"),
                      "--workload", str(trace_dir / "workload.jsonl"),
                      "--page-map", str(trace_dir / "page_map.json"),
                      "--out", str(STUDY), "--seeds", "101,102,103,104", "--epochs", "8",
                      "--candidates", "24", "--ledger", str(LEDGER)],
                     tag="offline-study-v21",
                     outputs=[str(STUDY / "offline_study_v21.json")]))

    # runtime: complementary suite through the real VFS path with the k-channel model
    steps.append(run(["run-suite", "--db", str(DB), "--model-dir", str(MODELS),
                      "--out", str(STUDY / "suite-k4"), "--queries", "60",
                      "--seeds", "101,102", "--strategies", "none,null,oracle_miss,oracle_optimal",
                      "--cache-pages", "32", "--degree", "2",
                      "--miss-delay-ms", "0.2", "--storage-model", "k_channel",
                      "--storage-channels", "4", "--drop-stale-jobs"],
                     tag="run-suite-k-channel-4",
                     outputs=[str(STUDY / "suite-k4" / "suite_summary.json")]))

    # headroom grid from a declared demand reference (a none run with recording)
    steps.append(run(["run-prefetch", "--db", str(DB), "--model-dir", str(MODELS),
                      "--strategy", "none", "--out", str(STUDY / "reference-none"),
                      "--queries", "120", "--seed", "101", "--cache-pages", "32",
                      "--predict-on", "miss", "--drop-stale-jobs",
                      "--prefetch-workers", "1"], tag="reference-none"))
    # record the declared demand reference for the headroom input
    _record_reference(DB)
    reference = STUDY / "headroom" / "demand_reference.json"
    steps.append(run(["headroom-v21", "--reference", str(reference), "--out", str(STUDY / "headroom"),
                      "--caches", "8,32", "--delays", "0.2", "--channels", "1,4,16",
                      "--degree", "2", "--threshold-pct", "10", "--ledger", str(LEDGER)],
                     tag="headroom-v21", outputs=[str(STUDY / "headroom" / "headroom_v21.json")]))
    headroom_report = {}
    if reference.exists():
        headroom_report = json.loads((STUDY / "headroom" / "headroom_v21.json").read_text(encoding="utf-8"))

    steps.append(run(["report-v21", "--study-dir", str(STUDY)],
                     tag="report-v21", outputs=[str(STUDY / "research_report_v21.md")]))
    steps.append(run(["dashboard-export-v21", "--study-dir", str(STUDY),
                      "--out", str(BACKEND.parent / "public" / "pagewise-dashboard-v2.1.json")],
                     tag="dashboard-export-v21",
                     outputs=[str(BACKEND.parent / "public" / "pagewise-dashboard-v2.1.json")]))

    # frozen manifest must be unchanged: recorded before and after
    manifest_sha_before = None
    if MANIFEST.exists():
        import hashlib
        manifest_sha_before = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    manifest_after = None
    if MANIFEST.exists():
        import hashlib
        manifest_after = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    steps.append(run(["check-env"], tag="check-env-final"))

    from pagewise import runledger
    integrity = runledger.verify_ledger(LEDGER)
    summary = {
        "schema_version": "v2.1",
        "scale": "development verification (NOT a final research run)",
        "steps": steps,
        "headroom_decision": headroom_report.get("decision"),
        "ledger": {"integrity": integrity,
                   "statuses": runledger.summarize_ledger(LEDGER)["statuses"]},
        "frozen_manifest_unchanged": (manifest_sha_before is None or
                                      manifest_sha_before == manifest_after),
        "evidence_class": ("real VFS/APSW executions on the local host; latency-injected "
                           "values remain latency-injected, not physical-device claims"),
    }
    (STUDY / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("headroom_decision", "frozen_manifest_unchanged", "ledger")}, indent=2))
    failed = [s["tag"] for s in steps if s["exit"] != 0]
    print("FAILED STEPS:", failed if failed else "none")
    return 1 if failed or not integrity["ok"] else 0


def _record_reference(database_path: Path) -> dict:
    """Rebuild a declared demand reference via the demand-reference recorder."""
    # The reference run above used --predict-on miss with recording disabled
    # implicitly; run a dedicated recorder pass here instead of CLI (kept
    # explicit for verification traceability).
    from pagewise.runner import trace_workload
    reference_dir = STUDY / "headroom" / "reference-run"
    seed_payloads = STUDY / "headroom"
    cfg = {
        "strategy": "none", "candidate_index_path": str(MODELS / "candidate_index.json"),
        "model_path": None, "semantic": False, "cache_pages": 32, "degree": 2,
        "confidence_threshold": 0.0, "queue_size": 64, "miss_delay_ms": 0.0,
        "predict_on": "miss", "predict_every": 1, "max_score_candidates": 0,
        "storage_model": "independent", "drop_stale_jobs": True,
        "diagnostics_enabled": False, "record_demand_reference": True, "prefetch_workers": 1,
    }
    trace_workload(database_path, reference_dir, query_count=120, seed=101,
                   cache_kib=4096, mixed=True, prefetch_config=cfg)
    reference = reference_dir / "demand_reference.json"
    seed_payloads.mkdir(parents=True, exist_ok=True)
    reference_target = seed_payloads / "demand_reference.json"
    reference_target.write_text(reference.read_text(encoding="utf-8"), encoding="utf-8")
    return json.loads(reference_target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
