"""v2.1 experiment orchestration: oracle headroom grid (correctly defined
oracle), NO_HEADROOM reopen rule, simulator-cell diagnosis, and ledger wiring.

Frozen-artifact discipline: nothing here edits ``runs/experiment-latest`` or any
frozen directory, and the FROZEN_MANIFEST is never modified; all v2.1 outputs go
to new ``v2.1`` output directories.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from . import runledger
from .sim import SimResult, Costs, load_reference, reference_stats, parse_storage_model, simulate


def run_headroom_v21(reference_path: str | Path, out_dir: str | Path, *,
                     caches: list[int] | None = None, delays: list[float] | None = None,
                     channels: list[int] | None = None, storage_model: str = "k_channel",
                     degree: int = 2, threshold_pct: float = 10.0, costs: Costs | None = None,
                     ledger_path: str | Path | None = None,
                     extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Oracle headroom grid under the v2.1 semantics.

    * Headroom is measured from ``oracle_optimal`` — the correctly defined
      offline policy that knows the future, optimizes admission/degree over the
      declared action family, and may select the no-prefetch action. The v2
      diagnostic ``oracle_miss`` remains reported as a diagnostic column only;
      its measured values (even when worse than no prefetching) are preserved
      unmodified.
    * Eligibility follows the v2 rule: cache <= working-set pages / 4.
    * Reopen rule (v2.1 amendment): NO_HEADROOM stands only when the correctly
      defined oracle has < threshold headroom in EVERY eligible k >= 4 config.
      Any eligible configuration with >= threshold headroom REOPENs the runtime
      comparison for that configuration.
    """
    caches = caches or [32, 64, 128, 256, 512]
    delays = delays or [0.1, 0.2, 0.5, 1.0]
    channels = channels or [1, 4, 16]
    costs = costs or Costs()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = load_reference(reference_path)
    stats = reference_stats(reference)
    working_set_pages = len({page for row in reference for page in row["pages"]})
    cells: list[dict[str, Any]] = []
    for cache_pages in caches:
        for delay in delays:
            for k in channels:
                params = {
                    "cache_pages": cache_pages, "delay_ms": delay, "channels": k,
                    "storage_model": ("shared" if k == 1 else storage_model),
                    "degree": degree, "reference_sha256": runledger.config_hash(
                        {"reference": str(reference_path)}),
                }
                model = "shared" if k == 1 else storage_model
                try:
                    base = simulate(reference, strategy="none", cache_pages=cache_pages,
                                    degree=degree, delay_ms=delay, costs=costs,
                                    storage_model=model, storage_channels=k)
                    oracle = simulate(reference, strategy="oracle_optimal", cache_pages=cache_pages,
                                      degree=degree, delay_ms=delay, costs=costs,
                                      storage_model=model, storage_channels=k)
                    diagnostic = simulate(reference, strategy="oracle_miss", cache_pages=cache_pages,
                                          degree=degree, delay_ms=delay, costs=costs,
                                          storage_model=model, storage_channels=k)
                except Exception as exc:  # failed configurations are recorded, never hidden
                    record = runledger.record_run(ledger_path or out / "run_log.jsonl",
                                                  params, status="failed",
                                                  reason=repr(exc)[:400])
                    cells.append({**params, "status": "failed", "error": str(exc),
                                  "ledger_sha": record["row_sha256"]})
                    continue
                headroom_pct = (100.0 * (base.total_ms - oracle.total_ms) / base.total_ms
                                if base.total_ms > 0 else 0.0)
                eligible = cache_pages * 4 <= working_set_pages
                cell = {
                    **params,
                    "status": "completed",
                    "k": k,
                    "none_total_ms": base.total_ms,
                    "oracle_optimal_total_ms": oracle.total_ms,
                    "oracle_miss_diagnostic_total_ms": diagnostic.total_ms,
                    "headroom_pct": round(headroom_pct, 4),
                    "eligible": eligible,
                    "cpu_fraction_of_total": base.stats.get("cpu_fraction_of_total"),
                    "amdahl_max_reduction_ms": base.stats.get("amdahl_max_reduction_ms"),
                    "oracle_action": (oracle.stats.get("oracle_invariants") or {}).get("chosen_action"),
                    "oracle_invariants": oracle.stats.get("oracle_invariants"),
                    "units": "virtual-time simulation, milliseconds",
                }
                cells.append(cell)
                runledger.record_run(
                    ledger_path or out / "run_log.jsonl", params, status="completed",
                    seed=None,
                    outputs=[str(out / "headroom_v21.json")],
                    metrics={"headroom_pct": cell["headroom_pct"],
                             "none_total_ms": base.total_ms,
                             "oracle_optimal_total_ms": oracle.total_ms,
                             "oracle_miss_diagnostic_total_ms": diagnostic.total_ms})
    eligible_cells = [c for c in cells if c.get("status") == "completed" and c.get("eligible")]
    eligible_k4 = [c for c in eligible_cells if c.get("k", 1) >= 4]
    reopen = [c for c in eligible_cells if c.get("headroom_pct", 0.0) >= threshold_pct]
    reopen_k4 = [c for c in reopen if c.get("k", 1) >= 4]
    if not eligible_k4 and not eligible_cells:
        decision = "no_eligible_cells"
        decision_note = ("No cell satisfied cache <= working-set/4; the v2 premise of the "
                         "eligibility rule must be rechecked before concluding NO_HEADROOM.")
    elif reopen_k4:
        decision = "REOPEN_NO_HEADROOM"
        decision_note = (f"{len(reopen_k4)} eligible k>=4 configuration(s) show >= "
                         f"{threshold_pct}% certified oracle_optimal headroom; the NO_HEADROOM "
                         "decision does not stand and the runtime comparison reopens for those "
                         "configurations.")
    else:
        decision = "NO_HEADROOM"
        decision_note = (f"The correctly defined oracle (oracle_optimal) has < {threshold_pct}% "
                         "headroom in every eligible k>=4 configuration under the modeled "
                         "constraints; NO_HEADROOM stands. Confidence scope: only the tested "
                         "cache/delay/k grid and the modeled constraints — not every possible "
                         "storage architecture.")
    report = {
        "schema_version": "v2.1",
        "oracle_semantics": {
            "headroom_source": "oracle_optimal (declared offline policy; no-prefetch action included)",
            "diagnostic_retained": "oracle_miss (future-aware diagnostic policy; results preserved as measured, never forced)",
            "oracle_next_role": "naive control, never a bound",
        },
        "working_set_pages": working_set_pages,
        "threshold_pct": threshold_pct,
        "decision": decision,
        "decision_note": decision_note,
        "reopen_configurations": [c for c in reopen_k4],
        "cells": cells,
        "storage_label_caveat": ("k=1 reproduces the v2 shared (serialized) model; k=4 and "
                                 "k=16 are abstract simulated concurrency scenarios without "
                                 "physical-device labels until a device-specific calibration "
                                 "section is executed on target hardware."),
        **(extra_metadata or {}),
    }
    (out / "headroom_v21.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def diagnose_sim_cells(calibration_path: str | Path, run_dirs: list[str | Path],
                       *, out_dir: str | Path, ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Diagnose simulator-validation cells against the declared 15% target.

    Requirement A: report each declared cell's configuration, error value and
    cause. Failed configurations are recorded — never removed. For runs whose
    demand reference is missing, the row is marked ``needs_reference`` with the
    exact path that must exist, not silently skipped.
    """
    from .runner import trace_workload  # noqa: F401  (kept for host-side re-runs)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    costs = Costs(**{key: value for key, value in calibration["costs"].items()
                     if key in Costs.__dataclass_fields__})
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir).resolve()
        metadata_path = run_path / "run_metadata.json"
        params: dict[str, Any] = {"run_dir": str(run_path)}
        if not metadata_path.exists():
            rows.append({**params, "status": "missing_metadata"})
            runledger.record_run(ledger_path or out / "run_log.jsonl", params,
                                 status="failed", reason="run_metadata.json missing")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cfg = metadata.get("prefetch_config") or {}
        cache_pages = int(cfg.get("cache_pages", 32))
        delay = float(cfg.get("miss_delay_ms", 0.0) or 0.0)
        model = str(cfg.get("storage_model", "shared"))
        params.update({"cache_pages": cache_pages, "delay_ms_requested": delay,
                       "storage_model": model, "strategy": cfg.get("strategy", "none")})
        reference_dir = None
        # the demand reference companion of the calibration cell
        for candidate in [run_path.parent / f"ref-c{cache_pages}" / "demand_reference.json",
                          run_path / "demand_reference.json"]:
            if candidate.exists():
                reference_dir = candidate
                break
        if reference_dir is None:
            rows.append({**params, "status": "needs_reference",
                         "expected_reference": str(run_path.parent / f"ref-c{cache_pages}" / "demand_reference.json"),
                         "cause": "demand reference absent; rerun the reference generation on the host"})
            runledger.record_run(ledger_path or out / "run_log.jsonl", params,
                                 status="interrupted",
                                 reason="demand reference missing; diagnosis blocked")
            continue
        reference = load_reference(reference_dir)
        sim = simulate(reference, strategy="none", cache_pages=cache_pages, delay_ms=delay,
                       costs=costs, storage_model=model)
        measured = float(metadata["query_latency_ms"]["total"])
        error_pct = 100.0 * (sim.total_ms - measured) / measured
        components = {
            "cpu_simulated_ms": sim.stats.get("cpu_simulated_ms"),
            "device_busy_ms": sim.stats.get("device_busy_ms"),
            "device_demand_queued_ms": sim.stats.get("device_demand_queued_ms"),
            "sleep_overshoot_ms_at_this_delay": calibration.get("sleep_measured_ms", {}).get(str(delay)),
            "measured_total_ms": round(measured, 4),
            "simulated_total_ms": sim.total_ms,
        }
        cause = _dominant_error_cause(components, measured)
        rows.append({**params, "status": "diagnosed", "error_pct": round(error_pct, 4),
                     "within_15pct": abs(error_pct) <= 15.0,
                     "dominant_error_cause": cause, "components": components})
        runledger.record_run(ledger_path or out / "run_log.jsonl", params, status="completed",
                             metrics={"error_pct": round(error_pct, 4),
                                      "dominant_error_cause": cause})
    flagged = [r for r in rows if r.get("status") == "diagnosed" and not r.get("within_15pct")]
    report = {
        "schema_version": "v2.1",
        "target_pct": 15.0,
        "cells": rows,
        "cells_missing_target": flagged,
        "note": ("Declared simulator-validation configurations that missed 15% are diagnosed "
                 "above with configuration, error value and dominant cause. Runs whose demand "
                 "reference is absent are marked needs_reference and must be completed on the "
                 "experiment host; none are silently omitted."),
    }
    (out / "simulator_cell_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _dominant_error_cause(components: dict[str, Any], measured_ms: float) -> str:
    """Rank known error causes by their modeled contribution share."""
    reasons: list[tuple[float, str]] = []
    cpu = components.get("cpu_simulated_ms") or 0.0
    device = components.get("device_busy_ms") or 0.0
    queue = components.get("device_demand_queued_ms") or 0.0
    total_model = cpu + device + queue
    if total_model <= 0:
        return "unmodeled components dominate (reference or device absent)"
    if queue / max(1.0, total_model) > 0.30:
        reasons.append((queue / total_model, "device queueing dominates the modeled time"))
    if cpu / max(1.0, total_model) > 0.60:
        reasons.append((cpu / total_model, "calibrated CPU cost per read dominates; residual timer/CPU noise scales with it"))
    if device / max(1.0, total_model) > 0.60:
        reasons.append((device / total_model, "device service holds dominate; sleep overshoot is the first suspect"))
    if not reasons:
        reasons.append((1.0, "mixed causes: no single component exceeds 60% of modeled time"))
    return sorted(reasons, reverse=True)[0][1]
