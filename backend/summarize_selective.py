"""Summarize matched held-out latency, predictive accuracy and worker costs."""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
import csv
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pagewise.statistics import paired_cluster_bootstrap_ci
from selective_adaptive import choose_action
from selective_deadline import deadline_feasible
from queue_experiments import ABLATIONS, BASELINES, OUTPUT, SCENARIOS, TEST, digest, source_digests, write
from calibrate_selective import offline_rows

POLICIES = BASELINES + ABLATIONS


def trial(scenario, seed, policy):
    return OUTPUT / "heldout" / scenario / f"seed-{seed}" / policy


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def comparison(sessions):
    if not sessions:
        return None
    return paired_cluster_bootstrap_ci(sessions, alternative="less", iterations=5000, seed=617)


def runtime_prediction_accuracy(scenario, policy):
    """Evaluate recorded scoring jobs against their own subsequent demand.

    New policies record query positions explicitly. Legacy logs only record
    history, so ambiguous history alignments are excluded and counted.
    """
    queries = defaultdict(lambda: [0, 0, 0, 0])
    families = defaultdict(lambda: [0, 0, 0, 0])
    audit = defaultdict(int)
    for seed in TEST:
        root = trial(scenario, seed, policy)
        reference = load(root / "demand_reference.json")
        reference_pages = {qid: [int(entry[0]) for entry in seq] for qid, seq in reference.items()}
        workload = {r["query_id"]: r for r in (json.loads(line) for line in
                    (root / "workload.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
        histories = {}
        cursors = defaultdict(int)
        if policy in BASELINES:
            for qid, pages in reference_pages.items():
                lookup = defaultdict(list)
                for pos in range(1, len(pages)):
                    lookup[tuple(pages[max(0, pos-8):pos])].append(pos)
                histories[qid] = lookup
        path = root / "predictions.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                ranking = [int(r["page"]) for r in event.get("ranked_candidates", [])]
                if not ranking:
                    audit["jobs_without_recorded_ranking"] += 1
                    continue
                qid, history = str(event.get("query_id")), tuple(int(x) for x in event.get("history", []))
                pages = reference_pages.get(qid)
                if not pages:
                    audit["missing_query_reference"] += 1
                    continue
                pos = event.get("query_page_position")
                if pos is None:
                    options = histories.get(qid, {}).get(history, [])
                    options = options[bisect_right(options, cursors[qid]):]
                    if len(options) != 1:
                        audit["ambiguous_legacy_history" if options else "unmatched_legacy_history"] += 1
                        continue
                    pos = options[0]
                    cursors[qid] = pos
                if not isinstance(pos, int) or not 1 <= pos < len(pages) or tuple(pages[max(0, pos-len(history)):pos]) != history:
                    audit["unaligned_or_last_position"] += 1
                    continue
                target = pages[pos]
                qkey = (seed, qid)
                family = workload.get(qid, {}).get("family", "unknown")
                q, g = queries[qkey], families[family]
                for v in (q, g):
                    v[0] += 1
                    v[1] += int(ranking[0] == target)
                    v[2] += int(target in ranking[:5])
                    v[3] += int(target in ranking)
    total = sum(v[0] for v in queries.values())
    return {"scored_aligned_jobs": total, "queries": len(queries), "alignment": dict(audit),
            "top1_macro_pct": statistics.mean(100*v[1]/v[0] for v in queries.values()) if queries else None,
            "top1_micro_pct": 100*sum(v[1] for v in queries.values())/total if total else None,
            "top5_micro_pct": 100*sum(v[2] for v in queries.values())/total if total else None,
            "target_in_logged_candidates_pct": 100*sum(v[3] for v in queries.values())/total if total else None,
            "by_family": {family: {"scored_aligned_jobs": v[0], "top1_micro_pct": 100*v[1]/v[0], "top5_micro_pct": 100*v[2]/v[0]}
                          for family, v in families.items()},
            "scope": "Only recorded scored jobs aligned to this policy's own actual demand stream. Unscored/rejected requests and ambiguous legacy histories are excluded; candidate logs are truncated, so target-in-logged-candidates is not full candidate recall. Different policies score different jobs."}


def runtime_summary(scenario):
    values = {p: [] for p in POLICIES}
    families = {p: defaultdict(list) for p in POLICIES}
    counters = {p: defaultdict(float) for p in POLICIES}
    resources = {p: [] for p in POLICIES}
    pairs = {p: defaultdict(list) for p in POLICIES if p != "none"}
    family_pairs = {p: defaultdict(lambda: defaultdict(list)) for p in POLICIES if p != "none"}
    workload_hashes = {}
    for seed in TEST:
        current = {}
        for policy in POLICIES:
            root = trial(scenario, seed, policy)
            fingerprint = digest(root / "workload.jsonl")
            if seed in workload_hashes and workload_hashes[seed] != fingerprint:
                raise ValueError(f"Mismatched workloads for seed {seed}")
            workload_hashes[seed] = fingerprint
            rows = read_csv(root / "query_results.csv")
            if len(rows) != 30 or len({r["query_id"] for r in rows}) != 30:
                raise ValueError(f"Incomplete or duplicate query timings: {root}")
            current[policy] = {r["query_id"]: (float(r["elapsed_ms"]), r["workload_family"]) for r in rows}
            stats, resource = load(root / "runtime_stats.json"), load(root / "resource_usage.json")
            if stats.get("prefetch_read_errors") or not stats.get("worker_shutdown_clean"):
                raise ValueError(f"Unclean worker: {root}")
            for name in ["prediction_observations", "prediction_requests", "queued_prediction_jobs", "completed_jobs",
                         "dropped_jobs", "coalesced_jobs", "stale_jobs_detected", "stale_jobs_dropped",
                         "inference_ms_total", "queue_wait_ms_total", "prefetch_read_ms_total", "pages_prefetched",
                         "useful_prefetches", "total_wasted_prefetches", "unused_prefetches_still_resident",
                         "demand_misses_while_prefetch_in_flight", "prefetch_read_errors", "policy_markov2", "policy_markov1",
                         "policy_neural", "policy_skipped", "policy_deadline_dropped", "policy_query_limit_dropped",
                         "policy_query_completed_dropped", "policy_duplicate_candidates", "policy_cached_candidates",
                         "policy_pending_candidates", "policy_device_busy_dropped", "policy_pages_read", "policy_late_read_discarded",
                         "policy_net_decisions", "policy_estimated_benefit_ms_total", "policy_estimated_cost_ms_total",
                         "policy_estimated_net_ms_total"]:
                counters[policy][name] += float(stats.get(name, 0))
            resources[policy].append(resource)
            for ms, family in current[policy].values():
                values[policy].append(ms)
                families[policy][family].append(ms)
        keys = current["none"].keys()
        if any(current[p].keys() != keys for p in POLICIES):
            raise ValueError(f"Missing matched query ids for {seed}")
        for policy in pairs:
            for qid, (base, family) in current["none"].items():
                ms, other_family = current[policy][qid]
                if family != other_family:
                    raise ValueError("Mismatched workload family")
                delta = ms - base
                pairs[policy][seed].append(delta)
                family_pairs[policy][family][seed].append(delta)
    summary = {}
    for policy in POLICIES:
        v = values[policy]
        c = dict(counters[policy])
        total = c["pages_prefetched"]
        c["useful_insertions_pct"] = 100 * c["useful_prefetches"] / total if total else None
        c["wasted_insertions_pct"] = 100 * c["total_wasted_prefetches"] / total if total else None
        completed = c["completed_jobs"]
        c["inference_ms_per_completed_job"] = c["inference_ms_total"] / completed if completed else None
        c["queue_wait_ms_per_completed_job"] = c["queue_wait_ms_total"] / completed if completed else None
        peaks = [r["peak_process_rss_bytes"] for r in resources[policy] if r["peak_process_rss_bytes"] is not None]
        summary[policy] = {"queries": len(v), "mean_ms": statistics.mean(v), "p95_ms": float(np.percentile(v, 95)),
            "cpu_seconds_total": sum(r["cpu_seconds"] for r in resources[policy]),
            "mean_peak_process_rss_bytes": statistics.mean(peaks) if peaks else None,
            "max_peak_process_rss_bytes": max(peaks) if peaks else None,
            "counters": c,
            "by_family": {f: {"queries": len(xs), "mean_ms": statistics.mean(xs), "p95_ms": float(np.percentile(xs, 95))}
                          for f, xs in families[policy].items()}}
        summary[policy]["runtime_prediction_accuracy"] = runtime_prediction_accuracy(scenario, policy)
        if policy != "none":
            summary[policy]["vs_none"] = comparison(pairs[policy])
            summary[policy]["per_seed_mean_delta_ms"] = {str(seed): statistics.mean(ds) for seed, ds in pairs[policy].items()}
            summary[policy]["seeds_with_lower_mean_latency"] = sum(statistics.mean(ds) < 0 for ds in pairs[policy].values())
            for family, sessions in family_pairs[policy].items():
                summary[policy]["by_family"][family]["vs_none"] = comparison(sessions)
    # Direct paired comparisons to both existing predictors are also required.
    for policy in ABLATIONS:
        baselines = ["markov2", "model_ranker"]
        if policy != "context":
            baselines.append("context")
        if policy == "combined":
            baselines.extend(["deadline", "adaptive"])
        for baseline in baselines:
            sessions = {}
            for seed in TEST:
                a = {r["query_id"]: float(r["elapsed_ms"]) for r in read_csv(trial(scenario, seed, policy) / "query_results.csv")}
                b = {r["query_id"]: float(r["elapsed_ms"]) for r in read_csv(trial(scenario, seed, baseline) / "query_results.csv")}
                sessions[seed] = [a[q] - b[q] for q in b]
            summary[policy][f"vs_{baseline}"] = comparison(sessions)
    return {"policies": summary, "workloads_matched_by_sha256": True,
            "latency_scope": "actual foreground query latency; lower is better; OS cache not reset",
            "completion_before_demand_scope": "useful_prefetches divided by successfully inserted prefetch pages; unused pages remain in denominator, and demand misses while a prefetch was in flight are reported separately",
            "resource_scope": "process CPU excludes dependency import; RSS is process peak working set, one fresh process per trial"}


def offline_summary(path, frozen, scenario):
    config = dict(frozen)
    config.update(config["by_scenario"][scenario])
    # Model decisions are approximated on the common reference trace. Queue and
    # cache pressure are zero here; actual scheduling outcomes are reported
    # separately from runtime logs and may differ.
    index_yes = SimpleNamespace(second_order={"1:2": [3]}, successors={2: [3]})
    index_no = SimpleNamespace(second_order={}, successors={2: [3]})
    index_empty = SimpleNamespace(second_order={}, successors={})
    agg = {p: defaultdict(lambda: [0, 0, 0, 0, 0]) for p in POLICIES if p != "none"}
    by_family = {p: defaultdict(lambda: [0, 0, 0, 0]) for p in POLICIES if p != "none"}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            exact, avail, family = bool(int(row["exact_context"])), bool(int(row["markov1_available"])), row["family"]
            idx = index_yes if exact else index_no if avail else index_empty
            history = [1, 2] if exact else [2]
            actions = {"model_ranker": "neural", "markov2": "markov2"}
            for policy in ABLATIONS:
                action, _, _ = choose_action(idx, history, family, policy, config, queue_fraction=0, cache_fraction=0)
                if policy in {"deadline", "combined"} and action != "skip":
                    costs = config.get("runtime_cost_ms", {}).get(action, {})
                    remaining = float(costs.get("score_ms", 0)) + float(costs.get("read_ms", 0))
                    if not deadline_feasible(0, 0, config.get("deadlines_ms", {}).get(family), remaining):
                        action = "skip"
                actions[policy] = action
            for policy, action in actions.items():
                query = (int(row["seed"]), row["query_id"], family)
                v = agg[policy][query]
                g = by_family[policy][family]
                v[0] += 1
                g[0] += 1
                if action == "skip":
                    v[4] += 1
                    continue
                v[1] += int(row[action + "_top1"])
                v[2] += int(row[action + "_top5"])
                candidate = "candidate_hit" if action == "neural" else action + "_candidate_hit"
                v[3] += int(row[candidate])
                g[1] += int(row[action + "_top1"])
                g[2] += int(row[action + "_top5"])
                g[3] += int(row[candidate])
    summary = {}
    for policy, qs in agg.items():
        total = sum(v[0] for v in qs.values())
        qvals = list(qs.values())
        family_rows = {}
        for family, v in by_family[policy].items():
            sessions = defaultdict(list)
            for (seed, _, qfamily), qv in qs.items():
                if qfamily == family:
                    sessions[seed].append(100 * qv[1] / qv[0])
            interval = paired_cluster_bootstrap_ci(dict(sessions), iterations=2000, seed=817) if len(sessions) >= 2 else None
            family_rows[family] = {"transitions": v[0], "queries": sum(len(s) for s in sessions.values()),
                "top1_macro_pct": statistics.mean(x for values in sessions.values() for x in values),
                "top1_macro_ci95": interval,
                "top1_micro_pct": 100*v[1]/v[0], "top5_micro_pct": 100*v[2]/v[0], "candidate_recall_pct": 100*v[3]/v[0]}
        summary[policy] = {"queries": len(qvals), "transitions": total,
            "top1_macro_pct": statistics.mean(100 * v[1] / v[0] for v in qvals),
            "top1_micro_pct": 100 * sum(v[1] for v in qvals) / total,
            "top5_micro_pct": 100 * sum(v[2] for v in qvals) / total,
            "candidate_recall_pct": 100 * sum(v[3] for v in qvals) / total,
            "skipped_transitions": sum(v[4] for v in qvals),
            "by_family": family_rows}
    for policy in ABLATIONS:
        summary[policy]["paired_family_accuracy_vs_baselines"] = {}
        for baseline in ("markov2", "model_ranker"):
            family_sessions = defaultdict(lambda: defaultdict(list))
            for query, v in agg[policy].items():
                reference = agg[baseline][query]
                if v[0] != reference[0]:
                    raise ValueError("Offline policies do not share transition counts")
                seed, _, family = query
                family_sessions[family][seed].append(100 * (v[1] - reference[1]) / v[0])
            summary[policy]["paired_family_accuracy_vs_baselines"][baseline] = {
                family: paired_cluster_bootstrap_ci(dict(sessions), iterations=2000, seed=919)
                if len(sessions) >= 2 else None for family, sessions in family_sessions.items()}
    return {"policies": summary, "scope": "offline no-prefetch reference trace; all eligible transitions; hypothetical decisions with zero queue and cache pressure, including skipped transitions as errors. This is not measured runtime job accuracy. Baseline predictor scores do not include its runtime confidence/stride filtering."}


def summarize():
    frozen = load(OUTPUT / "policy-frozen.json")
    if frozen["source_sha256"] != source_digests():
        raise ValueError("Policy source changed after freeze")
    summary_path = OUTPUT / "heldout-summary.json"
    if summary_path.exists():
        raise FileExistsError("Preserve previous held-out summary")
    # Reuse one scoring pass only if the measured page/miss streams match.
    matched = all(load(trial("zero", seed, "none") / "demand_reference.json") ==
                  load(trial("controlled-delay", seed, "none") / "demand_reference.json") for seed in TEST)
    zero_path = OUTPUT / "heldout-offline-zero.csv"
    offline_rows(TEST, "heldout", "zero", zero_path)
    delay_path = zero_path
    if not matched:
        delay_path = OUTPUT / "heldout-offline-controlled-delay.csv"
        offline_rows(TEST, "heldout", "controlled-delay", delay_path)
    report = {"policy_sha256": digest(OUTPUT / "policy-frozen.json"), "seeds": TEST,
              "same_reference_streams_across_scenarios": matched, "scenarios": {}}
    for scenario in SCENARIOS:
        report["scenarios"][scenario] = {"runtime": runtime_summary(scenario),
            "offline_accuracy": offline_summary(zero_path if scenario == "zero" else delay_path, frozen, scenario)}
    write(summary_path, report)
    for scenario, result in report["scenarios"].items():
        print(scenario, flush=True)
        for policy, r in result["runtime"]["policies"].items():
            pair = r.get("vs_none") or {}
            print(f"  {policy}: mean {r['mean_ms']:.3f}ms, p95 {r['p95_ms']:.3f}ms, "
                  f"delta {pair.get('mean')}ms CI [{pair.get('ci95_low')}, {pair.get('ci95_high')}]", flush=True)


if __name__ == "__main__":
    summarize()
