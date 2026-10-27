"""Measure development-only fallback evidence and prefetch-cost estimates."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from pagewise.dataset import build_sequences, load_jsonl
from pagewise.models import CandidateIndex
from pagewise.ranking import NeuralPredictor, build_groups, logits
from pagewise.statistics import paired_cluster_bootstrap_ci

from queue_experiments import DEV, FROZEN, OUTPUT, SCENARIOS, digest, write


def reference_sequences(directory):
    return build_sequences(directory / "trace.jsonl", FROZEN / "params/page_map.json",
                           directory / "offline-sequences.jsonl")


def offline_rows(seeds, phase, scenario, output_path):
    """Score all true next-page transitions using frozen predictor and index."""
    torch.set_num_threads(1)
    index = CandidateIndex.from_json(json.loads((FROZEN / "params/candidate_index.json").read_text(encoding="utf-8")))
    metadata = json.loads((FROZEN / "params/page_map.json").read_text(encoding="utf-8"))
    predictor = NeuralPredictor(FROZEN / "params/model_ranker.pt", index, metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "query_id", "family", "position", "target", "exact_context", "candidate_hit",
              "markov1_available", "markov1_candidate_hit", "markov2_candidate_hit",
              "neural_top1", "neural_top5", "markov1_top1", "markov1_top5", "markov2_top1", "markov2_top5"]
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            directory = OUTPUT / phase / scenario / f"seed-{seed}" / "none"
            sequences = reference_sequences(directory)
            cases = {str(row["query_id"]): row for row in load_jsonl(directory / "workload.jsonl")}
            for seq in sequences:
                if len(seq["pages"]) < 2:
                    continue
                data = build_groups([seq], index, predictor.page_map, predictor.counts, predictor.config,
                                    cases, predictor.parameter_scales, max_per_query=None)
                ordered = []
                with torch.inference_mode():
                    for start in range(0, len(data["target"]), 512):
                        x = torch.from_numpy(data["x"][start:start+512])
                        prior = torch.from_numpy(data["prior"][start:start+512])
                        valid = torch.from_numpy(data["pages"][start:start+512] != -1)
                        ordered.append(torch.argsort(logits(predictor.model, x, prior, valid, predictor.config),
                                                    dim=1, descending=True, stable=True).numpy())
                order = np.concatenate(ordered)
                for i, target in enumerate(data["target"]):
                    p = i + 1
                    history = seq["pages"][max(0, p-8):p]
                    key = f"{history[-2]}:{history[-1]}" if len(history) >= 2 else ""
                    target_page = int(data["target_page"][i])
                    first_candidates = index.successors.get(history[-1], [])
                    second_candidates = index.second_order.get(key, []) or first_candidates
                    neural_top1 = bool(target >= 0 and order[i, 0] == target)
                    neural_top5 = bool(target >= 0 and target in order[i, :5])
                    m1 = data["markov1_top5"][i]
                    m2 = data["markov2_top5"][i]
                    row = {"seed": seed, "query_id": seq["query_id"], "family": seq["workload_family"],
                           "position": p, "target": target_page, "exact_context": int(bool(index.second_order.get(key))),
                           "markov1_available": int(bool(first_candidates)),
                           "markov1_candidate_hit": int(target_page in first_candidates),
                           "markov2_candidate_hit": int(target_page in second_candidates),
                           "candidate_hit": int(target >= 0), "neural_top1": int(neural_top1), "neural_top5": int(neural_top5),
                           "markov1_top1": int(m1[0] == target_page), "markov1_top5": int(target_page in m1),
                           "markov2_top1": int(m2[0] == target_page), "markov2_top5": int(target_page in m2)}
                    writer.writerow(row)
                    count += 1
            print(f"Offline scoring {phase}/{scenario}/seed-{seed}: {len(sequences)} queries", flush=True)
    return count


def demand_observations(scenario):
    gaps = defaultdict(list)
    miss_costs = defaultdict(list)
    for seed in DEV:
        trace = OUTPUT / "baseline" / scenario / f"seed-{seed}" / "none/trace.jsonl"
        prev = {}
        for row in load_jsonl(trace):
            qid, family = row.get("query_id"), row.get("workload_family") or "unknown"
            if not qid or row.get("file_type") != "main_db" or row.get("result") != "success":
                continue
            end = int(row["monotonic_time_ns"])
            start = end - int(float(row.get("duration_us", 0)) * 1000)
            if qid in prev:
                gap = (start - prev[qid]) / 1_000_000
                if gap > 0:
                    gaps[family].append(gap)
            prev[qid] = end
            if row.get("cache_status") == "miss":
                first, last = row.get("first_page"), row.get("last_page")
                pages = max(1, int(last) - int(first) + 1) if first is not None and last is not None else 1
                miss_costs[family].append(float(row.get("duration_us", 0)) / (1000 * pages))
    return gaps, miss_costs


def runtime_costs(scenario, policy):
    stats = [json.loads((OUTPUT / "baseline" / scenario / f"seed-{s}" / policy / "runtime_stats.json").read_text()) for s in DEV]
    scored = sum(max(0, s["completed_jobs"] - s["stale_jobs_dropped"]) for s in stats)
    queued = sum(s["completed_jobs"] for s in stats)
    pages = sum(s["pages_prefetched"] for s in stats)
    return {"score_ms": sum(s["inference_ms_total"] for s in stats) / scored if scored else 0.0,
            "queue_wait_ms": sum(s["queue_wait_ms_total"] for s in stats) / queued if queued else 0.0,
            "read_ms": sum(s["prefetch_read_ms_total"] for s in stats) / pages if pages else 0.0,
            "usefulness_given_prediction": sum(s["useful_prefetches"] for s in stats) / pages if pages else 0.0,
            "scored_jobs": scored, "prefetched_pages": pages}


def calibrate():
    output = OUTPUT / "development-offline-transitions.csv"
    if output.exists() or (OUTPUT / "policy-development.json").exists():
        raise FileExistsError("Preserve previous development calibration")
    count = offline_rows(DEV, "baseline", "zero", output)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    paired = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    total_exact = 0
    with output.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            family, group = r["family"], "exact" if int(r["exact_context"]) else "fallback"
            total_exact += int(r["exact_context"])
            for policy in ("neural", "markov1", "markov2"):
                cell = groups[family][group][policy]
                cell[0] += int(r[policy + "_top1"])
                cell[1] += 1
            if group == "fallback":
                cell = paired[family][(int(r["seed"]), r["query_id"])]
                cell[0] += int(r["neural_top1"])
                cell[1] += int(r["markov1_top1"])
                cell[2] += 1
    family_evidence = {}
    enabled = []
    for family, queries in paired.items():
        sessions = defaultdict(list)
        for (seed, _), (neural, markov, total) in queries.items():
            sessions[seed].append(100 * (neural - markov) / total)
        ci = paired_cluster_bootstrap_ci(dict(sessions), alternative="greater", iterations=2000, seed=201)
        transitions = sum(v[2] for v in queries.values())
        allowed = len(queries) >= 10 and transitions >= 40 and len(sessions) >= 2 and ci["ci95_low"] > 0 and family != "recursive_traversal"
        family_evidence[family] = {"fallback_transitions": transitions, "queries": len(queries), "seeds": len(sessions),
                                   "neural_minus_markov1_macro_pp": ci["mean"], "ci95_low_pp": ci["ci95_low"],
                                   "ci95_high_pp": ci["ci95_high"], "allowed": allowed,
                                   "decision": "exploratory development evidence; recursive fallback disabled by default"}
        if allowed:
            enabled.append(family)
    zero_gaps, _ = demand_observations("zero")
    deadlines = {family: statistics.median(values) for family, values in zero_gaps.items() if values}
    scenario_configs = {}
    for scenario in SCENARIOS:
        _, demand = demand_observations(scenario)
        markov = runtime_costs(scenario, "markov2")
        neural = runtime_costs(scenario, "model_ranker")
        costs = {"markov2": {"score_ms": markov["score_ms"], "read_ms": markov["read_ms"]},
                 "markov1": {"score_ms": markov["score_ms"], "read_ms": markov["read_ms"]},
                 "neural": {"score_ms": neural["score_ms"], "read_ms": neural["read_ms"]}}
        estimates = {}
        for family, contexts in groups.items():
            estimates[family] = {}
            for group, policies in contexts.items():
                estimates[family][group] = {}
                for action in ("neural", "markov1", "markov2"):
                    hits, n = policies[action]
                    c = neural if action == "neural" else markov
                    estimates[family][group][action] = {"prediction_accuracy": hits / n,
                        "usefulness_given_prediction": c["usefulness_given_prediction"],
                        "demand_miss_ms": statistics.mean(demand[family]) if demand[family] else 0.0,
                        "score_ms": neural["score_ms"], "markov_score_ms": markov["score_ms"],
                        "read_ms": c["read_ms"], "queue_wait_ms": c["queue_wait_ms"],
                        "deadline_ms": deadlines.get(family, 0.0), "sample_transitions": n}
        scenario_configs[scenario] = {"runtime_cost_ms": costs, "estimates": estimates,
                                      "baseline_cost_measurements": {"markov2": markov, "neural": neural}}
    result = {"checkpoint_sha256": digest(FROZEN / "params/model_ranker.pt"), "development_seeds": DEV,
              "neural_fallback_families": sorted(enabled), "allow_recursive_neural": False, "max_jobs_per_query": 8,
              "score_threshold": None, "score_threshold_note": "No confidence threshold: conditional candidate scores were not calibrated against usefulness.",
              "deadlines_ms": deadlines, "by_scenario": scenario_configs, "fallback_evidence": family_evidence,
              "exact_context_fraction": total_exact / count, "transitions": count,
              "deadline_note": "Median observed time from end of one no-prefetch VFS read to start of next within query, zero-delay development only; estimator, not a real demand deadline.",
              "net_note": "Heuristic multiplies development top-1 rate, observed useful/prefetched fraction and estimated readiness; it subtracts score, queue and pressure-weighted read cost. Markov-1 uses Markov-2 cost/usefulness as a proxy. Zero queue/cache pressure is not assumed at runtime.",
              "source": "only development seeds 1101/1102, full no-prefetch transitions; policy unchanged using held-out results"}
    write(OUTPUT / "policy-development.json", result)
    print(json.dumps({"transitions": count, "exact_context_fraction": result["exact_context_fraction"],
                      "neural_fallback_families": enabled, "fallback_evidence": family_evidence,
                      "deadlines_ms": deadlines}, indent=2), flush=True)


if __name__ == "__main__":
    calibrate()
