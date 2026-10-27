"""Generate a compact, auditable report from completed selective-policy runs."""
import json
import math
import shutil
from pathlib import Path

from queue_experiments import OUTPUT, ROOT, ABLATIONS, digest


def f(value, digits=3):
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def ci(value):
    return "n/a" if not value else f"[{f(value['ci95_low'])}, {f(value['ci95_high'])}]"


def verdict(value):
    if not value or value['ci95_low'] is None:
        return "inconclusive"
    if value['ci95_high'] < 0:
        return "lower latency supported by this interval"
    if value['ci95_low'] > 0:
        return "higher latency supported by this interval"
    return "interval includes zero; no supported change"


def main():
    report = json.loads((OUTPUT / "heldout-summary-detailed.json").read_text(encoding="utf-8"))
    policy = json.loads((OUTPUT / "policy-frozen.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    assert report['policy_sha256'] == digest(OUTPUT / 'policy-frozen.json')
    body = ["# Selective Pagewise prefetching: measured results", "",
        "This is a new study on the existing synthetic 8.62 GB SQLite snapshot with fresh workload seeds. It is separate from the original frozen study. The earlier queue-optimization branch, H1 audit and related artifacts were unavailable, so the figures quoted in the request were not verified or reused as evidence.", "",
        "## Protocol", "",
        f"Development seeds: {protocol['development_seeds']}; held-out seeds: {report['seeds']}; {protocol['queries_per_seed']} queries per seed/policy/scenario. All policies use matched query parameters, 512 Pagewise cache pages, a 4096 KiB SQLite pager, one worker, diagnostic logging and randomized policy order. SQLite's pager is released between queries; the OS cache is not reset.", "",
        "Two conditions are shown: zero injected delay and a shared, single-channel model with 0.2 ms requested service delay. The delay is artificial and operating-system scheduling may make actual waits longer. It is not a physical-device benchmark. Demand priority is best-effort: a speculative operation already in progress and actual OS I/O cannot be preempted.", "",
        f"The selected frozen checkpoint SHA256 is `{policy['checkpoint_sha256']}`; the policy was frozen as `{report['policy_sha256']}` before held-out execution. The original predictor source and saved research data remain untouched. New policies are an opt-in extension outside the hashed frozen predictor directories.", "",
        "The no-prefetch, native Markov-2 and existing neural–Markov ranker are baselines. The context policy uses Markov-2 on known contexts and an evidence-gated fallback. Deadline and adaptive are individual additions to that context policy; combined enables both. This design keeps exact Markov-2 behavior or skips prefetching when cost/deadline rules reject the job.", "",
        "## Development calibration", "",
        f"Exact Markov-2 contexts occurred in {100*policy['exact_context_fraction']:.2f}% of {policy['transitions']:,} development transitions. This is a new result, not a reconstruction of the missing H1 audit. Neural fallback was enabled for: {', '.join(policy['neural_fallback_families']) if policy['neural_fallback_families'] else 'no families'}. No confidence threshold was added because candidate scores were not calibrated against usefulness.", "",
        "A family required at least 40 fallback transitions, 10 queries, both development seeds, and a positive paired 95% query-bootstrap lower bound against Markov-1. Recursive traversal remains disabled by default. These two-seed intervals are exploratory; indexed lookup remains especially uncertain.", "",
        "| Family | Fallback transitions | Queries | Neural − Markov-1 macro Top-1 (pp) | 95% interval | Neural enabled? |",
        "|---|---:|---:|---:|---:|---|"]
    for family, row in sorted(policy['fallback_evidence'].items()):
        body.append(f"| {family} | {row['fallback_transitions']} | {row['queries']} | {f(row['neural_minus_markov1_macro_pp'],2)} | [{f(row['ci95_low_pp'],2)}, {f(row['ci95_high_pp'],2)}] | {'yes' if row['allowed'] else 'no'} |")
    body += ["", "Deadlines are the median observed interval between no-prefetch VFS reads on zero-delay development traces. This is a conservative estimate of time until the next read, not a guaranteed future demand deadline. The cost controller uses development-estimated prediction accuracy, usefulness, miss cost, inference time, queue wait, read time and current queue/cache pressure; Markov-1 cost uses the Markov-2 measurement as a proxy. Unavailable estimates cause skipping. It does not use future accesses.", "",
             "An early development attempt was retained but excluded because instrumentation changed while it was running. All reported development ablations were rerun with the same fixed code before the policy freeze. No held-out outcome was used to tune the policy.", "",
             "## Held-out runtime", "",
             "Every row contains 150 matched query executions. Delta is policy minus no-prefetch mean query latency; negative favors the policy. The 95% paired interval resamples workload seeds and queries within seeds. Intervals are descriptive with only five workload seeds; this is not a claim of physical I/O improvement."]
    for scenario, result in report['scenarios'].items():
        runtime = result['runtime']['policies']
        body += ["", f"### {scenario}", "",
                 "| Policy | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Peak RSS max MB | CPU total s | Prefetched | Useful | Wasted |",
                 "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for name, row in runtime.items():
            pair = row.get('vs_none');c = row['counters']
            body.append(f"| {name} | {f(row['mean_ms'])} | {f(row['p95_ms'])} | {f(pair['mean']) if pair else 'baseline'} | {ci(pair)} | {f(row['max_peak_process_rss_bytes']/1_000_000 if row['max_peak_process_rss_bytes'] else None,1)} | {f(row['cpu_seconds_total'],2)} | {int(c['pages_prefetched'])} | {int(c['useful_prefetches'])} | {int(c['total_wasted_prefetches'])} |")
        body += ["", "Peak RSS is the largest per-process peak working set across five runs. CPU is total process CPU across those five runs, excluding dependency imports. Wasted means unused evicted plus unused resident prefetched insertions at run end.", "",
                 "| New policy | Verdict vs none | Seeds with lower mean | Delta vs Markov-2 ms (95% CI) | Delta vs existing neural hybrid ms (95% CI) |",
                 "|---|---|---:|---|---|"]
        for name in ABLATIONS:
            row = runtime[name];m=row['vs_markov2'];h=row['vs_model_ranker']
            body.append(f"| {name} | {verdict(row['vs_none'])} | {row['seeds_with_lower_mean_latency']}/5 | {f(m['mean'])} {ci(m)} | {f(h['mean'])} {ci(h)} |")
        body += ["", "| Added component | Delta vs context ms | 95% CI ms | Interpretation |", "|---|---:|---|---|"]
        for name in ('deadline', 'adaptive', 'combined'):
            row=runtime[name]['vs_context']
            body.append(f"| {name} | {f(row['mean'])} | {ci(row)} | {verdict(row)} |")
        body += ["", "Completion-before-demand is observable as useful prefetched insertions: useful / successfully inserted pages. Unused pages remain in the denominator. In-flight demand misses are separately counted; this is not an exact timing rate over every candidate.", "",
                 "| New policy | Queue wait ms/job | Inference ms/job | Timely useful / inserted | Demand misses while in flight | Deadline drops | Query-ended drops | Queue-full drops | Coalesced jobs |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ABLATIONS:
            c=runtime[name]['counters']
            useful = f"{f(c['useful_insertions_pct'],2)}%" if c['useful_insertions_pct'] is not None else 'n/a'
            body.append(f"| {name} | {f(c['queue_wait_ms_per_completed_job'],4)} | {f(c['inference_ms_per_completed_job'],4)} | {useful} | {int(c['demand_misses_while_prefetch_in_flight'])} | {int(c['policy_deadline_dropped'])} | {int(c['policy_query_completed_dropped'])} | {int(c['dropped_jobs'])} | {int(c['coalesced_jobs'])} |")
        body += ["", "### Query-family runtime: combined policy", "",
                 "| Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |",
                 "|---|---:|---:|---:|---:|---|---:|"]
        for family, row in sorted(runtime['combined']['by_family'].items()):
            pair=row.get('vs_none')
            body.append(f"| {family} | {row['queries']} | {f(row['mean_ms'])} | {f(row['p95_ms'])} | {f(pair['mean']) if pair else 'n/a'} | {ci(pair)} | {pair['seeds'] if pair else 0} |")
        accuracy = result['offline_accuracy']['policies']
        body += ["", "### Offline prediction accuracy", "",
                 "Scored on all next-page transitions from the shared no-prefetch reference. Selective decisions use the frozen development policy with zero queue/cache pressure; deadlines use estimated remaining work, and skips count as errors. This is an idealized offline approximation, not the behavior of every runtime job. The existing neural baseline is scored without its runtime confidence/stride filter. Candidate recall is target inclusion in the available full candidate set.", "",
                 "| Policy | Queries | Transitions | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Candidate recall | Skips |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in accuracy.items():
            body.append(f"| {name} | {row['queries']} | {row['transitions']} | {f(row['top1_macro_pct'],2)}% | {f(row['top1_micro_pct'],2)}% | {f(row['top5_micro_pct'],2)}% | {f(row['candidate_recall_pct'],2)}% | {row['skipped_transitions']} |")
        body += ["", "### Query-family prediction accuracy: all policies", "",
                 "Intervals below resample query accuracies within workload seeds and then resample seeds. Family intervals and paired family comparisons were added in a second analysis pass; the frozen policy and raw outcomes were unchanged.", "",
                 "| Policy | Family | Queries | Transitions | Query-average Top-1 | 95% CI | Transition Top-1 | Transition Top-5 | Candidate recall |",
                 "|---|---|---:|---:|---:|---|---:|---:|---:|"]
        for name, details in accuracy.items():
            for family, row in sorted(details['by_family'].items()):
                body.append(f"| {name} | {family} | {row['queries']} | {row['transitions']} | {f(row['top1_macro_pct'],2)}% | {ci(row['top1_macro_ci95'])} | {f(row['top1_micro_pct'],2)}% | {f(row['top5_micro_pct'],2)}% | {f(row['candidate_recall_pct'],2)}% |")
        body += ["", "### Query-family latency: all policies", "",
                 "| Policy | Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |",
                 "|---|---|---:|---:|---:|---:|---|---:|"]
        for name, details in runtime.items():
            for family, row in sorted(details['by_family'].items()):
                pair=row.get('vs_none')
                body.append(f"| {name} | {family} | {row['queries']} | {f(row['mean_ms'])} | {f(row['p95_ms'])} | {f(pair['mean']) if pair else 'baseline'} | {ci(pair)} | {pair['seeds'] if pair else 5} |")
        body += ["", "### Actual scored runtime jobs", "",
                 "| Policy | Aligned scored jobs | Queries | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Ambiguous legacy histories excluded |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in runtime.items():
            r=row['runtime_prediction_accuracy'];a=r['alignment']
            body.append(f"| {name} | {r['scored_aligned_jobs']} | {r['queries']} | {f(r['top1_macro_pct'],2)} | {f(r['top1_micro_pct'],2)} | {f(r['top5_micro_pct'],2)} | {a.get('ambiguous_legacy_history',0)} |")
        body += ["", "These are conditional on actual recorded scoring jobs that can be aligned to the same policy's observed demand stream; different policies score different jobs. Legacy job histories that match multiple positions are excluded. Logs truncate candidate lists, so their target-in-logged-candidates measure is not full candidate recall. Full counters, model actions and per-family values are in the JSON evidence."]
    zero_runtime = report['scenarios']['zero']['runtime']['policies']
    delayed_runtime = report['scenarios']['controlled-delay']['runtime']['policies']
    zero_acc = report['scenarios']['zero']['offline_accuracy']['policies']
    body += ["", "## Hypothesis verdicts", "",
             "- **Context / neural fallback:** development evidence did not authorize neural fallback in any family. The held-out context policy scored no neural jobs. On the offline common-reference comparison, its query-average Top-1 was "
             f"{f(zero_acc['context']['top1_macro_pct'],2)}%, versus {f(zero_acc['model_ranker']['top1_macro_pct'],2)}% for the existing hybrid; this run did not preserve the neural prediction advantage.",
             "- **Deadline:** the calibrated next-read windows rejected all deadline-policy held-out scoring requests before enqueue. It prefetched zero pages in both conditions; any measured latency difference from context primarily accompanies avoiding work, rather than a demonstrated timely prefetch gain.",
             "- **Adaptive net benefit:** the development-calibrated estimate rejected all eligible held-out actions in both conditions. It made zero prefetched insertions; this run does not establish that the controller can find profitable prefetch opportunities.",
             "- **Combined:** it made zero prefetched insertions. In the zero-delay condition its paired delta versus none was "
             f"{f(zero_runtime['combined']['vs_none']['mean'])} ms with CI {ci(zero_runtime['combined']['vs_none'])}; with requested delay it was "
             f"{f(delayed_runtime['combined']['vs_none']['mean'])} ms with CI {ci(delayed_runtime['combined']['vs_none'])}. The primary success criterion was not achieved.",
             "- **Resource costs:** all selective variants still load the frozen neural checkpoint in the adapter even when neural fallback is disabled. In zero-delay runs the combined policy's largest process peak was "
             f"{f(zero_runtime['combined']['max_peak_process_rss_bytes']/1_000_000,1)} MB versus {f(zero_runtime['none']['max_peak_process_rss_bytes']/1_000_000,1)} MB for none; total CPU was "
             f"{f(zero_runtime['combined']['cpu_seconds_total'],2)} s versus {f(zero_runtime['none']['cpu_seconds_total'],2)} s. Individual scheduling CPU time was not isolated.",
             "- **Next independent cycle:** consider lazily loading the neural model only when fallback is validated, and calibrating readiness on a less restrictive, observable demand horizon. Freeze any revised choices on development data and use new held-out seeds; do not tune to these results.",
             "", "## Interpretation and limitations", "",
             "Treat the paired intervals above as the verdict on the declared latency hypothesis for each condition. A negative mean alone is not a success if its 95% interval includes zero. Context selection may trade the neural ranker's offline accuracy for lower inference cost; deadline and adaptive controls can skip many or all predictions. No confidence threshold or policy was tuned on held-out results.", "",
             "The candidate-based offline comparison does not simulate queue timing, per-query admission caps or runtime cache state. The actual job analysis is conditional and can contain missing/ambiguous alignments. OS caching, small family sample sizes, five seeds, one host, an artificial storage delay, multiple unadjusted policy comparisons, and possible overlap of parameter values with prior training limit generalization. The controller's readiness/benefit values are estimates, not measured causal savings. The opt-in adapter reuses the existing neural request path, which can parse current page layout even for a queued Markov action; that work is not separately timed.", "",
             "Outputs and raw per-run traces live under ignored `backend/runs/selective-prefetch-20261027/`. The aggregate machine-readable results are `backend/selective-prefetch-results.json`. The frontend's SDSS and prior research exports remain separate.", ""]
    (ROOT / "SELECTIVE_PREFETCH_RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    shutil.copyfile(OUTPUT / "heldout-summary-detailed.json", ROOT / "selective-prefetch-results.json")
    print(f"Report written: {ROOT / 'SELECTIVE_PREFETCH_RESULTS.md'}")


if __name__ == '__main__':
    main()
