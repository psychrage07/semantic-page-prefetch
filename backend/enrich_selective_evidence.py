"""Add per-family accuracy intervals to preserved held-out analysis.

This does not change the frozen policy, workload, or any measured outcome.
"""
import json
from queue_experiments import OUTPUT, SCENARIOS, source_digests, write
from summarize_selective import offline_summary


def main():
    old = json.loads((OUTPUT / "heldout-summary.json").read_text(encoding="utf-8"))
    policy = json.loads((OUTPUT / "policy-frozen.json").read_text(encoding="utf-8"))
    if policy["source_sha256"] != source_digests():
        raise ValueError("Frozen decision source changed")
    target = OUTPUT / "heldout-summary-detailed.json"
    if target.exists():
        raise FileExistsError("Preserve prior detailed analysis")
    for scenario in SCENARIOS:
        reference = OUTPUT / ("heldout-offline-zero.csv" if scenario == "zero" or old["same_reference_streams_across_scenarios"] else "heldout-offline-controlled-delay.csv")
        new = offline_summary(reference, policy, scenario)
        prior = old["scenarios"][scenario]["offline_accuracy"]["policies"]
        for name, row in new["policies"].items():
            for field in ["top1_macro_pct", "top1_micro_pct", "top5_micro_pct", "candidate_recall_pct"]:
                if abs(row[field] - prior[name][field]) > 1e-10:
                    raise ValueError(f"Original accuracy changed: {scenario}/{name}/{field}")
        old["scenarios"][scenario]["offline_accuracy"] = new
    old["analysis_revision"] = "Family-level query-average accuracy intervals and paired family accuracy intervals added after the initial aggregate analysis; policies and raw outcomes unchanged."
    write(target, old)
    print("Saved family accuracy intervals; original summary and frozen policy preserved.")


if __name__ == "__main__":
    main()
