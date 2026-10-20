"""v2.1 statistics layer: declared comparison families, cluster-aware paired
bootstrap, and Holm step-down correction.

Relationship to v2 statistics
-----------------------------
v2 (`offline.paired_bootstrap_ci`, `suite._bootstrap_mean_ci`) resampled
individual paired observations. Query observations are clustered within workload
seeds: two queries from the same seed share the workload mix, the OS-cache
condition of their trial, and the query-parameter generator state, so treating
them as independent underestimates interval widths. The declared v2.1 family
therefore uses a two-level cluster bootstrap: workload *seeds* are resampled
with replacement, and within each drawn seed every matched query observation of
that seed is resampled with replacement. Decisions are unchanged in form
(CI excludes zero) but the family-level decision applies Holm correction.

Both the raw and corrected decisions, the raw effect sizes and CIs, and the
replication units (seeds, matched query pairs) are always reported together.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Iterable, Sequence

# A declared statistic for one comparison in a predeclared family.
Comparison = dict[str, Any]


def paired_cluster_bootstrap_ci(
    sessions: dict[Any, list[float]],
    *,
    alternative: str = "greater",
    iterations: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Cluster-aware bootstrap CI of the mean paired difference.

    ``sessions`` maps a replication unit (typically the workload seed) to its
    matched paired differences (candidate − baseline; negative favors candidate).
    Null/empty sessions are ignored, matching v2's matched-by-(seed, query id)
    denominator discipline.

    ``alternative``: "greater" (candidate > baseline), "less", or "two-sided".
    For runtime latency where negative favors the candidate, pass
    ``alternative="less"`` or filter at the call site; the returned CI is
    always the distribution of the *signed* mean delta.
    """
    effective = {unit: list(vals) for unit, vals in sessions.items() if vals}
    if not effective:
        return {
            "mean": None, "ci95_low": None, "ci95_high": None,
            "seeds": 0, "matched_pairs": 0, "iterations": iterations,
        }
    units = sorted(effective, key=repr)
    n_pairs = sum(len(v) for v in effective.values())
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        count = 0
        for _unit in units:
            session = effective[rng.choice(units)]
            for _q in session:
                total += session[rng.randrange(len(session))]
                count += 1
        means.append(total / count)
    means.sort()
    lo = means[int(0.025 * (iterations - 1))]
    hi = means[int(0.975 * (iterations - 1))]
    observed = statistics.mean(v for vals in effective.values() for v in vals)
    return {
        "mean": round(observed, 6),
        "median_paired_delta": round(statistics.median([v for vals in effective.values() for v in vals]), 6),
        "ci95_low": round(lo, 6),
        "ci95_high": round(hi, 6),
        "seeds": len(units),
        "matched_pairs": n_pairs,
        "iterations": iterations,
    }


def raw_decision(ci_low: float | None, ci_high: float | None, *,
                 favorable_direction: str) -> str:
    """One of 'supported', 'not supported', 'inconclusive' for one test."""
    if ci_low is None or ci_high is None:
        return "inconclusive"
    if ci_low <= 0 <= ci_high:
        return "inconclusive"
    favorable_high = favorable_direction == "low"  # negative deltas favor candidate
    in_favor = ci_high < 0 if favorable_high else ci_low > 0
    if in_favor:
        return "supported"
    return "not supported"


def holm_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm step-down: returns the list of familywise decisions at ``alpha``."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    decisions = [False] * len(p_values)
    running_max = 0.0
    rejected = True
    for rank, index in enumerate(order):
        adjusted = (len(p_values) - rank) * max(p_values[index], 0.0)
        running_max = max(running_max, adjusted)
        if running_max <= alpha and rejected:
            decisions[index] = True
        else:
            rejected = False
    return decisions


def holm_from_bootstrap(results: Sequence[dict[str, Any]], *, alpha: float = 0.05) -> list[bool]:
    """Holm decisions from cluster-bootstrap CIs by inverting the CI coverage.

    Each CI implies a two-sided p-value of at most 0.05 when the CI excludes
    zero, of 1.0 otherwise. Because CIs alone are order-incompatible for exact
    Holm p-values, v2.1 declares comparisons on *scaled* pivotal statistics:
    use :func:`declare_family` for the full pipeline. This helper is kept for
    callers that only have CI decisions and applies Holm on the convention
    p ∈ {0.001 (CI excludes zero in the favorable direction), 0.5 (otherwise)}.
    """
    p_values: list[float] = []
    for result in results:
        excluded = result["ci95_low"] is not None and not (
            result["ci95_low"] <= 0 <= result["ci95_high"]
        )
        favorable = excluded and (
            (result["ci95_high"] < 0) == (result.get("favorable_direction", "greater") == "less")
        ) or (excluded and result.get("favorable_direction", "greater") == "greater" and result["ci95_low"] > 0)
        p_values.append(0.001 if favorable else 0.5)
    return holm_correction(p_values, alpha=alpha)


def declare_family(
    comparisons: Iterable[tuple[str, dict[Any, list[float]]]],
    *,
    favorable_directions: dict[str, str] | None = None,
    iterations: int = 10_000,
    bootstrap_seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Predeclared family evaluation with Holm correction.

    ``comparisons`` is an ordered iterable of ``(label, sessions)`` — the family
    definition (order membership, direction per label) must be frozen in
    configuration *before* evaluation. Returns every member's raw CI, raw
    decision, adjusted decision and Holm step-down rank, plus the declared
    directions and the replication units.
    """
    labels: list[str] = []
    boot: dict[str, dict[str, Any]] = {}
    directions = favorable_directions or {}
    for label, sessions in comparisons:
        label = str(label)
        labels.append(label)
        boot[label] = paired_cluster_bootstrap_ci(
            sessions, alternative="two-sided", iterations=iterations,
            seed=bootstrap_seed + (abs(hash(label)) % 1000) * 149,  # deterministic, label-distinct
        )
    # Convert each CI to a Holm-orderable p-value via the CI-width transform:
    # p = 0.05 * max(0, |center| / max(0, dist to nearest CI edge))^-1 is unstable;
    # instead use the declared convention: the natural test statistic is the
    # pivotal z = observed mean / CI half-width, and the two-sided p-value from
    # the normal pair the CI implies.
    p_values: list[float] = []
    for label in labels:
        r = boot[label]
        if r["mean"] is None:
            p_values.append(1.0)
            continue
        center = r["mean"]
        half = (r["ci95_high"] - r["ci95_low"]) / 2.0
        if half <= 0:
            half = max(abs(center), 1e-9)
        z = center / half
        p_values.append(max(1e-12, 2.0 * (1.0 - _norm_cdf(z))))
    decisions = holm_correction(p_values, alpha=alpha)
    ordered = sorted(range(len(labels)), key=lambda i: (p_values[i], labels[i]))
    members: dict[str, Any] = {}
    for rank, index in enumerate(ordered, start=1):
        label = labels[index]
        result = boot[label]
        direction = directions.get(label, "greater")
        members[label] = {
            **result,
            "favorable_direction": direction,
            "raw_decision": raw_decision(result["ci95_low"], result["ci95_high"],
                                         favorable_direction="low" if direction == "less" else "greater"),
            "holm_p_unadjusted": p_values[index],
            "holm_adjusted_p": min(1.0, (len(labels) - rank + 1) * p_values[index]),
            "holm_decision": decisions[index],
            "holm_step_rank": rank,
        }
    return {
        "schema_version": "v2.1",
        "family_labels": labels,
        "alpha": alpha,
        "members": members,
        "note": ("Paired deltas matched by workload seed and query id; cluster bootstrap "
                 "resamples seed sessions and matched query pairs within seeds. Holm applied "
                 "across the declared family; raw CIs and effect sizes reported alongside "
                 "corrected decisions."),
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
