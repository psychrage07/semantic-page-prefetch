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
import hashlib
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
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("Unknown alternative")
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations")
    import numpy as np
    effective = {unit: np.asarray(vals, dtype=float) for unit, vals in sessions.items() if len(vals)}
    if any(not np.isfinite(vals).all() for vals in effective.values()):
        raise ValueError("Paired deltas must be finite")
    if not effective:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "p_value": 1.0,
                "seeds": 0, "clusters": 0, "matched_pairs": 0, "iterations": iterations}
    units = sorted(effective, key=repr)
    arrays = [effective[u] for u in units]
    flat = np.concatenate(arrays)
    observed = float(flat.mean())
    rng = np.random.default_rng(seed)
    # Draw clusters, then paired observations within each occurrence of a cluster.
    draws = rng.integers(len(units), size=(iterations, len(units)))
    total, counts = np.zeros(iterations), np.zeros(iterations)
    for i, vals in enumerate(arrays):
        rows, slots = np.where(draws == i)
        # Bound memory for large query/session arrays.
        for start in range(0, len(rows), max(1, 1_000_000 // len(vals))):
            row = rows[start:start + max(1, 1_000_000 // len(vals))]
            samples = rng.integers(len(vals), size=(len(row), len(vals)))
            np.add.at(total, row, vals[samples].sum(axis=1))
            np.add.at(counts, row, len(vals))
    means = total / counts
    lo, hi = np.quantile(means, [0.025, 0.975])
    # Center the bootstrap distribution at the null. Do not infer p-values
    # from CI width. The +1 correction prevents zero Monte Carlo p-values.
    centered = means - observed
    if alternative == "greater":
        extreme = centered >= observed - 1e-12
    elif alternative == "less":
        extreme = centered <= observed + 1e-12
    else:
        extreme = np.abs(centered) >= abs(observed) - 1e-12
    p_value = float((np.count_nonzero(extreme) + 1) / (iterations + 1))
    return {"mean": round(observed, 6), "median_paired_delta": round(float(np.median(flat)), 6),
            "ci95_low": round(float(lo), 6), "ci95_high": round(float(hi), 6),
            "p_value": p_value, "seeds": len(units), "clusters": len(units),
            "matched_pairs": len(flat), "iterations": iterations,
            "p_value_method": "centered hierarchical paired bootstrap (Monte Carlo +1 correction)"}


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
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and between zero and one")
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
    """Apply Holm to actual bootstrap p-values; CIs alone are insufficient."""
    if any("p_value" not in result for result in results):
        raise ValueError("Holm correction requires p-values; confidence intervals alone are insufficient")
    return holm_correction([r["p_value"] for r in results], alpha=alpha)


def declare_family(
    comparisons: Iterable[tuple[str, dict[Any, list[float]]]],
    *,
    favorable_directions: dict[str, str] | None = None,
    iterations: int = 10_000,
    bootstrap_seed: int = 17,
    alpha: float = 0.05,
    replication_unit: str = "caller-supplied independent clusters",
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
            seed=bootstrap_seed + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little"),
        )
    p_values = [boot[label]["p_value"] for label in labels]
    decisions = holm_correction(p_values, alpha=alpha)
    ordered = sorted(range(len(labels)), key=lambda i: (p_values[i], labels[i]))
    members: dict[str, Any] = {}
    running_adjusted = 0.0
    for rank, index in enumerate(ordered, start=1):
        label = labels[index]
        result = boot[label]
        direction = directions.get(label, "greater")
        running_adjusted = max(running_adjusted, min(1.0, (len(labels) - rank + 1) * p_values[index]))
        favorable = result["mean"] is not None and (result["mean"] < 0 if direction == "less" else result["mean"] > 0)
        members[label] = {
            **result,
            "favorable_direction": direction,
            "raw_decision": raw_decision(result["ci95_low"], result["ci95_high"],
                                         favorable_direction="low" if direction == "less" else "greater"),
            "holm_p_unadjusted": p_values[index],
            "holm_adjusted_p": running_adjusted,
            "holm_decision": decisions[index] and favorable,
            "holm_reject_equal_performance": decisions[index],
            "holm_step_rank": rank,
        }
    return {
        "schema_version": "v2.1",
        "family_labels": labels,
        "replication_unit": replication_unit,
        "alpha": alpha,
        "members": members,
        "note": ("Paired deltas matched within caller-declared clusters; hierarchical bootstrap "
                 "resamples clusters and matched observations within clusters. Holm applied "
                 "across the declared family; raw CIs and effect sizes reported alongside "
                 "corrected decisions."),
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
