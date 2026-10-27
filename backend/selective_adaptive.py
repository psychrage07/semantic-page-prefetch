"""Interpretable opt-in benefit heuristic; estimates must come from development."""
import math

from selective_context import context_action, has_second_order


def net_benefit(action, estimates, *, queue_fraction, cache_fraction):
    def value(name):
        n = float(estimates.get(name, 0))
        return max(0.0, n) if math.isfinite(n) else 0.0
    p = min(1.0, value("prediction_accuracy"))
    usefulness = min(1.0, value("usefulness_given_prediction"))
    demand_cost = value("demand_miss_ms")
    score_cost = value("score_ms") if action == "neural" else value("markov_score_ms")
    read_cost, wait, deadline = value("read_ms"), value("queue_wait_ms"), value("deadline_ms")
    ready = max(0.0, min(1.0, (deadline - score_cost - read_cost - wait) / deadline)) if deadline else 0.0
    pressure = max(0.0, min(1.0, queue_fraction)) + max(0.0, min(1.0, cache_fraction))
    benefit = p * usefulness * ready * demand_cost
    cost = score_cost + wait + read_cost * pressure
    return {"expected_benefit_ms": benefit, "estimated_cost_ms": cost,
            "estimated_net_ms": benefit - cost, "ready_probability_heuristic": ready,
            "queue_fraction": queue_fraction, "cache_fraction": cache_fraction}


def choose_action(index, history, family, policy, config, *, queue_fraction=0.0, cache_fraction=0.0):
    exact = has_second_order(index, history)
    if policy in {"context", "deadline", "adaptive", "combined"}:
        action, reason = context_action(index, history, family, config)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if policy not in {"adaptive", "combined"} or action == "skip":
        return action, reason, None
    group = "exact" if exact else "fallback"
    estimates = config.get("estimates", {}).get(family, {}).get(group, {}).get(action)
    if not estimates:
        return "skip", "missing_development_estimate", None
    net = net_benefit(action, estimates, queue_fraction=queue_fraction, cache_fraction=cache_fraction)
    if net["estimated_net_ms"] <= 0:
        return "skip", "nonpositive_expected_net", net
    return action, "positive_expected_net", net
