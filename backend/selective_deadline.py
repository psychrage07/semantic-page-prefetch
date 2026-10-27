"""Deadline admission using only measured past estimates and a monotonic clock."""
import math


def deadline_feasible(now_ns, request_ns, budget_ms, remaining_ms):
    if budget_ms is None or not math.isfinite(budget_ms) or budget_ms <= 0:
        return False
    if not math.isfinite(remaining_ms) or remaining_ms < 0:
        return False
    return now_ns + remaining_ms * 1_000_000 < request_ns + budget_ms * 1_000_000
