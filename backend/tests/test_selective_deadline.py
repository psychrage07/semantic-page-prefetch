import math

from selective_deadline import deadline_feasible


def test_requires_enough_time_for_remaining_work_before_deadline():
    request = 1_000_000_000
    assert deadline_feasible(request + 1_000_000, request, 5.0, 3.0)
    assert not deadline_feasible(request + 2_000_000, request, 5.0, 3.0)
    assert not deadline_feasible(request + 6_000_000, request, 5.0, 0.0)


def test_missing_or_invalid_estimates_never_create_a_deadline():
    for budget, remaining in [(None, 1), (0, 1), (math.inf, 1), (5, -1), (5, math.nan)]:
        assert not deadline_feasible(0, 0, budget, remaining)
