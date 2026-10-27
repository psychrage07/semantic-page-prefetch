from types import SimpleNamespace

from selective_adaptive import choose_action, net_benefit


def test_net_benefit_accounts_for_cost_and_pressure():
    estimates = {"prediction_accuracy": 0.9, "usefulness_given_prediction": 0.9,
                 "demand_miss_ms": 10, "score_ms": 1, "markov_score_ms": 0.1,
                 "read_ms": 1, "queue_wait_ms": 0.1, "deadline_ms": 10}
    low = net_benefit("neural", estimates, queue_fraction=0, cache_fraction=0)
    high = net_benefit("neural", estimates, queue_fraction=1, cache_fraction=1)
    assert low["estimated_net_ms"] > high["estimated_net_ms"]
    assert low["expected_benefit_ms"] > 0
    assert net_benefit("neural", {**estimates, "deadline_ms": 0}, queue_fraction=0, cache_fraction=0)["estimated_net_ms"] < 0


def test_adaptive_skips_when_estimates_are_missing_or_unprofitable():
    index = SimpleNamespace(second_order={"1:2": [7]}, successors={2: [8]})
    assert choose_action(index, [1, 2], "point_lookup", "combined", {})[:2] == ("skip", "missing_development_estimate")
    poor = {"prediction_accuracy": 0.1, "usefulness_given_prediction": 0.1,
            "demand_miss_ms": 0.01, "markov_score_ms": 2, "deadline_ms": 1}
    config = {"estimates": {"point_lookup": {"exact": {"markov2": poor}}}}
    action, reason, detail = choose_action(index, [1, 2], "point_lookup", "combined", config)
    assert (action, reason) == ("skip", "nonpositive_expected_net")
    assert detail["estimated_net_ms"] < 0


def test_adaptive_uses_markov_when_supported_and_profitable():
    index = SimpleNamespace(second_order={"1:2": [7]}, successors={2: [8]})
    good = {"prediction_accuracy": 0.9, "usefulness_given_prediction": 0.9,
            "demand_miss_ms": 10, "markov_score_ms": 0.01, "read_ms": 0.1, "deadline_ms": 10}
    config = {"estimates": {"point_lookup": {"exact": {"markov2": good}}}}
    assert choose_action(index, [1, 2], "point_lookup", "combined", config)[0] == "markov2"
