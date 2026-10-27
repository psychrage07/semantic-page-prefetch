from types import SimpleNamespace

from selective_context import context_action


def test_known_context_preserves_markov2_even_for_validated_neural_family():
    index = SimpleNamespace(second_order={"1:2": [9, 8]}, successors={2: [7]})
    assert context_action(index, [1, 2], "point_lookup", {"neural_fallback_families": ["point_lookup"]}) == ("markov2", "exact_second_order")


def test_only_validated_families_get_neural_fallback():
    index = SimpleNamespace(second_order={}, successors={2: [7]})
    config = {"neural_fallback_families": ["point_lookup", "recursive_traversal"]}
    assert context_action(index, [1, 2], "point_lookup", config)[0] == "neural"
    assert context_action(index, [1, 2], "indexed_lookup", config)[0] == "markov1"
    assert context_action(index, [1, 2], "recursive_traversal", config)[0] == "markov1"
    assert context_action(index, [5], "point_lookup", {})[0] == "skip"
