import numpy as np
import torch

from idle_optimized import CachedSemantics, OptimizedIdleHybrid
from pagewise import models as M
from test_idle_neural import examples, make_model


def trained():
    torch.set_num_threads(1)
    data = examples()
    model = make_model(data)
    model.fit(data, data, epochs=2, budget=1)
    return model


def test_cached_neural_scores_and_rankings_match_original_with_eviction_and_exclusions():
    model = trained()
    optimized = OptimizedIdleHybrid(model)
    optimized.features.capacity = 1  # Exercise batches larger than memo capacity.
    for previous in (2, 5, 2):
        history = [model.state("A", [1, previous])]
        for exclude in ((), (3,), (3, 4)):
            expected = model.predict(history, exclude_pages=exclude)
            actual = optimized.predict(history, exclude_pages=exclude)
            assert [p["page"] for p in actual["pages"]] == [p["page"] for p in expected["pages"]]
            np.testing.assert_allclose([p["neural_markov_score"] for p in actual["pages"]],
                                       [p["neural_markov_score"] for p in expected["pages"]], atol=1e-6)
            assert len(optimized.features.values) <= 1
    assert optimized.priors.stats["hits"] > 0


def test_static_metadata_cache_keeps_query_context_separate():
    metadata = {"2": {"object": "orders", "page_type": "internal", "depth": 2}}
    cached = CachedSemantics(metadata, capacity=1)
    for template, plan in (("A", "scan"), ("B", "search_index"), ("A", "scan")):
        context = {"template_id": template, "plan_class": plan, "workload_family": "range_scan"}
        expected = np.asarray([M.semantic_page_features(p, metadata, context) for p in (2, 999)], np.float32)
        np.testing.assert_array_equal(cached.semantic_matrix([2, 999], context), expected)
        assert len(cached.pages.values) <= 1


def test_selective_gate_is_logged_and_uncertain_cutoff_still_uses_neural_weights():
    model = trained()
    gate = OptimizedIdleHybrid(model, selective=True)
    history = [model.state("A", [1, 2])]
    all_fit = gate.predict(history, max_pages=2)
    assert not all_fit["neural_scored"] and all_fit["route"] == "markov_all_candidates_fit"
    ambiguous = gate.predict(history, max_pages=1)
    assert ambiguous["neural_scored"] and ambiguous["route"] == "semantic_neural"
    assert ambiguous["pages"] == model.predict(history, max_pages=1)["pages"]
    excluded = gate.predict(history, exclude_pages=[3, 4])
    assert excluded["pages"] == [] and excluded["route"] == "no_candidates"
