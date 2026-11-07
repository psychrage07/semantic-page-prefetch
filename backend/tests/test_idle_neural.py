import json

import numpy as np
import pytest
import torch

from idle_markov import QueryMarkov
from idle_neural import IdleNeuralHybrid, early_miss_records
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex


def examples():
    records = []
    for i in range(8):
        context_page, target = (2, 3) if i % 2 == 0 else (5, 4)
        for name, pages, misses in (("A", [1, context_page], [context_page]), ("B", [1, target], [target])):
            records.append({"session_id": str(i), "query_id": f"{i}-{name}", "template_id": name,
                            "pages": pages, "miss_pages": misses, "family": "range_scan", "plan_class": "scan"})
    return early_miss_records(records)


def make_model(records, no_semantics=False):
    prior = QueryMarkov(2).fit(records)
    index = CandidateIndex.build(records, max_candidates=64)
    metadata = {str(i): {"object": "sample", "page_type": "leaf", "depth": 0} for i in range(1, 6)}
    contexts = {"A": {"workload_family": "range_scan", "plan_class": "scan"}}
    return IdleNeuralHybrid(prior, index, ML.build_training_transition_counts(records), metadata,
                            contexts, hidden=16, no_semantics=no_semantics)


def test_neural_variant_is_trained_and_saved_predictions_match(tmp_path):
    torch.set_num_threads(1)
    data = examples()
    model = make_model(data)
    fit = model.fit(data, data, epochs=2, budget=1)
    assert fit["selected_epoch"] >= 1
    assert fit["selected_residual_weight_norm"] > 0
    directory = tmp_path / "weights"
    model.save(directory, fit=fit, database_identity={"test": 1}, metadata_path=tmp_path / "page_map.json")
    loaded = IdleNeuralHybrid.load(directory, model.metadata, {"test": 1})
    history = [model.state("A", [1, 2])]
    assert loaded.predict(history)["pages"] == model.predict(history)["pages"]
    with pytest.raises(ValueError, match="different database"):
        IdleNeuralHybrid.load(directory, model.metadata, {"test": 2})
    (directory / "query_prior.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        IdleNeuralHybrid.load(directory, model.metadata, {"test": 1})


def test_semantics_are_present_and_excluded_pages_never_reach_neural_scoring():
    data = examples()
    model, control = make_model(data), make_model(data, True)
    history = [model.state("A", [1, 2])]
    semantic, _, proposal, _ = model.features(history)
    ablated, _, _, _ = control.features(history)
    assert semantic.shape == ablated.shape
    assert np.any(semantic != ablated)
    before = json.dumps(model.prior.counts, sort_keys=True)
    features, _, prediction, _ = model.features(history, exclude_pages=[3])
    assert [p["page"] for p in prediction["pages"]] == [4]
    assert features.shape[0] == 1
    assert json.dumps(model.prior.counts, sort_keys=True) == before


def test_early_miss_targets_keep_full_completed_query_state():
    records = [{"session_id": "s", "template_id": "A", "pages": [1, 2, 3], "miss_pages": [3, 3]}]
    transformed = early_miss_records(records)
    assert transformed[0]["target_pages"] == [3]
    assert transformed[0]["pages"] == [1, 2, 3]
    assert "target_pages" not in records[0]


def test_validation_keeps_targets_when_candidate_generator_abstains():
    data = examples()
    model = make_model(data)
    model.fit(data, data, epochs=1, budget=1)
    unfamiliar = [
        {**data[0], "session_id": "unseen", "template_id": "unseen-template"},
        {**data[1], "session_id": "unseen"},
    ]
    groups = model.groups(unfamiliar, min_confidence=1.0)
    assert len(groups) == 1 and groups[0]["x"] is None
    measured = model.evaluate(groups)
    assert measured["targets"] == 1
    assert measured["early_miss_recall_micro_pct"] == 0
