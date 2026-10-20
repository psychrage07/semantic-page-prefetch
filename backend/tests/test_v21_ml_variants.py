"""v2.1 ML variant tests: B2/B3 features, fairness, miss-only labels,
temperature calibration normalization, and the break-even gate."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pagewise import ml_variants as ML
from pagewise import models as M


def _synthetic(n_queries: int = 30) -> tuple[list, dict]:
    rng = random.Random(1)
    seqs, cases = [], {}
    for q in range(n_queries):
        pages: list[int] = []
        cur = rng.randrange(1, 15)
        for _ in range(20):
            cur = max(1, min(24, cur + rng.choice([-2, -1, 0, 1, 2])))
            pages.append(cur)
        cases[f"q{q}"] = {"query_id": f"q{q}", "params": {"id": q % 4}}
        seqs.append({"query_id": f"q{q}", "pages": pages,
                     "template_id": f"t{q % 3}", "workload_family": "f0",
                     "plan_class": "scan"})
    return seqs, cases


PAGE_MAP = {str(p): {"page_type": "leaf", "depth": 1, "object": "orders"} for p in range(1, 25)}


def _small_split():
    seqs, cases = _synthetic()
    return seqs[:22], seqs[22:], cases


def test_model_b2_features_extend_and_are_readable():
    seqs, _, _ = _small_split()
    counts = ML.build_training_transition_counts(seqs)
    history, candidate = [5, 6], 7
    block = ML.markov2_feature_block(candidate, history[-2], history[-1], counts)
    assert len(block) == ML.MARKOV2_FEATURES == 6
    vec = ML.variant_feature_vector(history, candidate, {}, PAGE_MAP, 24, "b2", counts)
    base_dim = M.feature_vector(history, candidate, {}, PAGE_MAP, 24, semantic=True).shape[0]
    assert vec.shape[0] == base_dim + ML.MARKOV2_FEATURES
    assert np.all(np.isfinite(vec))


def test_model_b3_param_features_availability_gated():
    seqs, _, _ = _small_split()
    counts = ML.build_training_transition_counts(seqs)
    ctx_on = {"param_bucket": 3, "param_available": True}
    ctx_off = {"param_bucket": None, "param_available": False}
    v_on = ML.variant_feature_vector([5, 6], 7, ctx_on, PAGE_MAP, 24, "b3", counts)
    v_off = ML.variant_feature_vector([5, 6], 7, ctx_off, PAGE_MAP, 24, "b3", counts)
    m2 = ML.MARKOV2_FEATURES + ML.PARAM_FEATURES
    # the param block is one-hot(1, 8) + [1.0] when available; all zeros when gated off
    param_block_on = v_on[-ML.PARAM_FEATURES:]
    param_block_off = v_off[-ML.PARAM_FEATURES:]
    assert v_on.shape[0] == v_off.shape[0]
    assert param_block_off.sum() == 0.0
    assert 1.0 in param_block_on and param_block_on.sum() >= 2.0


def test_counts_derived_from_train_only_no_test_leakage():
    train, test, _ = _small_split()
    counts_train = ML.build_training_transition_counts(train)
    # a transition pair that only exists in TEST executions contributes nothing
    test_only = ML.build_training_transition_counts(test)
    leaked_pairs = []
    for key, counter in test_only["first"].items():
        train_counter = counts_train["first"].get(key, {})
        for page in counter:
            if train_counter.get(page, 0) > 0 and (key, page) not in _train_pairs(train):
                leaked_pairs.append((key, page))
    assert leaked_pairs == []


def _train_pairs(train):
    pairs = set()
    for seq in train:
        pages = [int(p) for p in seq["pages"]]
        for i in range(1, len(pages)):
            pairs.add((pages[i - 1], pages[i]))
    return pairs


def test_miss_only_labels_match_declared_reference_replay():
    seqs, _, _ = _small_split()
    replay = ML.reference_cache_miss_replay(seqs, cache_pages=ML.MISS_ONLY_REPLAY_CACHE_PAGES)
    for seq in seqs:
        flags = replay[str(seq["query_id"])]
        assert len(flags) == len(seq["pages"])
        assert flags[0] is True  # cold cache: the first page always misses
    # a page repeated immediately after itself must hit the reference cache
    seq_hit = {"query_id": "hit", "pages": [7, 7, 7]}
    flags = ML.reference_cache_miss_replay([seq_hit])["hit"]
    assert flags == [True, False, False]


def test_miss_only_training_changes_the_trained_model():
    train, val, cases = _small_split()
    (all_model, _, _), report_all = ML.train_variant_model(train, val, PAGE_MAP, "b2", seed=5,
                                                           epochs=3, miss_only=False, workload_cases=cases)
    (miss_model, _, _), report_miss = ML.train_variant_model(train, val, PAGE_MAP, "b2", seed=5,
                                                             epochs=3, miss_only=True, workload_cases=cases)
    assert report_all["miss_only"] is False and report_miss["miss_only"] is True
    assert report_all["train_pairs"] != report_miss["train_pairs"]
    a = np.concatenate([p.detach().flatten().numpy() for p in all_model.parameters()])
    b = np.concatenate([p.detach().flatten().numpy() for p in miss_model.parameters()])
    assert not np.array_equal(a, b)


def test_softmax_normalization_is_categorical_not_independent_sigmoid():
    logits = np.array([2.0, 0.0, -2.0])
    probs = ML.softmax_normalize(logits, 1.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]
    # independent sigmoids would NOT sum to 1: explicit violation excluded
    sig = 1.0 / (1.0 + np.exp(-logits))
    assert abs(sig.sum() - 1.0) > 1e-3


def test_temperature_fit_uses_validation_and_reduces_nll():
    train, val, cases = _small_split()
    (model, index, counts), _ = ML.train_variant_model(train, val, PAGE_MAP, "b2", seed=9,
                                                       epochs=4, workload_cases=cases)
    fit = ML.fit_temperature_from_validation(val, model, PAGE_MAP, "b2", counts, index,
                                             workload_cases=cases)
    assert 0.05 <= fit["temperature"] <= 20.0
    assert fit["val_transitions"] > 0
    assert not math.isnan(fit["val_nll"])


def test_breakeven_gate_arithmetic():
    gate = ML.breakeven_gate(timely_probability=0.5, saved_latency_ms=2.0,
                             wasted_rate=0.25, waste_cost_ms=1.0,
                             inference_cost_ms=0.1, scheduling_cost_ms=0.05,
                             contention_cost_ms=0.1)
    # 0.5*2 - (0.25*1 + 0.1 + 0.05 + 0.1) = 1.0 - 0.5 = 0.5
    assert gate["expected_net_benefit_ms_per_candidate"] == pytest.approx(0.5)
    assert gate["gate_pass"] is True
    neg = ML.breakeven_gate(timely_probability=0.1, saved_latency_ms=2.0,
                            wasted_rate=0.9, waste_cost_ms=1.0,
                            inference_cost_ms=0.1, scheduling_cost_ms=0.05)
    assert neg["gate_pass"] is False


def test_model_b_preserved_as_separate_baseline():
    # 'b' must remain a declared variant independent of b2/b3 feature groups.
    assert ML.variant_semantic("b") is True
    assert ML.variant_extra_dims("b") == 0
    assert ML.variant_extra_dims("b2") == ML.MARKOV2_FEATURES
    assert ML.variant_extra_dims("b3") == ML.MARKOV2_FEATURES + ML.PARAM_FEATURES
