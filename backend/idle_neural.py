"""Experimental query-level adaptation of Pagewise's semantic neural-Markov ranker.

All inputs describe completed queries. Target labels are the next training
query's first distinct demand misses; no next-query hints are used online.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from idle_markov import QueryMarkov
from pagewise import ml_variants as ML
from pagewise.models import CandidateIndex, MLPScorer
from pagewise.ranking import RankerConfig, group_features


def early_miss_records(records, budget=16):
    return [{**row, "target_pages": list(dict.fromkeys(row.get("miss_pages", row["pages"])))[:budget]} for row in records]


class IdleNeuralHybrid:
    order, signal = 2, "neural_semantic_query"

    def __init__(self, prior, feature_index, counts, metadata, contexts, *, hidden=64,
                 no_semantics=False, candidate_limit=64):
        self.prior, self.index, self.counts, self.metadata, self.contexts = prior, feature_index, counts, metadata, contexts
        self.config = RankerConfig("idle_query_neural_markov", variant="b2", hidden=hidden, epochs=16,
                                   prior_weight=1.0, no_semantics=no_semantics)
        self.candidate_limit, self.model = candidate_limit, None

    def state(self, template, pages):
        return {"template": str(template), "pages": [int(p) for p in pages[-8:]]}

    def features(self, history, *, min_confidence=0.25, exclude_pages=()):
        started = time.perf_counter_ns()
        prediction = self.prior.predict([row["template"] for row in history], max_pages=self.candidate_limit,
                                         min_confidence=min_confidence, exclude_pages=exclude_pages)
        candidate_done = time.perf_counter_ns()
        pages = [row["page"] for row in prediction["pages"]]
        if not history or not pages:
            return None, None, prediction, {}
        latest = history[-1]
        context = {"template_id": latest["template"], "position_in_query": 0.0,
                   **self.contexts.get(latest["template"], {"workload_family": "unknown", "plan_class": "other"})}
        # Uses exactly the existing B2 semantic/page-history feature pipeline.
        # Its context describes the completed query, not the unknown next query.
        features = group_features(latest["pages"] or [0], pages, context, self.metadata,
                                  self.index, self.counts, self.config, [1.0, 1.0, 1.0])
        prior = np.asarray([np.log(max(1e-4, row["confidence"])) for row in prediction["pages"]], np.float32)
        return features, prior, prediction, {"candidate_ns": candidate_done - started,
                                            "feature_ns": time.perf_counter_ns() - candidate_done}

    def predict(self, completed_history, *, max_pages=16, min_confidence=0.25, exclude_pages=()):
        x, prior, proposal, timings = self.features(completed_history, min_confidence=min_confidence,
                                                    exclude_pages=exclude_pages)
        if x is None:
            return {**proposal, "pages": [], "neural_scored": False, "stage_ns": timings}
        start = time.perf_counter_ns()
        with torch.inference_mode():
            scores = self.model(torch.from_numpy(x)) + torch.from_numpy(prior)
            order = torch.argsort(scores, descending=True, stable=True).numpy()
        timings["forward_and_sort_ns"] = time.perf_counter_ns() - start
        ranked = [{**proposal["pages"][int(i)], "neural_markov_score": float(scores[int(i)])}
                  for i in order[:max_pages]]
        return {**proposal, "pages": ranked, "neural_scored": True, "stage_ns": timings,
                "confidence_source": "Training query-level early-miss frequency; neural logits rank pages and are not calibrated probabilities",
                "torch_threads_observed": torch.get_num_threads()}

    def groups(self, records, min_confidence=.25):
        result, history, session = [], [], None
        for row in records:
            if row["session_id"] != session:
                history, session = [], row["session_id"]
            if history:
                x, prior, proposal, _ = self.features(history, min_confidence=min_confidence)
                targets = set(row["target_pages"])
                candidate_pages = [item["page"] for item in proposal["pages"]]
                result.append({"x": x, "prior": prior, "target": np.asarray([p in targets for p in candidate_pages], dtype=bool),
                               "pages": candidate_pages, "targets": targets, "query_id": row["query_id"]})
            history.append(self.state(row["template_id"], row["pages"]))
            history = history[-2:]
        return result

    def evaluate(self, groups, budget=16):
        hit, total, recalls = 0, 0, []
        self.model.eval()
        with torch.inference_mode():
            for group in groups:
                if group["x"] is None:
                    ordered = []
                else:
                    score = self.model(torch.from_numpy(group["x"])) + torch.from_numpy(group["prior"])
                    ordered = torch.argsort(score, descending=True, stable=True).numpy()[:budget]
                predicted = {group["pages"][i] for i in ordered}
                matched = len(predicted & group["targets"])
                hit, total = hit + matched, total + len(group["targets"])
                if group["targets"]:
                    recalls.append(matched / len(group["targets"]))
        return {"early_miss_recall_macro_pct": 100 * float(np.mean(recalls)) if recalls else 0.0,
                "early_miss_recall_micro_pct": 100 * hit / total if total else 0.0,
                "groups": len(groups), "eligible_groups": sum(bool(g["target"].any()) for g in groups), "targets": total}

    def fit(self, training, validation, *, seed=101, epochs=16, budget=16, min_confidence=.25):
        torch.manual_seed(seed)
        groups, val = self.groups(training, min_confidence), self.groups(validation, min_confidence)
        eligible = [group for group in groups if group["target"].any()]
        if not eligible or not val:
            raise ValueError("Insufficient query-level training/validation examples")
        self.model = MLPScorer(eligible[0]["x"].shape[-1], hidden=self.config.hidden)
        nn.init.zeros_(self.model.net[-1].weight); nn.init.zeros_(self.model.net[-1].bias)
        baseline = self.evaluate(val, budget)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed)
        best, best_epoch, best_score, stale, history = None, 0, -float("inf"), 0, []
        # Select a genuinely trained epoch on validation. The zero-residual
        # Markov checkpoint is reported separately, not relabeled as neural.
        for epoch in range(1, epochs + 1):
            self.model.train()
            losses = []
            for i in rng.permutation(len(eligible)):
                group = eligible[int(i)]
                logits = self.model(torch.from_numpy(group["x"])) + torch.from_numpy(group["prior"])
                positives = torch.from_numpy(group["target"])
                loss = -torch.log_softmax(logits, dim=0)[positives].mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5); optimizer.step()
                losses.append(float(loss.detach()))
            measured = self.evaluate(val, budget)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), **measured})
            score = measured["early_miss_recall_macro_pct"]
            if score > best_score + 1e-6:
                best, best_epoch, best_score, stale = copy.deepcopy(self.model.state_dict()), epoch, score, 0
            else:
                stale += 1
            if stale >= 5:
                break
        self.model.load_state_dict(best); self.model.eval()
        return {"seed": seed, "maximum_epochs": epochs, "selected_epoch": best_epoch, "epochs_run": len(history),
                "selected_residual_weight_norm": float(self.model.net[-1].weight.detach().norm()),
                "validation_zero_residual": baseline, "validation_selected": self.evaluate(val, budget),
                "selection": "Best trained epoch by validation early-miss recall at a fixed page budget; no evaluation outcomes used",
                "training_groups": len(groups), "eligible_training_groups": len(eligible), "history": history}

    def save(self, directory, *, fit, database_identity, metadata_path):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        self.prior.save(directory / "query_prior.json")
        payload = {"config": asdict(self.config), "candidate_limit": self.candidate_limit,
            "state_dict": self.model.state_dict(), "input_dim": self.model.net[0].in_features,
            "feature_index": self.index.to_json(), "counts": self.counts, "contexts": self.contexts,
            "database_identity": database_identity, "metadata_path": str(metadata_path), "fit": fit,
            "query_prior_sha256": hashlib.sha256((directory / "query_prior.json").read_bytes()).hexdigest()}
        torch.save(payload, directory / "idle_neural.pt")
        (directory / "training.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory, metadata, database_identity):
        directory = Path(directory)
        payload = torch.load(directory / "idle_neural.pt", map_location="cpu", weights_only=True)
        if payload["database_identity"] != database_identity:
            raise ValueError("Query-level neural checkpoint belongs to a different database")
        if hashlib.sha256((directory / "query_prior.json").read_bytes()).hexdigest() != payload["query_prior_sha256"]:
            raise ValueError("Query-level Markov prior checksum mismatch")
        result = cls(QueryMarkov.load(directory / "query_prior.json"), CandidateIndex.from_json(payload["feature_index"]),
                     payload["counts"], metadata, payload["contexts"], hidden=payload["config"]["hidden"],
                     no_semantics=payload["config"]["no_semantics"], candidate_limit=payload["candidate_limit"])
        result.model = MLPScorer(payload["input_dim"], hidden=result.config.hidden)
        result.model.load_state_dict(payload["state_dict"]); result.model.eval()
        return result
