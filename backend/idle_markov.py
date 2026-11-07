"""Training-only query-to-next-query page-set prediction; independent of frozen models."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


class QueryMarkov:
    def __init__(self, order=2, signal="template"):
        if order not in (1, 2) or signal not in ("template", "footprint"):
            raise ValueError("Use order 1/2 and template/footprint signal")
        self.order, self.signal = order, signal
        self.counts, self.support = {}, {}
        self.global_counts, self.training_queries = Counter(), 0

    def state(self, template, pages):
        if self.signal == "template":
            return str(template)
        unique = sorted(set(int(page) for page in pages))
        return hashlib.sha256(json.dumps(unique, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def key(history):
        return json.dumps(list(history), separators=(",", ":"))

    def fit(self, records):
        counts, support = defaultdict(Counter), Counter()
        global_counts = Counter()
        history, session = [], None
        for row in records:
            if row["session_id"] != session:
                history, session = [], row["session_id"]
            pages = set(int(page) for page in row["pages"])
            targets = set(int(page) for page in row.get("target_pages", pages))
            global_counts.update(targets)
            for length in range(1, min(2, len(history)) + 1):
                key = self.key(history[-length:])
                counts[key].update(targets)
                support[key] += 1
            history.append(self.state(row["template_id"], pages))
        self.counts = {key: dict(value) for key, value in counts.items()}
        self.support, self.global_counts = dict(support), global_counts
        self.training_queries = len(records)
        return self

    def predict(self, completed_history, *, max_pages=16, min_confidence=0.25, exclude_pages=()):
        if max_pages < 1 or not 0 <= min_confidence <= 1:
            raise ValueError("Invalid prediction admission settings")
        counts, support, used = self.global_counts, self.training_queries, 0
        for length in range(min(self.order, len(completed_history)), 0, -1):
            key = self.key(completed_history[-length:])
            if self.support.get(key):
                counts, support, used = self.counts[key], self.support[key], length
                break
        # Probability is the training frequency of a page in the next query's
        # unique page set, not a softmax or a prediction correctness percentage.
        excluded = set(exclude_pages)
        ranked = sorted(((int(page), count / max(1, support)) for page, count in counts.items() if int(page) not in excluded),
                        key=lambda pair: (-pair[1], pair[0]))
        return {"order_used": used, "support": support,
                "pages": [{"page": page, "confidence": confidence} for page, confidence in ranked
                          if confidence >= min_confidence][:max_pages]}

    def save(self, path):
        payload = {"schema_version": 1, "order": self.order, "signal": self.signal,
                   "counts": self.counts, "support": self.support,
                   "global_counts": dict(self.global_counts), "training_queries": self.training_queries}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(value["order"], value["signal"])
        model.counts = {key: {int(p): n for p, n in counts.items()} for key, counts in value["counts"].items()}
        model.support = value["support"]
        model.global_counts = Counter({int(p): n for p, n in value["global_counts"].items()})
        model.training_queries = value["training_queries"]
        return model
