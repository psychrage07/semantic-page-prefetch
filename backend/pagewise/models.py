from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import load_jsonl

PAGE_TYPES = ("unknown", "leaf", "interior", "internal", "overflow")
PLAN_CLASSES = ("unknown", "search_index", "search_other", "scan", "join", "recursive")
WORKLOAD_FAMILIES = ("unknown", "point_lookup", "indexed_join", "range_scan", "recursive_traversal", "indexed_lookup")
HISTORY_LENGTH = 8
HISTORY_FEATURE_DIM = HISTORY_LENGTH * 2 + 3
SEMANTIC_FEATURE_GROUPS = {
    "page_type": (0, len(PAGE_TYPES)),
    "depth": (len(PAGE_TYPES), len(PAGE_TYPES) + 1),
    "object": (len(PAGE_TYPES) + 1, len(PAGE_TYPES) + 1 + 32),
    "plan": (len(PAGE_TYPES) + 1 + 32, len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES)),
    "template": (len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES), len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16),
    "workload": (len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16, len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16 + len(WORKLOAD_FAMILIES)),
    "position": (len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16 + len(WORKLOAD_FAMILIES), len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16 + len(WORKLOAD_FAMILIES) + 1),
}
SEMANTIC_ABLATION_ALIASES = {
    "page_metadata": ("page_type", "depth", "object"),
    "query_context": ("plan", "template", "workload", "position"),
}


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def semantic_page_features(page: int, metadata: dict[str, Any], context: dict[str, Any]) -> list[float]:
    page_meta = metadata.get(str(page), metadata.get(page, {})) or {}
    page_type = str(page_meta.get("page_type", "unknown")).lower()
    if page_type == "internal":
        page_type = "interior"
    try:
        type_index = PAGE_TYPES.index(page_type)
    except ValueError:
        type_index = 0
    depth = max(0, min(32, int(page_meta.get("depth", 0) or 0))) / 32.0
    object_name = str(page_meta.get("object", "unknown"))
    object_hash = one_hot(stable_bucket(object_name, 32), 32)
    plan = str(context.get("plan_class", "unknown"))
    plan_vec = one_hot(PLAN_CLASSES.index(plan) if plan in PLAN_CLASSES else 0, len(PLAN_CLASSES))
    template = str(context.get("template_id", "unknown"))
    template_vec = one_hot(stable_bucket(template, 16), 16)
    family = str(context.get("workload_family", "unknown"))
    family_vec = one_hot(WORKLOAD_FAMILIES.index(family) if family in WORKLOAD_FAMILIES else 0, len(WORKLOAD_FAMILIES))
    # Query position is the number of main-database page requests observed so far,
    # normalized and capped to a bounded range. It is known at inference time.
    position = max(0.0, min(1.0, float(context.get("position_in_query", 0.0) or 0.0)))
    return one_hot(type_index, len(PAGE_TYPES)) + [depth] + object_hash + plan_vec + template_vec + family_vec + [position]


def normalize_semantic_ablation_groups(groups: Iterable[str] | None) -> set[str]:
    disabled: set[str] = set()
    for group in groups or []:
        name = str(group).strip()
        if not name:
            continue
        if name in SEMANTIC_ABLATION_ALIASES:
            disabled.update(SEMANTIC_ABLATION_ALIASES[name])
        elif name in SEMANTIC_FEATURE_GROUPS:
            disabled.add(name)
        else:
            valid = sorted([*SEMANTIC_FEATURE_GROUPS, *SEMANTIC_ABLATION_ALIASES])
            raise ValueError(f"Unknown semantic ablation group {name!r}; expected one of {valid}")
    return disabled


def apply_semantic_ablation(features: list[float], groups: Iterable[str] | None) -> list[float]:
    disabled = normalize_semantic_ablation_groups(groups)
    if not disabled:
        return features
    out = list(features)
    for group in disabled:
        start, end = SEMANTIC_FEATURE_GROUPS[group]
        for idx in range(start, end):
            out[idx] = 0.0
    return out


def feature_vector(history: list[int], candidate: int, context: dict[str, Any], metadata: dict[str, Any], max_page: int,
                   semantic: bool, disabled_semantic_groups: Iterable[str] | None = None) -> np.ndarray:
    denom = max(1, max_page)
    hist = [int(p) for p in history[-HISTORY_LENGTH:]]
    pad = [0] * (HISTORY_LENGTH - len(hist))
    hist = pad + hist
    norm_hist = [p / denom for p in hist]
    deltas = [0.0] * HISTORY_LENGTH
    for i in range(1, HISTORY_LENGTH):
        if hist[i] and hist[i - 1]:
            deltas[i] = max(-1.0, min(1.0, (hist[i] - hist[i - 1]) / denom))
    last = history[-1] if history else 0
    common = norm_hist + deltas + [candidate / denom, max(-1.0, min(1.0, (candidate - last) / denom)), float(candidate == last)]
    if semantic:
        common += apply_semantic_ablation(semantic_page_features(candidate, metadata, context), disabled_semantic_groups)
    return np.asarray(common, dtype=np.float32)


@dataclass
class CandidateIndex:
    successors: dict[int, list[int]]
    second_order: dict[str, list[int]]
    popular: list[int]
    max_page: int
    max_candidates: int = 32
    page_map_sha256: str = ""

    @classmethod
    def build(cls, sequences: list[dict[str, Any]], max_candidates: int = 32) -> "CandidateIndex":
        successors: dict[int, Counter[int]] = defaultdict(Counter)
        second: dict[str, Counter[int]] = defaultdict(Counter)
        popular: Counter[int] = Counter()
        max_page = 1
        for seq in sequences:
            pages = [int(p) for p in seq["pages"]]
            for p in pages:
                popular[p] += 1
                max_page = max(max_page, p)
            for i in range(len(pages) - 1):
                a, b = pages[i], pages[i + 1]
                successors[a][b] += 1
                if i >= 1:
                    second[f"{pages[i-1]}:{a}"][b] += 1
        return cls(
            successors={k: [p for p, _ in v.most_common(max_candidates)] for k, v in successors.items()},
            second_order={k: [p for p, _ in v.most_common(max_candidates)] for k, v in second.items()},
            popular=[p for p, _ in popular.most_common(max_candidates)],
            max_page=max_page,
            max_candidates=max_candidates,
        )

    def candidates(self, history: list[int]) -> list[int]:
        if not history:
            base = list(self.popular)
        else:
            base = list(self.successors.get(history[-1], []))
            if len(history) >= 2:
                for p in self.second_order.get(f"{history[-2]}:{history[-1]}", []):
                    if p not in base:
                        base.append(p)
        if history:
            next_page = history[-1] + 1
            if 1 <= next_page <= self.max_page and next_page not in base:
                base.append(next_page)
        for p in self.popular:
            if p not in base:
                base.append(p)
            if len(base) >= self.max_candidates:
                break
        return base[: self.max_candidates]

    def to_json(self) -> dict[str, Any]:
        return {
            "successors": {str(k): v for k, v in self.successors.items()},
            "second_order": self.second_order,
            "popular": self.popular,
            "max_page": self.max_page,
            "max_candidates": self.max_candidates,
            "page_map_sha256": self.page_map_sha256,
            "note": "Candidates are generated from training sequences only; pages absent from the candidate set cannot be predicted.",
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CandidateIndex":
        return cls(
            successors={int(k): [int(p) for p in v] for k, v in value["successors"].items()},
            second_order={str(k): [int(p) for p in v] for k, v in value["second_order"].items()},
            popular=[int(p) for p in value["popular"]],
            max_page=int(value["max_page"]), max_candidates=int(value.get("max_candidates", 32)),
            page_map_sha256=str(value.get("page_map_sha256", "")),
        )


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_sequences(sequences: list[dict[str, Any]], seed: int = 7, train_fraction: float = 0.70, val_fraction: float = 0.15):
    ids = [i for i, s in enumerate(sequences) if len(s.get("pages", [])) >= 2]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * train_fraction)) if n else 0
    n_val = max(0, int(n * val_fraction))
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return ([sequences[i] for i in ids if i in train_ids], [sequences[i] for i in ids if i in val_ids], [sequences[i] for i in ids if i in test_ids])


def _training_pairs(sequences: list[dict[str, Any]], index: CandidateIndex, metadata: dict[str, Any], semantic: bool,
                    rng: random.Random, negatives_per_positive: int = 16,
                    disabled_semantic_groups: Iterable[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for seq in sequences:
        pages = [int(p) for p in seq["pages"]]
        base_context = {"template_id": seq.get("template_id"), "workload_family": seq.get("workload_family"), "plan_class": seq.get("plan_class")}
        for i in range(1, len(pages)):
            history, target = pages[:i][-HISTORY_LENGTH:], pages[i]
            context = {**base_context, "position_in_query": min(i, 512) / 512.0}
            candidates = index.candidates(history)
            if target not in candidates:
                # Do not train on a candidate the inference system could not generate.
                continue
            negatives = [p for p in candidates if p != target]
            if len(negatives) > negatives_per_positive:
                negatives = rng.sample(negatives, negatives_per_positive)
            selected = [target] + negatives
            for candidate in selected:
                xs.append(feature_vector(history, candidate, context, metadata, index.max_page, semantic, disabled_semantic_groups))
                ys.append(1.0 if candidate == target else 0.0)
    if not xs:
        dims = 19 + (len(PAGE_TYPES) + 1 + 32 + len(PLAN_CLASSES) + 16 + len(WORKLOAD_FAMILIES) + 1 if semantic else 0)
        return np.empty((0, dims), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def _fit(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, seed: int,
         epochs: int = 20, batch_size: int = 512, hidden: int = 128, learning_rate: float = 0.001) -> tuple[MLPScorer, dict[str, Any]]:
    if len(x_train) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError("Insufficient ranking training samples. Collect more query executions or inspect the VFS trace.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPScorer(x_train.shape[1], hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    use_val = len(x_val) > 0 and len(np.unique(y_val)) > 1
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = 4
    no_improve = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(xb)
            train_n += len(xb)
        model.eval()
        with torch.no_grad():
            if use_val:
                val_logits = model(torch.from_numpy(x_val))
                val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val)).item())
            else:
                val_loss = train_loss_sum / max(1, train_n)
        history.append({"epoch": epoch, "train_loss": train_loss_sum / max(1, train_n), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}


def page_map_digest(page_map: dict[str, Any]) -> str:
    normalized = {str(k): page_map[k] for k in sorted(page_map, key=lambda value: int(value))}
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def train_models(sequences_path: str | Path, page_map_path: str | Path, output_dir: str | Path, *, seed: int = 7,
                 split_seed: int | None = None, epochs: int = 20, max_candidates: int = 32, hidden: int = 128,
                 semantic_ablations: dict[str, list[str]] | None = None) -> dict[str, Any]:
    sequences = load_jsonl(sequences_path)
    page_map = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    split_seed = seed if split_seed is None else split_seed
    train, val, test = split_sequences(sequences, seed=split_seed)
    if len(train) < 3 or len(test) < 1:
        raise ValueError(f"Need more distinct query executions for a train/validation/test split; got train={len(train)}, val={len(val)}, test={len(test)}")
    index = CandidateIndex.build(train, max_candidates=max_candidates)
    index.page_map_sha256 = page_map_digest(page_map)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng_a = random.Random(seed)
    results: dict[str, Any] = {
        "seed": seed, "split_seed": split_seed,
        "counts": {"sequences_total": len(sequences), "train_executions": len(train), "validation_executions": len(val), "test_executions": len(test)},
        "candidate_count": max_candidates, "models": {},
        "page_map_sha256": index.page_map_sha256,
        "split_note": "Split by query execution; candidate dictionary built from training executions only.",
    }
    (out / "candidate_index.json").write_text(json.dumps(index.to_json(), indent=2), encoding="utf-8")
    (out / "train_sequences.jsonl").write_text("\n".join(json.dumps(x) for x in train) + "\n", encoding="utf-8")
    (out / "validation_sequences.jsonl").write_text("\n".join(json.dumps(x) for x in val) + ("\n" if val else ""), encoding="utf-8")
    (out / "test_sequences.jsonl").write_text("\n".join(json.dumps(x) for x in test) + ("\n" if test else ""), encoding="utf-8")
    normalized_ablations = {
        label: sorted(normalize_semantic_ablation_groups(groups))
        for label, groups in (semantic_ablations or {}).items()
    }
    training_specs: list[tuple[str, bool, int, list[str]]] = [
        ("model_a", False, 101, []),
        ("model_b", True, 211, []),
    ]
    for offset, (label, groups) in enumerate(normalized_ablations.items(), start=1):
        safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)
        training_specs.append((f"model_b_retrained_{safe_label}", True, 300 + offset * 37, groups))

    for name, semantic, salt, disabled_groups in training_specs:
        x_train, y_train = _training_pairs(train, index, page_map, semantic, random.Random(seed + salt),
                                           disabled_semantic_groups=disabled_groups)
        x_val, y_val = _training_pairs(
            val, index, page_map, semantic, random.Random(seed + salt + 1),
            disabled_semantic_groups=disabled_groups,
        ) if val else (np.empty((0, x_train.shape[1]), np.float32), np.empty((0,), np.float32))
        model, fit_report = _fit(x_train, y_train, x_val, y_val, seed=seed + salt, epochs=epochs, hidden=hidden)
        checkpoint = out / f"{name}.pt"
        torch.save({
            "state_dict": model.state_dict(), "input_dim": x_train.shape[1],
            "hidden": hidden, "semantic": semantic, "seed": seed + salt,
            "disabled_semantic_groups": disabled_groups,
        }, checkpoint)
        results["models"][name] = {
            "checkpoint": checkpoint.name, "input_dim": int(x_train.shape[1]),
            "disabled_semantic_groups": disabled_groups,
            "train_candidate_pairs": int(len(x_train)), "validation_candidate_pairs": int(len(x_val)),
            **fit_report,
        }
    results["feature_schema"] = {
        "history_only": {"history_pages": HISTORY_LENGTH, "history_deltas": HISTORY_LENGTH, "candidate_page": 1, "candidate_delta": 1, "candidate_same_as_latest": 1},
        "semantic_additions": {"candidate_page_type_one_hot": len(PAGE_TYPES), "candidate_depth": 1, "candidate_object_hash_one_hot": 32, "plan_class_one_hot": len(PLAN_CLASSES), "template_hash_one_hot": 16, "workload_family_one_hot": len(WORKLOAD_FAMILIES), "query_position_normalized": 1},
        "semantic_ablation_groups": {
            "groups": {name: {"start_in_semantic_block": start, "end_in_semantic_block": end} for name, (start, end) in SEMANTIC_FEATURE_GROUPS.items()},
            "aliases": {name: list(groups) for name, groups in SEMANTIC_ABLATION_ALIASES.items()},
        },
    }
    (out / "training_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def load_model(checkpoint_path: str | Path) -> tuple[MLPScorer, bool]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MLPScorer(int(payload["input_dim"]), int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.disabled_semantic_groups = list(payload.get("disabled_semantic_groups", []))
    return model, bool(payload["semantic"])


def score_candidates(model: MLPScorer, history: list[int], candidates: list[int], context: dict[str, Any],
                     page_map: dict[str, Any], max_page: int, semantic: bool,
                     disabled_semantic_groups: Iterable[str] | None = None) -> list[tuple[int, float]]:
    if not candidates:
        return []
    x = np.stack([
        feature_vector(history, p, context, page_map, max_page, semantic, disabled_semantic_groups)
        for p in candidates
    ]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        scores = torch.sigmoid(logits).cpu().numpy()
    return sorted([(int(p), float(s)) for p, s in zip(candidates, scores)], key=lambda item: item[1], reverse=True)
