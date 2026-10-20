"""Append-only run ledger for v2.1 (Requirement C: record every configuration).

Every configuration attempt — successful, failed, interrupted, deliberately
excluded — is appended to ``runs/<study>/run_log.jsonl`` with an identity hash,
the reason for inclusion or exclusion, and outputs. Entries are never deleted
or silently omitted; a re-run with the same configuration_appended id appends a
new row with ``retry_of`` set, rather than rewriting history.

The ledger's integrity is protected two ways:

1. ``record_run`` appends one JSON line and fsyncs.
2. ``verify_ledger`` recomputes the per-row sha256 chain and confirms the
   ``prev_hash`` pointers, so any silent deletion or edit of a middle row is
   detectable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def source_hash() -> str:
    """Hash source contents, including uncommitted changes, without generated data."""
    backend = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted([*backend.joinpath("pagewise").glob("*.py"),
                    *backend.joinpath("scripts").glob("*.py")])
    for source in paths:
        digest.update(source.relative_to(backend).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_run(log_path: str | Path, config: dict[str, Any], *, status: str,
               reason: str = "", seed: int | None = None, outputs: list[str] | None = None,
               metrics: dict[str, Any] | None = None, error: str | None = None,
               retry_of: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one configuration attempt. Returns the stored row (with hashes)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = _last_hash(path)
    row: dict[str, Any] = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": config_hash(config),
        "config": config,
        "status": status,          # completed | failed | interrupted | excluded | exploratory
        "reason": reason,          # why included / excluded / aborted
        "seed": seed,
        "outputs": outputs or [],
        "metrics": metrics or {},
        "error": error,
        "retry_of": retry_of,
        "host": platform.platform(),
        "source_sha256": source_hash(),
        "prev_hash": prev_hash,
    }
    if extra:
        row.update(extra)
    body = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    row["row_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    return row


def read_ledger(log_path: str | Path) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_ledger(log_path: str | Path) -> dict[str, Any]:
    """Recompute the hash chain; returns ok=False with the first broken row."""
    rows = read_ledger(log_path)
    expected_prev = GENESIS
    for index, row in enumerate(rows):
        stored = row.get("row_sha256")
        body = {k: v for k, v in row.items() if k != "row_sha256"}
        recomputed = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if row.get("prev_hash") != expected_prev:
            return {"ok": False, "rows": len(rows), "first_broken_row": index,
                    "cause": "prev_hash pointer mismatch (row deleted or inserted)"}
        if stored != recomputed:
            return {"ok": False, "rows": len(rows), "first_broken_row": index,
                    "cause": "row content edited after append"}
        expected_prev = stored
    return {"ok": True, "rows": len(rows)}


def summarize_ledger(log_path: str | Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in read_ledger(log_path):
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"statuses": counts, "total": sum(counts.values()), "integrity": verify_ledger(log_path)}


def _last_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    if not last:
        return GENESIS
    return json.loads(last).get("row_sha256", GENESIS)
