from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def build_sequences(trace_path: str | Path, page_map_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    """Turn VFS read byte-ranges into per-query page-request sequences.

    This records pages represented by reads that reached VFS xRead. It is not a
    full logical page-reference trace and does not prove physical disk I/O.
    """
    page_map_raw = json.loads(Path(page_map_path).read_text(encoding="utf-8"))
    page_map = {int(page): meta for page, meta in page_map_raw.items()}
    trace_events = load_jsonl(trace_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in trace_events:
        if event.get("file_type") != "main_db" or event.get("result") != "success":
            continue
        query_id = event.get("query_id")
        if not query_id:
            # Setup reads, PRAGMAs, and schema initialization without a workload
            # context are excluded from supervised query examples.
            continue
        first_page = event.get("first_page")
        last_page = event.get("last_page")
        if first_page is None or last_page is None:
            continue
        for page in range(int(first_page), int(last_page) + 1):
            row = dict(event)
            row["page"] = page
            row["page_metadata"] = page_map.get(page, {
                "page": page, "object": "unknown", "page_type": "unknown", "depth": 0, "path": "?"
            })
            grouped[str(query_id)].append(row)

    sequences: list[dict[str, Any]] = []
    for query_id, events in grouped.items():
        events.sort(key=lambda row: (int(row.get("seq", 0)), int(row.get("page", 0))))
        first = events[0]
        sequences.append({
            "query_id": query_id,
            "template_id": first.get("template_id") or "unknown",
            "workload_family": first.get("workload_family") or "unknown",
            "plan_class": first.get("plan_class") or "unknown",
            "params": first.get("params"),
            "pages": [int(event["page"]) for event in events],
            "event_count": len(events),
            "page_metadata": {str(page): page_map.get(page, {"page": page, "object": "unknown", "page_type": "unknown", "depth": 0}) for page in set(int(e["page"]) for e in events)},
        })
    sequences.sort(key=lambda row: row["query_id"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sequences:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sequences
