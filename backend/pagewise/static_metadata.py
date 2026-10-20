"""Compact, trace-independent metadata for a static database snapshot.

The whole dbstat relation is read once, before model selection. No future trace
membership controls which candidates have metadata. Four bytes per database
page retain type, depth and object identity; the compressed artifact is reused.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

PAGE_TYPES = ("unknown", "leaf", "internal", "overflow")


def snapshot_identity(database: str | Path) -> str:
    path = Path(database)
    with path.open("rb") as handle:
        header = handle.read(100)
    return hashlib.sha256(header + path.stat().st_size.to_bytes(8, "big")).hexdigest()


def build_static_metadata(database: str | Path, output: str | Path) -> dict:
    import apsw
    database, output = Path(database).resolve(), Path(output)
    if output.exists():
        existing = StaticMetadata(output)
        if existing.snapshot_identity != snapshot_identity(database):
            raise ValueError("Static metadata belongs to a different snapshot")
        return {"pages": len(existing.types) - 1, "path": str(output), "reused": True}
    con = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        page_count = int(next(iter(con.execute("PRAGMA page_count")))[0])
        types = np.zeros(page_count + 1, np.uint8)
        depths = np.zeros(page_count + 1, np.uint8)
        objects = np.zeros(page_count + 1, np.uint16)
        names, object_ids = ["unknown"], {"unknown": 0}
        for name, path, page, kind in con.execute("SELECT name,path,pageno,pagetype FROM dbstat"):
            if name not in object_ids:
                object_ids[name] = len(names); names.append(name)
            types[page] = PAGE_TYPES.index(kind) if kind in PAGE_TYPES else 0
            depths[page] = min(32, len([p for p in path.split("/") if p]))
            objects[page] = object_ids[name]
    finally:
        con.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, types=types, depths=depths, objects=objects, names=np.asarray(names),
                        snapshot_identity=np.asarray(snapshot_identity(database)))
    return {"pages": page_count, "objects": names, "bytes_uncompressed": types.nbytes + depths.nbytes + objects.nbytes,
            "bytes_compressed": output.stat().st_size, "path": str(output)}


class StaticMetadata:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as payload:
            self.types, self.depths, self.objects = payload["types"], payload["depths"], payload["objects"]
            self.names = payload["names"].tolist()
            self.snapshot_identity = str(payload["snapshot_identity"])

    def _indices(self, pages):
        ids = np.asarray(pages, dtype=np.int64)
        ids = np.where((ids > 0) & (ids < len(self.types)), ids, 0)
        return np.where(self.types[ids] > 0, ids, 0)

    def semantic_matrix(self, pages, context):
        """Exactly the existing semantic vector, broadcast without Python row loops."""
        from . import models as M
        ids = self._indices(pages)
        width = len(M.PAGE_TYPES) + 1 + 32 + len(M.PLAN_CLASSES) + 16 + len(M.WORKLOAD_FAMILIES) + 1
        matrix = np.zeros((len(ids), width), dtype=np.float32)
        rows = np.arange(len(ids))
        # M.semantic_page_features maps internal -> interior.
        kind = np.asarray([0, 1, 2, 4])[self.types[ids]]
        matrix[rows, kind] = 1
        matrix[:, len(M.PAGE_TYPES)] = self.depths[ids].astype(float) / 32
        buckets = np.asarray([M.stable_bucket(str(name), 32) for name in self.names])
        matrix[rows, len(M.PAGE_TYPES) + 1 + buckets[self.objects[ids]]] = 1
        start = len(M.PAGE_TYPES) + 1 + 32
        matrix[:, start:] = M.semantic_page_features(0, {}, context)[start:]
        return matrix

    def relative_features(self, previous, pages):
        ids = self._indices([previous, *pages])
        object_ids = np.where(self.types[ids] > 0, self.objects[ids].astype(int), -1)
        return np.column_stack([object_ids[1:] == object_ids[0], self.types[ids[1:]] == self.types[ids[0]],
                                (self.depths[ids[1:]].astype(float) - float(self.depths[ids[0]])) / 32]).astype(np.float32)

    def get(self, page, default=None):
        value = self._get(int(page))
        return default if value is None else value

    @lru_cache(maxsize=65_536)
    def _get(self, page):
        if page < 1 or page >= len(self.types) or self.types[page] == 0:
            return None
        return {"page": page, "page_type": PAGE_TYPES[int(self.types[page])],
                "depth": int(self.depths[page]), "object": self.names[int(self.objects[page])]}
