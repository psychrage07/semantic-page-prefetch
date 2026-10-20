"""Static SQLite B-tree proposals from already-addressable page bytes.

The complete child list is parsed, never filtered by future trace membership.
SQLite file format: https://www.sqlite.org/fileformat2.html#b_tree_pages
Only a static read-only database is supported. Cached entries include the
file modification time, so replacing a snapshot does not reuse old pointers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def parse_children(raw: bytes, page: int, page_count: int) -> tuple[int, ...]:
    header = 100 if page == 1 else 0
    if len(raw) < header + 12 or raw[header] not in (0x02, 0x05):
        return ()
    cells = int.from_bytes(raw[header + 3:header + 5], "big")
    if header + 12 + 2 * cells > len(raw):
        raise ValueError("Invalid B-tree cell pointer array")
    children = []
    for i in range(cells):
        offset = int.from_bytes(raw[header + 12 + 2 * i:header + 14 + 2 * i], "big")
        if offset < header + 12 + 2 * cells or offset + 4 > len(raw):
            raise ValueError("Invalid B-tree cell offset")
        children.append(int.from_bytes(raw[offset:offset + 4], "big"))
    children.append(int.from_bytes(raw[header + 8:header + 12], "big"))
    if any(child < 1 or child > page_count or child == page for child in children):
        raise ValueError("Invalid child page number")
    return tuple(children)


@lru_cache(maxsize=32_768)
def _children(path: str, page: int, size: int, mtime_ns: int) -> tuple[int, ...]:
    with open(path, "rb") as handle:
        header = handle.read(100)
        if header[:16] != b"SQLite format 3\x00":
            raise ValueError("Invalid SQLite snapshot")
        page_size = int.from_bytes(header[16:18], "big")
        page_size = 65536 if page_size == 1 else page_size
        if page < 1 or page > size // page_size:
            return ()
        handle.seek((page - 1) * page_size)
        return parse_children(handle.read(page_size), page, size // page_size)


def children(database: str, page: int) -> tuple[int, ...]:
    path = Path(database).resolve()
    stat = path.stat()
    return _children(str(path), int(page), stat.st_size, stat.st_mtime_ns)
