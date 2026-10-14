from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def read_page_size(database_path: str | Path) -> int:
    with open(database_path, "rb") as f:
        header = f.read(100)
    if len(header) < 100 or header[:16] != b"SQLite format 3\x00":
        raise ValueError(f"Not a valid SQLite database file: {database_path}")
    value = int.from_bytes(header[16:18], "big")
    return 65536 if value == 1 else value


def _read_dbstat_rows(database_path: str | Path) -> list[tuple[Any, ...]]:
    """Read dbstat rows, preferring stdlib SQLite and falling back to APSW.

    Some Windows Python distributions compile their bundled SQLite without
    SQLITE_ENABLE_DBSTAT_VTAB, while APSW may ship a separate SQLite build with
    the feature enabled. Use APSW as a real metadata source in that case.
    """
    path = Path(database_path).resolve()
    stdlib_error: Exception | None = None
    con = None
    try:
        uri = f"file:{path.as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        return con.execute("SELECT name, path, pageno, pagetype FROM dbstat").fetchall()
    except Exception as exc:
        stdlib_error = exc
    finally:
        if con is not None:
            con.close()

    try:
        import apsw
        con_apsw = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            return list(con_apsw.execute("SELECT name, path, pageno, pagetype FROM dbstat"))
        finally:
            con_apsw.close()
    except Exception as apsw_error:
        raise RuntimeError(
            "SQLite dbstat is unavailable in both Python's sqlite3 and APSW. "
            "Install/use a SQLite build compiled with SQLITE_ENABLE_DBSTAT_VTAB. "
            f"sqlite3 error: {stdlib_error}; APSW error: {apsw_error}"
        ) from apsw_error


def dbstat_available(database_path: str | Path | None = None) -> dict[str, bool]:
    """Report dbstat support in stdlib sqlite3 and APSW independently.

    When no database path is supplied, probe temporary in-memory databases.
    """
    support = {"sqlite3": False, "apsw": False}
    if database_path is None:
        try:
            con = sqlite3.connect(":memory:")
            try:
                con.execute("SELECT count(*) FROM dbstat").fetchone()
                support["sqlite3"] = True
            finally:
                con.close()
        except Exception:
            pass
        try:
            import apsw
            con_apsw = apsw.Connection(":memory:")
            try:
                list(con_apsw.execute("SELECT count(*) FROM dbstat"))
                support["apsw"] = True
            finally:
                con_apsw.close()
        except Exception:
            pass
        return support

    path = Path(database_path).resolve()
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            con.execute("SELECT count(*) FROM dbstat").fetchone()
            support["sqlite3"] = True
        finally:
            con.close()
    except Exception:
        pass
    try:
        import apsw
        con_apsw = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            list(con_apsw.execute("SELECT count(*) FROM dbstat"))
            support["apsw"] = True
        finally:
            con_apsw.close()
    except Exception:
        pass
    return support


def build_page_map(database_path: str | Path) -> dict[int, dict[str, Any]]:
    """Return page-to-object metadata using dbstat where available.

    Page 1 may participate in sqlite_schema B-tree and also contains the file
    header. Pages not returned by dbstat are intentionally left unmapped.
    """
    rows = _read_dbstat_rows(database_path)
    result: dict[int, dict[str, Any]] = {}
    for name, path, pageno, pagetype in rows:
        path = path or "/"
        depth = len([segment for segment in path.split("/") if segment])
        result[int(pageno)] = {
            "page": int(pageno), "object": str(name), "path": str(path),
            "page_type": str(pagetype or "unknown").lower(), "depth": depth,
        }
    return result


def save_page_map(database_path: str | Path, output_path: str | Path) -> dict[int, dict[str, Any]]:
    mapping = build_page_map(database_path)
    normalized = {str(page): meta for page, meta in mapping.items()}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return mapping
