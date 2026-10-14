import sqlite3
import sys
from types import SimpleNamespace
from pathlib import Path

from pagewise import metadata


def test_build_page_map_falls_back_to_apsw_when_stdlib_sqlite_lacks_dbstat(tmp_path: Path, monkeypatch):
    db = tmp_path / "fallback.sqlite"
    db.write_bytes(b"placeholder")
    stdlib_connect = sqlite3.connect

    def connect_without_dbstat(database, *args, **kwargs):
        if isinstance(database, str) and "mode=ro" in database:
            raise sqlite3.OperationalError("no such table: dbstat")
        return stdlib_connect(database, *args, **kwargs)

    rows = [("orders", "/1/", 2, "leaf"), ("idx_orders", "/", 3, "internal")]

    class FakeApswConnection:
        def __init__(self, path, flags=None):
            self.path = path
            self.flags = flags

        def execute(self, sql):
            assert "FROM dbstat" in sql
            return iter(rows)

        def close(self):
            pass

    fake_apsw = SimpleNamespace(
        Connection=FakeApswConnection,
        SQLITE_OPEN_READONLY=1,
    )
    monkeypatch.setattr(metadata.sqlite3, "connect", connect_without_dbstat)
    monkeypatch.setitem(sys.modules, "apsw", fake_apsw)

    page_map = metadata.build_page_map(db)

    assert page_map[2]["object"] == "orders"
    assert page_map[2]["page_type"] == "leaf"
    assert page_map[2]["depth"] == 1
    assert page_map[3]["object"] == "idx_orders"
