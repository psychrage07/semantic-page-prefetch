from __future__ import annotations

import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    template_id: str
    family: str
    sql: str
    params: tuple[Any, ...]

    def serializable(self) -> dict[str, Any]:
        row = asdict(self)
        row["params"] = list(self.params)
        # Do not save unbounded or sensitive parameter material in real workloads.
        return row


TEMPLATES: dict[str, dict[str, str]] = {
    "customer_orders": {
        "family": "point_lookup",
        "sql": "SELECT order_id, customer_id, created_at, status, total_cents FROM orders WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
    },
    "order_items_join": {
        "family": "indexed_join",
        "sql": "SELECT o.order_id, oi.item_id, p.product_id, p.category_id, oi.quantity FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id JOIN products AS p ON p.product_id = oi.product_id WHERE o.customer_id = ? ORDER BY o.created_at DESC LIMIT ?",
    },
    "product_sales": {
        "family": "indexed_join",
        "sql": "SELECT p.product_id, p.category_id, oi.order_id, oi.quantity FROM products AS p JOIN order_items AS oi ON oi.product_id = p.product_id WHERE p.product_id = ? LIMIT ?",
    },
    "created_range": {
        "family": "range_scan",
        "sql": "SELECT order_id, customer_id, created_at, total_cents FROM orders WHERE created_at BETWEEN ? AND ? ORDER BY created_at LIMIT ?",
    },
    "graph_walk": {
        "family": "recursive_traversal",
        "sql": "WITH RECURSIVE walk(node, depth) AS (SELECT ?, 0 UNION ALL SELECT e.dst_node, w.depth + 1 FROM walk AS w JOIN account_edges AS e ON e.src_node = w.node WHERE w.depth < ?) SELECT node, depth FROM walk LIMIT ?",
    },
    "customer_profile": {
        "family": "indexed_lookup",
        "sql": "SELECT c.customer_id, c.region, c.segment, COUNT(o.order_id) AS order_count FROM customers AS c LEFT JOIN orders AS o ON o.customer_id = c.customer_id WHERE c.customer_id = ? GROUP BY c.customer_id",
    },
}


def generate_workload(db_path: str | Path, count: int = 500, seed: int = 1234, *, mixed: bool = True) -> list[QueryCase]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    con = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        max_customer = int(con.execute("SELECT MAX(customer_id) FROM customers").fetchone()[0])
        max_product = int(con.execute("SELECT MAX(product_id) FROM products").fetchone()[0])
        max_node = int(con.execute("SELECT MAX(src_node) FROM account_edges").fetchone()[0])
        min_ts, max_ts = con.execute("SELECT MIN(created_at), MAX(created_at) FROM orders").fetchone()
    finally:
        con.close()

    weights = [
        ("customer_orders", 0.28), ("order_items_join", 0.24),
        ("product_sales", 0.16), ("created_range", 0.14),
        ("graph_walk", 0.10), ("customer_profile", 0.08),
    ]
    ids = [w[0] for w in weights]
    ws = [w[1] for w in weights]
    cases: list[QueryCase] = []
    for i in range(1, count + 1):
        template_id = rng.choices(ids, weights=ws, k=1)[0] if mixed else ids[(i - 1) % len(ids)]
        if template_id == "customer_orders":
            # Skew some accesses toward lower IDs to model a hot subset.
            customer = min(max_customer, max(1, int(rng.paretovariate(1.8)))) if rng.random() < 0.7 else rng.randint(1, max_customer)
            params = (customer, rng.randint(5, 30))
        elif template_id == "order_items_join":
            customer = min(max_customer, max(1, int(rng.paretovariate(1.8)))) if rng.random() < 0.7 else rng.randint(1, max_customer)
            params = (customer, rng.randint(10, 50))
        elif template_id == "product_sales":
            product = rng.randint(1, max_product)
            params = (product, rng.randint(10, 100))
        elif template_id == "created_range":
            width = rng.randint(600, 14_400)
            start = rng.randint(int(min_ts), max(int(min_ts), int(max_ts) - width))
            params = (start, start + width, rng.randint(20, 200))
        elif template_id == "graph_walk":
            params = (rng.randint(1, max_node), rng.randint(2, 5), rng.randint(30, 200))
        else:
            customer = min(max_customer, max(1, int(rng.paretovariate(1.8)))) if rng.random() < 0.7 else rng.randint(1, max_customer)
            params = (customer,)
        spec = TEMPLATES[template_id]
        cases.append(QueryCase(f"Q{i:07d}", template_id, spec["family"], spec["sql"], tuple(params)))
    return cases


def explain_query_plan(db_path: str | Path, case: QueryCase) -> list[str]:
    """Get query-plan details using APSW when available, to match runtime SQLite.

    Falling back to stdlib sqlite3 supports offline planning/tests, but experiment
    metadata should record both SQLite versions if they differ.
    """
    try:
        import apsw
    except ImportError:
        apsw = None
    if apsw is not None:
        con = apsw.Connection(str(Path(db_path).resolve()), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            con.execute("PRAGMA mmap_size=0")
            return [str(row[3]) for row in con.execute("EXPLAIN QUERY PLAN " + case.sql, case.params)]
        finally:
            con.close()
    con = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA mmap_size=0")
        rows = con.execute("EXPLAIN QUERY PLAN " + case.sql, case.params).fetchall()
        return [str(row[3]) for row in rows]
    finally:
        con.close()


def classify_plan(details: list[str]) -> str:
    text = " ".join(details).lower()
    if "recursive" in text or "co-routine walk" in text:
        return "recursive"
    if "join" in text or "nested" in text or sum(("search " in item.lower() or "scan " in item.lower()) for item in details) >= 2:
        return "join"
    if "index" in text or "primary key" in text:
        return "search_index"
    if "scan" in text:
        return "scan"
    if "search" in text:
        return "search_other"
    return "other"
