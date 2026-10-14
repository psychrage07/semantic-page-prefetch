from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from .metadata import dbstat_available

SCHEMA = """
PRAGMA page_size=4096;
PRAGMA journal_mode=DELETE;
PRAGMA foreign_keys=ON;
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    segment TEXT NOT NULL
);
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES categories(category_id),
    name TEXT NOT NULL,
    price_cents INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    created_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    total_cents INTEGER NOT NULL
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL
);
CREATE TABLE account_edges (
    src_node INTEGER NOT NULL,
    dst_node INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (src_node, dst_node)
);
"""
INDEXES = """
CREATE INDEX idx_orders_customer_created ON orders(customer_id, created_at DESC);
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_edges_dst ON account_edges(dst_node);
"""

REGIONS = ("north", "south", "east", "west", "central")
SEGMENTS = ("consumer", "small_business", "enterprise")
STATUSES = ("pending", "paid", "shipped", "delivered", "cancelled")


def generate_database(
    output_path: str | Path,
    *,
    customers: int = 10_000,
    products: int = 1_000,
    orders: int = 50_000,
    avg_items_per_order: int = 3,
    graph_nodes: int = 5_000,
    edges_per_node: int = 3,
    seed: int = 17,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a reproducible SQLite e-commerce database for experiments.

    Defaults are intended for development, not a 5-20 GB benchmark. Increase
    counts only after verifying available disk space and confirming the pipeline.
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Database already exists: {path}; pass overwrite=True to replace it")
        path.unlink()
    if graph_nodes < 2:
        raise ValueError("graph_nodes must be >= 2 because the graph traversal workload requires at least one edge")
    for key, value in {
        "customers": customers, "products": products, "orders": orders,
        "avg_items_per_order": avg_items_per_order, "graph_nodes": graph_nodes,
        "edges_per_node": edges_per_node,
    }.items():
        if value < 1:
            raise ValueError(f"{key} must be >= 1")

    rng = random.Random(seed)
    started = time.perf_counter()
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA page_size=4096")
        con.executescript(SCHEMA)
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("PRAGMA synchronous=NORMAL")

        con.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            ((i, f"customer_{i:07d}", rng.choice(REGIONS), rng.choice(SEGMENTS)) for i in range(1, customers + 1)),
        )
        con.executemany("INSERT INTO categories VALUES (?, ?)", ((i, f"category_{i:03d}") for i in range(1, 41)))
        con.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            ((i, rng.randint(1, 40), f"product_{i:06d}", rng.randint(199, 250_000)) for i in range(1, products + 1)),
        )
        con.commit()

        # Skewed customer choices make the working set non-uniform while keeping
        # IDs valid. All values and workload generation are deterministic per seed.
        order_batch: list[tuple[int, int, int, str, int]] = []
        item_batch: list[tuple[int, int, int, int, int]] = []
        item_id = 1
        base_timestamp = 1_700_000_000
        for order_id in range(1, orders + 1):
            if rng.random() < 0.72:
                customer_id = min(customers, max(1, int(rng.paretovariate(1.8))))
            else:
                customer_id = rng.randint(1, customers)
            created_at = base_timestamp + rng.randrange(365 * 24 * 3600)
            status = rng.choice(STATUSES)
            n_items = max(1, min(12, int(rng.expovariate(1 / max(1, avg_items_per_order))) + 1))
            total = 0
            rows_for_order = []
            for _ in range(n_items):
                product_id = rng.randint(1, products)
                qty = rng.randint(1, 5)
                price = rng.randint(199, 100_000)
                total += qty * price
                rows_for_order.append((item_id, order_id, product_id, qty, price))
                item_id += 1
            order_batch.append((order_id, customer_id, created_at, status, total))
            item_batch.extend(rows_for_order)
            if len(order_batch) >= 1_000:
                con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", order_batch)
                con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", item_batch)
                con.commit()
                order_batch.clear()
                item_batch.clear()
        if order_batch:
            con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", order_batch)
            con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", item_batch)
            con.commit()

        # Small, data-dependent graph workload stored in the same database.
        edge_batch = []
        for src in range(1, graph_nodes + 1):
            chosen: set[int] = set()
            for _ in range(min(edges_per_node, max(0, graph_nodes - 1))):
                dst = rng.randint(1, graph_nodes)
                while dst == src or dst in chosen:
                    dst = rng.randint(1, graph_nodes)
                chosen.add(dst)
                edge_batch.append((src, dst, round(rng.random(), 6)))
            if len(edge_batch) >= 10_000:
                con.executemany("INSERT OR IGNORE INTO account_edges VALUES (?, ?, ?)", edge_batch)
                con.commit()
                edge_batch.clear()
        if edge_batch:
            con.executemany("INSERT OR IGNORE INTO account_edges VALUES (?, ?, ?)", edge_batch)
        con.executescript(INDEXES)
        con.execute("ANALYZE")
        con.commit()

        counts = {
            table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("customers", "categories", "products", "orders", "order_items", "account_edges")
        }
        page_size = int(con.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(con.execute("PRAGMA page_count").fetchone()[0])
        # Probe both SQLite providers: on some Windows installs stdlib sqlite3
        # lacks dbstat while APSW's bundled SQLite exposes it.
        dbstat_support = dbstat_available(path)
    finally:
        con.close()

    metadata = {
        "database_path": str(path), "seed": seed, "page_size": page_size,
        "page_count": page_count, "estimated_database_bytes": path.stat().st_size,
        "table_rows": counts, "dbstat_available": any(dbstat_support.values()),
        "dbstat_support": dbstat_support,
        "generation_seconds": round(time.perf_counter() - started, 3),
        "profile": "development; scale rows only after validating the pipeline",
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
