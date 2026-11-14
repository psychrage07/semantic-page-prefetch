"""Read-only SDSS workloads adapted from the existing isolated SDSS experiment.

Same six SQL templates and SHA256 parameter partitions as the original local
study. These are generated SQLite queries, not SeLeP's original workload logs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sqlite3

from pagewise.workloads import QueryCase

TEMPLATES = [
    ("sdss_object", "point_lookup", "SELECT objID,ra,dec,u,g,r,i,z FROM PhotoObj WHERE objID=?"),
    ("sdss_magnitudes", "range_scan", "SELECT objID,u,g,r,i,z FROM PhotoObj WHERE r BETWEEN ? AND ? ORDER BY r,objID LIMIT ?"),
    ("sdss_sky_window", "range_scan", "SELECT objID,ra,dec,r,type FROM PhotoObj WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ? ORDER BY ra,objID LIMIT ?"),
    ("sdss_plate_join", "indexed_join", "SELECT s.specObjID,p.objID,p.ra,p.dec,p.r,s.redshift,s.specClass FROM SpecObj s JOIN PhotoObj p ON p.objID=s.bestObjID WHERE s.plate=? ORDER BY s.specObjID LIMIT ?"),
    ("sdss_redshift_join", "indexed_join", "SELECT s.specObjID,p.objID,p.u,p.g,p.r,s.redshift FROM SpecObj s JOIN PhotoObj p ON p.objID=s.bestObjID WHERE s.redshift BETWEEN ? AND ? ORDER BY s.redshift,s.specObjID LIMIT ?"),
    ("sdss_field_summary", "range_scan", "SELECT type,COUNT(*),AVG(r) FROM PhotoObj WHERE run=? AND field BETWEEN ? AND ? GROUP BY type ORDER BY type"),
]


def signature(template, params):
    return json.dumps([template, list(params)], separators=(",", ":"))


def bucket(value):
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % 10


def make_cases(database, seed, count, split="test", seen=None):
    if split not in ("training", "validation", "test") or count < 1:
        raise ValueError("Choose a valid partition and positive query count")
    allowed = {"training": set(range(7)), "validation": {7}, "test": {8, 9}}[split]
    seen = seen if seen is not None else set()
    rng = random.Random(seed)
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as con:
        anchors = con.execute("SELECT objID,ra,dec,r,run,field FROM PhotoObj ORDER BY objID").fetchall()
        plates = [r[0] for r in con.execute("SELECT DISTINCT plate FROM SpecObj ORDER BY plate")]
        redshifts = [r[0] for r in con.execute("SELECT redshift FROM SpecObj ORDER BY specObjID")]
    cases = []
    for i in range(count):
        t = i % len(TEMPLATES); name, family, sql = TEMPLATES[t]
        for _ in range(100000):
            obj, ra, dec, magnitude, run, field = rng.choice(anchors)
            if t == 0: params = (obj,)
            elif t == 1:
                width = rng.uniform(.005, .08)
                params = (round(magnitude-width, 6), round(magnitude+width, 6), rng.randint(20, 100))
            elif t == 2:
                width = rng.uniform(.02, .3)
                params = (round(ra-width, 6), round(ra+width, 6), round(dec-width, 6), round(dec+width, 6), rng.randint(20, 100))
            elif t == 3: params = (rng.choice(plates), rng.randint(20, 100))
            elif t == 4:
                z = rng.choice(redshifts); width = rng.uniform(.0001, .008)
                params = (round(z-width, 7), round(z+width, 7), rng.randint(20, 100))
            else: params = (run, max(0, field-rng.randint(0, 8)), field+rng.randint(1, 8))
            key = signature(name, params)
            if bucket(key) in allowed and key not in seen: break
        else: raise RuntimeError("Could not generate disjoint SDSS query parameters")
        seen.add(key)
        cases.append(QueryCase(f"{seed}-{i+1:05d}", name, family, sql, params))
    rng.shuffle(cases)
    return cases
