"""Verify that the frozen run directories are byte-for-byte unchanged.

Run from the backend directory: python scripts/verify_frozen.py
Exits non-zero if any file was added, removed, or modified relative to
FROZEN_MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys


def main() -> int:
    manifest = json.loads(pathlib.Path("FROZEN_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual: dict[str, str] = {}
    for directory in manifest["directories"]:
        for path in sorted(pathlib.Path(directory).rglob("*")):
            if path.is_file():
                actual[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = sorted(
        path for path in set(expected) & set(actual)
        if expected[path]["sha256"] != actual[path]
    )

    if not (added or removed or changed):
        print(f"FROZEN OK: {len(actual)} files unchanged")
        return 0

    print("FROZEN MISMATCH")
    for path in added:
        print("  ADDED  ", path)
    for path in removed:
        print("  REMOVED", path)
    for path in changed:
        print("  CHANGED", path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
