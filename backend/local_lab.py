"""Loopback-only bridge between the dashboard and Pagewise's existing CLI.

Start with ``python backend/local_lab.py`` or ``npm run dev:local``.  This file
is deliberately outside pagewise/ and scripts/ so historical source hashes do
not change when the local UI evolves.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent
RUNS = BACKEND / "runs"
LAB = RUNS / "local-lab"
READ_ROOTS = [BACKEND / name for name in ("runs", "data", "models", "results")] + [REPO / "public"]
WRITE_ROOTS = READ_ROOTS[:4]
PATH_OPTIONS = {"db", "out", "trace", "page-map", "sequences", "workload", "model-dir", "model-path",
                "candidate-index", "model-a", "model-b", "retrained-ablation-dir", "oracle-trace",
                "oracle-miss-ref", "runs", "input", "run-dir", "optimized-suite", "study-dir",
                "reference", "calibration", "ledger", "trace-dir", "static-metadata", "sources", "screen",
                "training-trace", "study", "frozen", "offline", "headroom", "oracle-suite", "policy"}
OUTPUT_OPTIONS = {"out"}
JOBS: dict[str, dict] = {}
ACTIVE: dict[str, subprocess.Popen | None] = {"process": None}
LOCK = threading.Lock()


def _literal(node: ast.AST | None):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def command_schema() -> dict[str, dict]:
    """Read the actual argparse definitions so every CLI feature stays exposed."""
    tree = ast.parse((BACKEND / "pagewise" / "cli.py").read_text(encoding="utf-8"))
    return _schema_from_tree(tree)


def _schema_from_tree(tree: ast.AST, *, script: str | None = None) -> dict[str, dict]:
    names: dict[str, str] = {}
    result: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "add_parser":
            continue
        if not node.targets or not isinstance(node.targets[0], ast.Name):
            continue
        name = _literal(call.args[0]) if call.args else None
        if not isinstance(name, str):
            continue
        names[node.targets[0].id] = f"script:{script}:{name}" if script else name
        help_text = next((_literal(kw.value) for kw in call.keywords if kw.arg == "help"), "")
        result[names[node.targets[0].id]] = {"name": names[node.targets[0].id], "help": help_text or "", "options": []}
    if script and not result:
        result[f"script:{script}"] = {"name": f"script:{script}", "help": f"Run scripts/{script}.py", "options": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "ArgumentParser" and node.targets and isinstance(node.targets[0], ast.Name):
                    names[node.targets[0].id] = f"script:{script}"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not isinstance(node.func.value, ast.Name):
            continue
        command = names.get(node.func.value.id)
        flag = _literal(node.args[0]) if node.args else None
        if command is None or not isinstance(flag, str) or not flag.startswith("--"):
            continue
        kwargs = {kw.arg: _literal(kw.value) for kw in node.keywords}
        kind = "boolean" if kwargs.get("action") == "store_true" else "text"
        if isinstance(kwargs.get("type"), type):
            kind = kwargs["type"].__name__
        else:
            type_node = next((kw.value for kw in node.keywords if kw.arg == "type"), None)
            if isinstance(type_node, ast.Name) and type_node.id in ("int", "float"):
                kind = type_node.id
        result[command]["options"].append({
            "name": flag[2:], "required": bool(kwargs.get("required")), "default": kwargs.get("default"),
            "choices": kwargs.get("choices"), "help": kwargs.get("help") or "", "kind": kind,
            "multiple": kwargs.get("nargs") == "+", "path": flag[2:] in PATH_OPTIONS,
        })
    return result


COMMANDS = command_schema()
for _script_path in sorted((BACKEND / "scripts").glob("*.py")):
    if _script_path.stem in {"verify_frozen", "run_v21_dev_verification"}:
        COMMANDS[f"script:{_script_path.stem}"] = {"name": f"script:{_script_path.stem}",
                                                  "help": f"Run scripts/{_script_path.name}", "options": []}
        continue
    _tree = ast.parse(_script_path.read_text(encoding="utf-8"))
    if any(isinstance(node, ast.Attribute) and node.attr == "ArgumentParser" for node in ast.walk(_tree)):
        COMMANDS.update(_schema_from_tree(_tree, script=_script_path.stem))


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root.resolve() or root.resolve() in path.parents for root in roots)


def safe_path(value: str, *, write: bool = False) -> Path:
    if not value or "\x00" in value:
        raise ValueError("A local path is required")
    path = Path(value)
    path = (path if path.is_absolute() else BACKEND / path).resolve()
    if not _within(path, WRITE_ROOTS if write else READ_ROOTS):
        raise ValueError("Paths must stay inside backend/runs, data, models, results, or public (read only)")
    return path


def build_argv(command: str, options: dict) -> list[str]:
    if command not in COMMANDS or not isinstance(options, dict):
        raise ValueError("Unknown Pagewise command or invalid options")
    specs = {item["name"]: item for item in COMMANDS[command]["options"]}
    unknown = set(options) - set(specs)
    if unknown:
        raise ValueError(f"Unknown options: {', '.join(sorted(unknown))}")
    argv = [command]
    for name, spec in specs.items():
        value = options.get(name)
        if value is None or value == "" or value is False:
            if spec["required"]:
                raise ValueError(f"--{name} is required")
            continue
        if spec["kind"] == "boolean":
            if value is not True:
                raise ValueError(f"--{name} must be a boolean")
            argv.append(f"--{name}")
            continue
        values = value if spec["multiple"] and isinstance(value, list) else [value]
        if spec["multiple"] and not values:
            raise ValueError(f"--{name} requires at least one value")
        clean = []
        for item in values:
            if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                raise ValueError(f"Invalid --{name} value")
            item = str(item).strip()
            if spec["kind"] == "int":
                int(item)
            elif spec["kind"] == "float":
                float(item)
            if spec["choices"] and item not in [str(choice) for choice in spec["choices"]]:
                raise ValueError(f"Unsupported --{name} choice")
            if spec["path"]:
                item = str(safe_path(item, write=name in OUTPUT_OPTIONS or (command == "generate-db" and name == "db")))
            clean.append(item)
        argv.extend([f"--{name}", *clean])
    return argv


def demo_steps(root: Path, queries: int) -> list[list[str]]:
    db = root / "pagewise-demo.sqlite"
    trace = root / "trace"
    models = root / "models"
    evaluation = root / "evaluation"
    suite = root / "suite-local"
    entries = [
        ("generate-db", {"db": str(db), "customers": 500, "products": 150, "orders": 3000,
                         "avg-items": 2, "graph-nodes": 600, "edges-per-node": 2, "seed": 23}),
        ("trace", {"db": str(db), "out": str(trace), "queries": queries, "seed": 1234, "cache-kib": 128}),
        ("build-sequences", {"trace": str(trace / "trace.jsonl"), "page-map": str(trace / "page_map.json"),
                             "out": str(trace / "sequences.jsonl")}),
        ("train", {"sequences": str(trace / "sequences.jsonl"), "page-map": str(trace / "page_map.json"),
                   "out": str(models), "seed": 7, "epochs": 5, "candidates": 16, "hidden": 32}),
        ("evaluate", {"sequences": str(models / "test_sequences.jsonl"), "page-map": str(trace / "page_map.json"),
                      "candidate-index": str(models / "candidate_index.json"), "model-a": str(models / "model_a.pt"),
                      "model-b": str(models / "model_b.pt"), "out": str(evaluation)}),
        ("run-suite", {"db": str(db), "model-dir": str(models), "out": str(suite), "queries": 30,
                       "seeds": "101,202", "strategies": "stock,none,sequential,markov1,markov2,model_a,model_b,hybrid",
                       "cache-kib": 128, "cache-pages": 128, "degree": 2}),
        ("dashboard-export", {"run-dir": str(root), "out": str(root / "dashboard.json")}),
    ]
    return [build_argv(command, options) for command, options in entries]


def _job_public(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in ("steps", "process")}


def _save(job: dict) -> None:
    path = LAB / f"{job['id']}.json"
    path.write_text(json.dumps(_job_public(job), indent=2), encoding="utf-8")


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass


def _run(job: dict) -> None:
    try:
        log_path = LAB / f"{job['id']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            for index, argv in enumerate(job["steps"]):
                if job["status"] in ("cancelled", "cancelling"):
                    job["status"] = "cancelled"
                    break
                with LOCK:
                    job["step"] = index + 1
                    job["command"] = argv[0]
                    _save(job)
                if argv[0].startswith("script:"):
                    _, script, *subcommand = argv[0].split(":")
                    command = [sys.executable, "-u", str(BACKEND / "scripts" / f"{script}.py"), *subcommand, *argv[1:]]
                else:
                    command = [sys.executable, "-u", "-m", "pagewise.cli", *argv]
                log.write(f"\n[{index + 1}/{len(job['steps'])}] {' '.join(command)}\n")
                log.flush()
                with subprocess.Popen(command, cwd=BACKEND,
                                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=os.name != "nt",
                                      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0) as proc:
                    with LOCK:
                        ACTIVE["process"] = proc
                        cancelled = job["status"] == "cancelling"
                    if cancelled:
                        _stop_process(proc)
                    code = proc.wait()
                    with LOCK:
                        ACTIVE["process"] = None
                if job["status"] == "cancelling":
                    job["status"] = "cancelled"
                    break
                if code != 0:
                    job["status"] = "failed"
                    job["exit_code"] = code
                    break
            else:
                job["status"] = "completed"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        with LOCK:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save(job)


def start_job(payload: dict) -> dict:
    with LOCK:
        if any(job["status"] in ("running", "cancelling") for job in JOBS.values()):
            raise RuntimeError("Another local experiment is running")
        job_id = uuid.uuid4().hex[:12]
        if payload.get("preset") == "demo":
            queries = int(payload.get("queries", 100))
            if not 1 <= queries <= 2000:
                raise ValueError("Demo queries must be between 1 and 2000")
            root = LAB / job_id
            root.mkdir(parents=True, exist_ok=False)
            steps = demo_steps(root, queries)
        else:
            options = payload.get("options", {})
            steps = [build_argv(payload.get("command"), options)]
            output = options.get("out") or (options.get("db") if payload.get("command") == "generate-db" else None)
            if output:
                path = safe_path(str(output), write=True)
                root = path.parent if path.suffix else path
            else:
                root = LAB
        job = {"id": job_id, "status": "running", "step": 0, "total_steps": len(steps),
               "command": "", "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
               "root": str(root.relative_to(BACKEND)), "log": str((LAB / f"{job_id}.log").relative_to(BACKEND)),
               "steps": steps}
        JOBS[job_id] = job
        LAB.mkdir(parents=True, exist_ok=True)
        _save(job)
        threading.Thread(target=_run, args=(job,), daemon=True).start()
        return _job_public(job)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # The browser polls every two seconds; keep the terminal focused on jobs.
        return

    def _reply(self, payload, status=200):
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _origin_ok(self):
        origin = self.headers.get("Origin", "")
        return not origin or origin.startswith(("http://127.0.0.1:", "http://localhost:"))

    def _host_ok(self):
        return self.headers.get("Host", "").split(":", 1)[0] in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._host_ok():
            return self._reply({"error": "Loopback host required"}, 403)
        try:
            path = urlsplit(self.path)
            query = parse_qs(path.query)
            if path.path == "/api/health":
                return self._reply({"ok": True, "python": sys.executable, "backend": str(BACKEND)})
            if path.path == "/api/commands":
                return self._reply({"commands": list(COMMANDS.values())})
            if path.path == "/api/jobs":
                return self._reply({"jobs": [_job_public(job) for job in reversed(list(JOBS.values()))]})
            if path.path.startswith("/api/jobs/"):
                job = JOBS.get(path.path.rsplit("/", 1)[-1])
                if not job:
                    return self._reply({"error": "Job not found"}, 404)
                log_path = BACKEND / job["log"]
                log = ""
                if log_path.exists():
                    with log_path.open("rb") as handle:
                        handle.seek(max(0, log_path.stat().st_size - 65536))
                        log = handle.read().decode("utf-8", errors="replace")[-30000:]
                return self._reply({"job": _job_public(job), "log": log})
            if path.path == "/api/files":
                requested = query.get("path", ["runs/local-lab"])[0]
                directory = safe_path(requested)
                if not directory.is_dir():
                    return self._reply({"error": "Directory not found"}, 404)
                entries = [{"name": child.name, "path": str(child.relative_to(BACKEND)) if _within(child, [BACKEND]) else str(child),
                            "directory": child.is_dir(), "bytes": child.stat().st_size if child.is_file() else None}
                           for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:200]
                           if _within(child.resolve(), READ_ROOTS)]
                return self._reply({"path": str(directory), "entries": entries})
            if path.path == "/api/file":
                file = safe_path(query.get("path", [""])[0])
                if not file.is_file() or file.suffix.lower() not in {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}:
                    return self._reply({"error": "Readable artifact not found"}, 404)
                if file.stat().st_size > 2_000_000:
                    return self._reply({"error": "Artifact exceeds the 2 MB browser preview limit"}, 413)
                return self._reply({"path": str(file), "content": file.read_text(encoding="utf-8", errors="replace")})
            return self._reply({"error": "Not found"}, 404)
        except (ValueError, OSError) as exc:
            return self._reply({"error": str(exc)}, 400)

    def do_POST(self):
        if not self._host_ok() or not self._origin_ok() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._reply({"error": "Local JSON request required"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/jobs":
                return self._reply({"job": start_job(payload)}, 202)
            if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.split("/")[3]
                proc = None
                with LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        return self._reply({"error": "Job not found"}, 404)
                    if job["status"] == "running":
                        job["status"] = "cancelling"
                        proc = ACTIVE["process"]
                        _save(job)
                if proc:
                    _stop_process(proc)
                return self._reply({"job": _job_public(job)})
            return self._reply({"error": "Not found"}, 404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._reply({"error": str(exc)}, 400)
        except RuntimeError as exc:
            return self._reply({"error": str(exc)}, 409)


def main():
    parser = argparse.ArgumentParser(description="Pagewise local dashboard bridge")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    LAB.mkdir(parents=True, exist_ok=True)
    for path in sorted(LAB.glob("*.json")):
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("id") != path.stem:
                continue
            if old.get("status") in ("running", "cancelling"):
                old["status"] = "interrupted"
                old["finished_at"] = datetime.now(timezone.utc).isoformat()
                path.write_text(json.dumps(old, indent=2), encoding="utf-8")
            JOBS[old["id"]] = old
        except (ValueError, OSError):
            pass
    print(f"Pagewise local API on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
