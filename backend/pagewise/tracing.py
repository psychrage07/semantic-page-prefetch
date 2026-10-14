from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_QUERY_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("pagewise_query_context", default=None)


def set_query_context(context: dict[str, Any]):
    return _QUERY_CONTEXT.set(context)


def reset_query_context(token) -> None:
    _QUERY_CONTEXT.reset(token)


class TraceRecorder:
    """Thread-safe JSONL writer for main-database VFS read events."""

    def __init__(self, output_path: str | Path, database_path: str | Path, page_size: int, *, enabled: bool = True):
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = os.path.normcase(os.path.abspath(str(database_path)))
        self.page_size = int(page_size)
        self.enabled = enabled
        self._file = self.output_path.open("w", encoding="utf-8") if enabled else None
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False

    def matches_database(self, filename: str | None) -> bool:
        if not filename:
            return False
        return os.path.normcase(os.path.abspath(filename)) == self.database_path

    def record_read(self, *, filename: str, offset: int, amount: int, duration_ns: int,
                    result: str, error: str | None = None, is_main_db: bool = True,
                    cache_status: str = "not_instrumented", underlying_bytes: int | None = None) -> None:
        if not self.enabled or not is_main_db or not self.matches_database(filename):
            return
        ctx = _QUERY_CONTEXT.get() or {}
        first_page = offset // self.page_size + 1 if amount > 0 else None
        last_page = (offset + amount - 1) // self.page_size + 1 if amount > 0 else None
        with self._lock:
            if self._closed:
                return
            self._seq += 1
            row = {
                "schema_version": 1,
                "seq": self._seq,
                "wall_time_ns": time.time_ns(),
                "monotonic_time_ns": time.perf_counter_ns(),
                "duration_us": round(duration_ns / 1000.0, 3),
                "file": filename,
                "file_type": "main_db",
                "offset": int(offset),
                "amount": int(amount),
                "page_size": self.page_size,
                "first_page": first_page,
                "last_page": last_page,
                "result": result,
                "error": error,
                "cache_status": cache_status,
                "underlying_bytes": int(amount if underlying_bytes is None and result == "success" else (underlying_bytes or 0)),
                "query_id": ctx.get("query_id"),
                "template_id": ctx.get("template_id"),
                "workload_family": ctx.get("workload_family"),
                "plan_class": ctx.get("plan_class", "unknown"),
                "plan_details": ctx.get("plan_details", []),
                "source": ctx.get("source", "demand"),
            }
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Flush in small batches to avoid excessive synchronous trace overhead.
            if self._seq % 128 == 0:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                if self._file is not None:
                    self._file.flush()
                    self._file.close()
                self._closed = True


try:
    import apsw
except ImportError:  # APSW must be installed on the machine doing the trace.
    apsw = None


class _TraceVFSFile:
    """Created dynamically as an APSW VFSFile subclass when APSW is installed."""


if apsw is not None:
    class TraceVFSFile(apsw.VFSFile):
        def __init__(self, base_vfs: str, name: Any, flags: list[int, int], recorder: TraceRecorder,
                     runtime=None, read_delay_ms: float = 0.0, storage_device=None):
            self._recorder = recorder
            self._runtime = runtime
            self._storage_device = storage_device
            self._read_delay_ms = max(0.0, float(read_delay_ms))
            self._last_query_id = None
            self._page_history: list[int] = []
            self._query_page_position = 0
            if isinstance(name, apsw.URIFilename):
                try:
                    self._filename = name.filename()
                except Exception:
                    self._filename = None
            else:
                self._filename = str(name) if name is not None else None
            self._is_main_db = recorder.matches_database(self._filename)
            super().__init__(base_vfs, name, flags)

        def xRead(self, amount: int, offset: int) -> bytes:
            start = time.perf_counter_ns()
            if not self._is_main_db or self._runtime is None or amount <= 0:
                try:
                    if self._is_main_db:
                        pages_touched = 1 if amount <= 0 else ((offset + amount - 1) // self._recorder.page_size - offset // self._recorder.page_size + 1)
                        if self._storage_device is not None and self._storage_device.enabled:
                            self._storage_device.access("demand", max(1, pages_touched))
                        elif self._read_delay_ms:
                            time.sleep(self._read_delay_ms * max(1, pages_touched) / 1000.0)
                    data = super().xRead(amount, offset)
                except Exception as exc:
                    self._recorder.record_read(
                        filename=self._filename or "", offset=offset, amount=amount,
                        duration_ns=time.perf_counter_ns() - start,
                        result="error", error=f"{type(exc).__name__}: {exc}", is_main_db=self._is_main_db,
                        cache_status="not_instrumented", underlying_bytes=0,
                    )
                    raise
                self._recorder.record_read(
                    filename=self._filename or "", offset=offset, amount=amount,
                    duration_ns=time.perf_counter_ns() - start,
                    result="success", is_main_db=self._is_main_db,
                    cache_status="delayed_stock" if self._is_main_db and (self._read_delay_ms or (self._storage_device and self._storage_device.enabled)) else "not_instrumented",
                    underlying_bytes=amount,
                )
                return data

            runtime = self._runtime
            ctx = _QUERY_CONTEXT.get() or {}
            query_id = ctx.get("query_id")
            if query_id != self._last_query_id:
                self._last_query_id = query_id
                self._page_history = []
                self._query_page_position = 0
                runtime.set_active_query(query_id)
            file_size = super().xFileSize()
            end = offset + amount
            cursor = offset
            output = bytearray()
            pages_seen: list[int] = []
            page_access_statuses: list[str] = []
            hit_count = 0
            miss_count = 0
            underlying_bytes = 0
            try:
                while cursor < end:
                    page = cursor // runtime.page_size + 1
                    page_start = (page - 1) * runtime.page_size
                    within_page = cursor - page_start
                    take = min(runtime.page_size - within_page, end - cursor)
                    pages_seen.append(page)
                    page_data = runtime.cache.get(page, demand=True, query_id=query_id)
                    if page_data is not None and len(page_data) == runtime.page_size:
                        output.extend(page_data[within_page:within_page + take])
                        hit_count += 1
                        page_access_statuses.append("hit")
                    else:
                        miss_count += 1
                        page_access_statuses.append("miss")
                        runtime.record_demand_miss_in_flight(page)
                        if page_start + runtime.page_size <= file_size:
                            if self._storage_device is not None and self._storage_device.enabled:
                                self._storage_device.access("demand", 1)
                            elif runtime.miss_delay_ms:
                                time.sleep(runtime.miss_delay_ms / 1000.0)
                            page_data = super().xRead(runtime.page_size, page_start)
                            underlying_bytes += len(page_data)
                            if len(page_data) == runtime.page_size:
                                runtime.cache.put(page, page_data, origin="demand")
                                output.extend(page_data[within_page:within_page + take])
                            else:
                                fallback = super().xRead(take, cursor)
                                underlying_bytes += len(fallback)
                                output.extend(fallback)
                        else:
                            if self._storage_device is not None and self._storage_device.enabled:
                                self._storage_device.access("demand", 1)
                            elif runtime.miss_delay_ms:
                                time.sleep(runtime.miss_delay_ms / 1000.0)
                            partial = super().xRead(take, cursor)
                            underlying_bytes += len(partial)
                            output.extend(partial)
                    cursor += take

                if runtime.recording:
                    runtime.record_query_demand(query_id, pages_seen, page_access_statuses)

                if query_id:
                    for page, access_status in zip(pages_seen, page_access_statuses):
                        self._page_history.append(page)
                        self._page_history = self._page_history[-8:]
                        self._query_page_position += 1
                        live_context = dict(ctx)
                        live_context["position_in_query"] = min(self._query_page_position, 512) / 512.0
                        live_context["query_page_position"] = self._query_page_position
                        if runtime.should_predict(access_status):
                            runtime.request_prediction(self._page_history, live_context)
                if hit_count and miss_count:
                    cache_status = "mixed"
                elif hit_count:
                    cache_status = "hit"
                else:
                    cache_status = "miss"
                self._recorder.record_read(
                    filename=self._filename or "", offset=offset, amount=amount,
                    duration_ns=time.perf_counter_ns() - start, result="success",
                    is_main_db=True, cache_status=cache_status, underlying_bytes=underlying_bytes,
                )
                return bytes(output)
            except Exception as exc:
                self._recorder.record_read(
                    filename=self._filename or "", offset=offset, amount=amount,
                    duration_ns=time.perf_counter_ns() - start, result="error",
                    error=f"{type(exc).__name__}: {exc}", is_main_db=True,
                    cache_status="error", underlying_bytes=underlying_bytes,
                )
                raise


    class TraceVFS(apsw.VFS):
        """VFS wrapper that inherits the default VFS and records main DB xRead calls."""

        def __init__(self, recorder: TraceRecorder, name: str | None = None, base_vfs: str = "",
                     runtime=None, read_delay_ms: float = 0.0, storage_device=None):
            self.recorder = recorder
            self.runtime = runtime
            self.storage_device = storage_device
            self.read_delay_ms = max(0.0, float(read_delay_ms))
            self.base_vfs = base_vfs
            self.name = name or f"pagewise_trace_{uuid.uuid4().hex[:8]}"
            super().__init__(self.name, self.base_vfs)

        def xOpen(self, name: Any, flags: list[int, int]):
            return TraceVFSFile(self.base_vfs, name, flags, self.recorder, self.runtime, self.read_delay_ms, self.storage_device)
else:
    class TraceVFS:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("APSW is required for VFS tracing. Install dependencies with: pip install -r requirements.txt")
