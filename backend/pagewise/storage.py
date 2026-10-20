"""Abstract k-channel storage model for the threaded runtime (v2.1, Requirement A).

The v2 simulator and runtime modelled storage as either

* ``shared``: one serialized channel (a single FIFO queue, k = 1), or
* ``independent``: effectively infinite parallelism (every access gets its own
  private channel, k = infinity, and device-level contention is unobservable),
  or the legacy inline ``time.sleep`` path.

Neither endpoint is a calibrated physical HDD or NVMe model. v2.1 adds the
intermediate abstract model ``k_channel(k)``: k identical concurrent service
slots, each a FIFO queue, served with one per-slot service time equal to the
configured per-page service delay.

Queue discipline
----------------
Within a slot: FIFO — sleepers queue on the slot's own lock.
Across slots: join the slot with the fewest assigned-and-unfinished jobs
(earliest projected free time under equal service times; ties break to the
lowest slot index).

Reported results under this model are **simulated concurrency scenarios**,
not device measurements, unless a device-specific calibration section is
added here and executed on the target hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class KChannelStorageDevice:
    """k concurrent service slots with per-slot FIFO admission."""

    def __init__(self, delay_ms: float = 0.0, channels: int = 1, enabled: bool = False):
        self.delay_ms = max(0.0, float(delay_ms))
        self.channels = max(1, int(channels))
        self.enabled = bool(enabled and self.delay_ms > 0)
        self._slot_locks = [threading.Lock() for _ in range(self.channels)]
        self._accounting = threading.Lock()
        self._assigned = [0] * self.channels      # jobs ever assigned per slot
        self._finished = [0] * self.channels      # jobs ever finished per slot
        self.stats: dict[str, Any] = {
            "demand_ios": 0, "prefetch_ios": 0,
            "demand_wait_ms_total": 0.0, "demand_wait_ms_max": 0.0,
            "prefetch_wait_ms_total": 0.0, "prefetch_wait_ms_max": 0.0,
            "device_busy_ms_total": 0.0,
        }
        self._in_service = 0
        self._max_in_service = 0

    @property
    def storage_kind(self) -> str:
        return f"k_channel_{self.channels}" if self.channels > 1 else "shared_serialized"

    def access(self, source: str = "demand", pages: int = 1) -> float:
        """Occupy one channel for the service time; returns queued wait in ms."""
        if not self.enabled or self.delay_ms <= 0:
            return 0.0
        pages = max(1, int(pages))
        service_s = self.delay_ms * pages / 1000.0
        service_ms = service_s * 1000.0
        # Admission happens under a short, non-sleeping critical section: choose
        # the slot with the fewest outstanding jobs, then queue on that slot.
        with self._accounting:
            outstanding = [a - f for a, f in zip(self._assigned, self._finished)]
            slot = outstanding.index(min(outstanding))
            self._assigned[slot] += 1
        wait_started = time.perf_counter()
        with self._slot_locks[slot]:
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            with self._accounting:
                self._in_service += 1
                self._max_in_service = max(self._max_in_service, self._in_service)
            time.sleep(service_s)
            with self._accounting:
                self._in_service -= 1
                self._finished[slot] += 1
                if source == "demand":
                    self.stats["demand_ios"] += 1
                    self.stats["demand_wait_ms_total"] += wait_ms
                    self.stats["demand_wait_ms_max"] = max(self.stats["demand_wait_ms_max"], wait_ms)
                else:
                    self.stats["prefetch_ios"] += 1
                    self.stats["prefetch_wait_ms_total"] += wait_ms
                    self.stats["prefetch_wait_ms_max"] = max(self.stats["prefetch_wait_ms_max"], wait_ms)
                self.stats["device_busy_ms_total"] += service_ms
        return wait_ms

    def snapshot(self) -> dict[str, Any]:
        with self._accounting:
            stats = dict(self.stats)
            stats["outstanding_per_slot"] = [
                a - f for a, f in zip(self._assigned, self._finished)]
            stats["max_in_service"] = self._max_in_service
        stats.update({
            "delay_ms": self.delay_ms,
            "enabled": self.enabled,
            "channels": self.channels,
            "storage_kind": self.storage_kind,
        })
        demand_ios = max(1, stats["demand_ios"])
        prefetch_ios = max(1, stats["prefetch_ios"])
        stats["demand_wait_ms_mean"] = round(stats["demand_wait_ms_total"] / demand_ios, 3) if stats["demand_ios"] else 0.0
        stats["prefetch_wait_ms_mean"] = round(stats["prefetch_wait_ms_total"] / prefetch_ios, 3) if stats["prefetch_ios"] else 0.0
        return stats
