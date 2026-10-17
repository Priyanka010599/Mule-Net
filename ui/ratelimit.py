"""A small in-process sliding-window limiter for the public endpoints.

Per instance, like ``agents/limits.py``: it bounds abuse of one process, it is
not a distributed quota. Anything that costs real money (the live-proof VM)
gets a durable cap on top of this, in ``agents/live_proof_service.py``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from math import ceil

#: Distinct keys tolerated before keys with empty windows are swept.
_MAX_KEYS = 10_000


class SlidingWindow:
    """At most ``limit`` hits per ``window_s`` seconds, per key."""

    def __init__(
        self,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record one hit. ``None`` if allowed, else whole seconds until it would be."""
        now = self._clock()
        with self._lock:
            if len(self._hits) > _MAX_KEYS:
                self._hits = {
                    k: v for k, v in self._hits.items() if v and v[-1] > now - self._window_s
                }
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self._window_s:
                hits.popleft()
            if len(hits) < self._limit:
                hits.append(now)
                return None
            return max(1, ceil(hits[0] + self._window_s - now))
