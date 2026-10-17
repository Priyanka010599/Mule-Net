"""Cost ceilings for the HTTP service.

``agents/service.py`` is deployed without public invoker access, so
Cloud Run rejects an unauthenticated call before any code here runs. That
bounds *who* may call. It does not bound *how much* is spent, because the
one authorised caller is ``mulenet-ui`` -- a public page with no sign-in,
fanning a judge's clicks straight through. Behind that door the request
volume is still effectively anonymous, so cost stays a security property:
the failure mode is not a crash, it is a bill.

``docs/agents_api.md`` makes the same point and requires these limits
regardless of auth. They are sized for an unauthenticated caller, which
is what the UI's visitors are.

Three ceilings, in increasing order of how much they can be trusted:

**Per session.** Keyed on a session id the client sends. Stops the
obvious accident -- a UI in a retry loop, a held-down enter key.

**Per IP.** Keyed on ``X-Forwarded-For``. Stops one machine from
sweeping through session ids. A client can pad that header, so this is
friction rather than enforcement.

**Per day, whole service.** Counted on the server against nothing the
caller supplies. This is the only one a hostile caller cannot move, and
therefore the only one that actually bounds the bill. It is checked
first, so that the number protecting the budget is never reached through
the two that can be forged.

A rejected request consumes no quota. Rejections make no model call, so
charging them against the daily cap would let anyone exhaust the day's
budget for free -- turning the cost control into the denial of service it
exists to prevent.

The counters live in memory, so they are per Cloud Run instance. With
more than one instance the effective caps multiply by the instance count,
which is why the deployment must pin ``--max-instances=1`` for the
numbers here to mean what they say. That is recorded for the platform
engineer rather than worked around, because a shared counter needs a
store this service does not have.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from agents.config import ServiceLimits

#: Seconds in a day. The daily cap resets on the UTC date boundary.
_DAY_S = 86_400.0

#: Distinct keys tolerated before empty windows are swept.
_SWEEP_AT_KEYS = 1024


class LimitExceeded(Exception):
    """One ceiling was hit. Carries what a UI can put on screen.

    Attributes:
        scope: ``session``, ``ip`` or ``daily`` -- which ceiling.
        message: Plain text intended for a user, not a log line. The
            service returns it verbatim in the 429 body, because "429"
            with no explanation looks like a broken demo rather than a
            deliberate budget.
        retry_after_s: Whole seconds until the ceiling clears.
    """

    def __init__(self, scope: str, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.scope = scope
        self.message = message
        self.retry_after_s = retry_after_s


class _SlidingWindow:
    """Per-key request timestamps within a fixed window."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window_s = window_s
        self._hits: dict[str, deque[float]] = {}

    def would_exceed(self, key: str, now: float) -> float | None:
        """Seconds until ``key`` has room again, or ``None`` if it has room."""
        hits = self._prune(key, now)
        if len(hits) < self._limit:
            return None
        return max(0.0, self._window_s - (now - hits[0]))

    def record(self, key: str, now: float) -> None:
        """Count one accepted request against ``key``."""
        self._prune(key, now).append(now)
        if len(self._hits) > _SWEEP_AT_KEYS:
            # Every session id ever seen would otherwise keep an empty
            # deque forever. Sweeping on growth rather than on every call
            # keeps the common path O(1).
            self._hits = {
                other: hits for other, hits in self._hits.items() if hits
            }

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self._window_s
        while hits and hits[0] <= cutoff:
            hits.popleft()
        return hits


class RequestGuard:
    """Decides whether one model-backed request may proceed.

    Not a general rate limiter: it guards Gemini spend specifically, so
    only endpoints that call a model consult it. The precomputed-case
    endpoint reads a parquet file and costs nothing, so charging it
    against the daily model budget would make free reads exhaust the
    paid one.
    """

    def __init__(
        self,
        limits: ServiceLimits | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Build a guard.

        Args:
            limits: The ceilings. Defaults to the environment.
            now: Epoch-seconds clock. Injected so the tests can move
                time instead of sleeping through a 60-second window.
        """
        self.limits = limits or ServiceLimits.from_env()
        self._now = now
        self._lock = threading.Lock()
        self._sessions = _SlidingWindow(self.limits.session_limit, self.limits.window_s)
        self._ips = _SlidingWindow(self.limits.ip_limit, self.limits.window_s)
        self._day = -1
        self._day_count = 0
        self._day_tokens = 0

    def check_and_record(self, session_id: str, client_ip: str) -> None:
        """Admit one request, or refuse it.

        All three ceilings are evaluated before any of them is written
        to, so a refused request leaves every counter untouched.

        Raises:
            LimitExceeded: The first ceiling that has no room.
        """
        with self._lock:
            now = self._now()
            self._roll_day(now)

            if (
                self._day_count >= self.limits.daily_cap
                or self._day_tokens >= self.limits.daily_token_cap
            ):
                raise LimitExceeded(
                    scope="daily",
                    message=(
                        "The demo's daily question budget is used up. This "
                        "service runs on a public URL with no sign-in, so it "
                        f"answers at most {self.limits.daily_cap} questions a "
                        "day to keep the bill bounded. The precomputed cases "
                        "and the graph are still available; chat resets at "
                        "00:00 UTC."
                    ),
                    retry_after_s=self._seconds_to_midnight(now),
                )

            ip_wait = self._ips.would_exceed(client_ip, now)
            if ip_wait is not None:
                raise LimitExceeded(
                    scope="ip",
                    message=(
                        f"Too many questions from this network: at most "
                        f"{self.limits.ip_limit} every "
                        f"{int(self.limits.window_s)} seconds. Try again in "
                        f"{max(1, round(ip_wait))} seconds."
                    ),
                    retry_after_s=max(1, round(ip_wait)),
                )

            session_wait = self._sessions.would_exceed(session_id, now)
            if session_wait is not None:
                raise LimitExceeded(
                    scope="session",
                    message=(
                        f"You are asking faster than this demo answers: at "
                        f"most {self.limits.session_limit} questions every "
                        f"{int(self.limits.window_s)} seconds. Try again in "
                        f"{max(1, round(session_wait))} seconds."
                    ),
                    retry_after_s=max(1, round(session_wait)),
                )

            self._ips.record(client_ip, now)
            self._sessions.record(session_id, now)
            self._day_count += 1

    def add_tokens(self, tokens: int) -> None:
        """Charge a finished request's Gemini tokens to today's token budget.

        Counted after the fact, so one request can overshoot the cap by its
        own size; the next one is then refused. Like the request counter this
        is in memory (per instance, reset on restart): see the module notes
        on ``--max-instances=1``.
        """
        with self._lock:
            self._roll_day(self._now())
            self._day_tokens += max(0, tokens)

    def remaining_today(self) -> int:
        """Model-backed requests left in the current UTC day."""
        with self._lock:
            self._roll_day(self._now())
            return max(0, self.limits.daily_cap - self._day_count)

    def _roll_day(self, now: float) -> None:
        """Reset the daily counter when the UTC date changes."""
        day = int(now // _DAY_S)
        if day != self._day:
            self._day = day
            self._day_count = 0
            self._day_tokens = 0

    def _seconds_to_midnight(self, now: float) -> int:
        """Whole seconds until the daily counter resets."""
        return max(1, int(_DAY_S - (now % _DAY_S)))
