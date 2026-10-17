"""Warm everything the Ask tools would otherwise build on the first question.

Measured against the real local data (see the commit message), the first
question after a cold start paid, before any model call:

* ~3.8s reading ``rings`` and the other parquet tables,
* ~2.2s importing the BigQuery schema module behind ``describe_tables``,
* ~15s reading ``transfers`` and ~46s building the transfer adjacency index
  (now ~5s; see :class:`agents.tools._TransferGraph`),
* ~2.8s copying ``accounts`` into SQLite, on every ``run_sql`` call
  (now once; see ``agents.sql_guard._scratch_database``),
* seconds importing ADK and google-genai.

All of it happens here instead, in a daemon thread started from the
service's lifespan, so ``/health`` and every other route answer while it
runs and the first judge question finds it done. A request that arrives
mid-warm waits on the same lock (``DataStore`` tables, the graph, the SQL
images) rather than building a second copy. Every step is best-effort:
a failure is logged and the next step runs, because a warm-up that can take
the service down is worse than no warm-up.

Skipped when ``MULENET_AGENTS_WARM_TOOLS=0`` (``agents/conftest.py`` sets
that, so the test suite never builds the real graph).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

_log = logging.getLogger("agents.warm")

#: Rings whose summaries are precomputed. The demo has a few dozen; the cap
#: only stops a larger table turning warm-up into a long CPU burn.
MAX_RINGS_WARMED = 100

#: Cheap queries that load, and cache an image of, the tables the usual
#: dataset-count questions touch.
_PRIME_SQL = (
    "SELECT COUNT(*) FROM accounts",
    "SELECT COUNT(*) FROM rings",
    "SELECT COUNT(*) FROM ring_members",
    "SELECT COUNT(*) FROM scores",
    "SELECT COUNT(*) FROM devices",
    "SELECT COUNT(*) FROM kyc_documents",
)


def enabled() -> bool:
    return os.environ.get("MULENET_AGENTS_WARM_TOOLS", "1") != "0"


def warm(on_step: Callable[[str, float], None] | None = None) -> dict[str, float]:
    """Run every warm-up step; return seconds spent per step."""
    from agents import adk_agents, tool_cache, tools  # noqa: PLC0415 - heavy imports

    timings: dict[str, float] = {}

    def step(name: str, fn: Callable[[], object]) -> object:
        started = time.perf_counter()
        try:
            return fn()
        except Exception:  # noqa: BLE001 - warming must never take the service down
            _log.warning("tool warm-up step %s failed", name, exc_info=True)
            return None
        finally:
            timings[name] = round(time.perf_counter() - started, 3)
            if on_step is not None:
                on_step(name, timings[name])

    store = tools._store()
    step("adk_import", adk_agents._require_adk)
    step("data_version", lambda: tool_cache.data_key(store))
    step("describe_tables", tools.describe_tables)
    for sql in _PRIME_SQL:
        step(f"sql:{sql.rsplit(' ', 1)[-1]}", lambda sql=sql: tools.run_sql(sql))
    rings = step("rings_table", lambda: store.table("rings"))
    step("transfer_graph", tools._graph)
    if rings is not None:
        ring_ids = [str(r) for r in rings["ring_id"]][:MAX_RINGS_WARMED]

        def summaries() -> None:
            for ring_id in ring_ids:
                tools.get_ring_summary(ring_id)
                tools.get_ring_transfer_summary(ring_id)

        step("ring_summaries", summaries)
    _log.warning("tool warm-up finished: %s", timings)
    return timings
