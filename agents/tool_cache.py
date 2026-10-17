"""A small bounded cache for deterministic tool results.

Why this exists: on the Ask path a question spends its non-model time in
the tools, and the tools are pure functions of the published tables.
Measured against the real local data (see the commit message for the
numbers), ``run_sql`` over ``accounts`` took ~2.8s on **every** call, a
ring summary ~0.1s, ``describe_tables`` ~2s cold. The same question asked
twice, or two sessions asking the same demo question, paid that again.

What is cached is exactly the set in :data:`agents.tools` that decorates
itself with :func:`cached`: ``describe_tables``, ``get_ring_summary``,
``get_ring_transfer_summary`` and ``run_sql`` (which also serves the
dataset counts, since those are ``SELECT COUNT(*)``).

**The key is** ``(store, DataStore.data_version(...), tool, bound args)``.
The data version means a re-published parquet changes the key and nothing
computed from the old data is served. The store is part of the key (held
by identity) so two stores that happen to share a version -- every mock
store hashes to the same ``"mock"`` marker -- never share an answer.

**Not cached:** a call that raised (the model must see the error again,
and a transient failure must not stick), and anything not in the list
above. A hit hands back a deep copy so a caller that mutates a result
cannot poison the next reader.

The tool *ledger* (:data:`agents.adk_agents.TOOL_CALLS`) and the
``sources`` sink are written by the wrapper around the tool, before this
cache is consulted, so a cache hit is still an observed tool execution.

Set ``MULENET_AGENTS_TOOL_CACHE=0`` to disable, ``MULENET_AGENTS_TOOL_CACHE_SIZE``
to resize (default 128 entries).
"""

from __future__ import annotations

import copy
import functools
import inspect
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

#: Tables whose bytes decide a tool result. ``ring_members`` and
#: ``explanations`` resolve through ``DataStore.source_path`` to the file
#: they are derived from, so they need no entry of their own.
DATA_VERSION_TABLES: tuple[str, ...] = (
    "accounts",
    "transfers",
    "devices",
    "kyc_submissions",
    "kyc_documents",
    "rings",
    "scores",
    "ring_explanations",
    "account_explanations",
    "account_evidence",
    "cases",
)

DEFAULT_SIZE = 128


def _size() -> int:
    try:
        return max(1, int(os.environ.get("MULENET_AGENTS_TOOL_CACHE_SIZE", DEFAULT_SIZE)))
    except ValueError:
        return DEFAULT_SIZE


def enabled() -> bool:
    """Whether results are cached. Read per call so a test can flip it."""
    return os.environ.get("MULENET_AGENTS_TOOL_CACHE", "1") != "0"


class ToolCache:
    """Bounded LRU, safe to share between the request threads."""

    def __init__(self, maxsize: int | None = None) -> None:
        self._maxsize = maxsize
        self._entries: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize if self._maxsize is not None else _size()

    def get(self, store: Any, key: tuple[Any, ...]) -> tuple[bool, Any]:
        """``(True, value)`` on a hit. The store must be the very one stored."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] is not store:
                self.misses += 1
                return False, None
            self._entries.move_to_end(key)
            self.hits += 1
            return True, entry[1]

    def put(self, store: Any, key: tuple[Any, ...], value: Any) -> None:
        with self._lock:
            self._entries[key] = (store, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._entries)}


#: The process-wide cache the tool decorators share.
CACHE = ToolCache()


def data_key(store: Any) -> str:
    """The data version of everything a tool may read."""
    return store.data_version(DATA_VERSION_TABLES)


def cached(get_store: Callable[[], Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a deterministic tool so identical calls reuse one result.

    ``get_store`` is called per invocation (not captured) because
    :func:`agents.tools._store` is what tests and the service swap.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return func(*args, **kwargs)
            store = get_store()
            try:
                version = data_key(store)
            except Exception:  # noqa: BLE001 - an unkeyable store is uncacheable, not fatal
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = tuple(
                (name, value.strip() if isinstance(value, str) else value)
                for name, value in bound.arguments.items()
            )
            key = (version, func.__name__, repr(arguments))
            hit, value = CACHE.get(store, key)
            if hit:
                return copy.deepcopy(value)
            result = func(*args, **kwargs)
            CACHE.put(store, key, copy.deepcopy(result))
            return result

        return wrapper

    return decorate
