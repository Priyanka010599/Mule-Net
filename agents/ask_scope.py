"""What account-level data Ask may return: the scope rule.

**Rule.** Ask may state facts about an individual account only if that
account is a member of a detected ring (alertable or same-bank context) --
the accounts the UI itself puts in front of a user. Aggregates over the
whole dataset (counts, distinct values) are unaffected. An account outside
every ring is not shown by name, profile, score or explanation, however it
is asked for: ``get_account_profile`` and an account ``get_explanation``
refuse it, and a ``run_sql`` result that would list one is refused whole
(rewrite it as an aggregate).

Applied only to the ``analyst_qa`` tool set (see
:data:`agents.adk_agents.AGENT_TOOLS`); the bank-facing ``GET /screen`` and
the offline investigator are different surfaces with their own contracts.
"""

from __future__ import annotations

import functools
import inspect
import re
import threading
from typing import Any

from agents import tool_cache

_ACCOUNT_ID = re.compile(r"^[0-9a-f]{64}$")
_LOCK = threading.Lock()
_CACHE: dict[int, tuple[Any, str, frozenset[str]]] = {}


def visible_accounts(store: Any) -> frozenset[str]:
    """Hashed ids of every ring member, cached per (store, data version)."""
    version = tool_cache.data_key(store)
    with _LOCK:
        hit = _CACHE.get(id(store))
        if hit is not None and hit[0] is store and hit[1] == version:
            return hit[2]
    members = frozenset(str(a) for a in store.table("ring_members")["hashed_account_id"])
    with _LOCK:
        _CACHE[id(store)] = (store, version, members)
    return members


def _refuse(account: str) -> ValueError:
    return ValueError(
        f"account {account!r} is outside what Ask can show: only accounts that belong to a "
        "detected ring are available here. Aggregate counts over all accounts are allowed."
    )


def scoped_account_tool(func: Any, account_arg: str, only_when: Any = None) -> Any:
    """Wrap an account-keyed tool so out-of-scope accounts raise ``ValueError``."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from agents import tools  # noqa: PLC0415 - avoids a cycle

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if only_when is None or only_when(bound.arguments):
            account = str(bound.arguments[account_arg])
            if account not in visible_accounts(tools._store()):
                raise _refuse(account)
        return func(*args, **kwargs)

    return wrapper


def check_rows(rows: Any, store: Any) -> None:
    """Refuse query rows that name an account outside the scope."""
    if not isinstance(rows, list):
        return
    visible = visible_accounts(store)
    for row in rows:
        values = row.values() if isinstance(row, dict) else ()
        for value in values:
            if isinstance(value, str) and _ACCOUNT_ID.match(value) and value not in visible:
                raise _refuse(value)
