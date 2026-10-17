"""Deterministic answers for the suggested chat questions: no model call.

Three prompts are offered to every visitor as one-click questions. They
have exact answers in the rings table, so sending them through Gemini
would spend money and latency to get a number that can be read off a
table -- and risk a wrong one. ``POST /chat`` serves them from here first,
before the rate-limit guard and the model, and marks the reply
``source: "precomputed"``.

Nothing is hard-coded: every figure is computed from ``store.table("rings")``
and cached under :meth:`agents.data.DataStore.data_version`, so a
re-published rings file recomputes them.

"Largest" means **most members** (``members`` counts every member,
context-only ones included), ties broken by the smaller ``ring_id``.
The answers say so, because "largest" is otherwise ambiguous.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from agents.data import DataStore

#: The suggested prompts, exactly as the UI offers them.
Q_RING_COUNT = "How many rings are in the dataset?"
Q_LARGEST_BANKS = (
    "Which banks are in the largest cross-bank ring and how many accounts does it have?"
)
Q_LARGEST_ALERTABLE = "How many alertable members does the largest ring have?"
#: The "Explain this" button on the race-to-the-exit card sends this. It is about
#: ONE ring, the one in ``ui_context``, so it is answered here only when that ring
#: is known (see :func:`race_answer`); it is not in :data:`SUGGESTED_QUESTIONS`
#: because it has no answer without a ring.
Q_RACE = (
    "Explain the race to the exit for this ring: when did single-bank and cross-bank "
    "flag it, compared with when the money reached the exit account?"
)
SUGGESTED_QUESTIONS = (Q_RING_COUNT, Q_LARGEST_BANKS, Q_LARGEST_ALERTABLE)

#: The UI prefixes the open ring onto a question ("About ring X: ...").
#: That is context, not a different question, so it is stripped before matching.
_RING_PREFIX = re.compile(r"^about ring \S+?:\s*", re.IGNORECASE)


def normalise(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace."""
    text = _RING_PREFIX.sub("", text.strip())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


_KEYS = {normalise(q): q for q in SUGGESTED_QUESTIONS}
_RACE_KEY = normalise(Q_RACE)


@dataclass(frozen=True)
class Precomputed:
    """One deterministic answer and what it was computed from."""

    answer: str
    sources: tuple[str, ...]
    details: tuple[dict[str, str], ...]


_lock = threading.Lock()
#: ``data_version`` -> answers by prompt. One entry: an older version is dropped.
_cache: dict[str, dict[str, Precomputed]] = {}


def is_suggested(message: str) -> bool:
    return normalise(message) in _KEYS


def _compute(store: DataStore) -> dict[str, Precomputed]:
    rings = store.table("rings")
    if rings.empty:
        return {}
    rows: list[dict[str, Any]] = [
        {
            "ring_id": str(row.ring_id),
            "members": list(row.members),
            "banks": sorted(str(b) for b in row.banks_involved),
        }
        for row in rings.itertuples()
    ]
    top = min(rows, key=lambda r: (-len(r["members"]), r["ring_id"]))
    members = len(top["members"])
    alertable = sum(1 for m in top["members"] if dict(m).get("alertable"))
    banks = top["banks"]
    definition = "the ring with the most members, ties broken by ring id"
    sources = ("rings",)
    detail = ({"tool": "precomputed", "args_summary": "rings table"},)
    bank_text = ", ".join(banks[:-1]) + f" and {banks[-1]}" if len(banks) > 1 else banks[0]
    span = "cross-bank " if len(banks) > 1 else "single-bank "
    return {
        Q_RING_COUNT: Precomputed(
            f"There are {len(rows)} rings in the dataset.", sources, detail
        ),
        Q_LARGEST_BANKS: Precomputed(
            f"The largest ring ({definition}) is {top['ring_id']}, a {span}ring "
            f"with {members} member accounts across {bank_text}.",
            sources,
            detail,
        ),
        Q_LARGEST_ALERTABLE: Precomputed(
            f"The largest ring ({definition}), {top['ring_id']}, has {alertable} "
            f"alertable members out of {members} total; the rest are shown as "
            "context only.",
            sources,
            detail,
        ),
    }


def _scope_clause(name: str, scope: dict[str, Any]) -> str:
    """One scope's side of the race, in words, from the tool's own fields."""
    first = scope["first_flag_T"]
    if first is None:
        return f"{name} did not flag it at any measured horizon"
    before = scope["minutes_before_exit"]
    if before is None:
        return f"{name} first flagged it at T={first} minutes"
    when = (
        f"{before:g} minutes before the money reached the exit account"
        if before >= 0
        else f"{-before:g} minutes after the money reached the exit account"
    )
    return f"{name} first flagged it at T={first} minutes, {when}"


def race_answer(store: DataStore, ring_id: str) -> Precomputed | None:
    """The race to the exit for one detected ring, or ``None`` to use the model.

    Built only from :func:`agents.tools.ring_race_for`, so it states exactly what
    the ``get_ring_race`` tool would have returned and nothing the model would have
    had to add. ``None`` (so the model, and its own refusal wording, takes over) for
    an unknown ring, a cluster that mixes several simulated rings, or when the race
    data has not been published to this service.
    """
    from agents import tools  # noqa: PLC0415 - tools imports data like this module

    try:
        race = tools.ring_race_for(store, ring_id)
    except (KeyError, tools.RaceDataUnavailable):
        return None
    if not race["available"]:
        return None
    single, cross = race["scopes"]["single_bank"], race["scopes"]["cross_bank"]
    if race["cross_bank_earlier"]:
        verdict = (
            f"Cross-bank flagged it {race['earlier_by_minutes']:g} minutes earlier than "
            "single-bank, so joining the banks helped on this ring."
        )
    elif race["joining_helps"]:
        verdict = "Only cross-bank flagged it before the money left, so joining helped."
    elif single["first_flag_T"] is not None and single["first_flag_T"] == cross["first_flag_T"]:
        verdict = "Both flagged it at the same horizon, so joining did not make it earlier."
    else:
        verdict = "Joining the banks did not help on this ring."
    answer = (
        f"Ring {race['ring_id']}: the money reached the exit account "
        f"{race['exit_offset_minutes']:g} minutes after the ring began. "
        f"{_scope_clause('Single-bank', single)}. {_scope_clause('Cross-bank', cross)}. "
        f"{verdict} "
        f"This is one simulated evaluation run; the horizons measured are "
        f"{', '.join(f'T={t}' for t in race['horizon_grid'])} minutes."
    )
    return Precomputed(
        answer,
        ("get_ring_race",),
        ({"tool": "precomputed", "args_summary": "ring race artifact"},),
    )


def lookup(message: str, store: DataStore, ring_id: str | None = None) -> Precomputed | None:
    """The deterministic answer for ``message``, or ``None`` to use the model.

    ``ring_id`` is the ring the console has open (``ui_context.ring_id``); only the
    race-to-the-exit question uses it.
    """
    if ring_id and normalise(message) == _RACE_KEY:
        return race_answer(store, ring_id)
    prompt = _KEYS.get(normalise(message))
    if prompt is None:
        return None
    version = store.data_version(("rings",))
    with _lock:
        if version not in _cache:
            _cache.clear()
            _cache[version] = _compute(store)
        return _cache[version].get(prompt)
