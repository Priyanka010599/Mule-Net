"""Which detected rings does cross-bank joining help on, and by how much?

``GET /api/replay-index`` summarises the per-ring replays the snapshot already
carries (built from ``ml/out/ring_detection_timeline.parquet`` and
``ring_id_map.parquet`` by ``ui/build_snapshot.py``) so a client can pick a ring
where joining helps without fetching every replay. Nothing here is computed from
raw transfers or invented: each number is copied from, or is a difference of two
numbers in, a replay or ring row, and a value the replay does not have is null.

The arithmetic itself lives in ``shared/ring_race.py``, which the agents
service's ``get_ring_race`` tool uses too. Its definitions, all in minutes after
the ring's onset unless stated:

``flag_offset_minutes``
    When the scope first flagged at a measured horizon (the horizon ``T``).
``minutes_before_exit``
    ``exit_offset_minutes - flag_offset_minutes``; null if the scope did not
    flag inside the grid, or the ring has no exit time.
``cross_bank_earlier`` / ``earlier_by_minutes``
    True when cross-bank first flagged strictly before single-bank, and the
    difference of the two offsets. Both null unless BOTH flagged in the grid.
``joining_helps``
    Cross-bank caught it before the exit AND (single-bank did not, or
    cross-bank was strictly earlier). A rule over the replay, not a metric.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.ring_race import race_view


class ScopeSummary(BaseModel):
    scope: Literal["single_bank", "cross_bank"]
    #: Horizon ``T`` (minutes) at which this scope first flagged; null if never in the grid.
    first_flag_T: int | None = None
    flag_offset_minutes: float | None = None
    minutes_before_exit: float | None = None
    status: str
    caught_in_time: bool
    flagged_full_window: bool


class RingReplaySummary(BaseModel):
    ring_id: str
    true_ring_id: str | None = None
    available: bool
    unavailable_reason: str = ""
    #: The detected cluster mixes several true rings, so its replay shows the largest.
    match_is_ambiguous: bool | None = None
    #: Null for a cluster the snapshot's ring index does not carry.
    span: int | None = None
    banks_involved: list[str] = Field(default_factory=list)
    exit_offset_minutes: float | None = None
    scopes: list[ScopeSummary] = Field(default_factory=list)
    cross_bank_earlier: bool | None = None
    earlier_by_minutes: float | None = None
    joining_helps: bool | None = None
    #: Other rings whose cross-bank scope first flagged at the same horizon ``T``.
    same_time_ring_ids: list[str] = Field(default_factory=list)


class ReplayIndex(BaseModel):
    label: str
    horizon_grid: list[int]
    rings: list[RingReplaySummary]
    #: Cross-bank first-flag horizon ``T`` (as a string key) -> ring ids that flagged then.
    cross_bank_first_flag_groups: dict[str, list[str]]
    #: Rings where ``joining_helps`` is true, best (largest ``earlier_by_minutes``) first.
    joining_helps_ring_ids: list[str]


def _summarise(
    ring_id: str, replay: dict[str, Any], ring: dict[str, Any] | None
) -> RingReplaySummary:
    race = race_view(replay.get("exit_offset_minutes"), replay.get("scopes", []))
    ambiguity = replay.get("ambiguity")
    return RingReplaySummary(
        ring_id=ring_id,
        true_ring_id=replay.get("true_ring_id"),
        available=bool(replay["available"]),
        unavailable_reason=replay.get("unavailable_reason", ""),
        match_is_ambiguous=ambiguity["is_ambiguous"] if ambiguity else None,
        span=ring["span"] if ring else None,
        banks_involved=list(ring["banks_involved"]) if ring else [],
        exit_offset_minutes=race["exit_offset_minutes"],
        scopes=[ScopeSummary(**s) for s in race["scopes"]],
        cross_bank_earlier=race["cross_bank_earlier"],
        earlier_by_minutes=race["earlier_by_minutes"],
        joining_helps=race["joining_helps"],
    )


def build_replay_index(snapshot: dict[str, Any]) -> ReplayIndex:
    rings = {r["ring_id"]: r for r in snapshot["rings"]}
    replays: dict[str, dict[str, Any]] = snapshot["replays"]
    rows = [_summarise(rid, rp, rings.get(rid)) for rid, rp in sorted(replays.items())]

    groups: dict[str, list[str]] = {}
    for row in rows:
        cross = next((s for s in row.scopes if s.scope == "cross_bank"), None)
        if cross and cross.first_flag_T is not None:
            groups.setdefault(str(cross.first_flag_T), []).append(row.ring_id)
    for row in rows:
        cross = next((s for s in row.scopes if s.scope == "cross_bank"), None)
        if cross and cross.first_flag_T is not None:
            peers = groups[str(cross.first_flag_T)]
            row.same_time_ring_ids = [r for r in peers if r != row.ring_id]

    helped = [r for r in rows if r.joining_helps]
    helped.sort(key=lambda r: (-(r.earlier_by_minutes or 0.0), r.ring_id))
    first = next((rp for rp in replays.values() if rp.get("horizon_grid")), {})
    return ReplayIndex(
        label=next(iter(replays.values()))["label"] if replays else "",
        horizon_grid=list(first.get("horizon_grid", [])),
        rings=rows,
        cross_bank_first_flag_groups=dict(sorted(groups.items(), key=lambda kv: int(kv[0]))),
        joining_helps_ring_ids=[r.ring_id for r in helped],
    )
