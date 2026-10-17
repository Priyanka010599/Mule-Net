"""The race to the exit, per detected ring, with no ground truth in it.

Two consumers need the same arithmetic over the same replay data:

* ``ui/replay_index.py`` (``GET /api/replay-index``) summarises the replays the
  UI snapshot carries so a client can pick a ring where joining helps;
* the agents service's ``get_ring_race`` tool answers "when did single-bank and
  cross-bank flag this ring, against when the money left?" for the Ask panel.

The UI image ships no ``ml/`` or ``agents/`` and the agents image ships no
``ui/``, so the rule lives here, in ``shared/``, which both images copy. Nothing
below reads a file or computes a detection: each number is copied from, or is a
difference of two numbers in, the per-scope verdicts ``ui/build_snapshot.py``
already derives from ``ml/out/ring_detection_timeline.parquet``.

Definitions, all in minutes after the ring's onset unless stated:

``flag_offset_minutes``
    When the scope first flagged at a measured horizon (the horizon ``T``).
``minutes_before_exit``
    ``exit_offset_minutes - flag_offset_minutes``; positive means before the
    money reached the exit account, negative after. Null if the scope did not
    flag inside the grid, or the ring has no exit time.
``cross_bank_earlier`` / ``earlier_by_minutes``
    True when cross-bank first flagged strictly before single-bank, and the
    difference of the two offsets. Both null unless BOTH flagged in the grid.
``joining_helps``
    Cross-bank caught it before the exit AND (single-bank did not, or
    cross-bank was strictly earlier). A rule over the replay, not a metric.

The artifact (:func:`build_artifact`) is keyed by the DETECTED ring id and never
carries the simulator's ground-truth ring id, ``is_mule``, roles, cohorts or
pattern labels: the detector's ring ids and the generator's are different id
spaces, and Ask must not be able to bridge them before the UI's Reveal control.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: File name under ``agents/out/`` that the agents service reads.
ARTIFACT_NAME = "ring_race.json"

#: Bump when the artifact's shape changes, so a reader can refuse an old one.
SCHEMA_VERSION = 1

#: Said when a detected cluster has no evaluation timeline at all.
NO_TIMELINE_REASON = "No evaluation timeline exists for this detected cluster."

#: Said when a cluster merges parts of several simulated rings: any single
#: race would describe one of them, not the cluster.
AMBIGUOUS_REASON = (
    "This detected cluster mixes parts of several simulated rings, so no single "
    "race to the exit describes it."
)


def scope_view(raw: Mapping[str, Any], exit_offset: float | None) -> dict[str, Any]:
    """One scope's verdict, in the shape ``ScopeSummary`` has."""
    offset = raw.get("flag_offset_minutes")
    before = (
        round(exit_offset - offset, 2) if exit_offset is not None and offset is not None else None
    )
    return {
        "scope": raw["scope"],
        "first_flag_T": raw.get("first_flag_T"),
        "flag_offset_minutes": offset,
        "minutes_before_exit": before,
        "status": raw["status"],
        "caught_in_time": bool(raw["caught_in_time"]),
        "flagged_full_window": bool(raw["flagged_full_window"]),
    }


def race_view(
    exit_offset: float | None, raw_scopes: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """The comparison of the two scopes against the exit.

    ``raw_scopes`` are the per-scope verdicts of a replay (or of
    ``ui.build_snapshot._ring_timing``): dicts with ``scope``, ``first_flag_T``,
    ``flag_offset_minutes``, ``status``, ``caught_in_time`` and
    ``flagged_full_window``.
    """
    scopes = [scope_view(s, exit_offset) for s in raw_scopes]
    by_scope = {s["scope"]: s for s in scopes}
    single, cross = by_scope.get("single_bank"), by_scope.get("cross_bank")
    earlier: bool | None = None
    gap: float | None = None
    helps: bool | None = None
    if single and cross:
        if single["flag_offset_minutes"] is not None and cross["flag_offset_minutes"] is not None:
            gap = round(single["flag_offset_minutes"] - cross["flag_offset_minutes"], 2)
            earlier = gap > 0
        helps = cross["caught_in_time"] and (not single["caught_in_time"] or bool(earlier))
    return {
        "exit_offset_minutes": exit_offset,
        "exit_reached": exit_offset is not None,
        "scopes": scopes,
        "cross_bank_earlier": earlier,
        "earlier_by_minutes": gap if earlier else None,
        "joining_helps": helps,
    }


def entry(
    exit_offset: float | None,
    raw_scopes: list[Mapping[str, Any]] | None,
    ambiguous: bool | None,
) -> dict[str, Any]:
    """One detected cluster's artifact entry.

    ``raw_scopes`` is ``None`` when the cluster has no timeline (it matched no
    simulated ring, which is itself ground truth and so is not said). An
    ambiguous cluster is recorded as ambiguous and carries no race.
    """
    if raw_scopes is None:
        return {"available": False, "ambiguous": None, "reason": NO_TIMELINE_REASON}
    if ambiguous:
        return {"available": False, "ambiguous": True, "reason": AMBIGUOUS_REASON}
    return {"available": True, "ambiguous": False, **race_view(exit_offset, raw_scopes)}


def build_artifact(
    entries: Mapping[str, dict[str, Any]],
    *,
    horizon_grid: list[int],
    label: str,
    source: str,
) -> dict[str, Any]:
    """The JSON the agents service reads: ``{"rings": {detected_ring_id: entry}}``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "source": source,
        "horizon_grid": list(horizon_grid),
        "rings": dict(sorted(entries.items())),
    }
