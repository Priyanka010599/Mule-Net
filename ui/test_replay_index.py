"""Item 7b: /api/replay-index is derived from the snapshot's replays, never invented."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ui.config import UiConfig
from ui.main import app
from ui.replay_index import build_replay_index

FLAGS = "flags_before_exit"


def _scope(scope: str, t: int | None, offset: float | None, caught: bool, status: str) -> dict:
    return {
        "scope": scope,
        "first_flag_T": t,
        "flag_offset_minutes": offset,
        "caught_in_time": caught,
        "flagged_full_window": t is not None,
        "status": status,
    }


def _replay(single: dict, cross: dict) -> dict:
    return {
        "label": "Replay: test",
        "available": True,
        "true_ring_id": "ring_t",
        "ambiguity": {"is_ambiguous": False},
        "exit_offset_minutes": 100.0,
        "horizon_grid": [5, 10, 30],
        "scopes": [single, cross],
    }


def _snapshot() -> dict:
    return {
        "rings": [
            {"ring_id": "a", "span": 3, "banks_involved": ["A", "B", "C"]},
            {"ring_id": "b", "span": 2, "banks_involved": ["A", "B"]},
            {"ring_id": "c", "span": 1, "banks_involved": ["A"]},
        ],
        "replays": {
            # cross earlier by 25 min
            "a": _replay(
                _scope("single_bank", 30, 30.0, True, FLAGS),
                _scope("cross_bank", 5, 5.0, True, FLAGS),
            ),
            # same time: joining does not help
            "b": _replay(
                _scope("single_bank", 5, 5.0, True, FLAGS),
                _scope("cross_bank", 5, 5.0, True, FLAGS),
            ),
            # single never flags, cross does
            "c": _replay(
                _scope("single_bank", None, None, False, "never_flagged"),
                _scope("cross_bank", 10, 10.0, True, FLAGS),
            ),
            "d": {
                "label": "Replay: test",
                "available": False,
                "unavailable_reason": "no true ring",
                "ambiguity": None,
                "scopes": [],
            },
        },
    }


def _rows() -> dict:
    return {r.ring_id: r for r in build_replay_index(_snapshot()).rings}


def test_gain_is_a_difference_of_two_replay_numbers() -> None:
    row = _rows()["a"]
    assert row.cross_bank_earlier is True
    assert row.earlier_by_minutes == 25.0
    assert row.joining_helps is True
    assert [s.minutes_before_exit for s in row.scopes] == [70.0, 95.0]
    assert row.span == 3 and row.banks_involved == ["A", "B", "C"]


def test_same_time_means_no_gain() -> None:
    row = _rows()["b"]
    assert row.cross_bank_earlier is False
    assert row.earlier_by_minutes is None
    assert row.joining_helps is False


def test_single_never_flagged_gives_no_invented_gap() -> None:
    row = _rows()["c"]
    assert row.cross_bank_earlier is None and row.earlier_by_minutes is None
    assert row.joining_helps is True  # cross caught it in time, single never did


def test_unavailable_replay_stays_null_not_zero() -> None:
    row = _rows()["d"]
    assert row.available is False
    assert row.joining_helps is None and row.scopes == []
    assert row.exit_offset_minutes is None and row.span is None


def test_groups_same_time_and_ordering() -> None:
    index = build_replay_index(_snapshot())
    assert index.cross_bank_first_flag_groups == {"5": ["a", "b"], "10": ["c"]}
    rows = {r.ring_id: r for r in index.rings}
    assert rows["a"].same_time_ring_ids == ["b"]
    assert rows["c"].same_time_ring_ids == []
    assert index.joining_helps_ring_ids == ["a", "c"]  # 25 min gain before no-gap "c"
    assert index.horizon_grid == [5, 10, 30]


@pytest.mark.skipif(
    not UiConfig.from_env().snapshot_path.exists(), reason="run `python -m ui.build_snapshot`"
)
def test_real_snapshot_ring_092_flags_in_both_scopes_at_5m_90_4_min_before_exit() -> None:
    with TestClient(app) as client:
        body = client.get("/api/replay-index").json()
    row = next(r for r in body["rings"] if r["true_ring_id"] == "ring_092")
    by_scope = {s["scope"]: s for s in row["scopes"]}
    for scope in ("single_bank", "cross_bank"):
        assert by_scope[scope]["first_flag_T"] == 5
        assert round(by_scope[scope]["minutes_before_exit"], 1) == 90.4
    assert row["cross_bank_earlier"] is False
    assert row["joining_helps"] is False
    assert row["match_is_ambiguous"] is True  # the cluster mixes 4 true rings
