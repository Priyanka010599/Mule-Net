"""``get_ring_race``: when each scope flagged a ring, against when its money left.

The tool reads ``agents/out/ring_race.json`` (written by ``ui.build_snapshot``),
keyed by DETECTED ring id. What these tests pin: the numbers are the ones
``ui/replay_index.py`` derives for the same ring (one shared rule), a cluster that
mixes several simulated rings says so and gives no race, no ground-truth id or
label is ever returned, an unknown ring or a missing artifact is a tool error and
not a crash, and the answer grounding accepts numbers that came from this tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, cases, grounding, hidden_labels, tools
from agents.test_ask_safety import ASK, _service
from agents.test_chat_claims import Tiny, tiny  # noqa: F401 - the six-account fixture
from shared import ring_race
from ui.replay_index import build_replay_index

#: A simulator ground-truth ring id (``ring_092``), as opposed to a detected one.
TRUE_ID = re.compile(r"\bring_\d+\b")


def _scope(scope: str, t: int | None, offset: float | None, caught: bool, status: str) -> dict:
    return {
        "scope": scope,
        "first_flag_T": t,
        "flag_offset_minutes": offset,
        "caught_in_time": caught,
        "flagged_full_window": t is not None,
        "status": status,
    }


#: Joining helps: cross-bank flags at T=5, single-bank only at T=180 (after the exit).
HELPS = [
    _scope("single_bank", 180, 180.0, False, "flags_after_exit"),
    _scope("cross_bank", 5, 5.0, True, "flags_before_exit"),
]
#: Both scopes flag together at T=5, well before the exit: joining does not help.
TOGETHER = [
    _scope("single_bank", 5, 5.0, True, "flags_before_exit"),
    _scope("cross_bank", 5, 5.0, True, "flags_before_exit"),
]


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty ``agents/out`` the tool reads from, and a fresh artifact cache."""
    folder = tmp_path / "agents_out"
    folder.mkdir()
    monkeypatch.setattr(cases, "DEFAULT_OUT_DIR", folder)
    tools._PUBLISHED_CACHE.clear()
    return folder


def _publish(out_dir: Path, ring_id: str, entry: dict[str, Any]) -> None:
    artifact = ring_race.build_artifact(
        {ring_id: entry}, horizon_grid=[5, 10, 180], label="Replay: test", source="test"
    )
    (out_dir / ring_race.ARTIFACT_NAME).write_text(json.dumps(artifact), encoding="utf-8")


def _walk(value: Any) -> list[str]:
    """Every key and string value inside a JSON-like result."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk(item))
    elif isinstance(value, str):
        found.append(value)
    return found


def test_the_numbers_are_the_ones_replay_index_derives(tiny: Tiny, out_dir: Path) -> None:  # noqa: F811
    replay = {
        "label": "Replay: test",
        "available": True,
        "true_ring_id": "ring_t",
        "ambiguity": {"is_ambiguous": False},
        "exit_offset_minutes": 100.0,
        "horizon_grid": [5, 10, 180],
        "scopes": HELPS,
    }
    snapshot = {"rings": [], "replays": {tiny.ring_id: replay}}
    expected = build_replay_index(snapshot).rings[0]
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))

    result = tools.get_ring_race(tiny.ring_id)

    assert result["available"] is True and result["ambiguous"] is False
    for scope in expected.scopes:
        got = result["scopes"][scope.scope]
        assert got["first_flag_T"] == scope.first_flag_T
        assert got["minutes_before_exit"] == scope.minutes_before_exit
        assert got["caught_in_time"] == scope.caught_in_time
    assert result["scopes"]["single_bank"]["minutes_before_exit"] == -80.0
    assert result["scopes"]["cross_bank"]["minutes_before_exit"] == 95.0
    assert result["scopes"]["single_bank"]["minutes_after_exit"] == 80.0
    assert result["scopes"]["cross_bank"]["minutes_after_exit"] is None
    assert result["cross_bank_earlier"] is expected.cross_bank_earlier is True
    assert result["earlier_by_minutes"] == expected.earlier_by_minutes == 175.0
    assert result["joining_helps"] is expected.joining_helps is True
    assert result["exit_reached"] is True and result["exit_offset_minutes"] == 100.0
    assert result["span"] == len(result["banks_involved"])


def test_both_scopes_flagging_together_is_not_a_gain(tiny: Tiny, out_dir: Path) -> None:  # noqa: F811
    _publish(out_dir, tiny.ring_id, ring_race.entry(95.42, TOGETHER, False))
    result = tools.get_ring_race(tiny.ring_id)
    assert result["cross_bank_earlier"] is False
    assert result["earlier_by_minutes"] is None
    assert result["joining_helps"] is False
    assert result["scopes"]["single_bank"]["minutes_before_exit"] == 90.42


def test_an_ambiguous_cluster_says_so_and_gives_no_race(tiny: Tiny, out_dir: Path) -> None:  # noqa: F811
    _publish(out_dir, tiny.ring_id, ring_race.entry(95.42, TOGETHER, True))
    result = tools.get_ring_race(tiny.ring_id)
    assert result["ambiguous"] is True and result["available"] is False
    assert "mixes parts of several" in result["reason"]
    for withheld in ("scopes", "joining_helps", "cross_bank_earlier", "exit_offset_minutes"):
        assert withheld not in result


def test_a_cluster_with_no_timeline_is_unavailable_not_invented(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(None, None, None))
    result = tools.get_ring_race(tiny.ring_id)
    assert result["available"] is False and result["ambiguous"] is None
    assert "scopes" not in result


@pytest.mark.parametrize("scopes", [HELPS, TOGETHER])
def test_no_ground_truth_id_or_label_is_returned(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,
    scopes: list[dict],
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, scopes, False))
    result = adk_agents._model_tool(tools.get_ring_race)(tiny.ring_id)
    words = _walk(result)
    denied = hidden_labels.DENIED_COLUMNS | {"true_ring_id", "ground_truth", "is_mule"}
    assert not {w.lower() for w in words} & denied
    assert not [w for w in words if TRUE_ID.search(w)]


def test_an_unknown_ring_is_a_tool_error_not_a_guess(tiny: Tiny, out_dir: Path) -> None:  # noqa: F811
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))
    out = adk_agents._model_tool(tools.get_ring_race)("ring-mulenet-velocity-rule-v2-99999")
    assert out["error"]["error_type"] == "KeyError"
    assert "unknown ring" in out["error"]["message"]
    assert out["error"]["recoverable"] is True


def test_a_missing_artifact_is_a_clear_tool_error(tiny: Tiny, out_dir: Path) -> None:  # noqa: F811
    out = adk_agents._model_tool(tools.get_ring_race)(tiny.ring_id)
    assert out["error"]["error_type"] == "RaceDataUnavailable"
    assert "race data unavailable" in out["error"]["message"]


def test_an_artifact_without_this_ring_or_with_a_foreign_schema_is_unavailable(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,
) -> None:
    _publish(out_dir, "ring-some-other", ring_race.entry(100.0, HELPS, False))
    out = adk_agents._model_tool(tools.get_ring_race)(tiny.ring_id)
    assert "no entry for it" in out["error"]["message"]
    (out_dir / ring_race.ARTIFACT_NAME).write_text(json.dumps({"schema_version": 99}))
    out = adk_agents._model_tool(tools.get_ring_race)(tiny.ring_id)
    assert "unexpected schema" in out["error"]["message"]


def test_the_tool_is_registered_for_ask_and_in_the_tool_list() -> None:
    assert tools.get_ring_race in tools.ALL_TOOLS
    assert tools.get_ring_race in adk_agents.AGENT_TOOLS["analyst_qa"]


def test_grounding_accepts_numbers_from_the_tool_and_refuses_others(
    tiny: Tiny,  # noqa: F811
    out_dir: Path,
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))
    result = adk_agents._model_tool(tools.get_ring_race)(tiny.ring_id)
    grounded = "Cross-bank flagged at T=5, 95 minutes before the exit; single-bank 80 after."
    assert grounding.ungrounded_numbers(grounded, "q", [result]) == []
    invented = "Cross-bank flagged 42 minutes before the exit."
    assert grounding.ungrounded_numbers(invented, "q", [result]) == ["42"]


def test_a_race_answer_is_served_with_the_tool_in_its_sources(
    monkeypatch: pytest.MonkeyPatch,
    tiny: Tiny,  # noqa: F811
    out_dir: Path,
) -> None:
    _publish(out_dir, tiny.ring_id, ring_race.entry(100.0, HELPS, False))
    reply = {
        "answer": "Cross-bank flagged it at T=5, 95 minutes before the exit.",
        "claims": [],
    }
    client = _service(monkeypatch, [(tools.get_ring_race, (tiny.ring_id,))], reply)
    body = client.post("/chat", json=ASK).json()
    assert "get_ring_race" in body["sources"]
    assert isinstance(client, TestClient)


# ----------------------------------------------------- the real published data

ROOT = Path(__file__).resolve().parent.parent


def _real_artifact() -> dict[str, Any]:
    pytest.importorskip("pyarrow")
    from ui.build_snapshot import build_ring_race  # noqa: PLC0415 - needs pandas
    from ui.config import UiConfig  # noqa: PLC0415

    cfg = UiConfig.from_env()
    needed = ("ring_detection_timeline", "ring_id_map", "rings")
    if not all(cfg.ml_table_path(t).exists() for t in needed):
        pytest.skip("ml/out is not published; run `python -m ml.publish_all`")
    return build_ring_race(cfg)


def test_the_real_artifact_covers_every_detected_cluster_with_no_ground_truth() -> None:
    artifact = _real_artifact()
    text = json.dumps(artifact)
    assert not TRUE_ID.search(text), "a simulator ring id leaked into the artifact"
    assert "true_ring" not in text and "is_mule" not in text
    assert all(key.startswith("ring-") for key in artifact["rings"])
    ambiguous = [k for k, v in artifact["rings"].items() if v["ambiguous"]]
    assert ambiguous, "expected the ambiguous clusters to be recorded as ambiguous"
    for key in ambiguous:
        assert "scopes" not in artifact["rings"][key]


def test_the_real_artifact_agrees_with_the_ui_snapshots_replays() -> None:
    artifact = _real_artifact()
    snapshot_path = ROOT / "ui" / "out" / "demo_snapshot.json"
    if not snapshot_path.exists():
        pytest.skip("ui/out/demo_snapshot.json is not built; run `python -m ui.build_snapshot`")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    compared = 0
    for row in build_replay_index(snapshot).rings:
        entry = artifact["rings"].get(row.ring_id)
        if not row.available or row.match_is_ambiguous or entry is None:
            continue
        compared += 1
        assert entry["joining_helps"] == row.joining_helps
        assert entry["cross_bank_earlier"] == row.cross_bank_earlier
        assert entry["earlier_by_minutes"] == row.earlier_by_minutes
        for scope in row.scopes:
            got = next(s for s in entry["scopes"] if s["scope"] == scope.scope)
            assert got["first_flag_T"] == scope.first_flag_T
            assert got["minutes_before_exit"] == scope.minutes_before_exit
    assert compared > 0
