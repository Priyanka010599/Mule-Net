"""Item 7a: ground-truth labels in /api/banks are an explicit, testable choice."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ui.config import UiConfig
from ui.main import app

pytestmark = pytest.mark.skipif(
    not UiConfig.from_env().snapshot_path.exists(),
    reason="run `python -m ui.build_snapshot` first",
)

#: Every tile-level and queue-level field derived from the simulator's labels.
TILE_LABEL_FIELDS = ("mules_in_bank", "alert_true_positives", "alert_false_positives")
QUEUE_LABEL_FIELDS = (
    "queue_depth",
    "queue_true_positives",
    "queue_false_positives",
    "false_alerts_per_catch",
    "reached_recall",
)


def _assert_nothing_leaks(tiles: list[dict]) -> None:
    assert tiles
    for tile in tiles:
        assert tile["labels_revealed"] is False
        for field in TILE_LABEL_FIELDS:
            assert tile[field] is None, field
        for field in QUEUE_LABEL_FIELDS:
            assert tile["queue"][field] is None, field
        assert tile["alerts"], "alerts (scores) are not ground truth and stay"
        assert all(a["is_mule"] is None for a in tile["alerts"])
        assert tile["queue"]["recall_target"] is not None  # a parameter, not a label


def test_default_payload_today_carries_ground_truth_and_says_so() -> None:
    """Documents precisely what leaks by default: backward compatible, flagged."""
    with TestClient(app) as client:
        tiles = client.get("/api/banks").json()
    for tile in tiles:
        assert tile["labels_revealed"] is True
        assert isinstance(tile["mules_in_bank"], int)
        assert isinstance(tile["alert_true_positives"], int)
        assert isinstance(tile["queue"]["queue_true_positives"], int)
        assert all(isinstance(a["is_mule"], bool) for a in tile["alerts"])


def test_reveal_labels_0_withholds_everything() -> None:
    with TestClient(app) as client:
        _assert_nothing_leaks(client.get("/api/banks?reveal_labels=0").json())


def test_flipped_default_leaks_nothing_until_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULENET_UI_LABELS_DEFAULT_REVEALED", "0")
    with TestClient(app) as client:
        _assert_nothing_leaks(client.get("/api/banks").json())
        opted = client.get("/api/banks?reveal_labels=1").json()
    assert all(t["labels_revealed"] and isinstance(t["mules_in_bank"], int) for t in opted)


def test_withholding_changes_only_label_fields() -> None:
    with TestClient(app) as client:
        shown = client.get("/api/banks?reveal_labels=1").json()
        hidden = client.get("/api/banks?reveal_labels=0").json()
    for a, b in zip(shown, hidden, strict=True):
        assert a["bank_id"] == b["bank_id"]
        assert [x["hashed_account_id"] for x in a["alerts"]] == [
            x["hashed_account_id"] for x in b["alerts"]
        ]
        assert [x["score"] for x in a["alerts"]] == [x["score"] for x in b["alerts"]]
        assert a["queue"]["recall_target"] == b["queue"]["recall_target"]
        assert a["account_count"] == b["account_count"]
