"""Tests for the "one more bank joins" control and the endpoint behind it.

Written in two halves, for two different risks.

The **endpoint** half asserts the contract shape and, more importantly,
that the API *transcribes* `ml/coalition.py`'s numbers rather than
approximating them. Hard-coded literals would turn a legitimate `ml/`
re-run into a UI failure, so what is pinned is equality with the
configured source file -- the same convention as
``ui/test_api.py::test_metrics_reproduce_the_published_experiment_exactly``.

The **honesty** half is the one worth having. This control exists to make
a network-effect argument on screen, which is exactly the kind of feature
that drifts into overclaiming, so the tests below pin the things that
would make it dishonest:

* the four-bank curve must be labelled as not-a-real-partition, because
  its fourth bank is half of a real bank's customers relabelled;
* the real three-bank curve must come first, so the page opens on
  measured numbers;
* the payload must carry the caveat that `N = 1` reads 0.000 under the
  shipped rule because the bank-spread clause is uncomputable inside one
  bank -- not because one bank sees nothing;
* and it must say that this series is not the ring recall in
  `docs/claims.md`, which is the scorer's, not the rule's.

Every test is skipped rather than failed when the coalition file is
absent, matching this repo's convention for gitignored artefacts -- except
that a committed fallback ships in ``ui/data/``, so in practice it is
always present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ui.config import UiConfig
from ui.main import app
from ui.models import CoalitionPayload, CoalitionPoint, CoalitionWorld

CFG = UiConfig.from_env()

#: The file the endpoint will actually read, resolved the same way
#: ``DataSource._read_coalition`` resolves it.
SOURCE_PATH: Path = (
    CFG.coalition_path if CFG.coalition_path.exists() else CFG.coalition_fallback_path
)

_needs_coalition = pytest.mark.skipif(
    not SOURCE_PATH.exists(),
    reason="run `python -m ml.coalition` first (a fallback normally ships in ui/data/)",
)

#: The exact keys the endpoint returns. Checked exactly: an added field is
#: a contract change even when nothing reads it yet.
PAYLOAD_KEYS = {
    "generated_at",
    "source",
    "source_detail",
    "true_rings_total",
    "metric_definition",
    "not_comparable_to_claims",
    "two_variants",
    "worlds",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def body(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/coalition")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


@_needs_coalition
def test_the_endpoint_returns_exactly_the_contract_fields(body: dict[str, Any]) -> None:
    assert set(body) == PAYLOAD_KEYS
    assert body["source"] in {"ml", "fallback"}
    CoalitionPayload(**body)


@_needs_coalition
def test_every_point_validates_against_the_model(body: dict[str, Any]) -> None:
    for world in body["worlds"]:
        CoalitionWorld(**world)
        for point in world["points"]:
            CoalitionPoint(**point)


@_needs_coalition
def test_the_endpoint_transcribes_ml_s_numbers_exactly(
    body: dict[str, Any], raw: dict[str, Any]
) -> None:
    """A wrong key or a lost world is what this catches, not a re-run."""
    raw_worlds = list(raw["worlds"].values())
    assert len(body["worlds"]) == len(raw_worlds)
    for served, source in zip(body["worlds"], raw_worlds, strict=True):
        assert served["label"] == source["label"]
        assert served["is_real_partition"] == source["is_real_partition"]
        assert len(served["points"]) == len(source["by_size"])
        for point, row in zip(served["points"], source["by_size"], strict=True):
            assert point["banks_joined"] == row["banks_joined"]
            assert point["ring_recall"] == row["ring_recall"]
            assert point["ring_recall_no_spread"] == row["ring_recall_no_spread"]
            assert point["rings_caught"] == row["rings_caught"]


@_needs_coalition
def test_the_endpoint_runs_no_detector(client: TestClient) -> None:
    """It reads a file, so it must answer fast and answer identically.

    The sweep is 22 feature builds over the full transfer graph. If this
    endpoint ever grew a live recomputation, the second call would not be
    free and this would stop being a cheap assertion.
    """
    first = client.get("/api/coalition").json()
    second = client.get("/api/coalition").json()
    assert first == second


# --------------------------------------------------------------------------
# the curve is a curve, and it goes the right way
# --------------------------------------------------------------------------


@_needs_coalition
def test_each_world_covers_every_coalition_size_from_one(
    body: dict[str, Any],
) -> None:
    """No gaps: a missing size would make the toggle skip a step."""
    for world in body["worlds"]:
        sizes = [p["banks_joined"] for p in world["points"]]
        assert sizes == list(range(1, len(sizes) + 1)), world["label"]


@_needs_coalition
def test_ring_recall_rises_with_every_bank_that_joins(body: dict[str, Any]) -> None:
    """The claim the control is on screen to make.

    Non-strict at the top end on purpose: recall saturates at 1.0 once
    every bank has joined, and demanding a strict increase there would
    fail on a genuinely perfect result.
    """
    for world in body["worlds"]:
        recalls = [p["ring_recall"] for p in world["points"]]
        assert recalls == sorted(recalls), (world["label"], recalls)
        assert recalls[-1] > recalls[0], world["label"]


@_needs_coalition
def test_reachable_rings_also_rise_with_coalition_size(body: dict[str, Any]) -> None:
    """Joining widens what a coalition can even see, not just what it catches."""
    for world in body["worlds"]:
        reachable = [p["rings_reachable"] for p in world["points"]]
        assert reachable == sorted(reachable), world["label"]


@_needs_coalition
def test_rings_caught_never_exceeds_rings_reachable(body: dict[str, Any]) -> None:
    """A coalition cannot catch a ring it holds no account in."""
    for world in body["worlds"]:
        for point in world["points"]:
            assert point["rings_caught"] <= point["rings_reachable"] + 1e-9, point


@_needs_coalition
def test_the_full_coalition_is_a_single_coalition(body: dict[str, Any]) -> None:
    """There is exactly one way for every bank to have joined."""
    for world in body["worlds"]:
        assert world["points"][-1]["coalitions"] == 1, world["label"]


# --------------------------------------------------------------------------
# honesty: the things that would make this control a lie
# --------------------------------------------------------------------------


@_needs_coalition
def test_the_real_partition_comes_first(body: dict[str, Any]) -> None:
    """So the page opens on measured numbers, not split-derived ones.

    ``app.js`` defaults ``state.coalitionWorld`` to 0 and relies on this
    ordering; a viewer has to choose the synthetic curve deliberately.
    """
    assert body["worlds"][0]["is_real_partition"] is True


@_needs_coalition
def test_exactly_one_world_is_the_real_partition(body: dict[str, Any]) -> None:
    real = [w for w in body["worlds"] if w["is_real_partition"]]
    assert len(real) == 1


@_needs_coalition
def test_the_four_bank_world_is_labelled_as_not_measured(
    body: dict[str, Any],
) -> None:
    """Its fourth bank is half of a real bank's customers, relabelled.

    The page renders this note in a ``caution`` box next to the figure.
    If the flag were ever set true, the box would silently become a
    ``verified-note`` and a synthetic curve would be presented as
    measured.
    """
    synthetic = [w for w in body["worlds"] if not w["is_real_partition"]]
    assert synthetic, "the four-bank curve must be present and flagged"
    for world in synthetic:
        note = world["note"].lower()
        # The three things a reader must not be left to guess.
        assert "label" in note or "relabel" in note
        assert "same market" in note
        assert "larger than a real fourth bank" in note


@_needs_coalition
def test_the_payload_explains_the_zero_at_one_bank(body: dict[str, Any]) -> None:
    """0.000 is the rule refusing to fire, not one bank seeing nothing.

    Quoting a gain against that zero would be a true number used to say
    something false, so the explanation ships with the payload and the
    page renders it beside the toggle.
    """
    note = body["two_variants"]
    assert "uncomputable" in note
    assert "declining to fire" in note
    assert "ring_recall_no_spread" in note

    first = body["worlds"][0]["points"][0]
    assert first["banks_joined"] == 1
    assert first["ring_recall"] == 0.0
    # The generous reading is what the page quotes gains against, and it
    # must be a real number rather than the same zero.
    assert first["ring_recall_no_spread"] > 0.0


@_needs_coalition
def test_the_payload_disclaims_comparison_with_docs_claims(
    body: dict[str, Any],
) -> None:
    """This is the velocity rule's decision; claims.md is the scorer's."""
    note = body["not_comparable_to_claims"]
    assert "docs/claims.md" in note
    assert "velocity rule" in note
    assert "must not be read as one series" in note


@_needs_coalition
def test_the_definition_names_its_denominator(body: dict[str, Any]) -> None:
    """Unconditional, and it has to say so.

    ``docs/claims.md`` learned this the hard way: a coverage share quoted
    without its denominator is wrong rather than imprecise.
    """
    definition = body["metric_definition"]
    assert "every ring in the dataset" in definition
    assert "holds no account in" in definition


@_needs_coalition
def test_the_spread_across_coalitions_is_published(body: dict[str, Any]) -> None:
    """A mean over three pairings hides whether one lucky pairing carries it."""
    for world in body["worlds"]:
        for point in world["points"]:
            assert point["ring_recall_min"] <= point["ring_recall"] + 1e-9
            assert point["ring_recall"] <= point["ring_recall_max"] + 1e-9
            if point["coalitions"] == 1:
                assert point["ring_recall_min"] == point["ring_recall_max"]


@_needs_coalition
def test_the_provenance_names_the_file_that_was_opened(body: dict[str, Any]) -> None:
    """Not the configured path: on Cloud Run the live path does not exist.

    Same rule as ``MetricsPayload.source_detail`` -- a provenance line
    naming a file the process never opened is true on a dev box and false
    where the demo runs.
    """
    if body["source"] == "ml":
        assert CFG.coalition_path.as_posix() in body["source_detail"]
    else:
        assert CFG.coalition_fallback_path.name in body["source_detail"]
        assert "ml/ has not run here" in body["source_detail"]


# --------------------------------------------------------------------------
# the shipped fallback
# --------------------------------------------------------------------------


def test_a_coalition_fallback_ships_so_the_control_works_on_cloud_run() -> None:
    """``infra/cloudbuild.ignore`` excludes ``ml/experiments/`` from the image.

    Without a committed copy the control would be missing in the one
    place the demo is actually shown. Mirrors the two fallbacks
    ``ui/test_fallback_sync.py`` already guards.
    """
    assert CFG.coalition_fallback_path.exists()
    payload = json.loads(CFG.coalition_fallback_path.read_text(encoding="utf-8"))
    assert payload["worlds"]
    assert payload["true_rings_total"] > 0


def test_a_missing_coalition_file_raises_the_repo_s_own_error(tmp_path: Path) -> None:
    """Named clearly, with the command that fixes it.

    ``app.js`` fetches ``/api/coalition`` in its own try/catch so this
    failure cannot take the metrics tab down with it. This pins the
    server side of that bargain: it must raise the same error type the
    rest of the page uses, and say what to run.
    """
    from dataclasses import replace  # noqa: PLC0415 - UiConfig is frozen

    from ui.datasource import DataSource, SnapshotMissingError  # noqa: PLC0415

    missing = tmp_path / "absent.json"
    cfg = replace(
        UiConfig.from_env(),
        coalition_path=missing,
        coalition_fallback_path=missing,
    )
    with pytest.raises(SnapshotMissingError, match="no coalition sweep"):
        DataSource(cfg).coalition()
