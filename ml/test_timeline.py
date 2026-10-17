"""The published ring timeline must aggregate back to the published metrics.

A per-ring breakdown of a headline number is only worth publishing if it
is the *same* number seen closer up. So the test that matters here is not
that the table is well formed -- it is that re-deriving
``ring_catch`` from the table reproduces what ``ml.early`` recorded,
**exactly**.

Exactly, not within a tolerance. Both sides are computed from the same
stored out-of-fold scores in
``ml/experiments/early_detection/oof/``, over the same 120 rings, with
the same threshold read from the same file. There is no floating-point
path between them that could legitimately introduce drift: the shares
are the same integer division on both sides. A tolerance here would be
covering for a bug.

The target is the **per-seed** row, not the ``ring_catch`` mean. The
means are over ``ml.evaluate.SEEDS`` (42, 43, 44) and include values like
104.33 rings; a one-seed boolean table cannot and should not reproduce
those, and asserting against them with a tolerance loose enough to pass
would assert nothing. ``ml.early`` now records
``ring_catch_per_seed`` for exactly this reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.early import CUTOFF_MINUTES, OOF_DIRNAME, ring_onsets
from ml.evaluate import PASSTHROUGH_MULE_ROLES
from ml.features import load_tables
from ml.timeline import CANONICAL_SEED, SCOPES, TABLE_NAME, load_oof


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def timeline(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.ml_dir / "out" / f"{TABLE_NAME}.parquet"
    require_artefact(path, "python -m ml.publish_all")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def oof_store(cfg: MlConfig) -> Path:
    """The scores every flag in the timeline was read back from.

    Checked separately from the timeline parquet: the two are written by
    different commands, so the table can outlive the evidence it was
    derived from, and a test reading one without the other would be
    asserting against a stale artefact.
    """
    path = cfg.experiments_dir / "early_detection" / OOF_DIRNAME
    require_artefact(path, "python -m ml.early")
    return path


@pytest.fixture(scope="module")
def horizons(cfg: MlConfig) -> dict[str, dict]:
    path = cfg.experiments_dir / "early_detection" / "metrics.json"
    require_artefact(path, "python -m ml.early")
    metrics = json.loads(path.read_text("utf-8"))
    return {str(h["cutoff_minutes"]): h for h in metrics["horizons"]}


@pytest.fixture(scope="module")
def onsets(cfg: MlConfig) -> pd.Series:
    tables = load_tables(cfg)
    return ring_onsets(tables["transfers"], tables["ground_truth"])


def test_timeline_covers_every_ring_horizon_and_scope(
    timeline: pd.DataFrame, onsets: pd.Series
) -> None:
    """One row per (ring, horizon, scope), no duplicates, no gaps."""
    assert len(timeline) == len(onsets) * len(CUTOFF_MINUTES) * len(SCOPES)
    assert not timeline.duplicated(["ring_id", "T", "scope"]).any()
    assert set(timeline["ring_id"]) == set(onsets.index)
    assert set(timeline["scope"]) == set(SCOPES)
    # The full window is encoded as NULL minutes; every other horizon is
    # a positive integer count of minutes.
    present = {None if pd.isna(t) else int(t) for t in timeline["T"].unique()}
    assert present == set(CUTOFF_MINUTES)
    assert timeline["flagged"].dtype == bool
    assert timeline["onset_at"].notna().all()
    assert timeline["exit_reached_at"].notna().all()
    # onset_at is a property of the ring, so it cannot vary by horizon or
    # scope; exit_reached_at likewise.
    per_ring = timeline.groupby("ring_id")[["onset_at", "exit_reached_at"]].nunique()
    assert (per_ring == 1).all().all()


@pytest.mark.parametrize("minutes", CUTOFF_MINUTES)
@pytest.mark.parametrize("scope", SCOPES)
def test_onset_plus_T_equals_the_cutoff_the_flag_was_computed_at(
    cfg: MlConfig,
    timeline: pd.DataFrame,
    oof_store: Path,
    minutes: int | None,
    scope: str,
) -> None:
    """``onset_at + T`` is the observation instant the run actually used.

    Checked against the ``as_of`` column ``ml.early.write_oof``
    persisted, **not** against a cutoff recomputed here — recomputing it
    would assert that this test's arithmetic matches this test's
    arithmetic. The stored value is the one ``cutoff_series`` produced,
    clip included.

    The clip is why the comparison is to ``min(onset + T, last transfer)``
    rather than to ``onset + T`` directly: ``cutoff_series`` bounds every
    cutoff at the end of the data, so a ring whose onset sits within T of
    the horizon is observed at the horizon instead.
    """
    oof = load_oof(cfg, minutes, scope, CANONICAL_SEED).set_index("hashed_account_id")
    tables = load_tables(cfg)
    truth = tables["ground_truth"]
    horizon_end = pd.to_datetime(tables["transfers"]["timestamp"], utc=True).max()

    rows = timeline.loc[
        (timeline["scope"] == scope)
        & (timeline["T"].isna() if minutes is None else timeline["T"] == minutes)
    ].set_index("ring_id")

    members = truth.loc[
        truth["role"].isin(PASSTHROUGH_MULE_ROLES) & truth["ring_id"].isin(rows.index),
        ["hashed_account_id", "ring_id"],
    ]
    actual = oof.loc[members["hashed_account_id"], "as_of"]
    actual.index = members["ring_id"].to_numpy()

    if minutes is None:
        expected = pd.Series(horizon_end, index=actual.index)
    else:
        onset = rows["onset_at"].reindex(actual.index)
        expected = (onset + pd.Timedelta(minutes=minutes)).clip(upper=horizon_end)

    pd.testing.assert_series_equal(
        actual.sort_index(),
        expected.sort_index(),
        check_names=False,
        check_freq=False,
    )


@pytest.mark.parametrize("minutes", CUTOFF_MINUTES)
@pytest.mark.parametrize("scope", SCOPES)
def test_aggregation_reproduces_published_per_seed_ring_catch(
    timeline: pd.DataFrame,
    horizons: dict[str, dict],
    onsets: pd.Series,
    minutes: int | None,
    scope: str,
) -> None:
    """Re-deriving ring_catch from the table equals what ml.early recorded.

    The ``in_time`` clause is rebuilt the way
    ``ml.early.catch_share_before_exit`` builds it, from the ring's onset
    and the horizon; the full window counts as after every cash-out.
    """
    published = horizons[str(minutes)]["ring_catch_per_seed"][scope][str(CANONICAL_SEED)]

    rows = timeline.loc[
        (timeline["scope"] == scope)
        & (timeline["T"].isna() if minutes is None else timeline["T"] == minutes)
    ].set_index("ring_id")
    rings = sorted(onsets.index)
    assert len(rows) == len(rings)

    if minutes is None:
        in_time = dict.fromkeys(rings, False)
    else:
        cut = onsets + pd.Timedelta(minutes=minutes)
        exits = pd.to_datetime(rows["exit_reached_at"])
        in_time = {r: bool(cut[r] < exits[r]) for r in rings}

    flagged = rows["flagged"].to_dict()
    caught = sum(1 for r in rings if flagged[r])
    caught_in_time = sum(1 for r in rings if flagged[r] and in_time[r])

    assert caught == published["rings_caught"]
    assert caught / len(rings) == published["ring_recall"]
    assert sum(in_time.values()) == published["rings_money_not_yet_at_exit"]
    assert caught_in_time == published["rings_caught_before_exit"]
    assert caught_in_time / len(rings) == published["share_caught_before_exit"]


def test_published_means_are_not_the_single_seed_target(horizons: dict[str, dict]) -> None:
    """The 3-seed means are documented as NOT reproducible from this table.

    Guards the reason the per-seed block exists: if a later change made
    ``ring_catch`` equal to seed 42's row at every horizon, the means
    would have silently stopped being means and the one-seed table would
    be masquerading as the published headline.
    """
    means = [h["ring_catch"] for h in horizons.values()]
    per_seed = [h["ring_catch_per_seed"] for h in horizons.values()]
    seeded = [
        (
            mean[scope]["rings_caught_before_exit"],
            row[scope][str(CANONICAL_SEED)]["rings_caught_before_exit"],
        )
        for mean, row in zip(means, per_seed, strict=True)
        for scope in SCOPES
    ]
    assert any(m != s for m, s in seeded), "means collapsed onto a single seed"
