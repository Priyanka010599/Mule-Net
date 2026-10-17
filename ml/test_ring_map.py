"""Guarantees on the detector->ground-truth ring id map.

The UI is about to present these pairings to a judge as "this is the ring
being replayed", so the properties worth asserting are the ones that stop
a wrong claim reaching a screen: that the mapping is total and unique on
the detected side, that the false-positive cluster is still present
rather than quietly dropped, that the pairings actually resolve in the
timeline, and that every case the pairing cannot justify is flagged.

The tie-break is asserted directly, because a pairing that depends on
dict or file order would change between runs on identical inputs.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.ring_map import TABLE_NAME, best_match, build_map
from ml.timeline import TABLE_NAME as TIMELINE_TABLE


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def ring_map(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.ml_dir / "out" / f"{TABLE_NAME}.parquet"
    require_artefact(path, "python -m ml.publish_all")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def rings(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.ml_dir / "out" / "rings.parquet"
    require_artefact(path, "python -m ml.publish_all")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def timeline(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.ml_dir / "out" / f"{TIMELINE_TABLE}.parquet"
    require_artefact(path, "python -m ml.publish_all")
    return pd.read_parquet(path)


def test_one_row_per_detected_ring(ring_map: pd.DataFrame, rings: pd.DataFrame) -> None:
    """Total and unique on the detected side: every ring exactly once."""
    assert len(ring_map) == len(rings)
    assert not ring_map["detected_ring_id"].duplicated().any()
    assert set(ring_map["detected_ring_id"]) == set(rings["ring_id"])


def test_unmatched_detected_ring_is_kept_with_a_null_true_id(
    ring_map: pd.DataFrame,
) -> None:
    """A false-positive cluster is a finding, not a row to drop."""
    unmatched = ring_map.loc[ring_map["true_ring_id"].isna()]
    assert len(unmatched) >= 1, "the zero-overlap detected ring is missing"
    # Nothing is claimed for it: no overlap, no true ring, no size.
    assert (unmatched["overlap_members"] == 0).all()
    assert (unmatched["true_rings_touched"] == 0).all()
    assert unmatched["true_ring_size"].isna().all()
    # Its own size is still real — it is a cluster of accounts.
    assert (unmatched["detected_ring_size"] > 0).all()


def test_double_claimed_true_ring_is_flagged(ring_map: pd.DataFrame) -> None:
    """If two detected rings claim one true ring, both rows say so.

    Consistency between the recorded count and the actual pairings, so
    the flag cannot drift away from the data it describes.
    """
    claims = Counter(ring_map["true_ring_id"].dropna())
    for _, row in ring_map.iterrows():
        if row["true_ring_id"] is None or pd.isna(row["true_ring_id"]):
            continue
        assert row["detected_rings_claiming_true_ring"] == claims[row["true_ring_id"]]
        if claims[row["true_ring_id"]] > 1:
            assert row["match_is_ambiguous"], "a shared replay must be caveated"


def test_every_matched_true_ring_resolves_in_the_timeline(
    ring_map: pd.DataFrame, timeline: pd.DataFrame
) -> None:
    """The join the table exists to enable actually lands."""
    matched = set(ring_map["true_ring_id"].dropna())
    assert matched, "no pairings at all"
    assert matched <= set(timeline["ring_id"])


def test_ambiguous_flag_covers_every_unjustifiable_pairing(
    ring_map: pd.DataFrame,
) -> None:
    """``match_is_ambiguous`` is exactly the disjunction it claims to be."""
    expected = (
        ring_map["overlap_tied"]
        | (ring_map["true_rings_touched"] > 1)
        | (ring_map["detected_rings_touching_true_ring"] > 1)
    )
    pd.testing.assert_series_equal(ring_map["match_is_ambiguous"], expected, check_names=False)


def test_overlap_never_exceeds_either_ring(ring_map: pd.DataFrame) -> None:
    """A sanity bound: overlap is an intersection of both memberships."""
    assert (ring_map["overlap_members"] <= ring_map["detected_ring_size"]).all()
    matched = ring_map.loc[ring_map["true_ring_id"].notna()]
    assert (matched["overlap_members"] <= matched["true_ring_size"]).all()
    assert (matched["overlap_members"] > 0).all()


def test_tie_break_is_deterministic_and_order_independent() -> None:
    """Most overlap wins; an exact tie goes to the smallest true ring id.

    Asserted on the pure function with shuffled inputs rather than on the
    table, because the table happens to contain only one tie and a
    regression here would otherwise depend on that one row surviving.
    """
    # Clear winner regardless of insertion order.
    assert best_match(Counter({"ring_009": 2, "ring_001": 5})) == ("ring_001", 5, False)
    assert best_match(Counter({"ring_001": 5, "ring_009": 2})) == ("ring_001", 5, False)
    # Exact tie: smallest id wins, from either insertion order, and the
    # tie is reported rather than hidden.
    assert best_match(Counter({"ring_058": 6, "ring_016": 6})) == ("ring_016", 6, True)
    assert best_match(Counter({"ring_016": 6, "ring_058": 6})) == ("ring_016", 6, True)
    # No overlap at all.
    assert best_match(Counter()) == (None, 0, False)


def test_rebuilding_reproduces_the_published_table(cfg: MlConfig, ring_map: pd.DataFrame) -> None:
    """The published table is what the current code produces."""
    rebuilt = build_map(cfg)
    pd.testing.assert_frame_equal(
        rebuilt.sort_values("detected_ring_id").reset_index(drop=True),
        ring_map.sort_values("detected_ring_id").reset_index(drop=True),
    )
