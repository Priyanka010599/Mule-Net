"""The lead-time measurement must keep its three kinds of missing apart.

The whole point of this module is that "no warning" and "zero warning"
are different, and that a median over the rings that flagged is a
different number from one over all rings. Both are easy to collapse by
accident, and collapsing either inflates the result, so they are asserted
rather than trusted.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.early import CUTOFF_MINUTES
from ml.leadtime import CATEGORIES, describe, finite_horizons
from ml.timeline import SCOPES


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def per_ring(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.experiments_dir / "lead_time" / "per_ring.parquet"
    require_artefact(path, "python -m ml.leadtime")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def metrics(cfg: MlConfig) -> dict:
    path = cfg.experiments_dir / "lead_time" / "metrics.json"
    require_artefact(path, "python -m ml.leadtime")
    return json.loads(path.read_text("utf-8"))


def test_grid_is_derived_from_the_one_definition() -> None:
    """The horizon list is never hand-copied."""
    assert finite_horizons() == sorted(m for m in CUTOFF_MINUTES if m is not None)
    assert None not in finite_horizons()


def test_every_ring_has_exactly_one_category(per_ring: pd.DataFrame) -> None:
    assert set(per_ring["category"]) <= set(CATEGORIES)
    assert not per_ring.duplicated(["ring_id", "scope", "seed"]).any()


def test_warning_is_null_exactly_when_no_finite_horizon_flagged(
    per_ring: pd.DataFrame,
) -> None:
    """The load-bearing invariant: unflagged is NULL, never 0.0.

    A ring flagged only at the full window, or never flagged, has no
    meaningful warning magnitude. If either ever became 0.0 it would
    enter the means as "caught exactly on time", which is the single
    most flattering way this measurement could be wrong.
    """
    in_grid = per_ring["category"] == "flagged_in_grid"
    assert per_ring.loc[in_grid, "warning_minutes"].notna().all()
    assert per_ring.loc[in_grid, "first_flag_T"].notna().all()
    assert per_ring.loc[~in_grid, "warning_minutes"].isna().all()
    assert per_ring.loc[~in_grid, "first_flag_T"].isna().all()
    assert not (per_ring.loc[~in_grid, "warning_minutes"] == 0).any()


def test_warning_equals_exit_minus_first_flag(per_ring: pd.DataFrame) -> None:
    """Including the sign: a flag after the cash-out is negative."""
    g = per_ring.loc[per_ring["category"] == "flagged_in_grid"]
    expected = g["exit_minutes"] - g["first_flag_T"].astype("float64")
    pd.testing.assert_series_equal(
        g["warning_minutes"].astype("float64"), expected, check_names=False
    )
    # The sign is exercised by the data, not just supported in principle.
    assert (g["warning_minutes"] < 0).any(), "no late flag in the data to check the sign"


def test_first_flag_is_on_the_grid(per_ring: pd.DataFrame) -> None:
    flagged = per_ring.loc[per_ring["first_flag_T"].notna(), "first_flag_T"]
    assert set(flagged.astype(int)) <= set(finite_horizons())


def test_conditional_and_unconditional_are_reported_separately(
    metrics: dict, per_ring: pd.DataFrame
) -> None:
    """Both quantities present, with the conditional one carrying its n."""
    for seed, node in metrics["per_seed"].items():
        for scope in SCOPES:
            s = node["by_scope"][scope]
            block = per_ring.loc[(per_ring["scope"] == scope) & (per_ring["seed"] == int(seed))]
            in_grid = block["category"] == "flagged_in_grid"
            # Conditional n is the flagged count, not the ring count.
            assert s["warning_minutes_conditional"]["n"] == int(in_grid.sum())
            assert s["rings"] == len(block)
            # Unconditional denominator is every ring.
            positive = int((block.loc[in_grid, "warning_minutes"] > 0).sum())
            assert s["rings_with_positive_warning"] == positive
            assert s["share_with_positive_warning"] == pytest.approx(positive / len(block))
            assert sum(s["category_counts"].values()) == len(block)


def test_describe_ignores_nulls_rather_than_counting_them_as_zero() -> None:
    """``describe`` must drop NULLs, not coerce them."""
    assert describe(pd.Series([1.0, 3.0, pd.NA], dtype="Float64"))["n"] == 2
    assert describe(pd.Series([1.0, 3.0, pd.NA], dtype="Float64"))["p50"] == 2.0
    assert describe(pd.Series([pd.NA, pd.NA], dtype="Float64")) == {"n": 0}


def test_exit_side_censoring_is_resolved(per_ring: pd.DataFrame) -> None:
    """Every exit falls inside the grid, so no ring is right-censored.

    This is the reason the extended grid was built. If a horizon is ever
    removed and an exit falls past the end again, this fails rather than
    the shares quietly becoming free.
    """
    assert per_ring["exit_minutes"].max() <= max(finite_horizons())
