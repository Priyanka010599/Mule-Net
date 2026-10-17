"""Point-in-time correctness audit for the cutoff-aware feature table.

Each test corresponds to one line of the leakage audit: age measured
against the cutoff, transfers/devices/KYC observations never read past
it, features monotone in the cutoff, and no ground-truth column other
than the two derived labels reaching the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config import MlConfig
from ml.early import cutoff_series, ring_onsets
from ml.evaluate import attach_labels
from ml.features import build_features, feature_columns, full_window_cutoff, load_tables

#: Cutoff horizon the audit is run at. Short enough that most of the
#: dataset lies after it, so a leak has somewhere to show up.
AUDIT_MINUTES = 10

#: Ground-truth columns that must never appear in the feature table.
FORBIDDEN_COLUMNS = (
    "is_mule",
    "ring_id",
    "role",
    "pattern_type",
    "cohort",
    "is_hard_negative",
    "ring_bank_count",
    "is_passthrough_mule",
    "is_complicit",
)

#: Count-like features that can only grow as the cutoff moves later.
MONOTONE_IN_CUTOFF = (
    "in_degree",
    "out_degree",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "unique_counterparties",
    "distinct_counterparty_banks",
    "amount_in_total",
    "amount_out_total",
    "device_count",
    "shared_device_count",
    "document_reuse_count",
)


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def tables(cfg: MlConfig) -> dict[str, pd.DataFrame]:
    return load_tables(cfg)


@pytest.fixture(scope="module")
def cutoff(tables: dict[str, pd.DataFrame]) -> pd.Series:
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])
    return cutoff_series(tables, onsets, AUDIT_MINUTES, seed=42)


@pytest.fixture(scope="module")
def early(cfg: MlConfig, tables: dict[str, pd.DataFrame], cutoff: pd.Series) -> pd.DataFrame:
    return build_features(cfg, tables=tables, cutoff=cutoff).set_index("hashed_account_id")


@pytest.fixture(scope="module")
def full(cfg: MlConfig, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_features(cfg, tables=tables).set_index("hashed_account_id")


def test_as_of_is_the_requested_cutoff(early: pd.DataFrame, cutoff: pd.Series) -> None:
    assert (early["as_of"] == cutoff.reindex(early.index)).all()


def test_account_age_is_measured_against_the_cutoff(
    early: pd.DataFrame, tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> None:
    opened = tables["accounts"].set_index("hashed_account_id")["opened_at"].reindex(early.index)
    expected = (cutoff.reindex(early.index) - opened).dt.total_seconds() / 86400.0
    for prefix in ("sb_", "cb_"):
        assert np.allclose(early[f"{prefix}account_age_days"], expected)
    # And it is genuinely different from the end-of-simulation value.
    assert not np.allclose(early["cb_account_age_days"], _full_window_age(tables))


def _full_window_age(tables: dict[str, pd.DataFrame]) -> np.ndarray:
    end = tables["transfers"]["timestamp"].max()
    opened = tables["accounts"].set_index("hashed_account_id")["opened_at"]
    return ((end - opened).dt.total_seconds() / 86400.0).to_numpy()


def test_no_transfer_after_the_cutoff_is_aggregated(
    early: pd.DataFrame, tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> None:
    """Recompute cross-bank degree straight from transfers and compare."""
    transfers = tables["transfers"]
    counts: dict[str, pd.Series] = {}
    sides = (("cb_in_degree", "to_hashed_account_id"), ("cb_out_degree", "from_hashed_account_id"))
    for side, column in sides:
        account = transfers[column]
        visible = transfers["timestamp"] <= account.map(cutoff)
        counts[side] = account[visible].value_counts().reindex(early.index).fillna(0)
    for side, expected in counts.items():
        assert (early[side].to_numpy() == expected.to_numpy()).all(), side


def test_device_links_respect_first_seen(
    early: pd.DataFrame, tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> None:
    devices = tables["devices"]
    visible = devices["first_seen"] <= devices["hashed_account_id"].map(cutoff)
    expected = (
        devices.loc[visible]
        .groupby("hashed_account_id")["device_fingerprint"]
        .nunique()
        .reindex(early.index)
        .fillna(0)
    )
    assert (early["cb_device_count"].to_numpy() == expected.to_numpy()).all()
    # Truncation must actually bite, or the test proves nothing.
    assert (~visible).any()


def test_kyc_links_respect_submitted_at(
    early: pd.DataFrame, tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> None:
    submitted = tables["kyc_submissions"].groupby("hashed_account_id")["submitted_at"].min()
    unsubmitted = submitted.reindex(early.index) > cutoff.reindex(early.index)
    assert (early.loc[unsubmitted, "cb_document_reuse_count"] == 0).all()
    assert (early.loc[unsubmitted, "sb_document_reuse_count"] == 0).all()


def test_features_are_monotone_in_the_cutoff(early: pd.DataFrame, full: pd.DataFrame) -> None:
    """Nothing observable at T can exceed what is observable at the horizon."""
    for prefix in ("sb_", "cb_"):
        for name in MONOTONE_IN_CUTOFF:
            column = f"{prefix}{name}"
            violations = int((early[column] > full[column] + 1e-9).sum())
            assert violations == 0, f"{column}: {violations} accounts exceed the full window"


def test_no_ground_truth_column_reaches_the_features(
    cfg: MlConfig, full: pd.DataFrame
) -> None:
    assert not set(FORBIDDEN_COLUMNS) & set(full.reset_index().columns)
    labelled = attach_labels(full.reset_index(), cfg)
    model_columns = set(feature_columns("sb_")) | set(feature_columns("cb_"))
    assert not model_columns & set(FORBIDDEN_COLUMNS)
    assert set(labelled.columns) - model_columns - {"hashed_account_id", "as_of"} == {
        "ring_id",
        "role",
        "pattern_type",
        "cohort",
        "ring_bank_count",
        "is_passthrough_mule",
        "is_complicit",
        "group",
    }


def test_ring_bank_count_is_stratification_only(cfg: MlConfig, full: pd.DataFrame) -> None:
    """Span rides on the labelled frame for reporting but is never a feature.

    ``ring_bank_count`` is the ring's bank span (1/2/3; 0 for non-mules)
    and is the key every stratified metric is cut by. It must reach the
    reporting frame and must not reach either scope's column set.
    """
    labelled = attach_labels(full.reset_index(), cfg)
    assert "ring_bank_count" in labelled.columns
    spans = labelled.loc[labelled["is_passthrough_mule"] == 1, "ring_bank_count"].unique()
    assert set(spans) <= {1, 2, 3}
    assert (labelled.loc[labelled["ring_id"] == "", "ring_bank_count"] == 0).all()
    assert "ring_bank_count" not in set(feature_columns("sb_")) | set(feature_columns("cb_"))


def test_full_window_cutoff_is_the_last_transfer(tables: dict[str, pd.DataFrame]) -> None:
    cutoff = full_window_cutoff(tables)
    assert cutoff.nunique() == 1
    assert cutoff.iloc[0] == tables["transfers"]["timestamp"].max()
