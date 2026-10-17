"""Tests for the paired single-bank / cross-bank feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config import MlConfig
from ml.features import FEATURE_NAMES, build_features, feature_columns

#: Count-like features that can only grow when more banks become visible.
MONOTONE_FEATURES = (
    "in_degree",
    "out_degree",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "unique_counterparties",
    "distinct_counterparty_banks",
    "shared_device_count",
    "document_reuse_count",
)


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def features(cfg: MlConfig) -> pd.DataFrame:
    """The published feature table, rebuilt if it has not been written yet."""
    if cfg.features_path.exists():
        return pd.read_parquet(cfg.features_path)
    return build_features(cfg)


@pytest.fixture(scope="module")
def account_count(cfg: MlConfig) -> int:
    return len(pd.read_parquet(cfg.table_path("accounts"), columns=["hashed_account_id"]))


def test_one_row_per_account(features: pd.DataFrame, account_count: int) -> None:
    assert len(features) == account_count
    assert features["hashed_account_id"].is_unique


def test_no_nulls(features: pd.DataFrame) -> None:
    null_counts = features.isna().sum()
    assert not null_counts.any(), null_counts[null_counts > 0].to_dict()


def test_all_feature_values_finite(features: pd.DataFrame) -> None:
    for prefix in ("sb_", "cb_"):
        block = features[feature_columns(prefix)].to_numpy(dtype="float64")
        assert np.isfinite(block).all(), f"non-finite values in {prefix} features"


def test_both_scopes_present(features: pd.DataFrame) -> None:
    expected = {"hashed_account_id", "as_of"}
    expected |= set(feature_columns("sb_")) | set(feature_columns("cb_"))
    assert set(features.columns) == expected
    assert len(FEATURE_NAMES) * 2 + 2 == len(expected)


def test_single_bank_scope_sees_only_its_own_bank(features: pd.DataFrame) -> None:
    """sb_ must not leak cross-bank structure: at most one counterparty bank."""
    assert features["sb_distinct_counterparty_banks"].max() <= 1


def test_single_bank_never_exceeds_cross_bank(features: pd.DataFrame) -> None:
    """The single-bank view is a subgraph, so its counts can only be smaller."""
    for name in MONOTONE_FEATURES:
        violations = (features[f"sb_{name}"] > features[f"cb_{name}"]).sum()
        assert violations == 0, f"{name}: {violations} accounts where sb_ > cb_"
