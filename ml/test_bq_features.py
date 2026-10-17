"""Tests for the BigQuery re-implementation of the graph features.

Two kinds of test live here.

The ones that need no network check the things a typo would break and the
one arithmetic fact this module exists to pin down: that
``ml/features.py`` — the canonical path — loses money precision by
casting ``Decimal`` to ``float64`` before summing. That is checked
locally against :func:`ml.bq_features.exact_decimal_totals`, so it is a
standing assertion rather than a note in a report.

The ones that need BigQuery re-resolve the artefacts a live run wrote
(``ml/experiments/bq_features/parity.json`` and ``leakage.json``) rather
than issuing queries from the test suite. Same reason ``ml/conftest.py``
exists: a test that costs money and three minutes per query does not
belong in ``pytest ml/``, but its *verdict* does, so the run is separate
and the assertions here are on its recorded output. ``MULENET_REQUIRE_OOF=1``
turns a missing artefact from a skip into a failure.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import pytest

from ml.bq_features import (
    CORRUPTED_POST_CUTOFF_SOURCE,
    DECISION_RESOLUTION,
    DROPPED_POST_CUTOFF_SOURCE,
    INTEGER_FEATURES,
    MONEY_FEATURES,
    SQL_FEATURE_NAMES,
    UNREPRODUCIBLE,
    build_query,
    exact_decimal_totals,
    staggered_cutoff,
)
from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.features import FEATURE_NAMES, SCOPES, load_tables

REGENERATE_PARITY = "python -m ml.bq_features"
REGENERATE_LEAKAGE = "python -m ml.bq_features --leakage"


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def tables(cfg: MlConfig) -> dict[str, pd.DataFrame]:
    return load_tables(cfg)


# --------------------------------------------------------------------------
# offline: the SQL text and the feature inventory
# --------------------------------------------------------------------------


def test_feature_inventory_is_derived_from_features_py() -> None:
    """Every canonical feature is either reproduced or listed as not."""
    assert set(SQL_FEATURE_NAMES) | set(UNREPRODUCIBLE) == set(FEATURE_NAMES)
    assert not set(SQL_FEATURE_NAMES) & set(UNREPRODUCIBLE)
    # Order follows FEATURE_NAMES so the SQL column order is not accidental.
    assert list(SQL_FEATURE_NAMES) == [n for n in FEATURE_NAMES if n not in UNREPRODUCIBLE]


def test_every_inexact_feature_has_a_decision_resolution() -> None:
    """A float feature with no stated decision step cannot be given a tolerance."""
    for name in SQL_FEATURE_NAMES:
        if name in INTEGER_FEATURES:
            assert name not in DECISION_RESOLUTION, f"{name} is integral; no tolerance applies"
        else:
            assert name in DECISION_RESOLUTION, f"{name} needs a stated decision step"


@pytest.mark.parametrize("scope", sorted(SCOPES))
def test_query_renders_per_scope(scope: str) -> None:
    sql = build_query(scope, "SELECT * FROM t", -1.0)
    assert f"'{scope}' AS scope" in sql
    # The point-in-time predicate must be present and must compare against
    # the account's own cutoff, not a global one.
    assert "WHERE e.timestamp <= d.as_of" in sql
    assert "@gap_seconds" in sql
    for name in SQL_FEATURE_NAMES:
        assert f" {name}" in sql, name


def test_single_bank_scope_filters_to_one_bank() -> None:
    assert "WHERE from_bank = to_bank" in build_query("single_bank", "t", -1.0)
    assert "from_bank = to_bank" not in build_query("cross_bank", "t", -1.0)


def test_unknown_scope_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_query("all_banks", "t", -1.0)


def test_leakage_sources_reference_the_cutoff_not_a_literal() -> None:
    """The probe instant is derived from @as_ofs; nothing is interpolated."""
    for template in (DROPPED_POST_CUTOFF_SOURCE, CORRUPTED_POST_CUTOFF_SOURCE):
        assert "UNNEST(@as_ofs)" in template
    assert "timestamp <=" in DROPPED_POST_CUTOFF_SOURCE
    assert "timestamp >" not in DROPPED_POST_CUTOFF_SOURCE
    # The corrupted variant must both rewrite and duplicate post-cutoff rows.
    assert "NUMERIC '7'" in CORRUPTED_POST_CUTOFF_SOURCE
    assert "-dup" in CORRUPTED_POST_CUTOFF_SOURCE


# --------------------------------------------------------------------------
# offline: the exact-Decimal reference, and what it says about pandas
# --------------------------------------------------------------------------


def test_exact_decimal_totals_respects_each_account_cutoff() -> None:
    """A transfer after an account's cutoff must not reach its total."""
    transfers = pd.DataFrame(
        {
            "from_hashed_account_id": ["a", "a", "b"],
            "to_hashed_account_id": ["b", "b", "a"],
            "amount": [Decimal("10.01"), Decimal("20.02"), Decimal("5.55")],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-05T00:00:00Z", "2025-01-02T00:00:00Z"]
            ),
        }
    )
    cutoff = pd.Series(
        pd.to_datetime(["2025-01-03T00:00:00Z", "2025-01-09T00:00:00Z"]),
        index=pd.Index(["a", "b"], name="hashed_account_id"),
    )
    totals = exact_decimal_totals(transfers, cutoff)
    # a's cutoff is before its second payment, so only 10.01 leaves.
    assert totals.loc["a", "amount_out_total"] == Decimal("10.01")
    assert totals.loc["a", "amount_in_total"] == Decimal("5.55")
    # b sees both credits.
    assert totals.loc["b", "amount_in_total"] == Decimal("30.03")


def test_exact_decimal_totals_returns_decimals_not_floats() -> None:
    """The reference must stay in Decimal; one float anywhere voids it."""
    amounts = [Decimal("0.10"), Decimal("0.20"), Decimal("0.30")]
    transfers = pd.DataFrame(
        {
            "from_hashed_account_id": ["a"] * 3,
            "to_hashed_account_id": ["b"] * 3,
            "amount": amounts,
            "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3),
        }
    )
    cutoff = pd.Series(pd.to_datetime(["2025-01-02T00:00:00Z"] * 2), index=pd.Index(["a", "b"]))
    total = exact_decimal_totals(transfers, cutoff).loc["b", "amount_in_total"]
    assert isinstance(total, Decimal)
    assert total == Decimal("0.60")


def test_canonical_pandas_money_totals_are_not_exact(cfg: MlConfig, tables) -> None:
    """A standing record of the finding, not a note in a report.

    ``ml.features._edge_events`` does ``transfers["amount"].astype("float64")``
    before summing, so the canonical money totals carry float error. This
    pins the size of it. It deliberately does **not** demand that the
    canonical path change: pandas stays canonical, and $6e-11 on a cent-
    quantised column changes no decision (``DECISION_RESOLUTION`` puts the
    money step at one cent). If someone ever makes the pandas path exact,
    this test fails and should be deleted with the cast.
    """
    require_artefact(cfg.features_path, "python -m ml.features")
    features = pd.read_parquet(cfg.features_path).set_index("hashed_account_id").sort_index()
    cutoff = pd.Series(
        tables["transfers"]["timestamp"].max(),
        index=pd.Index(tables["accounts"]["hashed_account_id"]),
    )
    exact = exact_decimal_totals(tables["transfers"], cutoff).sort_index()

    inexact_columns = 0
    for name in sorted(MONEY_FEATURES):
        delta = (
            features[f"cb_{name}"].astype("float64") - exact[name].map(float).astype("float64")
        ).abs()
        assert delta.max() < DECISION_RESOLUTION[name], (
            f"cb_{name}: pandas float error {delta.max():g} now exceeds one cent"
        )
        inexact_columns += int(delta.max() > 0.0)
    assert inexact_columns == len(MONEY_FEATURES), (
        "the pandas money totals are exact now; drop this test and the "
        "float64 cast note in ml/bq_features.py"
    )


def test_staggered_cutoff_stays_inside_the_window(tables) -> None:
    timestamps = tables["transfers"]["timestamp"]
    cutoff = staggered_cutoff(tables, steps=8)
    assert cutoff.nunique() == 8
    assert cutoff.min() >= timestamps.min()
    assert cutoff.max() <= timestamps.max()


def test_staggered_cutoff_latest_leaves_post_cutoff_rows(tables) -> None:
    """``latest=0.5`` must leave transfers for the leakage check to corrupt."""
    half = staggered_cutoff(tables, latest=0.5)
    assert (tables["transfers"]["timestamp"] > half.max()).sum() > 0
    assert (tables["transfers"]["timestamp"] > staggered_cutoff(tables).max()).sum() == 0


# --------------------------------------------------------------------------
# live: the recorded verdicts of a real BigQuery run
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity(cfg: MlConfig) -> list[dict]:
    path = cfg.experiments_dir / "bq_features" / "parity.json"
    require_artefact(path, REGENERATE_PARITY)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def leakage(cfg: MlConfig) -> dict:
    path = cfg.experiments_dir / "bq_features" / "leakage.json"
    require_artefact(path, REGENERATE_LEAKAGE)
    return json.loads(path.read_text(encoding="utf-8"))


def test_parity_covers_both_cutoff_scenarios(parity: list[dict]) -> None:
    """Full window alone proves nothing: it makes the cutoff filter a no-op."""
    assert {r["scenario"] for r in parity} == {"full_window", "staggered"}


def test_parity_ran_on_every_account(parity: list[dict]) -> None:
    for report in parity:
        assert report["accounts"] == 30000, report["scenario"]


def test_counts_match_exactly(parity: list[dict]) -> None:
    """No tolerance on an integer. Ever."""
    for report in parity:
        for feature in report["features"]:
            name = feature["column"][3:]
            if name in INTEGER_FEATURES:
                assert feature["max_abs_delta"] == 0.0, (report["scenario"], feature)
                assert feature["mismatched_rows"] == 0, (report["scenario"], feature)


def test_sql_money_matches_exact_decimal(parity: list[dict]) -> None:
    """The NUMERIC sums are exact; the disagreement is on the pandas side."""
    for report in parity:
        attribution = report["money_attribution"]
        assert attribution, report["scenario"]
        for column, verdict in attribution.items():
            assert verdict["sql_is_exact"], (report["scenario"], column, verdict)
            assert verdict["sql_numeric_vs_exact_decimal_mismatched_rows"] == 0
            assert verdict["pandas_max_abs_error_usd"] < DECISION_RESOLUTION["amount_in_total"]


def test_float_features_are_within_their_decision_step(parity: list[dict]) -> None:
    """A tolerance is only acceptable below the step that changes a decision."""
    for report in parity:
        for feature in report["features"]:
            name = feature["column"][3:]
            if name in INTEGER_FEATURES or name in MONEY_FEATURES:
                continue
            limit = DECISION_RESOLUTION[name]
            assert feature["max_abs_delta"] < limit, (report["scenario"], feature)


def test_unreproducible_features_are_named(parity: list[dict]) -> None:
    for report in parity:
        assert set(report["unreproducible_features"]) == set(UNREPRODUCIBLE)
        assert set(report["reproduced_features"]) == set(SQL_FEATURE_NAMES)


def test_cost_is_recorded_per_query(parity: list[dict]) -> None:
    """Bytes scanned is the checkable figure; assert it is there and sane."""
    for report in parity:
        assert len(report["queries"]) == len(SCOPES)
        for query in report["queries"]:
            assert query["bytes_processed"] > 0
            assert not query["cache_hit"], "a cached query measures nothing"
            assert query["usd"] < 0.01


def test_no_post_cutoff_leak(leakage: dict) -> None:
    """The test that matters more than the equality test.

    A point-in-time leak does not show up as a mismatch against pandas —
    it shows up as a better-looking feature, and if both implementations
    read the same future rows they agree while both being wrong. So this
    holds the query fixed and changes the data: with every cutoff at the
    midpoint of the window, deleting every post-cutoff transfer, and
    separately rewriting and duplicating every post-cutoff transfer, must
    leave all 30,000 feature rows identical.
    """
    assert leakage["post_cutoff_transfers"] > 0, "nothing was after the cutoff; test is vacuous"
    assert leakage["dropped_matches_baseline"], "deleting post-cutoff rows changed a feature"
    assert leakage["corrupted_matches_baseline"], "corrupting post-cutoff rows changed a feature"
    assert leakage["verdict"]


def test_leakage_control_proves_the_query_reads_transfers(leakage: dict) -> None:
    """Without this, a query that read nothing would pass the test above."""
    assert leakage["control_is_live"]
    assert leakage["control_accounts_changed"] > 0
