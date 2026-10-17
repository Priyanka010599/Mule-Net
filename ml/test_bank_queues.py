"""``bank_alert_queues``: single-bank scope only, additive, and self-consistent.

The real-file tests read the published ``ml/out`` and skip (or fail under
``MULENET_REQUIRE_OOF=1``) when it is absent. The five contract tables are
pinned by sha256 of the bytes they had when this table was added: if any
regeneration rewrites them (``generated_at`` is ``now()`` in two of them)
this fails loudly -- the table must be addable without touching them.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ml.bank_queues import MODEL_VERSION, queue_at_recall
from ml.config import MlConfig
from ml.conftest import require_artefact

#: sha256 of the five published tables before bank_alert_queues existed.
PUBLISHED_FIVE_SHA256 = {
    "scores": "7ab48be9f7685713a48cb24dbe34b56dc5a1bfbc05dbb1908899cd5cd134ee90",
    "account_explanations": "ba2d1cca88aae5cee5594b2cf1a3cb99962876c1b904807c3504726105a94610",
    "rings": "41970ce490ddf61409d6918c76f7401c137138c7dd84878a527b0439048af078",
    "ring_explanations": "d20b911ff9eda08d13dd1dddd73dc7f85b7b762c80026c5cba6bf6c68153de92",
    "account_evidence": "36a24675e41afc342e90327d9a69da98a809eb3cd6a31b5d30db844b98a77837",
}


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def table(cfg: MlConfig) -> pd.DataFrame:
    path = cfg.ml_dir / "out" / "bank_alert_queues.parquet"
    require_artefact(path, "python -m ml.publish_all --only bank_alert_queues")
    return pd.read_parquet(path)


def test_no_cross_bank_anything(table: pd.DataFrame) -> None:
    assert not [c for c in table.columns if "cb_" in c]
    for factors in table["top_factors"]:
        assert len(factors) > 0
        for f in factors:
            assert f["feature_name"].startswith("sb_")
            assert "cb_" not in json.dumps(f, default=str)
    assert not (table["model_version"].str.contains("-cb-")).any()


def test_scores_in_unit_interval_and_version_names_scope(table: pd.DataFrame) -> None:
    assert table["score"].between(0, 1).all()
    assert set(table["model_version"]) == {MODEL_VERSION}
    assert "-sb-" in MODEL_VERSION


def test_one_row_per_account_matching_bank_counts(table: pd.DataFrame, cfg: MlConfig) -> None:
    accounts_path = cfg.table_path("accounts")
    require_artefact(accounts_path, "python -m simulator")
    accounts = pd.read_parquet(accounts_path)
    assert table["hashed_account_id"].is_unique
    assert len(table) == len(accounts)
    assert table.groupby("bank_id").size().to_dict() == accounts.groupby("bank_id").size().to_dict()
    merged = table.merge(accounts, on="hashed_account_id", suffixes=("", "_acct"))
    assert (merged["bank_id"] == merged["bank_id_acct"]).all()


def test_stored_queue_stats_match_recomputation(table: pd.DataFrame, cfg: MlConfig) -> None:
    truth_path = cfg.table_path("ground_truth")
    require_artefact(truth_path, "python -m simulator")
    truth = pd.read_parquet(truth_path).set_index("hashed_account_id")["is_mule"]
    for bank, group in table.groupby("bank_id"):
        ranked = group.sort_values("score", ascending=False, kind="stable")
        expected = queue_at_recall(truth.loc[ranked["hashed_account_id"]].to_numpy())
        stored = group.iloc[0]
        for key in ("queue_depth", "queue_true_positives", "queue_false_positives",
                    "mules_in_bank", "reached_recall"):
            assert stored[key] == expected[key], (bank, key)
        assert (group["queue_depth"] == stored["queue_depth"]).all()
        assert sorted(group["rank_in_bank"]) == list(range(1, len(group) + 1))


def test_queue_at_recall_hand_computed() -> None:
    # 5 positives -> need round(0.8 * 5) = 4; the 4th positive is at rank 6.
    labels = [True, False, True, True, False, True, False, True]
    out = queue_at_recall(labels)
    assert (out["queue_depth"], out["queue_true_positives"], out["queue_false_positives"]) == (
        6, 4, 2)
    assert queue_at_recall([False, False])["reached_recall"] is False


def test_existing_published_tables_byte_identical(cfg: MlConfig) -> None:
    out = cfg.ml_dir / "out"
    for name, expected in PUBLISHED_FIVE_SHA256.items():
        path = out / f"{name}.parquet"
        require_artefact(path, "python -m ml.publish_all")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"{name} changed"


def test_manifest_registers_table_and_keeps_other_entries(cfg: MlConfig) -> None:
    path = cfg.ml_dir / "out" / "manifest.json"
    require_artefact(path, "python -m ml.publish_all")
    tables = json.loads(path.read_text("utf-8"))["tables"]
    assert tables["bank_alert_queues"]["rows"] == 30000
    assert tables["bank_alert_queues"]["model_version"] == MODEL_VERSION
    assert {"scores", "account_explanations", "rings", "ring_explanations",
            "account_evidence"} <= set(tables)


def test_single_bank_fit_uses_only_sb_columns() -> None:
    """The fit's column set is built from the sb_ prefix, disjoint from cb_."""
    from ml.features import feature_columns

    cols = feature_columns("sb_")
    assert cols and not set(cols) & set(feature_columns("cb_"))
    assert np.all([c.startswith("sb_") for c in cols])
