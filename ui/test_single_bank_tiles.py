"""The bank tiles show the single-bank (``sb_``) queue, not the cross-bank model.

Fixtures carry the numbers in ``docs/single_bank_view.md`` so the test does not
depend on a locally generated ``ml/out/bank_alert_queues.parquet``. No GCP, no Gemini.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ui.build_snapshot import (
    ALERTS_PER_BANK,
    SINGLE_BANK_QUEUE_CAVEAT,
    _load,
    _single_bank_tiles,
)
from ui.config import UiConfig
from ui.datasource import DataSource

SB = "mulenet-xgboost-sb-oof-v1"
#: bank -> (mules, queue_depth, queue_true_positives, false positives), from the doc table.
DOC = {
    "BANK_A": (307, 397, 246, 151),
    "BANK_B": (323, 343, 258, 85),
    "BANK_C": (269, 334, 215, 119),
}
N = 40  # accounts per bank in the fixture; only the queue columns carry the doc numbers


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    rows, transfers, truth = [], [], {}
    for bank, (mules, depth, tp, fp) in DOC.items():
        for i in range(N):
            acct = f"{bank}-{i:03d}"
            truth[acct] = i % 4 == 0
            rows.append(
                {
                    "hashed_account_id": acct,
                    "bank_id": bank,
                    # Shuffled on purpose: ranking must follow rank_in_bank.
                    "score": 1.0 - i / N,
                    "model_version": SB,
                    "top_factors": [
                        {
                            "description": f"factor {k} of {acct}",
                            "contribution": 0.1,
                            "feature_name": f"sb_{k}",
                        }
                        for k in range(4)
                    ],
                    "rank_in_bank": i + 1,
                    "queue_recall_target": 0.8,
                    "queue_depth": depth,
                    "queue_true_positives": tp,
                    "queue_false_positives": fp,
                    "false_alerts_per_catch": round(fp / tp, 2),
                    "mules_in_bank": mules,
                    "reached_recall": True,
                }
            )
        transfers.append(
            {"from_hashed_account_id": f"{bank}-000", "to_hashed_account_id": "EXTERNAL"}
        )
    queues = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    return queues, pd.DataFrame(transfers), truth


def test_tiles_carry_single_bank_numbers_from_the_doc_table() -> None:
    queues, transfers, truth = _fixture()
    tiles = {t["bank_id"]: t for t in _single_bank_tiles(queues, transfers, truth)}
    assert set(tiles) == set(DOC)
    for bank, (mules, depth, tp, fp) in DOC.items():
        tile = tiles[bank]
        q = tile["queue"]
        assert (q["queue_depth"], q["queue_true_positives"], q["queue_false_positives"]) == (
            depth,
            tp,
            fp,
        )
        assert q["recall_target"] == 0.8 and q["reached_recall"] is True
        assert q["false_alerts_per_catch"] == round(fp / tp, 2)
        assert tile["mules_in_bank"] == mules
        assert tile["score_model_version"] == SB and tile["score_scope"] == "single_bank"


def test_alerts_follow_rank_in_bank_and_map_top_factors_to_strings() -> None:
    queues, transfers, truth = _fixture()
    tile = _single_bank_tiles(queues, transfers, truth)[0]
    assert [a["rank_in_bank"] for a in tile["alerts"]] == list(range(1, ALERTS_PER_BANK + 1))
    scores = [a["score"] for a in tile["alerts"]]
    assert scores == sorted(scores, reverse=True)
    top = tile["alerts"][0]
    assert top["top_factors"] == [f"factor {k} of {top['hashed_account_id']}" for k in range(4)]
    assert all(isinstance(f, str) for a in tile["alerts"] for f in a["top_factors"])


def test_caveat_no_longer_claims_a_missing_single_bank_column() -> None:
    queues, transfers, truth = _fixture()
    for tile in _single_bank_tiles(queues, transfers, truth):
        caveat = tile["queue_caveat"]
        assert "has not published" not in caveat and "not what this bank" not in caveat
        assert SB in caveat and "ceiling-bound" in caveat and "tab 3" in caveat
    assert "{model}" in SINGLE_BANK_QUEUE_CAVEAT


def _banks(reveal: bool) -> list[dict]:
    queues, transfers, truth = _fixture()
    ds = DataSource(UiConfig.from_env())
    ds._snapshot = {"banks": _single_bank_tiles(queues, transfers, truth)}
    return [t.model_dump() for t in ds.banks(None, reveal)]


def test_label_derived_fields_are_null_when_labels_are_not_revealed() -> None:
    for tile in _banks(False):
        assert tile["labels_revealed"] is False
        assert tile["mules_in_bank"] is None
        q = tile["queue"]
        for field in (
            "queue_depth",
            "queue_true_positives",
            "queue_false_positives",
            "false_alerts_per_catch",
            "reached_recall",
        ):
            assert q[field] is None, field
        assert all(a["is_mule"] is None for a in tile["alerts"])
        # Label-free fields stay.
        assert tile["score_model_version"] == SB
        assert all(
            a["score"] is not None and a["rank_in_bank"] and a["top_factors"]
            for a in tile["alerts"]
        )


def test_revealed_tiles_keep_the_doc_numbers() -> None:
    tiles = {t["bank_id"]: t for t in _banks(True)}
    assert tiles["BANK_A"]["queue"]["queue_depth"] == 397
    assert tiles["BANK_A"]["queue"]["queue_false_positives"] == 151
    assert tiles["BANK_B"]["queue"]["queue_depth"] == 343
    assert tiles["BANK_C"]["queue"]["queue_false_positives"] == 119


def test_missing_table_is_a_clear_error(tmp_path) -> None:
    cfg = UiConfig.from_env()
    if not cfg.raw_table_path("accounts").exists():
        pytest.skip("simulator/out not generated")
    from ui.build_snapshot import MlOutputMissingError

    class _Cfg:
        raw_table_path = staticmethod(cfg.raw_table_path)

        @staticmethod
        def ml_table_path(table: str):
            return tmp_path / f"{table}.parquet"

    with pytest.raises(MlOutputMissingError, match="bank_alert_queues|ml.publish"):
        _load(_Cfg())  # type: ignore[arg-type]


def test_joint_and_cross_bank_output_does_not_depend_on_the_queue_table(tmp_path) -> None:
    """Everything outside ``banks`` is derived from ``scores`` (cb_) and must not move.

    Builds the snapshot twice from the same published tables, once with the real
    ``bank_alert_queues`` and once with its scores reversed: only ``banks`` may differ.
    """
    import dataclasses
    import json
    import shutil

    from ui.build_snapshot import build

    cfg = UiConfig.from_env()
    needed = [cfg.ml_table_path("bank_alert_queues"), cfg.raw_table_path("accounts")]
    if not all(p.exists() for p in needed):
        pytest.skip("run `python -m ml.publish_all` and the simulator first")
    shutil.copytree(cfg.ml_out_dir, tmp_path / "ml", ignore=shutil.ignore_patterns("models"))
    base = build(dataclasses.replace(cfg, ml_out_dir=tmp_path / "ml"), 3)

    queues = pd.read_parquet(tmp_path / "ml" / "bank_alert_queues.parquet")
    queues["score"] = 1.0 - queues["score"]
    queues["rank_in_bank"] = queues.groupby("bank_id")["rank_in_bank"].transform(
        lambda r: r.max() + 1 - r
    )
    queues.to_parquet(tmp_path / "ml" / "bank_alert_queues.parquet")
    other = build(dataclasses.replace(cfg, ml_out_dir=tmp_path / "ml"), 3)

    assert base["banks"] != other["banks"]
    for key in base.keys() - {"banks", "generated_at"}:
        assert json.dumps(base[key], sort_keys=True) == json.dumps(other[key], sort_keys=True), key
    assert base["detector"]["score_model_version"].startswith("mulenet-xgboost-cb")
    assert all(b["score_model_version"] == SB for b in base["banks"])
