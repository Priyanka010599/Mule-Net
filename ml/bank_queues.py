"""Single-bank alert queues: what each bank would see running ``sb_`` features alone.

``ml/out/scores.parquet`` is the **cross-bank** model (``cb_`` features),
so a bank tile ranked by it shows cross-bank output. This module publishes
the like-for-like single-bank scope as a new, purely additive table,
``bank_alert_queues``: one row per account, scored by an out-of-fold
XGBoost fitted on the ``sb_`` columns only.

Recipe is the published one, unchanged, so the two scopes are comparable:
``ml.evaluate.XGB_PARAMS``, seed ``SEEDS[0]`` (42), ``fold_assignment``
grouped folds (whole rings per fold), label ``is_passthrough_mule``. The
OOF predictions did not already exist at full window (``ml.early`` writes
single-bank OOF only at its cutoff horizons), so they are produced here.

No cross-bank feature, score or factor reaches the table; the module
asserts it before writing. Run via ``python -m ml.publish_all`` (or
``python -m ml.publish_all --only bank_alert_queues`` to add just this
table without rewriting the others).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import MlConfig
from ml.evaluate import (
    SEEDS,
    XGB_PARAMS,
    attach_labels,
    fold_assignment,
    oof_scores_and_contributions,
)
from ml.features import feature_columns, load_tables
from ml.metrics import BURDEN_RECALL
from ml.publish import EXPLANATION_FACTORS, factor_description, register_table

TABLE_NAME = "bank_alert_queues"

#: Names the scope, so a consumer cannot mistake it for ``scores.parquet``.
MODEL_VERSION = "mulenet-xgboost-sb-oof-v1"

#: Prefix of the only features this table may use or mention.
SCOPE_PREFIX = "sb_"

#: Label the queue stats are scored against: the simulator's ``is_mule``
#: (mule + collector + source), which is what the UI bank tiles use, so the
#: numbers sit next to the tile numbers. The model itself is trained on
#: ``is_passthrough_mule`` (as is the published cross-bank model).
STATS_LABEL = "ground_truth.is_mule (role in mule/collector/source)"


def queue_at_recall(
    ranked_labels: list[bool] | np.ndarray, recall: float = BURDEN_RECALL
) -> dict[str, object]:
    """Depth of a score-ranked queue at the first point reaching ``recall``.

    Same rule as ``ui.build_snapshot._queue_at_recall`` (needed =
    ``max(1, round(recall * positives))``), re-stated here because ``ml/``
    must not import ``ui/``; the tests pin the two to the same answer.
    """
    labels = [bool(v) for v in ranked_labels]
    positives = sum(labels)
    if positives == 0:
        return {
            "queue_depth": 0,
            "queue_true_positives": 0,
            "queue_false_positives": 0,
            "false_alerts_per_catch": None,
            "mules_in_bank": 0,
            "reached_recall": False,
        }
    needed = max(1, int(round(recall * positives)))
    caught, depth = 0, len(labels)
    for idx, is_mule in enumerate(labels, start=1):
        caught += is_mule
        if caught >= needed:
            depth = idx
            break
    false_positives = depth - caught
    return {
        "queue_depth": depth,
        "queue_true_positives": caught,
        "queue_false_positives": false_positives,
        "false_alerts_per_catch": round(false_positives / caught, 2) if caught else None,
        "mules_in_bank": positives,
        "reached_recall": caught >= needed,
    }


def build_bank_alert_queues(
    frame: pd.DataFrame,
    bank_of: pd.Series,
    is_mule: pd.Series,
    scores: np.ndarray,
    contribs: np.ndarray,
    columns: list[str],
    scored_at: pd.Series,
) -> pd.DataFrame:
    """One row per account; ranks and queue stats are per bank."""
    assert all(c.startswith(SCOPE_PREFIX) for c in columns), "single-bank columns only"
    names = [c[len(SCOPE_PREFIX) :] for c in columns]
    values = frame[columns].to_numpy(dtype="float64")
    order = np.argsort(-np.abs(contribs[:, : len(columns)]), axis=1)

    rows = []
    for i, account in enumerate(frame["hashed_account_id"]):
        factors = [
            {
                "description": factor_description(names[j], values[i, j]),
                "contribution": float(contribs[i, j]),
                "feature_name": columns[j],
            }
            for j in order[i, :EXPLANATION_FACTORS]
        ]
        rows.append(
            {
                "hashed_account_id": account,
                "bank_id": str(bank_of[account]),
                "score": float(scores[i]),
                "model_version": MODEL_VERSION,
                "scored_at": scored_at.iloc[i],
                "top_factors": factors,
            }
        )
    table = pd.DataFrame(rows)
    table["scored_at"] = pd.to_datetime(table["scored_at"], utc=True)

    parts = []
    for _bank, group in table.groupby("bank_id", sort=True):
        # Stable sort on descending score: ties keep input order, exactly as
        # the UI tile ranks, so the depths are computed the same way.
        ranked = group.sort_values("score", ascending=False, kind="stable").copy()
        ranked["rank_in_bank"] = np.arange(1, len(ranked) + 1)
        stats = queue_at_recall(is_mule.loc[ranked["hashed_account_id"]].to_numpy())
        for key, value in stats.items():
            ranked[key] = value
        ranked["queue_recall_target"] = BURDEN_RECALL
        parts.append(ranked)
    out = pd.concat(parts, ignore_index=True)
    out["reached_recall"] = out["reached_recall"].astype(bool)
    out["false_alerts_per_catch"] = out["false_alerts_per_catch"].astype("float64")
    return out


def assert_single_bank_only(table: pd.DataFrame) -> None:
    """No ``cb_`` anywhere: column names, factor names, descriptions."""
    assert not any("cb_" in c for c in table.columns), "cb_ in column names"
    for factors in table["top_factors"]:
        for f in factors:
            assert f["feature_name"].startswith(SCOPE_PREFIX), f["feature_name"]
            assert "cb_" not in f["description"], f["description"]
    assert ((table["score"] >= 0) & (table["score"] <= 1)).all(), "score outside [0, 1]"


def provenance(seed: int) -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "feature_scope": f"{SCOPE_PREFIX} (single-bank) ONLY; no cb_ feature, score or factor",
        "recipe": (
            "ml.evaluate.XGB_PARAMS + oof_scores_and_contributions, grouped folds from "
            "ml.evaluate.fold_assignment (whole rings per fold), label is_passthrough_mule; "
            "identical to the cross-bank scores.parquet fit except for the column set"
        ),
        "seed": seed,
        "xgb_params": XGB_PARAMS,
        "fitted_by": "ml.bank_queues (additional grouped-CV fit on the sb_ columns; "
        "single-bank OOF at full window did not already exist)",
        "top_factors": (
            f"top {EXPLANATION_FACTORS} TreeSHAP contributions of the single-bank fold model, "
            "same struct shape as account_explanations.top_factors, sb_ names only"
        ),
        "queue_stats": (
            f"per bank, queue depth to {BURDEN_RECALL:.0%} recall over that bank's accounts "
            f"ranked by score, scored against {STATS_LABEL}; same rule as the UI bank tile"
        ),
        "written_by": "ml.bank_queues, appended to this manifest without touching other tables. "
        "Use `python -m ml.publish_all` (or `--only bank_alert_queues`).",
        "additivity": "no existing published table or manifest entry is modified by this step",
    }


def main() -> None:
    """Fit the single-bank OOF, build the table, register it."""
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    truth = pd.read_parquet(cfg.table_path("ground_truth")).set_index("hashed_account_id")
    bank_of = tables["accounts"].set_index("hashed_account_id")["bank_id"]

    columns = feature_columns(SCOPE_PREFIX)
    seed = SEEDS[0]
    scores, contribs = oof_scores_and_contributions(
        frame, columns, seed, fold_assignment(frame, seed)
    )
    table = build_bank_alert_queues(
        frame,
        bank_of,
        truth["is_mule"],
        scores,
        contribs,
        columns,
        pd.to_datetime(frame["as_of"], utc=True),
    )
    assert_single_bank_only(table)
    path = register_table(
        cfg.ml_dir / "out", TABLE_NAME, table, validated_against=None, **provenance(seed)
    )
    print(f"wrote {path} rows={len(table)}")
    for bank, g in table.groupby("bank_id", sort=True):
        r = g.iloc[0]
        print(
            f"{bank}: accounts={len(g)} queue_depth@80%={r.queue_depth} "
            f"tp={r.queue_true_positives} fp={r.queue_false_positives}"
        )


if __name__ == "__main__":
    main()
