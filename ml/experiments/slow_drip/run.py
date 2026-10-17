"""Slow-drip experiment: does anything catch rings that stay under the velocity ceiling?

Detectors (each in a single-bank ``sb_`` and a cross-bank ``cb_`` version)
--------------------------------------------------------------------------
``velocity_rule``
    ``ml.rule.velocity_decision`` unchanged. No fitting, no threshold choice
    here: the thresholds are the shipped constants. ``sb_`` drops the
    bank-spread clause exactly as ``ml/rule.py`` does.
``isolation_forest``
    ``sklearn.ensemble.IsolationForest`` fitted on the existing
    ``feature_columns(prefix)`` of the evaluated world's accounts. **No labels
    are read to fit, score or threshold it**: ``fit_isolation_forest`` receives
    a feature matrix only. Fixed hyper-parameters, one seed (42), never tuned.
``xgboost_oof``
    ``ml.evaluate`` recipe: ``XGB_PARAMS``, ``scale_pos_weight`` per fold,
    ``fold_assignment(frame, 42)`` grouped-ring folds, out-of-fold scores.
``xgboost_canonical_transfer`` (overlay world only)
    The same XGBoost fitted on **all** canonical accounts and scored on the
    overlay: "what a model trained on today's data does against slow drip".

Alert thresholds (all chosen without labels)
--------------------------------------------
* ``budget_1pct_per_bank``: each bank alerts its top 1% of accounts by score
  (``ceil(0.01 * n_bank)``, ties broken by position). Same alerts per bank for
  every score-based detector, in both scopes. Labels play no part in it.
* ``budget_matched_to_rule``: each bank alerts exactly as many accounts as the
  velocity rule fires on at that bank in the same world (skipped where the
  rule fires nowhere). Only the rule's own output is used.
* The rule itself is binary and alerts wherever its fixed conditions hold; its
  alert count is reported rather than forced to a budget.

Ring-level catch
----------------
A ring is **caught** when at least one of its pass-through members (role
``mule`` or ``collector``, the roles the detector targets) is alerted.
Also reported: ``majority`` catch (>= half of those members alerted).
**False alert** = an alerted account that is not complicit (role not in
mule/collector/source). Alerts on ``source`` accounts are counted separately and
are neither true catches nor false alerts. ``false_alerts_per_ring_caught`` =
false alerts / rings caught (``null`` when no ring is caught).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from ml.config import MlConfig
from ml.evaluate import XGB_PARAMS, attach_labels, fold_assignment, oof_scores
from ml.experiments.slow_drip.overlay import (
    DRIP_DELAY_RANGE,
    DRIP_SPACING_RANGE,
    OVERLAY_DIR,
    OVERLAY_SEED,
    OVERLAY_SPAN_MIX,
    SMALL_AMOUNT_CAP,
    generate_overlay,
)
from ml.features import build_features, feature_columns, load_tables
from ml.rule import VELOCITY_MAX_SECONDS, velocity_decision

SEED = 42
BUDGET_FRACTION = 0.01
SCOPES: dict[str, str] = {"single_bank": "sb_", "cross_bank": "cb_"}
OUT_PATH = Path(__file__).resolve().parent / "metrics.json"
IF_PARAMS: dict[str, Any] = {"n_estimators": 300, "max_samples": 256, "contamination": "auto"}


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------


def fit_isolation_forest(
    features: pd.DataFrame, columns: list[str], seed: int = SEED
) -> np.ndarray:
    """Unsupervised anomaly score (higher = more anomalous) for every row.

    Takes feature columns only; there is no label argument to misuse.
    """
    if any(c.startswith(("is_", "ring_", "role", "pattern", "cohort")) for c in columns):
        raise ValueError("isolation forest must only see feature columns")
    x = features[columns].to_numpy(dtype="float32")
    forest = IsolationForest(random_state=seed, n_jobs=1, **IF_PARAMS).fit(x)
    return -forest.score_samples(x)


def rule_alerts(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """The shipped velocity rule; single-bank drops the (uncomputable) bank clause."""
    return velocity_decision(frame, prefix, require_bank_spread=(prefix == "cb_"))


def xgb_fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int = SEED
) -> np.ndarray:
    """XGBoost on all of ``train`` (labelled), scored on ``test``."""
    y = train["is_passthrough_mule"].to_numpy(dtype="int64")
    n_pos = int(y.sum())
    model = XGBClassifier(
        **XGB_PARAMS,
        random_state=seed * 100,
        scale_pos_weight=(len(y) - n_pos) / max(n_pos, 1),
    )
    model.fit(train[columns].to_numpy(dtype="float32"), y, verbose=False)
    return model.predict_proba(test[columns].to_numpy(dtype="float32"))[:, 1]


# --------------------------------------------------------------------------
# budgets and ring-level scoring
# --------------------------------------------------------------------------


def budget_per_bank(bank: np.ndarray, fraction: float = BUDGET_FRACTION) -> dict[str, int]:
    """Alerts per bank for a fixed fraction of that bank's accounts (no labels)."""
    return {str(b): math.ceil(fraction * int((bank == b).sum())) for b in sorted(set(bank))}


def top_k_per_bank(scores: np.ndarray, bank: np.ndarray, budget: dict[str, int]) -> np.ndarray:
    """Boolean alert vector: each bank's top-``budget[bank]`` accounts by score."""
    alert = np.zeros(len(scores), dtype=bool)
    for b, k in budget.items():
        idx = np.flatnonzero(bank == b)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        alert[order[:k]] = True
    return alert


def ring_report(frame: pd.DataFrame, alert: np.ndarray) -> dict[str, Any]:
    """Account- and ring-level outcome of one alert vector (labels used to score only)."""
    passthrough = frame["is_passthrough_mule"].to_numpy().astype(bool)
    complicit = frame["is_complicit"].to_numpy().astype(bool)
    ring = frame["ring_id"].to_numpy()
    pattern = frame["pattern_type"].to_numpy()
    span = frame["ring_bank_count"].to_numpy()

    hits = pd.DataFrame({"ring": ring, "hit": alert}).loc[passthrough]
    per_ring = hits.groupby("ring")["hit"].agg(["sum", "size"])
    caught = per_ring["sum"] >= 1
    majority = per_ring["sum"] * 2 >= per_ring["size"]
    ring_meta = (
        frame.loc[passthrough, ["ring_id", "pattern_type", "ring_bank_count"]]
        .drop_duplicates("ring_id")
        .set_index("ring_id")
        .reindex(per_ring.index)
    )
    n_rings = int(len(per_ring))
    n_caught = int(caught.sum())
    false_alerts = int((alert & ~complicit).sum())
    del pattern, span
    role = frame["role"].to_numpy()

    def _by(col: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, idx in ring_meta.groupby(col).groups.items():
            out[str(key)] = {
                "rings": len(idx),
                "caught": int(caught.loc[idx].sum()),
                "recall": float(caught.loc[idx].mean()),
            }
        return out

    return {
        "alerts": int(alert.sum()),
        "rings": n_rings,
        "rings_caught": n_caught,
        "ring_recall": n_caught / n_rings if n_rings else float("nan"),
        "rings_caught_majority": int(majority.sum()),
        "ring_recall_majority": float(majority.mean()) if n_rings else float("nan"),
        "passthrough_accounts_caught": int((alert & passthrough).sum()),
        "passthrough_account_recall": float((alert & passthrough).sum() / passthrough.sum()),
        "mule_account_recall": float((alert & (role == "mule")).sum() / (role == "mule").sum()),
        "collector_account_recall": float(
            (alert & (role == "collector")).sum() / (role == "collector").sum()
        ),
        "source_alerts": int((alert & complicit & ~passthrough).sum()),
        "false_alerts": false_alerts,
        "false_alerts_per_ring_caught": false_alerts / n_caught if n_caught else None,
        "by_pattern": _by("pattern_type"),
        "by_span": _by("ring_bank_count"),
    }


def ranking_report(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    """Threshold-free ranking quality (labels used to score only)."""
    y = frame["is_passthrough_mule"].to_numpy().astype(int)
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "prevalence": float(y.mean()),
    }


# --------------------------------------------------------------------------
# worlds
# --------------------------------------------------------------------------


def _with_bank(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    out = frame.copy()
    out["bank_id"] = out["hashed_account_id"].map(bank)
    return out


def load_canonical(cfg: MlConfig) -> pd.DataFrame:
    """Canonical features + labels, read-only."""
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    return _with_bank(frame, load_tables(cfg))


def load_overlay(cfg: MlConfig, overlay_dir: Path = OVERLAY_DIR) -> pd.DataFrame:
    """Overlay features + labels; generates the overlay if it is not on disk."""
    if not (overlay_dir / "ground_truth.parquet").exists():
        generate_overlay(overlay_dir)
    ocfg = replace(cfg, data_dir=overlay_dir)
    tables = load_tables(ocfg)
    frame = attach_labels(build_features(ocfg, tables=tables), ocfg)
    return _with_bank(frame, tables)


def evaluate_world(
    frame: pd.DataFrame,
    transfer_from: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """All detectors, both scopes, both budgets, on one world."""
    bank = frame["bank_id"].to_numpy()
    fold = fold_assignment(frame, SEED)
    budget_fixed = budget_per_bank(bank)
    result: dict[str, Any] = {
        "accounts": len(frame),
        "passthrough_accounts": int(frame["is_passthrough_mule"].sum()),
        "budget_1pct_per_bank": budget_fixed,
        "detectors": {},
    }
    for scope, prefix in SCOPES.items():
        columns = feature_columns(prefix)
        rule = rule_alerts(frame, prefix)
        rule_by_bank = {b: int(rule[bank == b].sum()) for b in budget_fixed}
        matched = rule_by_bank if sum(rule_by_bank.values()) > 0 else None

        score_sets: dict[str, np.ndarray] = {
            "isolation_forest": fit_isolation_forest(frame, columns),
            "xgboost_oof": oof_scores(frame, columns, SEED, fold),
        }
        if transfer_from is not None:
            score_sets["xgboost_canonical_transfer"] = xgb_fit_predict(
                transfer_from, frame, columns
            )

        entry: dict[str, Any] = {
            "velocity_rule": {
                "threshold": "fixed shipped constants (ml/rule.py); not tuned here",
                "alerts_per_bank": rule_by_bank,
                **ring_report(frame, rule),
            }
        }
        for name, scores in score_sets.items():
            entry[name] = {
                "ranking": ranking_report(frame, scores),
                "budget_1pct_per_bank": ring_report(
                    frame, top_k_per_bank(scores, bank, budget_fixed)
                ),
                "budget_matched_to_rule": (
                    {
                        "alerts_per_bank": matched,
                        **ring_report(frame, top_k_per_bank(scores, bank, matched)),
                    }
                    if matched
                    else None
                ),
            }
        result["detectors"][scope] = entry
    return result


def run(cfg: MlConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MlConfig.from_env()
    canonical = load_canonical(cfg)
    overlay = load_overlay(cfg)
    return {
        "experiment": "slow_drip",
        "design": {
            "seed": SEED,
            "overlay_seed": OVERLAY_SEED,
            "overlay_span_mix_1_2_3_banks": list(OVERLAY_SPAN_MIX),
            "velocity_max_seconds_from_rule": VELOCITY_MAX_SECONDS,
            "drip_delay_range_seconds": list(DRIP_DELAY_RANGE),
            "drip_chunk_spacing_range_seconds": list(DRIP_SPACING_RANGE),
            "amount_cap": SMALL_AMOUNT_CAP,
            "budget_fraction_per_bank": BUDGET_FRACTION,
            "isolation_forest_params": {**IF_PARAMS, "random_state": SEED},
            "ring_caught_definition": (
                "ring caught = at least one pass-through member (mule/collector) alerted"
            ),
            "false_alert_definition": "alerted account whose role is not mule/collector/source",
            "thresholds": (
                "score-based detectors: top 1% of each bank's accounts (ceil) chosen "
                "without labels; second budget = per-bank alert count of the velocity "
                "rule in the same world; rule uses its fixed shipped constants"
            ),
            "folds": "ml.evaluate.fold_assignment(frame, 42), 5 grouped-ring folds, OOF",
        },
        "worlds": {
            "canonical": evaluate_world(canonical),
            "slow_drip_overlay": evaluate_world(overlay, transfer_from=canonical),
        },
    }


def main() -> None:
    metrics = run()
    OUT_PATH.write_text(json.dumps(metrics, indent=2, default=str) + "\n", "utf-8")
    print(f"wrote {OUT_PATH}")
    for world, w in metrics["worlds"].items():
        print(f"\n== {world}")
        for scope, dets in w["detectors"].items():
            for det, r in dets.items():
                b = r if det == "velocity_rule" else r["budget_1pct_per_bank"]
                print(
                    f"{scope:12} {det:28} alerts={b['alerts']:4d} "
                    f"rings_caught={b['rings_caught']:3d}/{b['rings']} "
                    f"false/ring={b['false_alerts_per_ring_caught']}"
                )


if __name__ == "__main__":
    main()
