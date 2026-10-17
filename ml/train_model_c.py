"""Train and save Model C: Model B's 12 features minus account_age_days.

Standalone, offline-only. Never touches Model A/B's artifacts
(ml/out/models/model_{a,b}.*) -- read-only references to confirm the
recipe, nothing there is written.

No training script exists in the repo for Model A or Model B either
(confirmed: ``grep -rln "XGBClassifier(" ml/`` finds only ml/evaluate.py
and ml/vertex/export_model.py, and export_model.py's own output is
explicitly ``servable: false``) -- they were produced by an uncommitted
one-off fit, reverse-engineered here from their own metadata:
``trained_at_cutoff: T=5m``, ``seed: 42``, 100% of the 30,000 rows, and
-- checked directly against both metadata files before writing this --
``xgb_params`` recorded there has no ``scale_pos_weight`` key. So Model C
is fit the same way: ``XGBClassifier(**ml.evaluate.XGB_PARAMS,
random_state=42)``, no ``scale_pos_weight``, on 100% of the data. This
is NOT the grouped-CV recipe (which adds a per-fold scale_pos_weight) --
that mismatch between the servable fit and the CV that validates it
already exists for Model A and Model B; Model C matches their
precedent, not a corrected one.

Threshold, deliberately different from A/B
--------------------------------------------
Model A and Model B's ``threshold_at_1pct_fpr`` is the 99th percentile
of the FINAL model's own scores over its OWN training-sample negatives
-- a model scoring data it already memorized. Model C's threshold is
instead the 99th percentile of OUT-OF-FOLD scores from a grouped 5-fold
CV (the same procedure, same seed, that produces validated_performance),
so the threshold is never set by a model grading its own homework. This
is a deliberate difference, not an oversight -- Model A and Model B are
unchanged; see each one's own ``threshold_note`` for its method.

validated_performance is copied from ml/experiments/ablation/'s
drop_all_identity_and_age entry (same recipe, already computed under
grouped 5-fold CV x 3 seeds), not re-derived here.

Usage::

    python -m ml.train_model_c
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from ml.config import MlConfig
from ml.early import cutoff_series, ring_onsets
from ml.evaluate import SEEDS, XGB_PARAMS, attach_labels, fold_assignment, oof_scores
from ml.features import build_features, load_tables

MODEL_A_METADATA = Path("ml/out/models/model_a_metadata.json")
MODEL_B_METADATA = Path("ml/out/models/model_b_metadata.json")
ABLATION_RESULTS = Path("ml/experiments/ablation/results_ablation.json")
OUT_DIR = Path("ml/out/models")

TRAIN_SEED = 42
TARGET_FPR = 0.01

#: Model B's 12 (ml/out/models/model_b_metadata.json's own feature_columns),
#: minus account_age_days, same relative order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "cb_in_degree",
    "cb_out_degree",
    "cb_unique_in_counterparties",
    "cb_unique_out_counterparties",
    "cb_unique_counterparties",
    "cb_distinct_counterparty_banks",
    "cb_amount_in_total",
    "cb_amount_out_total",
    "cb_passthrough_ratio",
    "cb_passthrough_speed_seconds",
    "cb_round_amount_ratio",
)


def _check_ab_recipe_assumption() -> None:
    """Fail loudly, not silently, if A/B's recorded recipe ever changes."""
    for path in (MODEL_A_METADATA, MODEL_B_METADATA):
        meta = json.loads(path.read_text())
        if "scale_pos_weight" in meta["xgb_params"]:
            raise SystemExit(
                f"{path} now records scale_pos_weight -- Model C's assumption "
                "that A/B were fit without it no longer holds. Stopping rather "
                "than silently fitting Model C a different way than this "
                "docstring claims."
            )
        if meta["trained_at_cutoff"] != "T=5m (per-account reference-time cutoff, ml.early.cutoff_series)" or meta["seed"] != 42:
            raise SystemExit(
                f"{path}: trained_at_cutoff/seed differ from what Model C "
                "assumes (T=5m, seed 42). Stopping."
            )


def main() -> None:
    _check_ab_recipe_assumption()

    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])
    cutoff = cutoff_series(tables, onsets, 5, TRAIN_SEED)
    frame = attach_labels(build_features(cfg, tables=tables, cutoff=cutoff), cfg)

    x = frame[list(FEATURE_COLUMNS)].to_numpy(dtype="float32")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")

    # --- threshold: out-of-fold grouped 5-fold CV scores, not in-sample ---
    fold = fold_assignment(frame, TRAIN_SEED)
    oof = oof_scores(frame, list(FEATURE_COLUMNS), TRAIN_SEED, fold)
    threshold = float(np.quantile(oof[y == 0], 1.0 - TARGET_FPR))

    pooled_flag = oof >= threshold
    realized_fpr_pooled = float((pooled_flag & (y == 0)).sum() / (y == 0).sum())
    realized_fpr_by_fold = {}
    for k in range(int(fold.max()) + 1):
        in_fold = fold == k
        neg_in_fold = in_fold & (y == 0)
        realized_fpr_by_fold[f"fold{k}"] = float(
            (pooled_flag & neg_in_fold).sum() / neg_in_fold.sum()
        )

    # --- servable artifact: fit once on 100% of data, A/B's own recipe ---
    model = XGBClassifier(**XGB_PARAMS, random_state=TRAIN_SEED)
    model.fit(x, y, verbose=False)
    booster = model.get_booster()
    booster.feature_names = list(FEATURE_COLUMNS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_DIR / "model_c.ubj"))

    ablation = json.loads(ABLATION_RESULTS.read_text())
    validated_performance = {
        horizon: {
            k: v
            for k, v in ablation["summary"][horizon]["drop_all_identity_and_age"].items()
            if k in ("pr_auc_mean", "recall_at_fpr_1pct_mean", "false_alerts_per_catch_at_80pct_recall_mean")
        }
        for horizon in ("T=5m", "T=180m")
    }

    metadata = {
        "model_name": "model_c",
        "generated_at": datetime.now(UTC).isoformat(),
        "servable": True,
        "served_by": "agents/subset_upload.py (replacing model_b for this path)",
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "xgb_params": {k: str(v) for k, v in XGB_PARAMS.items()},
        "seed": TRAIN_SEED,
        "trained_on": (
            "100% of labeled data (no held-out fold), same recipe as Model A/B's "
            "own servable fit (ml.evaluate.XGB_PARAMS, seed 42, no "
            "scale_pos_weight -- confirmed from their own metadata, not assumed) "
            "-- see validated_performance for the honest expected-performance "
            "estimate, from ml/experiments/ablation/'s drop_all_identity_and_age "
            "(grouped 5-fold CV x 3 seeds, same recipe)."
        ),
        "trained_at_cutoff": "T=5m (per-account reference-time cutoff, ml.early.cutoff_series)",
        "training_rows": int(len(frame)),
        "training_positives": int(y.sum()),
        "threshold_at_1pct_fpr": threshold,
        "threshold_method": (
            "99th percentile of OUT-OF-FOLD scores from a grouped 5-fold CV "
            "(same seed, same recipe as validated_performance) -- deliberately "
            "NOT the in-sample method Model A and Model B use (99th percentile "
            "of the final 100%-fit model's own scores over its own training "
            "negatives). Model A and Model B are unchanged; this is a one-off "
            "difference for Model C, not a retroactive fix applied to them."
        ),
        "threshold_realized_fpr_pooled": realized_fpr_pooled,
        "threshold_realized_fpr_by_fold": realized_fpr_by_fold,
        "validated_performance": validated_performance,
    }
    (OUT_DIR / "model_c_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"threshold_at_1pct_fpr={threshold:.6f}")
    print(f"realized_fpr_pooled={realized_fpr_pooled:.4f}")
    print(f"realized_fpr_by_fold={json.dumps(realized_fpr_by_fold, indent=2)}")
    print(f"wrote {OUT_DIR / 'model_c.ubj'} and {OUT_DIR / 'model_c_metadata.json'}")


if __name__ == "__main__":
    main()
