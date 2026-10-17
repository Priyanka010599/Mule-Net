"""XGBoost baseline: single-bank features vs cross-bank features.

The experiment this module exists to run: train the *same* model, with
the *same* grouped-CV folds and the *same* hyperparameters, twice — once
seeing only what one bank can see (``sb_`` columns), once seeing the
merged three-bank graph (``cb_`` columns). Nothing is tuned per scope,
so a difference in the numbers is a difference in the *information*, not
in the fitting.

Evaluation is grouped 5-fold cross-validation over rings, pooled
out-of-fold, over three seeds; see ``ml/evaluate.py``. Metrics are the
operational set in ``ml/metrics.py``.

Label: ``is_passthrough_mule`` — 1 when ``role`` is ``mule`` or
``collector`` (712 accounts). ``source`` accounts (187) are **not**
victims: they are complicit feeders allocated from the same mule pool,
sharing the ring's devices and reused KYC documents. They are left out
of this label because they are structurally undetectable by a
pass-through model — median in-degree 0, so no pass-through feature can
separate them — not because they are innocent. ``is_complicit`` (899)
is the honest guilt label and is carried alongside; see
``ml.evaluate.COMPLICIT_ROLES`` and ``ml/stage2.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from ml.config import MlConfig
from ml.evaluate import (
    SEEDS,
    SPAN_STRATA,
    attach_labels,
    run_scope,
    strip_scores,
    summarise,
    summarise_strata,
)
from ml.metrics import METRIC_KEYS

#: Experiments run back to back. The first is the headline result and
#: uses every feature. The second is a *robustness check*: the first
#: generation of this dataset leaked the answer through transfer amounts
#: and the data-engineer has since fixed it (ring transfers are now
#: 0.39% exact multiples of 100 against 0.10% elsewhere, and their median
#: size is 1.99x rather than 64x the background). Dropping the
#: amount-derived features asks whether graph structure, timing and
#: identity reuse alone favour the cross-bank view, independently of
#: whether any amount signal survived. Both are reported.
EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "xgboost_cv": (),
    "xgboost_cv_no_amount": (
        "round_amount_ratio",
        "amount_in_total",
        "amount_out_total",
    ),
}


def write_metrics(cfg: MlConfig, name: str, metrics: dict[str, object]) -> None:
    """Write one experiment's metrics where ``ml/compare.py`` will find it."""
    out_dir = cfg.experiments_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")


def print_strata(metrics: dict[str, object]) -> None:
    """Print the per-span table, with the PR-AUC comparability caveat."""
    strata: dict[str, dict[str, dict[str, float]]] = metrics["strata_summary"]  # type: ignore[assignment]
    if not strata:
        return
    print("")
    print("  per ring bank-span (train pooled, evaluate stratified; negatives held constant)")
    print("  CAVEAT: PR-AUC is NOT comparable across spans — positive counts differ.")
    print("  Compare spans on ROC-AUC and recall@fixed-FPR; both are prevalence-independent.")
    head = f"  {'span':<14}{'pos':>6}{'prev':>9}" + "".join(f"{k:>26}" for k in METRIC_KEYS)
    print(head)
    for stratum in (f"span{s}" for s in SPAN_STRATA):
        for scope in ("single_bank", "cross_bank"):
            entry = strata.get(scope, {}).get(stratum)
            if entry is None:
                continue
            cells = "".join(
                f"{entry[k + '_mean']:.4f}+/-{entry[k + '_std']:.4f}".rjust(26)
                for k in METRIC_KEYS
            )
            tag = f"{stratum}/{scope[:5]}"
            print(f"  {tag:<14}{int(entry['positives']):>6}{entry['prevalence']:>9.5f}{cells}")
        sb = strata.get("single_bank", {}).get(stratum)
        cb = strata.get("cross_bank", {}).get(stratum)
        if sb and cb:
            gaps = "  ".join(
                f"{k}={cb[k + '_mean'] - sb[k + '_mean']:+.4f}"
                for k in ("pr_auc", "roc_auc", "recall_at_fpr_1pct")
            )
            print(f"  {stratum} cross-minus-single: {gaps}")


def print_summary(metrics: dict[str, object]) -> None:
    """Print the single-bank vs cross-bank table for one experiment."""
    summary: dict[str, dict[str, float]] = metrics["summary"]  # type: ignore[assignment]
    print(f"\n{metrics['model']} — grouped 5-fold CV over rings, {len(SEEDS)} seeds")
    header = f"{'scope':<14}" + "".join(f"{k:>26}" for k in METRIC_KEYS)
    print(header)
    for scope in ("single_bank", "cross_bank"):
        if scope not in summary:
            continue
        cells = "".join(
            f"{summary[scope][k + '_mean']:.4f}+/-{summary[scope][k + '_std']:.4f}".rjust(26)
            for k in METRIC_KEYS
        )
        print(f"{scope:<14}{cells}")
    if {"single_bank", "cross_bank"} <= set(summary):
        delta = summary["cross_bank"]["pr_auc_mean"] - summary["single_bank"]["pr_auc_mean"]
        pooled = max(
            summary["cross_bank"]["pr_auc_std"], summary["single_bank"]["pr_auc_std"], 1e-9
        )
        print(f"cross-bank minus single-bank PR-AUC: {delta:+.4f}  ({delta / pooled:+.2f} sigma)")


def main() -> None:
    """Train both scopes over three seeds and write the metrics JSON."""
    cfg = MlConfig.from_env()
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    print(
        f"accounts={len(frame)} positives={int(frame['is_passthrough_mule'].sum())} "
        f"rings={frame.loc[frame['ring_id'] != '', 'ring_id'].nunique()}"
    )
    print(frame["cohort"].value_counts().to_string())

    for name, drop in EXPERIMENTS.items():
        runs = [
            run_scope(frame, prefix, seed, drop) for prefix in ("sb_", "cb_") for seed in SEEDS
        ]
        metrics = {
            "model": name,
            "generated_at": datetime.now(UTC).isoformat(),
            "label_definition": "is_passthrough_mule: role in (mule, collector)",
            "evaluation": "grouped 5-fold CV over ring_id, out-of-fold pooled",
            "dropped_features": list(drop),
            "total_positives": int(frame["is_passthrough_mule"].sum()),
            "total_accounts": len(frame),
            "positives_by_span": {
                f"span{s}": int(
                    ((frame["is_passthrough_mule"] == 1) & (frame["ring_bank_count"] == s)).sum()
                )
                for s in SPAN_STRATA
            },
            "stratification": (
                "train pooled on all data, evaluate restricted per ring_bank_count; "
                "negatives (is_passthrough_mule==0) held constant across strata; "
                "PR-AUC not comparable across strata (differing positive counts) — "
                "use roc_auc and recall_at_fpr for cross-span comparison"
            ),
            "runs": strip_scores(runs),
            "summary": summarise(runs),
            "strata_summary": summarise_strata(runs),
        }
        print_summary(metrics)
        print_strata(metrics)
        write_metrics(cfg, name, metrics)


if __name__ == "__main__":
    main()
