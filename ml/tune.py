"""Honest threshold tuning for the velocity rule: tune in-fold, score out-of-fold.

The problem this module fixes
-----------------------------
``rule.py`` shipped four velocity thresholds — hop window, flow band,
fan-in floor, fan-out floor — chosen by a grid search over **the whole
dataset**, and then reported precision on **the whole dataset**. Those
are the same rows twice. The resulting 0.9337 precision is a
*resubstitution* estimate: it tells you how well four numbers fit 30,000
accounts they were fitted to, which is not the question anyone is
asking. The question is what the rule does on rings it has never seen.

What is done instead
--------------------
The grid search moves **inside** the grouped 5-fold CV that
``evaluate.py`` already defines:

1. deal whole rings into 5 folds (``evaluate.fold_assignment``), so no
   ring is split across the train/test boundary;
2. for fold ``k``, run the entire grid over the **training** rows only
   and pick the best thresholds there;
3. apply exactly those thresholds to fold ``k``'s **held-out** rows and
   record the flags;
4. pool the held-out flags over all five folds — every account in the
   dataset is scored by a rule that never saw its ring — and compute
   precision/recall/FPR once on the pooled decisions.

Three seeds reshuffle the fold assignment; results are mean +/- std
across them. The in-sample number is computed too (tune on everything,
score everything) purely so the **gap** between the two can be printed
rather than glossed over.

Selection objective
-------------------
Within a training fold: **maximise recall subject to FPR <=
``FPR_BUDGET``**, tie-broken by precision. The budget, not an unbounded
F-score, is the operationally faithful constraint — an AML team staffs a
fixed number of alerts per day, and a rule that doubles recall by
tripling the alert queue is not an improvement to them. Ties are broken
deterministically by grid order, so the selection is reproducible.

The single-bank analogue, identically treated
---------------------------------------------
The same grid, the same folds, the same objective, run over ``sb_``
features with the bank-spread clause dropped (the most generous
single-bank reading, since that feature is capped at 1 in the ``sb_``
scope and the literal clause fires on nothing). Both scopes are tuned
per fold, so neither is handed an in-sample advantage the other lacks.
If the cross-bank margin survives honest tuning on both sides, it is a
margin and not an artefact of who got to peek.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import astuple, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml.config import MlConfig
from ml.evaluate import N_FOLDS, SEEDS, attach_labels, fold_assignment
from ml.metrics import binary_metrics

#: Alert budget the tuner must respect on the training fold: at most
#: 0.2% of legitimate accounts may be flagged. ~29,300 negatives here,
#: so roughly 59 alerts.
FPR_BUDGET = 0.002

#: Candidate values per threshold. Bracketed around the shipped rule
#: (2400s / 0.25-1.5 / 16 / 8) so that choice remains reachable, and wide
#: enough on both sides that a fold can genuinely disagree with it.
VELOCITY_GRID: dict[str, tuple[float, ...]] = {
    "speed_max_seconds": (1200.0, 1800.0, 2400.0, 3000.0, 3600.0),
    "ratio_min": (0.1, 0.25, 0.5),
    "ratio_max": (1.0, 1.5, 2.0),
    "min_in_degree": (8.0, 12.0, 16.0, 20.0, 24.0),
    "min_out_degree": (4.0, 6.0, 8.0, 12.0),
}

#: Threshold field names in grid order.
FIELDS: tuple[str, ...] = tuple(VELOCITY_GRID)

#: Hard-decision metrics summarised across folds and seeds.
RULE_METRICS: tuple[str, ...] = ("precision", "recall", "fpr")


@dataclass(frozen=True)
class Thresholds:
    """One point in the velocity grid."""

    speed_max_seconds: float
    ratio_min: float
    ratio_max: float
    min_in_degree: float
    min_out_degree: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FIELDS, astuple(self), strict=True))


def grid() -> list[Thresholds]:
    """Every threshold combination, with ``ratio_min < ratio_max`` enforced."""
    points = [
        Thresholds(*values)
        for values in itertools.product(*(VELOCITY_GRID[f] for f in FIELDS))
    ]
    return [p for p in points if p.ratio_min < p.ratio_max]


def scope_arrays(frame: pd.DataFrame, prefix: str) -> dict[str, np.ndarray]:
    """The five columns the velocity branch reads, as plain arrays."""
    return {
        "speed": frame[f"{prefix}passthrough_speed_seconds"].to_numpy(dtype="float64"),
        "ratio": frame[f"{prefix}passthrough_ratio"].to_numpy(dtype="float64"),
        "in_degree": frame[f"{prefix}in_degree"].to_numpy(dtype="float64"),
        "out_degree": frame[f"{prefix}out_degree"].to_numpy(dtype="float64"),
        "banks": frame[f"{prefix}distinct_counterparty_banks"].to_numpy(dtype="float64"),
    }


def velocity_mask(
    arrays: dict[str, np.ndarray], t: Thresholds, require_bank_spread: bool
) -> np.ndarray:
    """The velocity firing condition. This is the entire detection decision."""
    spread = arrays["banks"] >= 2.0 if require_bank_spread else np.ones_like(arrays["banks"], bool)
    return (
        (arrays["speed"] >= 0.0)
        & (arrays["speed"] <= t.speed_max_seconds)
        & (arrays["ratio"] >= t.ratio_min)
        & (arrays["ratio"] <= t.ratio_max)
        & (arrays["in_degree"] >= t.min_in_degree)
        & (arrays["out_degree"] >= t.min_out_degree)
        & spread
    )


def tune_on(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    rows: np.ndarray,
    require_bank_spread: bool,
    candidates: list[Thresholds],
) -> tuple[Thresholds, dict[str, float]]:
    """Pick thresholds on ``rows`` only: best recall subject to the FPR budget.

    ``rows`` is a boolean mask selecting the training fold. Nothing
    outside it is read, so the choice cannot depend on held-out rings.
    """
    sub = {k: v[rows] for k, v in arrays.items()}
    y_sub = y[rows].astype(bool)
    n_pos = int(y_sub.sum())
    n_neg = int((~y_sub).sum())

    best: tuple[Thresholds, dict[str, float]] | None = None
    best_key = (-1.0, -1.0)
    fallback: tuple[Thresholds, dict[str, float]] | None = None
    fallback_key = (-1.0, -1.0)
    for t in candidates:
        flag = velocity_mask(sub, t, require_bank_spread)
        tp = int((flag & y_sub).sum())
        fp = int((flag & ~y_sub).sum())
        recall = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        scored = {"precision": precision, "recall": recall, "fpr": fpr}
        if fpr <= FPR_BUDGET and (recall, precision) > best_key:
            best_key = (recall, precision)
            best = (t, scored)
        if (precision, recall) > fallback_key:
            fallback_key = (precision, recall)
            fallback = (t, scored)
    # Every grid point blowing the budget would mean the budget is wrong,
    # not that there is no rule; fall back to the most precise point and
    # let the held-out FPR say so out loud.
    chosen = best if best is not None else fallback
    assert chosen is not None, "empty threshold grid"
    return chosen


def cv_run(
    frame: pd.DataFrame,
    prefix: str,
    require_bank_spread: bool,
    seed: int,
    candidates: list[Thresholds],
) -> dict[str, object]:
    """One (scope, seed): tune per fold, pool the held-out decisions."""
    arrays = scope_arrays(frame, prefix)
    y = frame["is_passthrough_mule"].to_numpy().astype(bool)
    fold = fold_assignment(frame, seed)

    held_out = np.zeros(len(frame), dtype=bool)
    per_fold: list[dict[str, object]] = []
    for k in range(N_FOLDS):
        test = fold == k
        train = ~test
        t, train_metrics = tune_on(arrays, y, train, require_bank_spread, candidates)
        flag = velocity_mask(arrays, t, require_bank_spread)
        held_out[test] = flag[test]
        per_fold.append(
            {
                "fold": k,
                "thresholds": t.as_dict(),
                "train_metrics": train_metrics,
                "held_out_metrics": binary_metrics(y[test], flag[test]),
                "test_rows": int(test.sum()),
                "test_positives": int(y[test].sum()),
            }
        )

    in_sample_t, _ = tune_on(
        arrays, y, np.ones(len(frame), dtype=bool), require_bank_spread, candidates
    )
    in_sample_flag = velocity_mask(arrays, in_sample_t, require_bank_spread)
    return {
        "scope": prefix,
        "seed": seed,
        "held_out": binary_metrics(y, held_out),
        "in_sample": binary_metrics(y, in_sample_flag),
        "in_sample_thresholds": in_sample_t.as_dict(),
        "per_fold": per_fold,
        "threshold_variance": _threshold_variance(per_fold),
        "_held_out_flag": held_out,
    }


def _threshold_variance(per_fold: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Per threshold field: the values the five folds chose, and their spread.

    A field whose folds all agree is a stable property of the attack
    pattern. A field whose folds disagree wildly was being fitted to
    noise, and the in-sample number was buying that noise.
    """
    chosen = pd.DataFrame([f["thresholds"] for f in per_fold])  # type: ignore[index]
    return {
        field: {
            "values": [float(v) for v in chosen[field]],
            "mean": float(chosen[field].mean()),
            "std": float(chosen[field].std(ddof=0)),
            "n_distinct": int(chosen[field].nunique()),
        }
        for field in FIELDS
    }


def summarise(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Mean/std of held-out and in-sample metrics per scope, across seeds."""
    out: dict[str, dict[str, float]] = {}
    by_scope: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        by_scope.setdefault(str(run["scope_name"]), []).append(run)
    for scope, group in by_scope.items():
        entry: dict[str, float] = {}
        for which in ("held_out", "in_sample"):
            for key in RULE_METRICS:
                values = np.array([float(r[which][key]) for r in group])  # type: ignore[index]
                prefix = "" if which == "held_out" else "in_sample_"
                entry[f"{prefix}{key}_mean"] = float(values.mean())
                entry[f"{prefix}{key}_std"] = float(values.std(ddof=0))
        for key in RULE_METRICS:
            entry[f"{key}_optimism"] = entry[f"in_sample_{key}_mean"] - entry[f"{key}_mean"]
        out[scope] = entry
    return out


#: Scopes tuned, as (reported name, column prefix, require bank spread).
SCOPES: tuple[tuple[str, str, bool], ...] = (
    ("cross_bank", "cb_", True),
    ("single_bank", "sb_", False),
)


def main() -> None:
    """Tune per fold in every scope and write the honest metrics JSON."""
    cfg = MlConfig.from_env()
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    candidates = grid()
    print(f"grid points={len(candidates)} folds={N_FOLDS} seeds={len(SEEDS)}")

    runs: list[dict[str, object]] = []
    for name, prefix, spread in SCOPES:
        for seed in SEEDS:
            run = cv_run(frame, prefix, spread, seed, candidates)
            run["scope_name"] = name
            run["require_bank_spread"] = spread
            runs.append(run)
            h = run["held_out"]
            i = run["in_sample"]
            print(
                f"{name:<12} seed={seed} held_out P={h['precision']:.4f} R={h['recall']:.4f}"  # type: ignore[index]
                f" FPR={h['fpr']:.5f} | in_sample P={i['precision']:.4f}"  # type: ignore[index]
                f" R={i['recall']:.4f} FPR={i['fpr']:.5f}"  # type: ignore[index]
            )

    summary = summarise(runs)
    print("")
    print("honest gap (in-sample minus held-out)")
    for scope, entry in summary.items():
        print(
            f"  {scope:<12} precision {entry['precision_mean']:.4f}+/-{entry['precision_std']:.4f}"
            f" vs {entry['in_sample_precision_mean']:.4f} (optimism"
            f" {entry['precision_optimism']:+.4f})"
            f" | recall {entry['recall_mean']:.4f} vs {entry['in_sample_recall_mean']:.4f}"
            f" ({entry['recall_optimism']:+.4f})"
        )

    print("")
    print("per-fold threshold choices (spread across folds and seeds)")
    for name, _prefix, _spread in SCOPES:
        chosen = pd.DataFrame(
            [f["thresholds"] for r in runs if r["scope_name"] == name for f in r["per_fold"]]  # type: ignore[index,union-attr]
        )
        for field in FIELDS:
            counts = chosen[field].value_counts().sort_index()
            rendered = ", ".join(f"{k:g}x{v}" for k, v in counts.items())
            print(f"  {name:<12} {field:<20} std={chosen[field].std(ddof=0):>8.3f}  {rendered}")

    metrics = {
        "model": "rule_tuning_held_out",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation": (
            "grouped 5-fold CV over ring_id; thresholds grid-searched on each "
            "fold's TRAINING rings only and applied to its held-out rings; "
            "held-out decisions pooled across folds"
        ),
        "selection_objective": (
            f"maximise recall subject to training-fold FPR <= {FPR_BUDGET}, "
            "tie-broken by precision"
        ),
        "grid": {k: list(v) for k, v in VELOCITY_GRID.items()},
        "grid_points": len(candidates),
        "total_positives": int(frame["is_passthrough_mule"].sum()),
        "total_accounts": len(frame),
        "runs": [
            {k: v for k, v in r.items() if not k.startswith("_")} | {"scope": r["scope_name"]}
            for r in runs
        ],
        "summary": summary,
    }
    out_dir = cfg.experiments_dir / "rule_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    print("")
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
