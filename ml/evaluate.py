"""Grouped cross-validation harness shared by every MuleNet experiment.

The old 60/20/20 single split put roughly 24 rings in one test set; the
seed-to-seed spread that produced was larger than any effect we were
trying to measure. This module replaces it with **grouped 5-fold
cross-validation over rings**:

* the group of a ring member is its ``ring_id``, so every account in a
  ring lands in the same fold and no model can memorise a ring in
  training and be rewarded for recognising its siblings;
* rings are dealt into folds stratified by ``pattern_type``, so all four
  attack patterns appear in every fold;
* ring-free accounts (normals and hard negatives) are dealt round-robin
  after shuffling;
* out-of-fold predictions are pooled, so **every** positive in the
  dataset contributes to one PR curve per seed.

Three seeds (42, 43, 44) reshuffle both the fold assignment and the
model, and results are reported as mean +/- std across them.

Scope hygiene: ``run_scope`` selects columns by prefix and asserts the
selected set is disjoint from the other scope's set, so a single-bank
run cannot see a ``cb_`` column and vice versa.

Span stratification
-------------------
Rings now span 1, 2 or 3 banks (``ground_truth.ring_bank_count``,
0 for non-mules), crossed with ``pattern_type`` 9/12/9 per pattern, so
span and pattern are not confounded. Two deliberate choices:

*Train pooled, evaluate stratified.* One model is trained on all data
and its pooled out-of-fold scores are then **restricted** to each
stratum. Training three models — one per span — would cut each training
set roughly threefold and confound "wider rings are harder to see" with
"this model saw less data", which is the opposite of the question.

*Negatives held constant.* Every stratum keeps the **same** negative
set: all ``is_passthrough_mule == 0`` accounts. Only the positive set changes
(mules/collectors whose ring has that span). The alert budget an AML
team faces is the whole book of customers regardless of which ring a
mule sits in, so holding negatives fixed is the operationally faithful
choice and it keeps FPR-anchored metrics directly comparable.

Because the positive counts differ per stratum (184 / 302 / 226),
PR-AUC baselines differ per stratum too: ``stratified_metrics`` reports
``positives`` and ``prevalence`` alongside every PR-AUC, and cross-span
comparisons should be read off ROC-AUC and recall@fixed-FPR.

``ring_bank_count`` is carried on the frame for stratification and
reporting **only**; ``run_scope`` selects model columns strictly by
``sb_``/``cb_`` prefix, so it can never reach a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from ml.config import MlConfig
from ml.features import feature_columns
from ml.metrics import METRIC_KEYS, evaluate_scores

#: Roles the shipped pass-through detector is scored against: the
#: accounts that actually receive and forward ring money. 712 accounts.
#:
#: **``source`` is excluded here, and NOT because sources are innocent.**
#: An earlier version of this comment claimed ``source`` accounts were
#: "the victims/funders feeding a ring". That was wrong and is corrected
#: here: sources are allocated by the ring builder out of the same mule
#: pool as every other member, share the ring's devices and reused KYC
#: documents, and are opened on the herder's schedule. 74% of their
#: outflow goes to ring-mates. They are **complicit feeders**, not
#: victims — see :data:`COMPLICIT_ROLES`.
#:
#: They are excluded from this label because they are **structurally
#: undetectable by a pass-through detector**, which is a different claim.
#: Their median in-degree is 0: they originate funds rather than
#: receiving and forwarding them, so the rule's fan-in clause
#: (``in_degree >= 16``) can never fire on them and no pass-through
#: feature separates them from a normal account. Scoring the shipped
#: detector on sources would measure a detector we have not built.
#: Recovering them needs a different mechanism entirely — see
#: ``ml/stage2.py``, which reaches them through identity propagation
#: from an already-detected ring.
PASSTHROUGH_MULE_ROLES = frozenset({"mule", "collector"})

#: Every role that is part of the criminal operation, sources included.
#: 899 accounts. This is the honest ground truth of *guilt*;
#: :data:`PASSTHROUGH_MULE_ROLES` is the subset the shipped detector is
#: built to find. Reported alongside it so the gap between "complicit"
#: and "detectable by this detector" is a stated number, not a silence.
COMPLICIT_ROLES = frozenset({"mule", "collector", "source"})

#: Folds in the grouped cross-validation.
N_FOLDS = 5

#: Seeds. Everything else about the runs is identical across scopes.
SEEDS: tuple[int, ...] = (42, 43, 44)

#: Ring spans evaluated separately. 0 (non-mule) is never a stratum.
SPAN_STRATA: tuple[int, ...] = (1, 2, 3)

#: Prefix -> reported scope name.
SCOPE_NAMES: dict[str, str] = {"sb_": "single_bank", "cb_": "cross_bank"}

#: Fixed across both scopes, all folds and all seeds; no per-scope tuning.
#: ``early_stopping_rounds`` is gone with the validation split: every fold
#: trains the same 400 trees, which keeps the two scopes comparable.
XGB_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 4,
}


def attach_labels(features: pd.DataFrame, cfg: MlConfig) -> pd.DataFrame:
    """Join the feature table to ground truth and derive both labels and the CV group.

    Only ``ring_id``, ``role``, ``pattern_type``, ``cohort`` and
    ``ring_bank_count`` are taken from ground truth. ``role`` becomes the
    labels; the rest are grouping/stratification/reporting keys — none of
    them is ever passed to a model (see ``run_scope``'s prefix filter).

    **Two named labels, never one bare "is_mule".** ``is_passthrough_mule``
    (712) is the model/rule target; ``is_complicit`` (899) additionally
    covers ``source``, which is guilty but not pass-through. A single
    ``is_mule`` column conflates the two and is exactly the ambiguity
    these names exist to remove — ``ground_truth.parquet`` still ships
    one (``simulator/`` owns that file), so both labels are derived here
    from ``role`` and the raw column is never read.
    """
    truth = pd.read_parquet(cfg.table_path("ground_truth"))
    frame = features.merge(
        truth[
            [
                "hashed_account_id",
                "ring_id",
                "role",
                "pattern_type",
                "cohort",
                "ring_bank_count",
            ]
        ],
        on="hashed_account_id",
        validate="one_to_one",
    )
    frame["is_passthrough_mule"] = frame["role"].isin(PASSTHROUGH_MULE_ROLES).astype("int8")
    frame["is_complicit"] = frame["role"].isin(COMPLICIT_ROLES).astype("int8")
    in_ring = frame["ring_id"] != ""
    frame["group"] = np.where(in_ring, frame["ring_id"], "free:" + frame["hashed_account_id"])
    return frame


def fold_assignment(frame: pd.DataFrame, seed: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Deal whole rings into folds, stratified by pattern_type.

    Ring-free accounts are shuffled and dealt round-robin, so fold sizes
    stay balanced without any regard to the label.
    """
    rng = np.random.default_rng(seed)
    fold = np.full(len(frame), -1, dtype="int64")
    in_ring = (frame["ring_id"] != "").to_numpy()

    rings = frame.loc[in_ring, ["ring_id", "pattern_type"]].drop_duplicates()
    ring_fold: dict[str, int] = {}
    for _pattern, group in rings.groupby("pattern_type", sort=True):
        ring_ids = np.array(sorted(group["ring_id"].unique()))
        rng.shuffle(ring_ids)
        for i, ring_id in enumerate(ring_ids):
            ring_fold[ring_id] = i % n_folds
    fold[in_ring] = frame.loc[in_ring, "ring_id"].map(ring_fold).to_numpy()

    free = np.flatnonzero(~in_ring)
    rng.shuffle(free)
    fold[free] = np.arange(len(free)) % n_folds
    assert (fold >= 0).all()
    return fold


def oof_scores_and_contributions(
    frame: pd.DataFrame, columns: list[str], seed: int, fold: np.ndarray, n_folds: int = N_FOLDS
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities **and** per-row SHAP contributions.

    ``pred_contribs=True`` gives exact TreeSHAP values, so the returned
    contributions are the model's own attribution, not a surrogate. They
    are computed on the **held-out** fold by the model that did not see
    it, exactly like the scores: an explanation produced by a model that
    had already seen the account's ring would explain a memorised answer.

    Returns ``(scores, contribs)`` where ``contribs`` has shape
    ``(len(frame), len(columns) + 1)``; the trailing column is the
    XGBoost bias term and is not a feature.
    """
    x = frame[columns].to_numpy(dtype="float32")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    scores = np.zeros(len(frame), dtype="float64")
    contribs = np.zeros((len(frame), len(columns) + 1), dtype="float64")
    for k in range(n_folds):
        test = fold == k
        train = ~test
        n_pos = int(y[train].sum())
        n_neg = int(train.sum()) - n_pos
        model = XGBClassifier(
            **XGB_PARAMS,
            random_state=seed * 100 + k,
            scale_pos_weight=n_neg / max(n_pos, 1),
        )
        model.fit(x[train], y[train], verbose=False)
        scores[test] = model.predict_proba(x[test])[:, 1]
        contribs[test] = model.get_booster().predict(
            DMatrix(x[test], feature_names=columns), pred_contribs=True
        )
    return scores, contribs


def oof_scores(
    frame: pd.DataFrame, columns: list[str], seed: int, fold: np.ndarray, n_folds: int = N_FOLDS
) -> np.ndarray:
    """Out-of-fold predicted probabilities for every row of ``frame``."""
    return oof_scores_and_contributions(frame, columns, seed, fold, n_folds)[0]


def run_scope(
    frame: pd.DataFrame, prefix: str, seed: int, drop: tuple[str, ...] = ()
) -> dict[str, object]:
    """Grouped-CV run for one (scope, seed); returns metrics plus pooled scores."""
    columns = [c for c in feature_columns(prefix) if c[len(prefix) :] not in drop]
    other_prefix = "cb_" if prefix == "sb_" else "sb_"
    other = set(feature_columns(other_prefix))
    assert not set(columns) & other, "scope column sets must be disjoint"
    assert all(c.startswith(prefix) for c in columns), "scope columns must share one prefix"
    assert "ring_bank_count" not in columns, "span is a stratification key, never a feature"

    fold = fold_assignment(frame, seed)
    scores = oof_scores(frame, columns, seed, fold)
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")

    result: dict[str, object] = {
        "scope": SCOPE_NAMES[prefix],
        "seed": seed,
        "n_features": len(columns),
        "positives": int(y.sum()),
        "rows": len(frame),
        **evaluate_scores(y, scores),
    }
    result["per_pattern_recall_at_fpr_1pct"] = _per_pattern_recall(frame, scores)
    result["strata"] = stratified_metrics(frame, scores)
    result["_scores"] = scores
    return result


def stratified_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics restricted to each ring-span stratum, negatives held constant.

    Positives are the mules/collectors whose ring spans that many banks;
    the negative set is every ``is_passthrough_mule == 0`` account in all three strata.
    ``positives`` and ``prevalence`` ride along because PR-AUC is not
    comparable across strata with different positive counts.
    """
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    span = frame["ring_bank_count"].to_numpy(dtype="int64")
    is_negative = y == 0
    out: dict[str, dict[str, float]] = {}
    for stratum in SPAN_STRATA:
        keep = is_negative | ((y == 1) & (span == stratum))
        y_s = y[keep]
        entry: dict[str, float] = {
            "positives": int(y_s.sum()),
            "negatives": int((y_s == 0).sum()),
            "prevalence": float(y_s.mean()),
        }
        entry.update(evaluate_scores(y_s, scores[keep]))
        out[f"span{stratum}"] = entry
    return out


def summarise_strata(runs: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean/std of every metric per (scope, span stratum), across seeds."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for run in runs:
        scope = str(run["scope"])
        for stratum, entry in run["strata"].items():  # type: ignore[union-attr]
            out.setdefault(scope, {}).setdefault(stratum, {}).setdefault("_rows", []).append(entry)  # type: ignore[union-attr]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for scope, strata in out.items():
        for stratum, holder in strata.items():
            rows = pd.DataFrame(holder["_rows"])  # type: ignore[index]
            entry = {
                "positives": float(rows["positives"].iloc[0]),
                "prevalence": float(rows["prevalence"].iloc[0]),
            }
            for key in METRIC_KEYS:
                values = pd.to_numeric(rows[key], errors="coerce")
                entry[f"{key}_mean"] = float(values.mean())
                entry[f"{key}_std"] = float(values.std(ddof=0))
            summary.setdefault(scope, {})[stratum] = entry
    return summary


def _per_pattern_recall(frame: pd.DataFrame, scores: np.ndarray, target_fpr: float = 0.01) -> dict:
    """Recall per attack pattern at the global 1%-FPR threshold."""
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    negatives = scores[y == 0]
    if negatives.size == 0:
        return {}
    threshold = float(np.quantile(negatives, 1.0 - target_fpr))
    flagged = pd.Series(scores >= threshold, index=frame.index)
    out: dict[str, dict[str, float]] = {}
    positives = frame.loc[y == 1]
    for pattern, group in positives.groupby("pattern_type"):
        hits = int(flagged.loc[group.index].sum()) if len(group) else 0
        out[str(pattern)] = {"positives": len(group), "caught": hits, "recall": hits / len(group)}
    return out


def summarise(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Mean and std of every metric per scope, across seeds."""
    frame = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs])
    summary: dict[str, dict[str, float]] = {}
    for scope, group in frame.groupby("scope"):
        entry: dict[str, float] = {}
        for key in METRIC_KEYS:
            values = pd.to_numeric(group[key], errors="coerce")
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=0))
        summary[str(scope)] = entry
    return summary


def strip_scores(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop the private pooled-score arrays before the runs are serialised."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
