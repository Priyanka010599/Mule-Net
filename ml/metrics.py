"""Operationally meaningful detection metrics for a rare positive class.

Precision at a fixed 1% of the population was dropped: with ~2.3%
positives it is ceiling-bound near 1.0 for any decent model and near
0.37 when the population is mostly negatives, so it measures the class
balance rather than the model. What replaces it:

``precision_at_k``
    k is the number of positives actually present in the evaluated set,
    so a perfect ranker scores 1.0 and the metric is not capped by the
    budget being smaller than the truth.

``recall_at_fpr``
    The share of mules caught while the false-positive rate stays under
    a fixed alert budget (0.1% and 1% of legitimate accounts). This is
    the number an AML team can actually staff against.

``false_alerts_per_catch``
    FP/TP at the threshold that first reaches 80% recall: how many
    innocent customers an analyst must clear per real mule found. Lower
    is better; ``inf`` means 80% recall is unreachable at any threshold.

``roc_auc``
    Added for the span-stratified comparison. PR-AUC is **not**
    prevalence-independent: a stratum holding fewer positives has a
    lower random-baseline average precision for free, so raw PR-AUC
    cannot be compared across strata that differ in positive count.
    ROC-AUC and ``recall_at_fpr`` are invariant to the positive count
    (they are built from TPR and FPR separately), so they carry the
    cross-stratum comparison while PR-AUC stays the within-stratum,
    operationally meaningful number.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

#: Alert budgets, as a fraction of legitimate accounts.
FPR_TARGETS: tuple[float, ...] = (0.001, 0.01)

#: Recall level at which the false-alert burden is quoted.
BURDEN_RECALL = 0.8

#: Metric names produced by :func:`evaluate_scores`, in report order.
METRIC_KEYS: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "precision_at_k",
    "recall_at_fpr_0.1pct",
    "recall_at_fpr_1pct",
    "false_alerts_per_catch_at_80pct_recall",
)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision — area under the precision/recall curve."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve; prevalence-independent, unlike PR-AUC."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Precision among the top ``k`` scores; ``k`` defaults to the positive count."""
    k = int(y_true.sum()) if k is None else k
    if k <= 0:
        return float("nan")
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y_true[top].sum() / k)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Highest recall reachable while the false-positive rate stays <= target."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    allowed = fpr <= target_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def false_alerts_per_catch(
    y_true: np.ndarray, scores: np.ndarray, recall: float = BURDEN_RECALL
) -> float:
    """FP per TP at the first threshold reaching ``recall``; ``inf`` if unreachable."""
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = y_true[order].astype("int64")
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(1 - hits)
    reached = np.flatnonzero(true_positives >= math.ceil(recall * positives))
    if reached.size == 0:
        return float("inf")
    i = int(reached[0])
    return float(false_positives[i] / true_positives[i])


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """All reported metrics for one set of scores, keyed by ``METRIC_KEYS``."""
    y_true = np.asarray(y_true).astype("int64")
    scores = np.asarray(scores, dtype="float64")
    return {
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "precision_at_k": precision_at_k(y_true, scores),
        "recall_at_fpr_0.1pct": recall_at_fpr(y_true, scores, FPR_TARGETS[0]),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, scores, FPR_TARGETS[1]),
        "false_alerts_per_catch_at_80pct_recall": false_alerts_per_catch(y_true, scores),
    }


def binary_metrics(y_true: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    """Precision, recall and FPR for a hard 0/1 decision such as a rule."""
    y_true = np.asarray(y_true).astype(bool)
    flag = np.asarray(flag).astype(bool)
    tp = int((flag & y_true).sum())
    fp = int((flag & ~y_true).sum())
    fn = int((~flag & y_true).sum())
    tn = int((~flag & ~y_true).sum())
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "flagged": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
    }
