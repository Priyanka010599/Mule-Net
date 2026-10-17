"""Grouped-CV run on IBM AML HI-Small (Lane D2).

Standalone and offline. Reads the raw files from the gitignored
ml/experiments/public_benchmark/ folder and the committed grouping in
split.json (via the gitignored assignment.parquet it defines).

Label: account ever touches a laundering transaction (any transaction,
sender or receiver, full window). Groups: connected components over
positive transactions (make_split.py); folds are dealt by group.

Features (scale-free, transfer-only; amount sums are dropped because 15
currencies make raw sums meaningless):
  in_degree, out_degree, unique_in_counterparties,
  unique_out_counterparties, unique_counterparties,
  distinct_counterparty_banks, passthrough_ratio (same-currency flows
  only), passthrough_speed_seconds (ml.features burst logic, minute
  resolution).
Dropped on purpose: round_amount_ratio (multiple-of-100 is a unit
artefact across currencies) and account_age_days (no source field).

Two feature windows, point-in-time, identical for train and test:
  full: every transaction in the 18-day window.
  first3: transactions before 2022-09-04 00:00 (the first 3 days).

Models: XGBoost (ml.evaluate.XGB_PARAMS), logistic regression, and the
component-size baseline on the FULL unlabeled graph (all transactions,
no labels). The 1% FPR threshold is taken from training folds only: for
XGBoost and LR from inner out-of-fold scores of the training part; for
the baseline from training-fold negatives.

Usage::

    python ml/experiments/ibm_aml/run_grouped_cv.py
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.evaluate import N_FOLDS, SEEDS, XGB_PARAMS
from ml.experiments.ibm_aml.make_split import ASSIGNMENT_PATH, load_transactions
from ml.features import _edge_events, _passthrough_speed
from ml.metrics import evaluate_scores

HERE = Path(__file__).parent
OUT_PATH = HERE / "results_grouped.json"

FEATURES = [
    "in_degree",
    "out_degree",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "unique_counterparties",
    "distinct_counterparty_banks",
    "passthrough_ratio",
    "passthrough_speed_seconds",
]
RAW_AMOUNT_FEATURES = ["amount_in_total", "amount_out_total"]

FIRST3_END = pd.Timestamp("2022-09-04 00:00")
BURST_GAP_SECONDS = 1800
TARGET_FPR = 0.01
INNER_FOLDS = 3


def build_node_index(trans: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    keys = pd.Index(sorted(set(trans["from_key"]) | set(trans["to_key"])))
    frame = pd.DataFrame(
        {
            "src": keys.get_indexer(trans["from_key"]).astype("int32"),
            "dst": keys.get_indexer(trans["to_key"]).astype("int32"),
            "amount": trans["Amount Paid"].to_numpy(dtype="float32"),
            "same_currency": (trans["Receiving Currency"] == trans["Payment Currency"]).to_numpy(),
            "timestamp": trans["timestamp"].to_numpy(),
            "bank_src": trans["From Bank"].astype("int32").to_numpy(),
            "bank_dst": trans["To Bank"].astype("int32").to_numpy(),
        }
    )
    return frame, keys


def node_banks(frame: pd.DataFrame, n_nodes: int) -> pd.Series:
    banks = np.zeros(n_nodes, dtype="int32")
    banks[frame["src"].to_numpy()] = frame["bank_src"].to_numpy()
    banks[frame["dst"].to_numpy()] = frame["bank_dst"].to_numpy()
    return pd.Series(banks)


def point_in_time_features(
    frame: pd.DataFrame, n_nodes: int, banks: pd.Series, as_of_mask: np.ndarray
) -> pd.DataFrame:
    """Features from transactions in ``as_of_mask`` only; zeros for inactive accounts."""
    t = frame.loc[as_of_mask].reset_index(drop=True)
    edges = pd.DataFrame(
        {
            "from_hashed_account_id": t["src"],
            "to_hashed_account_id": t["dst"],
            "amount": t["amount"],
            "timestamp": t["timestamp"],
        }
    )
    events = _edge_events(edges, banks)
    index = pd.RangeIndex(n_nodes, name="node")
    inbound = events.loc[events["is_inbound"]]
    outbound = events.loc[~events["is_inbound"]]

    def count(frame_: pd.DataFrame, how: str, column: str | None = None) -> pd.Series:
        if frame_.empty:
            return pd.Series(0, index=index, dtype="int64")
        if how == "size":
            return frame_.groupby("account").size().reindex(index, fill_value=0).astype("int64")
        return frame_.groupby("account")[column].nunique().reindex(index, fill_value=0).astype("int64")

    out = pd.DataFrame(index=index)
    out["in_degree"] = count(inbound, "size")
    out["out_degree"] = count(outbound, "size")
    out["unique_in_counterparties"] = count(inbound, "nunique", "counterparty")
    out["unique_out_counterparties"] = count(outbound, "nunique", "counterparty")
    out["unique_counterparties"] = count(events, "nunique", "counterparty")
    out["distinct_counterparty_banks"] = count(events, "nunique", "counterparty_bank")

    same = t.loc[t["same_currency"]]
    out_sum = same.groupby("src")["amount"].sum().reindex(index, fill_value=0.0).to_numpy()
    in_sum = same.groupby("dst")["amount"].sum().reindex(index, fill_value=0.0).to_numpy()
    has_in = in_sum > 0.0
    out["passthrough_ratio"] = np.where(has_in, out_sum / np.where(has_in, in_sum, 1.0), 0.0)

    out["passthrough_speed_seconds"] = _passthrough_speed(events, index, BURST_GAP_SECONDS).to_numpy()
    out["amount_in_total"] = inbound.groupby("account")["amount"].sum().reindex(index, fill_value=0.0).to_numpy()
    out["amount_out_total"] = outbound.groupby("account")["amount"].sum().reindex(index, fill_value=0.0).to_numpy()
    return out


def full_graph_component_size(frame: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Size of each node's connected component over ALL transactions, no labels."""
    data = np.ones(len(frame), dtype="int8")
    adjacency = coo_matrix(
        (data, (frame["src"].to_numpy(), frame["dst"].to_numpy())), shape=(n_nodes, n_nodes)
    )
    _, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    return sizes[labels].astype("float32")


def full_graph_degree(frame: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Total in+out degree over ALL transactions, no cutoff, no labels."""
    degree = np.zeros(n_nodes, dtype="float32")
    np.add.at(degree, frame["src"].to_numpy(), 1.0)
    np.add.at(degree, frame["dst"].to_numpy(), 1.0)
    return degree


def assign_folds(groups: np.ndarray, has_positive: np.ndarray, seed: int, n_folds: int) -> np.ndarray:
    """Deal whole groups into folds; positive groups first so each fold gets positives."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(groups), -1, dtype="int64")
    positive_mask = has_positive == 1
    for mask in (positive_mask, ~positive_mask):
        sub = groups[mask]
        unique_groups = np.array(sorted(set(sub)))
        rng.shuffle(unique_groups)
        group_fold = {g: i % n_folds for i, g in enumerate(unique_groups)}
        fold[mask] = np.array([group_fold[g] for g in sub], dtype="int64")
    assert (fold >= 0).all()
    return fold


def logistic_matrix(x: np.ndarray) -> np.ndarray:
    counts = np.log1p(np.clip(x[:, :6], 0, None))
    ratio = np.log1p(np.clip(x[:, 6:7], 0, None))
    speed = x[:, 7:8]
    speed_known = (speed >= 0).astype("float64")
    speed_log = np.log1p(np.clip(speed, 0, None))
    return np.hstack([counts, ratio, speed_log, speed_known])


def fit_predict(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    if model_name.startswith("logistic_regression"):
        xt = logistic_matrix(x_train)
        xs = logistic_matrix(x_test)
        scaler = StandardScaler().fit(xt)
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(scaler.transform(xt), y_train)
        return model.predict_proba(scaler.transform(xs))[:, 1]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(**XGB_PARAMS, random_state=seed, scale_pos_weight=n_neg / max(n_pos, 1))
    model.fit(x_train, y_train, verbose=False)
    return model.predict_proba(x_test)[:, 1]


def nested_threshold(
    model_name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, has_positive: np.ndarray, seed: int
) -> float:
    """Inner grouped OOF on the training part; threshold = 99th pct of inner negatives' scores."""
    inner = assign_folds(groups, has_positive, seed + 1000, INNER_FOLDS)
    oof = np.zeros(len(y), dtype="float64")
    for k in range(INNER_FOLDS):
        tr = inner != k
        te = inner == k
        oof[te] = fit_predict(model_name, x[tr], y[tr], x[te], seed)
    return float(np.quantile(oof[y == 0], 1.0 - TARGET_FPR))


def run_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    has_positive: np.ndarray,
    seed: int,
    node_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Grouped outer CV. Pools out-of-fold scores; threshold per fold from training part only."""
    fold = assign_folds(groups, has_positive, seed, N_FOLDS)
    pooled_scores = np.zeros(len(y), dtype="float64")
    flagged = np.zeros(len(y), dtype=bool)
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        if node_score is not None:
            thr = float(np.quantile(node_score[tr & (y == 0)], 1.0 - TARGET_FPR))
            scores_te = node_score[te]
        else:
            thr = nested_threshold(model_name, x[tr], y[tr], groups[tr], has_positive[tr], seed + k)
            scores_te = fit_predict(model_name, x[tr], y[tr], x[te], seed + k)
        pooled_scores[te] = scores_te
        flagged[te] = scores_te >= thr
    metrics = evaluate_scores(y, pooled_scores)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    return {
        "model": model_name,
        "seed": seed,
        "pr_auc": metrics["pr_auc"],
        "false_alerts_per_catch_at_80pct_recall": metrics["false_alerts_per_catch_at_80pct_recall"],
        "recall_at_1pct_fpr_train_threshold": float((flagged & (y == 1)).sum() / positives),
        "realized_fpr": float((flagged & (y == 0)).sum() / negatives),
        "flagged_accounts": int(flagged.sum()),
        "positives": positives,
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "pr_auc",
        "false_alerts_per_catch_at_80pct_recall",
        "recall_at_1pct_fpr_train_threshold",
        "realized_fpr",
    ]
    out: dict[str, dict[str, float]] = {}
    for model in sorted({r["model"] for r in runs}):
        rows = [r for r in runs if r["model"] == model]
        entry: dict[str, float] = {}
        for k in keys:
            values = np.array([r[k] for r in rows], dtype="float64")
            entry[f"{k}_mean"] = float(values.mean())
            entry[f"{k}_std"] = float(values.std(ddof=0))
        out[model] = entry
    return out


def main() -> None:
    trans = load_transactions()
    frame, keys = build_node_index(trans)
    del trans
    gc.collect()
    n_nodes = len(keys)
    banks = node_banks(frame, n_nodes)

    assignment = pd.read_parquet(ASSIGNMENT_PATH)
    node = keys.get_indexer(assignment["acct"])
    order = np.argsort(node)
    node = node[order]
    groups = assignment["group"].to_numpy()[order]
    has_positive = assignment["has_positive"].to_numpy()[order].astype("int64")
    y = has_positive

    full_mask = np.ones(len(frame), dtype=bool)
    first3_mask = (frame["timestamp"] < FIRST3_END).to_numpy()

    full_component = full_graph_component_size(frame, n_nodes)[node]
    full_degree = full_graph_degree(frame, n_nodes)[node]
    component_stats = {
        "full_graph_component_count": int(len(np.unique(full_component))),
        "largest_component_size": int(full_component.max()),
        "share_of_accounts_in_largest_component": float((full_component == full_component.max()).mean()),
        "degree_median": float(np.median(full_degree)),
        "degree_p99": float(np.quantile(full_degree, 0.99)),
        "degree_max": float(full_degree.max()),
    }

    feats_full = point_in_time_features(frame, n_nodes, banks, full_mask).iloc[node]
    x_full = feats_full[FEATURES].to_numpy(dtype="float32")
    x_full_raw = feats_full[FEATURES + RAW_AMOUNT_FEATURES].to_numpy(dtype="float32")
    del feats_full
    gc.collect()

    feats_first3 = point_in_time_features(frame, n_nodes, banks, first3_mask).iloc[node]
    x_first3 = feats_first3[FEATURES].to_numpy(dtype="float32")
    del feats_first3, frame, banks
    gc.collect()

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_path = HERE / f"results_seed{seed}.json"
        if seed_path.exists():
            print(f"seed={seed}: reusing checkpoint {seed_path.name}", flush=True)
            runs.extend(json.loads(seed_path.read_text()))
            continue
        print(f"seed={seed}", flush=True)
        batch = []
        for args in (
            ("xgboost_full", x_full, None),
            ("xgboost_first3", x_first3, None),
            ("xgboost_full_raw_amounts_sensitivity", x_full_raw, None),
            ("logistic_regression_full", x_full, None),
            ("logistic_regression_first3", x_first3, None),
            ("component_size_baseline_full_graph", x_full, full_component),
            ("degree_threshold_baseline_full_graph", x_full, full_degree),
        ):
            name, x_arr, node_score = args
            batch.append(run_model(name, x_arr, y, groups, has_positive, seed, node_score=node_score))
            gc.collect()
        for r in batch:
            print(
                f"  {r['model']}: pr_auc={r['pr_auc']:.4f} "
                f"r@1%={r['recall_at_1pct_fpr_train_threshold']:.4f} "
                f"realized_fpr={r['realized_fpr']:.4f} "
                f"fa/catch={r['false_alerts_per_catch_at_80pct_recall']:.3f}",
                flush=True,
            )
        seed_path.write_text(json.dumps(batch, indent=2, default=float))
        print(f"  wrote {seed_path}", flush=True)
        runs.extend(batch)

    results = {
        "groups": "connected components over positive transactions; folds dealt by group",
        "label": "account touches a laundering transaction (full window)",
        "features": FEATURES,
        "sensitivity_extra_features": RAW_AMOUNT_FEATURES,
        "windows": {"full": "all 18 days", "first3": "2022-09-01 to 2022-09-04 00:00 (exclusive)"},
        "threshold": (
            "1% FPR from training part only (inner grouped OOF for trained models; "
            "training negatives for the baseline)"
        ),
        "accounts": int(len(y)),
        "positives": int(y.sum()),
        "component_baseline_graph": component_stats,
        "runs": runs,
        "summary": summarise(runs),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
