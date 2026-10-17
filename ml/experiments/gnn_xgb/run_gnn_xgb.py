"""Lane E: GNN embeddings + XGBoost, inductive (vs. ml.gnn's transductive setup).

Standalone, offline-only. Never touches ml/out/models (the shipped
artifacts), the live app, Vertex Model Registry, or Gemini. Reuses
ml.evaluate.run_scope/oof_scores/fold_assignment/XGB_PARAMS and
ml.gnn.GraphSAGE/build_edge_index UNMODIFIED -- only this script's own
inductive training loop is new.

Why ml.gnn.py's GNN is transductive, and what changes here
-------------------------------------------------------------
ml.gnn.build_edge_index builds one graph over all 30,000 accounts per
horizon/seed -- train and test rows together -- and ml.gnn.run_fold
trains on that single graph with the loss masked to train rows only.
The loss never sees a held-out label, but every one of the 150 training
epochs runs SAGEConv message passing that aggregates held-out nodes'
*features and structure* into their train-fold neighbours' embeddings,
and a held-out node's own embedding is built from its real neighbours
using weights simultaneously being fit against those same neighbours'
labels. See ml/gnn.py's own docstring, which states this as a declared,
uncorrected asymmetry versus XGBoost (which never instantiates a test
row until after `model.fit` on train rows only).

This script instead, per fold:
  1. builds edge_index_train from ONLY the train-fold accounts (both
     endpoints must be train-fold) -- test-fold accounts and their
     edges are completely ABSENT from the graph the GNN trains on;
  2. trains GraphSAGE on that small subgraph (every node present IS a
     train node, so the loss covers the whole graph, no mask needed);
  3. freezes the trained weights, builds a SEPARATE full-graph
     (all 30,000 accounts, same cutoff-admission rule as ml.gnn) via
     the same ml.gnn.build_edge_index, and runs ONE forward pass in
     eval mode to get embeddings/scores for every account, test
     accounts included.

build_edge_index itself needs no change for step 1: its `accounts`
argument already drives which positions exist in the graph, and any
transfer whose endpoint is not in that index is dropped by the
existing `src.notna() & dst.notna()` check -- passing a train-only
account index already gives a train-only subgraph for free.

Both the embedding-producing GNN and the downstream XGBoost stage are
built from the SAME (frame, seed) -> fold_assignment, so an XGBoost
test-fold row's embedding always comes from a GNN that was trained
excluding that exact fold; the two-stage combination introduces no
leak beyond what each stage already has on its own.

Gate: variant (i), XGBoost on all 16 cb_ features alone, must reproduce
~0.622 PR-AUC and ~1.97 false alerts/catch at T=5m
(ml/out/models/model_a_metadata.json's validated_performance). If it
doesn't, this script stops before running anything else.

Per-(seed, horizon) checkpointing: every variant's result for one
(seed, horizon) pair is written to results_seed{N}_T{minutes}.json as
soon as that pair finishes, so a crash or time-box cutoff loses at most
one (seed, horizon) of work, and a rerun skips whatever is already
checkpointed.

Usage::

    python ml/experiments/gnn_xgb/run_gnn_xgb.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml.config import MlConfig
from ml.early import cutoff_series, ring_onsets
from ml.evaluate import (
    SEEDS,
    XGB_PARAMS,
    attach_labels,
    fold_assignment,
    run_scope,
)
from ml.features import build_features, feature_columns, load_tables
from ml.gnn import (
    DROPOUT,
    EPOCHS,
    HIDDEN,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GraphSAGE,
    build_edge_index,
)
from ml.gnn import run_scope as gnn_run_scope_transductive
from ml.metrics import evaluate_scores

HERE = Path(__file__).parent
MODEL_A_METADATA = Path("ml/out/models/model_a_metadata.json")

HORIZONS: tuple[int, ...] = (5, 180)

#: Model C's 11-feature transfer-only set (no cb_ prefix). Matches
#: ml/experiments/ablation/run_ablation.py's TRANSFER_ONLY exactly.
TRANSFER_ONLY: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "unique_counterparties",
    "distinct_counterparty_banks",
    "amount_in_total",
    "amount_out_total",
    "passthrough_ratio",
    "passthrough_speed_seconds",
    "round_amount_ratio",
)
#: The 5 columns Model C drops relative to Model A (4 identity + age).
IDENTITY_AND_AGE: tuple[str, ...] = (
    "device_count",
    "shared_device_count",
    "document_reuse_count",
    "document_reuse_flag",
    "account_age_days",
)

ALL16_COLS: list[str] = feature_columns("cb_")
TRANSFER11_COLS: list[str] = [f"cb_{name}" for name in TRANSFER_ONLY]

#: Minimum free RAM before starting a heavy (GNN/XGBoost) job. Below
#: this the job waits rather than risk OOM/thrashing -- see the
#: module-level `_wait_for_memory` docstring.
MIN_FREE_GB = 3.0


def _wait_for_memory(min_gb: float = MIN_FREE_GB, poll_seconds: int = 15) -> None:
    """Block until at least ``min_gb`` is free, polling and printing status.

    Never silently proceeds under the floor, and never shrinks data or
    seeds to fit -- it waits instead, per the explicit instruction this
    lane was approved under.
    """
    import psutil

    while True:
        available_gb = psutil.virtual_memory().available / 1e9
        if available_gb >= min_gb:
            return
        print(
            f"  [memory] {available_gb:.2f} GB free < {min_gb} GB floor; waiting {poll_seconds}s...",
            flush=True,
        )
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Inductive GraphSAGE: train on a train-fold-only subgraph, score everyone
# with the frozen weights over the full graph.
# ---------------------------------------------------------------------------


def _log1p_signed(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _fit_norm(x_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the log1p-transformed TRAINING rows only."""
    x = _log1p_signed(x_train_raw)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def _apply_norm(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = _log1p_signed(x_raw)
    return torch.from_numpy(((x - mean) / std).astype("float32"))


def _train_inductive(x_train: torch.Tensor, edge_index_train: torch.Tensor, y_train: np.ndarray, seed: int) -> GraphSAGE:
    """Train GraphSAGE on a graph containing ONLY train-fold nodes.

    Every node in this graph is a train node, so the loss is computed
    over the whole graph -- there is no mask, unlike ml.gnn.run_fold,
    because there is nothing held out present to mask.
    """
    torch.manual_seed(seed)
    target = torch.from_numpy(y_train.astype("float32"))
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))

    model = GraphSAGE(x_train.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(EPOCHS):
        optimiser.zero_grad()
        logits = model(x_train, edge_index_train)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        loss.backward()
        optimiser.step()
    return model


def _embed_and_score_full(model: GraphSAGE, x_full: torch.Tensor, edge_index_full: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """One frozen forward pass over the full graph. Mirrors GraphSAGE.forward
    exactly (dropout is a no-op in eval mode either way) so embeddings and
    the model's own score come from the identical computation.
    """
    model.eval()
    with torch.no_grad():
        h = F.relu(model.conv1(x_full, edge_index_full))
        h = F.relu(model.conv2(h, edge_index_full))
        logits = model.head(h).squeeze(-1)
        scores = torch.sigmoid(logits).numpy()
        embeddings = h.numpy()
    return scores, embeddings


def inductive_gnn_per_fold(
    frame: pd.DataFrame,
    columns: list[str],
    transfers: pd.DataFrame,
    bank: pd.Series,
    cutoff: pd.Series,
    seed: int,
    fold: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Train one inductive GraphSAGE per fold; return each fold's OWN full-graph
    scores and embeddings, keyed by fold, WITHOUT stitching embeddings across
    folds into one shared matrix.

    Why not stitch embeddings the way ml.evaluate.oof_scores stitches scalar
    scores: each fold trains an independent GraphSAGE from a fresh random
    init, so fold j's 64-dim latent space has no alignment with fold k's --
    "dimension 23" means something different in each. A scalar probability
    is comparable across models (both are calibrated to [0, 1] against the
    same label), so OOF-stitching scores is fine (see inductive_gnn_oof
    below); OOF-stitching raw embedding VECTORS is not, because the
    downstream XGBoost stage would then see rows whose embedding columns
    come from several mutually-incompatible latent spaces within one
    training set. Early in this run that exact mistake made the embedding
    columns look like structured noise, collapsing PR-AUC from ~0.62 to
    ~0.03 for the "XGBoost + GNN" variant -- see ml/experiments/gnn_xgb/README.md.
    Callers needing GNN+XGBoost together must pair fold k's XGBoost
    fit/predict with fold k's OWN embeddings only (see xgb_oof_with_gnn below).
    """
    accounts_full = pd.Index(frame["hashed_account_id"])
    x_full_raw = frame[columns].to_numpy(dtype="float64")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    edge_index_full = build_edge_index(transfers, accounts_full, cutoff, bank, same_bank_only=False)

    n_folds = int(fold.max()) + 1
    per_fold: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(n_folds):
        test = fold == k
        train = ~test
        accounts_train = accounts_full[train]
        edge_index_train = build_edge_index(transfers, accounts_train, cutoff, bank, same_bank_only=False)
        x_train_raw = x_full_raw[train]
        y_train = y[train]
        mean, std = _fit_norm(x_train_raw)
        x_train = _apply_norm(x_train_raw, mean, std)

        model = _train_inductive(x_train, edge_index_train, y_train, seed * 100 + k)

        x_full = _apply_norm(x_full_raw, mean, std)  # train-fitted stats applied to everyone
        scores_full, embeddings_full = _embed_and_score_full(model, x_full, edge_index_full)
        per_fold[k] = (scores_full, embeddings_full)

    return per_fold


def inductive_gnn_oof(per_fold: dict[int, tuple[np.ndarray, np.ndarray]], fold: np.ndarray, n: int) -> np.ndarray:
    """Out-of-fold inductive GNN SCORES (scalar probabilities only -- see
    inductive_gnn_per_fold's docstring for why embeddings are not stitched
    this way). Fold k's rows get the score from the model trained with
    fold k held out entirely from the training graph.
    """
    scores_oof = np.zeros(n, dtype="float64")
    for k, (scores_full, _embeddings_full) in per_fold.items():
        test = fold == k
        scores_oof[test] = scores_full[test]
    return scores_oof


def xgb_oof_with_gnn(
    frame: pd.DataFrame,
    raw_columns: list[str],
    seed: int,
    fold: np.ndarray,
    per_fold_gnn: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Grouped-CV XGBoost OOF scores on raw_columns + GNN embeddings, with
    fold k's XGBoost model trained AND evaluated using fold k's own GNN
    embeddings only -- never mixing embeddings from different folds'
    GraphSAGE models within one XGBoost training set (see
    inductive_gnn_per_fold's docstring).
    """
    from xgboost import XGBClassifier

    x_raw = frame[raw_columns].to_numpy(dtype="float32")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    n = len(frame)
    scores = np.zeros(n, dtype="float64")
    n_folds = int(fold.max()) + 1
    for k in range(n_folds):
        test = fold == k
        train = ~test
        _scores_full, embeddings_full = per_fold_gnn[k]
        x_full = np.concatenate([x_raw, embeddings_full.astype("float32")], axis=1)
        n_pos = int(y[train].sum())
        n_neg = int(train.sum()) - n_pos
        model = XGBClassifier(**XGB_PARAMS, random_state=seed * 100 + k, scale_pos_weight=n_neg / max(n_pos, 1))
        model.fit(x_full[train], y[train], verbose=False)
        scores[test] = model.predict_proba(x_full[test])[:, 1]
    return scores


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def check_gate(runs_t5m: list[dict[str, Any]]) -> None:
    shipped = json.loads(MODEL_A_METADATA.read_text())["validated_performance"]["T=5m"]
    pr_auc = float(np.mean([r["pr_auc"] for r in runs_t5m]))
    fa = float(np.mean([r["false_alerts_per_catch_at_80pct_recall"] for r in runs_t5m]))
    shipped_pr = shipped["pr_auc_mean"]
    shipped_fa = shipped["false_alerts_per_catch_at_80pct_recall_mean"]
    print(
        f"GATE: variant (i) PR-AUC={pr_auc:.4f} (shipped {shipped_pr:.4f}), "
        f"false_alerts/catch={fa:.3f} (shipped {shipped_fa:.3f})",
        flush=True,
    )
    if abs(pr_auc - shipped_pr) > 0.02 or abs(fa - shipped_fa) > 0.3:
        raise SystemExit(
            f"GATE FAILED: variant (i) does not reproduce shipped Model A closely enough "
            f"(PR-AUC diff {pr_auc - shipped_pr:+.4f}, false_alerts diff {fa - shipped_fa:+.3f}). "
            "Stopping -- nothing past this point is trusted."
        )
    print("GATE PASSED.", flush=True)


def run_one(cfg: MlConfig, tables: dict[str, pd.DataFrame], onsets: pd.Series, minutes: int, seed: int) -> dict[str, Any]:
    """Every variant, plus the transductive leak-check, for one (seed, horizon)."""
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    cutoff = cutoff_series(tables, onsets, minutes, seed)
    frame = attach_labels(build_features(cfg, tables=tables, cutoff=cutoff), cfg)
    fold = fold_assignment(frame, seed)
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")

    out: dict[str, Any] = {"seed": seed, "horizon": f"T={minutes}m", "rows": len(frame), "positives": int(y.sum())}

    # (i) XGBoost, Model A's 16 cb_ features alone.
    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] (i) xgboost_model_a ...", flush=True)
    t0 = time.time()
    run_i = run_scope(frame, "cb_", seed)
    xgb16_scores = run_i.pop("_scores")
    run_i.pop("per_pattern_recall_at_fpr_1pct", None)
    run_i.pop("strata", None)
    out["xgboost_model_a"] = run_i
    print(f"    pr_auc={run_i['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # Inductive GNN on the full 16-feature set -- feeds (ii), the leak
    # check's inductive side, and (iv)'s score averaging. One model per
    # fold; embeddings kept per-fold (never stitched) per
    # inductive_gnn_per_fold's docstring.
    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] inductive GNN, 16 features ...", flush=True)
    t0 = time.time()
    per_fold_16 = inductive_gnn_per_fold(frame, ALL16_COLS, tables["transfers"], bank, cutoff, seed, fold)
    gnn16_scores = inductive_gnn_oof(per_fold_16, fold, len(frame))
    out["gnn_inductive_16feat"] = evaluate_scores(y, gnn16_scores)
    print(f"    pr_auc={out['gnn_inductive_16feat']['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # (ii) XGBoost on Model A's 16 features + the inductive 16-feature GNN's
    # embeddings, fold-matched (fold k's XGBoost uses ONLY fold k's GNN).
    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] (ii) xgboost_model_a_plus_gnn ...", flush=True)
    t0 = time.time()
    scores_ii = xgb_oof_with_gnn(frame, ALL16_COLS, seed, fold, per_fold_16)
    out["xgboost_model_a_plus_gnn"] = evaluate_scores(y, scores_ii)
    print(f"    pr_auc={out['xgboost_model_a_plus_gnn']['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # (iii) 11-feature transfer-only alone, and + GNN embeddings built ONLY from those 11.
    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] (iii) xgboost_transfer_only (11 feat) ...", flush=True)
    t0 = time.time()
    run_iii_alone = run_scope(frame, "cb_", seed, drop=IDENTITY_AND_AGE)
    run_iii_alone.pop("_scores", None)
    run_iii_alone.pop("per_pattern_recall_at_fpr_1pct", None)
    run_iii_alone.pop("strata", None)
    out["xgboost_transfer_only"] = run_iii_alone
    print(f"    pr_auc={run_iii_alone['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] inductive GNN, 11 transfer features only ...", flush=True)
    t0 = time.time()
    per_fold_11 = inductive_gnn_per_fold(frame, TRANSFER11_COLS, tables["transfers"], bank, cutoff, seed, fold)
    gnn11_scores = inductive_gnn_oof(per_fold_11, fold, len(frame))
    out["gnn_inductive_11feat_transfer_only"] = evaluate_scores(y, gnn11_scores)
    print(f"    pr_auc={out['gnn_inductive_11feat_transfer_only']['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] (iii) xgboost_transfer_only_plus_gnn ...", flush=True)
    t0 = time.time()
    scores_iii_plus = xgb_oof_with_gnn(frame, TRANSFER11_COLS, seed, fold, per_fold_11)
    out["xgboost_transfer_only_plus_gnn"] = evaluate_scores(y, scores_iii_plus)
    print(f"    pr_auc={out['xgboost_transfer_only_plus_gnn']['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # (iv) score averaging of (i)'s XGBoost and the 16-feature inductive GNN.
    averaged = 0.5 * xgb16_scores + 0.5 * gnn16_scores
    out["score_average_xgboost_and_gnn"] = evaluate_scores(y, averaged)

    # Leak check: transductive GNN (ml.gnn, unmodified) vs the inductive one above,
    # same frame/fold/seed/16 features.
    _wait_for_memory()
    print(f"  [seed={seed} T={minutes}m] transductive GNN (leak check) ...", flush=True)
    t0 = time.time()
    edge_index_full = build_edge_index(tables["transfers"], pd.Index(frame["hashed_account_id"]), cutoff, bank, same_bank_only=False)
    run_transductive = gnn_run_scope_transductive(frame, edge_index_full, "cb_", seed)
    run_transductive.pop("_scores", None)
    out["gnn_transductive_16feat_leak_check"] = {
        k: v for k, v in run_transductive.items() if k not in ("strata",)
    }
    print(f"    pr_auc={run_transductive['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    return out


def summarise(all_results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    keys = ["pr_auc", "recall_at_fpr_1pct", "false_alerts_per_catch_at_80pct_recall"]
    variant_keys = [
        "xgboost_model_a",
        "gnn_inductive_16feat",
        "xgboost_model_a_plus_gnn",
        "xgboost_transfer_only",
        "gnn_inductive_11feat_transfer_only",
        "xgboost_transfer_only_plus_gnn",
        "score_average_xgboost_and_gnn",
        "gnn_transductive_16feat_leak_check",
    ]
    out: dict[str, dict[str, dict[str, float]]] = {}
    horizons = sorted({r["horizon"] for r in all_results})
    for horizon in horizons:
        out[horizon] = {}
        rows = [r for r in all_results if r["horizon"] == horizon]
        for vkey in variant_keys:
            entry: dict[str, float] = {}
            for k in keys:
                values = np.array([r[vkey][k] for r in rows], dtype="float64")
                entry[f"{k}_mean"] = float(values.mean())
                entry[f"{k}_std"] = float(values.std(ddof=0))
            out[horizon][vkey] = entry
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        choices=list(SEEDS),
        help=(
            "Restrict this process to one seed only, so the 3 seeds' independent work "
            "(each seed's 2 horizons) can run as separate OS processes in parallel. "
            "Does not change which seeds are evaluated overall -- only which ones THIS "
            "invocation computes; the final aggregated results_gnn_xgb.json is written "
            "once checkpoints for every seed in ml.evaluate.SEEDS exist on disk, "
            "however many processes produced them."
        ),
    )
    args = parser.parse_args()
    seeds_to_run = (args.seed,) if args.seed is not None else SEEDS

    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])
    print(f"torch={torch.__version__} accounts={len(tables['accounts'])} transfers={len(tables['transfers'])}", flush=True)

    all_results: list[dict[str, Any]] = []
    gate_checked = False
    for seed in seeds_to_run:
        for minutes in HORIZONS:
            ckpt_path = HERE / f"results_seed{seed}_T{minutes}.json"
            if ckpt_path.exists():
                print(f"seed={seed} T={minutes}m: reusing checkpoint {ckpt_path.name}", flush=True)
                result = json.loads(ckpt_path.read_text())
                all_results.append(result)
                if not gate_checked and minutes == 5:
                    check_gate([r["xgboost_model_a"] for r in all_results if r["horizon"] == "T=5m"])
                    gate_checked = True
                continue

            print(f"seed={seed} T={minutes}m", flush=True)
            result = run_one(cfg, tables, onsets, minutes, seed)
            ckpt_path.write_text(json.dumps(result, indent=2, default=float))
            print(f"  wrote {ckpt_path}", flush=True)
            all_results.append(result)

            if not gate_checked and minutes == 5:
                check_gate([r["xgboost_model_a"] for r in all_results if r["horizon"] == "T=5m"])
                gate_checked = True

    print(f"this process (seeds={seeds_to_run}) finished its own work.", flush=True)

    # Multiple processes may run concurrently, each restricted to one seed
    # (see --seed above). Only write the combined results_gnn_xgb.json once
    # EVERY seed's checkpoints exist on disk -- read them fresh rather than
    # reuse all_results, since this process only computed its own subset.
    all_ckpts = [HERE / f"results_seed{seed}_T{minutes}.json" for seed in SEEDS for minutes in HORIZONS]
    if not all(p.exists() for p in all_ckpts):
        missing = [p.name for p in all_ckpts if not p.exists()]
        print(f"not all seeds done yet; skipping final aggregation. still missing: {missing}", flush=True)
        return
    all_results = [json.loads(p.read_text()) for p in all_ckpts]

    results = {
        "recipe": "ml.evaluate.XGB_PARAMS / run_scope / oof_scores, ml.gnn.GraphSAGE / build_edge_index, unmodified; "
        "new inductive train-fold-only-subgraph + frozen-full-graph-eval loop in this script",
        "seeds": list(SEEDS),
        "horizons": {"T=5m": 5, "T=180m": 180},
        "all16_cols": ALL16_COLS,
        "transfer11_cols": TRANSFER11_COLS,
        "identity_and_age_dropped_for_transfer_only": list(IDENTITY_AND_AGE),
        "xgb_params": XGB_PARAMS,
        "gnn_hyperparams": {
            "hidden": HIDDEN,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
        },
        "per_seed_per_horizon": all_results,
        "summary": summarise(all_results),
    }
    out_path = HERE / "results_gnn_xgb.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
