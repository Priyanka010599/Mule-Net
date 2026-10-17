"""GraphSAGE over the account graph, judged at T = 5/10/30 minutes.

Why the early-detection horizons are the *primary* evaluation
-------------------------------------------------------------
Over the full window both scopes sit near the ceiling and the
single-bank/cross-bank gap collapses — not because one bank can do the
job, but because by the end of the simulation every ring has finished
cashing out and the evidence is overwhelming in either view. That regime
flatters everyone and discriminates between nothing. The credible regime
is the one an AML team actually operates in: minutes after a ring wakes
up, with the ledger half-written. So ``CUTOFF_MINUTES`` is the headline
here and the full window is reported only as a reference row.

Point-in-time discipline on *edges*
-----------------------------------
The feature table's cutoff rules carry over unchanged, and one more rule
is added that only exists for a graph model: **an edge is admitted only
when its timestamp is at or before the cutoff of BOTH endpoints.** A
transfer that account ``u`` can legitimately see may still be in ``v``'s
future, and message passing is symmetric — admitting that edge would let
``v`` aggregate a neighbour's state from after ``v``'s own cutoff, which
is exactly the leak the whole harness exists to prevent. Taking the
stricter of the two cutoffs is the only choice that keeps both
directions honest.

Self-loops are added so an isolated account still has something to
aggregate; without them a node with no admitted edges returns zeros from
every convolution and the model cannot distinguish "quiet" from
"unobserved".

The transductive asymmetry, stated rather than hidden
------------------------------------------------------
This is a **transductive** node-classification setup: the graph is built
once per horizon and the model sees held-out nodes' *features and
structure* during training, though never their *labels*. XGBoost in
``evaluate.py`` is inductive — it sees nothing of the test rows at all.
That is a genuine asymmetry in the GNN's favour and it is not corrected
here, because correcting it (rebuilding a subgraph per fold) would also
strip the held-out nodes of the neighbourhoods that are the entire point
of using a graph model. It is instead **declared**: if the GNN wins by a
margin smaller than this advantage plausibly accounts for, it has not
really won. It is noted in the metrics JSON for the same reason.

Everything else is held identical to the XGBoost baseline: the same
grouped ring folds from ``evaluate.fold_assignment``, the same three
seeds, the same metrics, the same span strata, and ``ring_bank_count``
still never enters a feature.

Normalisation is fitted on **training rows only** per fold. Fitting the
scaler on all 30,000 accounts would be a quiet, real leak — small, but
exactly the kind this project has been careful about.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from ml.config import MlConfig
from ml.early import cutoff_series, ring_onsets
from ml.evaluate import (
    N_FOLDS,
    SCOPE_NAMES,
    SEEDS,
    attach_labels,
    fold_assignment,
    stratified_metrics,
    summarise,
    summarise_strata,
)
from ml.features import build_features, feature_columns, load_tables
from ml.metrics import METRIC_KEYS, evaluate_scores

#: Horizons evaluated. ``None`` (full window) is a reference row only.
CUTOFF_MINUTES: tuple[int | None, ...] = (5, 10, 30, None)

#: Hidden width of both SAGE layers.
HIDDEN = 64

#: Training epochs per fold. Fixed across scopes, folds and seeds, so no
#: scope can be advantaged by a longer budget.
EPOCHS = 150

LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE producing one logit per account."""

    def __init__(self, in_features: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)


def build_edge_index(
    transfers: pd.DataFrame,
    accounts: pd.Index,
    cutoff: pd.Series,
    bank: pd.Series,
    same_bank_only: bool,
) -> torch.Tensor:
    """Undirected edge index admitting only edges visible to *both* endpoints.

    The single-bank scope additionally keeps only intra-bank transfers,
    mirroring ``features._scope_features``: a bank does not see an edge
    whose far end is at another institution.
    """
    position = {account: i for i, account in enumerate(accounts)}
    frame = transfers
    if same_bank_only:
        intra = frame["from_hashed_account_id"].map(bank) == frame["to_hashed_account_id"].map(bank)
        frame = frame.loc[intra]

    src = frame["from_hashed_account_id"].map(position)
    dst = frame["to_hashed_account_id"].map(position)
    keep = src.notna() & dst.notna()
    # Strictest of the two cutoffs: message passing is symmetric.
    visible = (
        (frame["timestamp"] <= frame["from_hashed_account_id"].map(cutoff))
        & (frame["timestamp"] <= frame["to_hashed_account_id"].map(cutoff))
        & keep
    )
    u = src[visible].to_numpy(dtype="int64")
    v = dst[visible].to_numpy(dtype="int64")
    loops = np.arange(len(accounts), dtype="int64")
    rows = np.concatenate([u, v, loops])
    cols = np.concatenate([v, u, loops])
    return torch.from_numpy(np.stack([rows, cols]))


def _normalise(x: np.ndarray, train: np.ndarray) -> torch.Tensor:
    """log1p heavy tails, then z-score using **training rows only**."""
    x = np.sign(x) * np.log1p(np.abs(x))
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std == 0.0] = 1.0
    return torch.from_numpy(((x - mean) / std).astype("float32"))


def run_fold(
    x_raw: np.ndarray,
    edge_index: torch.Tensor,
    y: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train on ``train`` rows, return logits for every node."""
    torch.manual_seed(seed)
    x = _normalise(x_raw, train)
    target = torch.from_numpy(y.astype("float32"))
    train_mask = torch.from_numpy(train)

    n_pos = float(y[train].sum())
    n_neg = float(train.sum() - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))

    model = GraphSAGE(x.shape[1])
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.train()
    for _ in range(EPOCHS):
        optimiser.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], target[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, edge_index)).numpy()


def run_scope(
    frame: pd.DataFrame,
    edge_index: torch.Tensor,
    prefix: str,
    seed: int,
) -> dict[str, object]:
    """Grouped-CV GraphSAGE run for one (scope, seed)."""
    columns = feature_columns(prefix)
    other = set(feature_columns("cb_" if prefix == "sb_" else "sb_"))
    assert not set(columns) & other, "scope column sets must be disjoint"
    assert "ring_bank_count" not in columns, "span is a stratification key, never a feature"

    x_raw = frame[columns].to_numpy(dtype="float64")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    fold = fold_assignment(frame, seed)

    scores = np.zeros(len(frame), dtype="float64")
    for k in range(N_FOLDS):
        test = fold == k
        predictions = run_fold(x_raw, edge_index, y, ~test, seed * 100 + k)
        scores[test] = predictions[test]

    result: dict[str, object] = {
        "scope": SCOPE_NAMES[prefix],
        "seed": seed,
        "n_features": len(columns),
        "positives": int(y.sum()),
        "rows": len(frame),
        **evaluate_scores(y, scores),
    }
    result["strata"] = stratified_metrics(frame, scores)
    result["_scores"] = scores
    return result


def run_horizon(
    cfg: MlConfig, tables: dict[str, pd.DataFrame], onsets: pd.Series, minutes: int | None
) -> dict[str, object]:
    """One cutoff horizon, both scopes, all seeds."""
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    runs: list[dict[str, object]] = []
    edge_counts: dict[str, int] = {}

    for seed in SEEDS:
        cutoff = cutoff_series(tables, onsets, minutes, seed)
        frame = attach_labels(build_features(cfg, tables=tables, cutoff=cutoff), cfg)
        accounts = pd.Index(frame["hashed_account_id"])
        for prefix in ("sb_", "cb_"):
            edge_index = build_edge_index(
                tables["transfers"], accounts, cutoff, bank, same_bank_only=(prefix == "sb_")
            )
            edge_counts[SCOPE_NAMES[prefix]] = int(edge_index.shape[1])
            run = run_scope(frame, edge_index, prefix, seed)
            run["cutoff_minutes"] = minutes
            runs.append(run)
        if minutes is None:
            break  # the full window does not depend on the reference draw

    entry = {
        "cutoff_minutes": minutes,
        "label": "full" if minutes is None else f"T={minutes}m",
        "summary": summarise(runs),
        "strata_summary": summarise_strata(runs),
        "edges_including_self_loops": edge_counts,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
    }
    _print_horizon(entry)
    return entry


def _print_horizon(entry: dict[str, object]) -> None:
    summary: dict[str, dict[str, float]] = entry["summary"]  # type: ignore[assignment]
    print(f"\n--- GraphSAGE {entry['is_passthrough_mule']} ---")
    print(f"{'scope':<13}" + "".join(f"{k:>24}" for k in METRIC_KEYS))
    for scope in ("single_bank", "cross_bank"):
        if scope not in summary:
            continue
        cells = "".join(
            f"{summary[scope][k + '_mean']:.4f}+/-{summary[scope][k + '_std']:.4f}".rjust(24)
            for k in METRIC_KEYS
        )
        print(f"{scope:<13}{cells}")
    strata: dict[str, dict[str, dict[str, float]]] = entry["strata_summary"]  # type: ignore[assignment]
    for stratum in ("span1", "span2", "span3"):
        sb = strata.get("single_bank", {}).get(stratum)
        cb = strata.get("cross_bank", {}).get(stratum)
        if not (sb and cb):
            continue
        print(
            f"  {stratum} pos={int(sb['positives']):<4}"
            f" sb pr_auc={sb['pr_auc_mean']:.4f} | cb pr_auc={cb['pr_auc_mean']:.4f}"
            f" | d_pr_auc={cb['pr_auc_mean'] - sb['pr_auc_mean']:+.4f}"
        )


def main() -> None:
    """Run GraphSAGE at every horizon and write its metrics JSON."""
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])
    print(f"torch={torch.__version__} rings={len(onsets)}")

    horizons = [run_horizon(cfg, tables, onsets, minutes) for minutes in CUTOFF_MINUTES]
    primary = next(h for h in horizons if h["cutoff_minutes"] == 5)

    metrics = {
        "model": "graphsage",
        "generated_at": datetime.now(UTC).isoformat(),
        "architecture": f"2-layer GraphSAGE, hidden={HIDDEN}, dropout={DROPOUT}, {EPOCHS} epochs",
        "evaluation": (
            "grouped 5-fold CV over ring_id, out-of-fold pooled, per-account "
            "cutoffs; T=5/10/30m is the primary regime and the full window is "
            "a reference row only"
        ),
        "edge_admission": (
            "an edge is admitted only when its timestamp is at or before the "
            "cutoff of BOTH endpoints, because message passing is symmetric; "
            "self-loops added so isolated accounts still aggregate something"
        ),
        "transductive_caveat": (
            "the GNN sees held-out nodes' features and structure during "
            "training (never their labels); XGBoost in evaluate.py sees "
            "nothing of the test rows. This is a real asymmetry in the GNN's "
            "favour and is not corrected, only declared"
        ),
        "primary_horizon_minutes": 5,
        "horizons": horizons,
        # compare.py reads the top-level summary; use the primary regime,
        # not the flattering full window.
        "summary": primary["summary"],
        "runs": primary["runs"],
    }
    out_dir = cfg.experiments_dir / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    print(f"\nwrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
