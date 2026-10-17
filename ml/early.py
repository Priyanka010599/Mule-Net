"""Early-detection evaluation: how soon after a ring wakes up can we see it?

For each cutoff ``T`` in minutes, every account is scored from **only**
what its bank could have observed by its own cutoff instant, and the
single-bank and cross-bank models are compared at that horizon.

Ring onset
----------
A ring's onset is the timestamp of its first *intra-ring* transfer —
the first edge whose sender and receiver are both members of the same
ring. In this dataset those edges are exactly ``source -> mule`` and
``mule -> collector``, so the onset is the start of the first fan-in
wave. Using "first transfer into any ring member" instead would pick up
the member's ordinary background traffic, which starts on day 1 and has
nothing to do with the ring.

The negatives problem
---------------------
The cutoff is defined per ring, but 29,109 accounts belong to no ring
and therefore have no natural onset. Giving them the full window while
positives are truncated would invert the leak: mules would look
artificially quiet next to fully-observed normals, and every number
would be meaningless.

What is done instead: **each non-ring account is assigned a reference
timestamp resampled from the empirical distribution of ring onsets.**
Its cutoff is then ``reference + T`` exactly as a ring member's is, so
the two groups face statistically identical observation windows and the
comparison is between "an account whose ring started T minutes ago" and
"an account observed at an instant drawn from the same clock".

One constraint is mirrored from the generator: a ring cannot activate
until a day after its last member opened, so a positive's cutoff is
always at least a day after its own account opened. Negatives therefore
resample only from onsets at or after ``opened_at + 1 day``; every
account in this dataset has at least one such onset available. Without
this, a fraction of negatives would be handed a cutoff *before* they
existed and would trivially score zero on everything.

``sanity_check`` prints the cutoff distribution per group so the claim
that neither group is advantaged can be inspected rather than trusted.

Exit account
------------
The ring's **collector** — the single cash-out account that every mule
forwards to and that then pushes funds to accounts outside the ring.
Money "has reached the exit" by time ``t`` when at least one
``mule -> collector`` transfer inside that ring has settled at or before
``t``. A ring is *caught in time* when it is flagged at the model's
1%-FPR operating point at a cutoff strictly earlier than that instant.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import MlConfig
from ml.evaluate import (
    SEEDS,
    attach_labels,
    run_scope,
    strip_scores,
    summarise,
    summarise_strata,
)
from ml.features import build_features, full_window_cutoff, load_tables
from ml.metrics import METRIC_KEYS

#: Minutes after a ring's onset at which features are cut off. ``None``
#: means the full window: every account observed at the last transfer.
#:
#: **The single source of truth for the horizon grid.** Every consumer --
#: ``ml.timeline``, the tests, ``ui/`` -- derives its grid from this tuple
#: or from the published table, never from a hand-copied list.
#:
#: The grid runs to 180 minutes because the *exit* distribution does.
#: Money reaches the collector between 3.5 and 165.5 minutes after onset
#: (median 89.4), so a grid ending at 60 left 93 of 120 rings with their
#: cash-out beyond the last horizon. "Caught before exit" is close to
#: free for those rings: almost any flag at all precedes an exit that the
#: measurement never reaches. The single-bank/cross-bank contrast was
#: unaffected -- both scopes faced the identical grid -- but the absolute
#: share was flattered by the censoring. At T=180 every exit falls inside
#: the grid, so the number means what it says.
CUTOFF_MINUTES: tuple[int | None, ...] = (5, 10, 20, 30, 60, 90, 120, 180, None)

#: Ring activation lag baked into the simulator, mirrored when drawing
#: reference times for accounts that belong to no ring.
ACTIVATION_LAG = pd.Timedelta(days=1)

#: Operating point at which "caught" is decided.
CATCH_FPR = 0.01

#: Subdirectory of the experiment holding one parquet per (horizon,
#: scope, seed) with that run's pooled out-of-fold scores and the
#: :data:`CATCH_FPR` threshold derived from them.
#:
#: These exist because ``ml.evaluate.strip_scores`` deliberately drops
#: the score arrays before ``metrics.json`` is written, which left every
#: downstream per-account or per-ring question unanswerable without
#: refitting. ``ml/out/ring_detection_timeline.parquet`` is built from
#: these files and from nothing else.
OOF_DIRNAME = "oof"


def _ring_of(truth: pd.DataFrame) -> pd.Series:
    return truth.set_index("hashed_account_id")["ring_id"]


def _intra_ring_edges(transfers: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Transfers whose sender and receiver sit in the same ring."""
    ring = _ring_of(truth)
    role = truth.set_index("hashed_account_id")["role"]
    src_ring = transfers["from_hashed_account_id"].map(ring).fillna("")
    dst_ring = transfers["to_hashed_account_id"].map(ring).fillna("")
    keep = (src_ring != "") & (src_ring == dst_ring)
    return pd.DataFrame(
        {
            "ring_id": src_ring[keep],
            "timestamp": transfers.loc[keep, "timestamp"],
            "from_role": transfers.loc[keep, "from_hashed_account_id"].map(role),
            "to_role": transfers.loc[keep, "to_hashed_account_id"].map(role),
        }
    )


def ring_onsets(transfers: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """ring_id -> timestamp of the ring's first intra-ring transfer."""
    return _intra_ring_edges(transfers, truth).groupby("ring_id")["timestamp"].min()


def exit_reached_at(transfers: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """ring_id -> when money first reached the exit (collector) account."""
    edges = _intra_ring_edges(transfers, truth)
    to_exit = edges.loc[(edges["from_role"] == "mule") & (edges["to_role"] == "collector")]
    return to_exit.groupby("ring_id")["timestamp"].min()


def reference_times(tables: dict[str, pd.DataFrame], onsets: pd.Series, seed: int) -> pd.Series:
    """Per-account reference instant: its ring's onset, or a resampled one.

    Non-ring accounts draw uniformly from the ring onsets that are at
    least ``ACTIVATION_LAG`` after the account opened, so their windows
    match the positives' distribution without ever starting before the
    account existed.
    """
    accounts = tables["accounts"]
    truth = tables["ground_truth"]
    index = pd.Index(accounts["hashed_account_id"], name="hashed_account_id")
    ring = _ring_of(truth).reindex(index).fillna("")

    reference = ring.map(onsets)
    missing = reference.isna().to_numpy()
    if missing.any():
        rng = np.random.default_rng(seed)
        pool = pd.DatetimeIndex(sorted(onsets))
        opened = accounts.set_index("hashed_account_id")["opened_at"].reindex(index)
        earliest = (opened + ACTIVATION_LAG)[missing]
        # First eligible onset per account, then a uniform draw from the tail.
        lo = np.minimum(pool.searchsorted(earliest, side="left"), len(pool) - 1)
        draw = lo + (rng.random(len(lo)) * (len(pool) - lo)).astype("int64")
        pick = np.minimum(draw, len(pool) - 1)
        reference = reference.copy()
        reference.iloc[np.flatnonzero(missing)] = pool[pick]
    return pd.to_datetime(reference)


def cutoff_series(
    tables: dict[str, pd.DataFrame], onsets: pd.Series, minutes: int | None, seed: int
) -> pd.Series:
    """Per-account cutoff instant for one horizon."""
    if minutes is None:
        return full_window_cutoff(tables)
    horizon_end = tables["transfers"]["timestamp"].max()
    cutoff = reference_times(tables, onsets, seed) + pd.Timedelta(minutes=minutes)
    return cutoff.clip(upper=horizon_end)


def sanity_check(frame: pd.DataFrame, minutes: int | None) -> dict[str, object]:
    """Show that positives and negatives face comparable observation windows."""
    out: dict[str, object] = {"cutoff_minutes": minutes}
    groups = (
        ("positive", frame["is_passthrough_mule"] == 1),
        ("negative", frame["is_passthrough_mule"] == 0),
    )
    for name, mask in groups:
        group = frame.loc[mask]
        cut = pd.to_datetime(group["as_of"])
        out[name] = {
            "n": int(len(group)),
            "cutoff_median": str(cut.median()),
            "cutoff_q10": str(cut.quantile(0.10)),
            "cutoff_q90": str(cut.quantile(0.90)),
            "median_cb_in_degree": float(group["cb_in_degree"].median()),
            "median_cb_out_degree": float(group["cb_out_degree"].median()),
        }
    return out


def catch_share_before_exit(
    frame: pd.DataFrame,
    scores: np.ndarray,
    onsets: pd.Series,
    exits: pd.Series,
    minutes: int | None,
) -> dict[str, float]:
    """Share of rings flagged at 1% FPR before money reaches the collector."""
    y = frame["is_passthrough_mule"].to_numpy()
    negatives = scores[y == 0]
    threshold = float(np.quantile(negatives, 1.0 - CATCH_FPR))
    flagged = pd.Series(scores >= threshold, index=frame.index)

    positives = frame.loc[y == 1]
    caught_rings = set(positives.loc[flagged.loc[positives.index], "ring_id"].unique())
    rings = sorted(onsets.index)
    if minutes is None:
        in_time = {r: False for r in rings}  # full window is after every cash-out
        headroom = 0
    else:
        cut = onsets + pd.Timedelta(minutes=minutes)
        in_time = {r: bool(cut[r] < exits[r]) for r in rings}
        headroom = sum(in_time.values())

    caught = sum(1 for r in rings if r in caught_rings)
    caught_in_time = sum(1 for r in rings if r in caught_rings and in_time[r])
    return {
        "rings": len(rings),
        "rings_caught": caught,
        "ring_recall": caught / len(rings),
        "rings_money_not_yet_at_exit": headroom,
        "rings_caught_before_exit": caught_in_time,
        "share_caught_before_exit": caught_in_time / len(rings),
        # Carried out so the run's operating point can be persisted
        # beside its scores. The threshold is a property of the run's own
        # negative score distribution, not a constant, so a consumer that
        # cannot see it cannot reproduce the flag.
        "threshold": threshold,
    }


def horizon_tag(minutes: int | None) -> str:
    """Filename tag for one horizon; ``None`` is the full window.

    Mirrors the ``label`` convention already used in ``metrics.json``
    (``T=5m`` / ``full``) minus the characters a filename should not
    carry, so ``full`` stays the encoding of "no cutoff" everywhere.
    """
    return "full" if minutes is None else f"T{minutes}"


def write_oof(
    oof_dir: Path,
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    minutes: int | None,
    scope: str,
    seed: int,
) -> Path:
    """Persist one run's out-of-fold scores, its operating point and its cutoffs.

    ``as_of`` is the **actual** per-account observation instant the score
    was computed from — ``cutoff_series``' output after the clip to the
    end of the data. It is stored rather than left to be recomputed
    because a consumer that rebuilds it from ``onset + T`` is testing its
    own arithmetic, not the run: the clip, and the resampled reference
    instants that non-ring accounts get, are both invisible from the
    outside.
    """
    path = oof_dir / f"{horizon_tag(minutes)}_{scope}_seed{seed}.parquet"
    pd.DataFrame(
        {
            "hashed_account_id": frame["hashed_account_id"].to_numpy(),
            "score": np.asarray(scores, dtype="float64"),
            "as_of": pd.to_datetime(frame["as_of"].to_numpy(), utc=True),
            # Constant down the column by construction: it is the run's
            # single operating point, stored per row so the file is
            # self-contained and a reader never has to join it back.
            "threshold": np.full(len(frame), threshold, dtype="float64"),
        }
    ).to_parquet(path, index=False)
    return path


def main() -> None:
    """Run the full early-detection sweep and write its metrics JSON."""
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])
    exits = exit_reached_at(tables["transfers"], tables["ground_truth"])
    print(f"rings={len(onsets)} with-exit={len(exits)}")

    oof_dir = cfg.experiments_dir / "early_detection" / OOF_DIRNAME
    oof_dir.mkdir(parents=True, exist_ok=True)

    horizons: list[dict[str, object]] = []
    for minutes in CUTOFF_MINUTES:
        runs: list[dict[str, object]] = []
        checks: list[dict[str, object]] = []
        catches: dict[str, list[dict[str, float]]] = {"single_bank": [], "cross_bank": []}
        per_seed: dict[str, dict[str, dict[str, float]]] = {}
        for seed in SEEDS:
            cutoff = cutoff_series(tables, onsets, minutes, seed)
            frame = attach_labels(build_features(cfg, tables=tables, cutoff=cutoff), cfg)
            checks.append(sanity_check(frame, minutes))
            for prefix in ("sb_", "cb_"):
                run = run_scope(frame, prefix, seed)
                run["cutoff_minutes"] = minutes
                scope = str(run["scope"])
                catch = catch_share_before_exit(frame, run["_scores"], onsets, exits, minutes)
                catches[scope].append(catch)
                per_seed.setdefault(scope, {})[str(seed)] = catch
                write_oof(oof_dir, frame, run["_scores"], catch["threshold"], minutes, scope, seed)
                runs.append(run)
            if minutes is None:
                break  # full window does not depend on the reference draw

        label = "full" if minutes is None else f"T={minutes}m"
        entry = {
            "cutoff_minutes": minutes,
            "label": label,
            "summary": summarise(runs),
            "strata_summary": summarise_strata(runs),
            "runs": strip_scores(runs),
            "sanity_check": checks,
            "ring_catch": {
                scope: {
                    k: float(np.mean([c[k] for c in lst]))
                    for k in (
                        "ring_recall",
                        "rings_money_not_yet_at_exit",
                        "rings_caught_before_exit",
                        "share_caught_before_exit",
                    )
                }
                for scope, lst in catches.items()
                if lst
            },
            # The per-seed rows the means above are taken over. A single
            # seed's contribution is not recoverable from a mean, so any
            # artefact built from one seed's stored scores (see
            # :data:`OOF_DIRNAME`) has nothing to be checked against
            # unless the per-seed numbers are kept too. Carries the run's
            # ``threshold`` as well, so the operating point is in the
            # metrics file and not only in the parquet.
            "ring_catch_per_seed": per_seed,
        }
        horizons.append(entry)
        _print_horizon(entry)

    metrics = {
        "model": "early_detection",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation": "grouped 5-fold CV over ring_id, out-of-fold pooled, per-account cutoffs",
        "exit_account": "ring collector; reached at its first intra-ring mule->collector transfer",
        "onset": "first intra-ring transfer of the ring",
        "negative_reference": "resampled from ring onsets at or after opened_at + 1 day",
        "horizons": horizons,
        "summary": horizons[-1]["summary"],
    }
    out_dir = cfg.experiments_dir / "early_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    print(f"wrote {out_dir / 'metrics.json'}")


def _print_horizon(entry: dict[str, object]) -> None:
    summary: dict[str, dict[str, float]] = entry["summary"]  # type: ignore[assignment]
    catch: dict[str, dict[str, float]] = entry["ring_catch"]  # type: ignore[assignment]
    print(f"\n--- {entry['label']} ---")
    print(f"{'scope':<13}" + "".join(f"{k:>24}" for k in METRIC_KEYS) + f"{'caught<exit':>13}")
    for scope in ("single_bank", "cross_bank"):
        if scope not in summary:
            continue
        cells = "".join(
            f"{summary[scope][k + '_mean']:.4f}+/-{summary[scope][k + '_std']:.4f}".rjust(24)
            for k in METRIC_KEYS
        )
        share = catch.get(scope, {}).get("share_caught_before_exit", float("nan"))
        print(f"{scope:<13}{cells}{share:>13.3f}")
    strata: dict[str, dict[str, dict[str, float]]] = entry["strata_summary"]  # type: ignore[assignment]
    for stratum in ("span1", "span2", "span3"):
        sb = strata.get("single_bank", {}).get(stratum)
        cb = strata.get("cross_bank", {}).get(stratum)
        if not (sb and cb):
            continue
        print(
            f"  {stratum} pos={int(sb['positives']):<4}"
            f" sb pr_auc={sb['pr_auc_mean']:.4f} roc={sb['roc_auc_mean']:.4f}"
            f" r@1%={sb['recall_at_fpr_1pct_mean']:.4f} |"
            f" cb pr_auc={cb['pr_auc_mean']:.4f} roc={cb['roc_auc_mean']:.4f}"
            f" r@1%={cb['recall_at_fpr_1pct_mean']:.4f} |"
            f" d_pr_auc={cb['pr_auc_mean'] - sb['pr_auc_mean']:+.4f}"
            f" d_roc={cb['roc_auc_mean'] - sb['roc_auc_mean']:+.4f}"
            f" d_r@1%={cb['recall_at_fpr_1pct_mean'] - sb['recall_at_fpr_1pct_mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
