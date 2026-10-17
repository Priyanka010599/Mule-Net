"""Per-ring detection timeline: when each ring was flagged, against its cash-out.

``ml/experiments/early_detection/metrics.json`` answers "what share of
rings did we catch before the money left?" as a single number per
(horizon, scope), averaged over three seeds. That is the headline, but
it is not inspectable: it cannot say *which* rings were caught, and a
reader who wants to see the trade-off ring by ring has nowhere to look.

This module publishes that breakdown as
``ml/out/ring_detection_timeline.parquet``: one row per
(ring, horizon, scope), carrying whether the ring was flagged at that
horizon and when money first reached its exit account.

Derived, never refitted
-----------------------
**Nothing here trains a model.** Every flag is read back from the
out-of-fold scores ``ml.early`` persisted under
``ml/experiments/early_detection/oof/``, together with the
:data:`ml.early.CATCH_FPR` threshold that run derived from its own
negative score distribution. The threshold is a property of the run, not
a constant, so reusing the stored value is the only way to reproduce the
published operating point exactly -- recomputing one here from a
different score set would silently move the decision boundary.

One seed, and why the target is not the published mean
------------------------------------------------------
``metrics.json``'s ``ring_catch`` block is a mean over
``ml.evaluate.SEEDS`` (42, 43, 44). A per-ring table is a per-seed
object: a ring is either flagged or not in a given run, so no single
seed's boolean table can average to 104.33 rings. This table is
:data:`CANONICAL_SEED` alone, and the number it must reproduce is that
seed's own contribution, published alongside the means as
``ring_catch_per_seed``. ``ml/test_timeline.py`` asserts that
equality exactly, both sides being derived from the same stored scores.

Flagging is a ring-level claim built from account-level scores, exactly
as ``ml.early.catch_share_before_exit`` builds it: a ring is flagged at
a horizon when **at least one** of its pass-through mules
(``role`` in ``mule``/``collector``) scores at or above the threshold.
``source`` members are excluded here for the same reason they are
excluded from the label everywhere else -- see
``ml.evaluate.PASSTHROUGH_MULE_ROLES``.

Horizon encoding
----------------
``T`` is **minutes**, matching ``cutoff_minutes`` in ``metrics.json``,
and the full window is encoded as **NULL** -- the table-level spelling
of the ``None`` that :data:`ml.early.CUTOFF_MINUTES` already uses. The
column is a nullable ``Int64`` so the horizon stays numerically
sortable; a string column mixing ``5`` and ``"full"`` would not be.

Note that the full-window rows are flagged-but-never-in-time by
construction: ``ml.early.catch_share_before_exit`` treats the full window
as after every cash-out, so its ``share_caught_before_exit`` is 0.0 for
reasons of definition, not of model performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.config import MlConfig
from ml.early import (
    CUTOFF_MINUTES,
    OOF_DIRNAME,
    exit_reached_at,
    horizon_tag,
    ring_onsets,
)
from ml.evaluate import PASSTHROUGH_MULE_ROLES, SCOPE_NAMES
from ml.features import load_tables

#: The seed this table is built from. ``ml.evaluate.SEEDS[0]``; the same
#: seed ``ml.publish`` scores with, so the two published artefacts
#: describe the same fold assignment.
CANONICAL_SEED = 42

#: Name of the published table, without extension.
TABLE_NAME = "ring_detection_timeline"

#: Scopes emitted, in report order. Both are published: the ``scope``
#: column exists to carry the single-bank/cross-bank comparison that is
#: this project's headline result, and a column with one value in it
#: would not carry anything.
SCOPES: tuple[str, ...] = tuple(SCOPE_NAMES.values())


def oof_path(cfg: MlConfig, minutes: int | None, scope: str, seed: int) -> Path:
    """Where ``ml.early`` persisted one run's scores and threshold."""
    return (
        cfg.experiments_dir
        / "early_detection"
        / OOF_DIRNAME
        / f"{horizon_tag(minutes)}_{scope}_seed{seed}.parquet"
    )


def load_oof(cfg: MlConfig, minutes: int | None, scope: str, seed: int) -> pd.DataFrame:
    """Read back one run's out-of-fold scores; never recompute them."""
    path = oof_path(cfg, minutes, scope, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m ml.early` to persist the "
            "out-of-fold scores; this module must never refit to recreate them."
        )
    frame = pd.read_parquet(path)
    assert frame["threshold"].nunique() == 1, "a run has exactly one operating point"
    return frame


def flagged_rings(oof: pd.DataFrame, truth: pd.DataFrame) -> set[str]:
    """Rings with at least one pass-through mule at or above the threshold.

    Mirrors ``ml.early.catch_share_before_exit``'s ``caught_rings`` step
    exactly, including its restriction to ``PASSTHROUGH_MULE_ROLES``.
    """
    threshold = float(oof["threshold"].iloc[0])
    positives = truth.loc[
        truth["role"].isin(PASSTHROUGH_MULE_ROLES), ["hashed_account_id", "ring_id"]
    ]
    scored = positives.merge(oof[["hashed_account_id", "score"]], on="hashed_account_id")
    return set(scored.loc[scored["score"] >= threshold, "ring_id"].unique())


def build_timeline(cfg: MlConfig, seed: int = CANONICAL_SEED) -> pd.DataFrame:
    """One row per (ring, horizon, scope) for ``seed``'s stored predictions."""
    tables = load_tables(cfg)
    truth = tables["ground_truth"]
    onsets = ring_onsets(tables["transfers"], truth)
    exits = exit_reached_at(tables["transfers"], truth)
    rings = sorted(onsets.index)

    rows: list[dict[str, object]] = []
    for minutes in CUTOFF_MINUTES:
        for scope in SCOPES:
            caught = flagged_rings(load_oof(cfg, minutes, scope, seed), truth)
            rows.extend(
                {
                    "ring_id": ring,
                    "T": minutes,
                    "scope": scope,
                    "flagged": ring in caught,
                    "onset_at": onsets[ring],
                    "exit_reached_at": exits[ring],
                }
                for ring in rings
            )

    frame = pd.DataFrame(rows)
    frame["T"] = frame["T"].astype("Int64")
    for column in ("onset_at", "exit_reached_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def update_manifest(cfg: MlConfig, frame: pd.DataFrame, seed: int) -> Path:
    """Record the table in ``ml/out/manifest.json`` beside the others.

    Read-modify-write rather than rebuilt: ``ml.publish`` owns that file
    and writes it wholesale, so this module must run **after** it. A
    ``ml.publish`` run following this one drops the entry and it has to
    be re-added -- flagged in the entry itself so a reader who finds it
    missing knows why.
    """
    path = cfg.ml_dir / "out" / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    # Set here as well as in ``ml.publish`` so the guidance is present
    # even in a manifest written before that key existed.
    manifest["regenerated_by"] = "python -m ml.publish_all"
    manifest["tables"][TABLE_NAME] = {
        "rows": len(frame),
        # No shared/ contract covers this table; shared/ is frozen until
        # after submission, so it is published uncontracted rather than
        # by redefining a schema locally. Logged in docs/post_submission.md.
        "validated_against": None,
        "columns": list(frame.columns),
        "source": (
            "derived from ml/experiments/early_detection/oof/ "
            f"(seed {seed}); NO model is fitted by ml.timeline"
        ),
        "written_by": (
            "ml.timeline, which must run AFTER ml.publish (it rewrites this "
            "manifest wholesale). Use `python -m ml.publish_all`."
        ),
        "seed": seed,
        "horizon_encoding": (
            "T is minutes after ring onset, matching metrics.json cutoff_minutes; "
            "NULL is the full window (ml.early.CUTOFF_MINUTES's None)"
        ),
        "flagged": (
            "ring has >=1 account with role in {mule, collector} scoring at or above "
            "that run's stored ml.early.CATCH_FPR (1% FPR) threshold"
        ),
        "onset_at": (
            "the ring's first intra-ring transfer (ml.early.ring_onsets) -- the same "
            "definition the T cutoffs are measured from, so onset_at + T is the instant "
            "the flag was decided at (clipped to the last transfer in the data); asserted "
            "against the per-run as_of persisted in oof/ by ml/test_timeline.py"
        ),
        "exit_reached_at": (
            "first intra-ring mule->collector transfer for the ring (ml.early.exit_reached_at)"
        ),
        "reproduces": (
            "aggregates exactly to metrics.json horizons[].ring_catch_per_seed"
            f"[scope]['{seed}']; the ring_catch means are over 3 seeds and are NOT "
            "reproducible from this one-seed table"
        ),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    return path


def main() -> None:
    """Build the timeline table and record it in the manifest."""
    cfg = MlConfig.from_env()
    frame = build_timeline(cfg)
    out_dir = cfg.ml_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TABLE_NAME}.parquet"
    frame.to_parquet(path, index=False)
    print(f"wrote {path} rows={len(frame)}")
    print(f"wrote {update_manifest(cfg, frame, CANONICAL_SEED)}")


if __name__ == "__main__":
    main()
