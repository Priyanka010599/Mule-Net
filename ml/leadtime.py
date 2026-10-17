"""How much warning does a ring give before the money leaves?

``ring_catch`` answers "was the ring flagged before its cash-out?" as a
share. That is a yes/no over a grid, and it hides the quantity an AML
team actually staffs against: **how many minutes of warning** they got.
This module measures that, per ring and per scope, from the persisted
out-of-fold scores -- nothing is refitted.

Three kinds of missing, never averaged as zero
----------------------------------------------
A ring that gives no warning is not a ring with zero warning, and the
two must not land in the same mean. Every ring falls into exactly one of:

``flagged_in_grid``
    Flagged at a finite horizon. ``warning_minutes`` is
    ``exit - first_flag``, and is **negative** when the first flag lands
    after the money already left. A real, signed number.

``flagged_full_window_only``
    Flagged at no finite horizon, but flagged once the full window is
    observed. The full window ends at the last transfer in the dataset,
    days after the cash-out, so the honest warning time is "large and
    negative" with no meaningful magnitude. Recorded as a category;
    ``warning_minutes`` is NULL. Treating it as zero would claim the
    detector arrived exactly on time.

``never_flagged``
    Not flagged at any horizon, full window included.
    ``warning_minutes`` is NULL.

So two different quantities are reported, both named:

* **conditional** -- median warning over ``flagged_in_grid`` rings only.
  This is "when we catch a ring in time to act, how much time do we
  get?". Its denominator is smaller than 120 and is always stated.
* **unconditional** -- the share of *all* rings with positive warning.
  This is "what fraction of rings did we actually get ahead of?", and it
  is the number that belongs next to a recall claim.

Quoting the conditional median alone would be the classic survivorship
error: it improves as the detector flags fewer rings.

Exit censoring is now resolved
------------------------------
Exits run 3.5 to 165.5 minutes after onset, and
:data:`ml.early.CUTOFF_MINUTES` now reaches 180, so **every** exit falls
inside the grid. Before that, 93 of 120 rings had their cash-out beyond
the last horizon, which made "caught before exit" nearly free for them.
No ring is right-censored on the exit side any more, and that is why the
warning times here are comparable across rings at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml.config import MlConfig
from ml.early import CUTOFF_MINUTES, exit_reached_at, ring_onsets
from ml.evaluate import SEEDS
from ml.features import load_tables
from ml.timeline import SCOPES, flagged_rings, load_oof

#: Percentiles reported for every distribution, so the shape is visible
#: rather than just a centre.
PERCENTILES: tuple[int, ...] = (0, 10, 25, 50, 75, 90, 100)

#: The only seed ``ml.early`` measures the full window for; see
#: :func:`first_flag_table`. Derived from ``SEEDS`` rather than written as
#: 42, so it follows the canonical seed if that ever changes.
FULL_WINDOW_SEED = SEEDS[0]

#: Ring censoring categories, in report order.
CATEGORIES: tuple[str, ...] = (
    "flagged_in_grid",
    "flagged_full_window_only",
    "never_flagged",
)


def finite_horizons() -> list[int]:
    """The measured finite horizons, ascending, from the one grid."""
    return sorted(m for m in CUTOFF_MINUTES if m is not None)


def first_flag_table(cfg: MlConfig, seed: int) -> pd.DataFrame:
    """Per (ring, scope) first-flag horizon and warning time, for one seed."""
    tables = load_tables(cfg)
    truth = tables["ground_truth"]
    onsets = ring_onsets(tables["transfers"], truth)
    exits = exit_reached_at(tables["transfers"], truth)
    exit_minutes = (exits - onsets).dt.total_seconds() / 60.0
    span = truth.loc[truth["ring_id"] != ""].groupby("ring_id")["ring_bank_count"].first()
    rings = sorted(onsets.index)
    horizons = finite_horizons()

    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        flagged_at = {
            minutes: flagged_rings(load_oof(cfg, minutes, scope, seed), truth)
            for minutes in finite_horizons()
        }
        # ``ml.early`` runs the full window for SEEDS[0] only -- it breaks
        # out of the seed loop there, because the full-window cutoff does
        # not depend on the resampled reference draw. So there is no
        # per-seed full-window file to read for seeds 43/44, and this
        # borrows the canonical seed's, exactly as ``metrics.json``
        # reports a single-seed mean for that horizon.
        #
        # The approximation is confined to the *descriptive* split between
        # ``flagged_full_window_only`` and ``never_flagged``.
        # ``warning_minutes`` and ``first_flag_T`` are computed from finite
        # horizons alone and are unaffected.
        flagged_at[None] = flagged_rings(load_oof(cfg, None, scope, FULL_WINDOW_SEED), truth)
        for ring in rings:
            first = next((m for m in horizons if ring in flagged_at[m]), None)
            if first is not None:
                category, warning = "flagged_in_grid", exit_minutes[ring] - first
            elif ring in flagged_at[None]:
                category, warning = "flagged_full_window_only", None
            else:
                category, warning = "never_flagged", None
            rows.append(
                {
                    "ring_id": ring,
                    "scope": scope,
                    "seed": seed,
                    "banks_spanned": int(span[ring]),
                    "first_flag_T": first,
                    "category": category,
                    "exit_minutes": float(exit_minutes[ring]),
                    "warning_minutes": warning,
                }
            )
    frame = pd.DataFrame(rows)
    frame["first_flag_T"] = frame["first_flag_T"].astype("Int64")
    frame["warning_minutes"] = frame["warning_minutes"].astype("Float64")
    return frame


def describe(values: pd.Series) -> dict[str, float | int]:
    """Percentile summary of one distribution, with its own n."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    out: dict[str, float | int] = {"n": int(clean.size)}
    if clean.size:
        for p in PERCENTILES:
            out[f"p{p}"] = float(np.percentile(clean, p))
        out["mean"] = float(clean.mean())
    return out


def summarise_scope(block: pd.DataFrame) -> dict[str, object]:
    """Conditional and unconditional lead-time summary for one group."""
    in_grid = block.loc[block["category"] == "flagged_in_grid"]
    counts = {c: int((block["category"] == c).sum()) for c in CATEGORIES}
    positive = int((pd.to_numeric(in_grid["warning_minutes"]) > 0).sum())
    return {
        "rings": int(len(block)),
        "category_counts": counts,
        # Conditional: over flagged_in_grid rings only. Denominator stated.
        "warning_minutes_conditional": describe(in_grid["warning_minutes"]),
        "first_flag_T_conditional": describe(in_grid["first_flag_T"]),
        # Unconditional: denominator is every ring in the group.
        "rings_with_positive_warning": positive,
        "share_with_positive_warning": positive / len(block) if len(block) else 0.0,
    }


def cross_bank_lead(frame: pd.DataFrame) -> dict[str, object]:
    """Minutes of head start cross-bank gives over single-bank, per ring.

    Defined only where **both** scopes flagged inside the grid; a ring
    that only one scope ever flags has an unbounded lead, not a numeric
    one, so those are counted separately instead of being imputed.
    """
    wide = frame.pivot_table(index="ring_id", columns="scope", values="first_flag_T", dropna=False)
    both = wide.dropna()
    lead = both["single_bank"] - both["cross_bank"]
    only_cb = int(wide["single_bank"].isna().sum() - wide.isna().all(axis=1).sum())
    only_sb = int(wide["cross_bank"].isna().sum() - wide.isna().all(axis=1).sum())
    return {
        "both_flagged_in_grid": int(len(both)),
        "lead_minutes": describe(lead),
        "rings_cross_bank_flagged_first": int((lead > 0).sum()),
        "rings_tied": int((lead == 0).sum()),
        "rings_single_bank_flagged_first": int((lead < 0).sum()),
        "rings_only_cross_bank_flagged_in_grid": only_cb,
        "rings_only_single_bank_flagged_in_grid": only_sb,
        "rings_neither_flagged_in_grid": int(wide.isna().all(axis=1).sum()),
    }


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    """Everything, overall and stratified by ring span."""
    out: dict[str, object] = {"by_scope": {}, "by_scope_and_span": {}}
    for scope in SCOPES:
        block = frame.loc[frame["scope"] == scope]
        out["by_scope"][scope] = summarise_scope(block)  # type: ignore[index]
        for span in (1, 2, 3):
            stratum = block.loc[block["banks_spanned"] == span]
            if len(stratum):
                out["by_scope_and_span"].setdefault(scope, {})[f"span{span}"] = (  # type: ignore[union-attr]
                    summarise_scope(stratum)
                )
    out["cross_bank_lead"] = cross_bank_lead(frame)
    out["cross_bank_lead_by_span"] = {
        f"span{span}": cross_bank_lead(frame.loc[frame["banks_spanned"] == span])
        for span in (1, 2, 3)
    }
    return out


def mean_over_seeds(per_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    """Average every numeric leaf across seeds, preserving the structure."""

    def walk(nodes: list[object]) -> object:
        first = nodes[0]
        if isinstance(first, dict):
            return {k: walk([n[k] for n in nodes]) for k in first}  # type: ignore[index]
        if isinstance(first, bool) or not isinstance(first, (int, float)):
            return first
        return float(np.mean([float(n) for n in nodes]))  # type: ignore[arg-type]

    return walk(list(per_seed.values()))  # type: ignore[return-value]


def exit_distribution(frame: pd.DataFrame) -> dict[str, object]:
    """Shape of the cash-out clock, overall and per ring span.

    Recorded here because it is the fact that makes or breaks every
    "caught before exit" claim, and it was not in any metrics file --
    every quote of it so far has been a number computed ad hoc in a
    shell. It is a property of the generated data, not of a seed, so it
    is taken from one scope of one seed and asserted identical across
    the rest by ``ml/test_leadtime.py``.
    """
    one = frame.loc[(frame["scope"] == SCOPES[0]) & (frame["seed"] == SEEDS[0])]
    per_ring = one.set_index("ring_id")["exit_minutes"]
    grid_end = max(finite_horizons())
    return {
        "rings": int(len(per_ring)),
        "minutes_after_onset": describe(per_ring),
        "by_span": {
            f"span{span}": describe(one.loc[one["banks_spanned"] == span, "exit_minutes"])
            for span in (1, 2, 3)
        },
        "grid_end_minutes": grid_end,
        "exits_inside_grid": int((per_ring <= grid_end).sum()),
        "exits_outside_grid": int((per_ring > grid_end).sum()),
    }


def main() -> None:
    """Measure lead time for every seed and write the metrics JSON."""
    cfg = MlConfig.from_env()
    per_seed: dict[int, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        frame = first_flag_table(cfg, seed)
        frames.append(frame)
        per_seed[seed] = summarise(frame)
        print(f"seed {seed}: {len(frame)} (ring, scope) rows")

    metrics = {
        "model": "lead_time",
        "generated_at": datetime.now(UTC).isoformat(),
        "derived_from": (
            "ml/experiments/early_detection/oof/ out-of-fold scores; no model is fitted"
        ),
        "horizon_grid": list(finite_horizons()),
        "exit_distribution": exit_distribution(pd.concat(frames, ignore_index=True)),
        "censoring": {
            "flagged_in_grid": "first flag at a finite horizon; warning_minutes is signed",
            "flagged_full_window_only": (
                "flagged only once the full window is observed, days after cash-out; "
                "warning_minutes is NULL, never zero"
            ),
            "never_flagged": "not flagged at any horizon; warning_minutes is NULL",
            "exit_side": ("no ring is right-censored: all 120 exits fall within T=180"),
        },
        "conditional_note": (
            "warning_minutes_conditional is over flagged_in_grid rings only and its n "
            "is stated; share_with_positive_warning is over every ring in the group"
        ),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "seed_mean": mean_over_seeds(per_seed),
    }
    out_dir = cfg.experiments_dir / "lead_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    pd.concat(frames, ignore_index=True).to_parquet(out_dir / "per_ring.parquet", index=False)
    print(f"wrote {out_dir / 'metrics.json'} and per_ring.parquet")
    _print_report(metrics)


def _print_report(metrics: dict[str, object]) -> None:
    """Human-readable summary of the seed-mean results."""
    mean: dict[str, object] = metrics["seed_mean"]  # type: ignore[assignment]
    print("\n=== warning minutes before exit (3-seed mean) ===")
    for scope in SCOPES:
        s = mean["by_scope"][scope]  # type: ignore[index]
        w, c = s["warning_minutes_conditional"], s["category_counts"]
        print(
            f"{scope:<12} in_grid={c['flagged_in_grid']:.1f} "
            f"full_only={c['flagged_full_window_only']:.1f} "
            f"never={c['never_flagged']:.1f} | "
            f"median={w.get('p50', float('nan')):+7.1f}m "
            f"p25={w.get('p25', float('nan')):+7.1f} p75={w.get('p75', float('nan')):+7.1f} | "
            f"positive_warning={s['share_with_positive_warning']:.3f}"
        )
    lead = mean["cross_bank_lead"]  # type: ignore[index]
    print(
        f"\ncross-bank lead over single-bank: n={lead['both_flagged_in_grid']:.1f} "
        f"median={lead['lead_minutes'].get('p50', float('nan')):+.1f}m "
        f"cb_first={lead['rings_cross_bank_flagged_first']:.1f} "
        f"tied={lead['rings_tied']:.1f} sb_first={lead['rings_single_bank_flagged_first']:.1f} "
        f"only_cb={lead['rings_only_cross_bank_flagged_in_grid']:.1f} "
        f"only_sb={lead['rings_only_single_bank_flagged_in_grid']:.1f}"
    )


if __name__ == "__main__":
    main()
