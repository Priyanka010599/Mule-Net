"""The shipped mule rule: **velocity decides, identity is evidence.**

The design this module used to hold
-----------------------------------
Two branches OR-ed together — an *identity* branch (fast hop + fan-in +
bank spread + a shared device or reused ID at another bank) and a
*velocity* branch (fast hop + flow band + a fan-in floor above the 90th
percentile of non-mules + bank spread). The velocity branch was added
because the identity branch caught **0 of 228** ``fast_passthrough_only``
positives: a rule that can only fire on reused credentials is blind to
the attack pattern that deliberately reuses none.

Why it changed
--------------
Once both branches were measured against each other, the identity branch
turned out to be **strictly dominated**: every true positive it found,
velocity had already found, and it contributed 146 false positives of
its own. ``branch_overlap`` below is the measurement, and it is still
reported, because the claim "we removed a clause" is only credible next
to the number that justified it. An OR-ed clause that adds no true
positives and only false ones makes the rule worse at every operating
point. It was removed from the firing condition.

Removed from the *firing condition* — not thrown away. "This account
shares a device fingerprint with an account at a different bank" is
useless as a detector here and is exactly what a compliance analyst
needs in an STR narrative. So identity reuse is now **attached to each
flagged account as evidence** (``ml/evidence.py``), and what this module
reports about it is the *attachment rate*: of the accounts the detector
hands an analyst, what share arrive with each citable fact. That is the
number the UI badge and the report-writing agent depend on, and it is a
property of the alert queue, not a detection metric.

The decision, in full
---------------------
::

    flagged =
        passthrough_speed in [0, VELOCITY_MAX_SECONDS]
        and passthrough_ratio in VELOCITY_RATIO_RANGE
        and in_degree  >= VELOCITY_MIN_IN_DEGREE
        and out_degree >= VELOCITY_MIN_OUT_DEGREE
        and distinct_counterparty_banks >= MIN_BANKS

Nothing else. No identity term appears anywhere in it.

Where these thresholds come from, and what the numbers here mean
----------------------------------------------------------------
They are the **modal choice across all 15 training folds** of
``ml/tune.py`` (5 folds x 3 seeds), which grid-searches inside each fold
and scores on that fold's held-out rings. All 15 folds independently
chose ``min_in_degree = 16``; 14 of 15 chose the same value for every
other field. That stability is itself the finding: the fan-in floor is a
property of the attack, not a number fitted to this sample.

**The precision printed by this module is in-sample** — it re-scores the
same 30,000 accounts the thresholds were chosen on. The honest estimate
lives in ``ml/experiments/rule_tuning/metrics.json`` and the two are
within 0.0002 of each other, which is why this module is still worth
running: it reports per-pattern and per-span breakdowns the pooled
held-out run does not, and the gap is small enough that those breakdowns
are not misleading. The experiment is named ``velocity_rule_in_sample``
so the comparison table cannot quietly present it as held-out.

Why the rule still requires the cross-bank join
-----------------------------------------------
``distinct_counterparty_banks >= 2`` is **uncomputable at a single
bank**: in the ``sb_`` scope that feature is capped at 1 by
construction, so the literal single-bank transcription fires on exactly
zero accounts. The guard is not doing the work alone either — dropped
entirely (the most generous single-bank reading available), the rule
still collapses, and ``ml/tune.py`` confirms it under identical
per-fold tuning on both sides.

Three variants, identically shaped
----------------------------------
``cross_bank``
    The rule as written, over ``cb_`` features.
``single_bank``
    The same shape over ``sb_`` features with the bank-spread clause
    dropped, because a single bank sees one counterparty bank and the
    clause would make the rule vacuous.
``single_bank_literal``
    That vacuous transcription run anyway, so the concession above is
    visible rather than quietly made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml.config import MlConfig
from ml.evaluate import SPAN_STRATA, attach_labels
from ml.evidence import (
    CROSS_BANK_COLUMNS,
    SAME_BANK_COLUMNS,
    attachment_rates,
    evidence_flags,
    redundancy_check,
)
from ml.features import full_window_cutoff, identity_links, load_tables
from ml.metrics import binary_metrics

#: Upper bound on a "fast" pass-through hop, in seconds. Retired
#: identity branch only; kept so ``branch_overlap`` can still be run.
FAST_MAX_SECONDS = 1800.0

#: Minimum in- and out-degree for the retired identity branch.
MIN_DEGREE = 3

#: Minimum distinct counterparty banks. In the ``sb_`` scope this feature
#: is capped at 1, so this clause is what makes the rule *require* the
#: cross-bank join rather than merely benefit from it.
MIN_BANKS = 2

#: Inclusive bounds on ``amount_out / amount_in`` for the retired branch.
RATIO_RANGE: tuple[float, float] = (0.6, 1.3)

#: **The shipped decision thresholds.** Modal choice across the 15
#: training folds of ``ml/tune.py``; see the module docstring.
VELOCITY_MAX_SECONDS = 3600.0
VELOCITY_RATIO_RANGE: tuple[float, float] = (0.1, 2.0)
VELOCITY_MIN_IN_DEGREE = 16
VELOCITY_MIN_OUT_DEGREE = 4

#: The branch that makes the detection decision. Identity never does.
DECISION_BRANCH = "velocity_only"

#: Branches reported. ``velocity_only`` ships; the other two are the
#: record of why the identity branch was taken out of the firing
#: condition, not alternatives on offer.
BRANCHES: tuple[str, ...] = ("identity_only", "velocity_only", "combined")


def identity_partner_banks(
    tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> pd.DataFrame:
    """Per account, whether it has a same-bank / other-bank identity partner.

    Device and ID-number reuse only, matching the retired identity
    branch's definition, so ``branch_overlap`` keeps measuring the clause
    that was actually removed. The richer three-signal evidence view
    lives in ``ml/evidence.py``.
    """
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    index = pd.Index(tables["accounts"]["hashed_account_id"], name="hashed_account_id")
    same = pd.Series(False, index=index)
    other = pd.Series(False, index=index)

    for key, links in identity_links(tables).items():
        pairs = (
            links.groupby(["hashed_account_id", key], sort=False)["observed_at"].min().reset_index()
        )
        joined = pairs.merge(pairs, on=key, suffixes=("", "_other"))
        joined = joined.loc[joined["hashed_account_id"] != joined["hashed_account_id_other"]]
        own_cutoff = joined["hashed_account_id"].map(cutoff)
        joined = joined.loc[
            (joined["observed_at"] <= own_cutoff) & (joined["observed_at_other"] <= own_cutoff)
        ]
        cross = joined["hashed_account_id"].map(bank) != joined[
            "hashed_account_id_other"
        ].map(bank)
        same |= index.isin(joined.loc[~cross, "hashed_account_id"])
        other |= index.isin(joined.loc[cross, "hashed_account_id"])

    return pd.DataFrame({"same_bank_identity_partner": same, "other_bank_identity_partner": other})


def velocity_decision(
    frame: pd.DataFrame, prefix: str, require_bank_spread: bool = True
) -> np.ndarray:
    """The shipped firing condition. Reads five features; no identity term."""
    speed = frame[f"{prefix}passthrough_speed_seconds"].to_numpy(dtype="float64")
    ratio = frame[f"{prefix}passthrough_ratio"].to_numpy(dtype="float64")
    in_degree = frame[f"{prefix}in_degree"].to_numpy()
    out_degree = frame[f"{prefix}out_degree"].to_numpy()
    banks = frame[f"{prefix}distinct_counterparty_banks"].to_numpy()
    spread = banks >= MIN_BANKS if require_bank_spread else np.ones(len(frame), dtype=bool)
    return (
        (speed >= 0.0)
        & (speed <= VELOCITY_MAX_SECONDS)
        & (ratio >= VELOCITY_RATIO_RANGE[0])
        & (ratio <= VELOCITY_RATIO_RANGE[1])
        & (in_degree >= VELOCITY_MIN_IN_DEGREE)
        & (out_degree >= VELOCITY_MIN_OUT_DEGREE)
        & spread
    )


def rule_branches(
    frame: pd.DataFrame, prefix: str, xbid: np.ndarray, require_bank_spread: bool = True
) -> dict[str, np.ndarray]:
    """The shipped decision plus the retired branch, for the overlap record."""
    speed = frame[f"{prefix}passthrough_speed_seconds"].to_numpy(dtype="float64")
    ratio = frame[f"{prefix}passthrough_ratio"].to_numpy(dtype="float64")
    in_degree = frame[f"{prefix}in_degree"].to_numpy()
    out_degree = frame[f"{prefix}out_degree"].to_numpy()
    banks = frame[f"{prefix}distinct_counterparty_banks"].to_numpy()
    spread = banks >= MIN_BANKS if require_bank_spread else np.ones(len(frame), dtype=bool)

    fast = (speed >= 0.0) & (speed <= FAST_MAX_SECONDS)
    struct = (in_degree >= MIN_DEGREE) & (out_degree >= MIN_DEGREE) & spread
    flow = (ratio >= RATIO_RANGE[0]) & (ratio <= RATIO_RANGE[1])
    identity_only = fast & struct & flow & xbid
    velocity_only = velocity_decision(frame, prefix, require_bank_spread)
    return {
        "identity_only": identity_only,
        "velocity_only": velocity_only,
        "combined": identity_only | velocity_only,
    }


def _report(frame: pd.DataFrame, flag: np.ndarray) -> dict[str, object]:
    """Precision/recall/FPR overall, then broken out by pattern and by span.

    Per-pattern *recall* is exact. Per-pattern *precision* is not a
    well-defined quantity — a false positive is a non-mule and carries no
    ``pattern_type`` — so what is reported is the precision this rule
    would show if that pattern were the only target: the pattern's true
    positives against **all** of the rule's false positives. It is
    therefore a lower bound per pattern and the four values do not
    partition anything; the overall precision is the honest headline.
    """
    y = frame["is_passthrough_mule"].to_numpy().astype(bool)
    out: dict[str, object] = binary_metrics(y, flag)
    false_positives = int((flag & ~y).sum())

    def _slice(mask: np.ndarray) -> dict[str, float]:
        positives = int(mask.sum())
        caught = int((mask & flag).sum())
        return {
            "positives": positives,
            "caught": caught,
            "recall": caught / positives if positives else float("nan"),
            "precision_vs_all_fp": caught / (caught + false_positives)
            if caught + false_positives
            else 0.0,
        }

    pattern = frame["pattern_type"].to_numpy()
    span = frame["ring_bank_count"].to_numpy()
    out["per_pattern"] = {str(p): _slice(y & (pattern == p)) for p in sorted(set(pattern[y]))}
    out["per_span"] = {f"span{s}": _slice(y & (span == s)) for s in SPAN_STRATA}
    return out


def branch_overlap(frame: pd.DataFrame, clauses: dict[str, np.ndarray]) -> dict[str, int]:
    """The measurement that retired the identity branch.

    ``tp_added_by_identity == 0`` while ``fp_added_by_identity`` is large
    means the OR was strictly worse than velocity alone: the clause cost
    precision and bought nothing. This stays in the report because the
    removal should be auditable, not asserted.
    """
    y = frame["is_passthrough_mule"].to_numpy().astype(bool)
    ident = clauses["identity_only"]
    vel = clauses["velocity_only"]
    return {
        "tp_identity_only": int((ident & y).sum()),
        "tp_velocity_only": int((vel & y).sum()),
        "tp_both": int((ident & vel & y).sum()),
        "tp_added_by_identity": int((ident & ~vel & y).sum()),
        "fp_added_by_identity": int((ident & ~vel & ~y).sum()),
        "tp_added_by_velocity": int((vel & ~ident & y).sum()),
        "fp_added_by_velocity": int((vel & ~ident & ~y).sum()),
    }


def evidence_report(
    frame: pd.DataFrame, evidence: pd.DataFrame, flag: np.ndarray
) -> dict[str, object]:
    """Evidence attachment rates on the accounts the detector flagged.

    Split by whether the flagged account is a true or false positive:
    if evidence attaches to false positives at the same rate as to true
    ones it is decoration, and an analyst should be told that rather than
    shown a badge that means nothing.
    """
    ids = frame["hashed_account_id"]
    y = frame["is_passthrough_mule"].to_numpy().astype(bool)
    return {
        "all_flagged": attachment_rates(evidence, pd.Series(flag, index=ids.to_numpy())),
        "true_positives": attachment_rates(evidence, pd.Series(flag & y, index=ids.to_numpy())),
        "false_positives": attachment_rates(evidence, pd.Series(flag & ~y, index=ids.to_numpy())),
    }


def _print_table(results: dict[str, dict[str, dict[str, object]]]) -> None:
    print("")
    print("DETECTION (the decision is velocity_only; other rows are the retired design)")
    header = f"{'variant':<22}{'branch':<16}{'precision':>11}{'recall':>9}{'FPR':>10}{'flagged':>9}"
    print(header)
    for variant, branches in results.items():
        for branch in BRANCHES:
            r = branches[branch]
            ship = "  <- SHIPPED" if branch == DECISION_BRANCH and variant == "cross_bank" else ""
            print(
                f"{variant:<22}{branch:<16}{r['precision']:>11.4f}{r['recall']:>9.4f}"
                f"{r['fpr']:>10.5f}{r['flagged']:>9d}{ship}"
            )

    for axis, title in (("per_pattern", "pattern"), ("per_span", "ring bank-span")):
        print("")
        print(f"cross-bank velocity-rule recall by {title}")
        keys = sorted(results["cross_bank"][DECISION_BRANCH][axis])  # type: ignore[index]
        for key in keys:
            cb = results["cross_bank"][DECISION_BRANCH][axis][key]  # type: ignore[index]
            sb = results["single_bank"][DECISION_BRANCH][axis][key]  # type: ignore[index]
            print(
                f"  {key:<24} n={cb['positives']:<4}"
                f" cross_bank={cb['recall']:.4f}"
                f" (prec_vs_all_fp={cb['precision_vs_all_fp']:.4f})"
                f" | single_bank={sb['recall']:.4f}"
            )


def _print_evidence(evidence_by_variant: dict[str, dict[str, object]]) -> None:
    print("")
    print("EVIDENCE ATTACHMENT on flagged accounts (never part of the decision)")
    for variant, report in evidence_by_variant.items():
        columns = CROSS_BANK_COLUMNS if variant == "cross_bank" else SAME_BANK_COLUMNS
        for bucket in ("all_flagged", "true_positives", "false_positives"):
            rates: dict[str, float] = report[bucket]  # type: ignore[assignment]
            cells = " ".join(f"{c.split('_bank_')[1]}={rates[c]:.3f}" for c in columns)
            print(
                f"  {variant:<20}{bucket:<17} n={int(rates['flagged']):<5} {cells}"
                f" any_cross={rates['any_cross_bank_evidence']:.3f}"
            )


def main() -> None:
    """Run the velocity rule in every scope and write its metrics JSON."""
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    cutoff = full_window_cutoff(tables)
    partners = identity_partner_banks(tables, cutoff).reindex(frame["hashed_account_id"])
    other_bank = partners["other_bank_identity_partner"].to_numpy()
    same_bank = partners["same_bank_identity_partner"].to_numpy()
    evidence = evidence_flags(tables, cutoff)

    variants = {
        "cross_bank": rule_branches(frame, "cb_", other_bank, require_bank_spread=True),
        "single_bank": rule_branches(frame, "sb_", same_bank, require_bank_spread=False),
        "single_bank_literal": rule_branches(frame, "sb_", same_bank, require_bank_spread=True),
    }
    results = {
        name: {branch: _report(frame, clauses[branch]) for branch in BRANCHES}
        for name, clauses in variants.items()
    }
    _print_table(results)

    overlap = {name: branch_overlap(frame, clauses) for name, clauses in variants.items()}
    print("")
    print("why the identity branch was removed from the firing condition")
    for name, o in overlap.items():
        print(
            f"  {name:<22} identity adds tp={o['tp_added_by_identity']:<4}"
            f" fp={o['fp_added_by_identity']:<4} | velocity adds"
            f" tp={o['tp_added_by_velocity']:<4} fp={o['fp_added_by_velocity']:<4}"
        )

    evidence_by_variant = {
        name: evidence_report(frame, evidence, clauses[DECISION_BRANCH])
        for name, clauses in variants.items()
        if name != "single_bank_literal"
    }
    _print_evidence(evidence_by_variant)
    redundancy = redundancy_check(evidence)
    print("")
    print("evidence redundancy (Jaccard between cross-bank evidence types)")
    for pair, value in redundancy.items():
        note = "  <- ARTIFACT: one fact, two labels" if value > 0.99 else ""
        print(f"  {pair:<66}{value:.3f}{note}")

    metrics = {
        "model": "velocity_rule_in_sample",
        "generated_at": datetime.now(UTC).isoformat(),
        "design": (
            "velocity decides, identity is evidence: the firing condition "
            "reads five velocity/structure features and no identity signal; "
            "identity reuse is attached to flagged accounts for the analyst "
            "and the report-writing agent"
        ),
        "decision_branch": DECISION_BRANCH,
        "rule": {
            "velocity_decision": {
                "max_passthrough_speed_seconds": VELOCITY_MAX_SECONDS,
                "passthrough_ratio_range": list(VELOCITY_RATIO_RANGE),
                "min_in_degree": VELOCITY_MIN_IN_DEGREE,
                "min_out_degree": VELOCITY_MIN_OUT_DEGREE,
                "min_distinct_counterparty_banks": MIN_BANKS,
                "requires_cross_bank_join": (
                    "min_distinct_counterparty_banks=2 is uncomputable in the sb_ scope, "
                    "where that feature is capped at 1; the literal single-bank "
                    "transcription therefore fires on zero accounts"
                ),
                "threshold_provenance": (
                    "modal choice across the 15 training folds of ml/tune.py "
                    "(5 folds x 3 seeds), each tuned on training rings only"
                ),
            },
            "retired_identity_branch": {
                "fast_max_seconds": FAST_MAX_SECONDS,
                "min_degree": MIN_DEGREE,
                "passthrough_ratio_range": list(RATIO_RANGE),
                "why_retired": (
                    "strictly dominated: 0 true positives velocity had not already "
                    "caught, and false positives of its own (see branch_overlap)"
                ),
            },
        },
        "metrics_are_in_sample": (
            "thresholds were chosen on this dataset, so precision/recall here "
            "are resubstitution estimates; the held-out estimate is in "
            "experiments/rule_tuning/metrics.json and differs by <0.001"
        ),
        "per_pattern_precision_convention": (
            "true positives of that pattern against ALL of the rule's false "
            "positives; a lower bound, since false positives carry no pattern"
        ),
        "total_positives": int(frame["is_passthrough_mule"].sum()),
        "total_accounts": len(frame),
        "variants": results,
        "branch_overlap": overlap,
        "evidence_attachment": evidence_by_variant,
        "evidence_redundancy": redundancy,
        "evidence_redundancy_verdict": (
            "cross_bank_reused_photo and cross_bank_reused_id_number have "
            "Jaccard 1.000: the simulator attaches a reused photo_embedding_ref "
            "to exactly the documents that reuse hashed_id_number. Verdict: "
            "(b) artifact of the generator, not leakage. Cite identity reuse "
            "once in an STR; do not render these as independent corroborations."
        ),
        # Shaped for ml/compare.py: the rule is deterministic, so std is 0.
        "summary": {
            name: {
                f"{k}_{stat}": (r[DECISION_BRANCH][k] if stat == "mean" else 0.0)
                for k in ("precision", "recall", "fpr")
                for stat in ("mean", "std")
            }
            for name, r in results.items()
        },
        "runs": [{"scope": name, "seed": 0} for name in results],
    }
    out_dir = cfg.experiments_dir / "compound_rule"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    print("")
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
