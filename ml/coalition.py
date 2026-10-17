"""What does one more bank joining actually buy? Measured, per coalition size.

The claim this exists to test
-----------------------------
"More banks joining makes detection better" is the network-effect
argument, and everywhere else in this repo it is supported by a *pair* of
numbers: the ``sb_`` scope (one bank alone) against the ``cb_`` scope (all
three joined). A pair is two points. It cannot show a curve, and a curve
is what the claim is about.

This module measures the curve. For every **coalition** of banks -- every
subset of the banks in the dataset -- it computes what that coalition and
only that coalition could detect, then averages over all coalitions of
the same size. The result is ring recall as a function of ``N``, the
number of banks that joined.

How a coalition's view is built
-------------------------------
By deletion, not by a new code path. A coalition sees the accounts held
at its own banks and the transfers between them; everything else does not
exist. So the coalition's view is produced by *filtering the tables* and
handing them to the existing :func:`ml.features.build_features`, whose
``cb_`` scope over a two-bank world is exactly "what those two banks see
once they join".

Nothing in ``ml/`` is modified to support this. The alternative was a
visible-bank-set parameter threaded through ``_scope_features``, which
would have put a second definition of scope next to the one the published
features use.

The metric
----------
**Ring recall: the share of the dataset's true rings for which the
coalition flags at least one pass-through mule or collector.** Same
definition as ``ml.early.catch_share_before_exit``'s ``ring_recall``,
with one deliberate difference: the flag here is the **velocity rule's**
hard decision, not the XGBoost scorer's 1%-FPR operating point. That
makes this cheap enough to sweep every coalition, and it makes the
numbers **not comparable** to the ones in ``docs/claims.md``, which are
the scorer's. They answer different questions and must not be read as
one series.

The denominator is **every ring in the dataset**, including rings in
which the coalition holds no account at all. That is unconditional on
purpose: a bank cannot act on a ring whose members are all somebody
else's customers, and hiding those rings from the denominator would
delete exactly the effect being measured. ``rings_reachable`` is reported
beside it so the conditional reading is recoverable.

The fourth bank
---------------
The dataset has three banks. A fourth is produced by **splitting one real
bank's customers in two** on a hash of the account id: the accounts, the
transfers, the devices and the documents are untouched, and only the
``bank_id`` label on half of one bank's accounts changes.

That is a real fourth bank in the only sense that matters to this
measurement -- a fourth independent visibility boundary over a real
transfer graph -- and it is emphatically **not** a fourth bank arriving
with new customers and new money. It redivides the same market. Two
consequences, both reported rather than left for a reader to work out:

* Splitting a bank makes every individual bank smaller, so the ``N = 1``
  baseline in the four-bank world is **worse** than in the three-bank
  world. The 1-to-4 gain is therefore larger than a real fourth bank
  would produce, and the bias runs in the flattering direction.
* For the same reason the four-bank world's ``N = 3`` point is not the
  repo's headline three-bank number. It is three banks over a different
  partition.

So both worlds are computed and both are published. The three-bank curve
is the real one; the four-bank curve is the extrapolation, and the gap
between their shared sizes is the price of the split, quantified instead
of asserted.

Running it
----------
::

    python -m ml.coalition

Needs ``simulator/out/``. No model call, no Google Cloud, no ``ml/out/``.
Twenty-two feature builds at roughly seven seconds each.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import pandas as pd

from ml.config import MlConfig
from ml.evaluate import PASSTHROUGH_MULE_ROLES
from ml.features import build_features, load_tables
from ml.publish import high_risk_components
from ml.rule import velocity_decision

#: The bank whose customers are split to make a fourth bank. Any of the
#: three would do; naming one keeps the run reproducible.
SPLIT_SOURCE_BANK = "BANK_C"

#: The label the split half receives.
SYNTHETIC_BANK = "BANK_D"

#: Accounts whose id hashes below this go to :data:`SYNTHETIC_BANK`. A
#: hash of the id rather than a random draw, so the split is identical on
#: every run without carrying a seed, and so it cannot correlate with
#: anything -- ring membership, role, open date -- that would make the
#: new bank unrepresentative of the one it came from.
SPLIT_FRACTION = 0.5

#: Tables filtered to a coalition's accounts. ``ground_truth`` is
#: deliberately absent: it is the label set the coalition is being scored
#: against, and a coalition does not get to see it.
COALITION_TABLES: tuple[str, ...] = (
    "accounts",
    "transfers",
    "devices",
    "kyc_submissions",
    "kyc_documents",
)


@dataclass(frozen=True)
class CoalitionResult:
    """What one specific coalition of banks detects on its own."""

    banks: tuple[str, ...]
    accounts_visible: int
    transfers_visible: int
    flagged_accounts: int
    rings_formed: int
    true_rings_total: int
    rings_reachable: int
    rings_caught: int
    #: The same coalition scored with the bank-spread clause DROPPED.
    #: ``ml/rule.py`` reports its ``single_bank`` variant the same way and
    #: for the same reason: ``distinct_counterparty_banks >= 2`` is
    #: uncomputable inside one bank, so the shipped rule fires on exactly
    #: zero accounts at N=1. Without this column the curve starts at 0.0
    #: and a reader cannot tell the rule declining to fire from an absence
    #: of signal. This is the generous reading, and it is the honest
    #: baseline to quote the gain against.
    flagged_accounts_no_spread: int = 0
    rings_caught_no_spread: int = 0

    @property
    def size(self) -> int:
        return len(self.banks)

    @property
    def ring_recall(self) -> float:
        """Caught over EVERY ring in the dataset. See the module docstring."""
        return self.rings_caught / self.true_rings_total if self.true_rings_total else 0.0

    @property
    def ring_recall_reachable(self) -> float:
        """Caught over the rings the coalition holds any account in."""
        return self.rings_caught / self.rings_reachable if self.rings_reachable else 0.0

    @property
    def ring_recall_no_spread(self) -> float:
        """Ring recall with the bank-spread clause dropped."""
        return self.rings_caught_no_spread / self.true_rings_total if self.true_rings_total else 0.0


def split_one_bank(
    accounts: pd.DataFrame,
    source: str = SPLIT_SOURCE_BANK,
    label: str = SYNTHETIC_BANK,
) -> pd.DataFrame:
    """Relabel half of ``source``'s accounts as ``label``.

    Returns a copy; the caller's frame is untouched. Only ``bank_id``
    changes -- no account is added, removed or re-keyed, and no transfer
    is touched, so the money in the four-bank world is the same money in
    the same places.

    The half is chosen by hashing the account id, so the same accounts
    move on every run and membership cannot correlate with role, ring or
    open date. Choosing by row order would have put the split on
    whatever the generator's emission order happens to encode.
    """
    out = accounts.copy()
    from_source = out["bank_id"] == source
    digest = out.loc[from_source, "hashed_account_id"].map(_unit_hash)
    moves = (digest < SPLIT_FRACTION).reindex(out.index, fill_value=False)
    out.loc[from_source & moves, "bank_id"] = label
    return out


def _unit_hash(account_id: str) -> float:
    """The account id as a stable float in [0, 1).

    SHA-256 rather than Python's ``hash``, which is salted per process:
    with ``hash`` the split would differ between runs and between the
    published curve and anything recomputed from it.
    """
    return int(hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:8], 16) / 0x1_0000_0000


def coalition_tables(
    tables: dict[str, pd.DataFrame], banks: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    """The five tables as the coalition sees them: its accounts, nothing else.

    A transfer survives only when **both** endpoints are held inside the
    coalition. One-legged transfers are not half-visible -- a bank sees
    its own side of a payment to an outside account, but the coalition
    cannot build a counterparty feature from an account it does not hold,
    and pretending otherwise would let the coalition read structure it
    has no access to.
    """
    accounts = tables["accounts"]
    inside = accounts.loc[accounts["bank_id"].isin(banks)]
    ids = set(inside["hashed_account_id"])

    transfers = tables["transfers"]
    both_ends = transfers["from_hashed_account_id"].isin(ids) & transfers[
        "to_hashed_account_id"
    ].isin(ids)

    out: dict[str, pd.DataFrame] = {
        "accounts": inside,
        "transfers": transfers.loc[both_ends],
    }
    for name in COALITION_TABLES[2:]:
        frame = tables[name]
        out[name] = frame.loc[frame["hashed_account_id"].isin(ids)]
    return out


def _true_rings(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """The pass-through mules and collectors of every true ring.

    ``PASSTHROUGH_MULE_ROLES`` rather than ``COMPLICIT_ROLES``: sources
    are complicit but the velocity rule cannot fire on them (median
    in-degree 0, see ``ml/stage2.py``), so scoring the rule on them would
    measure a mechanism it does not have.
    """
    rows = ground_truth.loc[
        ground_truth["role"].isin(PASSTHROUGH_MULE_ROLES)
        & ground_truth["ring_id"].notna()
    ]
    return rows[["hashed_account_id", "bank_id", "ring_id"]]


def evaluate_coalition(
    tables: dict[str, pd.DataFrame],
    banks: tuple[str, ...],
    truth: pd.DataFrame,
    cfg: MlConfig,
) -> CoalitionResult:
    """Run the real rule over one coalition's view and score its ring recall."""
    view = coalition_tables(tables, banks)
    true_rings_total = int(truth["ring_id"].nunique())

    # A coalition holding no account in a ring cannot flag it. Counted
    # rather than excluded: it is the numerator's ceiling, and the gap
    # between it and true_rings_total is most of what joining buys.
    reachable = truth.loc[truth["bank_id"].isin(banks)]
    rings_reachable = int(reachable["ring_id"].nunique())

    if view["transfers"].empty:  # pragma: no cover - needs a bank with no internal edges
        return CoalitionResult(
            banks=banks,
            accounts_visible=len(view["accounts"]),
            transfers_visible=0,
            flagged_accounts=0,
            rings_formed=0,
            true_rings_total=true_rings_total,
            rings_reachable=rings_reachable,
            rings_caught=0,
        )

    # One feature build, two decisions. The generous variant costs
    # nothing beyond a second pass over the same frame, and building the
    # features twice to get it would have doubled the sweep.
    features = build_features(cfg=cfg, tables=view)
    flagged = velocity_decision(features, "cb_", require_bank_spread=True)
    flagged_ids = pd.Index(features.loc[flagged, "hashed_account_id"])
    generous = velocity_decision(features, "cb_", require_bank_spread=False)
    generous_ids = set(features.loc[generous, "hashed_account_id"])

    cutoff = pd.Timestamp(view["transfers"]["timestamp"].max())
    rings = high_risk_components(view["transfers"], flagged_ids, cutoff)

    # Same rule as ml.early: a ring counts as caught when any of its
    # pass-through mules or collectors is flagged. Ring FORMATION is
    # reported separately -- an analyst gets an alert either way, and
    # conflating the two would credit the coalition with a ring it only
    # produced a singleton alert on.
    caught = truth.loc[truth["hashed_account_id"].isin(set(flagged_ids)), "ring_id"]
    caught_generous = truth.loc[
        truth["hashed_account_id"].isin(generous_ids), "ring_id"
    ]
    return CoalitionResult(
        banks=banks,
        accounts_visible=len(view["accounts"]),
        transfers_visible=len(view["transfers"]),
        flagged_accounts=len(flagged_ids),
        rings_formed=len(rings),
        true_rings_total=true_rings_total,
        rings_reachable=rings_reachable,
        rings_caught=int(caught.nunique()),
        flagged_accounts_no_spread=len(generous_ids),
        rings_caught_no_spread=int(caught_generous.nunique()),
    )


def sweep(
    tables: dict[str, pd.DataFrame], cfg: MlConfig
) -> list[CoalitionResult]:
    """Every coalition of every size, in one world.

    Every subset, not a sample: with four banks there are fifteen, and a
    mean over "some of the pairs" is a number nobody can reproduce.
    """
    truth = _true_rings(tables["ground_truth"])
    banks = tuple(sorted(set(tables["accounts"]["bank_id"])))
    results: list[CoalitionResult] = []
    for size in range(1, len(banks) + 1):
        for coalition in combinations(banks, size):
            results.append(evaluate_coalition(tables, coalition, truth, cfg))
            print(
                f"  N={size} {'+'.join(coalition):<32} "
                f"recall={results[-1].ring_recall:.4f} "
                f"no_spread={results[-1].ring_recall_no_spread:.4f} "
                f"reachable={results[-1].rings_reachable} "
                f"flagged={results[-1].flagged_accounts}"
            )
    return results


def by_size(results: list[CoalitionResult]) -> list[dict[str, Any]]:
    """Mean over all coalitions of each size, with the spread kept.

    ``min`` and ``max`` ride along because the mean alone hides the thing
    a sceptic will ask about: whether one lucky pairing carries the
    average. With three banks the coalitions of size 2 are genuinely
    different views of the same market, and the range says how different.
    """
    sizes = sorted({r.size for r in results})
    out: list[dict[str, Any]] = []
    for size in sizes:
        group = [r for r in results if r.size == size]
        recalls = [r.ring_recall for r in group]
        out.append(
            {
                "banks_joined": size,
                "coalitions": len(group),
                "ring_recall": sum(recalls) / len(group),
                "ring_recall_min": min(recalls),
                "ring_recall_max": max(recalls),
                "ring_recall_no_spread": sum(r.ring_recall_no_spread for r in group)
                / len(group),
                "ring_recall_reachable": sum(r.ring_recall_reachable for r in group)
                / len(group),
                "rings_caught": sum(r.rings_caught for r in group) / len(group),
                "rings_reachable": sum(r.rings_reachable for r in group) / len(group),
                "rings_formed": sum(r.rings_formed for r in group) / len(group),
                "flagged_accounts": sum(r.flagged_accounts for r in group) / len(group),
                "accounts_visible": sum(r.accounts_visible for r in group) / len(group),
            }
        )
    return out


def build(cfg: MlConfig | None = None) -> dict[str, Any]:
    """Both worlds, both curves, and the caveats that make them readable."""
    cfg = cfg or MlConfig.from_env()
    tables = load_tables(cfg)
    true_rings_total = int(_true_rings(tables["ground_truth"])["ring_id"].nunique())

    print("three-bank world (the real partition)")
    three = sweep(tables, cfg)

    print("")
    print(f"four-bank world ({SPLIT_SOURCE_BANK} split, half relabelled {SYNTHETIC_BANK})")
    four_tables = dict(tables)
    four_tables["accounts"] = split_one_bank(tables["accounts"])
    four_tables["ground_truth"] = _reassign_truth_banks(
        tables["ground_truth"], four_tables["accounts"]
    )
    four = sweep(four_tables, cfg)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "metric": "ring_recall",
        "metric_definition": (
            "Share of the dataset's true rings for which the coalition flags at "
            "least one pass-through mule or collector, using the VELOCITY RULE's "
            "hard decision (ml.rule.velocity_decision over the coalition's joined "
            "view). Denominator is every ring in the dataset, including rings the "
            "coalition holds no account in."
        ),
        "two_variants": (
            "ring_recall is the SHIPPED rule, whose bank-spread clause "
            "(distinct_counterparty_banks >= 2) is uncomputable inside one bank -- so "
            "it fires on exactly zero accounts at N=1, and that point is the rule "
            "declining to fire, not an absence of signal. ring_recall_no_spread is the "
            "same rule with that clause dropped: the generous reading, and the honest "
            "baseline to quote a gain against. ml/rule.py reports its own single_bank "
            "variant the same way and for the same reason."
        ),
        "not_comparable_to_claims": (
            "docs/claims.md quotes ring recall at the XGBoost scorer's 1%-FPR "
            "operating point. This is the velocity rule's hard decision instead, "
            "which is what makes sweeping every coalition cheap. The two answer "
            "different questions and must not be read as one series."
        ),
        "true_rings_total": true_rings_total,
        "worlds": {
            "three_bank": {
                "label": "3 real banks",
                "is_real_partition": True,
                "note": (
                    "The dataset as generated. Every point here is measured on "
                    "the real partition."
                ),
                "by_size": by_size(three),
                "coalitions": [_as_row(r) for r in three],
            },
            "four_bank": {
                "label": f"4 banks ({SPLIT_SOURCE_BANK} split)",
                "is_real_partition": False,
                "note": (
                    f"{SYNTHETIC_BANK} is half of {SPLIT_SOURCE_BANK}'s customers, "
                    "chosen by hashing the account id. Every account, transfer, "
                    "device and document is real and unchanged; only the bank_id "
                    "label moved. This redivides the same market among four banks "
                    "rather than adding a fourth bank's customers, so the N=1 "
                    "baseline is worse than the real world's and the 1-to-4 gain "
                    "is larger than a real fourth bank would give. Its N=3 point "
                    "is three banks over a different partition, not the repo's "
                    "headline three-bank number."
                ),
                "by_size": by_size(four),
                "coalitions": [_as_row(r) for r in four],
            },
        },
    }


def _reassign_truth_banks(
    ground_truth: pd.DataFrame, accounts: pd.DataFrame
) -> pd.DataFrame:
    """Carry the split's new ``bank_id`` into ground truth.

    ``ground_truth`` holds its own ``bank_id`` column, and
    ``rings_reachable`` is computed from it. Left unmapped, a ring's
    members would still be recorded at the bank they were split out of,
    and every four-bank coalition's reachability would be wrong.
    """
    bank = accounts.set_index("hashed_account_id")["bank_id"]
    out = ground_truth.copy()
    out["bank_id"] = out["hashed_account_id"].map(bank).fillna(out["bank_id"])
    return out


def _as_row(result: CoalitionResult) -> dict[str, Any]:
    return {
        "banks": list(result.banks),
        "banks_joined": result.size,
        "accounts_visible": result.accounts_visible,
        "transfers_visible": result.transfers_visible,
        "flagged_accounts": result.flagged_accounts,
        "rings_formed": result.rings_formed,
        "rings_caught": result.rings_caught,
        "rings_reachable": result.rings_reachable,
        "true_rings_total": result.true_rings_total,
        "ring_recall": result.ring_recall,
        "ring_recall_no_spread": result.ring_recall_no_spread,
        "ring_recall_reachable": result.ring_recall_reachable,
        "flagged_accounts_no_spread": result.flagged_accounts_no_spread,
        "rings_caught_no_spread": result.rings_caught_no_spread,
    }


def _print_curve(payload: dict[str, Any]) -> None:
    print("")
    print("RING RECALL BY NUMBER OF BANKS JOINED (velocity rule, not the scorer)")
    for world in payload["worlds"].values():
        tag = "measured" if world["is_real_partition"] else "split-derived"
        print("")
        print(f"  {world['label']}  [{tag}]")
        for row in world["by_size"]:
            spread = (
                f"  (range {row['ring_recall_min']:.4f}-{row['ring_recall_max']:.4f}"
                f" over {row['coalitions']} coalitions)"
                if row["coalitions"] > 1
                else ""
            )
            print(
                f"    N={row['banks_joined']}  recall={row['ring_recall']:.4f}"
                f"  no_spread={row['ring_recall_no_spread']:.4f}"
                f"  rings_caught={row['rings_caught']:.1f}"
                f"/{payload['true_rings_total']}"
                f"  reachable={row['rings_reachable']:.1f}{spread}"
            )


def main() -> None:
    """Sweep both worlds and write the curve beside the other experiments."""
    cfg = MlConfig.from_env()
    payload = build(cfg)
    _print_curve(payload)

    out_dir = cfg.experiments_dir / "coalition"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    print("")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
