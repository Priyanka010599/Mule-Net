"""Stage 2: identity propagation out of an **already-detected** ring.

Why a second stage exists
-------------------------
Stage 1 (``ml/rule.py`` + ``ml/publish.high_risk_components``) is a
pass-through detector: velocity decides, and a flagged account only
joins a ring when the counterparty it forwards to was *also* flagged.
That leaves two measured holes:

* 87.8% account-level recall collapses to **53 / 120** true rings
  touched, because the clustering step drops flagged singletons;
* **0 of 187** ``source`` accounts are reachable at all. Sources have
  median in-degree 0 — they originate ring funds rather than forwarding
  them — so the rule's fan-in clause can never fire on them. No
  threshold on a pass-through rule fixes that; it is a different
  mechanism, and this module is that mechanism.

The design constraint, kept intact
----------------------------------
Identity reuse is **not** a firing condition and does not become one
here. Measured on its own it was strictly dominated as a detector (0
extra true positives, +146 false positives, see ``ml/rule.py``), and the
legitimate cover is dense by construction: of the 4,773 accounts
carrying any identity edge in this dataset, **4,102 are not complicit**
— families share devices, dual-bank customers reuse an id across banks,
and same-bank document reuse is ordinary.

What changes here is not the evidence, it is the **conditioning**. An
identity edge is evidence of a *link*, never of guilt. Conditioned on
one endpoint already sitting inside a ring that velocity and graph
structure independently detected, following that link is a legitimate
expansion path. So:

    stage-2 never creates a ring. It only grows one stage 1 already
    found.

That is the whole safety property, and ``ml/test_publish.py`` asserts
it: with the stage-1 ring set empty, stage-2 returns nothing.

One hop, and why not more
-------------------------
Propagation is a **single hop** from ring members. Two hops was measured
and is bit-identical to one (401 members either way): identity
components in this dataset close after one step, so iterating buys
nothing and only adds a knob that could later be turned up into an
unconditioned detector. ``PROPAGATION_HOPS`` is kept explicit so the
measurement is reproducible, not so it is tuned.

Point-in-time discipline
------------------------
Identical to ``ml/features.py`` and ``ml/evidence.py``: a link counts
only when **both** sightings land at or before the observing account's
cutoff. An analyst cannot follow a fingerprint sighting that has not
happened yet, so neither can this.

The result, in one line
-----------------------
Stage 2 recovers **58 of 187 sources** and 32 further pass-through
mules for **13 non-complicit accounts** — ring contamination moves
11.07% -> 11.47%, and rings whose every member is recovered go from
**0 / 120 to 34 / 120**. It recovers **zero** rings that stage 1 missed
entirely, by construction. Numbers and the caveats live in
``main`` and the written ``metrics.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from ml.config import MlConfig
from ml.evaluate import COMPLICIT_ROLES, PASSTHROUGH_MULE_ROLES, attach_labels
from ml.evidence import SCOPE_PREFIXES
from ml.features import full_window_cutoff, identity_links, load_tables
from ml.publish import DETECTOR_VERSION, STAGE2_VERSION, high_risk_components
from ml.rule import velocity_decision
from shared.contracts import EvidenceScope

#: Identity link keys followed by propagation. Both are *links*, not
#: evidence of guilt; see the module docstring on conditioning.
PROPAGATION_KEYS: tuple[str, ...] = ("device_fingerprint", "hashed_id_number")

#: Hops from a stage-1 ring member. Two hops was measured and is
#: identical to one on this dataset; see the module docstring.
PROPAGATION_HOPS = 1

#: ``STAGE2_VERSION`` is re-exported from :mod:`ml.publish`, which owns
#: every version string this pipeline stamps. It lives there rather than
#: here because ``publish`` now stamps it on the evidence rows stage 2
#: causes to exist, and ``stage2`` already imports ``publish`` — defining
#: it here would make that an import cycle.


def identity_edges(
    tables: dict[str, pd.DataFrame], cutoff: pd.Series, keys: tuple[str, ...] = PROPAGATION_KEYS
) -> pd.DataFrame:
    """Undirected identity links, point-in-time, as ``(account, other, kind)``.

    Both sightings must land at or before the *observing* account's
    cutoff, so the frame is directed in storage but symmetric in content.
    """
    links = identity_links(tables)
    rows: list[pd.DataFrame] = []
    for key in keys:
        pairs = (
            links[key]
            .groupby(["hashed_account_id", key], sort=False)["observed_at"]
            .min()
            .reset_index()
        )
        joined = pairs.merge(pairs, on=key, suffixes=("", "_other"))
        joined = joined.loc[joined["hashed_account_id"] != joined["hashed_account_id_other"]]
        own_cutoff = joined["hashed_account_id"].map(cutoff)
        joined = joined.loc[
            (joined["observed_at"] <= own_cutoff) & (joined["observed_at_other"] <= own_cutoff)
        ]
        edge = joined[["hashed_account_id", "hashed_account_id_other"]].copy()
        edge["kind"] = key
        rows.append(edge)
    return pd.concat(rows, ignore_index=True).drop_duplicates(ignore_index=True)


@dataclass(frozen=True)
class IdentityLink:
    """One account pulled into one ring, and exactly how it got there.

    This is the provenance the old ``dict[ring_id, list[str]]`` threw
    away. A bare account id says "this account is in the ring" and
    nothing else, so a consumer had no way to tell a detected mule from
    a relative who shares a phone — which is the mislabelling the whole
    ``RingMember`` contract exists to prevent. Every field here maps
    one-to-one onto a ``shared.contracts.RingMember`` field.
    """

    hashed_account_id: str
    linked_from_hashed_account_id: str
    kind: str
    scope: EvidenceScope
    link_evidence_type: str
    reason: str

    @property
    def alertable(self) -> bool:
        """Stage-2 alerting is cross-bank only; same-bank links are context."""
        return self.scope is EvidenceScope.CROSS_BANK


#: Edge kind -> the evidence name shared with ``ml.evidence.EVIDENCE_KEYS``.
#: Prefixed with the scope to give ``RingMember.link_evidence_type``, which
#: is a **join key** into ``account_evidence`` — so these are the existing
#: spellings, not new ones.
EDGE_EVIDENCE_NAMES: dict[str, str] = {
    "device_fingerprint": "shared_device",
    "hashed_id_number": "reused_id_number",
}

#: Edge kind -> the clause naming what is shared, in an analyst's words.
LINK_PHRASES: dict[str, str] = {
    "device_fingerprint": "shares a device fingerprint",
    "hashed_id_number": "shares a hashed KYC ID number",
}


def link_evidence_type(kind: str, scope: EvidenceScope) -> str:
    """``(edge kind, scope)`` -> the ``account_evidence.evidence_type`` key."""
    return f"{SCOPE_PREFIXES[scope]}_{EDGE_EVIDENCE_NAMES[kind]}"


def link_reason(kind: str, scope: EvidenceScope, linked_bank: str) -> str:
    """The sentence a ``RingMember`` carries, written as a **lead**.

    Never a finding. This account was not detected: nothing about its own
    transactions fired the velocity rule, and identity reuse was measured
    and found strictly dominated as a detector. So the sentence says what
    was observed (a shared credential), who with (a ring member, at which
    bank), and what it is (a thing to review). "Mule", "laundering" and
    any word asserting the account did something do not appear, and the
    same-bank variant additionally says out loud that it is not an alert.

    The word "mule" does not appear even inside a negation. A sentence
    ending "is not evidence that this account is a mule" reads correctly
    in full and reads as an accusation the moment a UI truncates it, so
    the denial is phrased without the noun. ``ml/test_publish.py``
    asserts the absence.
    """
    phrase = LINK_PHRASES[kind]
    if scope is EvidenceScope.CROSS_BANK:
        return (
            f"Linked, needs review: {phrase} with a ring member at another bank "
            f"({linked_bank}). This is a lead from identity reuse, not a detection — "
            "the account's own transaction behaviour did not trigger the velocity rule."
        )
    return (
        f"Context only, not an alert: {phrase} with a ring member at the same bank "
        f"({linked_bank}). Same-bank identity reuse is dominated by legitimate "
        "customers sharing a household device or refiling a document, so this is "
        "shown for context and carries no accusation against the account."
    )


def _preference(link: IdentityLink) -> tuple[object, ...]:
    """Ordering used to pick one link when several reach the same account.

    Cross-bank first: it is the citable fact, the one an STR can name a
    counterparty bank in, and the one the ruling makes alertable. Then a
    fixed key order and finally the counterparty id, so the choice is
    deterministic across runs rather than dependent on frame order.
    """
    return (
        not link.alertable,
        PROPAGATION_KEYS.index(link.kind),
        link.linked_from_hashed_account_id,
    )


def propagate(
    rings: dict[str, list[str]],
    edges: pd.DataFrame,
    bank: pd.Series,
    hops: int = PROPAGATION_HOPS,
) -> dict[str, list[IdentityLink]]:
    """Grow each **already-detected** ring along identity edges.

    Returns, per ring, the accounts stage 2 **adds** — each with the ring
    member it was reached from, the edge kind, and the scope derived from
    the two accounts' banks. Stage-1 members are deliberately *not* in
    the result: they are already the ring's velocity members and have no
    link provenance to carry.

    An empty ``rings`` returns an empty result: identity alone can never
    originate a ring, which is the safety property this stage is built
    around.
    """
    adjacency: dict[str, set[tuple[str, str]]] = {}
    for account, other, kind in edges[
        ["hashed_account_id", "hashed_account_id_other", "kind"]
    ].itertuples(index=False, name=None):
        adjacency.setdefault(account, set()).add((other, kind))

    added: dict[str, list[IdentityLink]] = {}
    for ring_id, members in rings.items():
        reached = set(members)
        frontier = set(members)
        links: dict[str, IdentityLink] = {}
        for _ in range(hops):
            nxt: dict[str, IdentityLink] = {}
            for member in sorted(frontier):
                member_bank = str(bank[member])
                for other, kind in sorted(adjacency.get(member, ())):
                    if other in reached:
                        continue
                    scope = (
                        EvidenceScope.SINGLE_BANK
                        if str(bank[other]) == member_bank
                        else EvidenceScope.CROSS_BANK
                    )
                    candidate = IdentityLink(
                        hashed_account_id=other,
                        linked_from_hashed_account_id=member,
                        kind=kind,
                        scope=scope,
                        link_evidence_type=link_evidence_type(kind, scope),
                        reason=link_reason(kind, scope, member_bank),
                    )
                    best = nxt.get(other)
                    if best is None or _preference(candidate) < _preference(best):
                        nxt[other] = candidate
            frontier = set(nxt)
            reached |= frontier
            links.update(nxt)
        added[ring_id] = [links[account] for account in sorted(links)]
    return added


def _membership_stats(members: set[str], truth: pd.DataFrame) -> dict[str, object]:
    """Role composition and ring coverage for one set of ring members."""
    role = truth["role"].reindex(sorted(members)).fillna("")
    in_ring = truth.loc[truth["ring_id"] != ""]
    fully_covered = sum(
        1 for _, group in in_ring.groupby("ring_id") if set(group.index) <= members
    )
    touched = {truth["ring_id"].get(account, "") for account in members} - {""}
    return {
        "members": len(members),
        "passthrough_mules": int(role.isin(PASSTHROUGH_MULE_ROLES).sum()),
        "sources": int((role == "source").sum()),
        "complicit": int(role.isin(COMPLICIT_ROLES).sum()),
        "non_complicit": int((~role.isin(COMPLICIT_ROLES)).sum()),
        "contamination": float((~role.isin(COMPLICIT_ROLES)).mean()) if members else 0.0,
        "true_rings_touched": len(touched),
        "true_rings_fully_covered": fully_covered,
    }


def evaluate_stage2(
    frame: pd.DataFrame, tables: dict[str, pd.DataFrame], cutoff: pd.Series
) -> dict[str, object]:
    """Run both stages and return every number the report quotes.

    ``frame`` must carry ``is_passthrough_mule`` / ``is_complicit`` from
    :func:`ml.evaluate.attach_labels`; ground truth is read from
    ``tables`` for role composition and ring coverage.
    """
    truth = tables["ground_truth"].set_index("hashed_account_id")
    flagged = velocity_decision(frame, "cb_", require_bank_spread=True)
    flagged_ids = pd.Index(frame.loc[flagged, "hashed_account_id"])
    stage1 = high_risk_components(tables["transfers"], flagged_ids, cutoff.iloc[0])

    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    edges = identity_edges(tables, cutoff)
    stage2 = propagate(stage1, edges, bank)

    members1 = {account for ids in stage1.values() for account in ids}
    all_links = [link for links in stage2.values() for link in links]
    # Net-new accounts. A link target can already be a velocity member of
    # a *different* ring; it is a genuine identity_link row in *this*
    # ring, but it is not an account stage 2 added to the ring set, and
    # counting it as one would inflate the headline.
    added = {link.hashed_account_id for link in all_links} - members1
    members2 = members1 | added
    added_role = truth["role"].reindex(sorted(added)).fillna("")

    alerts1 = set(flagged_ids)
    alerts2 = alerts1 | added
    alert_role1 = truth["role"].reindex(sorted(alerts1)).fillna("")
    alert_role2 = truth["role"].reindex(sorted(alerts2)).fillna("")

    total_complicit = int(truth["role"].isin(COMPLICIT_ROLES).sum())
    total_sources = int((truth["role"] == "source").sum())
    total_passthrough = int(truth["role"].isin(PASSTHROUGH_MULE_ROLES).sum())

    return {
        "stage2_version": STAGE2_VERSION,
        "stage1_detector_version": DETECTOR_VERSION,
        "propagation_hops": PROPAGATION_HOPS,
        "propagation_keys": list(PROPAGATION_KEYS),
        "identity_edges": int(len(edges)),
        "accounts_with_any_identity_edge": int(edges["hashed_account_id"].nunique()),
        "non_complicit_accounts_with_identity_edge": int(
            edges.loc[
                ~edges["hashed_account_id"].map(truth["role"]).isin(COMPLICIT_ROLES),
                "hashed_account_id",
            ].nunique()
        ),
        "ring_membership": {
            "stage1": _membership_stats(members1, truth),
            "stage1_plus_stage2": _membership_stats(members2, truth),
        },
        "added_by_stage2": {
            "accounts": len(added),
            "membership_rows": len(all_links),
            "alertable_rows": sum(1 for link in all_links if link.alertable),
            "context_rows": sum(1 for link in all_links if not link.alertable),
            "sources": int((added_role == "source").sum()),
            "passthrough_mules": int(added_role.isin(PASSTHROUGH_MULE_ROLES).sum()),
            "non_complicit": int((~added_role.isin(COMPLICIT_ROLES)).sum()),
            "precision": float(added_role.isin(COMPLICIT_ROLES).mean()) if added else 0.0,
            "non_complicit_cohorts": truth.loc[
                [a for a in sorted(added) if truth["role"].get(a, "") not in COMPLICIT_ROLES],
                "cohort",
            ]
            .value_counts()
            .to_dict(),
        },
        "alert_level": {
            "totals": {
                "complicit": total_complicit,
                "sources": total_sources,
                "passthrough_mules": total_passthrough,
            },
            "stage1": {
                "alerts": len(alerts1),
                "complicit_recall": float(
                    alert_role1.isin(COMPLICIT_ROLES).sum() / total_complicit
                ),
                "source_recall": float((alert_role1 == "source").sum() / total_sources),
                "passthrough_recall": float(
                    alert_role1.isin(PASSTHROUGH_MULE_ROLES).sum() / total_passthrough
                ),
                "non_complicit_alerts": int((~alert_role1.isin(COMPLICIT_ROLES)).sum()),
            },
            "stage1_plus_stage2": {
                "alerts": len(alerts2),
                "complicit_recall": float(
                    alert_role2.isin(COMPLICIT_ROLES).sum() / total_complicit
                ),
                "source_recall": float((alert_role2 == "source").sum() / total_sources),
                "passthrough_recall": float(
                    alert_role2.isin(PASSTHROUGH_MULE_ROLES).sum() / total_passthrough
                ),
                "non_complicit_alerts": int((~alert_role2.isin(COMPLICIT_ROLES)).sum()),
            },
        },
        "missed_rings": missed_ring_crosstab(truth, members1, members2),
        "source_recovery_by_pattern": source_recovery_crosstab(truth, members2),
    }


def missed_ring_crosstab(
    truth: pd.DataFrame, members1: set[str], members2: set[str]
) -> dict[str, object]:
    """Pattern x span cross-tab of recovered vs missed true rings.

    A true ring is "recovered" when at least one of its members lands in
    a detected ring. Reported for both stages so the (null) stage-2
    effect on ring *discovery* is visible rather than assumed.
    """
    in_ring = truth.loc[truth["ring_id"] != ""]
    info = in_ring.groupby("ring_id").agg(
        pattern=("pattern_type", "first"), span=("ring_bank_count", "first")
    )
    touched1 = {truth["ring_id"].get(a, "") for a in members1} - {""}
    touched2 = {truth["ring_id"].get(a, "") for a in members2} - {""}
    info["recovered_stage1"] = info.index.isin(sorted(touched1))
    info["recovered_stage1_plus_stage2"] = info.index.isin(sorted(touched2))

    def table(subset: pd.DataFrame) -> dict[str, dict[str, int]]:
        if subset.empty:
            return {}
        cross = pd.crosstab(subset["pattern"], subset["span"])
        return {str(p): {f"span{s}": int(v) for s, v in row.items()} for p, row in cross.iterrows()}

    return {
        "all_rings": table(info),
        "recovered_stage1": table(info.loc[info["recovered_stage1"]]),
        "missed_stage1": table(info.loc[~info["recovered_stage1"]]),
        "newly_recovered_by_stage2": table(
            info.loc[~info["recovered_stage1"] & info["recovered_stage1_plus_stage2"]]
        ),
    }


def source_recovery_crosstab(truth: pd.DataFrame, members2: set[str]) -> dict[str, object]:
    """Pattern x span cross-tab of which ``source`` accounts stage 2 reaches.

    The structural point the headline number hides: stage 2 can only
    follow an identity edge, so a ring whose pattern deliberately reuses
    no credentials leaks no sources to it.
    """
    sources = truth.loc[truth["role"] == "source"].copy()
    sources["recovered"] = sources.index.isin(sorted(members2))
    got = pd.crosstab(
        sources["pattern_type"],
        sources["ring_bank_count"],
        values=sources["recovered"],
        aggfunc="sum",
    ).fillna(0)
    total = pd.crosstab(sources["pattern_type"], sources["ring_bank_count"])
    return {
        str(pattern): {
            f"span{span}": {
                "recovered": int(got.loc[pattern, span]),
                "total": int(total.loc[pattern, span]),
            }
            for span in total.columns
        }
        for pattern in total.index
    }


def main() -> None:
    """Measure stage 2 and write ``ml/experiments/stage2/metrics.json``.

    This is the **measurement** entry point only. Stage 2 is now shipped:
    ``ml.publish.main`` calls :func:`propagate` and folds the result into
    ``ml/out/rings.parquet`` as ``identity_link`` members, each carrying
    the link it arrived by. That resolved the question this module used
    to leave open — what a published ring *means* once members can be
    there on an identity link — by making the basis part of every member
    row rather than merging the two claims into one flat list.

    What stays here is the evaluation against ground truth (role
    composition, ring coverage, the source-recovery cross-tab), which
    needs labels ``ml/out/`` must never carry.
    """
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    cutoff = full_window_cutoff(tables)

    result = evaluate_stage2(frame, tables, cutoff)
    result["generated_at"] = datetime.now(UTC).isoformat()

    out_dir = cfg.experiments_dir / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(result, indent=2, default=str), "utf-8")

    stage1 = result["ring_membership"]["stage1"]  # type: ignore[index]
    stage2 = result["ring_membership"]["stage1_plus_stage2"]  # type: ignore[index]
    added = result["added_by_stage2"]  # type: ignore[index]
    print(
        f"stage1 members={stage1['members']} contamination={stage1['contamination']:.2%} "
        f"sources={stage1['sources']} rings_touched={stage1['true_rings_touched']}/120"
    )
    print(
        f"stage2 members={stage2['members']} contamination={stage2['contamination']:.2%} "
        f"sources={stage2['sources']} rings_touched={stage2['true_rings_touched']}/120 "
        f"fully_covered={stage2['true_rings_fully_covered']}/120"
    )
    print(
        f"added={added['accounts']} (sources={added['sources']} "
        f"mules={added['passthrough_mules']} false={added['non_complicit']}) "
        f"precision={added['precision']:.1%}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
