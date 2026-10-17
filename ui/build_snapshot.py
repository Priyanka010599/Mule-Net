"""Offline builder for the demo snapshot the UI serves.

Joins the detector's published output in ``ml/out/`` to the transaction
graph in ``simulator/out/`` and writes one compact JSON bundle to
``ui/out/``. Run it once; the FastAPI app then never touches parquet at
request time, which is what keeps every endpoint fast.

Where each field comes from
---------------------------
``ml/out/rings.parquet``
    The rings actually found by the shipped velocity rule. **Not**
    ground truth. 45 rings over 298 accounts.
``ml/out/scores.parquet``
    Out-of-fold XGBoost P(mule) for all 30,000 accounts. This is what
    ranks the bank alert queues -- a real continuous score, not a rule
    of thumb invented here.
``ml/out/ring_explanations.parquet``
    The ring narrative and its median feature values next to the rule
    thresholds. Quoted verbatim; nothing is paraphrased.
``ml/out/account_evidence.parquet``
    Cross-bank identity reuse. Supporting evidence only -- it played no
    part in the firing condition (see ``ml/out/README.md``).
``ml/out/ring_detection_timeline.parquet``
    Per TRUE ring, per scope, per horizon T: whether that scope had
    flagged the ring yet, plus the ring's onset and the instant the money
    reached the exit account. Drives the replay's flag markers. T is
    minutes after onset; a NULL T is the full window.
``ml/out/ring_id_map.parquet``
    Joins the detector's 45 cluster ids to the 120 ground-truth ring ids.
    The two id spaces do not intersect, and the pairing is a
    best-overlap heuristic: ``match_is_ambiguous`` says when it must not
    be presented as identity.
``simulator/out/``
    Accounts, the transfer edges the graph draws, and ground-truth
    labels used for one thing only: telling the operator which flagged
    accounts are true positives, so the demo cannot flatter itself.

Why a snapshot and not a live query
-----------------------------------
The dataset is 30,000 accounts and 572,890 transfers. None of that can
be laid out in Cytoscape and almost none of it is interesting. Per ring
the snapshot keeps the members, their one-hop counterparties, and only
the transfers between that node set.

Usage::

    python -m ui.build_snapshot
    python -m ui.build_snapshot --rings 24 --out ui/out/demo_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from shared import ring_race
from shared.contracts import (
    Account,
    Investigation,
    Ring,
    RingExplanation,
    RingMember,
    RiskScore,
    StrReport,
    Transfer,
)
from ui.config import UiConfig

#: How many alerts each bank tile shows. A real console paginates; the
#: demo shows one screen and also reports the depth of the full queue.
ALERTS_PER_BANK = 25

#: Intra-ring edges are kept in full: they are the money trail the demo
#: animates, and truncating them would silently drop hops. Context edges
#: get their own reserved budget rather than whatever the trail leaves
#: over -- a busy ring would otherwise consume the whole allowance and
#: strand every counterparty as an isolated dot.
MAX_CONTEXT_EDGES = 120

#: Intra-ring hops are deliberately NOT capped, so the edge total per
#: subgraph cannot be promised in advance. ``bounding.max_transfers_per_ring``
#: is therefore reported as the largest count actually present in the
#: built snapshot -- a measured fact rather than a limit the data could
#: quietly exceed.

#: Upper bound on rings in one snapshot.
MAX_RINGS = 24

#: Recall level at which each bank's alert burden is quoted. Mirrors
#: ``ml/metrics.py:BURDEN_RECALL`` so the tile and the metrics page
#: describe the same operating point.
BURDEN_RECALL = 0.8

#: ``cross_bank_reused_id_number`` and ``cross_bank_reused_photo`` fire
#: on exactly the same accounts (Jaccard 1.000) because the generator
#: attaches a reused photo to precisely the documents that reuse an ID
#: number. They are collapsed into one citable finding; rendering them
#: as two independent corroborations would double-count one fact.
_IDENTITY_TYPES = frozenset(
    {
        "cross_bank_reused_id_number",
        "cross_bank_reused_photo",
        "same_bank_reused_id_number",
        "same_bank_reused_photo",
    }
)

SAME_BANK_CONTEXT_NOTE = (
    "Same-bank reuse: context only, not corroboration. Roughly 1,442 legitimate "
    "accounts share a document within a bank against ~292 mules, so this is "
    "dominated by households and refiled paperwork."
)


def _evidence_label(collapsed: bool, cross_bank: bool) -> str:
    """Name a finding so its strength is legible from the label alone."""
    if collapsed:
        return (
            "KYC identity reused at another bank"
            if cross_bank
            else "KYC identity reused within this bank (context)"
        )
    return (
        "Device fingerprint shared across banks"
        if cross_bank
        else "Device fingerprint shared within this bank (context)"
    )


IDENTITY_COLLAPSE_NOTE = (
    "Reused ID number and matching photo fire on the same accounts in this "
    "dataset - a property of the generator, not of the model. Cited once."
)

#: Placeholder citations. ``agents/`` replaces these with real passages
#: from Vertex AI Search over the AML corpus; they are marked unverified
#: in the payload so the UI never presents them as retrieved.
PLACEHOLDER_CITATIONS: list[str] = [
    "FATF Recommendation 20 - Reporting of suspicious transactions",
    "FATF Recommendation 16 - Wire transfers (originator/beneficiary information)",
    "31 CFR 1020.320 - Reports by banks of suspicious transactions",
    "Directive (EU) 2015/849 Art. 33 - Reporting obligations",
]

_SIM_TABLES = ("accounts", "transfers", "ground_truth")
_ML_TABLES = (
    "rings",
    "scores",
    "ring_explanations",
    "account_evidence",
    "ring_detection_timeline",
    "ring_id_map",
    "bank_alert_queues",
)

#: Every replay carries this, verbatim. One seed, one evaluation run --
#: not an average, not a live stream.
REPLAY_LABEL = "Replay: seed 42 evaluation run"

#: The three ways a scope can fail to catch a ring inside the horizon
#: grid. They are genuinely different outcomes and must never collapse
#: into one phrase:
#:
#: ``flagged_full_window_only``
#:     The scope does flag, but only over the full window -- which is
#:     observed at the end of the dataset, days after cash-out. It is a
#:     detection, and it is NOT a catch.
#: ``never_flagged``
#:     The scope does not flag at any horizon, full window included.
#:
#: A scope that flags inside the grid but after ``exit_reached_at`` is a
#: third distinct case again: it fired in time to be measured, but too
#: late to stop the money.
NEVER_FLAGGED_LABEL = "never flagged"
FULL_WINDOW_ONLY_LABEL = "flagged only at the full window (after the money left)"

EXIT_LABEL = "money reaches the exit account"


class MlOutputMissingError(RuntimeError):
    """Raised when ``ml/out/`` has not been published yet."""


def _load(cfg: UiConfig) -> dict[str, pd.DataFrame]:
    tables = {t: pd.read_parquet(cfg.raw_table_path(t)) for t in _SIM_TABLES}
    for table in _ML_TABLES:
        path = cfg.ml_table_path(table)
        if not path.exists():
            raise MlOutputMissingError(
                f"{path} is missing. The UI reads the detector's published "
                "tables; regenerate them with `python -m ml.publish` "
                "(`python -m ml.publish_all --only bank_alert_queues` for the "
                "single-bank queue table alone)."
            )
        tables[table] = pd.read_parquet(path)
    return tables


def _load_cases(cfg: UiConfig) -> dict[str, Investigation]:
    """Read the verified cases ``agents/`` published, keyed by ring.

    Absence is a normal state, not an error. ``agents/`` writes a row
    only for an investigation whose claims the verifier re-derived and
    matched; a withheld narrative produces no row, and a machine where
    the agents have never run has no file. Both cases come back as an
    empty mapping and the ring renders "no case yet".
    """
    path = cfg.agents_table_path("cases")
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    cases: dict[str, Investigation] = {}
    for row in frame.to_dict("records"):
        closed_at = row.get("closed_at")
        cases[row["ring_id"]] = Investigation(
            case_id=row["case_id"],
            ring_id=row["ring_id"],
            verified_at=pd.Timestamp(row["verified_at"]).to_pydatetime(),
            verification_attempts=int(row["verification_attempts"]),
            verified_claim_count=int(row["verified_claim_count"]),
            analyst_notes=row.get("analyst_notes") or "",
            status=row["status"],
            opened_at=pd.Timestamp(row["opened_at"]).to_pydatetime(),
            closed_at=pd.Timestamp(closed_at).to_pydatetime() if closed_at else None,
        )
    return cases


def _load_str_reports(cfg: UiConfig) -> dict[str, dict[str, Any]]:
    """Read the real Gemini-drafted STRs ``agents/batch_draft_reports.py``
    published, keyed by case id.

    Absence is a normal state, not an error, same as :func:`_load_cases`:
    a machine where that batch has never run has no file, and every ring
    falls back to the synthesized ``PLACEHOLDER_CITATIONS`` report below.
    """
    path = cfg.agents_table_path("str_reports")
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    return {
        row["case_id"]: {
            "report_id": row["report_id"],
            "case_id": row["case_id"],
            "narrative": row["narrative"],
            "regulation_citations": list(row["regulation_citations"]),
            "generated_at": pd.Timestamp(row["generated_at"]).to_pydatetime(),
            "status": row["status"],
            "citations_verified": bool(row["citations_verified"]),
        }
        for row in frame.to_dict("records")
    }


def _iso(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime().astimezone(UTC).isoformat()


def _queue_at_recall(
    ranked_ids: list[str],
    truth: dict[str, bool],
    recall: float = BURDEN_RECALL,
) -> dict[str, Any]:
    """Size a bank's alert queue at the threshold reaching ``recall``.

    The head of a score-ranked queue is nearly pure, and that head is
    not the operational picture. What an AML team staffs against is the
    depth needed to catch most of the mules, and that is where the
    single-bank false-alert burden shows up. Both are reported so a tile
    cannot flatter itself with its top rows alone.
    """
    labels = [bool(truth.get(a, False)) for a in ranked_ids]
    total_positives = sum(labels)
    if total_positives == 0:
        return {
            "queue_depth": 0,
            "queue_true_positives": 0,
            "queue_false_positives": 0,
            "false_alerts_per_catch": None,
            "recall_target": recall,
            "reached_recall": False,
        }
    needed = max(1, int(round(recall * total_positives)))
    caught = 0
    depth = len(labels)
    for idx, is_mule in enumerate(labels, start=1):
        if is_mule:
            caught += 1
        if caught >= needed:
            depth = idx
            break
    reached = caught >= needed
    false_positives = depth - caught
    return {
        "queue_depth": depth,
        "queue_true_positives": caught,
        "queue_false_positives": false_positives,
        "false_alerts_per_catch": round(false_positives / caught, 2) if caught else None,
        "recall_target": recall,
        "reached_recall": reached,
    }


#: What a single-bank tile says about its own ranking. The full-window queue is
#: ceiling-bound, so the early-horizon comparison on tab 3 stays the honest one.
SINGLE_BANK_QUEUE_CAVEAT = (
    "Ranked by this bank's own single-bank model ({model}), which sees only this "
    "bank's own accounts and transfers. It is a full-window number, which is "
    "ceiling-bound. The honest single-bank vs cross-bank comparison is on tab 3, "
    "at the early horizons."
)


def _single_bank_tiles(
    queues: pd.DataFrame,
    transfers: pd.DataFrame,
    truth: dict[str, bool],
) -> list[dict[str, Any]]:
    """One tile per bank from ``bank_alert_queues`` (single-bank ``sb_`` scores).

    ``score``, ``rank_in_bank`` and ``top_factors`` are label-free. The queue
    block and ``mules_in_bank`` are copied from the table's label-derived
    columns; ``ui.datasource`` withholds them unless labels are revealed.
    ``top_factors`` is mapped to the ``list[str]`` of each factor's ``description``.
    """
    tiles: list[dict[str, Any]] = []
    for bank_id, group in queues.groupby("bank_id", sort=True):
        ranked = group.sort_values("rank_in_bank", kind="stable")
        first = ranked.iloc[0]
        alerts = [
            {
                "hashed_account_id": row.hashed_account_id,
                "bank_id": bank_id,
                "score": round(float(row.score), 6),
                "rank_in_bank": int(row.rank_in_bank),
                "top_factors": [str(f["description"]) for f in row.top_factors],
                "is_mule": bool(truth.get(row.hashed_account_id, False)),
            }
            for row in ranked.head(ALERTS_PER_BANK).itertuples(index=False)
        ]
        bank_account_ids = set(ranked["hashed_account_id"])
        from_own = transfers["from_hashed_account_id"].isin(bank_account_ids)
        to_own = transfers["to_hashed_account_id"].isin(bank_account_ids)
        true_positives = sum(1 for a in alerts if a["is_mule"])
        model = str(first["model_version"])
        tiles.append(
            {
                "bank_id": bank_id,
                "account_count": len(bank_account_ids),
                "transfers_seen": int((from_own | to_own).sum()),
                "transfers_leaving_perimeter": int((from_own & ~to_own).sum()),
                "alerts": alerts,
                "alert_true_positives": true_positives,
                "alert_false_positives": len(alerts) - true_positives,
                "mules_in_bank": int(first["mules_in_bank"]),
                "queue": {
                    "queue_depth": int(first["queue_depth"]),
                    "queue_true_positives": int(first["queue_true_positives"]),
                    "queue_false_positives": int(first["queue_false_positives"]),
                    "false_alerts_per_catch": (
                        None
                        if pd.isna(first["false_alerts_per_catch"])
                        else float(first["false_alerts_per_catch"])
                    ),
                    "recall_target": float(first["queue_recall_target"]),
                    "reached_recall": bool(first["reached_recall"]),
                },
                "score_model_version": model,
                "score_scope": "single_bank",
                "queue_caveat": SINGLE_BANK_QUEUE_CAVEAT.format(model=model),
            }
        )
    return tiles


def _detector_version_of(ring_id: str) -> str:
    """The detector that CREATED this ring, read from its ``ring_id``.

    ``ml/publish.py`` mints every ring id as
    ``f"ring-{DETECTOR_VERSION}-{label:05d}"``, so the id carries its
    own provenance and nothing else has to be joined to recover it.

    This is deliberately NOT derived from the ring's evidence rows.
    ``account_evidence.detector_version`` is stamped per ACCOUNT -- the
    velocity rule for an account it flagged, stage-2 identity
    propagation for an account only pulled in by a link (see
    ``ml.publish.build_evidence``) -- so a ring that mixes the two has
    no single evidence-derived answer, and picking one row's value
    makes the field depend on parquet row order. Which detectors
    contributed evidence is already reported per finding, in
    ``evidence[].detector_versions``.
    """
    return ring_id.rsplit("-", 1)[0].removeprefix("ring-")


def _ring_evidence(
    evidence: pd.DataFrame,
    members: set[str],
    bank_of: dict[str, str],
) -> list[dict[str, Any]]:
    """Aggregate ``account_evidence`` rows into citable ring findings.

    One entry per (kind, scope), not per row. The reused-id-number and
    reused-photo rows collapse into a single identity finding (see
    :data:`IDENTITY_COLLAPSE_NOTE`), and scope is kept as part of the
    key rather than folded away: cross-bank reuse is the fact worth
    citing, while same-bank reuse is dominated by legitimate cover and
    is context only. Merging the two would let household noise be
    presented as corroboration.
    """
    rows = evidence[evidence["hashed_account_id"].isin(members)]
    if rows.empty:
        return []

    groups: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)
    # Sorted before grouping, so the per-group representatives picked
    # below (``detail``, ``source_table``) name a fixed account rather
    # than whichever one parquet happened to store first.
    ordered = rows.sort_values(
        ["hashed_account_id", "evidence_type"], kind="stable"
    )
    for _, row in ordered.iterrows():
        kind = (
            "identity_reuse"
            if row["evidence_type"] in _IDENTITY_TYPES
            else "shared_device"
        )
        groups[(kind, str(row["scope"]))].append(row)

    findings: list[dict[str, Any]] = []
    for (kind, scope), group in groups.items():
        collapsed = kind == "identity_reuse"
        is_cross_bank = scope == "cross_bank"
        member_ids = sorted({r["hashed_account_id"] for r in group})
        counterparties: set[str] = set()
        counterparty_banks: set[str] = set()
        for r in group:
            counterparties.update(r["counterparty_hashed_account_ids"])
            counterparty_banks.update(r["counterparty_bank_ids"])
        own_banks = sorted({bank_of.get(a, "?") for a in member_ids})
        raw_types = sorted({r["evidence_type"] for r in group})
        findings.append(
            {
                "kind": kind,
                "evidence_type": raw_types[0],
                "raw_evidence_types": raw_types,
                "label": _evidence_label(collapsed, is_cross_bank),
                "scope": str(group[0]["scope"]),
                "detail": str(group[0]["detail"]).strip(),
                "ring_member_hashed_account_ids": member_ids,
                "counterparty_hashed_account_ids": sorted(counterparties),
                "counterparty_bank_ids": sorted(counterparty_banks),
                "banks": sorted(set(own_banks) | counterparty_banks),
                # Scope is read from the row, never inferred by comparing
                # bank lists: that derivation is exactly the one that
                # turns household noise into an accusation.
                "cross_bank": is_cross_bank,
                "accounts_with_this": len(member_ids),
                "source_table": str(group[0]["source_table"]),
                "observed_at": _iso(min(r["observed_at"] for r in group)),
                # Every detector that stamped a row in this group, not
                # one row's. The column is per-account (velocity rule vs
                # stage-2 propagation), so a finding spanning both has
                # two answers and must report both.
                "detector_versions": sorted(
                    {str(r["detector_version"]) for r in group}
                ),
                "note": " ".join(
                    part
                    for part in (
                        IDENTITY_COLLAPSE_NOTE if collapsed else "",
                        "" if is_cross_bank else SAME_BANK_CONTEXT_NOTE,
                    )
                    if part
                ),
            }
        )
    # Cross-bank findings first: they are the citable ones.
    findings.sort(key=lambda f: (not f["cross_bank"], -f["accounts_with_this"]))
    return findings


def _build_ring(
    ring_row: pd.Series,
    explanation_row: pd.Series | None,
    accounts: pd.DataFrame,
    transfers: pd.DataFrame,
    evidence: pd.DataFrame,
    scores: pd.DataFrame,
    truth: dict[str, bool],
    roles: dict[str, str],
    bank_of: dict[str, str],
    cases: dict[str, Investigation],
    str_reports: dict[str, dict[str, Any]],
    cfg: UiConfig,
) -> dict[str, Any] | None:
    """Assemble one ring: subgraph, members, evidence, explanation, report."""
    if len(ring_row["members"]) == 0:
        return None

    # Build through the real contract so a member whose link fields
    # disagree with its basis fails here, at build time, rather than
    # being rendered to an analyst as a detection.
    ring = Ring(
        ring_id=ring_row["ring_id"],
        members=[RingMember(**dict(m)) for m in ring_row["members"]],
        banks_involved=list(ring_row["banks_involved"]),
        detected_at=pd.Timestamp(ring_row["detected_at"]).to_pydatetime(),
        status=ring_row["status"],
    )
    basis_of = {m.hashed_account_id: m for m in ring.members}
    # Subgraph expansion uses every member: a context member is drawn,
    # just never styled or counted as a detection.
    members = set(ring.all_member_ids)

    touching = transfers[
        transfers["from_hashed_account_id"].isin(members)
        | transfers["to_hashed_account_id"].isin(members)
    ]
    if touching.empty:
        return None

    context = (
        set(touching["from_hashed_account_id"]) | set(touching["to_hashed_account_id"])
    ) - members
    node_budget = max(0, cfg.max_graph_nodes - len(members))
    if len(context) > node_budget:
        # Keep the counterparties carrying the most money; the rest are
        # background traffic and only cost layout time.
        weight: dict[str, float] = defaultdict(float)
        for row in touching.itertuples(index=False):
            for side in (row.from_hashed_account_id, row.to_hashed_account_id):
                if side in context:
                    weight[side] += float(row.amount)
        context = set(sorted(context, key=lambda a: -weight[a])[:node_budget])

    node_set = members | context
    edges = touching[
        touching["from_hashed_account_id"].isin(node_set)
        & touching["to_hashed_account_id"].isin(node_set)
    ].sort_values("timestamp", kind="stable")

    intra_mask = edges["from_hashed_account_id"].isin(members) & edges[
        "to_hashed_account_id"
    ].isin(members)
    intra_edges = edges[intra_mask]
    context_edges = edges[~intra_mask]
    if len(context_edges) > MAX_CONTEXT_EDGES:
        # ``amount`` arrives as decimal128, which pandas holds as object
        # and cannot rank directly; rank on a float copy instead.
        ranked = context_edges.assign(_amount_f=context_edges["amount"].astype(float))
        keep = ranked.nlargest(MAX_CONTEXT_EDGES, "_amount_f").index
        context_edges = context_edges.loc[keep]
    edges = pd.concat([intra_edges, context_edges]).sort_values("timestamp", kind="stable")

    # Drop counterparties whose edges all fell outside the context
    # budget: an unconnected node renders as an isolated dot and tells
    # the viewer nothing.
    connected = set(edges["from_hashed_account_id"]) | set(edges["to_hashed_account_id"])
    context &= connected
    node_set = members | context

    account_rows = accounts[accounts["hashed_account_id"].isin(node_set)].set_index(
        "hashed_account_id"
    )
    score_rows = scores.set_index("hashed_account_id")

    member_payload: list[dict[str, Any]] = []
    for member_id in ring.all_member_ids:
        if member_id not in account_rows.index:
            continue
        acct_row = account_rows.loc[member_id]
        account = Account(
            hashed_account_id=member_id,
            bank_id=acct_row["bank_id"],
            opened_at=pd.Timestamp(acct_row["opened_at"]).to_pydatetime(),
            account_type=acct_row["account_type"],
            status=acct_row["status"],
        )
        basis = basis_of[member_id]
        score_row = score_rows.loc[member_id]
        risk = RiskScore(
            hashed_account_id=member_id,
            score=float(score_row["score"]),
            model_version=str(score_row["model_version"]),
            scored_at=pd.Timestamp(score_row["scored_at"]).to_pydatetime(),
            top_factors=list(score_row["top_factors"]),
        )
        member_payload.append(
            {
                "hashed_account_id": account.hashed_account_id,
                "bank_id": account.bank_id,
                "account_type": account.account_type,
                "status": account.status.value,
                "opened_at": _iso(account.opened_at),
                "role": roles.get(member_id, ""),
                "risk_score": risk.score,
                "model_version": risk.model_version,
                "top_factors": risk.top_factors,
                "is_true_positive": bool(truth.get(member_id, False)),
                # Provenance travels with the account, never in a
                # parallel array that could slip out of alignment.
                # ``alertable`` is copied from the row, not re-derived:
                # re-deriving it means joining evidence and comparing
                # banks, and getting that wrong publishes an alert
                # against somebody's relative.
                "membership_basis": basis.membership_basis.value,
                "alertable": basis.alertable,
                "link_scope": basis.link_scope.value if basis.link_scope else None,
                "linked_from_hashed_account_id": basis.linked_from_hashed_account_id,
                "link_evidence_type": basis.link_evidence_type,
                "reason": basis.reason,
            }
        )

    context_payload = [
        {
            "hashed_account_id": account_id,
            "bank_id": str(account_rows.loc[account_id, "bank_id"]),
            "account_type": str(account_rows.loc[account_id, "account_type"]),
            "risk_score": float(score_rows.loc[account_id, "score"])
            if account_id in score_rows.index
            else 0.0,
        }
        for account_id in sorted(context)
        if account_id in account_rows.index
    ]

    transfer_payload: list[dict[str, Any]] = []
    for row in edges.itertuples(index=False):
        transfer = Transfer(
            transfer_id=row.transfer_id,
            from_hashed_account_id=row.from_hashed_account_id,
            to_hashed_account_id=row.to_hashed_account_id,
            amount=row.amount,
            currency=row.currency,
            timestamp=row.timestamp.to_pydatetime(),
            channel=row.channel,
        )
        from_bank = bank_of.get(transfer.from_hashed_account_id, "?")
        to_bank = bank_of.get(transfer.to_hashed_account_id, "?")
        transfer_payload.append(
            {
                "transfer_id": transfer.transfer_id,
                "from_hashed_account_id": transfer.from_hashed_account_id,
                "to_hashed_account_id": transfer.to_hashed_account_id,
                "amount": f"{transfer.amount:.2f}",
                "currency": transfer.currency,
                "timestamp": _iso(transfer.timestamp),
                "channel": transfer.channel,
                "from_bank": from_bank,
                "to_bank": to_bank,
                "cross_bank": from_bank != to_bank,
                "intra_ring": (
                    transfer.from_hashed_account_id in members
                    and transfer.to_hashed_account_id in members
                ),
            }
        )

    findings = _ring_evidence(evidence, members, bank_of)
    intra = [t for t in transfer_payload if t["intra_ring"]]

    if explanation_row is not None:
        explanation = RingExplanation(
            ring_id=ring.ring_id,
            top_factors=list(explanation_row["top_factors"]),
            narrative=str(explanation_row["narrative"]),
            generated_at=pd.Timestamp(explanation_row["generated_at"]).to_pydatetime(),
        )
    else:  # pragma: no cover - ml publishes one explanation per ring
        explanation = RingExplanation(
            ring_id=ring.ring_id,
            top_factors=[],
            narrative="No explanation was published for this ring.",
        )

    # A case exists only where ``agents/`` ran an investigation whose
    # claims the verifier re-derived and matched. A withheld
    # investigation publishes no row, and the honest render of that is
    # "no case yet" -- never a fabricated open case, which would tell an
    # analyst that a narrative had been checked when none was written.
    case = cases.get(ring.ring_id)
    case_payload = (
        {
            "case_id": case.case_id,
            "ring_id": case.ring_id,
            "status": case.status.value,
            "analyst_notes": case.analyst_notes,
            "opened_at": _iso(case.opened_at),
            "verified_at": _iso(case.verified_at),
            "verification_attempts": case.verification_attempts,
            "verified_claim_count": case.verified_claim_count,
            "closed_at": _iso(case.closed_at) if case.closed_at else None,
        }
        if case
        else None
    )

    # An STR is drafted *for a case*. With no verified case there is
    # nothing to draft against, so the report is absent too rather than
    # being addressed to an invented case id.
    report_payload = None
    if case:
        real = str_reports.get(case.case_id)
        if real is not None:
            # A real Gemini draft exists for this case (batch_draft_reports.py
            # ran against the live regulation index) -- use it as published,
            # rather than resynthesizing the placeholder version beside it.
            report_payload = {
                "report_id": real["report_id"],
                "case_id": real["case_id"],
                "narrative": real["narrative"],
                "regulation_citations": real["regulation_citations"],
                "generated_at": _iso(real["generated_at"]),
                "status": real["status"],
                "citations_verified": real["citations_verified"],
            }
        else:
            evidence_paragraph = (
                " Supporting evidence, which took no part in the detection decision: "
                + " ".join(f["detail"].rstrip(".") + "." for f in findings)
                if findings
                else " No cross-bank identity reuse is attached to this ring; the "
                "finding rests on transaction structure alone."
            )
            report = StrReport(
                case_id=case.case_id,
                narrative=explanation.narrative + evidence_paragraph,
                regulation_citations=list(PLACEHOLDER_CITATIONS),
            )
            report_payload = {
                "report_id": report.report_id,
                "case_id": report.case_id,
                "narrative": report.narrative,
                "regulation_citations": report.regulation_citations,
                "generated_at": _iso(report.generated_at),
                "status": report.status.value,
                "citations_verified": report.citations_verified,
            }

    single_bank_view = {}
    for bank in ring.banks_involved:
        visible = [m["hashed_account_id"] for m in member_payload if m["bank_id"] == bank]
        visible_intra = [
            t
            for t in intra
            if t["from_bank"] == bank and t["to_bank"] == bank
        ]
        single_bank_view[bank] = {
            "visible_member_hashed_account_ids": visible,
            "flagged_hashed_account_ids": visible,
            "invisible_member_count": len(member_payload) - len(visible),
            "visible_alertable_hashed_account_ids": [
                m["hashed_account_id"]
                for m in member_payload
                if m["bank_id"] == bank and m["alertable"]
            ],
            "visible_intra_ring_transfers": len(visible_intra),
        }

    # Precision is quoted over the ALERTABLE population only. Counting
    # context members here would score the system on accounts it never
    # accused, in either direction.
    alertable_ids = set(ring.alertable_member_ids)
    true_positives = sum(
        1
        for m in member_payload
        if m["alertable"] and m["is_true_positive"]
    )
    return {
        "ring_id": ring.ring_id,
        # No flat member list: the contract removed it so that a linked
        # relative can never be read as a detected mule. Each population
        # is named, and a consumer must say which one it means.
        "all_member_hashed_account_ids": ring.all_member_ids,
        "alertable_member_hashed_account_ids": ring.alertable_member_ids,
        "context_member_hashed_account_ids": ring.context_member_ids,
        "velocity_member_hashed_account_ids": ring.velocity_member_ids,
        "identity_link_member_hashed_account_ids": ring.identity_link_member_ids,
        "alertable_member_count": len(alertable_ids),
        "context_member_count": len(ring.context_member_ids),
        "total_member_count": len(ring.members),
        "banks_involved": ring.banks_involved,
        "span": len(ring.banks_involved),
        "status": ring.status.value,
        "detected_at": _iso(ring.detected_at),
        "onset": min((t["timestamp"] for t in intra), default=None),
        "detector_version": _detector_version_of(ring.ring_id),
        "total_intra_ring_amount": f"{sum(Decimal(t['amount']) for t in intra):.2f}",
        "intra_ring_transfer_count": len(intra),
        "cross_bank_transfer_count": sum(1 for t in intra if t["cross_bank"]),
        "true_positive_alertable_members": true_positives,
        "members": member_payload,
        "context_accounts": context_payload,
        "transfers": transfer_payload,
        "evidence": findings,
        "explanation": {
            "ring_id": explanation.ring_id,
            "top_factors": explanation.top_factors,
            "narrative": explanation.narrative,
            "generated_at": _iso(explanation.generated_at),
        },
        "case": case_payload,
        "report": report_payload,
        "single_bank_view": single_bank_view,
    }


# ── replay ───────────────────────────────────────────────────────────


def _horizon_grid(timeline: pd.DataFrame) -> list[int]:
    """The measured finite horizons, read from the table, not assumed."""
    return sorted(int(t) for t in timeline["T"].dropna().unique())


def _ambiguity(row: pd.Series) -> dict[str, Any]:
    """Describe the detected -> true ring pairing, caveat included.

    ``match_is_ambiguous`` is copied from the row and never re-derived.
    When it is true the cluster is a merge, a split or a tie, and the
    caveat below is the only honest caption: a replay titled with one
    true ring id, on a cluster that holds parts of several, tells a
    viewer the cluster *is* that ring.
    """
    true_ring_id = row["true_ring_id"]
    ambiguous = bool(row["match_is_ambiguous"])
    touched = int(row["true_rings_touched"])
    overlap = int(row["overlap_members"])
    detected_size = int(row["detected_ring_size"])
    caveat = ""
    if ambiguous:
        largest = "one of the largest (tied)" if bool(row["overlap_tied"]) else "the largest"
        caveat = (
            f"This detected cluster contains parts of {touched} true rings; "
            f"the replay shows {largest}, {true_ring_id}, covering "
            f"{overlap} of {detected_size} members."
        )
    return {
        "is_ambiguous": ambiguous,
        "true_rings_touched": touched,
        "overlap_members": overlap,
        "detected_ring_size": detected_size,
        "true_ring_size": (
            int(row["true_ring_size"]) if not pd.isna(row["true_ring_size"]) else None
        ),
        "overlap_tied": bool(row["overlap_tied"]),
        "detected_rings_touching_true_ring": int(row["detected_rings_touching_true_ring"]),
        "caveat": caveat,
    }


def _scope_replay(
    rows: pd.DataFrame,
    scope: str,
    onset: pd.Timestamp,
    exit_offset_minutes: float,
) -> dict[str, Any]:
    """When one scope first flagged the ring, against the exit.

    The flag instant is ``onset + T`` for the earliest measured ``T`` at
    which that scope had flagged: per ``ml/out/manifest.json`` that is
    exactly the instant the decision was taken at. A NULL ``T`` is the
    full window, which flags every ring by construction and so is
    reported separately rather than counted as a timed flag -- quoting it
    as one would turn "we eventually noticed" into "we caught it early".
    """
    scope_rows = rows[rows["scope"] == scope]
    finite = scope_rows[scope_rows["T"].notna()].sort_values("T")
    horizons = [
        {
            "T": int(r.T),
            "label": f"T={int(r.T)}m",
            "flagged": bool(r.flagged),
        }
        for r in finite.itertuples(index=False)
    ]
    full_rows = scope_rows[scope_rows["T"].isna()]
    flagged_full_window = bool(full_rows["flagged"].iloc[0]) if not full_rows.empty else False
    horizons.append(
        {"T": None, "label": "full window", "flagged": flagged_full_window}
    )

    # The grid is read off the table, so extending the horizons in ml/
    # widens this sentence without anyone editing it here.
    grid_text = ", ".join(f"T={h['T']}m" for h in horizons if h["T"] is not None)

    hit = finite[finite["flagged"]]
    if hit.empty:
        # Two different outcomes, never one phrase. "It eventually fired,
        # days after cash-out" and "it never fired" are not the same
        # claim, and neither of them is a catch.
        full_window_only = flagged_full_window
        return {
            "scope": scope,
            "horizons": horizons,
            "first_flag_T": None,
            "first_flag_at": None,
            "flag_offset_minutes": None,
            "before_exit": None,
            "flagged_full_window": flagged_full_window,
            "status": (
                "flagged_full_window_only" if full_window_only else "never_flagged"
            ),
            "status_label": (
                FULL_WINDOW_ONLY_LABEL if full_window_only else NEVER_FLAGGED_LABEL
            ),
            "caught_in_time": False,
            "detail": (
                (
                    f"Not flagged at any measured horizon ({grid_text}). It does flag "
                    "over the full window, but the full window is the whole dataset, "
                    f"observed days after this ring's exit at T="
                    f"{exit_offset_minutes:.0f}m. That is a detection, not a catch."
                )
                if full_window_only
                else (
                    f"Not flagged at any measured horizon ({grid_text}), and not over "
                    "the full window either. This scope never flags this ring at all."
                )
            ),
        }
    first_t = int(hit["T"].iloc[0])
    before_exit = first_t <= exit_offset_minutes
    if before_exit:
        timing = f"{exit_offset_minutes - first_t:.1f} min BEFORE the exit"
    else:
        # A flag after the exit is a detection, not a save, and has to
        # read as one.
        timing = f"{first_t - exit_offset_minutes:.1f} min AFTER the money left"
    return {
        "scope": scope,
        "horizons": horizons,
        "first_flag_T": first_t,
        "first_flag_at": _iso(onset + pd.Timedelta(minutes=first_t)),
        "flag_offset_minutes": float(first_t),
        "before_exit": before_exit,
        "flagged_full_window": flagged_full_window,
        "status": "flags_before_exit" if before_exit else "flags_after_exit",
        "status_label": f"flagged at T={first_t}m, {timing}",
        "caught_in_time": before_exit,
        "detail": (
            ""
            if before_exit
            else (
                f"This scope did flag, at T={first_t}m, but the money had already "
                f"reached the exit account at T={exit_offset_minutes:.0f}m. A flag "
                "after the exit is a detection, not a save."
            )
        ),
    }


def _ring_timing(
    map_row: pd.Series | None,
    timeline: pd.DataFrame,
    grid: list[int],
) -> dict[str, Any] | None:
    """Onset, exit and per-scope flag verdicts for one detected cluster.

    Split out from :func:`_build_replay` because the demo picks must be
    chosen over ALL published clusters, not just the ones this snapshot
    happens to hold -- otherwise the ring that best makes the argument
    can be excluded by a size-ranked selection that knows nothing about
    detection timing, and the demo silently settles for second best.
    Needs no transfers, so it can run before any ring is built.
    """
    if map_row is None or pd.isna(map_row["true_ring_id"]):
        return None
    rows = timeline[timeline["ring_id"] == str(map_row["true_ring_id"])]
    if rows.empty:
        return None
    onset = pd.Timestamp(rows["onset_at"].iloc[0])
    exit_at = pd.Timestamp(rows["exit_reached_at"].iloc[0])
    exit_offset = (exit_at - onset).total_seconds() / 60.0
    return {
        "true_ring_id": str(map_row["true_ring_id"]),
        "onset": onset,
        "exit_at": exit_at,
        "exit_offset": exit_offset,
        "scopes": [
            _scope_replay(rows, scope, onset, exit_offset)
            for scope in ("single_bank", "cross_bank")
        ],
    }


def _build_replay(
    detected_ring_id: str,
    map_row: pd.Series | None,
    timing: dict[str, Any] | None,
    grid: list[int],
    subgraph_transfers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """One ring's replay payload, or an honest reason there is none.

    Three distinct absences, each named rather than rendered as a blank
    panel: the cluster is not in this snapshot, the cluster matched no
    true ring at all (a false positive, so there is no ground-truth
    timeline to replay), or the timeline published no rows for the ring
    it did match.
    """
    base: dict[str, Any] = {
        "detected_ring_id": detected_ring_id,
        "label": REPLAY_LABEL,
        "available": False,
        "unavailable_reason": "",
        "true_ring_id": None,
        "ambiguity": None,
        "onset_at": None,
        "exit_reached_at": None,
        "exit_offset_minutes": None,
        "exit_label": EXIT_LABEL,
        "horizon_grid": grid,
        "scopes": [],
        "hops": [],
        "hop_count": 0,
        "hops_before_window": 0,
        "hops_after_window": 0,
        "last_subgraph_hop_offset_minutes": None,
        "window_minutes": None,
        "window_note": "",
    }
    if map_row is None:
        base["unavailable_reason"] = (
            f"{detected_ring_id} is not in ml/out/ring_id_map.parquet, so there is "
            "no ground-truth ring to replay it against."
        )
        return base
    if pd.isna(map_row["true_ring_id"]):
        base["unavailable_reason"] = (
            "No replay available: this detected cluster overlaps no true ring at "
            "all. It is a false-positive cluster, so there is no ground-truth "
            "onset, exit or flag timeline to replay. The detection itself is still "
            "shown on the other tabs - what is missing is the ground truth, not "
            "the finding."
        )
        base["ambiguity"] = _ambiguity(map_row)
        return base

    true_ring_id = str(map_row["true_ring_id"])
    base["true_ring_id"] = true_ring_id
    base["ambiguity"] = _ambiguity(map_row)

    if timing is None:
        base["unavailable_reason"] = (
            f"No replay available: ml/out/ring_detection_timeline.parquet has no "
            f"rows for {true_ring_id}, the true ring this cluster matched."
        )
        return base
    if subgraph_transfers is None:
        base["unavailable_reason"] = (
            f"No replay available: {detected_ring_id} is outside this snapshot "
            "(the snapshot holds a span-balanced subset of the detector's "
            "clusters), so its hops were never built. Rebuild with a larger "
            "--rings budget to replay it."
        )
        return base

    onset = timing["onset"]
    exit_at = timing["exit_at"]
    exit_offset = timing["exit_offset"]
    scopes = timing["scopes"]

    # The replay window runs from onset to whichever comes last: the exit,
    # a flag, or the final measured horizon. That is the window the flags
    # are decisions about. It is NOT the cluster's whole life: a cluster's
    # members go on transacting for weeks, and stretching the axis over
    # that compresses the part being argued about into a pixel.
    window_minutes = max(
        [exit_offset, float(grid[-1])]
        + [s["flag_offset_minutes"] for s in scopes if s["flag_offset_minutes"] is not None]
    )

    # Every hop in the ring subgraph, not just the intra-cluster ones: the
    # account that FUNDS a mule is usually unflagged and so is not a
    # cluster member, and dropping its hop would show money appearing from
    # nowhere. Each hop says which it is.
    offsets = [
        (
            round((pd.Timestamp(t["timestamp"]) - onset).total_seconds() / 60.0, 2),
            t,
        )
        for t in subgraph_transfers
    ]
    hops = [
        {
            "index": i,
            "transfer_id": t["transfer_id"],
            "from_hashed_account_id": t["from_hashed_account_id"],
            "to_hashed_account_id": t["to_hashed_account_id"],
            "amount": t["amount"],
            "currency": t["currency"],
            "timestamp": t["timestamp"],
            "offset_minutes": offset,
            "from_bank": t["from_bank"],
            "to_bank": t["to_bank"],
            "cross_bank": t["cross_bank"],
            "intra_ring": t["intra_ring"],
        }
        for i, (offset, t) in enumerate(
            sorted(
                (p for p in offsets if 0.0 <= p[0] <= window_minutes),
                key=lambda p: p[1]["timestamp"],
            ),
            start=1,
        )
    ]
    before = sum(1 for offset, _ in offsets if offset < 0.0)
    after = sum(1 for offset, _ in offsets if offset > window_minutes)
    last_offset = max((offset for offset, _ in offsets), default=0.0)

    # Say out loud what the window leaves out. A viewer who is shown 13
    # hops out of 173 and not told so has been handed a filtered trail.
    parts: list[str] = []
    if before:
        parts.append(
            f"{before} earlier hop{'s' if before > 1 else ''} in this subgraph predate "
            f"{true_ring_id}'s onset"
            + (
                " - this cluster mixes several true rings, so some of its money was "
                "already moving"
                if bool(map_row["match_is_ambiguous"])
                else ""
            )
        )
    if after:
        parts.append(
            f"{after} later hop{'s' if after > 1 else ''} fall after the window, out to "
            f"{last_offset / 1440:.1f} days after onset"
        )
    window_note = ""
    if parts:
        window_note = (
            f"The replay covers onset to T={window_minutes:.0f}m, the window the "
            "flag decisions are about. "
            + "; ".join(parts)
            + ". Those are not shown here and are not part of the flag comparison."
        )

    base.update(
        {
            "available": True,
            "unavailable_reason": "",
            "onset_at": _iso(onset),
            "exit_reached_at": _iso(exit_at),
            "exit_offset_minutes": round(exit_offset, 2),
            "scopes": scopes,
            "hops": hops,
            "hop_count": len(hops),
            "hops_before_window": before,
            "hops_after_window": after,
            "last_subgraph_hop_offset_minutes": round(last_offset, 2),
            "window_minutes": round(window_minutes, 2),
            "window_note": window_note,
        }
    )
    return base


#: The primary pick's criterion, as a property of three published
#: numbers rather than of a status string. Stated in words: *cross-bank's
#: first flag lands before the money reaches the exit account, and
#: single-bank's first flag does not*.
#:
#: This is deliberately not "single-bank flags only at the full window".
#: That earlier wording was satisfied by no openable cluster at all, so it
#: measured nothing and every pick wore a failure badge. This wording is
#: satisfied by clusters and failed by clusters: it is failed whenever
#: cross-bank is also late, and failed whenever single-bank is also in
#: time -- which is most rings. What it asserts is exactly the thing worth
#: demonstrating, that the cross-bank scope buys warning *while the money
#: is still there*, and it is re-derivable from the timeline by anyone who
#: doubts it.
def _cross_bank_alone_in_time(
    cross_bank_first_flag_T: int | None,
    single_bank_first_flag_T: int | None,
    exit_offset_minutes: float,
) -> bool:
    """Does cross-bank flag before the exit while single-bank does not?

    A ``None`` first flag means the scope never fired at any horizon in
    the measured grid. For cross-bank that fails the criterion; for
    single-bank it satisfies the second half, because a scope that never
    flags certainly did not flag in time.
    """
    if cross_bank_first_flag_T is None or cross_bank_first_flag_T >= exit_offset_minutes:
        return False
    return (
        single_bank_first_flag_T is None
        or single_bank_first_flag_T > exit_offset_minutes
    )


def _pick_demo(
    clusters: list[dict[str, Any]],
    timings: dict[str, dict[str, Any] | None],
    ambiguous_of: dict[str, bool],
    grid: list[int],
) -> dict[str, Any]:
    """Choose the demo rings from the data, never by hard-coded id.

    A hard-coded "best ring" breaks silently the moment ``ml/`` is
    regenerated and is indistinguishable from cherry-picking. Every pick
    is derived here from ``ring_id_map`` and ``ring_detection_timeline``,
    every pick is restricted to clusters whose detected->true match is
    NOT ambiguous, and every pick reports whether its stated criterion
    was actually met.

    Selection runs over ALL published clusters, not only the ones a
    size-ranked snapshot happened to keep; ``pinned_ring_ids`` then forces
    the winners into that snapshot. Choosing from the subset instead would
    let ring size quietly overrule detection timing.

    Three picks, because they make three different points:

    ``primary``
        A multi-bank ring the cross-bank scope catches before the money
        leaves and the single-bank scope does not catch in time. This is
        the sharpest honest demo: single-bank may still flag eventually,
        but only after the money is gone. See
        :func:`_cross_bank_alone_in_time` for the criterion.
    ``secondary``
        A multi-bank ring both scopes catch in time, cross-bank first.
        Kept as a lead-time illustration -- and labelled as the exception
        it is, because across the table the median lead is zero and the
        real advantage is coverage, not speed.
    ``control``
        A one-bank ring both scopes flag at the same horizon. One bank
        already sees the whole ring, so joining up should buy nothing.
    """
    eligible = [
        c
        for c in clusters
        if timings.get(c["ring_id"]) is not None and not ambiguous_of[c["ring_id"]]
    ]

    def scopes_of(ring_id: str) -> dict[str, dict[str, Any]]:
        return {s["scope"]: s for s in timings[ring_id]["scopes"]}

    def entry(cluster: dict[str, Any] | None, **extra: Any) -> dict[str, Any] | None:
        if cluster is None:
            return None
        timing = timings[cluster["ring_id"]]
        scopes = scopes_of(cluster["ring_id"])
        return {
            "ring_id": cluster["ring_id"],
            "true_ring_id": timing["true_ring_id"],
            "span": cluster["span"],
            "banks_involved": cluster["banks_involved"],
            "exit_offset_minutes": round(timing["exit_offset"], 2),
            "single_bank_first_flag_T": scopes["single_bank"]["first_flag_T"],
            "cross_bank_first_flag_T": scopes["cross_bank"]["first_flag_T"],
            "single_bank_status": scopes["single_bank"]["status"],
            "cross_bank_status": scopes["cross_bank"]["status"],
            **extra,
        }

    grid_text = "/".join(str(t) for t in grid)

    # ── primary: cross-bank catches it in time, single-bank does not.
    primary_pool = [
        c
        for c in eligible
        if c["span"] >= 2
        and scopes_of(c["ring_id"])["cross_bank"]["status"] == "flags_before_exit"
        and not scopes_of(c["ring_id"])["single_bank"]["caught_in_time"]
    ]

    #: How badly single-bank misses, best demo first. A scope that only
    #: ever flags at the full window is the strongest case; a scope that
    #: flags inside the grid but after cash-out is ranked by how late.
    _MISS_RANK = {
        "flagged_full_window_only": 0,
        "never_flagged": 1,
        "flags_after_exit": 2,
    }

    def primary_rank(cluster: dict[str, Any]) -> tuple[int, float, int, int, str]:
        scopes = scopes_of(cluster["ring_id"])
        single = scopes["single_bank"]
        exit_offset = timings[cluster["ring_id"]]["exit_offset"]
        lateness = (
            single["first_flag_T"] - exit_offset
            if single["first_flag_T"] is not None
            else 0.0
        )
        return (
            _MISS_RANK[single["status"]],
            -lateness,
            -cluster["span"],
            -cluster["alertable_member_count"],
            # Deterministic last resort: the pick never depends on row order.
            cluster["ring_id"],
        )

    primary_cluster = min(primary_pool, key=primary_rank) if primary_pool else None
    primary = None
    if primary_cluster is not None:
        scopes = scopes_of(primary_cluster["ring_id"])
        single, cross = scopes["single_bank"], scopes["cross_bank"]
        # The same rounded value the pick publishes, so anyone recomputing
        # ``criterion_met`` from the response gets the same answer.
        exit_offset = round(timings[primary_cluster["ring_id"]]["exit_offset"], 2)
        met = _cross_bank_alone_in_time(
            cross["first_flag_T"], single["first_flag_T"], exit_offset
        )
        if single["first_flag_T"] is None:
            single_clause = (
                f"single-bank is {FULL_WINDOW_ONLY_LABEL} -- it never flags at any "
                f"horizon in the {grid_text} minute grid"
                if single["status"] == "flagged_full_window_only"
                else (
                    f"single-bank never flags at any horizon in the {grid_text} "
                    "minute grid"
                )
            )
        else:
            single_clause = (
                f"single-bank does not flag until T={single['first_flag_T']}m, "
                f"{single['first_flag_T'] - exit_offset:.1f} min AFTER the money left"
            )
        if met:
            claim = (
                f"Cross-bank flags this ring at T={cross['first_flag_T']}m, "
                f"{exit_offset - cross['first_flag_T']:.1f} min before the money "
                f"reaches the exit account at T={exit_offset:.1f}m -- and "
                f"{single_clause}. Same accounts, same horizons; the only difference "
                "is how much of the ring the scope can see."
            )
        else:
            claim = (
                f"Cross-bank {cross['status_label']} and {single_clause}, but the "
                f"published horizons do not bracket the exit at T={exit_offset:.1f}m "
                "the way the criterion requires."
            )
        primary = entry(
            primary_cluster,
            role="primary",
            criterion=(
                "multi-bank ring, match not ambiguous, cross-bank's first flag lands "
                "before the money reaches the exit account and single-bank's first "
                "flag lands after it (or single-bank never flags at all)"
            ),
            criterion_met=met,
            claim=claim,
            honest_note=(
                (
                    "Single-bank is not blind to this ring -- it does flag it, and "
                    "the timeline shows when. What it cannot do is flag it in time, "
                    "and that is the whole claim: warning while the money is still "
                    "recoverable, not detection eventually. Two things this is NOT. "
                    "It is not a speed claim: across all rings both scopes flag, the "
                    "median lead is zero, and the metrics tab quotes coverage "
                    "instead. And it is not the starkest shape available in "
                    "principle -- no openable cluster has single-bank silent at "
                    "every measured horizon -- so this is one real cluster's real "
                    "behaviour rather than the best story that could be told."
                )
                if met
                else (
                    "The timeline's own numbers do not satisfy the stated criterion: "
                    "cross-bank's first flag and single-bank's first flag do not "
                    "bracket the exit. The criterion was NOT loosened to make this "
                    "ring fit -- the closest available cluster is shown and labelled "
                    "as not meeting it."
                )
            ),
        )

    # ── secondary: both scopes catch it, cross-bank first.
    secondary_pool = [
        c
        for c in eligible
        if c["span"] >= 2
        and c["ring_id"] != (primary_cluster or {}).get("ring_id")
        and scopes_of(c["ring_id"])["cross_bank"]["status"] == "flags_before_exit"
        and scopes_of(c["ring_id"])["single_bank"]["status"] == "flags_before_exit"
        and scopes_of(c["ring_id"])["single_bank"]["first_flag_T"]
        > scopes_of(c["ring_id"])["cross_bank"]["first_flag_T"]
    ]

    def secondary_rank(cluster: dict[str, Any]) -> tuple[int, int, str]:
        scopes = scopes_of(cluster["ring_id"])
        lead = scopes["single_bank"]["first_flag_T"] - scopes["cross_bank"]["first_flag_T"]
        return (-lead, -cluster["alertable_member_count"], cluster["ring_id"])

    secondary_cluster = min(secondary_pool, key=secondary_rank) if secondary_pool else None
    secondary = None
    if secondary_cluster is not None:
        scopes = scopes_of(secondary_cluster["ring_id"])
        single, cross = scopes["single_bank"], scopes["cross_bank"]
        lead = single["first_flag_T"] - cross["first_flag_T"]
        secondary = entry(
            secondary_cluster,
            role="secondary",
            criterion=(
                "multi-bank ring, match not ambiguous, both scopes flag before the "
                "exit with cross-bank first"
            ),
            criterion_met=True,
            claim=(
                f"Both scopes catch this one in time, cross-bank first: T="
                f"{cross['first_flag_T']}m against T={single['first_flag_T']}m, a "
                f"{lead}-minute lead, with the exit at T="
                f"{timings[secondary_cluster['ring_id']]['exit_offset']:.0f}m."
            ),
            honest_note=(
                "This is an illustration, not the headline. Across the rings both "
                "scopes flag, the MEDIAN lead is zero -- most tie at the same "
                "horizon. Cross-bank is not systematically faster; it is "
                "systematically broader. The metrics tab quotes the coverage gap, "
                "which is the claim that holds."
            ),
        )

    # ── control: a 1-bank ring where both scopes flag at the same time.
    control_pool = [
        c
        for c in eligible
        if c["span"] == 1
        and scopes_of(c["ring_id"])["single_bank"]["first_flag_T"] is not None
        and scopes_of(c["ring_id"])["cross_bank"]["first_flag_T"] is not None
    ]

    def control_rank(cluster: dict[str, Any]) -> tuple[int, int, str]:
        scopes = scopes_of(cluster["ring_id"])
        gap = abs(
            scopes["single_bank"]["first_flag_T"] - scopes["cross_bank"]["first_flag_T"]
        )
        return (gap, -cluster["alertable_member_count"], cluster["ring_id"])

    control_cluster = min(control_pool, key=control_rank) if control_pool else None
    control = None
    if control_cluster is not None:
        scopes = scopes_of(control_cluster["ring_id"])
        gap = abs(
            scopes["single_bank"]["first_flag_T"] - scopes["cross_bank"]["first_flag_T"]
        )
        met = gap == 0
        control = entry(
            control_cluster,
            role="control",
            criterion=(
                "1-bank ring, match not ambiguous, both scopes first flag at the "
                "same measured horizon"
            ),
            criterion_met=met,
            claim=(
                f"Both scopes first flag at T="
                f"{scopes['single_bank']['first_flag_T']}m. One bank already sees "
                "this whole ring, so joining up buys nothing here -- which is the "
                "control the multi-bank rings are read against."
                if met
                else (
                    f"Single-bank first flags at T="
                    f"{scopes['single_bank']['first_flag_T']}m against cross-bank "
                    f"T={scopes['cross_bank']['first_flag_T']}m: {gap} minutes apart, "
                    "not the same time. This is the closest 1-bank ring available."
                )
            ),
            honest_note=(
                f"The horizons are a {grid_text} minute grid, so 'the same time' "
                "means the same earliest horizon; simultaneity finer than that is "
                "not measured. This is one ring, not the span-1 average on the "
                "metrics tab."
                if met
                else (
                    "No non-ambiguous 1-bank cluster has both scopes flagging at the "
                    "same horizon. The criterion was NOT loosened to make one fit; "
                    "the closest is shown and labelled as such."
                )
            ),
        )

    picks = [p for p in (primary, secondary, control) if p]
    return {
        "label": REPLAY_LABEL,
        "primary": primary,
        "secondary": secondary,
        "control": control,
        "eligible_ring_count": len(eligible),
        "primary_candidates": len(primary_pool),
        "secondary_candidates": len(secondary_pool),
        "control_candidates": len(control_pool),
        "pinned_ring_ids": [p["ring_id"] for p in picks],
        "selection_note": (
            "Every pick is derived from ml/out/ring_id_map.parquet and "
            "ml/out/ring_detection_timeline.parquet at snapshot-build time, over all "
            "published clusters rather than the snapshot subset. No ring id is "
            "hard-coded, and only clusters whose detected->true match is not "
            "ambiguous are eligible, so each caption can name one true ring honestly."
        ),
    }


def _select_rings(
    rings: pd.DataFrame, limit: int, always_include: list[str] | None = None
) -> list[str]:
    """Pick a balanced set: the largest rings of each bank-span.

    ``always_include`` is reserved for clusters that must be selectable
    whatever their size -- specifically the ones matching no true ring at
    all. A false positive the operator cannot open is a false positive the
    demo never has to admit to.
    """
    sized = rings.assign(
        span=rings["banks_involved"].apply(len),
        # Size rings by the ALERTABLE population: a ring padded with
        # same-bank context members is not a bigger detection.
        members=rings["members"].apply(
            lambda ms: sum(1 for m in ms if m["alertable"])
        ),
    )
    pinned = [r for r in (always_include or []) if r in set(rings["ring_id"])]
    per_span = max(1, limit // 3)
    chosen: list[str] = []
    for span in (3, 2, 1):
        block = sized[sized["span"] == span].nlargest(per_span, "members")
        chosen.extend(block["ring_id"].tolist())
    return pinned + [r for r in chosen if r not in set(pinned)][: limit - len(pinned)]


def dataset_scope_counts(accounts: pd.DataFrame, transfers: pd.DataFrame) -> dict[str, Any]:
    """The population numbers the header and the enclave card quote, side by side.

    ``accounts_generated`` is every account the simulator minted;
    ``accounts_with_activity`` is those with at least one transfer;
    ``edges_total`` is every transfer. The enclave only ever sees edges that
    cross between two banks (``enclave/graph.py`` builds from cross-bank
    transfers), so ``edges_in_enclave_scope`` and ``accounts_in_enclave_scope``
    count those, and an account whose only activity is same-bank is outside
    it. That is why 30,000 accounts and 29,973 accounts are both correct.
    """
    bank = accounts.set_index("hashed_account_id")["bank_id"]
    src, dst = transfers["from_hashed_account_id"], transfers["to_hashed_account_id"]
    cross = transfers.loc[src.map(bank) != dst.map(bank)]
    return {
        "accounts_generated": int(len(accounts)),
        "accounts_with_activity": int(len(set(src) | set(dst))),
        "edges_total": int(len(transfers)),
        "edges_in_enclave_scope": int(len(cross)),
        "accounts_in_enclave_scope": int(
            len(set(cross["from_hashed_account_id"]) | set(cross["to_hashed_account_id"]))
        ),
        "scope_reason": "enclave_sees_cross_bank_edges_only",
    }


def _data_basis(cfg: UiConfig) -> dict[str, Any]:
    """Generation time and seed of the ``ml/out`` manifest this build reads."""
    path = cfg.ml_out_dir / "manifest.json"
    if not path.exists():
        return {"ml_manifest_generated_at": None, "score_seed": None}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ml_manifest_generated_at": manifest.get("generated_at"),
        "score_seed": manifest.get("settings", {}).get("score_seed"),
    }


def _activity_counts(accounts: Any, transfers: Any) -> dict[str, int]:
    """The three account and edge counts that read as one number elsewhere.

    ``accounts_generated`` is every account the simulator wrote.
    ``accounts_with_cross_bank_activity`` is the accounts that appear on at
    least one transfer whose two ends sit at different banks, which is exactly
    the set of accounts the joined graph holds, because the joined graph is
    built from boundary-crossing transfers only. ``cross_bank_transfers`` is
    those transfers: the graph's edge count. An account that only ever moves
    money inside its own bank is generated but never joins the graph, so the
    second number can be smaller than the first without anything being lost.
    """
    bank_of = dict(zip(accounts["hashed_account_id"], accounts["bank_id"], strict=True))
    src = transfers["from_hashed_account_id"]
    dst = transfers["to_hashed_account_id"]
    crossing = src.map(bank_of) != dst.map(bank_of)
    touched = set(src[crossing]) | set(dst[crossing])
    return {
        "accounts_generated": int(len(accounts)),
        "accounts_with_cross_bank_activity": len(touched),
        "cross_bank_transfers": int(crossing.sum()),
    }


def build_ring_race(cfg: UiConfig) -> dict[str, Any]:
    """The label-free race artifact the agents service's ``get_ring_race`` reads.

    One entry per DETECTED cluster, for all of them: unlike the snapshot's
    replays, which exist only for the rings it chose to build hops for, the
    timing needs no transfers (see :func:`_ring_timing`). Reuses
    :func:`_ring_timing` and :func:`_ambiguity`'s source column, so the
    numbers are the ones the replays carry; the arithmetic comparing the two
    scopes is ``shared/ring_race.py``, shared with ``ui/replay_index.py``.

    Why an artifact built here rather than computed in the agents service at
    start-up: the timeline is keyed by the simulator's ground-truth ring ids,
    and the agents image should never hold that id space next to the detector's
    (``agents/hidden_labels.py``). The artifact is keyed by detected id only.
    """
    timeline = pd.read_parquet(cfg.ml_table_path("ring_detection_timeline"))
    ring_map = pd.read_parquet(cfg.ml_table_path("ring_id_map"))
    rings = pd.read_parquet(cfg.ml_table_path("rings"))
    grid = _horizon_grid(timeline)
    map_rows = {row["detected_ring_id"]: row for _, row in ring_map.iterrows()}
    entries: dict[str, dict[str, Any]] = {}
    for ring_id in sorted(set(rings["ring_id"]) | set(map_rows)):
        map_row = map_rows.get(ring_id)
        timing = _ring_timing(map_row, timeline, grid)
        entries[ring_id] = ring_race.entry(
            timing["exit_offset"] if timing else None,
            timing["scopes"] if timing else None,
            bool(map_row["match_is_ambiguous"]) if map_row is not None else None,
        )
    return ring_race.build_artifact(
        entries,
        horizon_grid=grid,
        label=REPLAY_LABEL,
        source="ml/out ring_detection_timeline joined through ring_id_map; detected ids only",
    )


def build(cfg: UiConfig, ring_limit: int) -> dict[str, Any]:
    """Build the full snapshot payload."""
    tables = _load(cfg)
    accounts = tables["accounts"]
    transfers = tables["transfers"]
    ground_truth = tables["ground_truth"]
    rings = tables["rings"]
    scores = tables["scores"]
    explanations = tables["ring_explanations"].set_index("ring_id")
    evidence = tables["account_evidence"]
    timeline = tables["ring_detection_timeline"]
    ring_map = tables["ring_id_map"]
    cases = _load_cases(cfg)
    str_reports = _load_str_reports(cfg)

    bank_of = dict(zip(accounts["hashed_account_id"], accounts["bank_id"], strict=True))
    truth = dict(zip(ground_truth["hashed_account_id"], ground_truth["is_mule"], strict=True))
    roles = dict(zip(ground_truth["hashed_account_id"], ground_truth["role"], strict=True))

    # ── bank tiles: each bank sees only its own accounts, ranked by the
    # single-bank (sb_) out-of-fold score in ml/out/bank_alert_queues.parquet,
    # and only the transfers it is a party to. The joint/cross-bank views keep
    # reading `scores` (cb_) below; nothing here touches them.
    banks_payload = _single_bank_tiles(tables["bank_alert_queues"], transfers, truth)

    # ── timings and demo picks come FIRST, before any ring is built.
    # The horizon grid is read off the timeline table, so extending the
    # horizons in ml/ widens every label and tick here with no edit.
    grid = _horizon_grid(timeline)
    map_rows = {row["detected_ring_id"]: row for _, row in ring_map.iterrows()}
    all_ring_ids = sorted(set(rings["ring_id"]) | set(map_rows))
    timings = {
        ring_id: _ring_timing(map_rows.get(ring_id), timeline, grid)
        for ring_id in all_ring_ids
    }
    ambiguous_of = {
        ring_id: bool(
            map_rows[ring_id]["match_is_ambiguous"] if ring_id in map_rows else False
        )
        for ring_id in all_ring_ids
    }
    clusters = [
        {
            "ring_id": row["ring_id"],
            "span": len(row["banks_involved"]),
            "banks_involved": list(row["banks_involved"]),
            "alertable_member_count": sum(1 for m in row["members"] if m["alertable"]),
        }
        for _, row in rings.iterrows()
    ]
    demo = _pick_demo(clusters, timings, ambiguous_of, grid)

    # ── rings. Two sets are pinned into the snapshot regardless of size:
    # the demo's own picks, chosen on detection timing rather than member
    # count, and the clusters matching no true ring -- a false positive the
    # operator cannot open is one the demo never has to admit to, and its
    # "no replay available" panel is the only place the absence of ground
    # truth is visible.
    no_true_ring = ring_map[ring_map["true_ring_id"].isna()]["detected_ring_id"].tolist()
    pinned = demo["pinned_ring_ids"] + no_true_ring
    built: list[dict[str, Any]] = []
    for ring_id in _select_rings(rings, min(ring_limit, MAX_RINGS), pinned):
        ring_row = rings[rings["ring_id"] == ring_id].iloc[0]
        explanation_row = (
            explanations.loc[ring_id] if ring_id in explanations.index else None
        )
        payload = _build_ring(
            ring_row,
            explanation_row,
            accounts,
            transfers,
            evidence,
            scores,
            truth,
            roles,
            bank_of,
            cases,
            str_reports,
            cfg,
        )
        if payload:
            built.append(payload)

    # ── replays. Built for EVERY detected cluster, not just the ones in
    # this snapshot, so that asking about a cluster the snapshot left out
    # -- or the one false-positive cluster with no ground truth at all --
    # returns a named reason instead of a missing key.
    transfers_by_ring = {r["ring_id"]: r["transfers"] for r in built}
    replays = {
        ring_id: _build_replay(
            ring_id,
            map_rows.get(ring_id),
            timings[ring_id],
            grid,
            transfers_by_ring.get(ring_id),
        )
        # Derived from the union of both id sources, so neither a cluster
        # missing from the map nor one missing from rings.parquet can
        # quietly vanish from the index.
        for ring_id in all_ring_ids
    }
    # Carry the match verdict onto the ring itself so the ring PICKER can
    # warn before the click, not only after it. Copied from the replay,
    # which copied it from ring_id_map: one source, no second derivation.
    for ring in built:
        replay = replays[ring["ring_id"]]
        ring["true_ring_id"] = replay["true_ring_id"]
        ring["match_is_ambiguous"] = bool(
            replay["ambiguity"]["is_ambiguous"] if replay["ambiguity"] else False
        )
        ring["has_replay"] = replay["available"]

    # The demo auto-selects the ring the replay argument rests on. That
    # pick is derived in ``_pick_demo`` and is restricted to clusters
    # whose true-ring match is not ambiguous; the fallback below is
    # restricted the same way, so no code path can feature a cluster the
    # caption cannot name honestly.
    def striking(ring: dict[str, Any]) -> tuple[int, int, int]:
        return (ring["span"], len(ring["members"]), ring["cross_bank_transfer_count"])

    unambiguous = [
        r
        for r in built
        if replays[r["ring_id"]]["available"]
        and not replays[r["ring_id"]]["ambiguity"]["is_ambiguous"]
    ]
    if demo["primary"]:
        featured = demo["primary"]["ring_id"]
    elif unambiguous:
        featured = max(unambiguous, key=striking)["ring_id"]
    else:  # pragma: no cover - needs an ml/out where every match is ambiguous
        featured = ""
    span_counts = rings["banks_involved"].apply(len).value_counts().to_dict()

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "ml/out (velocity rule v2 + XGBoost OOF scores) over simulator/out seed 42",
        # Which ml/out run this snapshot was built from, so /api/meta can tell
        # when a bundled metrics file predates it even where ml/out is absent.
        "data_basis": _data_basis(cfg),
        "dataset": {
            "accounts": int(len(accounts)),
            "transfers": int(len(transfers)),
            "detected_rings_total": int(len(rings)),
            "rings_in_snapshot": len(built),
            "banks": sorted(set(accounts["bank_id"])),
            **dataset_scope_counts(accounts, transfers),
            **_activity_counts(accounts, transfers),
        },
        "detector": {
            "detector_version": _detector_version_of(
                str(rings["ring_id"].iloc[0])
            ),
            "score_model_version": str(scores["model_version"].iloc[0]),
            "rings_found": int(len(rings)),
            "ring_span_counts": {str(k): int(v) for k, v in sorted(span_counts.items())},
            "caveat": (
                "45 rings is not '45 of 120 true rings found'. They touch 53 of "
                "the 120 true rings and 88.9% of their members are true positives. "
                "The rule flags 676 accounts at 87.8% account-level recall, but a "
                "flagged mule only joins a ring here when the collector it forwards "
                "to was flagged too - so ring-level coverage is weaker than "
                "account-level recall. That gap is the clustering step, not the "
                "detector."
            ),
        },
        "bounding": {
            "max_graph_nodes": cfg.max_graph_nodes,
            "graph_hops": cfg.graph_hops,
            "max_transfers_per_ring": max(
                (len(r["transfers"]) for r in built), default=0
            ),
            "max_context_edges": MAX_CONTEXT_EDGES,
            "note": (
                "Cytoscape receives one ring subgraph at a time: ring members "
                f"plus their {cfg.graph_hops}-hop counterparties, capped at "
                f"{cfg.max_graph_nodes} nodes. Intra-ring edges are kept in full "
                "so the money trail is never truncated; context edges get a "
                f"separate budget of {MAX_CONTEXT_EDGES}, largest amount first, "
                "so the reported edge maximum is measured, not promised, "
                "and counterparties left with no edge are dropped rather than "
                "drawn as isolated dots."
            ),
        },
        "cases": {
            "source": str(cfg.agents_table_path("cases").name),
            "available": bool(cases),
            "rings_with_verified_case": len(cases),
            "note": (
                "A case exists only where an investigation's claims were "
                "re-derived and matched by the verifier. A ring with no row "
                "here is shown as 'no case yet'; no placeholder case is ever "
                "created."
            ),
        },
        "str_reports": {
            "source": str(cfg.agents_table_path("str_reports").name),
            "available": bool(str_reports),
            "cases_with_real_draft": len(str_reports),
            "note": (
                "A real Gemini-drafted STR, cited against the live RBI/PMLA/"
                "FIU-IND regulation index (agents/batch_draft_reports.py), "
                "exists only for cases that batch has drafted for. Every "
                "other case's report falls back to PLACEHOLDER_CITATIONS, "
                "marked citations_verified=False so it is never mistaken "
                "for the real one."
            ),
        },
        "ring_match": {
            "source": "ml/out/ring_id_map.parquet",
            "detected_clusters": int(len(ring_map)),
            "ambiguous_matches": int(ring_map["match_is_ambiguous"].sum()),
            "no_true_ring": int(ring_map["true_ring_id"].isna().sum()),
            "note": (
                "Detector cluster ids and ground-truth ring ids are different id "
                "spaces. The pairing is best-member-overlap, and "
                f"{int(ring_map['match_is_ambiguous'].sum())} of {len(ring_map)} "
                "clusters span more than one true ring, are overlapped by more than "
                "one cluster, or tie - because clusters are connected components "
                "over flagged accounts. Those are never auto-selected, and a manual "
                "selection carries the caveat naming how many true rings are mixed "
                "in."
            ),
        },
        # Ring coverage is NOT derived here. It comes from
        # ml/experiments/lead_time/metrics.json at request time, because
        # the measure that matters -- flagged before the money reached the
        # exit -- is published there and cited by docs/claims.md. Deriving
        # a second, looser version from the timeline parquet is exactly how
        # the page and the claims file would come to disagree.
        "banks": banks_payload,
        "rings": built,
        "replays": replays,
        "demo": demo,
        "featured_ring_id": featured,
        "mock": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the UI demo snapshot.")
    parser.add_argument("--rings", type=int, default=MAX_RINGS, help="rings to include")
    parser.add_argument("--out", type=Path, default=None, help="snapshot output path")
    args = parser.parse_args(argv)

    cfg = UiConfig.from_env()
    out = args.out or cfg.snapshot_path
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = build(cfg, args.rings)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(payload['rings'])} rings)")

    # The race artifact for Ask (agents/out/ring_race.json). Written here
    # because this step already owns the timing logic; the deploy mirrors
    # agents/out to the agents bucket, and infra/agents_fetch.py lists it.
    race = build_ring_race(cfg)
    cfg.ring_race_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.ring_race_path.write_text(json.dumps(race, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {cfg.ring_race_path} ({len(race['rings'])} clusters)")

    # The bundled fallbacks are what the app serves on a machine where
    # ml/ has never run. They are refreshed here, and each one carries the
    # source's own ``generated_at``, so a stale copy announces its age on
    # the page instead of passing for live output.
    cfg.metrics_fallback_path.parent.mkdir(parents=True, exist_ok=True)
    for source, fallback in (
        (cfg.metrics_path, cfg.metrics_fallback_path),
        (cfg.lead_metrics_path, cfg.lead_metrics_fallback_path),
        (cfg.coalition_path, cfg.coalition_fallback_path),
        (cfg.redteam_path, cfg.redteam_fallback_path),
    ):
        if source.exists():
            shutil.copyfile(source, fallback)
            print(f"refreshed {fallback.name} from {source}")
        else:
            print(f"WARNING: {source} is missing; {fallback.name} left as it was")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
