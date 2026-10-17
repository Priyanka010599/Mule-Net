"""Write the output tables other agents consume, validated against ``shared/``.

Five tables land in ``ml/out/``. **Every one of them** conforms to a
model that exists in ``shared/contracts.py`` and is **built by importing
that model**, never by hand-writing field names — so a contract change
breaks this module loudly instead of silently producing a table the UI
cannot read.

What produces what
------------------
Two different things are shipped here and conflating them would mislead
downstream:

``scores.parquet`` (``RiskScore``)
    The **XGBoost baseline's pooled out-of-fold probability**, one row
    per account, all 30,000. Out-of-fold means no account's score came
    from a model that saw its ring, so this column is safe to rank on.
    It is a continuous score, which is what a UI needs to sort an alert
    queue and what ``RiskScore.score`` is typed for.

``rings.parquet`` (``Ring``), ``account_evidence.parquet``
    Driven by the **velocity rule**, which is the shipped detector. The
    rule makes a hard flag; the ring skeleton is the connected components
    of the transfer graph *restricted to flagged accounts*, and evidence
    is attached to those accounts.

Ring membership is two claims, not one
--------------------------------------
``Ring.members`` is a list of ``RingMember`` records, and each says how
that account got into the ring. A ``velocity`` member was flagged by the
detector on its own transaction behaviour. An ``identity_link`` member
was pulled in by **stage-2 propagation** (``ml/stage2.py``) because it
shares a device fingerprint or a hashed KYC id number with a member — a
lead, never a finding, and it carries the counterparty, the evidence
type and a reason sentence saying so.

``alertable`` follows the ruling: velocity always; identity links only
when the link crosses a bank boundary. A **same-bank** identity link is
published as CONTEXT (``alertable=False``), because same-bank identity
reuse is dominated by legitimate customers. An unqualified count of ring
members is therefore *not* an alert count — ``alertable_member_ids`` is.

``account_evidence.parquet`` consequently carries **both scopes**. It
used to be cross-bank only, which left a context member's
``same_bank_*`` link joining to nothing; a consumer following the join
would render the member with its qualifier missing, which is the flat
list problem again. Every member that names a ``link_evidence_type`` now
has a matching row, asserted by ``ml/test_publish.py``.

So the score column and the flag are two different models, deliberately.
A reader who assumes ``score`` ranks the ring members will be roughly
right and occasionally wrong, so ``top_factors`` states which decision
each row came from.

Ring clustering
---------------
Connected components over the high-risk edge set: a transfer whose
**both** endpoints were flagged by the velocity rule. Singletons are
dropped — a "ring" of one account is an alert, not a ring, and emitting
them would swamp the ring table with the rule's false positives. This is
the ``ml-engineer`` brief's "connected components over high-risk edges",
computed with ``scipy.sparse.csgraph`` rather than a bespoke union-find.

``detected_at`` is the feature table's ``as_of`` — the observation
instant the decision was actually made at — not ``datetime.now()``. A
detection timestamp that moves every time the file is rebuilt is not a
detection timestamp.

Per-account explanations
------------------------
``account_explanations.parquet`` exists because a consumer asking "why
this *account*?" previously had nowhere to look: ``ml/`` published ring
explanations only, and ``agents``' ``get_explanation(subject_type=
"account")`` raised ``KeyError`` for every account. Its ``top_factors``
are the account's own **out-of-fold TreeSHAP** attributions, rendered as
sentences quoting the account's own feature values — the model's actual
attribution, from the fold model that had not seen the account's ring.

It is validated against ``shared.contracts.AccountExplanation``, which
the architect has since defined and which is now authoritative. The
first cut of this table was shaped like ``RiskScore`` — a bare
``list[str]`` of factor sentences — and that was wrong in a specific
way: direction was carried **in prose**, as a trailing
``(raises risk)``/``(lowers risk)``. A consumer that did not parse the
suffix would render a *mitigating* factor as incriminating. The
contract's ``ExplanationFactor.contribution`` is a signed float, so the
sign is now typed rather than spelled, ``description`` is the analyst
sentence alone, and ``feature_name`` carries the raw column for
reproducibility (and, per the contract, is never displayed in place of
``description``). ``model_version`` equals ``scores.model_version`` for
the same account: both are the same out-of-fold XGBoost fit, and an
explanation shown next to a score it was not computed from is a wrong
explanation.

Identity evidence
-----------------
``AccountEvidence`` now lives in ``shared/contracts.py`` (promoted by
the architect, with ``scope`` as an ``EvidenceScope`` enum rather than a
bare ``str``); it is imported here, not redefined. The enum is why the
writer below flattens ``Enum`` members before the parquet write.

Type discipline
---------------
``MoneyAmount`` is ``Decimal`` and every datetime in ``shared/ddl.sql``
is ``TIMESTAMPTZ``. None of these four tables carries money, so the
Decimal question does not arise; the datetime one does, and every
timestamp written here is tz-aware UTC. Validation is not a spot check:
**every row** is constructed through its pydantic model, and a sample is
additionally round-tripped through JSON and re-parsed with an equality
assertion, so a field that survives construction but not serialisation
is caught too.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ml.config import MlConfig
from ml.evaluate import SEEDS, attach_labels, fold_assignment, oof_scores_and_contributions
from ml.evidence import EVIDENCE_RECORD_COLUMNS, evidence_flags, evidence_records, top_factors
from ml.features import NO_PASSTHROUGH, feature_columns, full_window_cutoff, load_tables
from ml.rule import (
    MIN_BANKS,
    VELOCITY_MAX_SECONDS,
    VELOCITY_MIN_IN_DEGREE,
    VELOCITY_MIN_OUT_DEGREE,
    VELOCITY_RATIO_RANGE,
    velocity_decision,
)
from shared.contracts import (
    AccountEvidence,
    AccountExplanation,
    EvidenceScope,
    MembershipBasis,
    Ring,
    RingExplanation,
    RingMember,
    RingStatus,
    RiskScore,
)

if TYPE_CHECKING:  # ml.stage2 imports this module; see main().
    from pathlib import Path

    from ml.stage2 import IdentityLink

#: Version string stamped on every score row.
MODEL_VERSION = "mulenet-xgboost-cb-oof-v1"

#: Version string identifying the detector behind rings and evidence.
DETECTOR_VERSION = "mulenet-velocity-rule-v2"

#: Version string for stage-2 identity propagation, distinct from the
#: stage-1 detector so a consumer can tell which produced a membership
#: row or an evidence row. It lives here, beside the other version
#: strings, rather than in ``ml/stage2.py``: ``stage2`` imports this
#: module, so defining it there and importing it back would be a cycle.
STAGE2_VERSION = "mulenet-identity-propagation-v1"

#: Version string stamped on every per-account explanation row.
EXPLAINER_VERSION = "mulenet-xgboost-cb-oof-shap-v1"

#: Field order of the nested ``Ring.members`` struct, read off the
#: contract. Recorded in the manifest because ``manifest.tables.columns``
#: lists top-level columns only, and "members" alone tells a consumer
#: nothing about the struct it has to read.
RING_MEMBER_FIELDS: tuple[str, ...] = tuple(RingMember.model_fields)

#: Rows round-tripped through JSON as an extra serialisation check.
ROUND_TRIP_SAMPLE = 25

#: SHAP factors carried per account. Four is what fits in a UI panel and
#: what an analyst reads before scrolling; the tail is near-zero anyway.
EXPLANATION_FACTORS = 4

#: Feature name (unprefixed) -> sentence quoting the account's own value.
#: Raw column names are not an explanation: "cb_passthrough_ratio=0.97"
#: tells an analyst nothing they can put in a case note.
FACTOR_PHRASES: dict[str, str] = {
    "in_degree": "received {value:,.0f} inbound transfers",
    "out_degree": "sent {value:,.0f} outbound transfers",
    "unique_in_counterparties": "received from {value:,.0f} distinct senders",
    "unique_out_counterparties": "paid out to {value:,.0f} distinct recipients",
    "unique_counterparties": "dealt with {value:,.0f} distinct counterparties",
    "distinct_counterparty_banks": "counterparties sit at {value:,.0f} distinct bank(s)",
    "amount_in_total": "took in {value:,.0f} in total",
    "amount_out_total": "paid out {value:,.0f} in total",
    "passthrough_ratio": "forwarded {value:.0%} of what it received",
    "passthrough_speed_seconds": "forwarded funds a median {value:,.0f}s after receiving them",
    "round_amount_ratio": "{value:.0%} of its transfers were round amounts",
    "account_age_days": "account was {value:,.0f} days old when observed",
    "device_count": "{value:,.0f} device(s) seen on the account",
    "shared_device_count": "shares {value:,.0f} device fingerprint(s) with other accounts",
    "document_reuse_count": "KYC details also appear on {value:,.0f} other account(s)",
    "document_reuse_flag": "KYC details are reused elsewhere",
}


def flatten_enums(value: object) -> object:
    """Replace every ``Enum`` member with its value, at any nesting depth.

    ``Ring.members`` is a list of structs carrying ``MembershipBasis``
    and ``EvidenceScope``, so flattening only the top level of a column
    leaves enum objects inside the nested records. Deliberately blind to
    everything that is not an ``Enum``, ``dict`` or ``list``: a
    ``pd.Timestamp`` must pass through untouched.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: flatten_enums(item) for key, item in value.items()}
    if isinstance(value, list):
        return [flatten_enums(item) for item in value]
    return value


def validate_rows(records: list[dict[str, object]], model: type[BaseModel]) -> pd.DataFrame:
    """Build every row through ``model``, round-trip a sample, return a frame.

    Construction alone would miss a field that validates but cannot
    serialise, so a sample is dumped to JSON and re-parsed with an
    equality assertion before anything is written.
    """
    validated = [model.model_validate(record) for record in records]
    step = max(1, len(validated) // ROUND_TRIP_SAMPLE)
    for item in validated[::step]:
        reparsed = model.model_validate_json(item.model_dump_json())
        assert reparsed == item, f"{model.__name__} did not survive a JSON round trip"
    return pd.DataFrame([item.model_dump() for item in validated])


def build_scores(
    frame: pd.DataFrame, evidence: pd.DataFrame, flagged: np.ndarray, scores: np.ndarray
) -> list[dict]:
    """Out-of-fold XGBoost probabilities for every account, as ``RiskScore`` dicts."""
    scored_at = pd.to_datetime(frame["as_of"], utc=True)

    records: list[dict[str, object]] = []
    for i, account in enumerate(frame["hashed_account_id"]):
        factors = list(top_factors(evidence.loc[account])) if account in evidence.index else []
        if flagged[i]:
            factors.insert(0, f"velocity rule fired ({DETECTOR_VERSION})")
        records.append(
            {
                "hashed_account_id": account,
                "score": float(scores[i]),
                "model_version": MODEL_VERSION,
                "scored_at": scored_at.iloc[i].to_pydatetime(),
                "top_factors": factors[:5],
            }
        )
    return records


def factor_description(name: str, value: float) -> str:
    """One SHAP factor as a sentence an analyst can paste into a case note.

    Direction is deliberately **not** in the text. It used to be, as a
    trailing ``(raises risk)`` / ``(lowers risk)``, because a bare
    ``list[str]`` had nowhere else to put it. ``ExplanationFactor`` has
    a signed ``contribution`` float, so encoding the same fact twice —
    once as prose a consumer must parse, once as a number it can sort on
    — would just be two places for it to disagree.
    """
    if name == "passthrough_speed_seconds" and value == NO_PASSTHROUGH:
        return "no in-then-out hop was observed"
    if name == "document_reuse_flag":
        return "KYC details are reused elsewhere" if value else "no KYC reuse observed"
    return FACTOR_PHRASES[name].format(value=value)


def explanation_narrative(score: float, factors: list[dict[str, object]], flagged: bool) -> str:
    """The paragraph ``AccountExplanation.narrative`` requires, for one account.

    States the score, which detector decision the account is on the
    wrong or right side of, and reads the signed factors back in the two
    directions they actually argue. An STR drafter can quote it; a UI can
    render it without re-deriving anything from ``top_factors``.
    """
    up = [f for f in factors if float(f["contribution"]) > 0]  # type: ignore[arg-type]
    down = [f for f in factors if float(f["contribution"]) <= 0]  # type: ignore[arg-type]
    verdict = (
        f"The velocity rule ({DETECTOR_VERSION}) flagged this account."
        if flagged
        else f"The velocity rule ({DETECTOR_VERSION}) did not flag this account."
    )
    parts = [
        f"Out-of-fold model score {score:.3f} from {MODEL_VERSION}, attributed with "
        f"TreeSHAP by the fold model that had not seen this account's ring. {verdict}"
    ]
    if up:
        parts.append("Raising the score: " + "; ".join(str(f["description"]) for f in up) + ".")
    if down:
        parts.append("Lowering the score: " + "; ".join(str(f["description"]) for f in down) + ".")
    return " ".join(parts)


def build_account_explanations(
    frame: pd.DataFrame,
    scores: np.ndarray,
    contribs: np.ndarray,
    columns: list[str],
    flagged: np.ndarray,
    generated_at: datetime,
) -> list[dict]:
    """Per-account explanations as ``AccountExplanation`` dicts, one per account.

    All 30,000, not just the flagged ones: an analyst clearing an account
    needs the reasons as much as one escalating it, and a consumer that
    has to catch ``KeyError`` for 29,000 of 30,000 subjects is not a
    working tool.

    ``model_version`` is :data:`MODEL_VERSION`, **the same string
    ``scores.parquet`` carries** — the contract requires the two to match
    for an account, and they genuinely do: both come from the same
    out-of-fold XGBoost fit. :data:`EXPLAINER_VERSION` names the
    attribution method and is recorded in ``manifest.json`` instead of
    being smuggled into a field typed for the model.
    """
    names = [column[len("cb_") :] for column in columns]
    values = frame[columns].to_numpy(dtype="float64")
    # Last column of a pred_contribs matrix is the bias term, not a feature.
    order = np.argsort(-np.abs(contribs[:, : len(columns)]), axis=1)

    records: list[dict[str, object]] = []
    for i, account in enumerate(frame["hashed_account_id"]):
        factors: list[dict[str, object]] = [
            {
                "description": factor_description(names[j], values[i, j]),
                "contribution": float(contribs[i, j]),
                # The raw column, for reproducibility only. The contract
                # is explicit that no consumer may render it to an
                # analyst in place of ``description``.
                "feature_name": columns[j],
            }
            for j in order[i, :EXPLANATION_FACTORS]
        ]
        records.append(
            {
                "hashed_account_id": account,
                "score": float(scores[i]),
                "model_version": MODEL_VERSION,
                "top_factors": factors,
                "narrative": explanation_narrative(float(scores[i]), factors, bool(flagged[i])),
                "generated_at": generated_at,
            }
        )
    return records


def high_risk_components(
    transfers: pd.DataFrame, flagged_ids: pd.Index, cutoff: pd.Timestamp
) -> dict[str, list[str]]:
    """Connected components over transfers with both endpoints flagged."""
    position = {account: i for i, account in enumerate(flagged_ids)}
    visible = transfers.loc[transfers["timestamp"] <= cutoff]
    src = visible["from_hashed_account_id"].map(position)
    dst = visible["to_hashed_account_id"].map(position)
    keep = src.notna() & dst.notna()
    rows = src[keep].to_numpy(dtype="int64")
    cols = dst[keep].to_numpy(dtype="int64")
    n = len(flagged_ids)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    _count, labels = connected_components(graph, directed=False)
    members: dict[int, list[str]] = {}
    for account, label in zip(flagged_ids, labels, strict=True):
        members.setdefault(int(label), []).append(account)
    # A component of one is an alert, not a ring.
    return {
        f"ring-{DETECTOR_VERSION}-{label:05d}": sorted(ids)
        for label, ids in sorted(members.items())
        if len(ids) > 1
    }


def build_rings(
    components: dict[str, list[str]],
    added: dict[str, list[IdentityLink]],
    bank: pd.Series,
    detected_at: datetime,
) -> list[dict]:
    """``Ring`` dicts, one per multi-account component, with typed members.

    Two populations, one list, each row saying which it belongs to.

    ``components`` are the **stage-1** connected components: every
    account in them was independently flagged by the velocity rule, so
    each becomes a ``velocity`` member and is alertable. ``added`` is
    stage-2 identity propagation out of those same components, keyed by
    ring id; each of those becomes an ``identity_link`` member carrying
    the ring member it was reached from, the evidence type that links
    them, and a reason written as a lead.

    ``alertable`` follows the ruling and nothing else: velocity always,
    identity links only when the link crosses a bank boundary. It is
    written on the row rather than left to the consumer, because
    re-deriving it means joining evidence and comparing banks, and a
    consumer that gets that wrong publishes an alert against a family
    member.

    ``banks_involved`` covers **all** members, alertable or not: it
    describes the footprint an analyst is looking at, which includes the
    context accounts.
    """
    records: list[dict[str, object]] = []
    for ring_id, ids in components.items():
        members: list[dict[str, object]] = [
            {
                "hashed_account_id": account,
                "membership_basis": MembershipBasis.VELOCITY,
                "alertable": True,
            }
            for account in ids
        ]
        links = added.get(ring_id, [])
        members.extend(
            {
                "hashed_account_id": link.hashed_account_id,
                "membership_basis": MembershipBasis.IDENTITY_LINK,
                "alertable": link.alertable,
                "link_scope": link.scope,
                "linked_from_hashed_account_id": link.linked_from_hashed_account_id,
                "link_evidence_type": link.link_evidence_type,
                "reason": link.reason,
            }
            for link in links
        )
        everyone = list(ids) + [link.hashed_account_id for link in links]
        records.append(
            {
                "ring_id": ring_id,
                "members": members,
                "banks_involved": sorted({str(bank[i]) for i in everyone}),
                "detected_at": detected_at,
                "status": RingStatus.CANDIDATE,
            }
        )
    return records


def build_explanations(
    components: dict[str, list[str]],
    added: dict[str, list[IdentityLink]],
    frame: pd.DataFrame,
    evidence: pd.DataFrame,
    bank: pd.Series,
    generated_at: datetime,
) -> list[dict]:
    """``RingExplanation`` dicts: the quantities the rule actually fired on.

    The narrative quotes the ring's own feature values rather than
    restating the thresholds, so an analyst can see *how far past* the
    bar the ring sits, not merely that it cleared it.

    **Counts here are alertable counts.** A bare "N accounts" in a
    narrative is read as "N accounts we are accusing", so the unqualified
    number is the alertable population — velocity members plus cross-bank
    identity links. Same-bank context members are counted only in their
    own sentence, which names them as context and not as alerts. The
    behavioural medians are computed over the **velocity** members alone
    and say so: an identity-link member was never measured against the
    rule, so folding it into a median of pass-through speed would quote a
    statistic about accounts that statistic does not describe.
    """
    indexed = frame.set_index("hashed_account_id")
    records: list[dict[str, object]] = []
    for ring_id, ids in components.items():
        links = added.get(ring_id, [])
        cross_ids = [link.hashed_account_id for link in links if link.alertable]
        context_ids = [link.hashed_account_id for link in links if not link.alertable]
        alertable_ids = list(ids) + cross_ids
        block = indexed.loc[ids]
        banks = sorted({str(bank[i]) for i in alertable_ids})
        context_banks = sorted({str(bank[i]) for i in context_ids})
        ev = evidence.reindex(alertable_ids).fillna(False)
        carried = [c for c in ev.columns if c.startswith("cross_bank_") and ev[c].any()]
        factors = [
            f"median fan-in {block['cb_in_degree'].median():.0f}"
            f" (rule floor {VELOCITY_MIN_IN_DEGREE})",
            f"median fan-out {block['cb_out_degree'].median():.0f}"
            f" (rule floor {VELOCITY_MIN_OUT_DEGREE})",
            f"median hop {block['cb_passthrough_speed_seconds'].median():.0f}s"
            f" (rule ceiling {VELOCITY_MAX_SECONDS:.0f}s)",
            f"median pass-through ratio {block['cb_passthrough_ratio'].median():.2f}"
            f" (rule band {VELOCITY_RATIO_RANGE[0]}-{VELOCITY_RATIO_RANGE[1]})",
            f"spans {len(banks)} bank(s): {', '.join(banks)}",
            f"membership: {len(ids)} velocity + {len(cross_ids)} cross-bank identity link "
            f"= {len(alertable_ids)} alertable"
            + (f", plus {len(context_ids)} same-bank context (not alerts)" if context_ids else ""),
        ]
        if carried:
            names = ", ".join(c[len("cross_bank_") :] for c in carried)
            factors.append(f"identity evidence: {names}")
        narrative = (
            f"{len(alertable_ids)} alertable accounts across {len(banks)} bank(s) "
            f"({', '.join(banks)}). {len(ids)} of them were independently flagged by the "
            f"velocity rule and form a connected component of the transfer graph"
            + (
                f"; the other {len(cross_ids)} share a device fingerprint or a hashed KYC "
                "ID number with one of those members at a different bank and are leads "
                "that need review, not detections. "
                if cross_ids
                else ". "
            )
            + f"The {len(ids)} velocity members receive a median of "
            f"{block['cb_in_degree'].median():.0f} inbound and "
            f"{block['cb_out_degree'].median():.0f} outbound transfers and forward funds "
            f"in a median of {block['cb_passthrough_speed_seconds'].median():.0f} seconds, "
            f"passing through a median "
            f"{block['cb_passthrough_ratio'].median():.0%} of what they receive. "
            + (
                "Identity reuse across banks is attached as supporting evidence; it did "
                "not contribute to the detection decision. "
                if carried
                else "No cross-bank identity reuse is attached: this ring was detected on "
                "transaction behaviour alone. "
            )
            + (
                f"A further {len(context_ids)} account(s) at "
                f"{', '.join(context_banks)} are linked to a member by same-bank identity "
                "reuse and are published as CONTEXT only: they are not alerts, are not "
                "included in the count above, and carry no accusation."
                if context_ids
                else ""
            )
        ).strip()
        records.append(
            {
                "ring_id": ring_id,
                "top_factors": factors,
                "narrative": narrative,
                "generated_at": generated_at,
            }
        )
    return records


def build_evidence(records: pd.DataFrame, stage1_accounts: set[str]) -> list[dict]:
    """``AccountEvidence`` dicts from the long-format evidence frame.

    ``detector_version`` names the run the row was attached to, so it is
    stamped per row rather than globally: a row on an account the
    velocity rule flagged belongs to :data:`DETECTOR_VERSION`, while a
    row that exists only because stage-2 propagation put the account in a
    ring belongs to :data:`STAGE2_VERSION`. Stamping one string over both
    would claim the stage-1 detector produced evidence about accounts it
    never flagged.
    """
    out: list[dict[str, object]] = []
    for row in records.to_dict("records"):
        account = row["hashed_account_id"]
        out.append(
            {
                **{k: row[k] for k in EVIDENCE_RECORD_COLUMNS},
                "counterparty_hashed_account_ids": list(row["counterparty_hashed_account_ids"]),
                "counterparty_bank_ids": list(row["counterparty_bank_ids"]),
                "observed_at": pd.Timestamp(row["observed_at"]).tz_convert(UTC).to_pydatetime(),
                "detector_version": (
                    DETECTOR_VERSION if account in stage1_accounts else STAGE2_VERSION
                ),
            }
        )
    return out


def register_table(
    out_dir: Path, name: str, table: pd.DataFrame, validated_against: str | None, **entry: object
) -> Path:
    """Write ONE table and add its entry to an existing manifest, touching nothing else.

    Additive counterpart of the loop in :func:`main` (which writes the
    manifest wholesale): read-modify-write of ``manifest.json`` with only
    ``tables[name]`` set, so every other entry and top-level key keeps its
    content and the other parquet files are not opened for writing.
    """
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].map(flatten_enums)
    path = out_dir / f"{name}.parquet"
    table.to_parquet(path, index=False)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["tables"][name] = {
        "rows": len(table),
        "validated_against": validated_against,
        "columns": list(table.columns),
        **entry,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    return path


def main() -> None:
    """Build, validate and write all four output tables plus their README."""
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    cutoff = full_window_cutoff(tables)
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]

    flagged = velocity_decision(frame, "cb_", require_bank_spread=True)
    flagged_ids = pd.Index(frame.loc[flagged, "hashed_account_id"])
    detected_at = pd.to_datetime(frame["as_of"], utc=True).max().to_pydatetime()
    print(f"flagged accounts={len(flagged_ids)} detected_at={detected_at.isoformat()}")

    columns = feature_columns("cb_")
    seed = SEEDS[0]
    scores, contribs = oof_scores_and_contributions(
        frame, columns, seed, fold_assignment(frame, seed)
    )

    evidence = evidence_flags(tables, cutoff)
    components = high_risk_components(tables["transfers"], flagged_ids, cutoff.iloc[0])

    # Imported here, not at module scope: ``ml.stage2`` imports this
    # module for ``high_risk_components`` and the version strings, so a
    # top-level import would be a cycle. By the time main() runs, this
    # module is fully initialised and the import is ordinary.
    from ml.stage2 import identity_edges, propagate

    added = propagate(components, identity_edges(tables, cutoff), bank)
    links = [link for ring_links in added.values() for link in ring_links]
    alertable_links = [link for link in links if link.alertable]
    context_links = [link for link in links if not link.alertable]
    print(
        f"rings={len(components)} velocity_members={sum(len(v) for v in components.values())} "
        f"identity_link_members={len(links)} (alertable={len(alertable_links)} "
        f"context={len(context_links)})"
    )

    # Every member that names a ``link_evidence_type`` must have the
    # matching ``account_evidence`` row to join to, or a UI following the
    # join renders the account with its qualifier missing. So the
    # evidence accounts are the flagged set *plus* the cross-bank
    # identity-link members, and the same-bank scope is emitted for the
    # context members it is the only join target for.
    cross_accounts = pd.Index(
        sorted(set(flagged_ids) | {link.hashed_account_id for link in alertable_links})
    )
    context_accounts = pd.Index(sorted({link.hashed_account_id for link in context_links}))
    records = pd.concat(
        [
            evidence_records(tables, cutoff, accounts=cross_accounts),
            evidence_records(
                tables, cutoff, accounts=context_accounts, scope=EvidenceScope.SINGLE_BANK
            ),
        ],
        ignore_index=True,
    ).sort_values(["hashed_account_id", "scope", "evidence_type"], ignore_index=True)

    generated_at = datetime.now(UTC)
    written: dict[str, tuple[pd.DataFrame, str]] = {
        "scores": (
            validate_rows(build_scores(frame, evidence, flagged, scores), RiskScore),
            "shared.contracts.RiskScore",
        ),
        "account_explanations": (
            validate_rows(
                build_account_explanations(frame, scores, contribs, columns, flagged, generated_at),
                AccountExplanation,
            ),
            "shared.contracts.AccountExplanation",
        ),
        "rings": (
            validate_rows(build_rings(components, added, bank, detected_at), Ring),
            "shared.contracts.Ring",
        ),
        "ring_explanations": (
            validate_rows(
                build_explanations(components, added, frame, evidence, bank, generated_at),
                RingExplanation,
            ),
            "shared.contracts.RingExplanation",
        ),
        "account_evidence": (
            validate_rows(build_evidence(records, set(flagged_ids)), AccountEvidence),
            "shared.contracts.AccountEvidence",
        ),
    }

    out_dir = cfg.ml_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The settings block is not decoration. Without it "45 rings" is an
    # unreproducible number: a reader cannot tell whether it came from a
    # score threshold or a rule, or at what observation horizon.
    manifest: dict[str, object] = {
        "generated_at": generated_at.isoformat(),
        # This module writes the manifest wholesale, so running it alone
        # drops ``ml.timeline``'s entry while leaving that table's parquet
        # on disk. ``ml.publish_all`` runs the two in the order that
        # leaves the manifest describing everything ``ml/out/`` contains.
        "regenerated_by": "python -m ml.publish_all",
        "detector_version": DETECTOR_VERSION,
        "score_model_version": MODEL_VERSION,
        "explainer_version": EXPLAINER_VERSION,
        "settings": {
            "ring_detector": "velocity rule (ml.rule.velocity_decision), NOT a score threshold",
            "score_threshold_for_rings": None,
            "feature_scope": "cb_ (cross-bank)",
            "observation_cutoff": "full window (ml.features.full_window_cutoff)",
            "cutoff_as_of": detected_at.isoformat(),
            "velocity_max_seconds": VELOCITY_MAX_SECONDS,
            "velocity_ratio_range": list(VELOCITY_RATIO_RANGE),
            "velocity_min_in_degree": VELOCITY_MIN_IN_DEGREE,
            "velocity_min_out_degree": VELOCITY_MIN_OUT_DEGREE,
            "min_distinct_counterparty_banks": MIN_BANKS,
            "clustering": "connected components over transfers with BOTH endpoints flagged",
            "clustering_min_ring_size": 2,
            "flagged_accounts": len(flagged_ids),
            "score_seed": seed,
            # Stage 2 IS folded into rings.parquet. Every member carries
            # how it got there, so the two claims stay distinguishable
            # inside one table instead of being merged into a flat list.
            "stage2_identity_propagation": (
                "APPLIED to rings.parquet as members with "
                "membership_basis='identity_link'; stage-1 members carry "
                "membership_basis='velocity'. Measured separately in "
                "ml/experiments/stage2/metrics.json"
            ),
            "stage2_version": STAGE2_VERSION,
            "stage2_alerting_scope": (
                "CROSS-BANK ONLY: a cross-bank identity link is alertable=True; a "
                "same-bank identity link is published as CONTEXT (alertable=False, "
                "link_scope='single_bank') and must never be counted as an alert or "
                "described as a mule"
            ),
            "ring_member_fields": list(RING_MEMBER_FIELDS),
            "ring_members_velocity": sum(len(v) for v in components.values()),
            "ring_members_identity_link": len(links),
            "ring_members_alertable": (
                sum(len(v) for v in components.values()) + len(alertable_links)
            ),
            "ring_members_context": len(context_links),
            "is_passthrough_mule": (
                "role in {mule, collector} (712) -- the label the shipped detector is scored on"
            ),
            "is_complicit": (
                "role in {mule, collector, source} (899) -- sources are complicit "
                "feeders, not victims; they are outside the detector label only "
                "because median in-degree 0 makes them structurally invisible to a "
                "pass-through rule"
            ),
        },
        "tables": {},
    }
    for name, (table, model) in written.items():
        # Enum members would otherwise land in parquet as opaque objects.
        # Tested with isinstance(Enum) and NOT hasattr(v, "value"):
        # pd.Timestamp also has a .value (epoch nanoseconds), so the
        # duck-typed check silently flattened every tz-aware datetime
        # column to bare int64 and threw the UTC zone away. The walk is
        # recursive because ``Ring.members`` is a list of structs and two
        # of its fields are enums: a top-level isinstance check sees a
        # list, not an Enum, and would let them through untouched.
        # Restricted to object columns so a datetime64 column is never
        # mapped back into loose Timestamps.
        for column in table.columns:
            if table[column].dtype == object:
                table[column] = table[column].map(flatten_enums)
        path = out_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest["tables"][name] = {  # type: ignore[index]
            "rows": len(table),
            "validated_against": model,
            "columns": list(table.columns),
        }
        print(f"wrote {path} rows={len(table)} validated against {model}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
