"""Shared Pydantic contracts for MuleNet.

All modules must import these models rather than redefining schemas
locally (see CLAUDE.md). Account identifiers are always hashed —
never store or pass raw/real account numbers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Annotated, Optional
from uuid import NAMESPACE_DNS, uuid4, uuid5

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    model_validator,
)

_CENTS = Decimal("0.01")


def _to_money(value: object) -> Decimal:
    """Coerce an incoming value to a Decimal rounded to 2 decimal places."""
    return Decimal(str(value)).quantize(_CENTS, rounding=ROUND_HALF_UP)


#: Monetary amount. Decimal-backed so money is never a binary float, and
#: serialized to JSON as a string with exactly two decimal places so it
#: round-trips through Pub/Sub without precision loss. Matches
#: ``NUMERIC(15, 2)`` in ``shared/ddl.sql`` and ``NUMERIC`` in
#: ``shared/bq_schemas.py``.
MoneyAmount = Annotated[
    Decimal,
    BeforeValidator(_to_money),
    PlainSerializer(lambda v: f"{v:.2f}", return_type=str, when_used="json"),
]


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Timezone-aware because every datetime column in ``shared/ddl.sql`` is
    ``TIMESTAMPTZ``; a naive datetime would silently mismatch it.
    """
    return datetime.now(timezone.utc)


class AccountStatus(str, Enum):
    """Lifecycle status of an account."""

    ACTIVE = "active"
    DORMANT = "dormant"
    FROZEN = "frozen"
    CLOSED = "closed"


class RingStatus(str, Enum):
    """Lifecycle status of a detected mule ring."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class InvestigationStatus(str, Enum):
    """Lifecycle status of an analyst investigation."""

    OPEN = "open"
    IN_REVIEW = "in_review"
    CLOSED = "closed"


class StrReportStatus(str, Enum):
    """Lifecycle status of a suspicious transaction report."""

    DRAFT = "draft"
    FILED = "filed"
    REJECTED = "rejected"


class EvidenceScope(str, Enum):
    """Whether a piece of evidence is observable within one bank or across banks.

    This is the load-bearing distinction on :class:`AccountEvidence`.
    Same-bank identity reuse is mostly noise — far more legitimate
    customers reuse a document within one bank than mules do — while
    cross-bank reuse is the fact worth citing. Matches the ``scope``
    values on the ``features`` BigQuery table in ``shared/bq_schemas.py``.
    """

    SINGLE_BANK = "single_bank"
    CROSS_BANK = "cross_bank"


class MembershipBasis(str, Enum):
    """How an account came to be in a ring. The two are different claims.

    ``velocity``: the account was detected by the stage-1 rule — fan-in,
    fast pass-through, cross-bank structure. That is a *finding* about
    the account's own behaviour.

    ``identity_link``: the account was pulled in by stage-2 propagation
    out of an already-detected ring, because it shares a device
    fingerprint or a hashed KYC id number with a member. That is a
    *lead*, never a finding. "Forwarded money fast" and "shares a phone
    with someone who did" are not the same accusation, and an
    ``identity_link`` member must be described as "linked, needs
    review" and never as a mule.

    Closed and enumerated, unlike :attr:`AccountEvidence.evidence_type`.
    That field is an open taxonomy of *observations* and still grows.
    This one names the *accusation class*, and every consumer branches
    on it exhaustively: ``agents/`` picks the wording, the verifier
    rejects the wrong wording, ``ui/`` picks the styling. An open
    ``str`` would let a producer emit a third value that every
    ``if basis == "identity_link"`` branch falls straight through,
    silently rendering a lead as a detected mule. If a stage-3
    mechanism appears, adding a member here is a deliberate contract
    change that forces every consumer to decide how to describe it —
    which is exactly the review that has to happen.
    """

    VELOCITY = "velocity"
    IDENTITY_LINK = "identity_link"


class Account(BaseModel):
    """A bank account, identified only by its hashed account id."""

    hashed_account_id: str
    bank_id: str
    opened_at: datetime
    account_type: str
    status: AccountStatus = AccountStatus.ACTIVE


class Transfer(BaseModel):
    """A single money transfer between two (possibly cross-bank) accounts."""

    transfer_id: str = Field(default_factory=lambda: str(uuid4()))
    from_hashed_account_id: str
    to_hashed_account_id: str
    amount: MoneyAmount
    currency: str = "USD"
    timestamp: datetime
    channel: str


class Device(BaseModel):
    """A device fingerprint associated with an account, hashed at rest."""

    device_id: str
    hashed_account_id: str
    device_fingerprint: str
    first_seen: datetime
    last_seen: datetime


class KycDocument(BaseModel):
    """A single parsed KYC document (ID card, address proof, ...)."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    hashed_account_id: str
    doc_type: str
    hashed_id_number: str
    normalized_address: str
    photo_embedding_ref: Optional[str] = None


class KycSubmission(BaseModel):
    """A bundle of KYC documents submitted for one account during onboarding."""

    submission_id: str = Field(default_factory=lambda: str(uuid4()))
    hashed_account_id: str
    bank_id: str
    documents: list[KycDocument] = Field(default_factory=list)
    submitted_at: datetime = Field(default_factory=_utcnow)


class RiskScore(BaseModel):
    """The XGBoost baseline risk score for an account.

    ``score`` is stored at full float precision -- ``FLOAT64`` in
    ``shared/bq_schemas.py`` and ``shared/ddl.sql`` -- but an instance
    that reached an agent through the tools in
    ``shared/tool_interfaces.py`` has had ``score`` rounded to
    :data:`~shared.tool_interfaces.SCORE_DECIMALS` places at that
    boundary. A consumer comparing a quoted score against a recomputed
    one must use :data:`~shared.tool_interfaces.SCORE_TOLERANCE`, which
    is derived from that rounding, and must never write its own
    tolerance literal.
    """

    hashed_account_id: str
    score: float
    model_version: str
    scored_at: datetime = Field(default_factory=_utcnow)
    top_factors: list[str] = Field(default_factory=list)


class RingMember(BaseModel):
    """One account in a ring, carrying *how* it got there.

    The basis lives on the member rather than in a parallel array beside
    the ring because a parallel array can be sliced, filtered, sorted or
    partially copied out of alignment, and a misaligned basis renders an
    innocent relative as a detected mule. Here the basis cannot detach
    from the account it describes.

    **Alert against context.** :attr:`alertable` is stored on the row,
    not left to the consumer. Deriving it means joining to
    ``account_evidence``, comparing ``bank_id`` across counterparties
    and reasoning about scope — precisely the derivation someone will
    get wrong, and getting it wrong publishes an alert against a family
    member. The validator below keeps it consistent with
    :attr:`link_scope`, so it is a redundant *statement* of an
    invariant, not a second source of truth that can disagree.

    Attributes:
        hashed_account_id: The member account.
        membership_basis: ``velocity`` (stage-1 detection) or
            ``identity_link`` (stage-2 propagation).
        alertable: Whether this membership may be published as an
            ALERT. Always ``True`` for ``velocity``. For
            ``identity_link`` it is ``True`` only when
            :attr:`link_scope` is ``cross_bank``: stage-2 alerting is
            cross-bank only, and same-bank identity links are published
            as CONTEXT. Same-bank identity reuse is dominated by
            legitimate cover — roughly 1,442 legitimate accounts share
            a document within a bank against ~292 mules — so a
            same-bank link is a thing to show an analyst, never a thing
            to accuse anyone with.
        link_scope: For an ``identity_link`` member, whether the link
            crosses a bank boundary. Reuses :class:`EvidenceScope`
            rather than introducing a second two-valued enum: it is the
            same distinction, drawn on the same facts (``devices`` and
            ``kyc_documents``), and one spelling means a consumer that
            already knows to read ``scope`` cannot fail to read this.
            ``None`` for a ``velocity`` member.
        linked_from_hashed_account_id: The ring member at the other end
            of the identity edge. ``None`` for a ``velocity`` member.
        link_evidence_type: Which reuse linked it, in the same open
            taxonomy as :attr:`AccountEvidence.evidence_type` (for
            example ``cross_bank_shared_device``), so a consumer that
            wants the full citable row — counterparty banks,
            ``observed_at``, ``detail`` — can join to
            ``account_evidence`` on
            ``(linked_from_hashed_account_id, link_evidence_type,
            link_scope)``. ``None`` for a ``velocity`` member.
        reason: A denormalised analyst-readable sentence saying why this
            account is in the ring. Required for ``identity_link``, and
            it must read as a lead ("Linked, needs review: shares a
            device fingerprint with a ring member at another bank"),
            never as a finding. Denormalised on purpose: ``ui/`` is
            required to show the reason, and a UI that must resolve a
            join before it can render will sooner or later render the
            member with the reason missing — which is the flat list
            again, an accusation with its qualifier dropped. It is also
            not a duplicate of ``account_evidence``: it states the
            *ring-scoped* edge (this member, this ring, that member),
            which no evidence row carries. Optional for ``velocity``,
            whose reason is already the ring's own
            :class:`RingExplanation`.
    """

    hashed_account_id: str
    membership_basis: MembershipBasis
    alertable: bool
    link_scope: Optional[EvidenceScope] = None
    linked_from_hashed_account_id: Optional[str] = None
    link_evidence_type: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _link_fields_match_basis(self) -> RingMember:
        """The link fields and ``alertable`` must agree with the basis."""
        link_fields = {
            "link_scope": self.link_scope,
            "linked_from_hashed_account_id": self.linked_from_hashed_account_id,
            "link_evidence_type": self.link_evidence_type,
        }
        if self.membership_basis is MembershipBasis.VELOCITY:
            present = sorted(name for name, value in link_fields.items() if value is not None)
            if present:
                raise ValueError(
                    "a velocity member carries no identity link; remove " + ", ".join(present)
                )
            if not self.alertable:
                raise ValueError("a velocity member is a detection and is always alertable")
            return self
        missing = sorted(name for name, value in link_fields.items() if value is None)
        if missing:
            raise ValueError(
                "an identity_link member must name the link it arrived by: missing "
                + ", ".join(missing)
            )
        if not self.reason:
            raise ValueError(
                "an identity_link member must carry a reason: it is a lead, not a "
                "finding, and every consumer is required to show why it was linked"
            )
        expected = self.link_scope is EvidenceScope.CROSS_BANK
        if self.alertable != expected:
            raise ValueError(
                "stage-2 alerting is cross-bank only: alertable must be "
                f"{expected} for a {self.link_scope.value} identity link"
            )
        return self


class Ring(BaseModel):
    """A cluster of accounts identified as a probable mule ring.

    Members arrive by two routes — stage-1 velocity detection and
    stage-2 identity propagation — and the route is part of the claim
    made about each account. See :class:`RingMember`.

    The flat ``member_hashed_account_ids: list[str]`` this model used to
    carry is **removed**, not kept alongside ``members``. It had nowhere
    to put a per-member basis, and keeping it would let every existing
    consumer go on reading a list in which a linked relative is
    indistinguishable from a detected mule — the exact failure this
    change exists to prevent, preserved under a compatibility alias.
    Removing it turns each of those call sites into an
    ``AttributeError``: a loud one-time migration instead of a silent
    mislabelling. The accessors below are deliberately *not* spelled
    ``member_hashed_account_ids``, so a caller has to state which
    population it means.

    ``banks_involved`` stays the banks of **all** members, alertable or
    not, since it describes the footprint an analyst is looking at.
    """

    ring_id: str = Field(default_factory=lambda: str(uuid4()))
    members: list[RingMember]
    banks_involved: list[str]
    detected_at: datetime = Field(default_factory=_utcnow)
    status: RingStatus = RingStatus.CANDIDATE

    @model_validator(mode="after")
    def _members_are_unique(self) -> Ring:
        """One row per account: an account cannot hold two bases at once."""
        ids = [member.hashed_account_id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("an account may appear at most once in a ring's members")
        return self

    @property
    def all_member_ids(self) -> list[str]:
        """Every member, regardless of basis. For subgraph expansion."""
        return [m.hashed_account_id for m in self.members]

    @property
    def velocity_member_ids(self) -> list[str]:
        """Members detected by the stage-1 velocity rule."""
        return [
            m.hashed_account_id
            for m in self.members
            if m.membership_basis is MembershipBasis.VELOCITY
        ]

    @property
    def identity_link_member_ids(self) -> list[str]:
        """Members pulled in by stage-2 identity propagation."""
        return [
            m.hashed_account_id
            for m in self.members
            if m.membership_basis is MembershipBasis.IDENTITY_LINK
        ]

    @property
    def alertable_member_ids(self) -> list[str]:
        """Members that may be published as an alert.

        Every velocity member, plus identity-link members whose link is
        cross-bank. This is the set an alert count, an STR narrative or
        a case is built from.
        """
        return [m.hashed_account_id for m in self.members if m.alertable]

    @property
    def context_member_ids(self) -> list[str]:
        """Members published as CONTEXT only — same-bank identity links.

        Shown to an analyst with their reason, never counted as alerts
        and never described as mules.
        """
        return [m.hashed_account_id for m in self.members if not m.alertable]


class RingExplanation(BaseModel):
    """Human-readable explanation of why a ring was flagged."""

    ring_id: str
    top_factors: list[str] = Field(default_factory=list)
    narrative: str
    generated_at: datetime = Field(default_factory=_utcnow)


class GnnScore(BaseModel):
    """The GraphSAGE (or other GNN) score for an account, if used instead of XGBoost."""

    hashed_account_id: str
    score: float
    model_version: str
    embedding_ref: Optional[str] = None
    scored_at: datetime = Field(default_factory=_utcnow)


class ExplanationFactor(BaseModel):
    """One human-meaningful reason an account scored where it did.

    ``description`` is a sentence an analyst can paste into a case note
    ("KYC ID number reused at another bank"), never a raw feature column
    name: ``cb_passthrough_speed_seconds`` is not a reason, it is a
    variable. ``feature_name`` keeps the raw column alongside it so a
    model engineer can still reproduce the attribution, but no consumer
    may render it to an analyst in place of ``description``.

    Attributes:
        description: Human-readable reason, in the analyst's language.
        contribution: Signed contribution to the score (SHAP-style).
            Positive pushes the score up, negative pushes it down. The
            sign is load-bearing: a factor list that drops it lets a
            mitigating factor be read as incriminating. Stored at full
            precision, but rounded to
            :data:`~shared.tool_interfaces.SCORE_DECIMALS` places when
            it crosses the agent tool boundary. That rounding preserves
            the sign and never moves a value across zero; a magnitude
            below half a unit in the last place does round to zero, and
            must then be read as "did not measurably move the score"
            rather than as a direction.
        feature_name: The underlying feature column, for reproducibility.
            ``None`` when the factor comes from a rule rather than a
            model feature (for example "velocity rule fired").
    """

    description: str
    contribution: float
    feature_name: Optional[str] = None


class AccountExplanation(BaseModel):
    """Why one account received the score it did.

    The per-account counterpart of :class:`RingExplanation`. Produced by
    whichever model is actually shipped — today the XGBoost baseline —
    so it is deliberately *not* named after a model family. The GNN was
    evaluated and not shipped; if one ever is, its explanation is still
    "top factors plus a narrative for one account" and belongs here,
    with a subgraph reference added as an optional field at that point.

    ``score`` and ``model_version`` are carried on the explanation
    itself rather than being looked up from ``scores``: an explanation
    shown next to a score it was not computed from is a wrong
    explanation, and this makes that mismatch detectable.

    As with :class:`RiskScore`, ``score`` is persisted at full
    precision, and both it and every ``top_factors[].contribution`` are
    rounded to :data:`~shared.tool_interfaces.SCORE_DECIMALS` places
    when an agent receives them through
    ``shared/tool_interfaces.py``.
    """

    hashed_account_id: str
    score: float
    model_version: str
    top_factors: list[ExplanationFactor] = Field(default_factory=list)
    narrative: str
    generated_at: datetime = Field(default_factory=_utcnow)


class AccountEvidence(BaseModel):
    """One citable identity-reuse fact attached to an account.

    **Evidence, not a detector.** Detection fires on velocity and
    structure alone; nothing on this model takes part in that decision.
    These rows exist so a compliance analyst and the report-writing
    agent can name *who*, *which bank* and *when* — an STR narrative
    that says "shares a device" without naming the counterparty bank is
    not worth filing.

    **Identity reuse is a weak signal on its own.** Thousands of
    legitimate accounts reuse a document within one bank, and dual-bank
    customers legitimately reuse an id across two banks. A row here
    therefore never implies guilt: it must always be read together with
    its :attr:`scope`, and rendered with the counterparty banks it
    actually names.

    One row per ``(hashed_account_id, evidence_type, scope)``.

    Attributes:
        hashed_account_id: The account the evidence is attached to.
        evidence_type: Which reuse was observed. Currently produced by
            ``ml/``: ``cross_bank_shared_device``,
            ``cross_bank_reused_id_number``, ``cross_bank_reused_photo``,
            with ``same_bank_`` counterparts for the single-bank scope.
            Deliberately a plain ``str`` rather than an ``Enum``: the
            taxonomy is open and still growing (``docai/`` also detects
            ``normalized_address`` reuse, see
            :func:`shared.tool_interfaces.get_document_reuse`), so a
            closed enum would force a contract change per new key.
        scope: ``single_bank`` or ``cross_bank``. Closed and enumerated
            because the whole interpretation of a row turns on it.
        detail: Ready-to-cite sentence naming the counterparty banks.
        counterparty_hashed_account_ids: The other accounts in the link,
            sorted. Never empty for a real row.
        counterparty_bank_ids: The banks those accounts sit at, sorted.
        observed_at: When the link first became *citable* — the later of
            the two sightings, since a bank cannot act on a pair before
            both sides exist. Required and never defaulted: defaulting
            it to "now" would fabricate a point-in-time claim that an
            analyst may quote in a filing.
        source_table: The table the fact is quotable from — ``devices``
            or ``kyc_documents`` — so a report can point at its source.
        detector_version: Version of the detector whose flagged accounts
            this evidence was attached to. ``None`` when evidence is
            produced independently of a detector run (``docai/``).
    """

    hashed_account_id: str
    evidence_type: str
    scope: EvidenceScope
    detail: str
    counterparty_hashed_account_ids: list[str] = Field(default_factory=list)
    counterparty_bank_ids: list[str] = Field(default_factory=list)
    observed_at: datetime
    source_table: str
    detector_version: Optional[str] = None


#: Namespace for deriving a case id from the ring it is about.
#: Fixed forever: changing it renames every case.
CASE_ID_NAMESPACE = uuid5(NAMESPACE_DNS, "cases.mulenet")

#: Re-attempts the verification loop allows **after** the first draft.
#: One, deliberately. The loop is investigator -> verifier: if the
#: verifier rejects a claim, the investigator gets exactly one more go
#: with the unsupported claims fed back, and if that also fails the
#: narrative is withheld. A second retry buys little — an agent that
#: failed the same check twice is not converging — and it doubles the
#: worst-case latency on the path an analyst is waiting on.
#:
#: It lives here, not in ``agents/loop.py``, because
#: :data:`MAX_VERIFICATION_ATTEMPTS` below is derived from it and that
#: derivation is a *record-level* constraint: the ``cases`` validator
#: refuses a row claiming an attempt the loop cannot make. A constant a
#: contract validator depends on cannot live in the folder being
#: validated. ``agents/loop.py`` imports this name instead of declaring
#: its own.
MAX_RETRIES = 1

#: The most investigator attempts a case may rest on: the first draft
#: plus the retries :data:`MAX_RETRIES` allows. A further attempt does
#: not exist, so a case claiming one is a fabricated row rather than a
#: longer run. At ``MAX_RETRIES = 1`` this evaluates to exactly ``2``.
#:
#: **Derived, never a literal** — the same rule as
#: :data:`shared.tool_interfaces.SCORE_TOLERANCE`, and for the same
#: reason. A frozen ``2`` is correct only by coincidence with the current
#: retry budget, and drifts wrong in **either** direction the moment the
#: loop changes:
#:
#: * Raise ``MAX_RETRIES`` to 2 and a legitimate three-attempt case is
#:   rejected by a validator citing a budget that no longer exists — a
#:   loud failure, but on correct data.
#: * Lower it to 0 and a frozen ``2`` silently **accepts a row claiming
#:   a retry that never happened**, which is the invisible failure: an
#:   analyst reads ``verification_attempts = 2`` as "passed on the
#:   retry" when no retry was ever run.
#:
#: Attempts are first draft + retries, so any consumer wanting the retry
#: budget imports :data:`MAX_RETRIES` and any consumer wanting the
#: attempt cap imports this. Neither may restate the other's number.
MAX_VERIFICATION_ATTEMPTS = MAX_RETRIES + 1


def case_id_for_ring(ring_id: str) -> str:
    """Return the one case id a ring's case is allowed to have.

    A case id is derived from its ring rather than minted at random so
    that "one case per ring" is a property of the id itself: a second
    verified investigation of the same ring produces the same id and
    therefore updates the same row, and re-running the producer is
    idempotent instead of minting a duplicate case. ``ui/`` can also
    reach a ring's case without a lookup.
    """
    return str(uuid5(CASE_ID_NAMESPACE, ring_id))


class Investigation(BaseModel):
    """A case opened against a ring by a **verified** investigation.

    **A case exists only where a narrative was checked against the
    data.** ``agents/`` runs an investigator that drafts a narrative
    with structured claims, and a deterministic Python verifier that
    re-derives every claim from tool data. On failure there is exactly
    one retry with the unsupported claims fed back; if it still fails
    the narrative is **withheld** and only raw data plus a warning is
    returned. A withheld investigation produces **no case row at all** —
    there is no ``withheld`` status here on purpose, because a case that
    can be unverified is a case an analyst has to check before trusting,
    which defeats the point of the table. An analyst reading ``cases``
    may take it that every row rests on claims that were re-derived and
    matched.

    That rule is enforced by shape rather than by a label, which is the
    difference from :attr:`StrReport.citations_verified`. A draft STR
    legitimately exists unverified, so it needs a flag that can be
    ``False``; an unverified case does not legitimately exist at all, so
    a ``verified: bool`` field here would be a constant ``True`` — a
    claim about the row rather than a constraint on it, and one a
    producer could simply assert. Instead:

    * :attr:`verified_at` is **required with no default**, so a case
      cannot come into being by omission.
    * :attr:`verified_claim_count` must be at least 1. A withheld
      investigation carries an empty claim list (``agents/loop.py``
      drops the claims along with the narrative), so the withheld shape
      cannot be written here without inventing a number.
    * :attr:`verification_attempts` is capped at
      :data:`MAX_VERIFICATION_ATTEMPTS`, the retry budget that actually
      exists.

    The matching ``CHECK`` constraints on the ``cases`` table in
    ``shared/ddl.sql`` refuse the same rows by ``UPDATE``.

    Produced by ``agents/``, one row per ring with a verified
    investigation, published as ``agents/out/cases.parquet`` (mirroring
    the ``ml/out/`` convention) and loaded by ``infra/``.

    There is deliberately no ``report_id`` back-pointer to
    :class:`StrReport`. The link already exists as ``StrReport.case_id``
    and points the way the cardinality runs: one case can accumulate
    several drafts. A back-pointer would have to name one of them, and
    the two sides could then disagree about which draft is current.

    There is also deliberately no ``assignee``: MuleNet has no user or
    identity model, so any value written there would be invented. Add it
    when real analyst identities exist, not before.

    Attributes:
        case_id: Always ``case_id_for_ring(ring_id)``. Required rather
            than defaulted, so a caller states the ring's case id
            instead of receiving a random one that happens to be unique.
        verified_at: When the verifier passed the investigation this
            case rests on. Required with no default, for the same
            reason as :attr:`AccountEvidence.observed_at`: it is a
            point-in-time claim about a compliance action, and a
            wall-clock default would fabricate it. On a re-investigation
            of the same ring it advances.
        verification_attempts: How many investigator calls it took, 1 or
            2. The live run makes the difference worth showing an
            analyst: rings 00018 and 00001 passed on the first attempt,
            ring 00002 only on the second.
        verified_claim_count: How many structured claims the verifier
            re-derived and confirmed. At least 1 — a narrative asserting
            things with nothing to check is unchecked, not verified.
        closed_at: When the case was closed. ``None`` while the case is
            open or in review; set exactly when ``status`` becomes
            ``closed``. Not defaulted to "now", for the same reason as
            :attr:`verified_at`.
    """

    case_id: str
    ring_id: str
    verified_at: datetime
    verification_attempts: int
    verified_claim_count: int
    analyst_notes: str = ""
    status: InvestigationStatus = InvestigationStatus.OPEN
    opened_at: datetime = Field(default_factory=_utcnow)
    closed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _rests_on_a_passed_verification(self) -> Investigation:
        """A case may only describe a verification that actually passed."""
        expected = case_id_for_ring(self.ring_id)
        if self.case_id != expected:
            raise ValueError(
                "a ring has exactly one case: case_id must be "
                f"case_id_for_ring({self.ring_id!r}) == {expected}"
            )
        if not 1 <= self.verification_attempts <= MAX_VERIFICATION_ATTEMPTS:
            raise ValueError(
                "verification_attempts must be between 1 and "
                f"{MAX_VERIFICATION_ATTEMPTS}: the investigator gets one draft "
                "and one retry, so any other number describes a run that did "
                "not happen"
            )
        if self.verified_claim_count < 1:
            raise ValueError(
                "a case must rest on at least one re-derived claim: a "
                "withheld investigation carries no claims, and a narrative "
                "with nothing to check is unchecked, not verified"
            )
        return self

    @model_validator(mode="after")
    def _closed_at_matches_status(self) -> Investigation:
        """``closed_at`` is set if and only if the case is closed."""
        if self.status is InvestigationStatus.CLOSED and self.closed_at is None:
            raise ValueError("a closed investigation must carry closed_at")
        if self.status is not InvestigationStatus.CLOSED and self.closed_at is not None:
            raise ValueError("closed_at may only be set on a closed investigation")
        return self


class StrReport(BaseModel):
    """A regulation-cited suspicious transaction report drafted for a case.

    **A draft must never be mistakable for a filing.** Every citation in
    the system today comes from
    ``docai/regulations/placeholder_corpus.json`` — synthetic
    placeholders written for development. ``citations_verified`` records
    that fact *on the report itself*, so it survives export, copy-paste
    and display in a UI that did not happen to render the warning
    banner. It defaults to ``False``, the safe value: verification is
    something a caller must assert, never something it gets by omission.

    The invariant below, mirrored by a ``CHECK`` constraint on the
    ``str_reports`` table in ``shared/ddl.sql``, makes an unverified
    report unfileable rather than merely labelled.

    Attributes:
        citations_verified: ``True`` only when every entry in
            ``regulation_citations`` was resolved against a real indexed
            regulation document. ``agents/`` sets this from
            ``agents.regulations.is_synthetic`` over its retrieved
            passages; while the corpus is the placeholder one it is
            always ``False``.
    """

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    narrative: str
    regulation_citations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    status: StrReportStatus = StrReportStatus.DRAFT
    citations_verified: bool = False

    @model_validator(mode="after")
    def _filed_requires_verified_citations(self) -> StrReport:
        """A report may not reach ``filed`` on unverified citations."""
        if self.status is StrReportStatus.FILED and not self.citations_verified:
            raise ValueError(
                "an STR cannot be filed while citations_verified is False: "
                "its citations are synthetic placeholders, not real regulations"
            )
        return self
