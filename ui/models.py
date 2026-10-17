"""Response models for the MuleNet UI API.

These are *view* models: what a browser needs, flattened and
JSON-ready. They deliberately reuse the field names from
``shared.contracts`` (``hashed_account_id``, ``from_hashed_account_id``,
``member_hashed_account_ids``, ``regulation_citations`` ...) so a
contract rename shows up here as a mismatch instead of silently
drifting. The snapshot builder validates every record against the real
contract models before it ever reaches these shapes.

Money crosses the wire as a fixed 2-decimal string, matching
``MoneyAmount``'s JSON serializer in ``shared/contracts.py``. The
browser formats it; it never does arithmetic on it as a float.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from shared import metric_catalog


class BoundingInfo(BaseModel):
    """How the graph payload is kept small enough to render."""

    max_graph_nodes: int
    graph_hops: int
    #: Ceiling on edges in one subgraph. Not a hard cap on the money
    #: trail: intra-ring edges are kept in full so no hop is silently
    #: dropped, and only context edges are budgeted.
    max_transfers_per_ring: int
    max_context_edges: int
    note: str


class DatasetInfo(BaseModel):
    """Scale of the underlying dataset, for the header strip."""

    accounts: int
    transfers: int
    detected_rings_total: int
    rings_in_snapshot: int
    banks: list[str]
    #: Population numbers behind ``accounts``/``transfers`` and the enclave
    #: card's smaller ones. ``None`` when the snapshot predates them.
    #: ``scope_reason`` is machine-readable: the enclave sees cross-bank
    #: edges only, so its edge and account counts are subsets of these.
    accounts_generated: int | None = None
    accounts_with_activity: int | None = None
    edges_total: int | None = None
    edges_in_enclave_scope: int | None = None
    accounts_in_enclave_scope: int | None = None
    scope_reason: str | None = None
    # Additive. ``accounts`` above is every account generated; the joined
    # graph holds only accounts on a cross-bank transfer, so it is smaller.
    # None on a snapshot built before these were written.
    accounts_generated: int | None = None
    accounts_with_cross_bank_activity: int | None = None
    cross_bank_transfers: int | None = None


class AlertRow(BaseModel):
    """One row of a single bank's own alert queue.

    ``score`` is the real out-of-fold single-bank XGBoost P(mule) from
    ``ml/out/bank_alert_queues.parquet``; ``top_factors`` is the
    description of each of that model's top contributions. ``is_mule`` is
    ground truth, shown only behind the tile's reveal toggle so the demo
    cannot quietly flatter itself.
    """

    hashed_account_id: str
    bank_id: str
    score: float
    #: 1 = highest score in the bank. Label-free, always present.
    rank_in_bank: int | None = None
    top_factors: list[str] = Field(default_factory=list)
    #: Simulator ground truth. ``None`` means WITHHELD (``labels_revealed`` is
    #: false on the tile), never "not a mule".
    is_mule: bool | None = None


class BankQueue(BaseModel):
    """Alert-queue burden for one bank at its 80%-recall threshold."""

    #: The four label-derived fields below are ``None`` when the tile's
    #: ``labels_revealed`` is false (withheld). ``false_alerts_per_catch`` is
    #: also ``None`` when revealed but nothing was caught; the flag tells the two
    #: apart.
    queue_depth: int | None = None
    queue_true_positives: int | None = None
    queue_false_positives: int | None = None
    false_alerts_per_catch: float | None = None
    recall_target: float
    reached_recall: bool | None = None


class BankTile(BaseModel):
    """Everything one bank dashboard shows, using only that bank's data."""

    bank_id: str
    account_count: int
    transfers_seen: int
    transfers_leaving_perimeter: int
    #: Label-derived counts: ``None`` when ``labels_revealed`` is false.
    mules_in_bank: int | None = None
    alerts: list[AlertRow]
    alert_true_positives: int | None = None
    alert_false_positives: int | None = None
    queue: BankQueue
    score_model_version: str
    #: Which model scored this tile: ``single_bank`` (``sb_`` features only,
    #: ``bank_alert_queues``). ``cross_bank`` for a snapshot built before that
    #: table existed.
    score_scope: str = "cross_bank"
    queue_caveat: str
    #: True when the simulator's ground-truth labels are in this tile. False
    #: means every label-derived field (``is_mule``, ``mules_in_bank``,
    #: ``alert_*_positives``, ``queue.queue_*``, ``queue.reached_recall``) is
    #: null because it was withheld, not because the value is zero or unknown.
    labels_revealed: bool = True
    #: Fragment of the currently featured ring this bank can see alone.
    visible_member_hashed_account_ids: list[str] = Field(default_factory=list)
    #: Of those, the ones it could actually raise an alert on. The rest
    #: are same-bank identity links, shown as context and never accused.
    visible_alertable_hashed_account_ids: list[str] = Field(default_factory=list)
    invisible_member_count: int = 0
    visible_intra_ring_transfers: int = 0


class RingSummary(BaseModel):
    """One row of the ring index."""

    ring_id: str
    span: int
    banks_involved: list[str]
    #: Members that may be published as an alert.
    alertable_member_count: int
    #: Same-bank identity links, shown as context and never accused.
    context_member_count: int
    total_member_count: int
    status: str
    detected_at: str
    onset: str | None = None
    #: The detector that CREATED the ring, read off its ``ring_id``.
    #: Not "the detectors that produced evidence about it": those are
    #: per finding, in ``EvidenceItem.detector_versions``.
    detector_version: str
    total_intra_ring_amount: str
    intra_ring_transfer_count: int
    cross_bank_transfer_count: int
    cross_bank_evidence_count: int
    #: Members that are true mules in the simulator's ground truth.
    #: Shown so a judge can see the detector's precision on this ring
    #: rather than being asked to assume it.
    #: Quoted over the ALERTABLE population only. Counting context
    #: members would score the system on accounts it never accused.
    true_positive_alertable_members: int
    #: True when a verified case exists for this ring.
    has_case: bool = False
    #: The ground-truth ring this detected cluster overlaps most, or
    #: ``None`` for a false-positive cluster that overlaps none.
    true_ring_id: str | None = None
    #: Copied from ``ml/out/ring_id_map.parquet``. When true the cluster
    #: spans several true rings (or the overlap ties), so it must never be
    #: captioned as being that one ring. Shown in the ring picker so a
    #: manual selection is informed before the click.
    match_is_ambiguous: bool = False
    has_replay: bool = False


class RingMemberView(BaseModel):
    """A ring member as the case panel and the graph show it.

    Carries its own provenance. ``shared.contracts.Ring`` removed the
    flat ``member_hashed_account_ids`` list precisely so that a linked
    relative cannot be rendered indistinguishably from a detected mule,
    and shipping a bare ``list[str]`` at the HTTP boundary would
    reintroduce that exact failure one layer down.

    ``alertable`` is passed through from the contract row, never
    re-derived here. Re-deriving means joining evidence and comparing
    banks, which is the computation that goes wrong, and going wrong
    publishes an alert against somebody's family member.
    """

    hashed_account_id: str
    bank_id: str
    account_type: str
    status: str
    opened_at: str
    role: str = ""
    risk_score: float
    model_version: str
    top_factors: list[str] = Field(default_factory=list)
    is_true_positive: bool = False
    #: ``velocity`` (a finding about this account's own behaviour) or
    #: ``identity_link`` (a lead: it shares an identifier with a member).
    membership_basis: Literal["velocity", "identity_link"]
    #: Whether this membership may be published as an ALERT. False for
    #: a same-bank identity link, which is CONTEXT only.
    alertable: bool
    link_scope: Literal["single_bank", "cross_bank"] | None = None
    linked_from_hashed_account_id: str | None = None
    link_evidence_type: str | None = None
    #: Analyst-readable sentence saying why this account is in the ring.
    #: Required by the contract for every identity link, and the UI is
    #: required to show it: an accusation with its qualifier dropped is
    #: the flat list all over again.
    reason: str | None = None


class ContextAccount(BaseModel):
    """A one-hop counterparty kept for context, not accused of anything."""

    hashed_account_id: str
    bank_id: str
    account_type: str
    risk_score: float


class TransferEdge(BaseModel):
    """A transfer in the ring subgraph, annotated for rendering."""

    transfer_id: str
    from_hashed_account_id: str
    to_hashed_account_id: str
    amount: str
    currency: str
    timestamp: str
    channel: str
    from_bank: str
    to_bank: str
    cross_bank: bool
    intra_ring: bool


class EvidenceItem(BaseModel):
    """One citable cross-bank finding, aggregated from ``account_evidence``.

    Supporting evidence, never proof. Per ``ml/out/README.md`` identity
    reuse took **no part** in the firing condition: it was measured as a
    detector and found strictly dominated (0 extra true positives, +146
    false positives), then re-attached for the analyst. It is a strong
    confirmation signal - 55.7% of flagged true positives carry it
    against 2.0% of flagged false positives - which is what makes it
    worth rendering at all.

    ``raw_evidence_types`` may hold two values where the reused ID
    number and the matching photo were collapsed into one finding;
    ``note`` then carries the reason.
    """

    kind: Literal["shared_device", "identity_reuse"]
    evidence_type: str
    raw_evidence_types: list[str]
    label: str
    scope: str
    detail: str
    ring_member_hashed_account_ids: list[str]
    counterparty_hashed_account_ids: list[str]
    counterparty_bank_ids: list[str]
    banks: list[str]
    cross_bank: bool
    accounts_with_this: int
    source_table: str
    observed_at: str
    #: Every detector that stamped a row behind this finding, sorted.
    #: ``account_evidence.detector_version`` is per ACCOUNT -- the
    #: velocity rule for an account it flagged, stage-2 identity
    #: propagation for one only pulled in by a link -- so a finding
    #: spanning both has two answers, and reporting one of them was a
    #: parquet-row-order coin flip.
    detector_versions: list[str]
    note: str = ""


class Explanation(BaseModel):
    """Why the ring was flagged, in analyst-readable form."""

    ring_id: str
    top_factors: list[str]
    narrative: str
    generated_at: str


class CaseInfo(BaseModel):
    """A case opened against the ring by a **verified** investigation.

    Only ever present when ``agents/`` published a row: a case exists
    only where a narrative's claims were re-derived from tool data and
    matched. A withheld investigation produces no row, and the ring
    renders "no case yet" rather than a placeholder — a fabricated open
    case would tell an analyst a narrative had been checked when none
    was ever written.
    """

    case_id: str
    ring_id: str
    status: str
    analyst_notes: str
    opened_at: str
    verified_at: str
    #: 1 or 2. Worth showing: a case that passed only on the retry was
    #: drafted once with claims the verifier rejected, and an analyst
    #: reading it should know that.
    verification_attempts: int
    verified_claim_count: int
    closed_at: str | None = None


class ReportInfo(BaseModel):
    """The drafted, regulation-cited suspicious transaction report."""

    report_id: str
    case_id: str
    narrative: str
    regulation_citations: list[str]
    generated_at: str
    status: str
    #: False while citations are placeholders. Flips to True once
    #: ``agents/`` retrieves real passages via Vertex AI Search.
    citations_verified: bool


class SingleBankView(BaseModel):
    """What one bank, alone, can see of this ring."""

    visible_member_hashed_account_ids: list[str]
    visible_alertable_hashed_account_ids: list[str] = Field(default_factory=list)
    flagged_hashed_account_ids: list[str]
    invisible_member_count: int
    visible_intra_ring_transfers: int = 0


class RingDetail(RingSummary):
    """Full ring payload: subgraph, evidence, explanation, case, report."""

    #: Named populations rather than one flat list, mirroring
    #: ``shared.contracts.Ring``: a caller must state which set it means.
    all_member_hashed_account_ids: list[str]
    alertable_member_hashed_account_ids: list[str]
    context_member_hashed_account_ids: list[str]
    velocity_member_hashed_account_ids: list[str]
    identity_link_member_hashed_account_ids: list[str]
    members: list[RingMemberView]
    context_accounts: list[ContextAccount]
    transfers: list[TransferEdge]
    evidence: list[EvidenceItem]
    explanation: Explanation
    #: ``None`` when no verified investigation exists for this ring.
    case: CaseInfo | None = None
    #: ``None`` when there is no case: an STR is drafted *for* a case,
    #: and with none there is nothing to address it to.
    report: ReportInfo | None = None
    single_bank_view: dict[str, SingleBankView]
    bounding: BoundingInfo


class RingMatchInfo(BaseModel):
    """How detector cluster ids were paired to ground-truth ring ids."""

    source: str
    detected_clusters: int
    ambiguous_matches: int
    no_true_ring: int
    note: str


class MatchAmbiguity(BaseModel):
    """Whether a replay's caption may name one true ring as *the* ring.

    ``is_ambiguous`` is copied from ``ml/out/ring_id_map.parquet`` and
    never re-derived. When it is true, ``caveat`` is the sentence that
    must be shown: a detector cluster is a connected component over
    flagged accounts, so it can hold parts of several true rings, and a
    replay captioned with one ring id then misleads.
    """

    is_ambiguous: bool
    true_rings_touched: int
    overlap_members: int
    detected_ring_size: int
    true_ring_size: int | None = None
    overlap_tied: bool
    detected_rings_touching_true_ring: int
    #: Empty exactly when the match is not ambiguous.
    caveat: str = ""


class HorizonFlag(BaseModel):
    """Whether one scope had flagged the ring by one horizon."""

    #: Minutes after onset. ``None`` is the full window, which flags every
    #: ring by construction and so is never quoted as an early catch.
    T: int | None
    label: str
    flagged: bool


class ScopeReplay(BaseModel):
    """When one scope first flagged, relative to the exit.

    ``status`` distinguishes four outcomes that must never be shown the
    same way:

    ``flags_before_exit``
        Flagged inside the grid, before the money left. A catch.
    ``flags_after_exit``
        Flagged inside the grid, but after cash-out. A detection, NOT a
        save, and it has to read as one.
    ``flagged_full_window_only``
        Flagged only over the full window, which is observed at the end
        of the dataset -- days after cash-out. Also not a catch.
    ``never_flagged``
        Not flagged at any horizon at all.

    An absent marker, a late marker and a "never flagged" marker are
    three different facts about the detector.
    """

    scope: Literal["single_bank", "cross_bank"]
    horizons: list[HorizonFlag]
    first_flag_T: int | None = None
    first_flag_at: str | None = None
    flag_offset_minutes: float | None = None
    #: ``None`` only when the scope never flagged at a measured horizon.
    before_exit: bool | None = None
    flagged_full_window: bool
    status: Literal[
        "flags_before_exit",
        "flags_after_exit",
        "flagged_full_window_only",
        "never_flagged",
    ]
    status_label: str
    #: True only for ``flags_before_exit``. The single field a caller can
    #: read to answer "did this scope actually stop the money?", so that
    #: no consumer has to re-derive it and get it wrong.
    caught_in_time: bool
    #: Spelled-out reason. Non-empty for every status except a clean
    #: in-time catch.
    detail: str = ""


class ReplayHop(BaseModel):
    """One intra-ring transfer, with its offset from onset precomputed.

    The browser renders ``offset_minutes``; it never subtracts timestamps
    to derive a timing, so no display can disagree with the timeline.
    """

    index: int
    transfer_id: str
    from_hashed_account_id: str
    to_hashed_account_id: str
    amount: str
    currency: str
    timestamp: str
    offset_minutes: float
    from_bank: str
    to_bank: str
    cross_bank: bool
    #: True when both ends are cluster members. False for the hop that
    #: FUNDS a mule from an unflagged account, which is part of the trail
    #: but not part of the detection.
    intra_ring: bool


class ReplayPayload(BaseModel):
    """Everything the replay panel draws, or why it cannot draw one.

    ``available`` false is a first-class state with a named
    ``unavailable_reason``: a cluster outside this snapshot, a
    false-positive cluster with no ground-truth ring, and a ring with no
    timeline rows each say which one it is rather than rendering blank.
    """

    detected_ring_id: str
    label: str
    available: bool
    unavailable_reason: str = ""
    true_ring_id: str | None = None
    #: Present even when ``available`` is false, so long as the cluster is
    #: in the map at all.
    ambiguity: MatchAmbiguity | None = None
    onset_at: str | None = None
    exit_reached_at: str | None = None
    exit_offset_minutes: float | None = None
    exit_label: str
    horizon_grid: list[int]
    scopes: list[ScopeReplay] = Field(default_factory=list)
    #: Only the hops inside the replay window. The window is the axis, so
    #: the browser plots ``offset_minutes / window_minutes`` and nothing
    #: it draws can disagree with the timeline table.
    hops: list[ReplayHop] = Field(default_factory=list)
    hop_count: int = 0
    #: Subgraph hops the window excludes, counted rather than dropped in
    #: silence: a trail shown as complete when it is not is a lie by
    #: omission.
    hops_before_window: int = 0
    hops_after_window: int = 0
    last_subgraph_hop_offset_minutes: float | None = None
    #: Length of the timeline axis in minutes: onset to whichever of the
    #: exit, the last flag or the last measured horizon comes last.
    window_minutes: float | None = None
    #: Non-empty whenever hops were excluded, naming how many and why.
    window_note: str = ""


class DemoPick(BaseModel):
    """One of the two rings demo mode walks through.

    ``criterion_met`` is the honesty valve. The pick is derived at
    snapshot-build time from ``ring_id_map`` and
    ``ring_detection_timeline``; when the data does not actually satisfy
    the criterion, the closest ring is taken, ``criterion_met`` is false
    and ``honest_note`` says so rather than the criterion being loosened
    until something fits.
    """

    ring_id: str
    true_ring_id: str | None = None
    span: int
    banks_involved: list[str]
    role: Literal["primary", "secondary", "control"]
    exit_offset_minutes: float | None = None
    single_bank_first_flag_T: int | None = None
    cross_bank_first_flag_T: int | None = None
    #: Carried so a caller can tell "flagged late" from "flagged only at
    #: the full window" without re-reading the replay.
    single_bank_status: str
    cross_bank_status: str
    criterion: str
    criterion_met: bool
    claim: str
    honest_note: str = ""


class DemoPlan(BaseModel):
    """The derived two-ring demo walkthrough."""

    label: str
    #: ``None`` when no published cluster fits even loosely.
    primary: DemoPick | None = None
    #: A lead-time illustration, kept deliberately secondary: the median
    #: lead across rings both scopes flag is zero, so speed is not the
    #: claim. Coverage is.
    secondary: DemoPick | None = None
    control: DemoPick | None = None
    eligible_ring_count: int
    primary_candidates: int
    secondary_candidates: int
    control_candidates: int
    #: The picks forced into the snapshot so they are openable at all.
    pinned_ring_ids: list[str] = Field(default_factory=list)
    selection_note: str


class DetectorInfo(BaseModel):
    """Which model produced what, and the caveat on the ring count.

    Two different models are in play and one does not explain the other:
    ``score_model_version`` ranks the alert queues (out-of-fold XGBoost),
    while ``detector_version`` is the velocity rule whose hard decision
    produced the rings. An account can score high without being flagged.
    """

    detector_version: str
    score_model_version: str
    rings_found: int
    ring_span_counts: dict[str, int]
    caveat: str


class RingListSource(BaseModel):
    """Where the ring index in this payload actually came from.

    Two flags, not one. ``mode`` is what was *configured*; ``served_by``
    is what *answered*. They differ exactly when ``MULENET_READ_MODE=
    bigquery`` was set and the query did not answer, and that is the case
    the page must not hide: a silent fallback would let a judge read a
    file as a BigQuery result. Same rule the chat dock already follows
    for the agents service, deliberately worded differently so the two
    degradations are never mistaken for each other.
    """

    #: Configured: ``snapshot`` or ``bigquery``.
    mode: Literal["snapshot", "bigquery"]
    #: What produced the rings below.
    served_by: Literal["snapshot", "bigquery"]
    #: Short on-screen label. ``None`` in plain snapshot mode, where
    #: there is nothing to disclose.
    label: str | None = None
    #: One sentence for the page, naming the tables or naming the file.
    detail: str = ""
    #: ``RingSummary`` fields this query derived, and the ones the
    #: dataset cannot answer and that came from the snapshot even in
    #: ``bigquery`` mode. See ``ui/bqrings.py`` for why, per field.
    bigquery_fields: list[str] = Field(default_factory=list)
    snapshot_fields: list[str] = Field(default_factory=list)
    #: Measured job facts. Cost and latency are a real demo regression
    #: against an instant file read, so they are stated, not hidden.
    bytes_processed: int | None = None
    bytes_billed: int | None = None
    cache_hit: bool | None = None
    latency_ms: float | None = None
    table_ref: str | None = None
    #: Why the fallback fired, verbatim from the exception. Set only when
    #: ``served_by`` is ``snapshot`` while ``mode`` is ``bigquery``.
    fallback_reason: str | None = None


class Overview(BaseModel):
    """Header payload: dataset scale, banks, ring index, featured ring."""

    generated_at: str
    source: str
    mock: bool
    dataset: DatasetInfo
    detector: DetectorInfo
    bounding: BoundingInfo
    featured_ring_id: str
    rings: list[RingSummary]
    ring_match: RingMatchInfo
    demo: DemoPlan
    #: Provenance of ``rings`` above. Always present, so a consumer never
    #: has to guess whether the field was omitted or the mode was default.
    ring_list_source: RingListSource


class ScopeMetrics(BaseModel):
    """Mean metric values for one scope (single-bank or cross-bank)."""

    pr_auc: float
    roc_auc: float
    precision_at_k: float
    recall_at_fpr_1pct: float
    false_alerts_per_catch: float
    positives: float | None = None


class SpanComparison(BaseModel):
    """Single-bank vs cross-bank inside one ring-span stratum."""

    span: int
    label: str
    single_bank: ScopeMetrics
    cross_bank: ScopeMetrics
    pr_auc_delta: float
    roc_auc_delta: float


class HorizonMetrics(BaseModel):
    """One early-detection horizon, with its span stratification."""

    cutoff_minutes: int | None
    label: str
    headline: bool
    single_bank: ScopeMetrics
    cross_bank: ScopeMetrics
    pr_auc_delta: float
    spans: list[SpanComparison]
    ring_recall_single_bank: float | None = None
    ring_recall_cross_bank: float | None = None
    #: How many rings still had money in flight at this horizon. Kept
    #: because it is the thing that retired
    #: ``share_caught_before_exit``: that metric is bounded by this count
    #: and tracks it, so it measures the horizon grid rather than the
    #: detector. It is reported, never headlined.
    rings_money_not_yet_at_exit: float | None = None


class SeedCoverage(BaseModel):
    """One seed's coverage for one span, with integer ring counts.

    The headline is a 3-seed mean, whose ring counts are fractional. This
    carries a single run's whole numbers alongside it so a reader can see
    what an actual count looks like, clearly marked as one seed.
    """

    seed: str
    single_bank_rings: int
    cross_bank_rings: int
    single_bank_rate: float
    cross_bank_rate: float


class CoverageSpan(BaseModel):
    """Positive-warning rate over ALL true rings of one span.

    "Positive warning" means flagged at a measured horizon **before the
    money reached the exit account**. That qualifier is the whole point:
    the looser "flagged at some horizon" rate is 1.000 for both scopes at
    span 1 and hides the control result.
    """

    span: int
    label: str
    #: 3-seed mean, so fractional. Counts travel with the rates by
    #: contract: a rate alone can hide the count that produced it.
    rings: float
    single_bank_rings: float
    cross_bank_rings: float
    single_bank_rate: float
    cross_bank_rate: float
    #: Cross minus single. NEGATIVE at span 1, and published rather than
    #: rounded away: a ring wholly inside one bank is already fully
    #: visible to it, so cross-bank buying nothing — or a little less
    #: than nothing — is the control working. It is the strongest
    #: evidence the span-2 and span-3 gaps are not an artefact.
    delta: float
    per_seed: SeedCoverage | None = None


class LeadStats(BaseModel):
    """How much earlier cross-bank flags — which is: not much.

    Published next to the coverage table on purpose. The coverage gap is
    real; a speed gap is not, and quoting one while showing the other is
    the easiest way to mislead with these numbers. Reported on the same
    basis as the coverage table so no sentence mixes a 3-seed rate with a
    single-seed count.
    """

    basis: str
    rings_both_scopes_flag: float
    ties: float
    cross_bank_earlier: float
    single_bank_earlier: float
    median_lead_minutes: float
    mean_lead_minutes: float


class RingCoverage(BaseModel):
    """Ring coverage by span, read from ``ml/experiments/lead_time``.

    Resolved by identity — ``by_scope_and_span[scope][f"span{n}"]`` — and
    never by position in a list. ``docs/claims.md`` cites the same paths,
    so the page cannot quote a number the claims file disagrees with.
    """

    source: str
    source_path: str
    basis: str
    #: One line, on the page, so nobody re-derives the looser measure.
    definition: str
    generated_at: str
    rings: float
    horizon_grid: list[int]
    single_bank_rings: float
    cross_bank_rings: float
    single_bank_rate: float
    cross_bank_rate: float
    spans: list[CoverageSpan]
    lead: LeadStats
    note: str


class RetiredMetric(BaseModel):
    """A metric withdrawn from display, with the reason it was withdrawn.

    Rendered rather than deleted quietly: a number that was on the page
    yesterday and is gone today needs an account of itself, and the
    account has to be the real one.
    """

    name: str
    reason: str
    evidence: list[str]


class CoalitionPoint(BaseModel):
    """Ring recall at one coalition size, averaged over every coalition."""

    banks_joined: int
    #: How many distinct bank subsets of this size were measured. With
    #: three banks there are three pairs, and they are genuinely different
    #: views of the same market, so the mean needs its count beside it.
    coalitions: int
    #: The SHIPPED rule. Its bank-spread clause is uncomputable inside one
    #: bank, so this is 0.0 at ``banks_joined == 1`` -- the rule declining
    #: to fire, not an absence of signal. Quote gains against
    #: :attr:`ring_recall_no_spread`.
    ring_recall: float
    #: The same rule with the bank-spread clause dropped: the generous
    #: reading, and the honest baseline. ``ml/rule.py`` reports its own
    #: ``single_bank`` variant the same way and for the same reason.
    ring_recall_no_spread: float
    ring_recall_min: float
    ring_recall_max: float
    rings_caught: float
    #: Rings in which this coalition holds at least one account. The
    #: numerator's ceiling: a bank cannot act on a ring whose members are
    #: all somebody else's customers, and the gap between this and
    #: ``true_rings_total`` is most of what joining buys.
    rings_reachable: float


class CoalitionWorld(BaseModel):
    """One curve: every coalition size in one partition of the market."""

    label: str
    #: ``False`` for the four-bank curve, whose fourth bank is half of a
    #: real bank's customers relabelled. The page must say so next to the
    #: number, not only in a doc.
    is_real_partition: bool
    note: str
    points: list[CoalitionPoint]


class CoalitionPayload(BaseModel):
    """What the "one more bank joins" control reads.

    Every point is precomputed by ``ml/coalition.py``. The control
    switches between measured numbers; it recomputes nothing, and no
    request to this app runs a detector.
    """

    generated_at: str
    source: Literal["ml", "fallback"]
    #: Built the same way as ``MetricsPayload.source_detail``, so the two
    #: provenance lines on one page cannot name different files.
    source_detail: str
    true_rings_total: int
    metric_definition: str
    #: Why this series must not be read alongside ``docs/claims.md``'s
    #: ring recall: that one is the XGBoost scorer at 1% FPR, this one is
    #: the velocity rule's hard decision.
    not_comparable_to_claims: str
    two_variants: str
    worlds: list[CoalitionWorld]


class RedteamVerdict(BaseModel):
    """One Gemini-designed evasion strategy, run through the real detector.

    ``caught`` is the flat bool a card can badge on; ``caught_by`` and
    ``reasoning`` are the detector's own account of why, copied verbatim
    from ``ml/redteam.py`` rather than re-derived, so the card can never
    disagree with the run that produced it.
    """

    name: str
    description: str
    targets: str
    aimed_at_clause: str
    caught: bool
    caught_by: str
    reasoning: str


class RedteamStrategyRow(BaseModel):
    """One Gemini strategy, reduced to what the counter needs."""

    name: str
    targets_clause: str
    result: str  # the detector's ``caught_by``: missed | velocity_ring | velocity_alert_no_ring
    caught: bool
    failing_clauses: list[str]
    aimed_clause_passed: bool | None
    is_designed_miss: bool


class RedteamPayload(BaseModel):
    """What the red-team card reads.

    Precomputed by ``ml/redteam.py`` against live Gemini -- unlike the
    coalition sweep this is not something any offline tooling in this repo
    ever recomputes, because the script has no rate limit on the model
    call. There is deliberately no "run again" affordance anywhere this
    payload is consumed.
    """

    generated_at: str
    source: Literal["ml", "fallback"]
    source_detail: str
    strategies_run: int
    caught: int
    missed: int
    #: Why a MISSED verdict does not transfer to production traffic, while
    #: a CAUGHT one does. See ``docs/redteam.md``.
    isolated_graph_caveat: str
    #: The XGBoost alert-queue scorer is not in this loop at all; only the
    #: velocity rule is attacked. See ``docs/redteam.md``.
    scorer_not_attacked: str
    verdicts: list[RedteamVerdict]
    #: One row per strategy for the "Gemini's strategies" counter: its name,
    #: the clause it targets, its result and which clauses actually stopped
    #: it. Derived from the same verdicts, never re-run.
    strategies: list[RedteamStrategyRow] = Field(default_factory=list)


class EnclaveOutboxProof(BaseModel):
    """One bank's outbox object, shown as ciphertext and as what it decrypts to.

    The point of this card is a contrast the reader can check themselves:
    ``ciphertext_preview`` is what the operator (and anyone with the GCS
    object) actually sees; ``own_members_sample``/``edges_sample`` is what
    only ``bank_id``'s own private key recovers from it
    (``enclave/outbox_crypto.py``). ``decrypt_failure_*`` records that a
    *different* bank's key on the same object fails outright, rather than
    silently returning garbage.
    """

    bank_id: str
    object_path: str
    ciphertext_bytes_total: int
    #: Truncated -- the full object is a multi-KB base64 blob and showing
    #: more of it would not make the "this is opaque" point any harder.
    ciphertext_preview: str
    own_members_sample: list[str]
    own_members_total: int
    other_bank_member_counts: dict[str, int]
    edges_sample: list[dict[str, str]]
    decrypt_failure_bank_id: str
    decrypt_failure_error: str


class EnclaveDebugRefusal(BaseModel):
    """A second, separate piece of frozen evidence: the unchanged,
    approved image booted on the confidential-space-DEBUG family,
    refused at workload identity federation itself -- before a
    credential, let alone a KMS call, ever existed. Proves a different
    property from the tamper test (operator access, not code identity),
    so it is shown next to it, never folded into the same claim.
    """

    run_id: str
    image_digest: str
    dbgstat: str
    refusal_kind: str
    refusal_message: str
    note: str


class EnclavePayload(BaseModel):
    """What the enclave evidence card reads.

    Captured from one real, credentialed Confidential Space run against
    Google Cloud (docs/enclave.md, Stage 3) -- not something this app, or
    any script in this repo, can recompute: that needs an actual GCP
    project, KMS and a Confidential VM. This payload itself is frozen;
    the live "Run live proof" / "Try tampered code" buttons next to it
    are a separate, real-time orchestration (agents/live_proof_service.py),
    not a recompute of this data.
    """

    generated_at: str
    run_id: str
    attempt: str
    outcome: str
    key_release_ok: bool
    running_debug_image: bool
    image_digest: str
    dbgstat: str
    secboot: bool
    hwmodel: str
    support_attributes: list[str]
    contributing_banks: list[str]
    edge_count: int
    account_count: int
    flagged_count: int
    ring_count: int
    bundles_written: list[str]
    outbox_proof: EnclaveOutboxProof
    note: str
    debug_refusal: EnclaveDebugRefusal
    #: Whether the "Try tampered code" button is shown and served.
    tamper_enabled: bool = False
    #: ``edge_count``/``account_count`` restated under names that say what
    #: they count, with the reason they are smaller than the dataset's
    #: 572,890 transfers / 30,000 accounts (the enclave sees cross-bank
    #: edges only; see ``ui.build_snapshot.dataset_scope_counts``).
    edges_in_enclave_scope: int = 0
    accounts_in_enclave_scope: int = 0
    scope_reason: str = "enclave_sees_cross_bank_edges_only"
    #: ``edge_count``/``account_count`` restated under names that say what
    #: they count, with the reason they are smaller than the dataset's
    #: 572,890 transfers / 30,000 accounts (the enclave sees cross-bank
    #: edges only; see ``ui.build_snapshot.dataset_scope_counts``).
    edges_in_enclave_scope: int = 0
    accounts_in_enclave_scope: int = 0
    scope_reason: str = "enclave_sees_cross_bank_edges_only"
    #: Additive, from the snapshot's dataset block (null if it predates them).
    #: ``accounts_generated`` is every account minted; ``accounts_with_activity``
    #: those with any transfer; ``accounts_in_enclave_scope`` (above) the ones on
    #: a cross-bank transfer, which is all the enclave sees.
    accounts_generated: int | None = None
    accounts_with_activity: int | None = None


class MetricsPayload(BaseModel):
    """The metrics page in one response."""

    model: str
    generated_at: str
    evaluation: str
    onset: str
    exit_account: str
    source: Literal["ml", "fallback"]
    #: Human-readable provenance for the numbers above, DERIVED from
    #: which file was actually opened. ``infra/cloudbuild.ignore``
    #: excludes ``ml/experiments/`` from the container image, so on Cloud
    #: Run the live path does not exist and the bundled copy is what is
    #: served -- a page that names the live file there would be naming a
    #: file it never opened. Built the same way as
    #: ``RingCoverage.source`` so the two lines cannot drift apart.
    source_detail: str
    horizons: list[HorizonMetrics]
    caveat: str
    #: The headline horizon for the ring-level claim: the last finite
    #: horizon, where exit-side censoring is gone.
    headline_ring_recall_horizon: str
    headline_ring_recall_single_bank: float | None = None
    headline_ring_recall_cross_bank: float | None = None
    coverage: RingCoverage | None = None
    retired: list[RetiredMetric] = Field(default_factory=list)
    #: Set ONLY when ``MULENET_READ_MODE=bigquery``. Says that the switch
    #: moved the ring index and nothing on this page, because the
    #: BigQuery dataset has no metrics table to read. ``source_detail``
    #: above already names the file that was opened; this stops a viewer
    #: who saw "ring list from BigQuery" in the header from inferring
    #: that these numbers came from there too.
    read_mode_note: str | None = None


class ChatUiContext(BaseModel):
    """What the console is showing, forwarded to ``agents/`` to ground an answer.

    Validated there against its fixed menus (tab, horizon, bank) and its
    rings table (ring_id); here only bounded, so nothing large is forwarded.
    """

    tab: str | None = Field(default=None, max_length=32)
    ring_id: str | None = Field(default=None, max_length=128)
    horizon: str | None = Field(default=None, max_length=16)
    bank: str | None = Field(default=None, max_length=16)
    #: The chart the question is about. Checked here against the same fixed menu
    #: ``agents/`` uses, so a bad value is a 422 with a field-level error and
    #: nothing is forwarded.
    chart: str | None = Field(default=None, max_length=64)

    @field_validator("chart")
    @classmethod
    def _known_chart(cls, value: str | None) -> str | None:
        if value is not None and value not in metric_catalog.CHART_MENU:
            raise ValueError(f"chart must be one of {list(metric_catalog.CHART_MENU)}")
        return value


class ChatRequest(BaseModel):
    """A question typed into the chat panel."""

    message: str
    ring_id: str | None = None
    #: Optional. ``ring_id`` above still works on its own; when both are
    #: present ``ui_context.ring_id`` wins.
    ui_context: ChatUiContext | None = None


class ChatTrace(BaseModel):
    """How a chat answer was produced: for the "How this was answered" line.

    Kept apart from ``ChatReply.reply`` so the answer text is only the
    answer. ``labels`` holds any internal agent label (``router``,
    ``narrative``, ...) that was found written into the text and removed
    from it. Additive: older clients ignore it.
    """

    agent: str = ""
    backend: str = ""
    sources: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    verified: bool | None = None
    labels: list[str] = Field(default_factory=list)


class ChatReply(BaseModel):
    """The router agent's answer.

    ``backend`` names what actually produced the text. ``"gemini"`` is
    the literal ``agents/`` returns (docs/agents_api.md) and is set only
    when that service answered under that label; the offline mock always
    says ``"mock"``, never ``"gemini"``. That is the whole point of the
    field: a judge can tell a Gemini answer from a keyword match.

    ``offline_demo`` is the visible degradation flag. When the agents
    service errors, times out, is unreachable or rate-limits, the mock
    answers instead and ``offline_notice`` carries the wording the page
    puts on screen. A silent fallback would let a judge believe they were
    talking to Gemini when they were talking to a keyword matcher.
    """

    reply: str
    backend: Literal["mock", "agents", "gemini", "precomputed"]
    agent: str
    citations: list[str] = Field(default_factory=list)
    #: Every regulation passage search_regulations actually returned
    #: this turn (keys: citation, excerpt, source_uri, doc_name,
    #: section, authority), rendered as citation cards independent of
    #: whether `reply`'s own prose quotes them. Distinct from
    #: `citations` above, which is the precomputed STR report's
    #: citation strings -- a different feature. Empty from the mock.
    regulation_citation_cards: list[dict[str, str]] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    #: Tool or table names the answer drew on, as ``agents/`` reports
    #: them. Provenance, not citation. Empty from the mock.
    sources: list[str] = Field(default_factory=list)
    #: ``agents/``'s verifier result: true means every quoted claim was
    #: re-derived from tool data. ``False`` means "nothing here was
    #: machine-checkable", NOT "suspect", and must never render as a
    #: warning. ``None`` when no verifier ran at all (the mock).
    verified: bool | None = None
    #: Server-side wall time reported by ``agents/``.
    latency_ms: int | None = None
    #: True whenever this answer came from the mock while the configured
    #: backend was ``agents``.
    offline_demo: bool = False
    #: The on-screen sentence for this degradation, distinct per failure
    #: case. Empty unless ``offline_demo``.
    offline_notice: str = ""
    #: ``precomputed``: a suggested question answered from the rings table
    #: with no model call. ``model``: a Gemini answer. ``mock``: the offline
    #: stand-in (the default, so real answers must set it explicitly).
    source: str = "mock"
    #: Additive metadata, separate from the prose. ``answer`` repeats
    #: ``reply`` (clean prose, no internal labels); the rest come from
    #: ``agents/`` as-is and are empty from the mock.
    answer: str = ""
    source_details: list[dict[str, str]] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    request_id: str = ""
    #: Set when the live agent failed and the mock answered instead:
    #: ``{code, message, retry_after_seconds}`` in the shape of the UI error
    #: envelope, so a client can react to a 429 or 504 without parsing prose.
    error: dict[str, Any] | None = None
    #: Machine-readable form of ``offline_demo``: ``"live"`` only when the
    #: agents service answered under backend ``gemini``; ``"offline"`` for
    #: the mock, whether it was configured or a fallback. Additive.
    mode: Literal["live", "offline"] = "offline"
    #: Why the live answer was not used, when it was not: the agents
    #: service's ``error_type`` (``malformed_response``, ``no_tool_evidence``,
    #: ``unreachable``...) and its message when it gave one. ``None`` on a
    #: live answer or a mock-only configuration. Additive.
    failure_reason: str | None = None
    failure_detail: str | None = None
    #: Provenance and any internal labels, apart from the answer text.
    trace: ChatTrace = Field(default_factory=ChatTrace)


class ScreenResult(BaseModel):
    """Whether ``hashed_account_id`` is a known mule, at payment time.

    Mirrors ``agents/service.py``'s ``ScreenOut`` (``GET /screen/{id}``),
    with the same ``offline_demo``/``offline_notice`` degradation pair
    :class:`ChatReply` uses: the agents service being unreachable must be
    visible on screen, never silently read as "not flagged".
    """

    hashed_account_id: str
    flagged: bool
    #: "agents" for a real check against agents/'s tables, "mock" when
    #: this server is configured mock-only -- the mock has no honest way
    #: to answer this (see MockRouter.screen), so its own flagged=False
    #: must never be read the same as a real all-clear.
    backend: Literal["mock", "agents"] = "mock"
    ring_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    risk_score: float | None = None
    offline_demo: bool = False
    offline_notice: str = ""


class SimulatePatternValues(BaseModel):
    """One evasion pattern, after ``agents/``'s bounds-clamping.

    Mirrors ``ml.redteam.EvasionPattern`` field for field. ``clamped``
    names any field the judge's raw input pushed outside
    ``ml.redteam.PATTERN_BOUNDS``, exactly as it does for a Gemini-designed
    strategy in ``ml/redteam.py`` -- a verdict on a silently altered
    pattern would be a verdict about the wrong thing.
    """

    collectors: int
    mules_per_collector: int
    onward_accounts: int
    hop_delay_seconds: float
    amount_per_inbound_hop: float
    retained_fraction: float
    counterparty_banks: int
    reuse_device: bool
    reuse_kyc_id: bool
    clamped: list[str] = Field(default_factory=list)


class ClauseValue(BaseModel):
    """One velocity clause's value on one collector, next to its threshold."""

    value: float
    passes: bool
    requirement: str


class SimulateVerdict(BaseModel):
    """One live verdict for a judge-built evasion pattern.

    Mirrors ``agents/service.py``'s ``SimulateOut`` -- itself shaped like
    one entry of ``ml.redteam.as_json()['verdicts']`` -- computed by the
    real, unmodified detection code on this request, not read from a
    snapshot. Carries the same ``offline_demo``/``offline_notice``
    degradation pair :class:`ChatReply` and :class:`ScreenResult` use:
    unlike those two, the mock backend has no honest way to answer this at
    all (there is no offline detector to run), so a mock or unreachable
    ``agents/`` must never be rendered as though a real verdict came back.
    """

    name: str
    description: str
    targets: str
    aimed_at_clause: str
    why_it_should_evade: str
    pattern: SimulatePatternValues
    caught: bool
    caught_by: str
    reasoning: str
    flagged_accounts: int
    flagged_collectors: int
    rings_formed: int
    stage2_alertable: int
    stage2_context: int
    clause_values: dict[str, dict[str, ClauseValue]]
    #: Structured verdict (see shared/evasion_verdict.py) so the page can
    #: write its own copy instead of parsing ``reasoning``.
    verdict: Literal["CAUGHT", "MISSED", ""] = ""
    failing_clauses: list[str] = Field(default_factory=list)
    aimed_clause: str = ""
    aimed_clause_passed: bool | None = None
    is_designed_miss: bool = False
    cash_out_accounts_flagged: int = 0
    #: Per clause: slider ``input_field``/``input_value``, ``expected_measured``,
    #: ``measured`` (+ per collector), ``min``/``max`` bounds, ``passes``.
    clauses: list[dict[str, Any]] = Field(default_factory=list)
    #: Additive. The speed clause's slider value, and what the detector
    #: measured on the first cash-out account (derived from ``clauses``; null
    #: when the verdict carries no speed clause, e.g. the offline mock).
    #: ``delay_added_seconds`` is measured minus input when positive. It is the
    #: inbound leg (the speed feature counts from the FIRST credit while the
    #: slider is held from the LAST), not detector latency; ``delay_explanation``
    #: says so in plain words and is empty when the two agree.
    input_delay_seconds: float | None = None
    measured_delay_seconds: float | None = None
    delay_added_seconds: float | None = None
    delay_explanation: str = ""
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""

    @model_validator(mode="after")
    def _derive_delay(self) -> SimulateVerdict:
        if self.input_delay_seconds is not None or self.measured_delay_seconds is not None:
            return self
        row = next(
            (c for c in self.clauses if c.get("clause") == "passthrough_speed_seconds"), None
        )
        if not row or row.get("measured") is None:
            return self
        given, measured = float(row["input_value"]), float(row["measured"])
        self.input_delay_seconds, self.measured_delay_seconds = given, measured
        added = round(measured - given, 3)
        if added > 0:
            self.delay_added_seconds = added
            leg = (row.get("decomposition") or {}).get("inbound_leg_seconds")
            self.delay_explanation = (
                f"{added:g}s were added because the detector starts the clock at the first "
                "credit into the cash-out account, while your delay is held from the last"
                + (f" (the inbound credits take {float(leg):g}s to arrive)" if leg else "")
                + ". This is how the speed feature is defined, not detector latency."
            )
        return self


class SimulateFieldBound(BaseModel):
    """One pattern field's slider range, off ``ml.redteam.PATTERN_BOUNDS``."""

    low: float
    high: float
    default: float
    integer: bool


class SimulateBounds(BaseModel):
    """Field bounds and clause names for the live pattern-builder form.

    Fetched once, from ``agents/``'s ``GET /simulate/bounds``, so the
    numbers on the sliders can never disagree with what
    ``ml.redteam._coerce_pattern`` will actually clamp to.
    """

    fields: dict[str, SimulateFieldBound]
    evadable_clauses: list[str]
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class SubsetHorizonOption(BaseModel):
    """One selectable horizon for the live "load a subset" panel."""

    key: str
    label: str
    minutes: int


class SubsetOptions(BaseModel):
    """Horizon and bank choices for the live subset panel.

    Fetched once, from ``agents/``'s ``GET /subset/options``, so the
    dropdown can never offer a choice ``agents.subset`` would reject.
    """

    horizons: list[SubsetHorizonOption] = Field(default_factory=list)
    banks: list[str] = Field(default_factory=list)
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class SubsetLoadResult(BaseModel):
    """The "load" step alone: features built for one subset, no verdict.

    Mirrors ``agents/service.py``'s ``SubsetLoadOut``. Carries the same
    ``offline_demo``/``offline_notice`` pair every live-detector panel
    uses: there is no offline equivalent of a real feature build, the
    same reasoning as :class:`SimulateVerdict`.
    """

    horizon: str = ""
    horizon_label: str = ""
    bank: str = "all"
    cutoff: str = ""
    transfers_considered: int = 0
    accounts_in_scope: int = 0
    load_ms: int = 0
    #: ``hit``/``miss`` on the server's cached feature build for this horizon,
    #: what that build cost, and the data it was built from. Empty on a mock.
    cache: str = ""
    feature_build_ms: int = 0
    data_version: str = ""
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class SubsetFlaggedAccount(BaseModel):
    """One account the velocity rule flagged in a live subset run."""

    hashed_account_id: str
    bank_id: str
    caught_cross_bank: bool
    caught_single_bank: bool
    in_degree: int
    out_degree: int
    distinct_counterparty_banks: int
    passthrough_ratio: float
    passthrough_speed_seconds: float


class SubsetScoreResult(BaseModel):
    """Load, then rule on, one subset -- computed live, on this request.

    Mirrors ``agents/service.py``'s ``SubsetScoreOut``. Same
    ``offline_demo``/``offline_notice`` pair as :class:`SubsetLoadResult`.
    """

    horizon: str = ""
    horizon_label: str = ""
    bank: str = "all"
    cutoff: str = ""
    transfers_considered: int = 0
    accounts_in_scope: int = 0
    cross_bank_flagged: int = 0
    single_bank_flagged: int = 0
    flagged_accounts: list[SubsetFlaggedAccount] = Field(default_factory=list)
    flagged_accounts_truncated: bool = False
    #: "model_a" or "rule_fallback". See agents/service.py:SubsetScoreOut.
    detector: str = ""
    load_ms: int = 0
    score_ms: int = 0
    total_ms: int = 0
    cache: str = ""
    feature_build_ms: int = 0
    data_version: str = ""
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class LiveProofStartResult(BaseModel):
    """Answer to ``POST /api/live-proof/start``.

    Mirrors ``agents/service.py``'s ``LiveProofStartOut``. A cap hit is
    ``capped: true`` with no ``run_id`` -- never an error -- so the page
    can fall back to the static evidence card with a plain count rather
    than a dead button. ``offline_demo``/``offline_notice`` cover the
    other case the mock has no honest answer for: the agents service
    being unreachable at all.
    """

    capped: bool = False
    run_id: str | None = None
    runs_today: int | None = None
    limit: int | None = None
    #: True: ``run_id`` is a run already in flight that this request joined.
    attached: bool = False
    #: When ``capped``: seconds until a slot frees up.
    retry_after_seconds: int | None = None
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class LiveProofStatusResult(BaseModel):
    """Answer to ``GET /api/live-proof/status/{run_id}``.

    Mirrors ``agents/service.py``'s ``LiveProofStatusOut``. ``extra``'s
    shape depends on ``stage`` -- see ``enclave/live_proof.py``'s
    ``on_stage`` call sites.
    """

    run_id: str = ""
    stage: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    #: Cloud Logging's record of the run, from the agents service; see
    #: ``agents/live_proof_logs.py`` for the shape.
    log_trail: dict[str, Any] | None = None
    #: Real per-step progress (``plan`` is what will happen, ``steps`` what has,
    #: each with start/finish times) and the finished run's headline facts.
    tampered: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    plan: list[dict[str, str]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    steps_done: int = 0
    steps_total: int = 0
    summary: dict[str, Any] | None = None
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""


class UploadFlaggedAccount(BaseModel):
    """One account the velocity rule flagged in a judge-uploaded run.

    Same shape as :class:`SubsetFlaggedAccount` on purpose: both render
    through the same card in ``ui/static/app.js``.
    """

    hashed_account_id: str
    bank_id: str
    caught_cross_bank: bool
    caught_single_bank: bool
    caught_by_rule: bool = False
    in_degree: int
    out_degree: int
    distinct_counterparty_banks: int
    passthrough_ratio: float
    passthrough_speed_seconds: float


class UploadScoreResult(BaseModel):
    """A judge's own per-bank CSV upload, scored live.

    Mirrors ``agents/service.py``'s ``UploadScoreOut``. Same
    ``offline_demo``/``offline_notice`` pair as :class:`SubsetScoreResult`:
    there is no offline equivalent of a real feature build over uploaded
    data either.
    """

    banks: list[str] = Field(default_factory=list)
    transfers_considered: int = 0
    duplicate_transfers_removed: int = 0
    accounts_in_scope: int = 0
    out_of_perimeter_accounts: int = 0
    ownership_ambiguous_accounts: int = 0
    cross_bank_flagged: int = 0
    single_bank_flagged: int = 0
    rule_cross_bank_flagged: int = 0
    flagged_accounts: list[UploadFlaggedAccount] = Field(default_factory=list)
    flagged_accounts_truncated: bool = False
    #: "model_c" or "rule_fallback"; never "model_a" or "model_b". See
    #: agents/service.py:UploadScoreOut.
    detector: str = ""
    #: Model C's honest T=5m/T=180m validated performance, present only
    #: when detector == "model_c". The judge-upload panel must show both
    #: numbers plainly, not bury them -- see ui/static/app.js's
    #: renderUploadResult. Model C (ml/train_model_c.py) was retrained
    #: without account_age_days specifically because this path cannot
    #: supply a real account-open date -- see
    #: ml/experiments/ablation/README.md.
    model_c_validated_performance: dict[str, dict[str, float]] | None = None
    load_ms: int = 0
    score_ms: int = 0
    total_ms: int = 0
    backend: Literal["mock", "agents"] = "mock"
    offline_demo: bool = False
    offline_notice: str = ""
    #: A validation failure (malformed/oversized/conflicting upload), as
    #: agents.subset_upload.UploadValidationError's own message -- kept
    #: separate from offline_notice/offline_demo on purpose: those mean
    #: "the agents service is unreachable", and the judge's file being
    #: rejected is a completely different, non-degraded outcome that
    #: should never be labelled "offline demo mode".
    error: str | None = None
    #: The same rejection as data: ``[{bank, row, field, message}]``, at most
    #: 20 per file, never containing a cell's content. Empty when accepted.
    errors: list[dict[str, Any]] = Field(default_factory=list)


class DemoState(BaseModel):
    """Result of pressing Demo Mode."""

    reset_at: str
    featured_ring_id: str
    reason: str
    replay_transfer_count: int
    replay_duration_ms: int
    #: The derived two-ring walkthrough: the 3-bank ring the argument
    #: rests on, then the 1-bank control it is read against.
    plan: DemoPlan
