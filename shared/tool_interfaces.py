"""Tool interfaces for the MuleNet ADK agents.

Signatures and docstrings only -- ``agents/`` provides the
implementations. Types come from :mod:`shared.contracts`; no module
may redefine them locally (see CLAUDE.md). Every account identifier
crossing these boundaries is a hashed account id.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from shared.contracts import (
    Account,
    AccountEvidence,
    AccountExplanation,
    ExplanationFactor,
    GnnScore,
    Ring,
    RingExplanation,
    RiskScore,
    Transfer,
)

#: Tables :func:`run_sql` may query. Any other table is rejected.
#:
#: ``str_reports`` is deliberately **not** in this set, although it has a
#: table in ``shared/ddl.sql`` and a schema in ``shared/bq_schemas.py``.
#: It holds generated narratives with citations that are currently
#: synthetic placeholders. An agent able to ``SELECT`` prior reports
#: could retrieve one and cite it as corroboration, turning an
#: unverified draft into apparent evidence for the next draft — a
#: laundering-of-provenance loop in which a placeholder citation
#: eventually reads as an established prior finding. Every fact in an
#: STR must trace to a primary table (``transfers``, ``rings``,
#: ``account_evidence``, ``scores``), never to another report. ``ui/``
#: still reads ``str_reports`` directly; the restriction is on the
#: agent's SQL tool, not on the data. Revisit only when reports carry a
#: human sign-off, and then admit ``status = 'filed'`` rows alone.
ALLOWED_SQL_TABLES: frozenset[str] = frozenset(
    {
        "accounts",
        "transfers",
        "devices",
        "kyc_submissions",
        "kyc_documents",
        "rings",
        # Ring membership with its basis. In the allowlist because the
        # basis is the difference between "this account forwarded money
        # fast" and "this account shares a phone with someone who did":
        # an agent that can query rings but not ring_members can only
        # get member accounts through a join it is not allowed to make,
        # or by asking for a flat list that has lost the distinction.
        # Any query counting alerts must filter on alertable.
        "ring_members",
        "cases",
        # ``features`` is deliberately **not** here, although it has a
        # schema in ``shared/bq_schemas.py``. Nothing publishes it: the
        # only file of that name is ``ml/features.parquet``, the ML
        # feature frame, which is one wide row per account with
        # ``sb_``/``cb_`` prefixed columns and no ``scope`` column --
        # not the long (account, as_of, scope) shape this contract's
        # ``features`` table specifies. The BigQuery table is created
        # empty and has 0 rows. Allowlisted, it made ``SELECT ... FROM
        # features`` succeed and return nothing, which reads to an agent
        # as "no such account" rather than "no such table". An
        # allowlisted table that cannot be read is a promise the system
        # does not keep. Re-admit it when something publishes rows in
        # the contract's shape.
        "scores",
        "ring_explanations",
        "account_explanations",
        "account_evidence",
    }
)

#: Row limit enforced on every :func:`run_sql` query.
SQL_ROW_LIMIT = 100

#: Ceilings on the lists :func:`get_ring_subgraph` returns.
#:
#: These live in the contract rather than in ``agents/`` for the same
#: reason :data:`SQL_ROW_LIMIT` does: a caller cannot read the answer
#: correctly without knowing the answer may be capped.
#:
#: They exist because the unbounded version killed a live request. The
#: model called the tool without ``max_hops``, took the former default
#: of 2, and got back 29,582 accounts and 185,011 transfers --
#: 49,001,023 tokens against a 1,048,576 limit, 47x over. The same ring
#: at one hop is 1,289 accounts and 2,199 transfers. Two hops is not a
#: slightly larger answer, it is a different order of magnitude: every
#: mule account is a hub, so the second hop pulls in the counterparties
#: of every counterparty. Hence ``max_hops`` also defaults to 1.
MAX_SUBGRAPH_ACCOUNTS = 200
MAX_SUBGRAPH_TRANSFERS = 400


#: Decimal places every score-like float is rounded to **at the tool
#: boundary**, before it reaches an agent.
#:
#: It applies to exactly three fields and nothing else:
#:
#: * :attr:`shared.contracts.RiskScore.score`
#: * :attr:`shared.contracts.AccountExplanation.score`
#: * :attr:`shared.contracts.ExplanationFactor.contribution`
#:
#: **This is not a storage precision.** ``ml/`` writes full-precision
#: floats, and ``shared/ddl.sql`` and ``shared/bq_schemas.py`` keep them
#: as ``FLOAT64``. It is a promise about what the tools in this file
#: hand back, so that the number an agent quotes and the number a
#: checker recomputes are the same number.
#:
#: It exists because the raw float is what leaks: a live answer came
#: back as ``0.99999809265136719`` where ``1.000`` was expected and
#: failed a ``1e-6`` comparison by ``1.9e-6``. Three places is what the
#: prompts ask agents to quote, so the tool hands them a number already
#: in that form rather than trusting every consumer to round it
#: identically.
#:
#: It lives here rather than in :mod:`shared.contracts` for the same
#: reason :class:`RingSubgraph` does. ``contracts.py`` describes
#: persisted entities, each with a row in ``ddl.sql`` or
#: ``bq_schemas.py`` behind it, and **the stored rows are not rounded**
#: -- putting the promise on the record would make ``contracts.py``
#: state something untrue of the data it models. This is a property of
#: the *call*, and it is the same kind of fact as :data:`SQL_ROW_LIMIT`
#: and :data:`MAX_SUBGRAPH_ACCOUNTS`: a caller cannot read the answer
#: correctly without knowing it.
#:
#: **Rounding preserves sign.** ``contribution`` is signed and the sign
#: is load-bearing: it is the whole difference between a mitigating
#: factor and an incriminating one. :func:`round` never moves a value
#: across zero, so a negative contribution stays negative (a magnitude
#: below half a unit in the last place becomes ``-0.0``, which still
#: compares ``<= 0`` and still formats with its minus sign). Any
#: implementation of this rounding must keep that property; a rounding
#: that took an absolute value, clamped at zero, or normalised ``-0.0``
#: to ``0.0`` would silently reclassify factors.
#:
#: The one thing rounding does take away is direction at very small
#: magnitudes: a contribution whose magnitude is under
#: :data:`SCORE_TOLERANCE` rounds to zero, and a consumer must read that
#: as "did not measurably move the score", not as evidence of a
#: direction.
SCORE_DECIMALS = 3

#: Tolerance for comparing a score that crossed the tool boundary
#: against one recomputed from the tools -- used by the investigator's
#: claim verifier and by anything else grading a quoted number.
#:
#: **Derived from :data:`SCORE_DECIMALS`. It is never a literal.** Half
#: of one unit in the last retained place is exactly the error rounding
#: to ``SCORE_DECIMALS`` places can introduce: a value rounded to ``d``
#: places sits at most ``0.5 * 10**-d`` from the value it was rounded
#: from. Two numbers agreeing inside that bound are therefore the same
#: number seen at two precisions; two that do not agree inside it
#: differ for a reason other than rounding. At ``SCORE_DECIMALS = 3``
#: this evaluates to exactly ``0.0005``.
#:
#: The derivation is the point. A hand-written ``0.0005`` is correct
#: only by coincidence with the current rounding, and drifts wrong in
#: **either** direction the moment the rounding moves:
#:
#: * Round to fewer places (2) and real rounding differences reach
#:   ``0.005``. A frozen ``0.0005`` is then ten times too tight and
#:   rejects correct answers as discrepancies.
#: * Round to more places (4) and rounding differences never exceed
#:   ``0.00005``. A frozen ``0.0005`` is then ten times too loose and
#:   **accepts a genuine discrepancy of 0.0004 as a rounding artefact**
#:   -- an invisible failure, because everything still passes.
#:
#: **No consumer may define its own tolerance for these values.** A
#: module that writes ``0.0005`` (or ``1e-6``, or any other literal) has
#: reintroduced exactly the bug this constant exists to make
#: impossible. Import this name. If a consumer needs a tolerance at a
#: different precision, derive it from :data:`SCORE_DECIMALS` here and
#: give it a name here, rather than inventing a number locally.
SCORE_TOLERANCE = 0.5 * 10**-SCORE_DECIMALS


#: The score and ratio fields :func:`round_score_fields` rounds, named
#: model by model. **An explicit registry, not a name convention.**
#:
#: A convention (round every field whose name ends ``_score``, ``score``
#: or ``_ratio``) is automatic and covers a new field for free, and it
#: was rejected anyway. The two failure modes are not symmetric. A
#: *missed* field is the bug that exists today — a tool returns an
#: unrounded value, and :func:`assert_score_fields_rounded` now catches
#: it loudly at the boundary. A *wrong match* is silent and permanent:
#: the helper quietly degrades a value that had to keep its precision,
#: and nothing downstream can tell. The obvious candidate for a wrong
#: match is money. :attr:`shared.contracts.Transfer.amount` is a
#: ``Decimal`` quantized to two places and specified as
#: ``NUMERIC(15, 2)`` in ``shared/ddl.sql``; a numeric-looking-field
#: rule that reached it would corrupt money, and "the amount is off by a
#: tenth of a cent" is not a bug anyone finds by reading a narrative.
#: The registry excludes it by construction — ``Transfer`` is not a key
#: here — and :func:`round_score_fields` additionally refuses any
#: registered field that is not a ``float``, so registering money by
#: mistake is a ``TypeError`` rather than a rounding.
#:
#: The cost of the registry is that it must be maintained: a new score
#: field is rounded only once it is added here. That cost is paid at
#: review time by a human who can say whether the field is a score, and
#: the contract's own docstrings (:class:`~shared.contracts.RiskScore`,
#: :class:`~shared.contracts.AccountExplanation`,
#: :class:`~shared.contracts.ExplanationFactor`) point at this promise
#: so the reader of the model is told to come here.
#:
#: **What is in it, and what is deliberately not:**
#:
#: * :attr:`shared.contracts.RiskScore.score` and
#:   :attr:`shared.contracts.AccountExplanation.score` — the two scores
#:   a tool hands an agent, via :func:`get_account_profile` and
#:   :func:`get_explanation`.
#: * :attr:`shared.contracts.ExplanationFactor.contribution` — signed,
#:   and reached by recursion through ``AccountExplanation.top_factors``.
#: * :attr:`shared.contracts.GnnScore.score` — no tool returns it today,
#:   because the GNN was evaluated and not shipped. It is registered
#:   anyway: it is the same quantity as ``RiskScore.score`` under a
#:   different model family, lands in the same ``scores.score``
#:   ``FLOAT64`` column with ``model_type = 'gnn'``, and is an explicit
#:   documented swap-in for the XGBoost baseline. Registering it now
#:   means shipping the GNN does not also require remembering this.
#: * The ratio-shaped values ``passthrough_ratio`` and
#:   ``round_amount_ratio`` are **not** here, and cannot be: they are
#:   columns of ``FEATURES_SCHEMA`` in ``shared/bq_schemas.py`` with no
#:   Pydantic model behind them, so nothing typed ever carries them
#:   across this boundary. They must also stay unrounded on principle —
#:   they are model *inputs*, and rounding a feature changes the score it
#:   produces. If a ``Features`` model is ever added to
#:   ``contracts.py``, that is the moment to decide field by field, not
#:   a reason to match on the suffix now.
_ROUNDED_FIELDS: dict[type[BaseModel], frozenset[str]] = {
    RiskScore: frozenset({"score"}),
    GnnScore: frozenset({"score"}),
    AccountExplanation: frozenset({"score"}),
    ExplanationFactor: frozenset({"contribution"}),
}

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _round_registered(model: BaseModel, name: str, value: Any) -> Any:
    """Round one registered field, refusing anything that is not a float."""
    if value is None:
        return None
    if not isinstance(value, float):
        raise TypeError(
            f"{type(model).__name__}.{name} is registered as a score field but "
            f"holds {type(value).__name__} {value!r}. Only float scores and "
            "ratios are rounded here: money is a Decimal (MoneyAmount, "
            "NUMERIC(15, 2)) and must never pass through this rounding."
        )
    return round(value, SCORE_DECIMALS)


def round_score_fields(model: _ModelT) -> _ModelT:
    """Return ``model`` with every registered score field rounded.

    This is the mechanical form of the promise :data:`SCORE_DECIMALS`
    describes. A tool that returns a contract model should pass it
    through here rather than rounding named fields by hand, so that a
    model gaining a score field does not silently return an unrounded
    one from every tool that forgot to be updated.

    **Which fields.** Exactly those in the registry above, keyed by the
    model's own class. A model with no entry is returned unchanged —
    which is what makes recursion safe rather than a blanket numeric
    sweep. Money is never touched: ``Transfer`` has no entry, and a
    registered field holding anything other than a ``float`` raises
    ``TypeError`` instead of being rounded.

    **Nesting.** Recursion follows the *typed contract structure*, not a
    depth limit: every field whose value is a ``BaseModel``, and every
    ``BaseModel`` inside a ``list`` field, is visited. That is how
    ``AccountExplanation.top_factors`` — a ``list[ExplanationFactor]`` —
    gets its contributions rounded. No model in
    :mod:`shared.contracts` refers to itself or to an ancestor, so the
    graph is a finite tree and today is at most two levels deep
    (``AccountExplanation`` -> ``ExplanationFactor``, ``Ring`` ->
    ``RingMember``, ``KycSubmission`` -> ``KycDocument``); the recursion
    is written against the structure rather than that number so a third
    level costs nothing.

    **It never mutates.** A new instance comes back via
    ``model_copy``; the input is left exactly as it was, and the same
    instance is returned when nothing needed rounding. Mutating in place
    would be wrong, not merely impure: ``agents/tools.py`` hands out
    models read from a shared in-memory store, so rounding in place
    would round the *stored* row. The next consumer needing full
    precision — the verifier re-deriving a claim — would then compare a
    rounded value against a rounded value and find them equal by
    construction, making :data:`SCORE_TOLERANCE` vacuous. Rounding is a
    property of the call, never of the record.

    **The sign survives.** :func:`round` never moves a value across
    zero, and nothing is done to the result afterwards — no ``abs``, no
    clamp, no ``+ 0.0`` — so a negative contribution stays negative and
    a magnitude under :data:`SCORE_TOLERANCE` becomes ``-0.0``, which
    still compares ``<= 0`` and still formats with its minus sign.

    Args:
        model: Any :mod:`shared.contracts` model.

    Returns:
        ``model`` itself when no registered field needed rounding,
        otherwise a copy with the rounded values.

    Raises:
        TypeError: If a registered field holds a non-``float``.
    """
    rounded_names = _ROUNDED_FIELDS.get(type(model), frozenset())
    updates: dict[str, Any] = {}

    for name in type(model).model_fields:
        value = getattr(model, name)

        if name in rounded_names:
            new_value = _round_registered(model, name, value)
            # ``!=`` and not ``is not``: ``round`` returns a fresh object
            # even when the value is unchanged, and comparing by identity
            # would copy every model on every call.
            if new_value != value:
                updates[name] = new_value
            continue

        if isinstance(value, BaseModel):
            nested = round_score_fields(value)
            if nested is not value:
                updates[name] = nested
        elif isinstance(value, list):
            items = [
                round_score_fields(item) if isinstance(item, BaseModel) else item
                for item in value
            ]
            if any(new is not old for new, old in zip(items, value, strict=True)):
                updates[name] = items

    return model if not updates else model.model_copy(update=updates)


def assert_score_fields_rounded(model: BaseModel) -> None:
    """Raise unless every registered score field on ``model`` is rounded.

    The check, kept separate from :func:`round_score_fields` on
    purpose. The rounder makes a value correct; this says whether it
    already was. A tool that rounds by hand — or one whose model grew a
    score field the tool does not know about — can call this on the
    value it is about to return and fail loudly at the boundary instead
    of handing an agent a raw float that a downstream comparison later
    rejects for reasons no one can trace. That is the live q16 failure:
    ``0.99999809265136719`` reached a comparison that wanted ``1.000``
    and missed by ``1.9e-6``, with nothing in between to object.

    Recursion, field selection and the money exclusion are exactly those
    of :func:`round_score_fields`.

    **It never changes anything, and ``-0.0`` is not a violation.**
    ``round(-0.0, d)`` is ``-0.0``, so a negative contribution that
    rounded to zero passes unflagged; and because ``-0.0 == 0.0`` in
    Python the check cannot tell the two apart and therefore cannot
    normalise one into the other. A validator that "fixed" ``-0.0`` to
    ``0.0`` would reclassify a mitigating factor, which is the failure
    this whole promise exists to prevent.

    Args:
        model: Any :mod:`shared.contracts` model.

    Raises:
        ValueError: If a registered field is not rounded to
            :data:`SCORE_DECIMALS` places. The message names the full
            field path — ``AccountExplanation.top_factors[2]
            .contribution`` — and both the value found and the value
            expected, because a tool author reading it needs to know
            which field broke the promise, not merely that one did.
            ``ValueError`` and not ``assert``: this is a contract
            violation of the same class as the model validators in
            ``shared/contracts.py``, and ``assert`` vanishes under
            ``python -O``, which would silently remove the check from
            exactly the production run it guards.
        TypeError: If a registered field holds a non-``float``.
    """
    _assert_rounded(model, type(model).__name__)


def _assert_rounded(model: BaseModel, path: str) -> None:
    """Recursive worker for :func:`assert_score_fields_rounded`."""
    rounded_names = _ROUNDED_FIELDS.get(type(model), frozenset())

    for name in type(model).model_fields:
        value = getattr(model, name)
        here = f"{path}.{name}"

        if name in rounded_names:
            expected = _round_registered(model, name, value)
            if value is not None and value != expected:
                raise ValueError(
                    f"{here} is not rounded to {SCORE_DECIMALS} places: found "
                    f"{value!r}, expected {expected!r}. Pass the value through "
                    "shared.tool_interfaces.round_score_fields() before it "
                    "crosses the tool boundary."
                )
            continue

        if isinstance(value, BaseModel):
            _assert_rounded(value, here)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _assert_rounded(item, f"{here}[{index}]")


@dataclass(frozen=True)
class RingSubgraph:
    """A ring's local subgraph, carrying whether it had to be cut down.

    **Ring members are never truncated; only counterparties are.** This
    is a correctness property, not a convenience. Every claim the
    verifier re-derives -- ``ring_member_count``,
    ``ring_alertable_member_count``, ``ring_bank_count``,
    ``ring_banks``, ``account_in_ring`` -- is about the member
    population. A capped member list would not make the answer smaller,
    it would make it **wrong**: the investigator would state a member
    count derived from a truncated list, and a size limit would have
    turned itself into a false statement about how many accounts were
    implicated. The cap therefore falls entirely on the counterparty
    accounts, which are context, and on transfers.

    **Transfers are kept largest-by-amount.** If only some can be shown,
    the ones moving the most money are the ones worth seeing.

    **The flag matters more than the cap.** A model that believes it saw
    the whole subgraph will confidently state totals derived from a
    partial one. :attr:`truncated` and :attr:`note` exist so the model
    knows data was cut, and :attr:`total_accounts` /
    :attr:`total_transfers` give it the real numbers to quote instead.

    **Iterating yields ``(ring, accounts, transfers)``**, so the
    three-tuple unpacking this tool used to return keeps working
    unchanged at all fifteen existing ``ring, accounts, transfers = ...``
    call sites. The truncation data is on attributes rather than in the
    tuple for exactly that reason: putting it in the tuple would break
    every caller in order to tell them something only the model needs.

    Not a Pydantic model and not in :mod:`shared.contracts`: this is a
    per-call tool response, not a persisted entity. Every model in
    ``contracts.py`` has a row in ``ddl.sql`` or ``bq_schemas.py``
    behind it; this has neither and never will. It is also never parsed
    from untrusted input -- it is assembled from already-validated
    :class:`~shared.contracts.Ring`, :class:`~shared.contracts.Account`
    and :class:`~shared.contracts.Transfer` values -- so validation
    would buy nothing, while a Pydantic ``BaseModel`` would make the
    :meth:`__iter__` override above actively harmful: Pydantic's own
    ``__iter__`` yields ``(field_name, value)`` pairs and backs
    ``dict(model)``.

    Attributes:
        ring: The ring itself, with all of its members. Never truncated.
        accounts: Accounts in the subgraph -- every ring member, plus as
            many counterparties as fit under
            :data:`MAX_SUBGRAPH_ACCOUNTS`.
        transfers: Transfers connecting them, capped at
            :data:`MAX_SUBGRAPH_TRANSFERS`, largest by amount kept.
        truncated: Whether either list was cut. Check this before
            counting anything that is not a member count.
        total_accounts: How many accounts the subgraph actually has.
        total_transfers: How many transfers it actually has.
        note: Plain-language warning naming what was cut. Empty string
            when nothing was.
    """

    ring: Ring
    accounts: list[Account]
    transfers: list[Transfer]
    truncated: bool
    total_accounts: int
    total_transfers: int
    note: str = ""

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(ring, accounts, transfers)`` for tuple unpacking."""
        return iter((self.ring, self.accounts, self.transfers))


def get_ring_subgraph(
    ring_id: str,
    max_hops: int = 1,
) -> RingSubgraph:
    """Fetch a detected ring together with its local transaction subgraph.

    **Ring members are not one population.** Each
    :class:`~shared.contracts.RingMember` carries a
    ``membership_basis``: ``velocity`` members were detected on their
    own behaviour, ``identity_link`` members were pulled in because they
    share a device fingerprint or hashed id number with a member. A
    caller must describe an ``identity_link`` member as "linked, needs
    review" and never as a mule. ``member.alertable`` says whether the
    membership may be published as an alert at all — it is ``False`` for
    same-bank identity links, which are context only. Use
    ``ring.all_member_ids`` for subgraph expansion and
    ``ring.alertable_member_ids`` for anything that counts or accuses.

    Args:
        ring_id: Identifier of the ring to expand.
        max_hops: How many transfer hops out from the ring members to
            include. Defaults to 1. Expansion starts from
            ``ring.all_member_ids``: a context-only member's
            counterparties are still worth showing an analyst. Two hops
            on a real ring returns tens of thousands of accounts (see
            :data:`MAX_SUBGRAPH_ACCOUNTS`); ask for it only when one hop
            was not enough.

    Returns:
        A :class:`RingSubgraph`. It unpacks as the
        ``(ring, accounts, transfers)`` triple this tool used to return,
        and additionally reports whether those lists were capped at
        :data:`MAX_SUBGRAPH_ACCOUNTS` and
        :data:`MAX_SUBGRAPH_TRANSFERS`, with the true totals.

        **Check ``truncated`` before counting anything.** Ring members
        are always returned in full, so member counts are always safe;
        counterparty and transfer counts are not, when the flag is set.
        Quote ``total_accounts`` and ``total_transfers`` rather than
        ``len()`` of the returned lists.

    Raises:
        KeyError: If no ring exists with ``ring_id``.
        ValueError: If ``max_hops`` is less than 1.
    """
    raise NotImplementedError


def get_explanation(
    subject_id: str,
    subject_type: str = "ring",
) -> RingExplanation | AccountExplanation:
    """Fetch the stored model explanation for a ring or an account.

    The two subject types are backed by two tables, ``ring_explanations``
    and ``account_explanations``, not by one polymorphic table.

    Args:
        subject_id: A ring id when ``subject_type`` is ``"ring"``, or a
            hashed account id when it is ``"account"``.
        subject_type: Either ``"ring"`` or ``"account"``.

    Returns:
        A :class:`~shared.contracts.RingExplanation` for a ring, or an
        :class:`~shared.contracts.AccountExplanation` for an account.
        The account form is produced by whichever model is shipped —
        today the XGBoost baseline — and its ``top_factors`` are
        :class:`~shared.contracts.ExplanationFactor` records carrying
        analyst-readable text and a signed contribution, never raw
        feature column names.

        ``AccountExplanation.score`` and every
        ``top_factors[].contribution`` are rounded to
        :data:`SCORE_DECIMALS` places before being returned; the
        rounding preserves the sign on a contribution. Compare a quoted
        value against a recomputed one with :data:`SCORE_TOLERANCE`.

    Raises:
        KeyError: If no explanation has been generated for ``subject_id``.
        ValueError: If ``subject_type`` is not ``"ring"`` or ``"account"``.
    """
    raise NotImplementedError


def trace_money_flow(
    hashed_account_id: str,
    direction: str = "downstream",
    max_hops: int = 3,
    since: Optional[datetime] = None,
) -> list[list[Transfer]]:
    """Trace chains of transfers into or out of an account.

    Follows funds hop by hop to expose pass-through behaviour, including
    across bank boundaries.

    Args:
        hashed_account_id: Hashed account id to start from.
        direction: ``"downstream"`` to follow outgoing transfers,
            ``"upstream"`` to follow incoming ones.
        max_hops: Maximum chain length to return.
        since: Only consider transfers at or after this timestamp. When
            ``None``, the full retained history is used.

    Returns:
        One list of transfers per traced path, ordered along the flow of
        funds and sorted by descending total amount.

    Raises:
        KeyError: If ``hashed_account_id`` is unknown.
        ValueError: If ``direction`` is invalid or ``max_hops`` is less than 1.
    """
    raise NotImplementedError


def get_account_profile(
    hashed_account_id: str,
) -> tuple[Account, Optional[RiskScore], list[str]]:
    """Fetch an account with its latest risk score and ring memberships.

    Args:
        hashed_account_id: Hashed account id to look up.

    Returns:
        A tuple of the account, its most recent risk score (``None`` if the
        account has never been scored), and the ids of rings it belongs to.

        :attr:`~shared.contracts.RiskScore.score` is rounded to
        :data:`SCORE_DECIMALS` places before being returned. Compare a
        quoted score against a recomputed one with
        :data:`SCORE_TOLERANCE`; do not write a tolerance literal.

    Raises:
        KeyError: If ``hashed_account_id`` is unknown.
    """
    raise NotImplementedError


def get_document_reuse(
    hashed_account_id: str,
) -> list[AccountEvidence]:
    """Find KYC documents this account shares with other accounts.

    Reuse is detected on the hashed id number, the normalized address and
    the photo embedding reference produced by ``docai/``.

    **This is evidence, not a detection.** Document reuse plays no part
    in the firing decision, and it is a weak signal on its own: far more
    legitimate customers reuse a document within one bank than mules do,
    and dual-bank customers legitimately reuse an id across two banks.
    Callers must therefore read every row together with its
    :attr:`~shared.contracts.AccountEvidence.scope` and must not present
    a ``single_bank`` row as incriminating.

    Args:
        hashed_account_id: Hashed account id to check.

    Returns:
        One :class:`~shared.contracts.AccountEvidence` row per reuse
        observed, each naming the reused attribute in ``evidence_type``,
        whether it is ``single_bank`` or ``cross_bank`` in ``scope``, the
        counterparty accounts and the banks they sit at. Rows from this
        tool always carry ``source_table="kyc_documents"``. Empty when
        the account reuses nothing.

        The flat ``dict[str, list[KycDocument]]`` this once returned is
        deliberately gone: it could not carry the counterparty bank or
        the scope, which is exactly what makes a reuse fact citable in a
        report rather than merely alarming.

    Raises:
        KeyError: If ``hashed_account_id`` has no KYC submission on file.
    """
    raise NotImplementedError


def search_regulations(
    query: str,
    top_k: int = 5,
) -> list[dict[str, str]]:
    """Search the AML regulation corpus for passages to cite in a report.

    Backed by Vertex AI Search over the indexed regulation documents.

    Args:
        query: Natural-language description of the suspicious behaviour.
        top_k: Maximum number of passages to return.

    Returns:
        A list of results ordered by relevance, each with the keys
        ``"citation"`` (regulation and section reference), ``"excerpt"``
        (the matching text) and ``"source_uri"``.

    Raises:
        ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        RuntimeError: If the search backend is unavailable.
    """
    raise NotImplementedError


def run_sql(query: str) -> list[dict[str, Any]]:
    """Run an analyst-supplied read-only query against the MuleNet tables.

    The query is validated before execution and must satisfy all of:

    * It is a single read-only ``SELECT`` statement. DML, DDL, and any
      other statement type is rejected, as are multiple statements.
    * Every table it references is in :data:`ALLOWED_SQL_TABLES`.
    * A ``LIMIT`` of at most :data:`SQL_ROW_LIMIT` (100) is enforced; if
      the query omits a limit, one is appended.

    Args:
        query: The ``SELECT`` statement to run.

    Returns:
        Up to :data:`SQL_ROW_LIMIT` rows, each as a column-name to value
        mapping.

    Raises:
        ValueError: If the query is not a single ``SELECT``, references a
            table outside the allowlist, or requests more than
            :data:`SQL_ROW_LIMIT` rows.
        RuntimeError: If the query is valid but fails to execute.
    """
    raise NotImplementedError
