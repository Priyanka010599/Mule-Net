"""Deterministic offline implementations of the four agents.

This environment has no Google Cloud credentials and no Gemini access,
so every agent needs a path that runs with no cloud at all. These are
those paths: same inputs, same tools, same output types as the Gemini
agents in :mod:`agents.adk_agents`, but the language step is a template
and the routing step is a keyword matcher.

**What this does and does not prove.** Everything below reads the real
``ml/out`` and ``simulator/out`` tables through :mod:`agents.tools`, so
running the eval here genuinely exercises the tools, the data layer, the
SQL guard and the verifier. It does not exercise language understanding:
a keyword matcher is not a model, and eval accuracy measured in this
mode is a measurement of the tool and data path, not of Gemini. Read it
as a floor and a regression guard, not as a model score.

The rules the prompts encode are enforced here in code, so mock mode
cannot drift from the documented behaviour:

* Narratives cite velocity and structure as the reason for a detection.
* Identity reuse appears only under supporting evidence, always with its
  scope, and a ``single_bank`` row is labelled weak.
* The report agent drops any ground it cannot attach a citation to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents import regulations, tools
from agents.verify import Claim, ClaimKind, InvestigationResult
from shared.contracts import EvidenceScope, MembershipBasis
from shared.tool_interfaces import SCORE_DECIMALS

#: Destinations the router may choose. Matches ``prompts/router.md``.
DESTINATIONS = ("analyst_qa", "investigator", "report")

#: Warning the report agent must carry while the corpus is synthetic.
SYNTHETIC_CORPUS_WARNING = (
    "WARNING: The regulation passages cited below are synthetic placeholders "
    "from a development corpus, not real regulatory text. This draft is not "
    "fileable."
)

_RING_ID = re.compile(r"\bring[-_][A-Za-z0-9_\-]+", re.IGNORECASE)
_ACCOUNT_ID = re.compile(r"\b[0-9a-f]{64}\b")

#: Returned when a question cannot be served from the allowed tables.
#: Must stay in step with ``agents.eval.questions.DECLINED``.
CANNOT_ANSWER = "cannot be answered from the allowed tables"

#: Vocabulary that marks a question as asking about the *labels* rather
#: than the detections. ``ground_truth`` is not in ``ALLOWED_SQL_TABLES``
#: -- deliberately, since an agent that can read the answer key is not
#: measuring anything -- so these questions must be declined, never
#: approximated from a detection table.
_GROUND_TRUTH_WORDS = (
    "ground truth",
    "ground-truth",
    "truly",
    "actually a mule",
    "according to the label",
    "labelled",
    "labeled",
    "cohort",
    "hard_negative",
    "role of",
    "pattern type",
    "pattern_type",
    "fast_passthrough",
    "shared_device pattern",
    "true ring",
    "true rings",
    "recall",
    "precision",
    "false positive",
)


def _asks_about_labels(lowered: str) -> bool:
    """True when a question can only be answered from ``ground_truth``."""
    return any(word in lowered for word in _GROUND_TRUTH_WORDS)


_REPORT_WORDS = ("report", "str ", "sar ", "file a", "draft", "narrative", "compliance")
_INVESTIGATE_WORDS = (
    "why", "walk me through", "explain", "trace", "where did", "how did",
    "investigate", "tell me about",
)


@dataclass(frozen=True)
class RouterDecision:
    """Which specialist should handle a request."""

    destination: str
    subject_id: str | None
    reason: str


@dataclass(frozen=True)
class QaAnswer:
    """A short factual answer with the tool call that produced it.

    ``claims`` is what makes the answer checkable. It carries the same
    :class:`~agents.verify.Claim` objects the investigator emits, and
    :func:`agents.verify.verify_answer` re-derives every one of them from
    the tools. Empty is a valid and common answer: a refusal, or a
    question whose answer is not of a kind the verifier knows, has
    nothing to check, and ``docs/agents_api.md`` defines that as
    ``verified: false`` rather than as a doubt about the answer.

    Nothing anywhere derives ``claims`` from ``answer``. Inferring a kind
    from the wording of a number would let a coincidence of phrasing
    produce ``verified: true``, which is the one thing that field must
    never mean.
    """

    answer: str
    basis: str
    unit: str = ""
    #: Checkable assertions, as structure. See above.
    claims: list[Claim] = field(default_factory=list)
    #: Every passage ``search_regulations`` actually returned this turn,
    #: for rendering as citation cards independent of whether ``answer``
    #: quotes them. Always empty in mock mode -- there is no real
    #: index to retrieve from offline.
    regulation_citations: list[dict[str, str]] = field(default_factory=list)

    @property
    def quoted(self) -> str:
        """The answer as a caller reads it: value and unit, one string.

        Defined here rather than at the two call sites because it is what
        ``verified`` is judged against -- if the service published
        ``"12 rings"`` and the verifier matched claims against ``"12"``,
        the two would disagree about what the answer said.
        """
        return f"{self.answer} {self.unit}".strip() if self.unit else self.answer


@dataclass(frozen=True)
class Ground:
    """One ground of suspicion, with the passage that supports it."""

    ground: str
    citation: str
    source_uri: str


@dataclass
class ReportDraft:
    """A drafted STR. ``grounds`` is empty when nothing could be cited."""

    narrative: str
    citations: list[str] = field(default_factory=list)
    grounds: list[Ground] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: False whenever any citation is a development placeholder. Mirrors
    #: ``StrReport.citations_verified``, which blocks filing while False,
    #: so the signal travels with the draft instead of being rediscovered
    #: by whoever turns it into a report.
    citations_verified: bool = False
    #: The narrative that was suppressed because it asserted a ground
    #: whose citation could not be verified. Kept for review, never for
    #: publication -- mirroring ``VerifiedInvestigation.raw
    #: ["withheld_narrative"]``. Empty when nothing was withheld.
    withheld_narrative: str = ""

    @property
    def narrative_withheld(self) -> bool:
        """True when prose was produced but suppressed as unsupported."""
        return not self.narrative and bool(self.withheld_narrative)


# ------------------------------------------------------------------ router


def extract_subject(text: str) -> str | None:
    """Pull a ring id or hashed account id out of a request, if present."""
    ring = _RING_ID.search(text)
    if ring:
        return ring.group(0)
    account = _ACCOUNT_ID.search(text)
    return account.group(0) if account else None


def route(request: str) -> RouterDecision:
    """Classify a request to exactly one specialist agent.

    A report request wins over everything, because it is the only
    destination a caller asks for explicitly.
    """
    lowered = request.lower()
    subject = extract_subject(request)

    if any(word in lowered for word in _REPORT_WORDS):
        return RouterDecision("report", subject, "request asks for a drafted report")
    if any(word in lowered for word in _INVESTIGATE_WORDS):
        return RouterDecision(
            "investigator", subject, "request asks why or how, needing a narrative"
        )
    return RouterDecision("analyst_qa", subject, "request has a single checkable answer")


# -------------------------------------------------------------- analyst qa

#: Question intent -> SQL -> the claim that backs the answer. Ordered:
#: the first whose every trigger word is present wins, so more specific
#: intents must come first.
#:
#: The fourth element is the ``(kind, subject)`` of the claim this intent
#: can offer, or ``None`` where no claim kind covers the question -- an
#: intra-ring transfer count has none, and the honest result there is an
#: answer with nothing checkable rather than a claim that only looks like
#: one. The **value** never appears here: it comes from the query, and a
#: value written beside the query would be a second source of truth for
#: the same number.
_QA_INTENTS: tuple[tuple[tuple[str, ...], str, str, tuple[str, str] | None], ...] = (
    (
        ("how many", "rings", "three banks"),
        "SELECT COUNT(*) AS n FROM rings WHERE banks_involved LIKE '%BANK_A%'"
        " AND banks_involved LIKE '%BANK_B%' AND banks_involved LIKE '%BANK_C%'",
        "rings",
        (ClaimKind.RING_BANK_SPAN_COUNT, "3"),
    ),
    (
        ("how many", "rings", "two banks"),
        "SELECT COUNT(*) AS n FROM rings WHERE (LENGTH(banks_involved)"
        " - LENGTH(REPLACE(banks_involved, 'BANK_', ''))) / 5 = 2",
        "rings",
        (ClaimKind.RING_BANK_SPAN_COUNT, "2"),
    ),
    (
        ("how many", "rings", "cross-bank"),
        "SELECT COUNT(*) AS n FROM rings WHERE (LENGTH(banks_involved)"
        " - LENGTH(REPLACE(banks_involved, 'BANK_', ''))) / 5 > 1",
        "rings",
        (ClaimKind.RING_BANK_SPAN_COUNT, ">1"),
    ),
    (
        ("how many", "rings"),
        "SELECT COUNT(*) AS n FROM rings",
        "rings",
        (ClaimKind.TABLE_ROW_COUNT, "rings"),
    ),
    # Evidence-scoped account questions must beat the generic account
    # count below, which would otherwise swallow anything saying
    # "how many accounts".
    (
        ("how many", "accounts", "evidence"),
        "SELECT COUNT(DISTINCT hashed_account_id) AS n FROM account_evidence",
        "accounts",
        (ClaimKind.DISTINCT_VALUE_COUNT, "account_evidence.hashed_account_id"),
    ),
    (
        ("how many", "accounts", "flagged"),
        "SELECT COUNT(DISTINCT hashed_account_id) AS n FROM account_evidence",
        "accounts",
        (ClaimKind.DISTINCT_VALUE_COUNT, "account_evidence.hashed_account_id"),
    ),
    (
        ("how many", "evidence", "cross_bank"),
        "SELECT COUNT(*) AS n FROM account_evidence WHERE scope = 'cross_bank'",
        "rows",
        (ClaimKind.FILTERED_ROW_COUNT, "account_evidence.scope=cross_bank"),
    ),
    (
        ("how many", "shared device"),
        "SELECT COUNT(*) AS n FROM account_evidence"
        " WHERE evidence_type = 'cross_bank_shared_device'",
        "rows",
        (ClaimKind.FILTERED_ROW_COUNT,
         "account_evidence.evidence_type=cross_bank_shared_device"),
    ),
    (
        ("how many", "accounts", "bank_a"),
        "SELECT COUNT(*) AS n FROM accounts WHERE bank_id = 'BANK_A'",
        "accounts",
        (ClaimKind.FILTERED_ROW_COUNT, "accounts.bank_id=BANK_A"),
    ),
    (
        ("how many", "accounts"),
        "SELECT COUNT(*) AS n FROM accounts",
        "accounts",
        (ClaimKind.TABLE_ROW_COUNT, "accounts"),
    ),
    (
        ("how many", "banks"),
        "SELECT COUNT(DISTINCT bank_id) AS n FROM accounts",
        "banks",
        (ClaimKind.DISTINCT_VALUE_COUNT, "accounts.bank_id"),
    ),
    (
        ("how many", "transfers"),
        "SELECT COUNT(*) AS n FROM transfers",
        "transfers",
        (ClaimKind.TABLE_ROW_COUNT, "transfers"),
    ),
    (
        ("highest", "score"),
        "SELECT hashed_account_id FROM scores ORDER BY score DESC LIMIT 1",
        "account",
        (ClaimKind.TOP_RISK_ACCOUNT, "scores"),
    ),
    (
        ("model", "version", "score"),
        "SELECT DISTINCT model_version FROM scores",
        "",
        (ClaimKind.COLUMN_DISTINCT_VALUES, "scores.model_version"),
    ),
)


def answer_question(question: str) -> QaAnswer:
    """Answer a short factual question with a tool call.

    Subject-specific questions (about one ring or one account) are served
    from the typed tools; aggregate questions go through ``run_sql``.

    A question about the labels is declined before anything else. The
    failure this guards against is specific and nasty: "how many
    accounts are truly mules?" keyword-matches the generic account count
    and would confidently answer 30,000. Substituting a number you *can*
    get for the one that was asked for is worse than refusing, because
    the answer looks authoritative.
    """
    lowered = question.lower()
    subject = extract_subject(question)

    if _asks_about_labels(lowered):
        return QaAnswer(CANNOT_ANSWER, "", "")

    if subject is not None:
        answer = _answer_about_subject(lowered, subject)
        if answer is not None:
            return answer

    for triggers, sql, unit, claim in _QA_INTENTS:
        if all(trigger in lowered for trigger in triggers):
            rows = tools.run_sql(sql)
            if not rows:
                return QaAnswer("no rows", sql, unit)
            value = next(iter(rows[0].values()))
            return QaAnswer(
                str(value), sql, unit, claims=_aggregate_claims(claim, rows)
            )

    return QaAnswer(CANNOT_ANSWER, "", "")


def _aggregate_claims(
    template: tuple[str, str] | None,
    rows: list[dict[str, Any]],
) -> list[Claim]:
    """Turn an intent's claim template plus its rows into claims.

    The value is read out of the rows the query actually returned, never
    written beside the template: the claim and the answer have to be the
    same number, and two literals in two places is how they stop being.

    ``column_distinct_values`` is the one intent whose claim spans every
    row rather than the first, because that is what the kind means. Its
    answer still quotes one value, so the claim is *stated* by the answer
    only where the column really has one value -- which is the honest
    outcome: an answer naming one of several versions has not answered
    the question its claim checks.
    """
    if template is None:
        return []
    kind, subject = template
    first = next(iter(rows[0].values()))
    if kind == ClaimKind.COLUMN_DISTINCT_VALUES:
        value: Any = sorted(str(next(iter(row.values()))) for row in rows)
    elif kind == ClaimKind.TOP_RISK_ACCOUNT:
        value = str(first)
    else:
        value = int(first)
    return [
        Claim(
            id="c1",
            kind=str(kind),
            subject=subject,
            value=value,
            text=f"{kind} for {subject} is {value}",
        )
    ]


def _answer_about_subject(lowered: str, subject: str) -> QaAnswer | None:
    """Answer a question scoped to one named ring or account."""
    is_ring = subject.lower().startswith("ring")

    if is_ring:
        try:
            ring, accounts, transfers = tools.get_ring_subgraph(subject, max_hops=1)
        except KeyError:
            return QaAnswer("no rows", f"get_ring_subgraph({subject})", "")
        if "how many" in lowered and "bank" in lowered:
            return QaAnswer(
                str(len(ring.banks_involved)),
                f"get_ring_subgraph({subject})",
                "banks",
                claims=[
                    _claim(
                        ClaimKind.RING_BANK_COUNT, subject, len(ring.banks_involved)
                    )
                ],
            )
        if "which bank" in lowered or "what bank" in lowered:
            return QaAnswer(
                ", ".join(sorted(ring.banks_involved)),
                f"get_ring_subgraph({subject})",
                "",
                claims=[
                    _claim(
                        ClaimKind.RING_BANKS, subject, sorted(ring.banks_involved)
                    )
                ],
            )
        if "how many" in lowered and ("account" in lowered or "member" in lowered):
            # An unqualified count answers with the alertable population.
            # Someone asking "how many accounts are in this ring" will
            # treat the answer as a number of alerts, and same-bank
            # context members are explicitly not alerts.
            wants_all = any(
                word in lowered
                for word in ("including", "context", "all members", "in total", "total")
            )
            population = ring.all_member_ids if wants_all else ring.alertable_member_ids
            kind = (
                ClaimKind.RING_MEMBER_COUNT
                if wants_all
                else ClaimKind.RING_ALERTABLE_MEMBER_COUNT
            )
            return QaAnswer(
                str(len(population)),
                f"get_ring_subgraph({subject})",
                "accounts" if wants_all else "alertable accounts",
                claims=[_claim(kind, subject, len(population))],
            )
        if "transfer" in lowered:
            return QaAnswer(
                str(len(transfers)), f"get_ring_subgraph({subject})", "transfers"
            )
        return None

    try:
        account, score, rings = tools.get_account_profile(subject)
    except KeyError:
        return QaAnswer("no rows", f"get_account_profile({subject})", "")
    if "score" in lowered:
        if score is None:
            return QaAnswer("not scored", f"get_account_profile({subject})", "")
        rounded = f"{score.score:.{SCORE_DECIMALS}f}"
        return QaAnswer(
            rounded,
            f"get_account_profile({subject})",
            "",
            claims=[_claim(ClaimKind.ACCOUNT_SCORE, subject, float(rounded))],
        )
    if "bank" in lowered:
        return QaAnswer(
            account.bank_id,
            f"get_account_profile({subject})",
            "",
            claims=[_claim(ClaimKind.ACCOUNT_BANK, subject, account.bank_id)],
        )
    if "ring" in lowered:
        # One claim per ring, not one naming the first: the answer lists
        # them all, and a claim set that covers part of an answer leaves
        # the rest of it unchecked.
        return QaAnswer(
            ", ".join(rings) if rings else "none",
            f"get_account_profile({subject})",
            "",
            claims=[
                _claim(ClaimKind.ACCOUNT_IN_RING, subject, ring_id, index=index)
                for index, ring_id in enumerate(rings, start=1)
            ],
        )
    return None


def _claim(kind: str, subject: str, value: Any, index: int = 1) -> Claim:
    """One claim about a named ring or account."""
    return Claim(
        id=f"c{index}",
        kind=str(kind),
        subject=subject,
        value=value,
        text=f"{kind} for {subject} is {value}",
    )


# ------------------------------------------------------------ investigator


def investigate(
    subject_id: str,
    feedback: str = "",
) -> InvestigationResult:
    """Explain why a subject was flagged, emitting checkable claims.

    Args:
        subject_id: A ring id or a hashed account id.
        feedback: Unsupported claims from a failed verification. The
            deterministic path recomputes every value from the tools on
            each call, so a retry is already corrected; the text is
            recorded on ``raw`` so the retry is visible in a trace.

    Returns:
        A narrative, the claims backing it, and the supporting evidence.
    """
    if subject_id.lower().startswith("ring"):
        result = _investigate_ring(subject_id)
    else:
        result = _investigate_account(subject_id)
    if feedback:
        result.raw["retry_feedback"] = feedback
    return result


def _investigate_ring(ring_id: str) -> InvestigationResult:
    ring, accounts, transfers = tools.get_ring_subgraph(ring_id, max_hops=1)
    explanation = tools.get_explanation(ring_id, "ring")

    banks = sorted(ring.banks_involved)
    members = ring.all_member_ids
    alertable = ring.alertable_member_ids
    context = ring.context_member_ids

    claims = [
        Claim(
            "c1",
            "ring_alertable_member_count",
            ring_id,
            len(alertable),
            f"The ring has {len(alertable)} alertable member accounts.",
        ),
        Claim(
            "c2",
            "ring_member_count",
            ring_id,
            len(members),
            f"The ring has {len(members)} members in total, including context.",
        ),
        Claim(
            "c3",
            "ring_bank_count",
            ring_id,
            len(banks),
            f"The ring spans {len(banks)} bank(s).",
        ),
        Claim("c4", "ring_banks", ring_id, banks, f"The banks are {', '.join(banks)}."),
    ]

    evidence = _evidence_lines(members)
    context_clause = (
        f", plus {len(context)} same-bank member(s) linked for review and not "
        "counted as alerts"
        if context
        else ""
    )
    narrative = (
        f"Ring {ring_id} comprises {len(alertable)} alertable accounts spanning "
        f"{len(banks)} bank(s): {', '.join(banks)}{context_clause}. It was flagged "
        "on transfer velocity and graph structure -- "
        f"{'; '.join(explanation.top_factors[:3])} -- observed across "
        f"{len(transfers)} transfers in the local subgraph. Identity signals "
        "played no part in the detection and are listed separately as "
        "supporting evidence."
    )
    membership_lines = _describe_memberships(ring)
    return InvestigationResult(
        narrative=narrative,
        claims=claims,
        evidence=membership_lines + evidence,
        raw={
            "ring_id": ring_id,
            "members": members,
            "alertable_members": alertable,
            "context_members": context,
            "membership_basis": membership_lines,
            "banks": banks,
            "subgraph_accounts": len(accounts),
            "subgraph_transfers": len(transfers),
            "top_factors": list(explanation.top_factors),
            "explanation_narrative": explanation.narrative,
        },
    )


def _investigate_account(hashed_account_id: str) -> InvestigationResult:
    account, score, rings = tools.get_account_profile(hashed_account_id)
    reuse = tools.get_document_reuse(hashed_account_id)
    cross = [row for row in reuse if row.scope is EvidenceScope.CROSS_BANK]

    claims = [
        Claim(
            "c1",
            "account_bank",
            hashed_account_id,
            account.bank_id,
            f"The account is held at {account.bank_id}.",
        ),
        Claim(
            "c2",
            "cross_bank_evidence_count",
            hashed_account_id,
            len(cross),
            f"There are {len(cross)} cross-bank identity evidence rows.",
        ),
    ]
    if score is not None:
        claims.append(
            Claim(
                "c3",
                "account_score",
                hashed_account_id,
                round(score.score, SCORE_DECIMALS),
                f"Its risk score is {score.score:.{SCORE_DECIMALS}f}.",
            )
        )
    if rings:
        claims.append(
            Claim(
                "c4",
                "account_in_ring",
                hashed_account_id,
                rings[0],
                f"It belongs to ring {rings[0]}.",
            )
        )

    factors = list(score.top_factors) if score is not None else []
    driver = "; ".join(factors[:3]) if factors else "no factor above threshold"
    ring_clause = (
        f" It clusters into ring {rings[0]}." if rings else " It is not in a ring."
    )
    narrative = (
        f"Account {hashed_account_id[:16]}... at {account.bank_id} carries a risk "
        f"score of {score.score:.{SCORE_DECIMALS}f} driven by velocity and structure: {driver}."
        f"{ring_clause} Identity reuse, where present, is recorded below as "
        "supporting evidence and did not contribute to the score."
    )
    return InvestigationResult(
        narrative=narrative,
        claims=claims,
        evidence=_format_evidence(reuse),
        raw={
            "hashed_account_id": hashed_account_id,
            "bank_id": account.bank_id,
            "score": score.score if score is not None else None,
            "top_factors": factors,
            "rings": rings,
            "evidence_rows": [row.model_dump(mode="json") for row in reuse],
        },
    )


def _describe_memberships(ring: Any) -> list[str]:
    """Describe each member by how it came to be in the ring.

    The branch on ``membership_basis`` is exhaustive and ends in a
    raise, not a fallthrough. ``MembershipBasis`` is closed precisely so
    that a future third value stops here and forces someone to decide
    how it should be described -- rather than quietly inheriting the
    velocity wording and publishing a detection that was never made.
    """
    lines: list[str] = []
    for member in ring.members:
        short = member.hashed_account_id[:16]
        basis = member.membership_basis

        if basis is MembershipBasis.VELOCITY:
            lines.append(
                f"[detected] {short}...: flagged by the velocity rule on its own "
                "transaction behaviour."
            )
        elif basis is MembershipBasis.IDENTITY_LINK:
            # The reason is used verbatim: it is written as a lead, it
            # has been reviewed, and paraphrasing is how a lead turns
            # into an accusation.
            scope = member.link_scope.value if member.link_scope else "unknown scope"
            tag = "linked, needs review" if member.alertable else "context only"
            lines.append(f"[{tag}] {short}... ({scope}): {member.reason}")
        else:  # pragma: no cover - unreachable while the enum has two members
            raise NotImplementedError(
                f"unhandled membership basis {basis!r} for {short}...: every basis "
                "must have agreed wording before it can be shown to an analyst"
            )
    return lines


def _evidence_lines(members: list[str]) -> list[str]:
    """Collect evidence lines across a ring's members, de-duplicated."""
    lines: list[str] = []
    for member in members:
        try:
            lines.extend(_format_evidence(tools.get_document_reuse(member)))
        except KeyError:
            continue
    return list(dict.fromkeys(lines))


def _format_evidence(rows: list[Any]) -> list[str]:
    """Render evidence rows, always naming the scope.

    Same-bank rows are kept but explicitly marked weak: dropping them
    would hide from the analyst that the match is the routine kind.
    """
    lines = []
    for row in rows:
        banks = ", ".join(row.counterparty_bank_ids) or "unknown bank"
        if row.scope is EvidenceScope.CROSS_BANK:
            lines.append(
                f"[cross_bank] {row.evidence_type}: shared with "
                f"{len(row.counterparty_hashed_account_ids)} account(s) at {banks}; "
                "corroborating only, since dual-bank customers reuse identifiers "
                "legitimately."
            )
        else:
            lines.append(
                f"[single_bank] {row.evidence_type}: shared with "
                f"{len(row.counterparty_hashed_account_ids)} account(s) at {banks}; "
                "weak signal, routine among households and joint businesses, "
                "not incriminating."
            )
    return lines


# ----------------------------------------------------------------- report


#: One query per candidate ground. The ground is only asserted if the
#: search returns a passage, and only if the underlying fact holds.
_GROUND_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "rapid pass-through: funds received and forwarded within minutes",
        "rapid pass-through of funds through an intermediary account without "
        "commercial rationale",
    ),
    (
        "high fan-in: deposits from many unrelated senders",
        "funnel account receiving deposits from multiple unrelated parties then "
        "transferred onward",
    ),
    (
        "cross-bank layering: funds move onward to another institution",
        "funds moving across institutions so no single bank sees the whole pattern",
    ),
)


def draft_report(subject_id: str) -> ReportDraft:
    """Draft an STR for a ring or account.

    Every ground must carry a regulation passage from
    ``search_regulations``; a ground with no passage is dropped. If no
    ground can be cited, the draft says so and lists no grounds -- an
    empty report is a correct outcome, not a failure.
    """
    investigation = investigate(subject_id)
    warnings: list[str] = []

    grounds: list[Ground] = []
    for ground_text, query in _GROUND_QUERIES:
        passages = tools.search_regulations(query, top_k=1)
        if not passages:
            warnings.append(f"ground dropped, no citation retrieved: {ground_text}")
            continue
        passage = passages[0]
        grounds.append(
            Ground(
                ground=ground_text,
                citation=passage["citation"],
                source_uri=passage["source_uri"],
            )
        )

    synthetic = any(
        regulations.is_synthetic({"citation": g.citation, "source_uri": g.source_uri})
        for g in grounds
    )
    if synthetic:
        warnings.append(
            "citations are synthetic placeholders from a development corpus"
        )

    narrative = _render_report(subject_id, investigation, grounds)
    return ReportDraft(
        narrative=narrative,
        citations=[g.citation for g in grounds],
        grounds=grounds,
        warnings=warnings,
        # Only a ground list that is non-empty AND free of placeholders
        # could ever be filed.
        citations_verified=bool(grounds) and not synthetic,
    )


def _render_report(
    subject_id: str,
    investigation: InvestigationResult,
    grounds: list[Ground],
) -> str:
    parts = [SYNTHETIC_CORPUS_WARNING, "", "## Subject", investigation.narrative, ""]

    parts.append("## Grounds of suspicion")
    if not grounds:
        parts.append(
            "No citable ground was established: no regulation passage could be "
            "retrieved for any candidate ground, so none is asserted."
        )
    else:
        for index, ground in enumerate(grounds, start=1):
            parts.append(
                f"{index}. {ground.ground}\n"
                f"   Citation: {ground.citation} ({ground.source_uri})"
            )
    parts.append("")

    parts.append("## Supporting evidence")
    if investigation.evidence:
        # Identity reuse is cited once, however many keys fired: the
        # reused id number and reused photo fire on identical accounts in
        # this dataset, so listing both overstates the corroboration.
        parts.extend(f"- {line}" for line in _dedupe_identity(investigation.evidence))
        parts.append(
            "\nIdentity reuse is corroborating evidence only. It took no part in "
            "the detection, and on its own it is a weak signal: far more "
            "legitimate customers reuse a document than mules do."
        )
    else:
        parts.append("No corroborating identity evidence was found.")
    parts.append("")

    parts.append("## Limitations")
    parts.append(
        "This draft rests on the transaction and KYC data held by the "
        "participating banks. It does not establish intent, and it does not "
        "show activity at institutions outside the cross-bank view. The "
        "citations above are synthetic placeholders and must be replaced with "
        "real regulatory text before any filing."
    )
    return "\n".join(parts)


def _dedupe_identity(lines: list[str]) -> list[str]:
    """Collapse id-number and photo reuse into one line per scope.

    Those two keys fire on exactly the same accounts in this dataset, an
    artifact of the generator. Rendering them separately would present
    one fact as two independent corroborations.
    """
    kept: list[str] = []
    seen_identity: set[str] = set()
    for line in lines:
        if "reused_id_number" in line or "reused_photo" in line:
            scope = "cross_bank" if "[cross_bank]" in line else "single_bank"
            if scope in seen_identity:
                continue
            seen_identity.add(scope)
            kept.append(
                line.replace("reused_id_number", "reused identity document")
                .replace("reused_photo", "reused identity document")
            )
            continue
        kept.append(line)
    return kept
