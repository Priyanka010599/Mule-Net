"""Claim types and the verifier that checks them against the tools.

The verifier re-derives every claim from the tools and compares. It does
*not* ask a model whether a claim looks right: an LLM checking its own
arithmetic is strictly weaker than recomputing the number, and this is
the gate that decides whether a narrative reaches an analyst.

**This module is the verifier. There is no verifier agent.** One was
built as an ``LlmAgent`` with a prompt and a tool subset, and was never
invoked -- ``_run_agent`` was only ever called for the router, the
analyst_qa, the investigator and the report agents. It has been removed
rather than left in place to make the system look like it has five
agents when four do the work. What an LLM judge could add over this
module is real but narrow -- catching a narrative that is *misleading*
without tripping any keyword rule -- and unimplemented; it is recorded
in ``docs/post_submission.md`` rather than claimed here.

Two kinds of check run:

*Claim checks* re-derive each structured claim and compare values.
A claim whose ``kind`` is not in :class:`ClaimKind` is unsupported by
definition -- it cannot be checked, so it cannot pass.

*Prose checks* scan the narrative for the two framings the project
forbids: presenting identity reuse as the reason something was flagged,
and presenting a ``single_bank`` reuse row as incriminating. Detection
fires on velocity and structure alone; the identity branch was measured
and found strictly dominated (zero extra true positives, 146 extra false
positives), so a narrative that credits it is wrong on the facts, not
merely impolite.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agents import hidden_labels, tools
from agents.data import default_store
from shared.contracts import EvidenceScope
from shared.tool_interfaces import ALLOWED_SQL_TABLES, SCORE_TOLERANCE


class ClaimKind(StrEnum):
    """The claim kinds the verifier knows how to re-derive.

    Anything outside this set is unverifiable and therefore rejected.
    Keep in step with the tables in ``agents/prompts/investigator.md``
    (the subject-scoped kinds) and ``agents/prompts/analyst_qa.md``
    (all of them). ``test_service`` fails if either prompt drifts.

    Two groups, differing in what ``subject`` means:

    *Subject-scoped* kinds are about one ring or one account, and
    ``subject`` is its id. These are the kinds the investigator emits.

    *Population* kinds are about a whole table, which is what a chat
    question usually asks about ("how many rings are there"). There
    ``subject`` names the population rather than an entity -- a table, a
    ``table.column``, or a ``table.column=value`` filter. The model names
    *what* to count and the verifier decides *how*: it resolves the table
    itself, against :data:`~shared.tool_interfaces.ALLOWED_SQL_TABLES`
    and the frame's real columns, and counts with pandas. **No
    model-supplied SQL is ever executed as part of a check** -- that
    would be re-running the model's own query and calling it
    verification.
    """

    RING_MEMBER_COUNT = "ring_member_count"
    RING_ALERTABLE_MEMBER_COUNT = "ring_alertable_member_count"
    RING_BANK_COUNT = "ring_bank_count"
    RING_BANKS = "ring_banks"
    ACCOUNT_SCORE = "account_score"
    ACCOUNT_IN_RING = "account_in_ring"
    ACCOUNT_BANK = "account_bank"
    CROSS_BANK_EVIDENCE_COUNT = "cross_bank_evidence_count"
    TABLE_ROW_COUNT = "table_row_count"
    FILTERED_ROW_COUNT = "filtered_row_count"
    DISTINCT_VALUE_COUNT = "distinct_value_count"
    COLUMN_DISTINCT_VALUES = "column_distinct_values"
    RING_BANK_SPAN_COUNT = "ring_bank_span_count"
    TOP_RISK_ACCOUNT = "top_risk_account"



#: Phrasings that credit identity reuse with causing a detection. Matched
#: case-insensitively against the narrative.
_REUSE_AS_CAUSE = (
    "flagged because",
    "flagged for sharing",
    "flagged due to",
    "detected because",
    "was flagged as a result of",
)

#: Words that, near a reuse mention, turn evidence into an accusation.
_GUILT_WORDS = (
    "proves",
    "confirms guilt",
    "is guilty",
    "establishes guilt",
    "demonstrates that the account is a mule",
)

#: Negators that flip the meaning of a marker word just after them. A
#: line reading "not incriminating" is the correct rendering of a
#: same-bank row, so a bare substring match on "incriminat" would reject
#: exactly the wording the project requires.
_NEGATORS = ("not ", "never ", "n't ", "no ", "rather than ", "cannot ", "without ")

#: How far back to look for a negator before a marker word.
_NEGATION_WINDOW = 24


def _asserted(text: str, marker: str) -> bool:
    """True when ``marker`` appears in ``text`` and is not negated.

    Checks the characters immediately before each occurrence for a
    negator. Crude, but it is the difference between rejecting "proves
    guilt" and rejecting "proves nothing on its own".
    """
    index = text.find(marker)
    while index != -1:
        window = text[max(0, index - _NEGATION_WINDOW) : index]
        if not any(negator in window for negator in _NEGATORS):
            return True
        index = text.find(marker, index + 1)
    return False


#: Identity vocabulary. Bare nouns rather than fixed phrases: "shares a
#: device" and "shared device" are the same error, and a phrase list
#: misses whichever wording it was not written for. Safe to keep broad
#: because a hit only counts inside the window after a causal phrase.
_REUSE_WORDS = (
    "device",
    "document",
    "photo",
    "id number",
    "identifier",
    "identity",
    "address",
    "kyc",
)


@dataclass(frozen=True)
class Claim:
    """One checkable assertion made by the investigator.

    Attributes:
        id: Short handle the verifier quotes back on failure.
        kind: One of :class:`ClaimKind`, as a raw string so an unknown
            kind survives to be reported rather than blowing up parsing.
        subject: The ring id or hashed account id the claim is about.
        value: The claimed value. Type depends on ``kind``.
        text: The sentence this claim backs, for the feedback message.
    """

    id: str
    kind: str
    subject: str
    value: Any
    text: str = ""


@dataclass
class InvestigationResult:
    """What the investigator produced for one subject."""

    narrative: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """The verdict on one investigation."""

    passed: bool
    unsupported: list[str] = field(default_factory=list)

    @property
    def feedback(self) -> str:
        """The unsupported claims, formatted for a retry prompt."""
        if self.passed:
            return ""
        lines = "\n".join(f"- {item}" for item in self.unsupported)
        return (
            "Your previous answer contained claims the tools do not support:\n"
            f"{lines}\n"
            "Correct or drop each one. Do not restate a claim you cannot "
            "verify with a tool call."
        )


def _check_ring_member_count(subject: str, value: Any) -> str | None:
    """Every member, whatever basis brought them in."""
    ring, _, _ = tools.get_ring_subgraph(subject, max_hops=1)
    actual = len(ring.all_member_ids)
    if int(value) != actual:
        return f"claimed {value} members, tools return {actual}"
    return None


def _check_ring_alertable_member_count(subject: str, value: Any) -> str | None:
    """Members that may be published as an alert.

    Velocity members plus cross-bank identity links. Same-bank links are
    context and are never counted here.
    """
    ring, _, _ = tools.get_ring_subgraph(subject, max_hops=1)
    actual = len(ring.alertable_member_ids)
    if int(value) != actual:
        return f"claimed {value} alertable members, tools return {actual}"
    return None


def _check_ring_bank_count(subject: str, value: Any) -> str | None:
    ring, _, _ = tools.get_ring_subgraph(subject, max_hops=1)
    actual = len(ring.banks_involved)
    if int(value) != actual:
        return f"claimed {value} banks, tools return {actual}"
    return None


def _check_ring_banks(subject: str, value: Any) -> str | None:
    ring, _, _ = tools.get_ring_subgraph(subject, max_hops=1)
    actual = sorted(ring.banks_involved)
    if sorted(str(v) for v in value) != actual:
        return f"claimed banks {sorted(value)}, tools return {actual}"
    return None


def _check_account_score(subject: str, value: Any) -> str | None:
    _, score, _ = tools.get_account_profile(subject)
    if score is None:
        return "claimed a score, but the account has never been scored"
    if abs(float(value) - score.score) > SCORE_TOLERANCE:
        return f"claimed score {value}, tools return {score.score:.3f}"
    return None


def _check_account_in_ring(subject: str, value: Any) -> str | None:
    _, _, rings = tools.get_account_profile(subject)
    if str(value) not in rings:
        return f"claimed membership of {value}, tools return {rings or 'no rings'}"
    return None


def _check_account_bank(subject: str, value: Any) -> str | None:
    account, _, _ = tools.get_account_profile(subject)
    if str(value) != account.bank_id:
        return f"claimed bank {value}, tools return {account.bank_id}"
    return None


def _check_cross_bank_evidence_count(subject: str, value: Any) -> str | None:
    rows = tools.get_document_reuse(subject)
    actual = sum(1 for row in rows if row.scope is EvidenceScope.CROSS_BANK)
    if int(value) != actual:
        return f"claimed {value} cross-bank evidence rows, tools return {actual}"
    return None


# ----------------------------------------------------------- populations
#
# A chat question is usually about a population rather than an entity:
# "how many rings has the detector found", "how many evidence rows have
# scope cross_bank". The checks below re-derive those counts from the
# frames the tools read, so a chat answer can carry a claim at all. They
# all go through :func:`_frame`, which is the gate: a table outside
# ``ALLOWED_SQL_TABLES`` and a column the frame does not have are both
# failures, reported as such rather than raised.


class _Unverifiable(Exception):
    """A claim that cannot be re-derived, carrying the reason.

    Raised inside the population checks and turned into a failure message
    by :func:`check_claim`. An exception rather than a returned string
    only because the parsing helpers are several calls deep.
    """


def _frame(table: str) -> Any:
    """The frame for ``table``, or raise :class:`_Unverifiable`.

    The allowlist is the same one ``run_sql`` enforces, so a claim cannot
    reach a table an agent was never allowed to read -- including
    ``ground_truth``, which is the whole point of that list.
    """
    name = table.strip()
    if name not in ALLOWED_SQL_TABLES:
        raise _Unverifiable(
            f"table {name!r} is not one this system publishes; "
            f"verifiable tables are {sorted(ALLOWED_SQL_TABLES)}"
        )
    try:
        return default_store().table(name)
    except (FileNotFoundError, KeyError, OSError) as exc:
        raise _Unverifiable(f"table {name!r} could not be read: {exc}") from None


def _column(table: str, column: str) -> Any:
    """One column of ``table``, by its real name in the frame."""
    frame = _frame(table)
    name = column.strip()
    if hidden_labels.is_denied_column(name):
        raise _Unverifiable(f"{table.strip()}.{name} is a hidden simulator label")
    if name not in frame.columns:
        raise _Unverifiable(
            f"{table.strip()} has no column {name!r}; its columns are "
            f"{sorted(str(c) for c in frame.columns)}"
        )
    return frame[name]


def _split_column(subject: str) -> tuple[str, str]:
    """Parse ``table.column``. Lenient about spacing, strict about shape."""
    table, _, column = subject.partition(".")
    if not table.strip() or not column.strip() or "." in column:
        raise _Unverifiable(
            f"subject {subject!r} is not a 'table.column' reference"
        )
    return table.strip(), column.strip()


def _split_filter(subject: str) -> tuple[str, str, str]:
    """Parse ``table.column=value``.

    Surrounding quotes on the value are stripped, because the model
    writes SQL all day and reaches for them here too. Tolerating that is
    not a weakened check: the comparison is still the verifier's, against
    the frame, and a value that matches nothing is a failure either way.
    """
    reference, sep, value = subject.partition("=")
    if not sep:
        raise _Unverifiable(
            f"subject {subject!r} is not a 'table.column=value' filter"
        )
    table, column = _split_column(reference)
    return table, column, value.strip().strip("'\"")


def _matching_rows(table: str, column: str, value: str) -> int:
    """Rows of ``table`` whose ``column`` equals ``value``.

    Compared as text first and numerically only if that matched nothing,
    which is what makes ``3`` match an integer column and ``BANK_A`` a
    string one without the claim having to say which it is. Booleans come
    out of the text branch: ``astype("string")`` renders them ``True`` /
    ``False`` and the comparison is case-folded.
    """
    series = _column(table, column)
    text = value.casefold()
    mask = series.astype("string").str.strip().str.casefold() == text
    count = int(mask.sum())
    if count:
        return count
    try:
        return int((series == float(value)).sum())
    except (TypeError, ValueError):
        return 0


def _check_table_row_count(subject: str, value: Any) -> str | None:
    actual = len(_frame(subject))
    if int(value) != actual:
        return f"claimed {value} rows in {subject.strip()}, tools return {actual}"
    return None


def _check_filtered_row_count(subject: str, value: Any) -> str | None:
    table, column, wanted = _split_filter(subject)
    actual = _matching_rows(table, column, wanted)
    if int(value) != actual:
        return (
            f"claimed {value} rows of {table} with {column} = {wanted!r}, "
            f"tools return {actual}"
        )
    return None


def _check_distinct_value_count(subject: str, value: Any) -> str | None:
    table, column = _split_column(subject)
    actual = int(_column(table, column).nunique())
    if int(value) != actual:
        return (
            f"claimed {value} distinct {column} values in {table}, "
            f"tools return {actual}"
        )
    return None


def _check_column_distinct_values(subject: str, value: Any) -> str | None:
    """Every distinct value of one column, as a set.

    Accepts a bare string for the common case of a column with exactly
    one value -- "which model version produced the scores" -- because
    requiring a one-element list there is a rule the model would break
    and a reader would not miss.
    """
    table, column = _split_column(subject)
    actual = sorted({str(item) for item in _column(table, column).dropna().unique()})
    claimed = sorted(
        str(item) for item in (value if isinstance(value, list | tuple) else [value])
    )
    if claimed != actual:
        return f"claimed {column} values {claimed}, tools return {actual}"
    return None


#: ``3``, ``>1``, and the wordier forms a model reaches for -- ``> 1``,
#: ``3 banks``. Leniency about the spelling of a subject is not leniency
#: about the check: the count is still the verifier's, and a spelling it
#: rejects withholds an answer that may well be right.
_BANK_SPAN = re.compile(r"^(>?)\s*(\d+)\s*(?:banks?)?$", re.IGNORECASE)

#: Accepted subjects for ``top_risk_account``. The derivation is fixed --
#: the maximum of ``scores.score`` -- so naming the column instead of the
#: table is a spelling difference and not a different claim.
_TOP_RISK_SUBJECTS = frozenset({"scores", "scores.score"})


def _check_ring_bank_span_count(subject: str, value: Any) -> str | None:
    """Detected rings whose bank span is exactly ``N``, or more than ``N``.

    ``subject`` is ``"3"`` or ``">1"``. Two shapes rather than a
    comparison language: those are the questions this system is asked,
    and a grammar the verifier has to interpret is a grammar the model
    can get subtly wrong in a way that reads as a data disagreement.
    """
    match = _BANK_SPAN.match(subject.strip())
    if match is None:
        raise _Unverifiable(
            f"subject {subject!r} is not a bank span: expected '3' or '>1'"
        )
    more_than, span = match.group(1) == ">", int(match.group(2))
    spans = _column("rings", "banks_involved").map(len)
    if more_than:
        actual = int((spans > span).sum())
        if int(value) != actual:
            return (
                f"claimed {value} rings spanning more than {span} banks, "
                f"tools return {actual}"
            )
        return None
    actual = int((spans == span).sum())
    if int(value) != actual:
        return (
            f"claimed {value} rings spanning exactly {span} banks, "
            f"tools return {actual}"
        )
    return None


def _check_top_risk_account(subject: str, value: Any) -> str | None:
    """The highest-scoring account. Ties all count as correct.

    Naming one of two accounts that share the maximum is not a wrong
    answer, and re-deriving with ``idxmax`` alone would have made it one.
    """
    if subject.strip() not in _TOP_RISK_SUBJECTS:
        raise _Unverifiable(
            f"subject {subject!r} is not 'scores'; the highest risk score is a "
            "fact about that table"
        )
    frame = _frame("scores")
    if frame.empty:
        return "claimed a highest-scoring account, but no account has been scored"
    top = float(frame["score"].max())
    holders = {
        str(row)
        for row, score in zip(
            frame["hashed_account_id"], frame["score"], strict=True
        )
        if abs(float(score) - top) <= SCORE_TOLERANCE
    }
    if str(value) not in holders:
        return (
            f"claimed {value} holds the highest score, tools return "
            f"{sorted(holders)} at {top:.3f}"
        )
    return None


#: One re-derivation per claim kind. A kind absent from this map cannot
#: be checked and is rejected outright.
_CHECKS: dict[str, Callable[[str, Any], str | None]] = {
    ClaimKind.RING_MEMBER_COUNT.value: _check_ring_member_count,
    ClaimKind.RING_ALERTABLE_MEMBER_COUNT.value: _check_ring_alertable_member_count,
    ClaimKind.RING_BANK_COUNT.value: _check_ring_bank_count,
    ClaimKind.RING_BANKS.value: _check_ring_banks,
    ClaimKind.ACCOUNT_SCORE.value: _check_account_score,
    ClaimKind.ACCOUNT_IN_RING.value: _check_account_in_ring,
    ClaimKind.ACCOUNT_BANK.value: _check_account_bank,
    ClaimKind.CROSS_BANK_EVIDENCE_COUNT.value: _check_cross_bank_evidence_count,
    ClaimKind.TABLE_ROW_COUNT.value: _check_table_row_count,
    ClaimKind.FILTERED_ROW_COUNT.value: _check_filtered_row_count,
    ClaimKind.DISTINCT_VALUE_COUNT.value: _check_distinct_value_count,
    ClaimKind.COLUMN_DISTINCT_VALUES.value: _check_column_distinct_values,
    ClaimKind.RING_BANK_SPAN_COUNT.value: _check_ring_bank_span_count,
    ClaimKind.TOP_RISK_ACCOUNT.value: _check_top_risk_account,
}


def check_claim(claim: Claim) -> str | None:
    """Re-derive one claim. Returns a failure message, or ``None`` if it holds."""
    checker = _CHECKS.get(claim.kind)
    if checker is None:
        return (
            f"claim kind {claim.kind!r} cannot be verified; "
            f"verifiable kinds are {sorted(_CHECKS)}"
        )
    try:
        return checker(claim.subject, claim.value)
    except _Unverifiable as exc:
        return str(exc)
    except KeyError:
        return f"subject {claim.subject!r} is unknown to the tools"
    except (TypeError, ValueError) as exc:
        return f"claimed value {claim.value!r} is not comparable: {exc}"


#: Words that turn a sentence into an accusation. An identity-link
#: member may carry none of them.
#:
#: Matched on word boundaries, not as substrings. The rule is about the
#: *word* "mule", and every ring id in this system is spelled
#: ``ring-mulenet-velocity-rule-v2-NNNNN`` -- a substring match would
#: reject every narrative that names the ring it is about, which is all
#: of them.
_ACCUSATION_RE = re.compile(
    r"\b(mule|mules|launderer|launderers|laundering|suspect|suspects|"
    r"criminal|criminals|perpetrator|perpetrators|offender|offenders|"
    r"fraudster|fraudsters)\b"
)

#: Vocabulary marking a sentence as being about an identity-link member.
_LINK_SENTENCE_WORDS = (
    "linked",
    "identity link",
    "identity_link",
    "needs review",
    "shares a device",
    "shares a hashed",
    "propagation",
    "lead",
)

#: Qualifiers that make an all-members count honest by naming what the
#: extra members are.
_CONTEXT_QUALIFIERS = (
    "context",
    "linked",
    "needs review",
    "including",
    "of which",
    "not alertable",
    "same-bank",
    "same bank",
)

#: Nouns a member count attaches to, used to find counts in prose.
_COUNT_NOUNS = ("member", "members", "account", "accounts")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: A count is a free-standing number, so the digits must not be glued to a
#: preceding word or hyphen. Every ring id ends in digits
#: (``ring-mulenet-velocity-rule-v2-00002``), and without the lookbehind
#: "ring-...-00002 has linked members" parses as a count of 2.
_COUNT_NEAR_NOUN = re.compile(
    r"(?<![-\w.])(\d+)\s+(?:\w+\s+){0,2}?(" + "|".join(_COUNT_NOUNS) + r")\b"
)


def check_identity_link_wording(narrative: str) -> list[str]:
    """Reject accusation vocabulary in a sentence about an identity link.

    The bar is deliberately absolute: the word "mule" may not appear in
    an identity-link sentence *even inside a negation*. "...is not
    evidence that this account is a mule" reads correctly in full and
    reads as an accusation the instant a UI truncates it, and the
    published ``reason`` strings clear that bar with zero occurrences.
    A narrative should too.
    """
    problems: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(narrative):
        lowered = sentence.lower()
        if not any(word in lowered for word in _LINK_SENTENCE_WORDS):
            continue
        found = _ACCUSATION_RE.search(lowered)
        if found:
            problems.append(
                f"identity-link sentence uses the word {found.group(0)!r}: "
                f"{sentence.strip()!r}. An identity_link member is a lead, "
                "described as 'linked, needs review', never as a mule -- and "
                "the word must not appear even inside a negation, because a "
                "truncated sentence reads as an accusation."
            )
    return problems


def check_counting(narrative: str, ring: Any) -> list[str]:
    """Reject an unqualified member count that is not the alertable count.

    An analyst reading "this ring has 12 accounts" takes all 12 as
    alerts. If three of them are same-bank context members, that
    sentence has just accused three people the system explicitly
    declined to alert on. The all-members figure is publishable only
    when the sentence says what the extra members are.
    """
    alertable = len(ring.alertable_member_ids)
    total = len(ring.all_member_ids)
    if alertable == total:
        return []

    problems: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(narrative):
        lowered = sentence.lower()
        qualified = any(word in lowered for word in _CONTEXT_QUALIFIERS)
        for match in _COUNT_NEAR_NOUN.finditer(lowered):
            counted = int(match.group(1))
            if counted == alertable:
                continue
            if counted == total and qualified:
                continue
            if counted == total:
                problems.append(
                    f"unqualified count of {counted} in {sentence.strip()!r}: that "
                    f"is every member, but only {alertable} are alertable. State "
                    f"{alertable}, or name the {total - alertable} context "
                    "member(s) explicitly."
                )
    return problems


def check_uncheckable_counting(narrative: str) -> list[str]:
    """Reject a member count that no ring subject was resolved to check.

    The counting rule is only as good as its ability to run. When the
    claims name no ring, :func:`check_counting` has no alertable
    population to compare against and would otherwise be skipped --
    which means an unqualified "34 accounts" sentence passes silently,
    and a verdict of "passed" stops distinguishing a count that was
    checked from one that never was.

    So this fails closed. The message says explicitly that the count was
    *unchecked* rather than wrong, because those two verdicts call for
    different responses from whoever reads them.
    """
    counts = sorted(
        {int(match.group(1)) for match in _COUNT_NEAR_NOUN.finditer(narrative.lower())}
    )
    if not counts:
        return []
    return [
        f"narrative states member count(s) {counts} but no ring subject could be "
        "resolved from the claims, so the count could not be checked against the "
        "ring's alertable population. This count is unchecked, not wrong. Emit a "
        "ring claim naming the ring the count is about, or drop the count."
    ]


def check_prose(
    narrative: str,
    evidence: list[str],
    ring: Any | None = None,
) -> list[str]:
    """Scan a narrative for framings the project forbids.

    Returns one message per violation. These are not style notes: a
    narrative that says an account was flagged for sharing a device
    describes a detector this system does not have.
    """
    problems: list[str] = []
    lowered = narrative.lower()

    # Sentence-scoped, not a character window. A window is arbitrary:
    # a correct narrative opening "This ring was flagged because it
    # involves 11 alertable accounts..." and mentioning a shared device
    # two sentences later passed only because that word landed at offset
    # 241 rather than 159. Reordering the same correct sentences would
    # have failed it. Causation is a claim made *within* a sentence, so
    # that is the unit to check.
    for sentence in _SENTENCE_SPLIT.split(narrative):
        sentence_lower = sentence.lower()
        phrase = next((p for p in _REUSE_AS_CAUSE if p in sentence_lower), None)
        if phrase is None:
            continue
        if any(word in sentence_lower for word in _REUSE_WORDS):
            problems.append(
                "narrative credits identity reuse with causing the detection in "
                f"{sentence.strip()!r} (near {phrase!r}); detection fires on "
                "velocity and structure only"
            )

    for word in _GUILT_WORDS:
        if _asserted(lowered, word):
            problems.append(
                f"narrative asserts guilt ({word!r}); report behaviour and why it "
                "warrants review, never a conclusion of guilt"
            )

    for line in evidence:
        low = line.lower()
        if "single_bank" not in low:
            continue
        hit = next(
            (
                marker
                for marker in ("strong", "damning", "proves", "incriminat")
                if _asserted(low, marker)
            ),
            None,
        )
        if hit is not None:
            problems.append(
                f"single_bank evidence rendered as incriminating ({hit!r}): {line!r}; "
                "same-bank reuse is routine among households and joint businesses"
            )

    problems.extend(check_identity_link_wording(narrative))
    if ring is not None:
        problems.extend(check_counting(narrative, ring))
    else:
        problems.extend(check_uncheckable_counting(narrative))

    return problems


#: Claim kinds whose ``subject`` is a ring id.
_RING_KINDS = frozenset(
    {
        ClaimKind.RING_MEMBER_COUNT.value,
        ClaimKind.RING_ALERTABLE_MEMBER_COUNT.value,
        ClaimKind.RING_BANK_COUNT.value,
        ClaimKind.RING_BANKS.value,
    }
)


def _ring_subject(result: InvestigationResult) -> Any | None:
    """Resolve the ring an investigation is about, if it names one.

    The counting rule needs the ring to know what the alertable
    population is. Returning ``None`` does not excuse a count: see
    :func:`check_uncheckable_counting`, which fails closed when the
    narrative counts members and no ring was resolved to count them
    against.
    """
    for claim in result.claims:
        if claim.kind in _RING_KINDS:
            try:
                ring, _, _ = tools.get_ring_subgraph(claim.subject, max_hops=1)
            except (KeyError, ValueError):
                return None
            return ring
    return None


def verify(result: InvestigationResult) -> VerificationResult:
    """Check every claim and the narrative prose against the tools.

    Args:
        result: What the investigator produced.

    Returns:
        A :class:`VerificationResult`. ``unsupported`` is empty when and
        only when ``passed`` is true.
    """
    unsupported: list[str] = []

    for claim in result.claims:
        problem = check_claim(claim)
        if problem is not None:
            unsupported.append(f"{claim.id}: {problem}")

    unsupported.extend(
        check_prose(result.narrative, result.evidence, _ring_subject(result))
    )

    # An empty result is not a verified one. This looks like a
    # degenerate case and is not: when a live request failed with a 400,
    # the structured error came back in place of a narrative, parsed to
    # an InvestigationResult with no narrative and no claims, and passed
    # verification in 4 seconds -- a vacuous pass that would have
    # published a case recording that nothing was checked. "No claims
    # failed" is not the same as "the claims held".
    if not result.narrative.strip() and not result.claims:
        unsupported.append(
            "the investigator returned neither a narrative nor any claims; "
            "there is nothing to verify, which is a failure and not a pass"
        )

    # A narrative that asserts things while offering nothing to check is
    # not verified, it is merely unchecked.
    if result.narrative.strip() and not result.claims:
        unsupported.append(
            "narrative makes assertions but carries no claims; every asserted "
            "fact must have a claim the verifier can re-derive"
        )

    return VerificationResult(passed=not unsupported, unsupported=unsupported)


# ---------------------------------------------------------- chat answers
#
# An investigation is a narrative plus claims, and :func:`verify` gates it.
# A chat answer is a short value plus claims, and this is its gate. The
# two are deliberately different functions: the prose rules above are
# about paragraphs, and a one-word answer has no paragraph to scan.


class AnswerStatus(StrEnum):
    """Why one chat answer came out verified or not.

    ``docs/agents_api.md`` exposes a single boolean, which is right for a
    caller and too coarse for measuring: "false" covers a refusal that
    could not be checked and a claim set that missed the answer, and
    those two want different fixes. The eval records this instead.
    """

    #: Every claim re-derived, and at least one of them is the answer.
    VERIFIED = "verified"
    #: A claim was checked and failed. The answer is withheld: ``409``.
    WITHHELD = "withheld"
    #: No claims at all, so nothing in the answer was checkable.
    NOTHING_CHECKABLE = "nothing_checkable"
    #: Claims held, but none of them states the answer. Not verified, and
    #: not withheld either: what was checked was true, it just was not
    #: what the answer said.
    UNBACKED = "unbacked"


#: A free-standing number in an answer. The lookbehind and lookahead are
#: load-bearing: every ring id ends in digits
#: (``ring-mulenet-velocity-rule-v2-00002``) and carries a ``v2``, so a
#: bare ``\d+`` reads an answer naming a ring as claiming 2.
_FREE_NUMBER = re.compile(r"(?<![-\w.])-?\d+(?:\.\d+)?(?![\w-])")


def _numbers(text: str) -> list[float]:
    """Every free-standing number in ``text``, as floats."""
    return [float(match.group(0)) for match in _FREE_NUMBER.finditer(text)]


def states_claim(answer: str, claim: Claim) -> bool:
    """True when ``answer`` actually states the value ``claim`` carries.

    This is **not** claim inference. It never invents a claim, never
    guesses a kind, and cannot make an unclaimed number checkable: with
    no claims there is nothing for it to match and the answer is reported
    as having nothing checkable. It runs the other way round -- given a
    claim the model emitted as structure, it asks whether that claim is
    the one the answer quotes.

    Without it, ``verified: true`` would mean "the model emitted some
    true claim", and a model could satisfy the field with any easy fact
    while the answer said something else entirely. The contract says
    *every quoted claim in* ``answer``; a claim the answer does not quote
    is not one of those.

    Numbers compare numerically, within
    :data:`~shared.tool_interfaces.SCORE_TOLERANCE`, so ``0.987`` matches
    a score claim of ``0.9870000001`` and ``45`` matches ``45``. A list
    matches when every element appears. Anything else is a case-folded
    substring, which is how a bank id or a hashed account id matches.
    """
    value = claim.value
    if isinstance(value, bool):
        return str(value).casefold() in answer.casefold()
    if isinstance(value, int | float):
        return any(abs(number - float(value)) <= SCORE_TOLERANCE for number in _numbers(answer))
    if isinstance(value, list | tuple):
        return bool(value) and all(
            str(item).strip().casefold() in answer.casefold() for item in value
        )
    text = str(value).strip()
    if not text:
        return False
    try:
        number = float(text)
    except ValueError:
        return text.casefold() in answer.casefold()
    return any(abs(found - number) <= SCORE_TOLERANCE for found in _numbers(answer))


@dataclass(frozen=True)
class AnswerVerification:
    """The verdict on one chat answer.

    Attributes:
        answer: The answer as a caller would read it, unit included.
        claims: What the model emitted as structure. Never derived from
            ``answer``.
        failures: One message per claim that was checked and did not
            hold. Non-empty means the answer must not be published.
        stated: Ids of the claims the answer actually quotes.
    """

    answer: str
    claims: list[Claim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    stated: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """True when a claim was checked and failed: withhold the answer."""
        return bool(self.failures)

    @property
    def verified(self) -> bool:
        """The boolean ``docs/agents_api.md`` specifies for ``verified``."""
        return bool(self.stated) and not self.failures

    @property
    def status(self) -> AnswerStatus:
        """The finer-grained outcome, for the eval record."""
        if self.failures:
            return AnswerStatus.WITHHELD
        if not self.claims:
            return AnswerStatus.NOTHING_CHECKABLE
        if not self.stated:
            return AnswerStatus.UNBACKED
        return AnswerStatus.VERIFIED

    @property
    def detail(self) -> str:
        """One line for a log or an error body."""
        if self.failures:
            return "; ".join(self.failures)
        if self.status is AnswerStatus.UNBACKED:
            return (
                f"{len(self.claims)} claim(s) held but none of them states the "
                f"answer {self.answer!r}, so the answer itself was not checked"
            )
        return ""


def verify_answer(answer: str, claims: Sequence[Claim]) -> AnswerVerification:
    """Re-derive every claim a chat answer carries.

    The one gate behind ``POST /chat``'s ``verified`` field, and the same
    function the eval records per question, so the number in a report and
    the number a caller is served cannot come from two different rules.

    Args:
        answer: The answer as published, unit included. Used only to see
            which claims it states -- never to invent one.
        claims: What the model emitted. Every one is checked, including
            claims the answer does not state: a claim the model asserted
            and got wrong is wrong whether or not it made the sentence.

    Returns:
        An :class:`AnswerVerification`. The caller maps ``failed`` to
        ``409 unverified`` and ``verified`` to the contract field.
    """
    checked = list(claims)
    failures = [
        f"{claim.id}: {problem}"
        for claim, problem in ((claim, check_claim(claim)) for claim in checked)
        if problem is not None
    ]
    return AnswerVerification(
        answer=answer,
        claims=checked,
        failures=failures,
        stated=[claim.id for claim in checked if states_claim(answer, claim)],
    )
