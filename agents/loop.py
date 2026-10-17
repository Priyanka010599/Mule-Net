"""The verification loop and the top-level request entrypoint.

The loop is the safety property of this system: an analyst must never
receive a narrative that the tools do not support. Three outcomes, in
order of preference:

1. The investigator's answer verifies. Return the narrative.
2. It does not. Retry **once**, feeding back exactly which claims failed.
   If the retry verifies, return that narrative.
3. It still does not. **Return the raw tool data with a warning, and no
   narrative.** Suppressing the narrative is the point: a plausible
   paragraph that the tools contradict is worse than a table of numbers,
   because it reads as though someone checked it.

Exactly one retry, never more. A second retry buys little -- an agent
that failed the same check twice is not converging -- and it doubles the
worst-case latency on the path an analyst is waiting on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.config import AgentsConfig
from agents.verify import Claim, VerificationResult, verify

#: Prefix on the warning returned when verification fails twice.
UNVERIFIED_WARNING = (
    "UNVERIFIED: the drafted narrative was withheld because claims in it could "
    "not be confirmed against the tools after one correction attempt. The raw "
    "tool output is provided below so it can be reviewed directly. Do not treat "
    "the withheld narrative as evidence."
)


@dataclass
class VerifiedInvestigation:
    """The outcome of running the investigator through the verifier.

    Attributes:
        narrative: The verified narrative, or ``""`` when verification
            failed twice. Empty is meaningful: there is no narrative to
            show, rather than one that happens to be blank.
        claims: The claims backing the narrative.
        evidence: Supporting evidence lines.
        raw: Tool output. Always populated, and the only content when
            ``verified`` is false.
        verification: The final verdict.
        attempts: How many investigator calls were made (1 or 2).
        warnings: Human-readable warnings; non-empty when unverified.
        verified: Whether a narrative passed.
    """

    narrative: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    verification: VerificationResult | None = None
    attempts: int = 0
    warnings: list[str] = field(default_factory=list)
    verified: bool = False

    @property
    def withheld(self) -> bool:
        """True when a narrative was produced but suppressed as unverified."""
        return not self.verified and self.attempts > 0


def investigate_verified(
    subject_id: str,
    config: AgentsConfig | None = None,
) -> VerifiedInvestigation:
    """Run the investigator and gate its answer through the verifier.

    Args:
        subject_id: A ring id or hashed account id.
        config: Selects the mock or Gemini investigator. Defaults to the
            environment.

    Returns:
        A :class:`VerifiedInvestigation`. When ``verified`` is false the
        narrative is empty and ``warnings`` explains why.
    """
    config = config or AgentsConfig.from_env()
    investigator = _investigator_for(config)

    attempt = investigator(subject_id, "")
    verdict = verify(attempt)
    attempts = 1

    if not verdict.passed:
        # One retry, with the failures fed back verbatim.
        attempt = investigator(subject_id, verdict.feedback)
        verdict = verify(attempt)
        attempts = 2

    if verdict.passed:
        return VerifiedInvestigation(
            narrative=attempt.narrative,
            claims=attempt.claims,
            evidence=attempt.evidence,
            raw=attempt.raw,
            verification=verdict,
            attempts=attempts,
            warnings=[],
            verified=True,
        )

    # Failed twice: withhold the narrative, keep the data.
    return VerifiedInvestigation(
        narrative="",
        claims=[],
        evidence=attempt.evidence,
        raw={
            **attempt.raw,
            "withheld_narrative": attempt.narrative,
            "unsupported_claims": list(verdict.unsupported),
        },
        verification=verdict,
        attempts=attempts,
        warnings=[UNVERIFIED_WARNING, *verdict.unsupported],
        verified=False,
    )


def _investigator_for(config: AgentsConfig):
    """Return the investigator callable for the configured mode."""
    if config.is_mock:
        from agents.mock_agents import investigate  # noqa: PLC0415 - mode dispatch

        return investigate

    from agents.adk_agents import investigate_with_gemini  # noqa: PLC0415

    return investigate_with_gemini


@dataclass
class HandledRequest:
    """What the system returns for one analyst request."""

    destination: str
    subject_id: str | None
    reason: str
    answer: Any | None = None
    investigation: VerifiedInvestigation | None = None
    report: Any | None = None
    warnings: list[str] = field(default_factory=list)


def handle(
    request: str,
    config: AgentsConfig | None = None,
) -> HandledRequest:
    """Route one analyst request and run the agent it belongs to.

    The report path runs its investigation through the same verification
    loop, so a report can never be built on an unverified narrative.
    """
    config = config or AgentsConfig.from_env()

    if config.is_mock:
        from agents import mock_agents as impl  # noqa: PLC0415 - mode dispatch
    else:
        from agents import adk_agents as impl  # noqa: PLC0415

    decision = impl.route(request)

    if decision.destination == "analyst_qa":
        return HandledRequest(
            destination=decision.destination,
            subject_id=decision.subject_id,
            reason=decision.reason,
            answer=impl.answer_question(request),
        )

    if decision.subject_id is None:
        return HandledRequest(
            destination=decision.destination,
            subject_id=None,
            reason=decision.reason,
            warnings=[
                "request names no ring id or hashed account id, so there is "
                "nothing to investigate or report on"
            ],
        )

    investigation = investigate_verified(decision.subject_id, config)

    if decision.destination == "investigator":
        return HandledRequest(
            destination=decision.destination,
            subject_id=decision.subject_id,
            reason=decision.reason,
            investigation=investigation,
            warnings=list(investigation.warnings),
        )

    # Report. Refuse to draft on top of an unverified investigation.
    if not investigation.verified:
        return HandledRequest(
            destination=decision.destination,
            subject_id=decision.subject_id,
            reason=decision.reason,
            investigation=investigation,
            warnings=[
                "report not drafted: the underlying investigation did not verify",
                *investigation.warnings,
            ],
        )

    report = impl.draft_report(decision.subject_id)
    return HandledRequest(
        destination=decision.destination,
        subject_id=decision.subject_id,
        reason=decision.reason,
        investigation=investigation,
        report=report,
        warnings=list(report.warnings),
    )
