"""Score the eval set: answer accuracy and verifier pass rate.

Run it with::

    python -m agents.eval.run_eval

Two numbers come out, and they measure different things.

**Accuracy** is over the 25 questions in :mod:`agents.eval.questions`.
The agent answers through the router, the tools and the SQL guard; the
expected answer is derived independently with pandas. Note what this
does *not* measure in mock mode: the offline agent's language step is a
keyword matcher, not a model, so accuracy here is a regression guard on
the tool and data path. It is not a Gemini score, and it would be
dishonest to quote it as one.

**Chat verification** is the same question asked of the chat path: of the
25 answers, how many carried a claim the verifier re-derived. It is
reported as the four outcomes of :class:`agents.verify.AnswerStatus`
rather than one rate, because "not verified" covers two different things
-- a refusal with nothing to check, and a claim set that missed the
answer -- and a single percentage would hide which one moved. ``withheld``
is the count that would have been a ``409 unverified`` over HTTP: an
answer whose claim was checked and failed is not published at all.

Accuracy is graded on the answer the agent produced, **not** on whether
it verified. The two measure different things and collapsing them would
make the before/after comparison meaningless: a question can be answered
correctly and carry no claim, and a question can verify while the number
is the wrong one for what was asked. A ``withheld`` on an answer this
harness grades *correct* is the interesting case, and it is in the
records per question so it can be found rather than averaged away.

**Verifier pass rate** is the fraction of investigations whose narrative
survived :func:`agents.verify.verify`. This is the number that says
whether the system is safe to show an analyst, because a failed
verification withholds the narrative. It is reported alongside the retry
rate, since a pass that needed a retry is a weaker result than a pass
first time.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.adk_agents import (
    LAST_USAGE,
    CallBudgetExceeded,
    ModelUnusableError,
    reset_usage,
)
from agents.config import AgentsConfig
from agents.data import default_store
from agents.eval.questions import EvalQuestion, build_questions, ground_truth_summary
from agents.eval.record import EvalRecorder
from agents.loop import handle, investigate_verified
from agents.verify import AnswerStatus, verify_answer
from shared.tool_interfaces import SCORE_TOLERANCE

#: Wordings that count as a refusal on a ``declined`` question.
_REFUSALS = (
    "cannot be answered",
    "not an allowed table",
    "ground truth is not",
    "no rows",
    "not available",
)


#: Recorded where no answer was produced at all -- a route elsewhere, an
#: exhausted budget, an unusable model. Not a verification outcome, and
#: kept out of :class:`agents.verify.AnswerStatus` so it cannot be
#: mistaken for one.
NO_ANSWER = "no_answer"


@dataclass
class QuestionResult:
    """One graded question.

    ``correct`` grades the answer. ``verification`` says what the
    deterministic verifier made of the claims that came with it, as an
    :class:`agents.verify.AnswerStatus` value or :data:`NO_ANSWER`. The
    two are independent on purpose: see the module docstring.
    """

    id: str
    question: str
    category: str
    expected: str
    actual: str
    basis: str
    correct: bool
    verification: str = NO_ANSWER
    claim_count: int = 0
    unsupported: list[str] = field(default_factory=list)


@dataclass
class VerificationSample:
    """One investigation run through the verifier."""

    subject: str
    verified: bool
    attempts: int
    unsupported: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Everything the harness measured."""

    mode: str
    accuracy: float
    correct: int
    total: int
    accuracy_by_category: dict[str, float]
    verifier_pass_rate: float
    verifier_passed: int
    verifier_total: int
    retry_rate: float
    withheld: int
    data_provenance: dict[str, bool]
    ground_truth: dict[str, object]
    questions: list[QuestionResult] = field(default_factory=list)
    verifications: list[VerificationSample] = field(default_factory=list)
    #: Chat answers by verification outcome, keyed by
    #: :class:`agents.verify.AnswerStatus` value plus :data:`NO_ANSWER`.
    #: Every question appears in exactly one bucket, so the values sum to
    #: ``total``.
    chat_verification: dict[str, int] = field(default_factory=dict)
    #: Questions graded correct whose answer was withheld. Each one means
    #: the verifier or the claim emission is wrong, never that the answer
    #: was: worth a name in the report rather than a number.
    withheld_but_correct: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    cleaned = text.strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", cleaned)


def score_answer(question: EvalQuestion, actual: str) -> bool:
    """Grade one answer against its expected value.

    Numbers compare numerically so ``45`` and ``45.0`` agree. A
    ``declined`` question is correct when the answer is a clear refusal
    and contains no digits: a refusal that still leaks a number has
    answered the question it was meant to decline.
    """
    expected = _normalise(question.expected)
    got = _normalise(actual)

    if question.category == "declined":
        refused = any(phrase in got for phrase in _REFUSALS)
        leaked_number = bool(re.search(r"\d", got))
        return refused and not leaked_number

    if got == expected:
        return True

    try:
        # <= SCORE_TOLERANCE, not a literal, and not <. The tolerance is
        # defined as the maximum error rounding to SCORE_DECIMALS can
        # introduce, so a difference of exactly that much is a rounding
        # artefact rather than a disagreement. The literal 1e-6 that used
        # to be here is what failed q16 live: a 3dp-rounded score differed
        # from the expected value by 1.9e-6 and was graded wrong.
        return abs(float(got) - float(expected)) <= SCORE_TOLERANCE
    except ValueError:
        return False


def run_accuracy(
    questions: list[EvalQuestion],
    config: AgentsConfig,
    recorder: EvalRecorder | None = None,
) -> list[QuestionResult]:
    """Put every question to the system and grade the answer.

    Each graded question is written to ``recorder`` as it completes, not
    collected and dumped at the end: a run killed at question 20 has to
    leave 19 questions of evidence behind. Two live runs were lost before
    this existed.
    """
    results = []
    for question in questions:
        started = time.perf_counter()
        reset_usage()
        try:
            handled = handle(question.question, config)
        except CallBudgetExceeded as exc:
            # One question that will not converge must not cost the other
            # 24. Recorded as a failure with the budget named, because a
            # model that needs more turns than it is allowed is a fact
            # about the model and belongs in the score.
            results.append(
                _record_question(
                    recorder,
                    question,
                    actual=f"(call budget exhausted: {exc})",
                    basis="",
                    started=started,
                )
            )
            continue
        except ModelUnusableError as exc:
            # The model produced nothing parseable after its retries. That
            # is a wrong answer, not a reason to abandon the other 24: an
            # eval that aborts on the first bad response measures nothing
            # and hides how often this happens. Recorded as a failure with
            # the mode trail, so the rate is visible in the report.
            results.append(
                _record_question(
                    recorder,
                    question,
                    actual=f"(no usable response from {exc.agent}: {exc.trail})",
                    basis="",
                    started=started,
                )
            )
            continue
        if handled.answer is not None:
            actual, basis = handled.answer.answer, handled.answer.basis
        else:
            # Routed somewhere other than analyst_qa. For a factual
            # question that is itself a failure, recorded as such rather
            # than retried through a different agent.
            actual = f"(routed to {handled.destination})"
            basis = ""
        results.append(
            _record_question(
                recorder,
                question,
                actual=actual,
                basis=basis,
                started=started,
                answer=handled.answer,
            )
        )
    return results


def _record_question(
    recorder: EvalRecorder | None,
    question: EvalQuestion,
    *,
    actual: str,
    basis: str,
    started: float,
    answer: Any | None = None,
) -> QuestionResult:
    """Grade one question, verify its claims, persist it, return the result.

    Grading happens here so the persisted record and the in-memory result
    can never disagree about whether a question passed.

    The verifier runs over exactly what a caller would be served -- the
    answer with its unit, through ``QaAnswer.quoted`` -- and over the
    claims the model emitted, never over the graded ``expected`` value. A
    verifier that could see the expected answer would be measuring
    nothing.

    Args:
        answer: The ``QaAnswer`` the agent produced, or ``None`` when no
            answer was produced at all.
    """
    verdict = None if answer is None else verify_answer(answer.quoted, answer.claims)
    result = QuestionResult(
        id=question.id,
        question=question.question,
        category=question.category,
        expected=question.expected,
        actual=actual,
        basis=basis,
        correct=score_answer(question, actual),
        verification=NO_ANSWER if verdict is None else str(verdict.status),
        claim_count=0 if verdict is None else len(verdict.claims),
        unsupported=[] if verdict is None else list(verdict.failures),
    )
    if recorder is not None:
        recorder.question(
            question_id=result.id,
            question=result.question,
            category=result.category,
            expected=result.expected,
            actual=result.actual,
            correct=result.correct,
            latency_s=time.perf_counter() - started,
            usage=dict(LAST_USAGE),
            verification=result.verification,
            claim_count=result.claim_count,
            unsupported=result.unsupported,
        )
    return result


def run_verifier_pass_rate(
    config: AgentsConfig,
    limit: int = 20,
    recorder: EvalRecorder | None = None,
) -> list[VerificationSample]:
    """Investigate a sample of real rings and record what verified."""
    store = default_store()
    ring_ids = store.table("rings")["ring_id"].tolist()[:limit]

    samples = []
    for ring_id in ring_ids:
        started = time.perf_counter()
        reset_usage()
        try:
            outcome = investigate_verified(str(ring_id), config)
        except CallBudgetExceeded as exc:
            samples.append(
                _record_verification(
                    recorder,
                    VerificationSample(
                        subject=str(ring_id),
                        verified=False,
                        attempts=0,
                        unsupported=[f"call budget exhausted: {exc}"],
                    ),
                    started,
                )
            )
            continue
        except ModelUnusableError as exc:
            # Counted as not verified, for the same reason as above: a
            # ring the investigator could not answer for is a ring
            # without a verified narrative, which is exactly what the
            # pass rate is measuring. Dropping it would flatter the rate.
            samples.append(
                _record_verification(
                    recorder,
                    VerificationSample(
                        subject=str(ring_id),
                        verified=False,
                        attempts=0,
                        unsupported=[
                            f"no usable response from {exc.agent}: {exc.trail}"
                        ],
                    ),
                    started,
                )
            )
            continue
        samples.append(
            _record_verification(
                recorder,
                VerificationSample(
                    subject=str(ring_id),
                    verified=outcome.verified,
                    attempts=outcome.attempts,
                    unsupported=(
                        list(outcome.verification.unsupported)
                        if outcome.verification
                        else []
                    ),
                ),
                started,
            )
        )
    return samples


def _record_verification(
    recorder: EvalRecorder | None,
    sample: VerificationSample,
    started: float,
) -> VerificationSample:
    """Persist one verification sample and return it unchanged."""
    if recorder is not None:
        recorder.verification(
            subject=sample.subject,
            verified=sample.verified,
            attempts=sample.attempts,
            unsupported=list(sample.unsupported),
            latency_s=time.perf_counter() - started,
            usage=dict(LAST_USAGE),
        )
    return sample


def run_eval(
    config: AgentsConfig | None = None,
    verification_limit: int = 20,
    recorder: EvalRecorder | None = None,
) -> EvalReport:
    """Run both halves of the eval and assemble the report.

    ``recorder`` persists each question and each investigation as it
    finishes. Defaults to one under ``agents/out/eval/`` -- the default is
    on rather than off because the runs worth recording are the live ones,
    and those are exactly the runs nobody remembers to pass a flag for.
    """
    config = config or AgentsConfig.from_env()
    if recorder is None:
        recorder = EvalRecorder(model=config.model, location=config.location)
    questions = build_questions(config)

    graded = run_accuracy(questions, config, recorder)
    samples = run_verifier_pass_rate(config, verification_limit, recorder)

    correct = sum(1 for result in graded if result.correct)
    by_category: dict[str, float] = {}
    for category in sorted({result.category for result in graded}):
        subset = [r for r in graded if r.category == category]
        by_category[category] = sum(1 for r in subset if r.correct) / len(subset)

    verified = sum(1 for sample in samples if sample.verified)
    retried = sum(1 for sample in samples if sample.attempts > 1)

    # Every bucket named, including the empty ones: a breakdown that
    # omits what did not happen cannot be compared against a run where it
    # did. Derived from AnswerStatus rather than listed here, so a new
    # outcome shows up in the report without anyone remembering to add it.
    buckets = {status.value: 0 for status in AnswerStatus}
    buckets[NO_ANSWER] = 0
    for result in graded:
        buckets[result.verification] = buckets.get(result.verification, 0) + 1

    report = EvalReport(
        mode=config.mode,
        accuracy=correct / len(graded) if graded else 0.0,
        correct=correct,
        total=len(graded),
        accuracy_by_category=by_category,
        verifier_pass_rate=verified / len(samples) if samples else 0.0,
        verifier_passed=verified,
        verifier_total=len(samples),
        retry_rate=retried / len(samples) if samples else 0.0,
        withheld=len(samples) - verified,
        data_provenance=default_store().provenance(),
        ground_truth=ground_truth_summary(config),
        questions=graded,
        verifications=samples,
        chat_verification=buckets,
        withheld_but_correct=[
            result.id
            for result in graded
            if result.correct and result.verification == AnswerStatus.WITHHELD
        ],
    )
    recorder.summary(asdict(report))
    return report


def format_report(report: EvalReport) -> str:
    """Render the report for a terminal."""
    lines = [
        "MuleNet agent eval",
        "=" * 60,
        f"mode                : {report.mode}",
        f"accuracy            : {report.correct}/{report.total} "
        f"({report.accuracy:.1%})",
    ]
    for category, value in report.accuracy_by_category.items():
        lines.append(f"  {category:<18}: {value:.1%}")
    lines.append(
        "chat verification   : "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(report.chat_verification.items())
        )
    )
    if report.withheld_but_correct:
        lines.append(
            "  WITHHELD BUT CORRECT: "
            + ", ".join(report.withheld_but_correct)
            + " -- a correct answer was withheld, so the claim emission or the "
            "verifier is wrong. Investigate; do not relax the check."
        )
    lines += [
        f"verifier pass rate  : {report.verifier_passed}/{report.verifier_total} "
        f"({report.verifier_pass_rate:.1%})",
        f"  needed a retry    : {report.retry_rate:.1%}",
        f"  narrative withheld: {report.withheld}",
        "",
        "mocked tables       : "
        + (
            ", ".join(name for name, mocked in report.data_provenance.items() if mocked)
            or "none, all tables are real"
        ),
        "",
        "failures:",
    ]
    failures = [result for result in report.questions if not result.correct]
    if not failures:
        lines.append("  none")
    for result in failures:
        lines.append(
            f"  {result.id} [{result.category}] {result.question}\n"
            f"      expected: {result.expected!r}\n"
            f"      actual  : {result.actual!r}\n"
            f"      verifier: {result.verification}"
            + (f" ({'; '.join(result.unsupported)})" if result.unsupported else "")
        )
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Score the MuleNet agent eval set.")
    parser.add_argument(
        "--json", action="store_true", help="emit the full report as JSON"
    )
    parser.add_argument(
        "--verification-limit",
        type=int,
        default=20,
        help="how many rings to investigate for the verifier pass rate",
    )
    args = parser.parse_args()

    report = run_eval(verification_limit=args.verification_limit)
    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
