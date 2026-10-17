"""Durable, incremental evidence for an eval run.

Why this exists: ``run_eval`` printed its report and persisted nothing, so
a run that died at question 20 of 25 left **no** evidence of the 19 that
had succeeded. Two live runs were lost that way -- one to an account rate
limit, one to a crash -- and each time the paid-for numbers died with the
process. Model calls cost money; a measurement that cannot survive its own
process is a measurement you have to buy twice.

The format is JSON Lines, written and flushed **per record**. That is the
whole design:

* One ``run`` header line, then one line per question and per verified
  ring. A crash truncates the file at a line boundary and everything
  before it is still readable.
* Appended and flushed immediately, never buffered to the end. Anything
  held in memory for tidiness is exactly what gets lost.
* A reader tolerates a torn final line, because a process killed
  mid-write leaves one.

Files land in ``agents/out/eval/``, beside the other published artefacts,
one per run so concurrent runs of different models cannot interleave.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Where runs are written, beside ``cases.parquet``.
DEFAULT_EVAL_DIR = Path(__file__).resolve().parent.parent / "out" / "eval"

# ALLOWED-MODEL-LITERALS: these are pricing-tier prefixes, not model ids
# used to call a model. Nothing here is ever passed to Vertex; the id that
# is comes from agents.config.DEFAULT_MODEL alone. A new family needs a
# new rate, so the literals have to live somewhere, and a rate table is
# the honest place.

#: Rough per-million-token prices, USD, for turning token counts into a
#: cost estimate. Kept beside the recorder rather than inlined so the
#: number in a record can be re-derived when prices change.
#:
#: These are list prices at the time of the run, not billed amounts. The
#: token counts in each record are the objective part; ``cost_usd`` is
#: arithmetic over them.
RATES_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    # prefix: (input, output). Longest matching prefix wins.
    "gemini-2.5": (0.15, 1.25),
    # 3.5-flash specifically: $1.50/$9.00, not the generic "gemini-3"
    # bucket's $0.75/$3.75 -- confirmed against multiple third-party
    # pricing trackers (Google's own pricing page would not render
    # through this session's fetch tooling), not Google's page directly.
    # Re-check against cloud.google.com/vertex-ai/generative-ai/pricing
    # before trusting this for a real budget decision.
    "gemini-3.5": (1.50, 9.00),
    "gemini-3": (0.75, 3.75),
}


def estimate_cost(model: str, usage: dict[str, dict[str, int]]) -> float:
    """Cost of one question's token usage, in USD.

    Thinking tokens bill as output, which is why they are added to it
    rather than reported on their own: a total that hides the split makes
    the bill look inexplicable, but a total that omits them understates
    it.
    """
    rate_in, rate_out = next(
        (
            rates
            for prefix, rates in sorted(
                RATES_USD_PER_MILLION.items(), key=lambda kv: -len(kv[0])
            )
            if model.startswith(prefix)
        ),
        (0.0, 0.0),
    )
    total = 0.0
    for bucket in usage.values():
        total += bucket.get("prompt", 0) / 1e6 * rate_in
        total += (bucket.get("output", 0) + bucket.get("thinking", 0)) / 1e6 * rate_out
    return round(total, 6)


@dataclass
class EvalRecorder:
    """Appends one JSON line per result, flushing each time.

    Args:
        model: The model id under test.
        location: The Vertex location it was called in. Recorded because
            the same model id behaves differently by region, and a result
            without its region cannot be compared to anything.
        out_dir: Defaults to ``agents/out/eval``.
        started_at: Overridable for tests.
    """

    model: str
    location: str
    out_dir: Path = field(default_factory=lambda: DEFAULT_EVAL_DIR)
    started_at: datetime | None = None
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        stamp = (self.started_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
        safe = f"{self.model}__{self.location}".replace("/", "-")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{safe}__{stamp}.jsonl"
        self._write(
            {
                "kind": "run",
                "model": self.model,
                "location": self.location,
                "started_at": (self.started_at or datetime.now(UTC)).isoformat(),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        """Append one line and get it onto the disk.

        ``flush`` alone only clears Python's buffer; ``fsync`` is what
        survives the process being killed, which is the case this whole
        module exists for.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def question(
        self,
        *,
        question_id: str,
        question: str,
        category: str,
        expected: str,
        actual: str,
        correct: bool,
        latency_s: float,
        usage: dict[str, dict[str, int]],
        verification: str = "",
        claim_count: int = 0,
        unsupported: list[str] | None = None,
    ) -> None:
        """Record one graded question.

        Args:
            verification: What :func:`agents.verify.verify_answer` made of
                the claims this answer carried, as an
                :class:`agents.verify.AnswerStatus` value. Recorded per
                question rather than only totalled, because the question
                worth looking at is the one whose answer was *correct* and
                withheld, and a total cannot point at it.
            claim_count: How many claims the model emitted. Zero is the
                honest reading of "nothing checkable was claimed".
            unsupported: One message per claim that was checked and
                failed. Empty unless ``verification`` is ``withheld``.
        """
        self._write(
            {
                "kind": "question",
                "model": self.model,
                "location": self.location,
                "id": question_id,
                "question": question,
                "category": category,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "verification": verification,
                "claim_count": claim_count,
                "unsupported": list(unsupported or []),
                "latency_s": round(latency_s, 2),
                "usage": usage,
                "cost_usd": estimate_cost(self.model, usage),
                "at": datetime.now(UTC).isoformat(),
            }
        )

    def verification(
        self,
        *,
        subject: str,
        verified: bool,
        attempts: int,
        unsupported: list[str],
        latency_s: float,
        usage: dict[str, dict[str, int]],
    ) -> None:
        """Record one ring investigation put through the verifier."""
        self._write(
            {
                "kind": "verification",
                "model": self.model,
                "location": self.location,
                "subject": subject,
                "verified": verified,
                "attempts": attempts,
                "unsupported": unsupported,
                "latency_s": round(latency_s, 2),
                "usage": usage,
                "cost_usd": estimate_cost(self.model, usage),
                "at": datetime.now(UTC).isoformat(),
            }
        )

    def summary(self, report: dict[str, Any]) -> None:
        """Record the finished report. Absent when a run died early."""
        self._write(
            {
                "kind": "summary",
                "model": self.model,
                "location": self.location,
                "report": report,
                "at": datetime.now(UTC).isoformat(),
            }
        )


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read back one run's records, tolerating a torn final line.

    A process killed mid-write leaves a partial last line. Dropping it is
    correct: the records before it are complete and are the evidence the
    run was paying for.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Only ever the last line, and only when the writer died
            # inside it. Anything else would mean the file was corrupted
            # some other way, which this reader cannot repair either.
            break
    return records
