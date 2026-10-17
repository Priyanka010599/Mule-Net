"""Chat answers carry typed claims, and the verifier re-derives them.

Every test here is offline and free. Two things are faked and nothing
else: the model, by replacing :func:`agents.adk_agents._run_agent`, and
the published tables, by writing a six-account dataset to a temporary
directory and pointing ``MULENET_DATA_DIR`` / ``MULENET_ML_OUT_DIR`` at
it. The data layer, the tool layer, the verifier and the HTTP service all
run their real code over it.

**Why a synthetic dataset rather than the published parquet.** The rest
of the agents suite reads ``simulator/out`` and ``ml/out``, which are
build artefacts and absent from a fresh clone, so those tests cannot run
until a 30,000-account dataset has been generated. The property under
test here -- that a claim the model emits is re-derived from the tables
and that a wrong one withholds the answer -- does not need 30,000
accounts, and a test that only runs on a developer's machine does not
guard anything on a clean checkout. The numbers below are small enough to
state in the test and derived from the same tables the verifier reads, so
an agreement between them is still an agreement between two paths.

The four properties ``docs/agents_api.md`` fixes, one test each:

* a claim that re-derives -> ``verified: true``
* an answer with no checkable claim -> ``verified: false``, not an error
* a claim that is checked and **fails** -> ``409 unverified``, no answer
* claims are never inferred from the answer text
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import adk_agents, data, mock_agents, service, tools, verify
from agents.config import DEFAULT_MODEL, AgentsConfig, ServiceLimits
from agents.verify import AnswerStatus, Claim, ClaimKind

#: The ring the fixture builds, spanning all three banks.
RING_ID = "ring-mulenet-velocity-rule-v2-00001"

BANKS = ("BANK_A", "BANK_B", "BANK_C")


def _account_id(index: int) -> str:
    """A 64-hex id, the only account identifier this system accepts."""
    return f"{index:064x}"


@dataclass(frozen=True)
class Tiny:
    """The fixture dataset's ids and the counts derived from it."""

    ring_id: str
    accounts: list[str]
    #: Accounts in ``ring_id``, one per bank.
    members: list[str]
    #: The two accounts sharing a device across a bank boundary.
    reusers: list[str]
    account_count: int
    bank_count: int
    bank_a_count: int
    evidence_rows: int
    top_account: str
    top_score: float
    model_version: str


def _write_tables(root: Path) -> Tiny:
    """Write a six-account, three-bank dataset and return what it contains.

    ``rings``, ``scores`` and ``account_evidence`` are **not** written:
    ``agents.data`` derives them from these tables when ``ml/out`` has no
    parquet, which is the same code path a developer runs before ``ml/``
    has published anything.
    """
    import pandas as pd

    data_dir = root / "simulator_out"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "ml_out").mkdir(parents=True, exist_ok=True)

    ids = [_account_id(n) for n in range(1, 7)]
    banks = ["BANK_A", "BANK_B", "BANK_C", "BANK_A", "BANK_B", "BANK_A"]
    opened = datetime(2026, 1, 1, tzinfo=UTC)

    pd.DataFrame(
        {
            "hashed_account_id": ids,
            "bank_id": banks,
            "opened_at": [opened + timedelta(days=n) for n in range(len(ids))],
            "account_type": ["individual"] * len(ids),
            "status": ["active"] * len(ids),
        }
    ).to_parquet(data_dir / "accounts.parquet")

    # a1 -> a2 -> a3 inside the window, so the mock scorer sees
    # pass-through behaviour and the three ring members score highest.
    hops = [
        (ids[0], ids[1], 0),
        (ids[1], ids[2], 5),
        (ids[3], ids[0], 60),
        (ids[4], ids[1], 65),
        (ids[5], ids[2], 70),
    ]
    pd.DataFrame(
        {
            "transfer_id": [f"t{n}" for n in range(len(hops))],
            "from_hashed_account_id": [src for src, _, _ in hops],
            "to_hashed_account_id": [dst for _, dst, _ in hops],
            "amount": [Decimal("100.00")] * len(hops),
            "currency": ["INR"] * len(hops),
            "timestamp": [
                opened + timedelta(minutes=minute) for _, _, minute in hops
            ],
            "channel": ["imps"] * len(hops),
        }
    ).to_parquet(data_dir / "transfers.parquet")

    pd.DataFrame(
        {
            "hashed_account_id": ids,
            "bank_id": banks,
            "ring_id": [RING_ID, RING_ID, RING_ID, "", "", ""],
            "role": ["mule", "mule", "collector", "clean", "clean", "clean"],
            "cohort": ["mule_ring"] * 3 + ["clean"] * 3,
            "pattern_type": ["fast_passthrough_only"] * 3 + [""] * 3,
            "is_mule": [True, True, True, False, False, False],
            "opened_at": [opened + timedelta(days=n) for n in range(len(ids))],
        }
    ).to_parquet(data_dir / "ground_truth.parquet")

    # One device shared by a BANK_A and a BANK_B account: exactly two
    # cross-bank evidence rows, one per side of the link.
    pd.DataFrame(
        {
            "hashed_account_id": [ids[0], ids[1], ids[3]],
            "device_fingerprint": ["shared-device", "shared-device", "solo-device"],
        }
    ).to_parquet(data_dir / "devices.parquet")

    pd.DataFrame(
        {
            "doc_id": [f"d{n}" for n in range(len(ids))],
            "hashed_account_id": ids,
            "doc_type": ["passport"] * len(ids),
            # Distinct per account, so identity reuse contributes nothing
            # and the evidence count is the device link alone.
            "hashed_id_number": [f"id-{n}" for n in range(len(ids))],
            "normalized_address": [f"{n} main street" for n in range(len(ids))],
            "photo_embedding_ref": [None] * len(ids),
        }
    ).to_parquet(data_dir / "kyc_documents.parquet")

    pd.DataFrame(
        {
            "hashed_account_id": ids,
            "submitted_at": [opened + timedelta(days=n) for n in range(len(ids))],
        }
    ).to_parquet(data_dir / "kyc_submissions.parquet")

    return Tiny(
        ring_id=RING_ID,
        accounts=ids,
        members=ids[:3],
        reusers=[ids[0], ids[1]],
        account_count=len(ids),
        bank_count=len(set(banks)),
        bank_a_count=banks.count("BANK_A"),
        evidence_rows=2,
        top_account="",  # filled by the fixture, once scores are derived
        top_score=0.0,
        model_version=data.MOCK_MODEL_VERSION,
    )


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tiny:
    """A loaded dataset the verifier can re-derive claims from.

    The caches are the fiddly part: ``default_store`` and the transfer
    graph are process-wide ``lru_cache``es, so they are cleared on the way
    in *and* on the way out. Leaving a store built over ``tmp_path``
    behind would hand the next test a directory that no longer exists.
    """
    described = _write_tables(tmp_path)
    monkeypatch.setenv("MULENET_DATA_DIR", str(tmp_path / "simulator_out"))
    monkeypatch.setenv("MULENET_ML_OUT_DIR", str(tmp_path / "ml_out"))
    data.default_store.cache_clear()
    tools._graph.cache_clear()
    try:
        scores = data.default_store().table("scores")
        top = scores.sort_values("score", ascending=False).iloc[0]
        yield Tiny(
            **{
                **described.__dict__,
                "top_account": str(top["hashed_account_id"]),
                "top_score": float(top["score"]),
            }
        )
    finally:
        data.default_store.cache_clear()
        tools._graph.cache_clear()


# --------------------------------------------------------------------------
# the population claim kinds, re-derived from the tables
# --------------------------------------------------------------------------


def test_every_claim_kind_has_a_check(tiny: Tiny) -> None:
    """A kind the verifier cannot re-derive must not exist in the enum.

    ``ClaimKind`` is what the prompts advertise. A member with no entry in
    ``_CHECKS`` would be advertised to the model and then fail every time
    it was used, withholding correct answers.
    """
    assert {kind.value for kind in ClaimKind} == set(verify._CHECKS)


@pytest.mark.parametrize(
    ("kind", "subject", "value"),
    [
        (ClaimKind.TABLE_ROW_COUNT, "accounts", 6),
        (ClaimKind.TABLE_ROW_COUNT, "rings", 1),
        (ClaimKind.FILTERED_ROW_COUNT, "accounts.bank_id=BANK_A", 3),
        # Quoted the way a model that writes SQL all day writes it.
        (ClaimKind.FILTERED_ROW_COUNT, "accounts.bank_id='BANK_A'", 3),
        (ClaimKind.FILTERED_ROW_COUNT, "account_evidence.scope=cross_bank", 2),
        (
            ClaimKind.FILTERED_ROW_COUNT,
            "account_evidence.evidence_type=cross_bank_shared_device",
            2,
        ),
        (ClaimKind.DISTINCT_VALUE_COUNT, "accounts.bank_id", 3),
        (ClaimKind.DISTINCT_VALUE_COUNT, "account_evidence.hashed_account_id", 2),
        (ClaimKind.RING_BANK_SPAN_COUNT, "3", 1),
        (ClaimKind.RING_BANK_SPAN_COUNT, "2", 0),
        (ClaimKind.RING_BANK_SPAN_COUNT, ">1", 1),
        # The wordier spellings a model reaches for. Leniency about the
        # spelling of a subject, never about the count.
        (ClaimKind.RING_BANK_SPAN_COUNT, "> 1", 1),
        (ClaimKind.RING_BANK_SPAN_COUNT, "3 banks", 1),
        (ClaimKind.RING_MEMBER_COUNT, RING_ID, 3),
        (ClaimKind.RING_ALERTABLE_MEMBER_COUNT, RING_ID, 3),
        (ClaimKind.RING_BANK_COUNT, RING_ID, 3),
        (ClaimKind.RING_BANKS, RING_ID, list(BANKS)),
    ],
)
def test_a_true_claim_re_derives(
    tiny: Tiny, kind: ClaimKind, subject: str, value: Any
) -> None:
    claim = Claim(id="c1", kind=kind.value, subject=subject, value=value, text="")
    assert verify.check_claim(claim) is None


@pytest.mark.parametrize(
    ("kind", "subject", "value"),
    [
        (ClaimKind.TABLE_ROW_COUNT, "accounts", 7),
        (ClaimKind.FILTERED_ROW_COUNT, "accounts.bank_id=BANK_A", 4),
        (ClaimKind.DISTINCT_VALUE_COUNT, "accounts.bank_id", 2),
        (ClaimKind.RING_BANK_SPAN_COUNT, "3", 2),
        (ClaimKind.RING_BANKS, RING_ID, ["BANK_A", "BANK_B"]),
        (ClaimKind.COLUMN_DISTINCT_VALUES, "scores.model_version", ["made-up"]),
        (ClaimKind.TOP_RISK_ACCOUNT, "scores", _account_id(6)),
    ],
)
def test_a_false_claim_is_caught(
    tiny: Tiny, kind: ClaimKind, subject: str, value: Any
) -> None:
    claim = Claim(id="c1", kind=kind.value, subject=subject, value=value, text="")
    assert verify.check_claim(claim) is not None


def test_the_model_version_and_top_account_re_derive(tiny: Tiny) -> None:
    """The two kinds whose value is a string rather than a count."""
    version = Claim(
        id="c1",
        kind=ClaimKind.COLUMN_DISTINCT_VALUES.value,
        subject="scores.model_version",
        value=[tiny.model_version],
        text="",
    )
    top = Claim(
        id="c2",
        kind=ClaimKind.TOP_RISK_ACCOUNT.value,
        subject="scores",
        value=tiny.top_account,
        text="",
    )
    assert verify.check_claim(version) is None
    assert verify.check_claim(top) is None
    # Naming the column instead of the table is a spelling difference, not
    # a different claim: the derivation is the maximum of scores.score
    # either way.
    by_column = Claim(
        id="c3",
        kind=ClaimKind.TOP_RISK_ACCOUNT.value,
        subject="scores.score",
        value=tiny.top_account,
        text="",
    )
    assert verify.check_claim(by_column) is None


def test_a_single_string_stands_in_for_a_one_value_list(tiny: Tiny) -> None:
    """``column_distinct_values`` accepts the bare value where there is one."""
    claim = Claim(
        id="c1",
        kind=ClaimKind.COLUMN_DISTINCT_VALUES.value,
        subject="scores.model_version",
        value=tiny.model_version,
        text="",
    )
    assert verify.check_claim(claim) is None


def test_a_claim_cannot_reach_a_table_the_agents_may_not_read(tiny: Tiny) -> None:
    """``ground_truth`` is on disk here and still not verifiable.

    The labels are deliberately outside ``ALLOWED_SQL_TABLES``, and a
    claim is another way to ask a question about a table. If a claim could
    count ``ground_truth`` rows, an agent could get the labelled mule
    count published as a verified fact by claiming it.
    """
    claim = Claim(
        id="c1",
        kind=ClaimKind.TABLE_ROW_COUNT.value,
        subject="ground_truth",
        value=6,
        text="",
    )
    problem = verify.check_claim(claim)
    assert problem is not None
    assert "ground_truth" in problem


@pytest.mark.parametrize(
    "subject",
    [
        "accounts",  # no column
        "accounts.no_such_column",
        "accounts.bank_id.extra",
        "no_such_table.bank_id",
    ],
)
def test_a_malformed_population_subject_fails_rather_than_raising(
    tiny: Tiny, subject: str
) -> None:
    """A bad subject is a claim failure, never a 500.

    The message has to say what was wrong with it, because the same text
    is what an analyst is shown in place of the withheld answer.
    """
    claim = Claim(
        id="c1",
        kind=ClaimKind.DISTINCT_VALUE_COUNT.value,
        subject=subject,
        value=3,
        text="",
    )
    assert verify.check_claim(claim) is not None


def test_an_invented_claim_kind_cannot_pass(tiny: Tiny) -> None:
    """The model does not get to define what counts as verified."""
    claim = Claim(
        id="c1", kind="ring_vibes", subject=RING_ID, value=3, text=""
    )
    problem = verify.check_claim(claim)
    assert problem is not None
    assert "cannot be verified" in problem


# --------------------------------------------------------------------------
# states_claim: which claim the answer actually quotes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "value", "expected"),
    [
        ("45 rings", 45, True),
        ("45 rings", 44, False),
        ("0.987", 0.987, True),
        ("0.987", 0.912, False),
        ("BANK_A, BANK_B", ["BANK_A", "BANK_B"], True),
        ("BANK_A", ["BANK_A", "BANK_B"], False),
        ("BANK_C", "BANK_C", True),
        # The trap this regex exists for: a ring id ends in digits and
        # contains "v2", so a bare \\d+ makes any answer naming a ring
        # look like it states a count of 2.
        (f"{RING_ID} has been reviewed", 2, False),
        (f"{RING_ID} has been reviewed", 1, False),
        (f"{RING_ID}", RING_ID, True),
    ],
)
def test_states_claim_matches_only_what_the_answer_says(
    answer: str, value: Any, expected: bool
) -> None:
    claim = Claim(id="c1", kind="table_row_count", subject="rings", value=value, text="")
    assert verify.states_claim(answer, claim) is expected


# --------------------------------------------------------------------------
# verify_answer: the four outcomes
# --------------------------------------------------------------------------


def _claim(kind: ClaimKind, subject: str, value: Any) -> Claim:
    return Claim(id="c1", kind=kind.value, subject=subject, value=value, text="")


def test_verified_when_the_claim_re_derives_and_is_the_answer(tiny: Tiny) -> None:
    verdict = verify.verify_answer(
        "6 accounts", [_claim(ClaimKind.TABLE_ROW_COUNT, "accounts", 6)]
    )
    assert verdict.status is AnswerStatus.VERIFIED
    assert verdict.verified is True
    assert verdict.failed is False


def test_nothing_checkable_when_no_claims_were_emitted(tiny: Tiny) -> None:
    """The answer is right and says so; it just carries nothing to check."""
    verdict = verify.verify_answer("6 accounts", [])
    assert verdict.status is AnswerStatus.NOTHING_CHECKABLE
    assert verdict.verified is False
    assert verdict.failed is False


def test_withheld_when_a_claim_is_checked_and_fails(tiny: Tiny) -> None:
    verdict = verify.verify_answer(
        "7 accounts", [_claim(ClaimKind.TABLE_ROW_COUNT, "accounts", 7)]
    )
    assert verdict.status is AnswerStatus.WITHHELD
    assert verdict.failed is True
    assert verdict.verified is False
    assert "tools return 6" in verdict.detail


def test_a_true_claim_about_something_else_does_not_verify(tiny: Tiny) -> None:
    """The hole this closes, and the reason ``states_claim`` exists.

    Without it a model could satisfy ``verified`` with any easy true fact
    while answering something else entirely: here the claim is a correct
    account count and the answer is a ring count. Nothing failed, so the
    answer is not withheld -- but nothing in the answer was checked, so it
    is not verified either.
    """
    verdict = verify.verify_answer(
        "1 ring", [_claim(ClaimKind.TABLE_ROW_COUNT, "accounts", 6)]
    )
    assert verdict.status is AnswerStatus.UNBACKED
    assert verdict.verified is False
    assert verdict.failed is False


def test_every_claim_is_checked_even_the_ones_the_answer_ignores(tiny: Tiny) -> None:
    """A claim the model got wrong is wrong whether it made the sentence.

    The answer here is backed by a true claim, so the temptation is to
    stop looking. The second claim is false, and publishing an answer that
    arrived alongside a false assertion is exactly what the withhold rule
    is for.
    """
    verdict = verify.verify_answer(
        "6 accounts",
        [
            _claim(ClaimKind.TABLE_ROW_COUNT, "accounts", 6),
            Claim(
                id="c2",
                kind=ClaimKind.TABLE_ROW_COUNT.value,
                subject="rings",
                value=99,
                text="",
            ),
        ],
    )
    assert verdict.status is AnswerStatus.WITHHELD
    assert verdict.failed is True


# --------------------------------------------------------------------------
# the offline agent emits the same claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("How many accounts are in the dataset?", ClaimKind.TABLE_ROW_COUNT),
        ("How many rings has the detector found?", ClaimKind.TABLE_ROW_COUNT),
        ("How many banks are in the dataset?", ClaimKind.DISTINCT_VALUE_COUNT),
        ("How many accounts are held at BANK_A?", ClaimKind.FILTERED_ROW_COUNT),
        (
            "How many detected rings span three banks?",
            ClaimKind.RING_BANK_SPAN_COUNT,
        ),
        ("Which account has the highest risk score?", ClaimKind.TOP_RISK_ACCOUNT),
        (
            "Which model version produced the risk scores?",
            ClaimKind.COLUMN_DISTINCT_VALUES,
        ),
        (f"How many banks does {RING_ID} span?", ClaimKind.RING_BANK_COUNT),
        (f"Which banks are involved in {RING_ID}?", ClaimKind.RING_BANKS),
        (
            f"How many accounts are in {RING_ID}?",
            ClaimKind.RING_ALERTABLE_MEMBER_COUNT,
        ),
    ],
)
def test_the_offline_agent_answers_verify(
    tiny: Tiny, question: str, kind: ClaimKind
) -> None:
    """Mock mode is where the eval runs without a bill, so it must verify.

    This is also a check on the claim kinds themselves: the offline agent
    answers from its own query and claims through the verifier's
    re-derivation, and the two have to agree on every question in the eval
    set's shape.
    """
    answer = mock_agents.answer_question(question)
    assert [claim.kind for claim in answer.claims] == [kind.value]
    verdict = verify.verify_answer(answer.quoted, answer.claims)
    assert verdict.status is AnswerStatus.VERIFIED, verdict.detail


def test_a_declined_question_carries_no_claims(tiny: Tiny) -> None:
    """A refusal has nothing to check, and must not pretend otherwise."""
    answer = mock_agents.answer_question(
        "How many accounts are truly mules according to the labels?"
    )
    assert answer.answer == mock_agents.CANNOT_ANSWER
    assert answer.claims == []
    verdict = verify.verify_answer(answer.quoted, answer.claims)
    assert verdict.status is AnswerStatus.NOTHING_CHECKABLE


def test_an_account_question_verifies_its_score_and_bank(tiny: Tiny) -> None:
    subject = tiny.top_account
    score = mock_agents.answer_question(f"What is the risk score for {subject}?")
    bank = mock_agents.answer_question(f"Which bank holds account {subject}?")
    for answer in (score, bank):
        verdict = verify.verify_answer(answer.quoted, answer.claims)
        assert verdict.status is AnswerStatus.VERIFIED, verdict.detail


def test_a_question_no_claim_kind_covers_is_answered_unclaimed(tiny: Tiny) -> None:
    """No kind covers an intra-ring transfer count, and none is invented.

    Stretching ``table_row_count`` to cover it would be a false claim
    about the ``transfers`` table. Answering with no claim is the honest
    outcome the contract has a value for.
    """
    answer = mock_agents.answer_question(f"How many transfers are in {RING_ID}?")
    assert answer.claims == []


# --------------------------------------------------------------------------
# over HTTP: the contract in docs/agents_api.md
# --------------------------------------------------------------------------


def _config(**overrides: Any) -> AgentsConfig:
    base = AgentsConfig.from_env()
    fields: dict[str, Any] = {
        "mode": "vertex",
        "data_dir": base.data_dir,
        "ml_out_dir": base.ml_out_dir,
        "model": DEFAULT_MODEL,
        "project": "mulenet",
        "location": base.location,
        "use_vertex": True,
    }
    fields.update(overrides)
    return AgentsConfig(**fields)


def _client(monkeypatch: pytest.MonkeyPatch, analyst_qa: dict[str, Any]) -> TestClient:
    """A service whose only fake is the model's reply.

    ``analyst_qa`` is the raw payload the model "returns", claims
    included, so the claims reaching the verifier have travelled the real
    parsing path in :func:`agents.adk_agents.answer_question`.
    """

    def fake_run_agent(
        name: str,
        message: str,
        config: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        if name == "analyst_qa":
            # On a bare thread, the way ADK's synchronous Runner does it,
            # so the provenance sink is exercised as it is in production.
            def run_tools() -> None:
                adk_agents._model_tool(tools.describe_tables, sink)()
                # The counts these tests have the model state, as a query
                # would return them: the grounding check refuses a number
                # no tool produced.
                for query in (
                    "SELECT COUNT(*) AS n FROM accounts",
                    "SELECT COUNT(*) AS n FROM rings",
                    "SELECT COUNT(DISTINCT bank_id) AS n FROM accounts",
                ):
                    adk_agents._model_tool(tools.run_sql, sink)(query)

            thread = threading.Thread(target=run_tools)
            thread.start()
            thread.join()
            return json.dumps(analyst_qa)
        return json.dumps(
            {"destination": "analyst_qa", "subject_id": None, "reason": "a count"}
        )

    monkeypatch.setattr(adk_agents, "_run_agent", fake_run_agent)
    # The readiness gate is not what these tests are about, and it cannot
    # pass here for a reason unrelated to them: it stats the *published*
    # paths, including ``agents/out/cases.parquet``, which no fixture can
    # relocate. The ``ml/out`` tables the fixture does provide are derived
    # by ``agents.data`` from the fixture's own simulator tables, so the
    # answer and the verification still read one store. The gate has its
    # own tests in ``test_service.py``.
    monkeypatch.setattr(service, "missing_tables", lambda _config: [])
    limits = ServiceLimits(
        session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000
    )
    return TestClient(service.create_app(_config(), limits))


QUESTION = {"session_id": "s1", "message": "How many accounts are in the dataset?"}


def test_chat_is_verified_when_the_claim_re_derives(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path that has never been reachable: ``verified: true`` on chat."""
    client = _client(
        monkeypatch,
        {
            "answer": "6",
            "basis": "SELECT COUNT(*) FROM accounts",
            "unit": "accounts",
            "claims": [
                {
                    "id": "c1",
                    "kind": "table_row_count",
                    "subject": "accounts",
                    "value": 6,
                    "text": "There are 6 accounts.",
                }
            ],
        },
    )
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verified"] is True
    assert body["answer"] == "6 accounts"
    assert body["backend"] == "gemini"


def test_chat_is_unverified_when_nothing_checkable_was_claimed(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """False is "not machine-checked". It is still a ``200`` with an answer."""
    client = _client(
        monkeypatch,
        {
            "answer": "6",
            "basis": "SELECT COUNT(*) FROM accounts",
            "unit": "accounts",
            "claims": [],
        },
    )
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 200, response.text
    assert response.json()["verified"] is False
    assert response.json()["answer"] == "6 accounts"


def test_chat_withholds_an_answer_whose_claim_fails(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``409 unverified``: the answer is not published, not flagged.

    Reachable for the first time. The claim says seven accounts, the
    tables hold six, and the caller gets no answer at all -- the failure
    message rather than a number with a caveat beside it.
    """
    client = _client(
        monkeypatch,
        {
            "answer": "7",
            "basis": "SELECT COUNT(*) FROM accounts",
            "unit": "accounts",
            "claims": [
                {
                    "id": "c1",
                    "kind": "table_row_count",
                    "subject": "accounts",
                    "value": 7,
                    "text": "There are 7 accounts.",
                }
            ],
        },
    )
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 409
    body = response.json()
    assert set(body) == {"error_type", "message", "retry_after_s"}
    assert body["error_type"] == "unverified"
    assert "withheld" in body["message"]
    assert "7" in body["message"] and "6" in body["message"]
    # The withheld answer must not be in the envelope: publishing it in a
    # message is publishing it.
    assert "answer" not in body


def test_chat_never_infers_a_claim_from_the_answer_text(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing negative test.

    The answer here is *correct and re-derivable*: the tables really do
    hold six accounts, and a verifier that read the number out of the
    sentence would confirm it and report ``verified: true``. Nothing may
    do that. A claim has to arrive from the model as structure, because a
    number matched out of prose says nothing about whether the model meant
    it as a count of accounts -- and once phrasing can produce
    ``verified: true``, the field stops meaning anything.
    """
    client = _client(
        monkeypatch,
        {
            "answer": "6 accounts, across 3 banks, 1 ring",
            "basis": "SELECT COUNT(*) FROM accounts",
            "unit": "",
            "claims": [],
        },
    )
    body = client.post("/chat", json=QUESTION).json()
    assert body["verified"] is False


def test_a_claim_the_answer_does_not_state_is_not_published_as_verified(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A true claim about something else does not verify the answer.

    Over HTTP this is a ``200`` with ``verified: false``: nothing failed,
    so there is nothing to withhold, and nothing in the answer was checked,
    so there is nothing to claim either.
    """
    client = _client(
        monkeypatch,
        {
            "answer": "1",
            "basis": "SELECT COUNT(*) FROM rings",
            "unit": "ring",
            "claims": [
                {
                    "id": "c1",
                    "kind": "table_row_count",
                    "subject": "accounts",
                    "value": 6,
                    "text": "There are 6 accounts.",
                }
            ],
        },
    )
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 200, response.text
    assert response.json()["verified"] is False


def test_an_invented_claim_kind_withholds_the_answer(
    tiny: Tiny, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kind outside the table is uncheckable, and uncheckable is withheld.

    The alternative -- publish it as "nothing checkable" -- would let a
    model opt out of verification by naming a kind that does not exist,
    and would report that as the same thing as a question with no numbers
    in it.
    """
    client = _client(
        monkeypatch,
        {
            "answer": "6",
            "basis": "SELECT COUNT(*) FROM accounts",
            "unit": "accounts",
            "claims": [
                {
                    "id": "c1",
                    "kind": "account_count",
                    "subject": "accounts",
                    "value": 6,
                    "text": "six accounts",
                }
            ],
        },
    )
    response = client.post("/chat", json=QUESTION)
    assert response.status_code == 409
    assert response.json()["error_type"] == "unverified"


# --------------------------------------------------------------------------
# the prompt and the enum cannot drift
# --------------------------------------------------------------------------


def test_the_analyst_qa_prompt_documents_every_claim_kind() -> None:
    """A kind the prompt omits is a kind the model will never emit.

    Derived from ``ClaimKind`` rather than listed here, so adding a kind
    without documenting it fails instead of quietly never being used.
    """
    from agents.config import load_prompt

    prompt = load_prompt("analyst_qa")
    for kind in ClaimKind:
        assert f"`{kind.value}`" in prompt, f"analyst_qa.md never names {kind.value}"


def test_the_prompt_says_a_refusal_carries_no_claims() -> None:
    """The rule that keeps a refusal out of the withhold path."""
    from agents.config import load_prompt

    prompt = load_prompt("analyst_qa")
    assert '"claims": []' in prompt
    assert "refusal carries no claims" in prompt.lower()


def test_the_eval_records_a_verification_outcome_for_every_question() -> None:
    """Every bucket is named in the report, including the empty ones.

    A breakdown that omits what did not happen cannot be compared with a
    run where it did, which is the entire job of a before/after eval.
    """
    from agents.eval.run_eval import NO_ANSWER, EvalReport

    buckets = {status.value for status in AnswerStatus} | {NO_ANSWER}
    assert "chat_verification" in EvalReport.__dataclass_fields__
    assert len(buckets) == 5
