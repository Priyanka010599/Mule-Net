"""Tests for the tools, the data layer, the agents and the loop.

The expensive fixtures here are module-scoped: the transfers table is
570k rows and building the graph index off it takes seconds, so tests
that need a real subject share one.

Several tests assert on the *framing* of generated text, not just its
values. That is deliberate. "Identity reuse did not cause this
detection" is a factual claim about the system -- the identity branch
was measured and found to add zero true positives -- so a narrative that
gets it wrong is wrong, and a test is the only thing that keeps it right
as the templates change.
"""

from __future__ import annotations

import pytest

from agents import mock_agents, regulations, tools
from agents.config import AgentsConfig, load_prompt
from agents.data import DataStore, MissingTableError, default_store
from agents.eval.questions import DECLINED, EXPECTED_COUNT, build_questions
from agents.eval.run_eval import run_eval, score_answer
from agents.loop import UNVERIFIED_WARNING, handle, investigate_verified
from agents.verify import Claim, InvestigationResult, check_prose, verify
from shared.contracts import (
    Account,
    AccountEvidence,
    EvidenceScope,
    Ring,
    RingExplanation,
    RiskScore,
    Transfer,
)


@pytest.fixture(scope="module")
def store() -> DataStore:
    return default_store()


@pytest.fixture(scope="module")
def ring_id(store: DataStore) -> str:
    return str(store.table("rings")["ring_id"].iloc[0])


@pytest.fixture(scope="module")
def reuse_account(store: DataStore) -> str:
    """An account that actually reuses a document across banks."""
    evidence = store.table("account_evidence")
    return str(
        evidence[evidence["evidence_type"] == "cross_bank_reused_id_number"][
            "hashed_account_id"
        ].iloc[0]
    )


# ------------------------------------------------------------- data layer


def test_real_ml_tables_are_used_not_mocks(store: DataStore) -> None:
    """The published ml/out tables are read, with only ``cases`` mocked."""
    provenance = store.provenance()
    for table in ("rings", "scores", "ring_explanations", "account_evidence"):
        assert provenance[table] is False, f"{table} unexpectedly mocked"


def test_mock_fallback_engages_when_ml_out_is_absent(tmp_path) -> None:
    """Pointing at an empty ml/out builds mock tables instead of failing.

    This is the property that kept the agents unblocked while ml/ was
    still being written, and it has to keep working: a half-published
    ml/out must degrade per table, not break the system. In ``mock`` mode
    only -- see the next test.
    """
    config = AgentsConfig.from_env()
    empty = AgentsConfig(
        mode=config.mode,
        data_dir=config.data_dir,
        ml_out_dir=tmp_path / "nothing",
        model=config.model,
        project=config.project,
        location=config.location,
        use_vertex=config.use_vertex,
    )
    isolated = DataStore(empty)
    rings = isolated.table("rings")
    assert isolated.status("rings").is_mock is True
    assert len(rings) > 0
    assert set(rings.columns) == {
        "ring_id",
        "members",
        "banks_involved",
        "detected_at",
        "status",
    }


def test_a_missing_table_is_a_hard_error_on_the_vertex_path(tmp_path) -> None:
    """Deployed, a mock table is a fluent wrong answer with real provenance.

    The fallback above is right locally and dangerous in cloud mode: the
    reply still reports ``backend: "gemini"`` and carries real-looking
    ``sources``, so the wrong number arrives with authentic provenance. It
    is refused at the read, not only at readiness, so a store built after
    startup cannot reopen the hole.
    """
    config = AgentsConfig.from_env()
    cloud = AgentsConfig(
        mode="vertex",
        data_dir=config.data_dir,
        ml_out_dir=tmp_path / "nothing",
        model=config.model,
        project=config.project,
        location=config.location,
        use_vertex=config.use_vertex,
    )
    assert cloud.allow_mock_tables is False
    with pytest.raises(MissingTableError, match="mock fallback is disabled"):
        DataStore(cloud).table("rings")


def test_unknown_table_is_rejected(store: DataStore) -> None:
    with pytest.raises(KeyError):
        store.table("definitely_not_a_table")


# ----------------------------------------------------------------- tools


def test_get_ring_subgraph_returns_contract_types(ring_id: str) -> None:
    ring, accounts, transfers = tools.get_ring_subgraph(ring_id, max_hops=1)
    assert isinstance(ring, Ring)
    assert ring.all_member_ids
    assert all(isinstance(a, Account) for a in accounts)
    assert all(isinstance(t, Transfer) for t in transfers)
    # Members must appear in the subgraph they anchor.
    returned = {a.hashed_account_id for a in accounts}
    assert set(ring.all_member_ids) <= returned


def test_get_ring_subgraph_rejects_zero_hops(ring_id: str) -> None:
    with pytest.raises(ValueError):
        tools.get_ring_subgraph(ring_id, max_hops=0)


def test_get_ring_subgraph_unknown_ring_raises_key_error() -> None:
    with pytest.raises(KeyError):
        tools.get_ring_subgraph("ring-does-not-exist")


def test_get_explanation_ring(ring_id: str) -> None:
    explanation = tools.get_explanation(ring_id, "ring")
    assert isinstance(explanation, RingExplanation)
    assert explanation.narrative


def test_get_explanation_rejects_bad_subject_type(ring_id: str) -> None:
    with pytest.raises(ValueError):
        tools.get_explanation(ring_id, "galaxy")


def test_get_account_profile(reuse_account: str) -> None:
    account, score, rings = tools.get_account_profile(reuse_account)
    assert isinstance(account, Account)
    assert score is None or isinstance(score, RiskScore)
    assert isinstance(rings, list)


def test_get_account_profile_unknown_account_raises() -> None:
    with pytest.raises(KeyError):
        tools.get_account_profile("0" * 64)


def test_trace_money_flow_rejects_bad_direction(reuse_account: str) -> None:
    with pytest.raises(ValueError):
        tools.trace_money_flow(reuse_account, direction="sideways")


def test_trace_money_flow_paths_are_bounded(reuse_account: str) -> None:
    paths = tools.trace_money_flow(reuse_account, max_hops=2)
    assert len(paths) <= tools.MAX_TRACE_PATHS
    assert all(len(path) <= 2 for path in paths)


# -------------------------------------------- document reuse as evidence


def test_get_document_reuse_returns_account_evidence(reuse_account: str) -> None:
    """The contract returns AccountEvidence rows, not the old dict."""
    rows = tools.get_document_reuse(reuse_account)
    assert rows, "expected reuse for an account known to reuse an id"
    assert all(isinstance(row, AccountEvidence) for row in rows)
    assert all(row.source_table == "kyc_documents" for row in rows)
    assert all(row.observed_at is not None for row in rows)


def test_get_document_reuse_separates_scopes(reuse_account: str) -> None:
    """Same-bank and cross-bank reuse are distinct rows, never merged.

    The scope is what decides whether a row is citable, so collapsing
    them would destroy the only thing that makes the evidence usable.
    """
    rows = tools.get_document_reuse(reuse_account)
    for row in rows:
        assert row.scope in (EvidenceScope.SINGLE_BANK, EvidenceScope.CROSS_BANK)
        if row.scope is EvidenceScope.CROSS_BANK:
            own_bank = tools.get_account_profile(reuse_account)[0].bank_id
            assert own_bank not in row.counterparty_bank_ids


def test_document_reuse_detail_carries_its_caveat(reuse_account: str) -> None:
    """Every reuse sentence qualifies itself, so a quote cannot drop it."""
    for row in tools.get_document_reuse(reuse_account):
        lowered = row.detail.lower()
        assert "not" in lowered and (
            "suspicion" in lowered or "incriminating" in lowered
        )


def test_legitimate_dual_bank_customer_also_shows_cross_bank_reuse(
    store: DataStore,
) -> None:
    """Reuse cannot decide guilt, and the data proves it.

    A labelled-legitimate dual-bank customer produces cross-bank reuse
    evidence indistinguishable from a mule's. If this test ever fails,
    reuse has become separable and the weak-signal caveats could be
    revisited -- until then they are load-bearing.
    """
    import pandas as pd

    ground_truth = pd.read_parquet(
        store.config.data_dir / "ground_truth.parquet"
    )
    dual_bank = ground_truth[
        ground_truth["cohort"] == "hard_negative_dual_bank"
    ]["hashed_account_id"]

    found = False
    for account in dual_bank.head(20):
        rows = tools.get_document_reuse(str(account))
        if any(row.scope is EvidenceScope.CROSS_BANK for row in rows):
            found = True
            break
    assert found, "expected a legitimate dual-bank customer to reuse an id"


# ----------------------------------------------------------- regulations


def test_search_regulations_returns_required_keys() -> None:
    results = tools.search_regulations("rapid pass-through of funds", top_k=3)
    assert results
    for result in results:
        assert {"citation", "excerpt", "source_uri"} <= set(result)


@pytest.mark.parametrize("bad", ["", "   "])
def test_search_regulations_rejects_empty_query(bad: str) -> None:
    with pytest.raises(ValueError):
        tools.search_regulations(bad)


def test_search_regulations_rejects_zero_top_k() -> None:
    with pytest.raises(ValueError):
        tools.search_regulations("pass-through", top_k=0)


def test_corpus_is_flagged_as_synthetic() -> None:
    """Placeholder citations must be detectable as placeholders."""
    results = tools.search_regulations("pass-through of funds", top_k=1)
    assert regulations.is_synthetic(results[0])


# --------------------------------------------------------------- prompts


@pytest.mark.parametrize(
    "name", ["router", "investigator", "report", "analyst_qa"]
)
def test_every_prompt_file_exists_and_is_substantial(name: str) -> None:
    """Prompts are files, not inline strings."""
    assert len(load_prompt(name)) > 200


def test_there_is_no_verifier_prompt() -> None:
    """Verification is deterministic Python, not an agent.

    A prompt file for an agent that does not exist is exactly the sort of
    thing that misleads a reader about what the system does -- and so is
    an agent registered with no prompt to run on, so this checks the
    correspondence in both directions.

    The expected set is derived from ``AGENT_TOOLS`` rather than typed
    out. It was typed out, and adding the ``redteam`` agent then failed
    this test for having done the registration correctly, which is a test
    asserting a hand-copied list rather than the property it is named
    for.
    """
    from agents.adk_agents import AGENT_TOOLS  # noqa: PLC0415 - one call site
    from agents.config import PROMPTS_DIR  # noqa: PLC0415 - one call site

    assert not (PROMPTS_DIR / "verifier.md").exists()
    assert {p.stem for p in PROMPTS_DIR.glob("*.md")} == set(AGENT_TOOLS)


def test_report_prompt_forbids_reuse_as_a_ground() -> None:
    text = load_prompt("report").lower()
    assert "identity reuse is never a ground" in text
    assert "no citation, no ground" in text


# -------------------------------------------------------------- verifier


def test_verify_accepts_claims_the_tools_confirm(ring_id: str) -> None:
    ring, _, _ = tools.get_ring_subgraph(ring_id, max_hops=1)
    result = InvestigationResult(
        narrative="A ring was flagged on velocity and structure.",
        claims=[
            Claim("c1", "ring_member_count", ring_id,
                  len(ring.all_member_ids), "")
        ],
    )
    assert verify(result).passed


def test_verify_rejects_a_wrong_count(ring_id: str) -> None:
    result = InvestigationResult(
        narrative="A ring was flagged.",
        claims=[Claim("c1", "ring_member_count", ring_id, 99999, "")],
    )
    verdict = verify(result)
    assert not verdict.passed
    assert "c1" in verdict.unsupported[0]


def test_verify_rejects_an_uncheckable_claim_kind(ring_id: str) -> None:
    """A claim the verifier cannot re-derive does not get the benefit of doubt."""
    result = InvestigationResult(
        narrative="A ring was flagged.",
        claims=[Claim("c1", "vibes", ring_id, "very suspicious", "")],
    )
    assert not verify(result).passed


def test_verify_rejects_a_narrative_with_no_claims() -> None:
    """An unchecked narrative is not a verified one."""
    result = InvestigationResult(narrative="This account is highly suspicious.")
    assert not verify(result).passed


def test_verify_rejects_unknown_subject() -> None:
    result = InvestigationResult(
        narrative="A ring was flagged.",
        claims=[Claim("c1", "ring_member_count", "ring-nope", 3, "")],
    )
    assert not verify(result).passed


def test_prose_check_rejects_reuse_as_the_cause() -> None:
    """Crediting identity reuse with a detection is a factual error."""
    problems = check_prose(
        "The account was flagged because it shares a device with two others.", []
    )
    assert problems and "velocity and structure" in problems[0]


def test_prose_check_rejects_asserted_guilt() -> None:
    assert check_prose("The shared document proves the account is guilty.", [])


def test_prose_check_allows_correct_single_bank_wording() -> None:
    """The required wording must not trip the check that enforces it.

    "not incriminating" contains "incriminat"; a naive substring match
    rejects exactly the sentence the project asks for.
    """
    assert not check_prose(
        "Flagged on pass-through velocity.",
        ["[single_bank] reused id: weak signal, routine, not incriminating."],
    )


def test_prose_check_rejects_incriminating_single_bank_wording() -> None:
    problems = check_prose(
        "Flagged on pass-through velocity.",
        ["[single_bank] reused id: strong evidence of a mule network."],
    )
    assert problems


# ------------------------------------------------------ investigator/loop


def test_investigation_verifies_first_time(ring_id: str) -> None:
    outcome = investigate_verified(ring_id)
    assert outcome.verified
    assert outcome.attempts == 1
    assert outcome.narrative


def test_investigation_never_blames_identity_reuse(ring_id: str) -> None:
    narrative = mock_agents.investigate(ring_id).narrative.lower()
    assert "velocity" in narrative or "structure" in narrative
    assert "flagged because it shares" not in narrative


def test_evidence_lines_always_state_their_scope(reuse_account: str) -> None:
    lines = mock_agents.investigate(reuse_account).evidence
    assert lines
    assert all(("[cross_bank]" in line) or ("[single_bank]" in line) for line in lines)


def test_loop_retries_once_then_withholds(monkeypatch, ring_id: str) -> None:
    """An investigator that keeps lying gets exactly one retry, then muzzled."""
    calls = []

    def always_wrong(subject_id: str, feedback: str = "") -> InvestigationResult:
        calls.append(feedback)
        return InvestigationResult(
            narrative="This ring has nine hundred members.",
            claims=[Claim("c1", "ring_member_count", subject_id, 900, "")],
        )

    monkeypatch.setattr("agents.mock_agents.investigate", always_wrong)
    outcome = investigate_verified(ring_id)

    assert outcome.attempts == 2, "exactly one retry"
    assert calls[0] == "", "first attempt gets no feedback"
    assert "c1" in calls[1], "the retry is told which claim failed"
    assert not outcome.verified
    assert outcome.narrative == "", "the unverified narrative is withheld"
    assert outcome.withheld
    assert UNVERIFIED_WARNING in outcome.warnings
    # The data survives even though the narrative does not.
    assert outcome.raw["withheld_narrative"]


def test_loop_returns_raw_data_when_unverified(monkeypatch, ring_id: str) -> None:
    def always_wrong(subject_id: str, feedback: str = "") -> InvestigationResult:
        return InvestigationResult(
            narrative="Fabricated.",
            claims=[Claim("c1", "ring_bank_count", subject_id, 42, "")],
        )

    monkeypatch.setattr("agents.mock_agents.investigate", always_wrong)
    outcome = investigate_verified(ring_id)
    assert outcome.raw["unsupported_claims"]


def test_loop_recovers_when_the_retry_is_correct(monkeypatch, ring_id: str) -> None:
    """A corrected retry passes, and the narrative is returned."""
    ring, _, _ = tools.get_ring_subgraph(ring_id, max_hops=1)
    state = {"n": 0}

    def wrong_then_right(subject_id: str, feedback: str = "") -> InvestigationResult:
        state["n"] += 1
        value = 999 if state["n"] == 1 else len(ring.all_member_ids)
        return InvestigationResult(
            narrative="Flagged on velocity and structure.",
            claims=[Claim("c1", "ring_member_count", subject_id, value, "")],
        )

    monkeypatch.setattr("agents.mock_agents.investigate", wrong_then_right)
    outcome = investigate_verified(ring_id)
    assert outcome.verified
    assert outcome.attempts == 2
    assert outcome.narrative


# --------------------------------------------------------------- router


@pytest.mark.parametrize(
    ("request_text", "destination"),
    [
        ("How many rings are there?", "analyst_qa"),
        ("What is the score for that account?", "analyst_qa"),
        ("Why was this ring flagged?", "investigator"),
        ("Walk me through ring-x", "investigator"),
        ("Trace the money out of that account", "investigator"),
        ("Draft an STR for ring-x", "report"),
        ("Prepare a compliance report", "report"),
    ],
)
def test_router_destinations(request_text: str, destination: str) -> None:
    assert mock_agents.route(request_text).destination == destination


def test_router_extracts_a_ring_id(ring_id: str) -> None:
    assert mock_agents.route(f"Why was {ring_id} flagged?").subject_id == ring_id


def test_router_invents_no_subject() -> None:
    assert mock_agents.route("How many rings are there?").subject_id is None


# ------------------------------------------------------------ analyst qa


def test_qa_answers_a_count() -> None:
    answer = mock_agents.answer_question("How many rings has the detector found?")
    assert answer.answer.isdigit()
    assert answer.basis


def test_qa_declines_ground_truth_questions() -> None:
    """The agent must not substitute a detection number for a label."""
    answer = mock_agents.answer_question(
        "How many accounts are truly mules according to the labels?"
    )
    assert answer.answer == DECLINED
    assert not any(char.isdigit() for char in answer.answer)


def test_qa_declines_recall_questions() -> None:
    answer = mock_agents.answer_question("What is the detector's recall?")
    assert answer.answer == DECLINED


# ---------------------------------------------------------------- report


def test_report_cites_every_ground(ring_id: str) -> None:
    """No citation, no ground."""
    draft = mock_agents.draft_report(ring_id)
    assert draft.grounds
    for ground in draft.grounds:
        assert ground.citation
        assert ground.source_uri
    assert len(draft.citations) == len(draft.grounds)


def test_report_carries_the_synthetic_warning(ring_id: str) -> None:
    draft = mock_agents.draft_report(ring_id)
    assert mock_agents.SYNTHETIC_CORPUS_WARNING in draft.narrative


def test_report_never_grounds_suspicion_on_identity(ring_id: str) -> None:
    """Reuse appears under evidence, after the grounds, never as one."""
    draft = mock_agents.draft_report(ring_id)
    for ground in draft.grounds:
        lowered = ground.ground.lower()
        assert "document" not in lowered
        assert "device" not in lowered
        assert "identity" not in lowered


def test_report_drops_grounds_it_cannot_cite(monkeypatch, ring_id: str) -> None:
    """With no corpus, an empty report is the correct outcome."""
    monkeypatch.setattr(mock_agents.tools, "search_regulations", lambda q, top_k=5: [])
    draft = mock_agents.draft_report(ring_id)
    assert draft.grounds == []
    assert "No citable ground" in draft.narrative
    assert draft.warnings


def test_report_states_limitations(ring_id: str) -> None:
    assert "Limitations" in mock_agents.draft_report(ring_id).narrative


def test_handle_refuses_to_report_on_an_unverified_investigation(
    monkeypatch, ring_id: str
) -> None:
    """A report is never built on a narrative that failed verification."""

    def always_wrong(subject_id: str, feedback: str = "") -> InvestigationResult:
        return InvestigationResult(
            narrative="Fabricated.",
            claims=[Claim("c1", "ring_member_count", subject_id, 12345, "")],
        )

    monkeypatch.setattr("agents.mock_agents.investigate", always_wrong)
    handled = handle(f"Draft an STR for {ring_id}")
    assert handled.report is None
    assert any("did not verify" in w for w in handled.warnings)


# ------------------------------------------------------------------ eval


def test_eval_set_has_twenty_five_questions() -> None:
    assert len(build_questions()) == EXPECTED_COUNT


def test_eval_set_ids_are_unique() -> None:
    questions = build_questions()
    assert len({q.id for q in questions}) == len(questions)


def test_eval_set_covers_both_categories() -> None:
    categories = {q.category for q in build_questions()}
    assert categories == {"detection", "declined"}


def test_scorer_accepts_numeric_equivalence() -> None:
    question = build_questions()[0]
    assert score_answer(question, question.expected)


def test_scorer_rejects_a_declined_question_answered_with_a_number() -> None:
    """A refusal that still leaks a number has not refused."""
    declined = next(q for q in build_questions() if q.category == "declined")
    assert not score_answer(declined, "cannot be answered, but it is 899")
    assert score_answer(declined, DECLINED)


def test_eval_runs_and_scores_both_metrics(tmp_path) -> None:
    # The recorder defaults to agents/out/eval/, which is published
    # output. A test run must not leave artefacts there pretending to be
    # a real measurement, so it is pointed at a temporary directory.
    from agents.eval.record import EvalRecorder

    report = run_eval(
        verification_limit=3,
        recorder=EvalRecorder(model="mock", location="none", out_dir=tmp_path),
    )
    assert report.total == EXPECTED_COUNT
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.verifier_pass_rate <= 1.0
    assert report.verifier_total == 3
