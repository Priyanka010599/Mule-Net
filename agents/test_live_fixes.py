"""Regression tests for the defects the first live Gemini run exposed.

Every test here corresponds to something that actually broke against
the real model, and most of them could not have been caught in mock
mode: the mock investigator always calls ``get_ring_subgraph`` with
``max_hops=1``, never writes a hallucinated column name, and never
orders its sentences differently from one run to the next.

That is the general lesson worth recording. Mock mode pins the shape of
the system; only a live model explores the inputs a real caller
produces.
"""

from __future__ import annotations

import os

import pytest

from agents import tools
from agents.adk_agents import (
    LAST_USAGE,
    THINKING_BUDGETS,
    _accumulate_usage,
    _error_payload,
    _model_tool,
    _status_code,
    reset_usage,
)
from agents.data import default_store
from agents.verify import check_prose
from shared.tool_interfaces import (
    ALLOWED_SQL_TABLES,
    SQL_ROW_LIMIT,
    assert_score_fields_rounded,
)

#: Ring 00001's verbatim narrative from the first live run. It passed
#: verification, but only by luck of sentence order -- see the tests
#: below.
LIVE_NARRATIVE_SENTENCES = [
    "This ring was flagged because it involves 11 alertable accounts across "
    "2 banks (BANK_A, BANK_B).",
    "Nine of these accounts were independently flagged by the velocity rule "
    "and form a connected component of the transfer graph.",
    "The remaining 2 accounts share a device fingerprint or a hashed KYC ID "
    "number with one of the flagged members at a different bank.",
    "The 9 velocity members show a median of 32 inbound and 37 outbound "
    "transfers, forwarding funds in a median of 1049 seconds, and passing "
    "through a median of 87% of what they receive.",
    "Identity reuse across banks is attached as supporting evidence but did "
    "not contribute to the detection decision.",
]


@pytest.fixture(scope="module")
def big_ring_id() -> str:
    """The ring whose two-hop subgraph killed a live request."""
    return "ring-mulenet-velocity-rule-v2-00002"


# ======================================================== 1. bounded subgraph


def test_default_max_hops_is_one() -> None:
    """The old default of 2 returned 29,582 accounts and blew the context."""
    import inspect

    assert inspect.signature(tools.get_ring_subgraph).parameters[
        "max_hops"
    ].default == 1


def test_subgraph_is_capped(big_ring_id: str) -> None:
    result = tools.get_ring_subgraph(big_ring_id)
    assert len(result.accounts) <= tools.MAX_SUBGRAPH_ACCOUNTS
    assert len(result.transfers) <= tools.MAX_SUBGRAPH_TRANSFERS


def test_two_hops_is_capped_too(big_ring_id: str) -> None:
    """Two hops is 29,582 accounts uncapped; the cap must hold there too."""
    result = tools.get_ring_subgraph(big_ring_id, max_hops=2)
    assert len(result.accounts) <= tools.MAX_SUBGRAPH_ACCOUNTS
    assert result.truncated
    assert result.total_accounts > tools.MAX_SUBGRAPH_ACCOUNTS


def test_truncation_reports_true_totals(big_ring_id: str) -> None:
    """The flag exists so a model does not count from a partial list."""
    result = tools.get_ring_subgraph(big_ring_id)
    assert result.truncated
    assert result.total_accounts > len(result.accounts)
    assert "TRUNCATED" in result.note
    assert str(result.total_accounts) in result.note


def test_every_ring_member_survives_truncation(big_ring_id: str) -> None:
    """Members are never dropped: member counts must stay exact.

    Truncating members would turn a size cap into a wrong answer, since
    every claim the verifier checks is about the member population.
    """
    result = tools.get_ring_subgraph(big_ring_id)
    returned = {account.hashed_account_id for account in result.accounts}
    assert set(result.ring.all_member_ids) <= returned


def test_subgraph_still_unpacks_as_a_triple(big_ring_id: str) -> None:
    """Existing callers unpack three values; that must keep working."""
    ring, accounts, transfers = tools.get_ring_subgraph(big_ring_id)
    assert ring.ring_id == big_ring_id
    assert isinstance(accounts, list)
    assert isinstance(transfers, list)


def test_small_ring_is_not_flagged_as_truncated(store_ring_id: str) -> None:
    result = tools.get_ring_subgraph(store_ring_id)
    if len(result.accounts) < tools.MAX_SUBGRAPH_ACCOUNTS:
        assert not result.truncated
        assert result.note == ""


@pytest.fixture(scope="module")
def store_ring_id() -> str:
    return str(default_store().table("rings")["ring_id"].iloc[0])


# ==================================================== 2. structured tool errors


def test_tool_error_is_returned_not_raised() -> None:
    """A raising tool killed the whole live request. It must not raise."""
    wrapped = _model_tool(tools.run_sql)
    result = wrapped("SELECT COUNT(*) FROM rings WHERE num_banks = 3")
    assert isinstance(result, dict)
    assert result["error"]["error_type"]
    assert "num_banks" in result["error"]["message"]


def test_sql_error_carries_the_schema() -> None:
    """The model must be able to self-correct from the error alone."""
    result = _model_tool(tools.run_sql)("SELECT nope FROM rings")
    error = result["error"]
    assert sorted(error["allowed_tables"]) == sorted(ALLOWED_SQL_TABLES)
    assert "banks_involved" in [c["name"] for c in error["schema"]["rings"]]
    assert error["row_limit"] == SQL_ROW_LIMIT


def test_rejected_sql_is_an_error_value_not_an_exception() -> None:
    """A blocked write is reported, not raised, so the model can retry."""
    result = _model_tool(tools.run_sql)("DROP TABLE accounts")
    assert result["error"]["recoverable"] is True


def test_unknown_ring_is_an_error_value() -> None:
    result = _model_tool(tools.get_ring_subgraph)("ring-does-not-exist")
    assert result["error"]["error_type"] == "KeyError"


def test_successful_tool_call_is_untouched() -> None:
    assert _model_tool(tools.run_sql)("SELECT COUNT(*) AS n FROM rings") == [
        {"n": 45}
    ]


def test_wrapper_preserves_the_tool_declaration() -> None:
    """ADK builds the model-facing declaration from the signature."""
    wrapped = _model_tool(tools.run_sql)
    assert wrapped.__name__ == "run_sql"
    assert wrapped.__doc__ == tools.run_sql.__doc__


def test_non_sql_error_has_no_schema_payload() -> None:
    payload = _error_payload("trace_money_flow", ValueError("bad direction"))
    assert "allowed_tables" not in payload["error"]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (type("E", (Exception,), {"code": 429})(), 429),
        (type("E", (Exception,), {"code": 400})(), 400),
        (Exception("429 RESOURCE_EXHAUSTED"), 429),
        (Exception("400 INVALID_ARGUMENT token count exceeds"), 400),
        (Exception("something else entirely"), None),
    ],
)
def test_status_code_classification(exc: Exception, expected: int | None) -> None:
    assert _status_code(exc) == expected


# ================================================= 4. schema visibility


def test_describe_tables_covers_every_allowed_table() -> None:
    described = tools.describe_tables()["tables"]
    assert set(described) == set(ALLOWED_SQL_TABLES)


def test_describe_tables_answers_the_question_that_failed_live() -> None:
    """q05 was refused as unanswerable; banks_involved makes it answerable."""
    columns = tools.describe_tables("rings")["tables"]["rings"]
    names = [column["name"] for column in columns]
    assert "banks_involved" in names
    assert "num_banks" not in names, "the hallucinated column must not appear"


def test_array_columns_say_how_to_filter_them() -> None:
    """A model told only 'ARRAY' writes SQL that silently returns nothing."""
    columns = tools.describe_tables("rings")["tables"]["rings"]
    banks = next(c for c in columns if c["name"] == "banks_involved")
    assert "LIKE" in banks["description"]


def test_array_hint_offers_json1_not_only_like() -> None:
    """The hint must not talk a model out of an answerable question.

    Live, q05 was refused with "cannot determine the number of banks in
    the JSON array... described as JSON text to be filtered with LIKE
    rather than parsed". The schema was right and the refusal was wrong,
    and the hint caused it: SQLite's JSON1 is built in and available.
    """
    columns = tools.describe_tables("rings")["tables"]["rings"]
    banks = next(c for c in columns if c["name"] == "banks_involved")
    assert "json_array_length" in banks["description"]


def test_json1_functions_execute_through_the_guard() -> None:
    """The hint would be a lie if the validator rejected these."""
    rows = tools.run_sql(
        "SELECT COUNT(*) AS n FROM rings WHERE json_array_length(banks_involved) = 3"
    )
    assert rows[0]["n"] == 15


@pytest.mark.parametrize(
    "fabricated",
    [
        # Both of these came back from live report-agent runs. Neither
        # carries a placeholder marker, because an invented citation is
        # not labelled synthetic -- it is simply made up.
        "https://www.fincen.gov/regulations/statutes/bank-secrecy-act",
        "Synthetic Citation: Section 10.1 - Rapid movement of funds "
        "through multiple accounts.",
        "31 CFR 1099.999 - Totally Invented Section",
        "",
    ],
)
def test_fabricated_citations_do_not_verify(fabricated: str) -> None:
    """A citation is checked against the corpus, never against its looks."""
    from agents import regulations

    assert not regulations.verify_citation(fabricated, "pass-through of funds")


def test_a_real_corpus_citation_verifies() -> None:
    from agents import regulations

    real = regulations.search("rapid pass-through of funds", 1)[0]["citation"]
    assert regulations.verify_citation(real)


# ------------------------------- the real 825-page corpus breaks re-search
#
# The four citations below are what the live ``mulenet-aml-regulations``
# index returned for RECORD_KEEPING_QUERY on 2026-09-27. Every one of
# them failed ``verify_citation`` under the old behaviour, which used the
# citation string as its own search query: "PMLA, 2002, p. 14" carries
# almost no semantic signal about what is on page 14, so on a real corpus
# the passage never came back in the top 25 and the report agent dropped
# every genuine ground it had. Verification is now by identity against
# what retrieval actually returned.

#: The query that produced the four citations.
RECORD_KEEPING_QUERY = "record keeping obligations for reporting entities"

#: Verbatim from the live index, in returned order.
LIVE_CITATIONS = (
    "PML (Maintenance of Records) Rules, 2005, p. 10",
    "PMLA, 2002, p. 14",
    "RBI Master Direction - KYC, 2016, p. 49",
    "FIU-IND Reporting Format, p. 12",
)

#: A page that document does not have. Same document as a real citation,
#: so only the page is invented -- the hardest fabrication to catch.
FABRICATED_PAGE = "PMLA, 2002, p. 9999"

#: Gate for tests that query the live Vertex AI Search index. Separate
#: from MULENET_RUN_LIVE_GEMINI because it bills Discovery Engine, not
#: Gemini, and a reader should be able to tell which service a run costs.
LIVE_SEARCH_ENV = "MULENET_RUN_LIVE_SEARCH"


def _fake_cloud_index(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Offline stand-in for the live index, with its measured behaviour.

    Answers RECORD_KEEPING_QUERY with the four real citations and
    *nothing else at all* -- which is the point. On the real corpus a
    citation string used as its own query does not retrieve its own
    passage, so a fake that answered it would test a corpus that does not
    exist.
    """
    if query.strip() != RECORD_KEEPING_QUERY:
        return []
    return [
        {
            "citation": citation,
            "excerpt": "Reporting entities shall maintain records of all "
            "transactions for a period of five years.",
            "source_uri": "https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf",
        }
        for citation in LIVE_CITATIONS
    ][:top_k]


@pytest.fixture
def clean_ledger():
    """Isolate the retrieval ledger, which is process-global."""
    from agents import regulations

    regulations.reset_retrieved()
    yield regulations
    regulations.reset_retrieved()


def test_a_real_cloud_citation_verifies_after_retrieval(monkeypatch, clean_ledger) -> None:
    """The fix, offline: retrieval provenance, not a second search.

    Fails against the old behaviour -- the final assertion was ``False``
    for all four of these citations against the live index.
    """
    regulations = clean_ledger
    monkeypatch.setattr(regulations, "_docai_search", lambda: _fake_cloud_index)

    citation = "PMLA, 2002, p. 14"
    # Nothing retrieved yet, and the citation cannot find itself. This is
    # exactly the state the report agent used to verify in.
    assert not regulations.verify_citation(citation)

    returned = [result["citation"] for result in regulations.search(RECORD_KEEPING_QUERY, 4)]
    assert citation in returned

    for real in LIVE_CITATIONS:
        assert regulations.verify_citation(real), real


def test_a_fabricated_page_still_fails_after_retrieval(monkeypatch, clean_ledger) -> None:
    """Identity, not shape: the right document and an invented page fails.

    A ``documents.get`` by-id check would pass this -- ``pmla_act_2002``
    is genuinely in the index -- which is why document identity alone is
    not the gate.
    """
    regulations = clean_ledger
    monkeypatch.setattr(regulations, "_docai_search", lambda: _fake_cloud_index)

    regulations.search(RECORD_KEEPING_QUERY, 4)
    assert not regulations.verify_citation(FABRICATED_PAGE)
    assert not regulations.verify_citation(
        "https://www.fincen.gov/regulations/statutes/bank-secrecy-act"
    )


@pytest.mark.skipif(
    os.environ.get(LIVE_SEARCH_ENV) != "1",
    reason=(
        f"BILLED: queries the live Vertex AI Search index. Set {LIVE_SEARCH_ENV}=1 "
        "with MULENET_DOCAI_MODE=cloud, GOOGLE_CLOUD_PROJECT, "
        "MULENET_REGULATIONS_DATASTORE, MULENET_REGULATIONS_ENGINE and "
        "application default credentials to run it."
    ),
)
def test_live_index_citations_verify(clean_ledger) -> None:
    """The same property against the real 825-page corpus."""
    regulations = clean_ledger

    results = regulations.search(RECORD_KEEPING_QUERY, 4)
    assert results, "the live index returned nothing"
    for result in results:
        assert not regulations.is_synthetic(result), result["citation"]
        assert regulations.verify_citation(result["citation"]), result["citation"]
    assert not regulations.verify_citation(FABRICATED_PAGE)


def test_placeholder_corpus_never_reports_verified_citations() -> None:
    """The corpus is placeholders, so nothing drafted from it is fileable."""
    from agents import mock_agents

    draft = mock_agents.draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert draft.grounds, "expected the mock path to cite its grounds"
    assert draft.citations_verified is False
    assert mock_agents.SYNTHETIC_CORPUS_WARNING in draft.narrative


def test_report_with_no_grounds_is_not_verified() -> None:
    from agents.mock_agents import ReportDraft

    assert ReportDraft(narrative="x").citations_verified is False


# --------------------------------------- a dropped ground loses its prose
#
# Verbatim from the first live report-agent run on ring 00001. The model
# fabricated a real-looking FinCEN URL for every ground, and the draft
# came back claiming citations_verified=True. The structured side is now
# gated by verify_citation(); these tests cover the prose.

_FABRICATED_PAYLOAD = {
    "narrative": (
        "Subject: Ring ring-mulenet-velocity-rule-v2-00001.\n\n"
        "Grounds of Suspicion:\n"
        "1. The ring exhibits a high median fan-in of 32, which exceeds the "
        "typical threshold.\n"
        "   Source: https://www.fincen.gov/regulations/statutes/bank-secrecy-act\n"
        "2. The detected ring involves accounts across multiple banks, which is "
        "indicative of a complex money laundering scheme designed to evade "
        "detection.\n"
        "   Source: https://www.fincen.gov/regulations/statutes/bank-secrecy-act\n"
    ),
    "grounds": [
        {
            "ground": "The ring exhibits a high median fan-in of 32.",
            "citation": "https://www.fincen.gov/regulations/statutes/bank-secrecy-act",
            "source_uri": "https://www.fincen.gov/regulations/statutes/bank-secrecy-act",
        },
        {
            "ground": "The detected ring involves accounts across multiple banks.",
            "citation": (
                "Synthetic Citation: Section 10.1 - Rapid movement of funds "
                "through multiple accounts."
            ),
            "source_uri": "placeholder://invented",
        },
    ],
}


@pytest.fixture
def fabricating_model(monkeypatch) -> None:
    """Make the report agent return the recorded fabricated draft."""
    import json as _json

    from agents import adk_agents

    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda name, message, config=None, sink=None, citations=None: _json.dumps(
            _FABRICATED_PAYLOAD
        ),
    )


def test_dropped_ground_leaves_no_trace_in_the_narrative(
    fabricating_model: None,
) -> None:
    """The prose is the part a human reads; it must not assert a dropped ground."""
    from agents.adk_agents import draft_report

    draft = draft_report("ring-mulenet-velocity-rule-v2-00001")

    assert draft.grounds == [], "both citations are fabricated"
    assert draft.narrative == "", "prose asserting a dropped ground is withheld"
    assert "fan-in of 32" not in draft.narrative
    assert "money laundering scheme" not in draft.narrative
    assert "fincen.gov" not in draft.narrative
    assert draft.citations_verified is False


def test_withheld_narrative_is_kept_for_review(fabricating_model: None) -> None:
    """Suppressed, not destroyed -- the same trade the investigator makes."""
    from agents.adk_agents import draft_report

    draft = draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert draft.narrative_withheld
    assert "fan-in of 32" in draft.withheld_narrative


def test_withholding_says_why_and_counts_the_survivors(
    fabricating_model: None,
) -> None:
    from agents.adk_agents import draft_report

    draft = draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert any("narrative withheld" in warning for warning in draft.warnings)
    assert any("not in the corpus" in warning for warning in draft.warnings)


def test_a_partially_fabricated_draft_keeps_its_verified_grounds(
    monkeypatch,
) -> None:
    """Three grounds, one fabricated: the survivors stay, the prose goes.

    This is the case the choice actually turns on. The narrative asserts
    all three in one pass, so it cannot be published -- but the two
    verified grounds and their citations are real findings and are still
    returned.
    """
    import json as _json

    from agents import adk_agents, regulations

    real = [r["citation"] for r in regulations.search("pass-through of funds", 2)]
    assert len(real) >= 2

    payload = {
        "narrative": "1. real one. 2. real two. 3. INVENTED THIRD GROUND.",
        "grounds": [
            {"ground": "g1", "citation": real[0], "source_uri": "u1"},
            {"ground": "g2", "citation": real[1], "source_uri": "u2"},
            {"ground": "g3", "citation": "31 CFR 9999 - Not Real", "source_uri": "u3"},
        ],
    }
    monkeypatch.setattr(
        adk_agents, "_run_agent", lambda name, message, config=None, sink=None, citations=None: _json.dumps(payload)
    )

    draft = adk_agents.draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert len(draft.grounds) == 2
    assert draft.narrative == ""
    assert "INVENTED THIRD GROUND" not in draft.narrative
    assert "INVENTED THIRD GROUND" in draft.withheld_narrative
    assert len(draft.citations) == 2


def test_a_fully_verified_draft_keeps_its_narrative(monkeypatch) -> None:
    """Withholding applies only when a ground was dropped."""
    import json as _json

    from agents import adk_agents, regulations

    real = regulations.search("pass-through of funds", 1)[0]["citation"]
    payload = {
        "narrative": "1. A properly cited ground.",
        "grounds": [{"ground": "g1", "citation": real, "source_uri": "u1"}],
    }
    monkeypatch.setattr(
        adk_agents, "_run_agent", lambda name, message, config=None, sink=None, citations=None: _json.dumps(payload)
    )

    draft = adk_agents.draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert draft.narrative == "1. A properly cited ground."
    assert not draft.narrative_withheld
    assert len(draft.grounds) == 1


def test_mock_path_builds_prose_only_from_surviving_grounds() -> None:
    """The mock report renders after filtering, so it cannot have this bug.

    Worth pinning: the defect is specific to a model writing the whole
    narrative in one pass, and the offline path is structurally immune.
    """
    from agents import mock_agents

    draft = mock_agents.draft_report("ring-mulenet-velocity-rule-v2-00001")
    assert draft.grounds
    assert draft.narrative
    assert not draft.narrative_withheld
    for ground in draft.grounds:
        assert ground.citation in draft.narrative


def test_empty_investigation_does_not_verify() -> None:
    """"Nothing failed" is not "everything held".

    A live 400 produced an error payload in place of a narrative, which
    parsed to an empty InvestigationResult and passed verification in
    4 seconds -- publishing a case that recorded nothing was checked.
    """
    from agents.verify import InvestigationResult, verify

    verdict = verify(InvestigationResult(narrative="", claims=[]))
    assert not verdict.passed
    assert "nothing to verify" in verdict.unsupported[0]


def test_no_agent_declares_a_duplicate_tool() -> None:
    """Vertex rejects the whole request on a duplicate declaration.

    It fails as 400 INVALID_ARGUMENT "Duplicate function declaration",
    which is not obviously a tool-wiring problem when read in a stack
    trace.
    """
    from agents.adk_agents import AGENT_TOOLS

    for agent, registered in AGENT_TOOLS.items():
        names = [func.__name__ for func in registered]
        assert len(names) == len(set(names)), f"{agent} declares a tool twice"


def test_investigator_leads_with_the_summary_tool() -> None:
    """Tool order is the cheapest nudge toward the cheap tool."""
    from agents.adk_agents import AGENT_TOOLS

    names = [func.__name__ for func in AGENT_TOOLS["investigator"]]
    assert "get_ring_summary" in names
    assert names.index("get_ring_summary") < names.index("get_ring_subgraph")


def test_summary_is_far_smaller_than_the_subgraph() -> None:
    """The reason the summary exists at all."""
    import json

    from agents.adk_agents import _json_safe

    ring_id = "ring-mulenet-velocity-rule-v2-00002"
    summary = len(json.dumps(_json_safe(tools.get_ring_summary(ring_id))))
    subgraph = len(json.dumps(_json_safe(tools.get_ring_subgraph(ring_id))))
    assert subgraph > summary * 10


def test_summary_counts_every_member() -> None:
    """Nothing in the summary is truncated, unlike the subgraph."""
    summary = tools.get_ring_summary("ring-mulenet-velocity-rule-v2-00002")
    assert (
        summary["alertable_member_count"] + summary["context_member_count"]
        == summary["member_count"]
    )
    assert (
        summary["velocity_member_count"] + summary["identity_link_member_count"]
        == summary["member_count"]
    )
    assert str(summary["alertable_member_count"]) in summary["counting_note"]


def test_money_trace_reports_truncation(store_account_id: str) -> None:
    """Silently capping paths is the same bug as the unbounded subgraph."""
    trace = tools.trace_money_flow(store_account_id, max_hops=3)
    assert len(trace.paths) <= tools.MAX_TRACE_PATHS
    if trace.truncated:
        assert trace.total_paths > len(trace.paths)
        assert "TRUNCATED" in trace.note


def test_money_trace_total_is_marked_as_a_lower_bound(
    store_account_id: str,
) -> None:
    """"At least 500" is honest; "500" would not be.

    The search stops at TRACE_COUNT_CEILING, so beyond it the tool knows
    only that more exist -- and must say so rather than report its own
    ceiling as the answer.
    """
    trace = tools.trace_money_flow(store_account_id, max_hops=3)
    if trace.total_is_lower_bound:
        assert trace.total_paths == tools.TRACE_COUNT_CEILING
        assert "at least" in trace.note


def test_money_trace_still_iterates_and_counts_like_a_list(
    store_account_id: str,
) -> None:
    """Callers wrote `for path in ...` and `len(...)` against a bare list."""
    trace = tools.trace_money_flow(store_account_id, max_hops=2)
    assert len(trace) == len(trace.paths)
    assert [path for path in trace] == trace.paths


def test_untruncated_trace_carries_no_warning(store_account_id: str) -> None:
    trace = tools.trace_money_flow(store_account_id, max_hops=1)
    if not trace.truncated:
        assert trace.note == ""
        assert trace.total_paths == len(trace.paths)
        assert not trace.total_is_lower_bound


@pytest.fixture(scope="module")
def store_account_id() -> str:
    return str(default_store().table("account_evidence")["hashed_account_id"].iloc[0])


def test_subgraph_result_is_json_serialisable() -> None:
    """ADK serialises every tool result into the next request's contents.

    ``RingSubgraph`` is a dataclass, which ``_json_safe`` fell through
    untouched, so ``Transfer.amount`` reached google-genai as a raw
    ``Decimal`` and killed the live request. Serialising it field by
    field is also what puts the truncation flag in front of the model.
    """
    import json

    from agents.adk_agents import _json_safe

    payload = _json_safe(
        tools.get_ring_subgraph("ring-mulenet-velocity-rule-v2-00002")
    )
    encoded = json.dumps(payload)
    assert "truncated" in payload
    assert payload["total_accounts"] > len(payload["accounts"])
    assert len(encoded) > 0


def test_describe_tables_rejects_an_unknown_table() -> None:
    with pytest.raises(ValueError, match="not a queryable table"):
        tools.describe_tables("definitely_not_a_table")


def test_ring_members_is_described_with_its_parent_key() -> None:
    columns = tools.describe_tables("ring_members")["tables"]["ring_members"]
    names = [column["name"] for column in columns]
    assert names[0] == "ring_id"
    assert "membership_basis" in names
    assert "alertable" in names


def test_every_allowlisted_table_actually_executes() -> None:
    """Validation passing is not enough; the table must resolve.

    ``ring_members`` was allowlisted before anything served it, so a
    query against it passed the guard and then failed at execution.
    """
    for table in sorted(ALLOWED_SQL_TABLES):
        assert isinstance(tools.run_sql(f"SELECT * FROM {table} LIMIT 1"), list)


# ============================================================== 5. rounding


def test_scores_are_rounded_at_the_tool_boundary() -> None:
    """q16 returned 0.99999809265136719 and failed a 1e-6 comparison."""
    store = default_store()
    top = str(
        store.table("scores").sort_values("score", ascending=False)[
            "hashed_account_id"
        ].iloc[0]
    )
    _, score, _ = tools.get_account_profile(top)
    assert score is not None
    # One shared check rather than a per-field list: it covers every
    # registered score field, including any the contract grows later.
    assert_score_fields_rounded(score)
    assert abs(score.score - 1.0) < 1e-9


def test_explanation_scores_and_contributions_are_rounded() -> None:
    store = default_store()
    account = str(
        store.table("account_explanations")["hashed_account_id"].iloc[0]
    )
    explanation = tools.get_explanation(account, "account")
    # Covers the nested ExplanationFactor.contribution list too.
    assert_score_fields_rounded(explanation)


def test_rounding_preserves_the_sign_of_a_contribution() -> None:
    """A negative contribution is a mitigating factor; the sign is the point."""
    store = default_store()
    for row in store.table("account_explanations").head(100).itertuples():
        for factor in list(row.top_factors):
            raw = dict(factor)["contribution"]
            if raw < 0:
                assert round(raw, tools.SCORE_DECIMALS) <= 0
                return
    pytest.skip("no negative contribution in the sample")


# ================================== 6. reuse-as-cause is sentence-scoped


@pytest.fixture(scope="module")
def ring_00001() -> object:
    """The real ring the live narrative describes.

    11 members, all alertable, 9 velocity and 2 cross-bank identity
    links -- so the narrative's "11 alertable", "Nine of these" and
    "remaining 2" are all correct, the counting rule is inert, and what
    these tests actually exercise is the reuse-as-cause rule.
    """
    return tools.get_ring_subgraph("ring-mulenet-velocity-rule-v2-00001").ring


@pytest.mark.parametrize("rotation", range(len(LIVE_NARRATIVE_SENTENCES)))
def test_live_narrative_passes_in_every_sentence_order(
    rotation: int, ring_00001: object
) -> None:
    """Ring 00001's real narrative, rotated. It must pass in all orders.

    The old rule scanned 160 characters after "flagged because". This
    narrative passed only because the word "device" happened to sit at
    offset 241. Any reordering that pulled it closer would have rejected
    a correct narrative, and the pass was luck rather than judgement.
    """
    sentences = (
        LIVE_NARRATIVE_SENTENCES[rotation:] + LIVE_NARRATIVE_SENTENCES[:rotation]
    )
    assert not check_prose(" ".join(sentences), [], ring_00001)


def test_live_narrative_passes_as_written(ring_00001: object) -> None:
    assert not check_prose(" ".join(LIVE_NARRATIVE_SENTENCES), [], ring_00001)


def test_adjacent_sentences_do_not_leak_causation(ring_00001: object) -> None:
    """The causal phrase and the reuse noun in *different* sentences is fine."""
    assert not check_prose(
        "This ring was flagged because of its pass-through velocity. "
        "Two accounts share a device fingerprint across banks.",
        [],
        ring_00001,
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "The account was flagged because it shares a device with two others.",
        "This ring was flagged due to reused KYC documents across banks.",
        "The account was detected because of its shared identity documents.",
    ],
)
def test_genuine_reuse_as_cause_is_still_rejected(sentence: str) -> None:
    """Narrowing the scope must not weaken the rule it enforces."""
    problems = check_prose(sentence, [])
    assert problems
    assert "velocity and structure" in problems[0]


# ====================================================== 7. thinking budgets


def test_every_agent_has_a_thinking_budget() -> None:
    from agents.adk_agents import AGENT_TOOLS

    assert set(THINKING_BUDGETS) == set(AGENT_TOOLS)


def test_router_does_not_think() -> None:
    """The router classifies into three labels and was the slowest step."""
    assert THINKING_BUDGETS["router"] == 0


def test_investigator_keeps_a_real_budget() -> None:
    """It must still choose claims that survive verification."""
    assert THINKING_BUDGETS["investigator"] >= 256


# ------------------------------------------------------------ usage metering


class _FakeUsage:
    prompt_token_count = 100
    candidates_token_count = 27
    thoughts_token_count = 538
    total_token_count = 665


class _FakeEvent:
    author = "investigator"
    usage_metadata = _FakeUsage()


def test_usage_is_accumulated_with_thinking_split_out() -> None:
    reset_usage()
    _accumulate_usage(_FakeEvent())
    _accumulate_usage(_FakeEvent())
    bucket = LAST_USAGE["investigator"]
    assert bucket["calls"] == 2
    assert bucket["thinking"] == 1076
    assert bucket["output"] == 54
    reset_usage()
    assert LAST_USAGE == {}


def test_usage_ignores_events_without_metadata() -> None:
    reset_usage()
    _accumulate_usage(object())
    assert LAST_USAGE == {}
