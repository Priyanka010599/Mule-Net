"""Tests for the ``search_regulations`` tool in mock mode.

Two things are checked here, and the second is the important one:

1. The result shape matches what ``shared/tool_interfaces.py`` documents,
   so the report agent can consume it.
2. Every bundled passage is *unmistakably* labelled as a placeholder.
   The report agent cites whatever this tool returns. An unlabelled
   placeholder would be rendered as a genuine regulation citation inside
   a compliance narrative, which is exactly the failure this corpus must
   not be capable of.

The Vertex AI Search path is not exercised: no credentials exist here,
and ``google-cloud-discoveryengine`` is not installed.
"""

from __future__ import annotations

import pytest

from docai.config import DocaiConfig
from docai.regulations import (
    PLACEHOLDER_MARKER,
    CorpusDocument,
    MockRegulationSearch,
    RegulationPassage,
    build_search,
    load_corpus_manifest,
    load_placeholder_corpus,
    passage_from_search_result,
    search_regulations,
)

#: Keys ``shared.tool_interfaces.search_regulations`` promises callers.
DOCUMENTED_KEYS: frozenset[str] = frozenset({"citation", "excerpt", "source_uri"})

#: Every bundled citation must carry one of these, in upper case.
_LABEL_MARKERS: tuple[str, ...] = ("PLACEHOLDER", "SYNTHETIC", "NOT A REAL")


def test_build_search_returns_the_offline_backend_in_mock_mode() -> None:
    backend = build_search(DocaiConfig.from_env())
    assert isinstance(backend, MockRegulationSearch)
    assert backend.backend == "placeholder"


def test_search_returns_the_documented_result_shape() -> None:
    """Keys and value types match the shared tool interface exactly."""
    results = search_regulations("accounts sharing an identity document across banks")

    assert results, "expected at least one passage"
    for result in results:
        assert DOCUMENTED_KEYS <= set(result)
        assert all(isinstance(value, str) for value in result.values())
        assert result["citation"].strip()
        assert result["excerpt"].strip()
        assert result["source_uri"].strip()


def test_search_also_breaks_out_document_name_and_section() -> None:
    """The docai brief asks for the document name and section separately."""
    result = search_regulations("suspicious transaction report", top_k=1)[0]
    assert result["doc_name"].strip()
    assert result["section"].strip()
    # The citation stays self-contained, so a caller that only reads the
    # documented keys still renders a complete reference.
    assert result["section"].split()[0] in result["citation"]


def test_every_bundled_passage_is_unmistakably_labelled_a_placeholder() -> None:
    """No result may be presentable as a genuine regulation citation."""
    results = search_regulations("customer due diligence", top_k=20)
    assert results

    for result in results:
        citation = result["citation"].upper()
        assert any(marker in citation for marker in _LABEL_MARKERS), (
            f"citation is not marked as a placeholder: {result['citation']!r}"
        )
        assert PLACEHOLDER_MARKER in result["doc_name"].upper()
        assert PLACEHOLDER_MARKER in result["excerpt"].upper()
        # A placeholder must not masquerade as a fetchable public source.
        assert result["source_uri"].startswith("placeholder://")


def test_the_whole_bundled_corpus_is_labelled_not_just_search_hits() -> None:
    """Guards the passages a query happens not to rank."""
    passages = load_placeholder_corpus(DocaiConfig.from_env().regulations_dir)
    assert len(passages) >= 5

    for passage in passages:
        assert PLACEHOLDER_MARKER in passage.citation.upper()
        assert PLACEHOLDER_MARKER in passage.doc_name.upper()
        assert PLACEHOLDER_MARKER in passage.text.upper()


def test_top_k_bounds_the_result_count() -> None:
    assert len(search_regulations("reuse of identity documents", top_k=2)) <= 2
    assert len(search_regulations("reuse of identity documents", top_k=1)) == 1


def test_results_are_ranked_and_deterministic() -> None:
    """The same query twice gives the same order; no randomness."""
    query = "identity document reused across two different institutions"
    assert search_regulations(query, top_k=5) == search_regulations(query, top_k=5)


def test_ranking_surfaces_the_cross_institution_passage() -> None:
    """A cross-bank reuse question should reach the cross-institution section."""
    results = search_regulations("reuse across different institutions", top_k=3)
    sections = [result["section"] for result in results]
    assert any("P-3" in section for section in sections), sections


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        search_regulations(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        search_regulations("any query", top_k=top_k)


def test_a_missing_corpus_raises_runtime_error(tmp_path) -> None:
    """``RuntimeError`` is what the shared interface documents for this."""
    with pytest.raises(RuntimeError, match="not found"):
        load_placeholder_corpus(tmp_path)


def test_an_empty_corpus_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        MockRegulationSearch([])


def test_a_query_matching_nothing_returns_no_results() -> None:
    """Better to return nothing than to hand the report agent a bad citation."""
    assert search_regulations("zzzqqq aardvark wombat quixotic", top_k=5) == []


def test_passage_renders_all_five_result_keys() -> None:
    passage = RegulationPassage(
        doc_name="PLACEHOLDER DOC",
        section="P-9 Example",
        citation="PLACEHOLDER-CITE s.P-9",
        text="PLACEHOLDER TEXT.",
        source_uri="placeholder://example",
    )
    assert passage.as_result() == {
        "citation": "PLACEHOLDER-CITE s.P-9",
        "excerpt": "PLACEHOLDER TEXT.",
        "source_uri": "placeholder://example",
        "doc_name": "PLACEHOLDER DOC",
        "section": "P-9 Example",
    }


# --- The official corpus and its provenance guard -------------------------
#
# The cloud path is what makes citations real, so these tests are about
# provenance, not ranking. A live report agent once emitted five citations
# that all pointed at a fincen.gov URL the tool had never returned. The
# guard against that is mechanical: a passage is built only from a document
# the API returned *and* that appears in the corpus manifest, so a citation
# string cannot exist unless a real retrieval produced it.

#: One live Discovery Engine result, copied verbatim from a real response
#: of the ``mulenet-aml-regulations-app`` engine on 2026-09-27, trimmed to
#: the fields the mapper reads.
LIVE_RESULT: dict[str, object] = {
    "id": "pmla_act_2002",
    "document": {
        "id": "pmla_act_2002",
        "structData": {
            "title": "The Prevention of Money-Laundering Act, 2002",
            "short_title": "PMLA, 2002",
            "source_url": "https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf",
        },
        "derivedStructData": {
            "title": "The Prevention of Money-Laundering Act, 2002",
            "link": "gs://mulenet-mulenet-data/regulations/pmla_act_2002.pdf",
            "extractive_segments": [
                {
                    "content": "[12. Reporting entity to maintain records.",
                    "relevanceScore": 0.74,
                    "pageNumber": "14",
                }
            ],
        },
    },
}


@pytest.fixture(scope="module")
def manifest() -> dict[str, CorpusDocument]:
    documents = load_corpus_manifest(DocaiConfig.from_env().regulations_dir)
    return {document.doc_id: document for document in documents}


def test_the_manifest_lists_the_four_official_documents(
    manifest: dict[str, CorpusDocument],
) -> None:
    assert set(manifest) == {
        "rbi_kyc_master_direction",
        "pmla_act_2002",
        "pml_rules_2005",
        "fiu_ind_reporting_format",
    }
    for document in manifest.values():
        assert document.filename.endswith(".pdf")
        assert document.source_url.startswith("https://")
        assert document.size_bytes > 0
        # Nothing in the real corpus may be labelled a placeholder.
        assert PLACEHOLDER_MARKER not in document.title.upper()


def test_a_live_result_maps_to_a_passage_with_title_section_and_source(
    manifest: dict[str, CorpusDocument],
) -> None:
    passage = passage_from_search_result(LIVE_RESULT, manifest)

    assert passage is not None
    assert passage.doc_name == "The Prevention of Money-Laundering Act, 2002"
    assert passage.section == "p. 14"
    assert passage.citation == "PMLA, 2002, p. 14"
    assert passage.text.startswith("[12. Reporting entity to maintain records.")
    # The citable source is the issuing authority's URL, not the gs:// object.
    assert passage.source_uri == (
        "https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf"
    )


def test_a_result_naming_a_document_outside_the_manifest_is_dropped(
    manifest: dict[str, CorpusDocument],
) -> None:
    """This is the fabricated-citation guard, and it is not cosmetic."""
    foreign = {"document": {"id": "bsa_aml_examination_manual", "structData": {}}}
    assert passage_from_search_result(foreign, manifest) is None


def test_a_result_with_no_extractive_text_is_dropped(
    manifest: dict[str, CorpusDocument],
) -> None:
    """No quotable text means nothing to cite, so nothing is returned."""
    bare = {"document": {"id": "pmla_act_2002", "derivedStructData": {}}}
    assert passage_from_search_result(bare, manifest) is None

    empty = {
        "document": {
            "id": "pmla_act_2002",
            "derivedStructData": {"extractive_segments": [{"content": "  "}]},
        }
    }
    assert passage_from_search_result(empty, manifest) is None


def test_no_manifest_document_can_produce_the_fabricated_citation(
    manifest: dict[str, CorpusDocument],
) -> None:
    """The exact shape of the live failure: a BSA citation on a FinCEN URL.

    Every field of a passage comes from the API response or the manifest
    entry it resolved to, so a citation naming a document that is not in
    the corpus is unreachable by construction.
    """
    fabricated = "BSA/AML Examination Manual - Funnel and pass-through accounts"
    for doc_id in manifest:
        result = {
            "document": {
                "id": doc_id,
                "derivedStructData": {
                    "extractive_segments": [{"content": "text", "pageNumber": "1"}]
                },
            }
        }
        passage = passage_from_search_result(result, manifest)
        assert passage is not None
        assert passage.citation != fabricated
        assert "fincen.gov" not in passage.source_uri


def test_a_missing_manifest_raises_runtime_error(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        load_corpus_manifest(tmp_path)


def test_cloud_search_refuses_to_start_without_the_search_app(monkeypatch) -> None:
    """A data store serving config cannot return extractive segments.

    Discovery Engine answers such a request with HTTP 400 ("Cannot use
    enterprise edition features ... in a standard edition search engine"),
    because Enterprise edition is enabled only at the app level. The
    engine id is therefore required, not optional.
    """
    monkeypatch.setenv("MULENET_DOCAI_MODE", "cloud")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mulenet")
    monkeypatch.setenv("MULENET_REGULATIONS_DATASTORE", "mulenet-aml-regulations")
    monkeypatch.delenv("MULENET_REGULATIONS_ENGINE", raising=False)

    with pytest.raises(ValueError, match="MULENET_REGULATIONS_ENGINE"):
        build_search(DocaiConfig.from_env())
