"""Tests for the bounded cloud evaluation, all offline.

Nothing here calls Document AI or Vertex. What is tested is the part
that decides *what* is measured and *how the numbers are computed* --
sample selection, ground truth, and the roll-up. That is where a
mistake would quietly produce a flattering result, which is worse than
no result at all.
"""

from __future__ import annotations

import pytest

from docai.config import DocaiConfig
from docai.eval_cloud import (
    DHASH_COLLIDING_REFS,
    FORM_PARSER_USD_PER_PAGE,
    MULTIMODAL_EMBEDDING_USD_PER_IMAGE,
    ExtractionResult,
    expected_fields,
    select_sample,
    summarize,
)
from docai.parsing import DocumentRecord


def _result(**overrides: object) -> ExtractionResult:
    """An all-correct result, so a test only states what it changes."""
    defaults = {
        "ref": "kyc_images/a_id_card.png",
        "doc_type": "id_card",
        "latency_s": 2.0,
        "name_ok": True,
        "id_number_ok": True,
        "address_ok": True,
        "extracted_name": "SYNTHETIC HOLDER AAAAAA",
        "extracted_id_number": "SYN-0000-0000-0000",
        "extracted_address": "1 TEST ST",
        "expected_name": "SYNTHETIC HOLDER AAAAAA",
        "expected_id_number": "SYN-0000-0000-0000",
        "expected_address": "1 TEST ST",
    }
    return ExtractionResult(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_expected_fields_come_from_the_renderer_not_from_this_module() -> None:
    """Ground truth is what ``simulator.kyc`` actually draws."""
    record = DocumentRecord(
        doc_id="d1",
        hashed_account_id="a1",
        doc_type="id_card",
        hashed_id_number="753ddb24607c8baa0d8fd6c8b4a27dc9",
        normalized_address="  3572 birch rd,  unit 33, testville zz-1009 ,",
        photo_embedding_ref="kyc_images/0d908a8e20a031fc_id_card.png",
    )
    name, id_number, address = expected_fields(record)

    assert name == "SYNTHETIC HOLDER 0D908A"
    assert id_number == "SYN-753D-DB24-607C"
    assert address == "3572 BIRCH RD, UNIT 33, TESTVILLE ZZ-1009"


def test_an_address_proof_is_expected_to_carry_no_document_number() -> None:
    """Only the ID card renders one, so extracting one would be an error."""
    record = DocumentRecord(
        doc_id="d2",
        hashed_account_id="a1",
        doc_type="address_proof",
        hashed_id_number="753ddb24607c8baa",
        normalized_address="1 TEST ST",
        photo_embedding_ref="kyc_images/0d908a8e20a031fc_address_proof.png",
    )
    assert expected_fields(record)[1] is None


def test_the_sample_always_contains_the_colliding_pair(
    seed42_config: DocaiConfig,
) -> None:
    """A sample without them could not answer the question being asked."""
    sample = select_sample(seed42_config, 50)

    assert len(sample) == 50
    refs = [record.photo_embedding_ref for record in sample]
    assert set(DHASH_COLLIDING_REFS) <= set(refs)
    assert len(set(refs)) == 50, "each image must appear once"


def test_the_sample_is_deterministic(seed42_config: DocaiConfig) -> None:
    first = [record.doc_id for record in select_sample(seed42_config, 12)]
    assert first == [record.doc_id for record in select_sample(seed42_config, 12)]


def test_a_smaller_limit_keeps_the_colliding_pair(seed42_config: DocaiConfig) -> None:
    refs = [record.photo_embedding_ref for record in select_sample(seed42_config, 2)]
    assert set(refs) == set(DHASH_COLLIDING_REFS)


def test_a_limit_below_the_mandatory_refs_is_rejected(seed42_config: DocaiConfig) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        select_sample(seed42_config, 1)


def test_summarize_reports_per_field_accuracy() -> None:
    extraction = [
        _result(),
        _result(name_ok=False),
        _result(doc_type="address_proof", address_ok=False),
        _result(doc_type="address_proof", id_number_ok=False),
    ]
    summary = summarize(extraction, reuse={})["extraction"]

    assert summary["name_accuracy"] == 0.75
    assert summary["address_accuracy"] == 0.75
    assert summary["id_number_accuracy"] == 0.75
    # Two ID cards, one of which failed only on the name.
    assert summary["id_number_accuracy_id_cards_only"] == 1.0
    assert summary["all_fields_correct"] == 0.25


def test_summarize_prices_the_run_from_the_billed_units() -> None:
    """One Document AI page per document, one embedding plus the repeat."""
    extraction = [_result() for _ in range(10)]
    cost = summarize(extraction, reuse={})["cost_usd"]

    assert cost["document_ai_pages"] == 10
    assert cost["document_ai"] == pytest.approx(10 * FORM_PARSER_USD_PER_PAGE)
    assert cost["embedding_calls"] == 11
    assert cost["embeddings"] == pytest.approx(11 * MULTIMODAL_EMBEDDING_USD_PER_IMAGE)
    assert cost["total"] == pytest.approx(cost["document_ai"] + cost["embeddings"])
    assert cost["per_document"] == pytest.approx(cost["total"] / 10)


def test_summarize_reports_the_latency_spread_not_just_a_mean() -> None:
    latencies = summarize([_result(latency_s=s) for s in (1.0, 2.0, 9.0)], reuse={})[
        "latency_s"
    ]
    assert latencies["min"] == 1.0
    assert latencies["max"] == 9.0
    assert latencies["median"] == 2.0
    assert latencies["mean"] == pytest.approx(4.0)
    assert latencies["total"] == pytest.approx(12.0)
