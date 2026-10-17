"""Tests for mock-mode KYC document parsing.

Mock path only. The Document AI path is never exercised here: no
credentials exist in this environment, and ``conftest._force_mock_mode``
strips any that might.
"""

from __future__ import annotations

import pandas as pd
import pytest

from docai.config import DocaiConfig
from docai.conftest import TinyCorpus
from docai.parsing import (
    DOCUMENT_COLUMNS,
    DocumentRecord,
    MockDocumentParser,
    build_parser,
    load_document_records,
    normalize_address,
    parse_kyc_documents,
    write_parsed_documents,
)
from shared.contracts import KycDocument


def test_the_suite_runs_in_mock_mode() -> None:
    """The whole suite must stay offline.

    Mode selection itself is pinned in ``test_config.py``; cloud is
    opt-in and is never inferred from ambient credentials.
    """
    assert DocaiConfig.from_env().mode == "mock"


def test_build_parser_returns_the_mock_backend_in_mock_mode(tiny_corpus: TinyCorpus) -> None:
    parser = build_parser(tiny_corpus.config)
    assert isinstance(parser, MockDocumentParser)
    assert parser.source == "mock"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "  8470 cedar st, unit 36,  testville zz-4273 ",
            "8470 CEDAR ST, UNIT 36, TESTVILLE ZZ-4273",
        ),
        ("1 Alder Rd\n  Testville", "1 ALDER RD TESTVILLE"),
        ("trailing comma,", "TRAILING COMMA"),
    ],
)
def test_normalize_address_uppercases_and_collapses_whitespace(raw: str, expected: str) -> None:
    """Address matching is exact equality, so both sides must normalize alike."""
    assert normalize_address(raw) == expected


def test_parsing_produces_valid_kyc_documents(tiny_corpus: TinyCorpus) -> None:
    """Every parsed row is a valid ``KycDocument`` carrying the intake fields."""
    documents = parse_kyc_documents(tiny_corpus.config)

    assert len(documents) == 10  # 5 accounts x 2 document types
    assert all(isinstance(doc, KycDocument) for doc in documents)

    raw = pd.read_parquet(tiny_corpus.config.documents_path).set_index("doc_id")
    for doc in documents:
        row = raw.loc[doc.doc_id]
        assert doc.hashed_account_id == row["hashed_account_id"]
        assert doc.doc_type == row["doc_type"]
        assert doc.hashed_id_number == row["hashed_id_number"]
        assert doc.normalized_address == normalize_address(row["normalized_address"])
        assert doc.photo_embedding_ref == row["photo_embedding_ref"]
        # Re-validating proves the model accepts what we emit, not just
        # that we constructed it once.
        assert KycDocument.model_validate(doc.model_dump()) == doc


def test_reused_identities_parse_to_the_same_identity_fields(tiny_corpus: TinyCorpus) -> None:
    """Two accounts on one identity share id number, address and image ref."""
    by_account: dict[str, KycDocument] = {
        doc.hashed_account_id: doc
        for doc in parse_kyc_documents(tiny_corpus.config)
        if doc.doc_type == "id_card"
    }
    left, right = tiny_corpus.cross_bank_accounts

    assert by_account[left].hashed_id_number == by_account[right].hashed_id_number
    assert by_account[left].normalized_address == by_account[right].normalized_address
    assert by_account[left].photo_embedding_ref == by_account[right].photo_embedding_ref
    assert by_account[left].hashed_account_id != by_account[right].hashed_account_id


def test_parser_drops_an_image_ref_whose_file_is_absent(tiny_corpus: TinyCorpus) -> None:
    """A dangling ref becomes ``None`` rather than a path nothing can read."""
    record = DocumentRecord(
        doc_id="doc_missing",
        hashed_account_id="acct_missing",
        doc_type="id_card",
        hashed_id_number="idnum_missing",
        normalized_address="1 nowhere st",
        photo_embedding_ref="kyc_images/does_not_exist.png",
    )
    parsed = MockDocumentParser().parse(record, tiny_corpus.config.images_dir)

    assert parsed.photo_embedding_ref is None
    assert parsed.normalized_address == "1 NOWHERE ST"


def test_document_record_coerces_a_null_image_ref(tiny_corpus: TinyCorpus) -> None:
    """Most seed-42 rows have no rendered image; NaN must become ``None``."""
    record = DocumentRecord.from_row(
        {
            "doc_id": "d",
            "hashed_account_id": "a",
            "doc_type": "id_card",
            "hashed_id_number": "i",
            "normalized_address": "addr",
            "photo_embedding_ref": float("nan"),
        }
    )
    assert record.photo_embedding_ref is None


def test_limit_parses_only_the_requested_rows(tiny_corpus: TinyCorpus) -> None:
    assert len(parse_kyc_documents(tiny_corpus.config, limit=3)) == 3


def test_parsed_documents_round_trip_through_parquet(tiny_corpus: TinyCorpus) -> None:
    documents = parse_kyc_documents(tiny_corpus.config)
    path = write_parsed_documents(documents, tiny_corpus.config.parsed_documents_path)

    frame = pd.read_parquet(path)
    assert len(frame) == len(documents)
    assert set(KycDocument.model_fields) <= set(frame.columns)
    restored = [KycDocument.model_validate(row) for row in frame.to_dict("records")]
    assert restored == documents


def test_load_document_records_reads_the_contract_columns(tiny_corpus: TinyCorpus) -> None:
    """Guards against the simulator renaming a column out from under us."""
    frame = pd.read_parquet(tiny_corpus.config.documents_path)
    assert set(DOCUMENT_COLUMNS) <= set(frame.columns)
    assert len(list(load_document_records(tiny_corpus.config))) == len(frame)


def test_seed42_corpus_parses_into_valid_models(seed42_documents: list[KycDocument]) -> None:
    """The real 60,000-row corpus parses cleanly end to end."""
    assert len(seed42_documents) == 60_000
    assert all(isinstance(doc, KycDocument) for doc in seed42_documents)
    assert len({doc.doc_id for doc in seed42_documents}) == 60_000
    assert {doc.doc_type for doc in seed42_documents} == {"id_card", "address_proof"}
    assert len({doc.hashed_account_id for doc in seed42_documents}) == 30_000
    assert all(doc.hashed_id_number and doc.normalized_address for doc in seed42_documents)
    # Addresses are already normalized, so parsing is idempotent on them.
    assert all(
        doc.normalized_address == normalize_address(doc.normalized_address)
        for doc in seed42_documents
    )
