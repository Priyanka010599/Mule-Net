"""Tests for cross-bank KYC document reuse detection.

The seed-42 tests assert *both* sides of the ambiguity on purpose:

* cross-bank reuse recovers the reuse rings, and
* same-bank reuse is dominated by legitimate customers.

Both matter. If only the first were asserted, the suite would quietly
endorse treating document reuse as a detector, which is precisely the
thing this project's design says it is not. The measured rates below are
the simulator's by construction; they are pinned here so a regression in
either direction is visible.

All offline: reuse runs on the mock parser and the difference-hash
embedder. No Google Cloud path is exercised.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from docai.config import DocaiConfig
from docai.conftest import TinyCorpus
from docai.reuse import (
    ACCOUNT_EVIDENCE_COLUMNS,
    REUSE_PAIR_COLUMNS,
    ReusePair,
    account_pair_summary,
    detect_reuse,
    evidence_from_pairs,
    exact_reuse_pairs,
    load_submitted_at_map,
    write_account_evidence,
    write_reuse_pairs,
)
from shared.contracts import AccountEvidence, EvidenceScope, KycDocument

# --------------------------------------------------------------------
# Fixture corpus: reuse that is known by construction
# --------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_pairs(tiny_corpus: TinyCorpus) -> list[ReusePair]:
    """All reuse pairs in the five-account fixture corpus."""
    from docai.parsing import parse_kyc_documents

    documents = parse_kyc_documents(tiny_corpus.config)
    return detect_reuse(documents, tiny_corpus.banks, tiny_corpus.config)


def _account_pairs(pairs: list[ReusePair]) -> set[tuple[str, str]]:
    return {(p.left_hashed_account_id, p.right_hashed_account_id) for p in pairs}


def test_cross_bank_reuse_is_found_and_flagged_cross_bank(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    left, right = tiny_corpus.cross_bank_accounts
    found = [
        p
        for p in tiny_pairs
        if {p.left_hashed_account_id, p.right_hashed_account_id} == {left, right}
    ]

    assert found
    assert all(p.cross_bank for p in found)
    assert {p.bank_pair for p in found} == {"BANK_A|BANK_B"}
    assert {p.left_bank_id for p in found} == {"BANK_A"}
    assert {p.right_bank_id for p in found} == {"BANK_B"}


def test_same_bank_reuse_is_found_and_flagged_same_bank(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """Same-bank reuse is still reported -- as evidence, not as a hit."""
    left, right = tiny_corpus.same_bank_accounts
    found = [
        p
        for p in tiny_pairs
        if {p.left_hashed_account_id, p.right_hashed_account_id} == {left, right}
    ]

    assert found
    assert not any(p.cross_bank for p in found)
    assert {p.bank_pair for p in found} == {"BANK_A|BANK_A"}


def test_all_three_match_types_fire(tiny_pairs: list[ReusePair]) -> None:
    assert {p.match_type for p in tiny_pairs} == {
        "hashed_id_number",
        "normalized_address",
        "photo_embedding_ref",
    }


def test_photo_matches_are_exact_for_a_shared_file(tiny_pairs: list[ReusePair]) -> None:
    photo = [p for p in tiny_pairs if p.match_type == "photo_embedding_ref"]
    assert photo
    assert all(p.similarity == pytest.approx(1.0) for p in photo)
    assert {p.backend for p in photo} == {"dhash"}


def test_an_account_reusing_nothing_is_never_paired(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    accounts = {p.left_hashed_account_id for p in tiny_pairs} | {
        p.right_hashed_account_id for p in tiny_pairs
    }
    assert tiny_corpus.unique_account not in accounts


def test_an_account_is_never_paired_with_itself(tiny_pairs: list[ReusePair]) -> None:
    """Each account holds two documents on one identity; that is not reuse."""
    assert all(p.left_hashed_account_id != p.right_hashed_account_id for p in tiny_pairs)


def test_account_ids_are_sorted_and_pair_ids_unique(tiny_pairs: list[ReusePair]) -> None:
    assert all(p.left_hashed_account_id < p.right_hashed_account_id for p in tiny_pairs)
    assert len({p.pair_id for p in tiny_pairs}) == len(tiny_pairs)


def test_exactly_two_account_pairs_are_summarised(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    summary = account_pair_summary(tiny_pairs)
    assert len(summary) == 2
    assert int(summary["cross_bank"].sum()) == 1
    assert _account_pairs(tiny_pairs) == {
        tuple(sorted(tiny_corpus.cross_bank_accounts)),
        tuple(sorted(tiny_corpus.same_bank_accounts)),
    }


def test_empty_input_summarises_to_an_empty_frame() -> None:
    assert account_pair_summary([]).empty


def test_reuse_pairs_round_trip_through_parquet(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    path = write_reuse_pairs(tiny_pairs, tiny_corpus.config.reuse_pairs_path)
    frame = pd.read_parquet(path)

    assert list(frame.columns) == list(REUSE_PAIR_COLUMNS)
    assert len(frame) == len(tiny_pairs)


def test_a_document_whose_account_has_no_submission_is_an_error(
    tiny_corpus: TinyCorpus,
) -> None:
    """Silently defaulting the bank would corrupt the cross_bank flag."""
    orphan = [
        KycDocument(
            doc_id=f"orphan_{index}",
            hashed_account_id=f"unknown_{index}",
            doc_type="id_card",
            hashed_id_number="shared_idnum",
            normalized_address="1 ORPHAN ST",
            photo_embedding_ref=None,
        )
        for index in range(2)
    ]
    with pytest.raises(KeyError, match="no KYC submission on file"):
        detect_reuse(orphan, tiny_corpus.banks, tiny_corpus.config)


# --------------------------------------------------------------------
# AccountEvidence reshaping (shared/contracts.py)
# --------------------------------------------------------------------


def test_evidence_validates_against_the_shared_contract(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    evidence = evidence_from_pairs(
        tiny_pairs, load_submitted_at_map(tiny_corpus.config)
    )

    assert evidence
    for row in evidence:
        assert isinstance(row, AccountEvidence)
        assert AccountEvidence.model_validate(row.model_dump()) == row
        assert row.source_table == "kyc_documents"
        assert row.detector_version is None
        assert row.counterparty_hashed_account_ids
        assert row.counterparty_bank_ids
        assert row.hashed_account_id not in row.counterparty_hashed_account_ids


def test_evidence_scope_follows_the_bank_comparison(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """``scope`` is exactly "do the two accounts' banks differ"."""
    evidence = evidence_from_pairs(tiny_pairs, load_submitted_at_map(tiny_corpus.config))
    by_account: dict[str, list[AccountEvidence]] = {}
    for row in evidence:
        by_account.setdefault(row.hashed_account_id, []).append(row)

    for account in tiny_corpus.cross_bank_accounts:
        rows = by_account[account]
        assert {row.scope for row in rows} == {EvidenceScope.CROSS_BANK}
        assert all(row.evidence_type.startswith("cross_bank_") for row in rows)

    for account in tiny_corpus.same_bank_accounts:
        rows = by_account[account]
        assert {row.scope for row in rows} == {EvidenceScope.SINGLE_BANK}
        assert all(row.evidence_type.startswith("same_bank_") for row in rows)

    assert tiny_corpus.unique_account not in by_account


def test_evidence_types_cover_the_documented_taxonomy(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """Keys line up with the ones ml/ writes, plus docai's address key."""
    evidence = evidence_from_pairs(tiny_pairs, load_submitted_at_map(tiny_corpus.config))
    assert {row.evidence_type for row in evidence} == {
        "cross_bank_reused_id_number",
        "cross_bank_reused_address",
        "cross_bank_reused_photo",
        "same_bank_reused_id_number",
        "same_bank_reused_address",
        "same_bank_reused_photo",
    }


def test_evidence_observed_at_is_the_later_of_the_two_submissions(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """A bank cannot act on a pair before both sides exist."""
    submitted = load_submitted_at_map(tiny_corpus.config)
    evidence = evidence_from_pairs(tiny_pairs, submitted)

    left, right = tiny_corpus.cross_bank_accounts
    expected = max(submitted[left], submitted[right])
    rows = [row for row in evidence if row.hashed_account_id in (left, right)]
    assert rows
    assert all(row.observed_at == expected for row in rows)


def test_evidence_detail_names_the_counterparty_banks(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """An STR narrative needs the bank named, not just "shares a document"."""
    evidence = evidence_from_pairs(tiny_pairs, load_submitted_at_map(tiny_corpus.config))
    left, _ = tiny_corpus.cross_bank_accounts
    rows = [row for row in evidence if row.hashed_account_id == left]

    assert rows
    for row in rows:
        assert "BANK_B" in row.detail
        for bank in row.counterparty_bank_ids:
            assert bank in row.detail


def test_evidence_requires_a_submission_time(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """``observed_at`` is required by the contract and must not be invented."""
    with pytest.raises(KeyError, match="no KYC submission time"):
        evidence_from_pairs(tiny_pairs, {})


def test_evidence_round_trips_through_parquet(
    tiny_corpus: TinyCorpus, tiny_pairs: list[ReusePair]
) -> None:
    """The file re-validates against the shared model, like ml/'s does."""
    evidence = evidence_from_pairs(tiny_pairs, load_submitted_at_map(tiny_corpus.config))
    path = write_account_evidence(evidence, tiny_corpus.config.account_evidence_path)

    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(ACCOUNT_EVIDENCE_COLUMNS)
    assert len(frame) == len(evidence)

    restored = [
        AccountEvidence.model_validate(
            {
                **row,
                "counterparty_hashed_account_ids": list(
                    row["counterparty_hashed_account_ids"]
                ),
                "counterparty_bank_ids": list(row["counterparty_bank_ids"]),
            }
        )
        for row in frame.to_dict("records")
    ]
    assert restored == evidence


# --------------------------------------------------------------------
# Real seed-42 corpus
# --------------------------------------------------------------------


@pytest.fixture(scope="module")
def seed42_labelled(
    seed42_documents: list[KycDocument],
    seed42_banks: dict[str, str],
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Ground truth with per-account reuse flags from the real corpus.

    Uses the exact id-number match only: it runs in a fraction of a
    second over all 60,000 rows, where the full detector takes ~100s
    because of the image pass.

    Measured on the full run, the id-number and address matches flag an
    identical set of 2,177 accounts (both derive from the identity), and
    the photo match flags a subset of 860 -- plus exactly one extra
    account, from a difference-hash collision between two unrelated
    address proofs. So the id-number match is the right basis for the
    cohort rates below, and it does not inherit the mock embedder's
    false positives.
    """
    pairs = exact_reuse_pairs(
        seed42_documents, seed42_banks, "hashed_id_number", datetime.now(UTC)
    )
    summary = account_pair_summary(pairs)
    cross = summary[summary["cross_bank"]]
    same = summary[~summary["cross_bank"]]

    flagged = set(summary["left_hashed_account_id"]) | set(summary["right_hashed_account_id"])
    cross_flagged = set(cross["left_hashed_account_id"]) | set(cross["right_hashed_account_id"])
    same_flagged = set(same["left_hashed_account_id"]) | set(same["right_hashed_account_id"])

    labelled = ground_truth.copy()
    labelled["reuse"] = labelled["hashed_account_id"].isin(flagged)
    labelled["cross_bank_reuse"] = labelled["hashed_account_id"].isin(cross_flagged)
    labelled["same_bank_reuse"] = labelled["hashed_account_id"].isin(same_flagged)
    labelled.attrs["summary"] = summary
    return labelled


def _rate(labelled: pd.DataFrame, cohort: str, column: str) -> float:
    rows = labelled[labelled["cohort"] == cohort]
    return 100.0 * rows[column].mean()


def test_seed42_cross_bank_reuse_recovers_every_multi_bank_reuse_ring(
    seed42_labelled: pd.DataFrame,
) -> None:
    """The signal side: every cross-bank reuse ring is caught.

    A ring that recycles one identity across two or three banks must show
    up as cross-bank reuse on at least two of its members, or the
    cross-bank evidence is worthless.
    """
    rings = seed42_labelled[
        (seed42_labelled["ring_bank_count"] >= 2)
        & (seed42_labelled["pattern_type"].isin(["reused_document", "mixed"]))
    ]
    assert not rings.empty

    per_ring = rings.groupby("ring_id")["cross_bank_reuse"].sum()
    assert len(per_ring) >= 40
    assert (per_ring >= 2).all(), per_ring[per_ring < 2]


def test_seed42_cross_bank_reuse_is_confined_to_mules_and_dual_bank_customers(
    seed42_labelled: pd.DataFrame,
) -> None:
    """Cross-bank reuse is sharp -- but not clean.

    Only two cohorts ever reuse an identity across banks: mules, and the
    400 dual-bank customers who do it legitimately. Those 400 are why
    cross-bank reuse still cannot be treated as a detection.
    """
    flagged = seed42_labelled[seed42_labelled["cross_bank_reuse"]]
    assert set(flagged["cohort"]) <= {"mule", "hard_negative_dual_bank"}

    assert _rate(seed42_labelled, "hard_negative_dual_bank", "cross_bank_reuse") == 100.0
    assert int(
        seed42_labelled[seed42_labelled["cohort"] == "hard_negative_dual_bank"][
            "cross_bank_reuse"
        ].sum()
    ) == 400
    assert _rate(seed42_labelled, "normal", "cross_bank_reuse") == 0.0
    assert _rate(seed42_labelled, "hard_negative_family", "cross_bank_reuse") == 0.0
    assert _rate(seed42_labelled, "hard_negative_business", "cross_bank_reuse") == 0.0

    # Even at its sharpest, well under half of cross-bank reuse is a mule.
    assert 0.35 < flagged["is_mule"].mean() < 0.45


def test_seed42_same_bank_reuse_is_dominated_by_legitimate_customers(
    seed42_labelled: pd.DataFrame,
) -> None:
    """The ambiguity side: legitimate same-bank reuse is the common case.

    Rates are the simulator's by construction. Mules reuse documents
    within a bank at roughly the same rate businesses do, so a single
    bank acting on same-bank reuse alone would be swamped.
    """
    assert _rate(seed42_labelled, "hard_negative_business", "same_bank_reuse") == pytest.approx(
        28.0, abs=1.0
    )
    assert _rate(seed42_labelled, "hard_negative_family", "same_bank_reuse") == pytest.approx(
        8.01, abs=0.5
    )
    assert _rate(seed42_labelled, "normal", "same_bank_reuse") == pytest.approx(4.9, abs=0.3)
    assert _rate(seed42_labelled, "mule", "same_bank_reuse") == pytest.approx(31.5, abs=1.5)

    same_bank = seed42_labelled[seed42_labelled["same_bank_reuse"]]
    legitimate = int((~same_bank["is_mule"]).sum())
    mules = int(same_bank["is_mule"].sum())
    assert legitimate > 1_000
    assert legitimate > 4 * mules, (legitimate, mules)


def test_seed42_reuse_alone_is_a_weak_signal(seed42_labelled: pd.DataFrame) -> None:
    """Flagging on reuse of any kind is mostly wrong, and scope helps."""
    flagged = seed42_labelled[seed42_labelled["reuse"]]
    assert flagged["is_mule"].mean() < 0.20

    cross_precision = seed42_labelled[seed42_labelled["cross_bank_reuse"]]["is_mule"].mean()
    same_precision = seed42_labelled[seed42_labelled["same_bank_reuse"]]["is_mule"].mean()
    assert cross_precision > same_precision
    assert cross_precision > 2 * same_precision


def test_seed42_every_bank_pair_appears(seed42_labelled: pd.DataFrame) -> None:
    """Three same-bank and three cross-bank combinations, all populated."""
    summary = seed42_labelled.attrs["summary"]
    bank_pairs = set(summary["bank_pair"])
    assert bank_pairs == {
        "BANK_A|BANK_A",
        "BANK_B|BANK_B",
        "BANK_C|BANK_C",
        "BANK_A|BANK_B",
        "BANK_A|BANK_C",
        "BANK_B|BANK_C",
    }
    cross = summary[summary["cross_bank"]]
    assert set(cross["bank_pair"]) == {"BANK_A|BANK_B", "BANK_A|BANK_C", "BANK_B|BANK_C"}
    assert (cross.groupby("bank_pair").size() > 100).all()


def test_seed42_evidence_validates_against_the_shared_contract(
    seed42_config: DocaiConfig,
    seed42_documents: list[KycDocument],
    seed42_banks: dict[str, str],
) -> None:
    """Real-corpus evidence rows are accepted by the shared model.

    Runs on a slice of the corpus: the reshaping is per pair, so a slice
    exercises the same code path as the whole table.
    """
    subset = seed42_documents[:4_000]
    pairs = exact_reuse_pairs(
        subset, seed42_banks, "hashed_id_number", datetime.now(UTC)
    )
    assert pairs

    evidence = evidence_from_pairs(pairs, load_submitted_at_map(seed42_config))
    assert evidence
    for row in evidence:
        assert AccountEvidence.model_validate(row.model_dump()) == row
        assert row.source_table == "kyc_documents"
        assert row.detector_version is None
        expected_scope = (
            EvidenceScope.CROSS_BANK
            if row.evidence_type.startswith("cross_bank_")
            else EvidenceScope.SINGLE_BANK
        )
        assert row.scope == expected_scope
        assert len(row.counterparty_bank_ids) == len(set(row.counterparty_bank_ids))


def test_seed42_reuse_respects_a_narrowed_image_root(
    seed42_config: DocaiConfig, tmp_path
) -> None:
    """Config drives where images are read from; nothing is hard-coded."""
    narrowed = replace(seed42_config, data_dir=tmp_path)
    assert narrowed.images_dir == tmp_path
    assert narrowed.documents_path == tmp_path / "kyc_documents.parquet"
