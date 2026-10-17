"""The hand-built sample upload (``ui/build_sample_upload.py``) must keep
showing what the upload tab promises: the planted ring's collectors are caught
by the velocity rule only when the banks are joined, and the decoys and
background traffic stay clean."""

from __future__ import annotations

from agents.subset_upload import MAX_FILE_BYTES, MAX_ROWS_PER_BANK, score_upload
from ui import build_sample_upload as bsu

COLLECTORS = {bsu.acct(f"collector/{bank}") for bank in bsu.BANKS}
DECOYS = {bsu.acct("merchant/BANK_A"), *(bsu.acct(f"family/{i}") for i in range(5))}


def _rule_flagged(files: dict[str, str], banks: tuple[str, ...]) -> set[str]:
    result = score_upload([(bank, files[bank]) for bank in banks])
    return {a["hashed_account_id"] for a in result["flagged_accounts"] if a["caught_by_rule"]}


def test_build_is_deterministic_and_within_the_upload_limits() -> None:
    first, second = bsu.build(), bsu.build()
    assert first == second
    for text in first.values():
        assert len(text.encode()) <= MAX_FILE_BYTES
        assert text.count("\n") - 1 <= MAX_ROWS_PER_BANK
        assert text.splitlines()[0] == ",".join(bsu.HEADER)


def test_joined_banks_flag_the_collectors_and_nothing_else() -> None:
    flagged = _rule_flagged(bsu.build(), bsu.BANKS)
    assert flagged == COLLECTORS


def test_one_bank_alone_flags_nothing() -> None:
    files = bsu.build()
    assert _rule_flagged(files, bsu.BANKS[:1]) == set()


def test_decoys_are_never_flagged() -> None:
    files = bsu.build()
    result = score_upload([(bank, files[bank]) for bank in bsu.BANKS])
    flagged = {a["hashed_account_id"] for a in result["flagged_accounts"]}
    assert not flagged & DECOYS
