"""Tests for the judge-upload live-scoring path (agents/subset_upload.py).

No GCP, no Gemini, no real published dataset: every test builds its own
tiny per-bank CSVs and asserts on score_upload()'s output directly.
"""

from __future__ import annotations

import pytest

import agents.subset_upload as subset_upload_module
from agents.subset_upload import (
    MAX_BANKS,
    MAX_FILE_BYTES,
    MAX_ROWS_PER_BANK,
    UploadValidationError,
    score_upload,
)

HEADER = "from_account,to_account,amount,timestamp,channel\n"


def _csv(*rows: str) -> str:
    return HEADER + "\n".join(rows) + "\n"


def test_single_bank_round_trip():
    bank_a = _csv("acct_a1,acct_a2,100.00,2025-01-01T00:00:00Z,card")
    result = score_upload([("Bank A", bank_a)])
    assert result["banks"] == ["Bank A"]
    assert result["transfers_considered"] == 1
    assert result["accounts_in_scope"] == 1  # only acct_a1 is from_account anywhere
    assert result["out_of_perimeter_accounts"] == 1  # acct_a2


def test_dedup_removes_identical_transfer_reported_by_both_sides():
    row = "acct_a1,acct_b1,50.00,2025-01-01T00:00:00Z,card"
    bank_a = _csv(row)
    bank_b = _csv(row)  # Bank B's export happens to carry the identical row
    result = score_upload([("Bank A", bank_a), ("Bank B", bank_b)])
    assert result["transfers_considered"] == 1
    assert result["duplicate_transfers_removed"] == 1


def test_account_claimed_by_two_banks_resolves_to_first_without_erroring():
    """A receiving bank's own export can legitimately report from_account =
    the external sender (direction describes the money, not the reporter),
    so this must not be treated as a hard conflict -- see module docstring."""
    bank_a = _csv("acct_x,acct_a2,10.00,2025-01-01T00:00:00Z,card")
    bank_b = _csv("acct_x,acct_b2,10.00,2025-01-01T00:00:01Z,card")
    result = score_upload([("Bank A", bank_a), ("Bank B", bank_b)])
    assert result["ownership_ambiguous_accounts"] == 1


def test_out_of_perimeter_counterparty_is_not_an_error():
    bank_a = _csv("acct_a1,acct_unknown,10.00,2025-01-01T00:00:00Z,card")
    result = score_upload([("Bank A", bank_a)])
    assert result["out_of_perimeter_accounts"] == 1
    assert all(row["hashed_account_id"] != "acct_unknown" for row in result["flagged_accounts"])


def test_missing_required_column_is_rejected():
    bad = "from_account,to_account,amount,channel\nacct1,acct2,10.00,card\n"
    with pytest.raises(UploadValidationError, match="missing column"):
        score_upload([("Bank A", bad)])


def test_malformed_amount_is_rejected_with_row_number():
    bad = _csv("acct1,acct2,not-a-number,2025-01-01T00:00:00Z,card")
    with pytest.raises(UploadValidationError, match="row 2"):
        score_upload([("Bank A", bad)])


def test_empty_file_is_rejected():
    with pytest.raises(UploadValidationError, match="empty"):
        score_upload([("Bank A", "")])


def test_too_many_rows_is_rejected():
    rows = [f"acct{i},acct_x,1.00,2025-01-01T00:00:00Z,card" for i in range(MAX_ROWS_PER_BANK + 1)]
    with pytest.raises(UploadValidationError, match="more than"):
        score_upload([("Bank A", _csv(*rows))])


def test_oversized_file_is_rejected():
    huge = "x" * (MAX_FILE_BYTES + 1)
    with pytest.raises(UploadValidationError, match="larger than"):
        score_upload([("Bank A", huge)])


def test_too_many_banks_is_rejected():
    one_bank = _csv("acct1,acct2,1.00,2025-01-01T00:00:00Z,card")
    banks = [(f"Bank {i}", one_bank) for i in range(MAX_BANKS + 1)]
    with pytest.raises(UploadValidationError, match="at most"):
        score_upload(banks)


def test_duplicate_bank_names_rejected():
    one = _csv("acct1,acct2,1.00,2025-01-01T00:00:00Z,card")
    with pytest.raises(UploadValidationError, match="unique"):
        score_upload([("Bank A", one), ("Bank A", one)])


def _ring_shaped_banks(base=None) -> list[tuple[str, str]]:
    """A small, deliberately ring-shaped two-bank sample.

    ml.rule.VELOCITY_MIN_IN_DEGREE is 16, so the hub needs at least that
    many distinct incoming transfers to ever be eligible under the rule --
    not a smaller, more "obvious" number, because the rule-fallback test
    below specifically exercises the shipped rule.

    ``base`` is settable (default 2025-01-01) so a caller can shift every
    timestamp -- and therefore score_upload's derived ``opened_at``
    placeholder -- by an arbitrary amount while keeping every relative
    gap identical, to test that the placeholder's absolute value cannot
    move Model C's score.
    """
    import datetime as dt

    base = base if base is not None else dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    rows_a = []
    rows_b = []
    # acct_hub (owned by Bank B) receives from many Bank A senders, then
    # fans out fast to many Bank C accounts -- a textbook pass-through.
    for i in range(16):
        ts = (base + dt.timedelta(seconds=i)).isoformat()
        rows_a.append(f"acct_a{i},acct_hub,100.00,{ts},card")
    for i in range(16):
        ts = (base + dt.timedelta(seconds=20 + i)).isoformat()
        rows_b.append(f"acct_hub,acct_c{i},95.00,{ts},card")
    return [("Bank A", _csv(*rows_a)), ("Bank B", _csv(*rows_b))]


def test_normal_run_uses_model_c():
    """score_upload's live, primary path is Model C, not the rule."""
    result = score_upload(_ring_shaped_banks())
    assert result["detector"] == "model_c"
    assert result["model_c_validated_performance"] is not None
    assert "T=5m" in result["model_c_validated_performance"]
    assert "T=180m" in result["model_c_validated_performance"]


def test_falls_back_to_rule_when_model_c_unavailable(monkeypatch):
    """A ring-shaped sample should still fire the rule when the model cannot be used.

    Forces the model-unavailable path the same way a missing/corrupt
    model file would: _get_model_c returns None, never raises.
    """
    monkeypatch.setattr(subset_upload_module, "_get_model_c", lambda: None)
    result = score_upload(_ring_shaped_banks())
    assert result["detector"] == "rule_fallback"
    assert result["model_c_validated_performance"] is None
    flagged_ids = {row["hashed_account_id"] for row in result["flagged_accounts"]}
    assert "acct_hub" in flagged_ids


def test_upload_path_never_reads_account_age():
    """Model C's feature set, as actually loaded by this path, excludes age.

    Static check on the live, loaded model (not just the metadata file in
    isolation): if someone later points agents/subset_upload.py back at a
    model that does read account_age_days, this catches it.
    """
    model_c = subset_upload_module._get_model_c()
    assert model_c is not None, "model_c.ubj must be present for this test to mean anything"
    assert "cb_account_age_days" not in model_c.feature_columns


def test_identical_scores_for_two_different_opened_at_placeholders():
    """score_upload's opened_at placeholder cannot move Model C's score.

    Two runs of the exact same relative transfer pattern, shifted by a
    full year, get two different ``earliest`` placeholders internally
    (see agents/subset_upload.py) -- but Model C never reads
    account_age_days, so the flagged accounts and their evidence must be
    identical regardless.
    """
    import datetime as dt

    early = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    late = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    result_early = score_upload(_ring_shaped_banks(base=early))
    result_late = score_upload(_ring_shaped_banks(base=late))

    assert result_early["detector"] == result_late["detector"] == "model_c"
    evidence_early = {
        row["hashed_account_id"]: {k: v for k, v in row.items() if k != "hashed_account_id"}
        for row in result_early["flagged_accounts"]
    }
    evidence_late = {
        row["hashed_account_id"]: {k: v for k, v in row.items() if k != "hashed_account_id"}
        for row in result_late["flagged_accounts"]
    }
    assert evidence_early == evidence_late
