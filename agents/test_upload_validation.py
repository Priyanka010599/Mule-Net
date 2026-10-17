"""Item 5: strict, structured, non-echoing upload validation."""

from __future__ import annotations

import logging

import pytest

from agents import subset_upload as su

HEADER = "from_account,to_account,amount,timestamp,channel\n"
GOOD = "acct1,acct2,10.00,2026-03-01T10:00:00Z,p2p\n"


def _errors(csv_text: str) -> list[dict]:
    with pytest.raises(su.UploadValidationError) as info:
        su.score_upload([("Bank A", csv_text)])
    return info.value.errors


@pytest.mark.parametrize(
    ("row", "field"),
    [
        ("=cmd|' /C calc'!A0,acct2,10,2026-03-01T10:00:00Z,p2p", "from_account"),
        ("acct1,+1+1,10,2026-03-01T10:00:00Z,p2p", "to_account"),
        ("acct1,acct2,10,2026-03-01T10:00:00Z,@SUM(A1)", "channel"),
        ("-acct1,acct2,10,2026-03-01T10:00:00Z,p2p", "from_account"),
        ("acct1,acct2,ten,2026-03-01T10:00:00Z,p2p", "amount"),
        ("acct1,acct2,0,2026-03-01T10:00:00Z,p2p", "amount"),
        ("acct1,acct2,-5,2026-03-01T10:00:00Z,p2p", "amount"),
        ("acct1,acct2,10,not-a-time,p2p", "timestamp"),
        ("a" * 129 + ",acct2,10,2026-03-01T10:00:00Z,p2p", "from_account"),
    ],
)
def test_each_bad_cell_is_a_row_level_error_naming_the_field(row: str, field: str) -> None:
    errors = _errors(HEADER + GOOD + row + "\n")
    assert [(e["bank"], e["row"], e["field"]) for e in errors] == [("Bank A", 3, field)]


def test_errors_never_echo_the_cell_content() -> None:
    secret = "SECRETACCOUNT123"
    errors = _errors(HEADER + f"{secret},acct2,notanumber,2026-03-01T10:00:00Z,p2p\n")
    assert secret not in str(errors)
    with pytest.raises(su.UploadValidationError) as info:
        su.score_upload([("Bank A", HEADER + f"{secret},acct2,xx,bad,p2p\n")])
    assert secret not in str(info.value)


def test_only_the_first_20_row_errors_are_reported() -> None:
    errors = _errors(HEADER + "acct1,acct2,bad,2026-03-01T10:00:00Z,p2p\n" * 50)
    assert len(errors) == su.MAX_ERRORS_REPORTED
    assert [e["row"] for e in errors] == list(range(2, 2 + su.MAX_ERRORS_REPORTED))


def test_unexpected_columns_nul_bytes_and_ragged_rows_are_rejected() -> None:
    extra = _errors(HEADER.strip() + ",note\n" + GOOD.strip() + ",x\n")
    assert "unexpected column" in extra[0]["message"]
    assert "NUL" in _errors(HEADER + GOOD + "\x00")[0]["message"]
    ragged = _errors(HEADER + GOOD.strip() + ",extra,cells\n")
    assert ragged[0]["field"] == "row" and ragged[0]["row"] == 2


def test_bank_names_that_could_start_a_formula_are_rejected() -> None:
    for name in ("=HYPERLINK(1)", "@x", "-1", "a" * 41, "Bank<script>"):
        with pytest.raises(su.UploadValidationError, match="bank names"):
            su.score_upload([(name, HEADER + GOOD)])


def test_a_naive_timestamp_is_read_as_utc_and_scores_without_error() -> None:
    rows = [
        "acct1,acct2,10.00,2026-03-01 10:00:00,p2p\n",
        "acct2,acct3,10.00,2026-03-01T10:05:00Z,p2p\n",
    ]
    result = su.score_upload([("Bank A", HEADER + "".join(rows))])
    assert result["transfers_considered"] == 2


def test_file_size_is_checked_before_parsing() -> None:
    big = HEADER + ("acct1,acct2,10.00,2026-03-01T10:00:00Z,p2p\n" * 20_000)
    assert "larger than" in _errors(big)[0]["message"]


def test_upload_logs_and_returns_no_raw_account_ids(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    ids = ("LOGPROBEACCOUNT1", "LOGPROBEACCOUNT2")
    good = f"{ids[0]},{ids[1]},10.00,2026-03-01T10:00:00Z,p2p\n"
    su.score_upload([("Bank A", HEADER + good)])
    bad = f"{ids[0]},{ids[1]},bad,2026-03-01T10:00:00Z,p2p\n"
    with pytest.raises(su.UploadValidationError):
        su.score_upload([("Bank A", HEADER + bad)])
    assert not any(i in record.getMessage() for record in caplog.records for i in ids)
