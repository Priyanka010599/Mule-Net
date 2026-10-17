"""Judge-uploaded data, scored live through the real detector -- one CSV per bank.

The sibling of ``agents/subset.py``: that module slices the *fixed*
published dataset live; this one accepts a judge's own small dataset,
split one CSV per bank the way the live demo's "three banks, three
fragments" framing already works, and runs it through the exact same,
unmodified :func:`ml.features.build_features` and
:func:`ml.rule.velocity_decision`. No new model, no new detection logic --
only a new way to get data into the same pipeline.

Why one CSV per bank, not one CSV with a bank column
-------------------------------------------------------
A single flat file with a ``bank_id`` column would let one upload claim
to speak for every bank at once, which is exactly the thing the rest of
this project's design (``docs/data_sharing.md``, ``enclave/``) exists to
make impossible: no party gets to author another party's data. Bank
identity here comes from *which upload slot a file was placed in* --
there is no ``bank_id`` column in a per-bank CSV at all (see
:data:`REQUIRED_COLUMNS`) -- which is the closest a browser upload can get
to the same separation.

How "my own account" is inferred without an accounts file
-------------------------------------------------------------
A judge uploads only transfers, never a separate accounts file, so
ownership has to be inferred. The rule: **an account belongs to bank X if
it appears as ``from_account`` in bank X's file.** That matches how a
bank's own transaction export naturally reads -- it lists what left (or
moved through) its own books. An account that never appears as
``from_account`` in *any* uploaded file is out of perimeter: it belongs to
a bank the judge did not upload, exactly as the live demo's single-bank
fragment view already treats a cross-bank counterparty it cannot see
inside. See :func:`_infer_ownership` for where this is enforced, and
:class:`UploadScoreOut` for how it is surfaced rather than hidden.

Two banks independently reporting the same real-world transfer (each
sees its own side) is expected, not an error -- see :func:`_dedup`.
"""

from __future__ import annotations

import csv
import io
import re
import time
from datetime import UTC
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ml.config import MlConfig
from ml.features import build_features
from ml.model_runtime import MODEL_C, load_model, score as model_score
from ml.rule import velocity_decision
from shared.contracts import Transfer

#: How many bank upload slots the UI offers. Matches "1 to N banks" --
#: bounded so the merged frame this endpoint builds stays small regardless
#: of how many slots a judge fills.
MAX_BANKS = 5

#: Per-file row cap. At this ceiling, MAX_BANKS files is at most 10,000
#: transfers total -- two orders of magnitude under the 118,406-transfer,
#: ~1.7s-measured 7-day slice ``agents/subset.py`` already serves live, so
#: no separate wall-clock timeout is needed: the row cap already bounds
#: the cost to well under a second.
MAX_ROWS_PER_BANK = 2_000

#: Per-file byte cap, checked before any CSV parsing. Generous for a few
#: thousand short rows; a tight stop against one slot holding something
#: that was never meant to be a small demo sample.
MAX_FILE_BYTES = 512 * 1_024

#: A per-bank CSV carries no bank_id column at all: identity comes from the
#: upload slot, not the data (see module docstring).
REQUIRED_COLUMNS: tuple[str, ...] = ("from_account", "to_account", "amount", "timestamp", "channel")

#: Placeholder bank_id for a counterparty account whose own bank was not
#: uploaded. One shared label rather than per-account None/NaN: it still
#: counts as one real (if unnamed) distinct bank for
#: ``cb_distinct_counterparty_banks``, which is the honest number -- an
#: account bridging two named banks and one unnamed one really does touch
#: three banks' worth of structure, even though only two can be named.
UNKNOWN_BANK = "(bank not uploaded)"

#: Mirrors ``agents.subset._EVIDENCE_COLUMNS``.
_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "distinct_counterparty_banks",
    "passthrough_ratio",
    "passthrough_speed_seconds",
)

#: Mirrors ``agents.subset.MAX_FLAGGED_ACCOUNTS_RETURNED``.
MAX_FLAGGED_ACCOUNTS_RETURNED = 50


#: Row-level problems reported per file before the rest are summarised.
MAX_ERRORS_REPORTED = 20

#: Cells that are echoed back to the browser (account ids, channel, bank
#: names) must not be able to start a spreadsheet formula if the result is ever
#: exported to CSV: no leading ``= + - @``, tab or CR. Hashed ids are hex, so
#: this costs a real upload nothing.
ACCOUNT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
CHANNEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_ -]{0,31}")
BANK_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}")


class UploadValidationError(ValueError):
    """Raised for malformed, oversized, or conflicting upload input.

    ``str(exc)`` is one message with multiple problems joined by ``"; "``
    (what the page has always shown). ``errors`` is the same information as
    data, ``[{bank, row, field, message}]`` (``row`` and ``field`` are ``None``
    for a file-level problem), capped at :data:`MAX_ERRORS_REPORTED` per
    file. Neither ever repeats a cell's content, so a bad row cannot leak an
    account id into an error message or a log line.
    """

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [{"bank": None, "row": None, "field": None, "message": message}]


def _file_error(bank_name: str, message: str) -> UploadValidationError:
    text = f"{bank_name}: {message}"
    return UploadValidationError(
        text, [{"bank": bank_name, "row": None, "field": None, "message": message}]
    )


def _check_row(raw: dict[str, str | None]) -> tuple[str, str] | None:
    """The first problem in one parsed row as ``(field, message)``, else ``None``."""
    if not ACCOUNT_ID_RE.fullmatch(raw.get("from_account") or ""):
        return "from_account", "invalid from_account"
    if not ACCOUNT_ID_RE.fullmatch(raw.get("to_account") or ""):
        return "to_account", "invalid to_account"
    if not CHANNEL_RE.fullmatch(raw.get("channel") or ""):
        return "channel", "invalid channel"
    return None


def _parse_bank_csv(bank_name: str, csv_text: str) -> list[Transfer]:
    """Parse and validate one bank's CSV into :class:`Transfer` rows.

    Reuses ``shared.contracts.Transfer`` for every row rather than
    hand-rolling amount/timestamp coercion: it is the same validator the
    real ingest path trusts (``_to_money``'s ``Decimal`` rounding, Pydantic's
    own datetime parsing), so a judge's upload is held to the same bar as
    real data, not a looser one.
    """
    # Characters never exceed bytes, so this cheap check rejects a huge
    # payload before it is encoded (a second full copy in memory).
    if len(csv_text) > MAX_FILE_BYTES or len(csv_text.encode("utf-8")) > MAX_FILE_BYTES:
        raise _file_error(bank_name, f"file is larger than {MAX_FILE_BYTES // 1024} KB")
    if "\x00" in csv_text:
        raise _file_error(bank_name, "file is not a text CSV (contains NUL bytes)")
    reader = csv.DictReader(io.StringIO(csv_text))
    try:
        fieldnames = reader.fieldnames
    except csv.Error:
        fieldnames = None
    if fieldnames is None:
        raise _file_error(bank_name, "file is empty")
    missing = set(REQUIRED_COLUMNS) - set(fieldnames)
    extra = [name for name in fieldnames if name not in REQUIRED_COLUMNS]
    if missing or extra:
        problem = []
        if missing:
            problem.append(f"missing column(s) {sorted(missing)}")
        if extra:
            problem.append(f"unexpected column(s) {len(extra)} (header names are not echoed)")
        raise _file_error(bank_name, f"{'; '.join(problem)}; expected exactly {list(REQUIRED_COLUMNS)}")

    rows: list[Transfer] = []
    errors: list[dict[str, Any]] = []

    def reject(row: int, field: str, message: str) -> bool:
        """Record one row problem; True once the per-file report is full."""
        errors.append({"bank": bank_name, "row": row, "field": field, "message": message})
        return len(errors) >= MAX_ERRORS_REPORTED

    try:
        for i, raw in enumerate(reader, start=2):  # header is row 1
            if i - 1 > MAX_ROWS_PER_BANK:
                raise _file_error(bank_name, f"more than {MAX_ROWS_PER_BANK} rows; trim the file")
            if None in raw:  # more cells than headers
                if reject(i, "row", "too many cells"):
                    break
                continue
            problem = _check_row(raw)
            if problem is not None:
                if reject(i, *problem):
                    break
                continue
            try:
                row = Transfer(
                    from_hashed_account_id=raw["from_account"],
                    to_hashed_account_id=raw["to_account"],
                    amount=raw["amount"],
                    timestamp=raw["timestamp"],
                    channel=raw["channel"],
                )
            except ValidationError as exc:
                field = exc.errors()[0]["loc"][0] if exc.errors() else "row"
                if reject(i, str(field), f"invalid {field}"):
                    break
                continue
            except Exception:  # noqa: BLE001
                # Transfer.amount's BeforeValidator raises a raw
                # decimal.InvalidOperation rather than a wrapped
                # ValidationError for a non-numeric amount -- confirmed live,
                # not assumed. Any row-level failure becomes a readable row
                # error here, never an uncaught 500 from untrusted input, and
                # the exception text is dropped: it can carry the cell.
                if reject(i, "amount", "invalid amount"):
                    break
                continue
            if row.amount <= 0:
                if reject(i, "amount", "amount must be greater than zero"):
                    break
                continue
            if row.timestamp.tzinfo is None:
                row.timestamp = row.timestamp.replace(tzinfo=UTC)  # naive means UTC
            rows.append(row)
    except csv.Error:
        raise _file_error(bank_name, "file is not a well-formed CSV") from None
    if errors:
        text = "; ".join(f"{bank_name} row {e['row']}: {e['message']}" for e in errors)
        if len(errors) >= MAX_ERRORS_REPORTED:
            text += "; ... further errors in this file not shown"
        raise UploadValidationError(text, errors)
    if not rows:
        raise _file_error(bank_name, "no data rows")
    return rows


def _infer_ownership(
    banks: dict[str, list[Transfer]],
) -> tuple[dict[str, str], int]:
    """hashed_account_id -> owning bank name, plus how many were ambiguous.

    See the module docstring for the base rule. One real wrinkle, found by
    testing this against a realistic two-sided report rather than assumed
    away: a RECEIVING bank's own export can legitimately carry
    ``from_account`` = the external sender's id (direction describes the
    money, not who is reporting it), so the same account can appear as
    ``from_account`` in more than one bank's file without either bank
    being wrong. There is no way to tell "genuinely contested" apart from
    "ordinary cross-bank reporting" from the upload alone, so this does
    not error: the first bank (by upload order) that names an account as
    ``from_account`` keeps it, and every account claimed by more than one
    bank is counted in the second return value so the caller can surface
    it, rather than silently picking a winner with no trace.
    """
    owner_of: dict[str, str] = {}
    ambiguous: set[str] = set()
    for bank_name, transfers in banks.items():
        for t in transfers:
            acct = t.from_hashed_account_id
            if acct in owner_of:
                if owner_of[acct] != bank_name:
                    ambiguous.add(acct)
                continue
            owner_of[acct] = bank_name
    return owner_of, len(ambiguous)


def _dedup(transfers: list[Transfer]) -> tuple[list[Transfer], int]:
    """Drop a transfer reported by both sides of a cross-bank pair.

    Keyed on (from, to, amount, timestamp) -- the four fields both sides of
    a real transfer would agree on. ``channel`` is deliberately excluded
    from the key: a mismatch there is more likely a labelling quirk between
    two banks' own systems than a different transfer, and including it
    risks under-deduping (double-counting) rather than over-deduping.

    Returns ``(deduped, removed_count)``.
    """
    seen: set[tuple[str, str, str, str]] = set()
    kept: list[Transfer] = []
    for t in transfers:
        key = (
            t.from_hashed_account_id,
            t.to_hashed_account_id,
            f"{t.amount:.2f}",
            t.timestamp.isoformat(),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(t)
    return kept, len(transfers) - len(kept)


def _empty_link_tables() -> dict[str, pd.DataFrame]:
    """Correctly-shaped, empty devices/KYC tables.

    A judge's upload carries no device or KYC data, so these contribute no
    identity-link features -- honest for a quick live test, not a gap: the
    detector still runs its full velocity/structure logic, just with zero
    identity evidence, same as it would for any account with none.
    """
    # dtype matters, not just column names: an empty column defaults to
    # object dtype, which fails a later comparison against the real
    # datetime64[ns, UTC] cutoff series (ml/features.py:_link_counts).
    datetime_utc = pd.Series([], dtype="datetime64[ns, UTC]")
    # kyc_submissions/kyc_documents additionally cannot both be truly
    # zero-row: ml.features.identity_links() maps kyc_documents' account
    # ids against a groupby of kyc_submissions, and pandas' own .map()
    # internally casts a zero-length datetime groupby result to float64
    # before anything else runs -- a pandas quirk, not a bug in ml/,
    # confirmed live by the TypeError this produced. One placeholder row
    # on each side, using sentinel ids that cannot collide with a real
    # hashed_account_id, keeps the groupby non-empty and dtype-correct;
    # the placeholder document's id never matches the placeholder
    # submission's id, so identity_links()'s own dropna() removes it --
    # the effective result is still no real identity-link evidence.
    now = pd.Timestamp.now(tz="UTC")
    return {
        "devices": pd.DataFrame(
            {
                "hashed_account_id": pd.Series([], dtype="object"),
                "device_fingerprint": pd.Series([], dtype="object"),
                "first_seen": datetime_utc,
            }
        ),
        "kyc_submissions": pd.DataFrame(
            {
                "hashed_account_id": ["__no_kyc_submission__"],
                "submitted_at": [now],
            }
        ),
        "kyc_documents": pd.DataFrame(
            {
                "hashed_account_id": ["__no_kyc_document__"],
                "hashed_id_number": ["__none__"],
            }
        ),
    }


#: Loaded once per process -- see agents/subset.py's identical pattern for
#: Model A. Model C never trained on identity features OR account_age_days,
#: so there is no skew risk here even though this path can never supply
#: either (see ml/train_model_c.py and ml/experiments/ablation/README.md
#: for why account_age_days was dropped specifically for this path).
_model_c_cache: list[Any] = []
_NO_MODEL = object()


def _get_model_c() -> Any:
    if not _model_c_cache:
        from agents.config import AgentsConfig

        loaded = load_model(AgentsConfig.from_env().models_dir, MODEL_C)
        _model_c_cache.append(loaded if loaded is not None else _NO_MODEL)
    cached = _model_c_cache[0]
    return None if cached is _NO_MODEL else cached


def score_upload(bank_csvs: list[tuple[str, str]]) -> dict[str, Any]:
    """Validate, merge, and score a judge's per-bank upload -- live, on this request.

    Args:
        bank_csvs: ``[(bank_name, csv_text), ...]``, one entry per filled
            upload slot, ``1 <= len <= MAX_BANKS``.

    Returns a dict shaped like ``agents.subset.score_subset``'s, plus the
    upload-specific fields in :class:`UploadScoreOut`.

    Raises:
        UploadValidationError: for anything malformed, oversized, or
            conflicting -- always with a message naming which bank and
            what was wrong, never a bare parse failure.
    """
    if not bank_csvs:
        raise UploadValidationError("at least one bank's file is required")
    if len(bank_csvs) > MAX_BANKS:
        raise UploadValidationError(f"at most {MAX_BANKS} banks are supported")
    names = [name.strip() for name, _ in bank_csvs]
    if any(not name for name in names):
        raise UploadValidationError("every bank needs a non-empty name")
    if any(not BANK_NAME_RE.fullmatch(name) for name in names):
        raise UploadValidationError(
            "bank names must be 1-40 characters: letters, digits, space, '_', '.', '-', "
            "starting with a letter or digit"
        )
    if len(set(names)) != len(names):
        raise UploadValidationError("bank names must be unique")

    banks: dict[str, list[Transfer]] = {
        name: _parse_bank_csv(name, csv_text) for name, csv_text in zip(names, (c for _, c in bank_csvs))
    }
    owner_of, ownership_ambiguous_count = _infer_ownership(banks)

    all_transfers = [t for rows in banks.values() for t in rows]
    deduped, duplicates_removed = _dedup(all_transfers)

    out_of_perimeter = {
        acct
        for t in deduped
        for acct in (t.from_hashed_account_id, t.to_hashed_account_id)
        if acct not in owner_of
    }
    # opened_at still has to exist with a valid dtype -- build_features()
    # computes account_age_days for every account unconditionally, whether
    # or not the serving model reads it. Model C (this path's model as of
    # ml/train_model_c.py) does NOT include account_age_days among its
    # features, so this placeholder -- the earliest transfer in the sample,
    # same for every account -- is now harmless rather than consequential:
    # it used to silently zero out one of Model B's features' discriminative
    # power for every upload; Model C was retrained specifically to not need
    # it at all (see ml/experiments/ablation/README.md).
    earliest = min((t.timestamp for t in deduped), default=pd.Timestamp.now(tz="UTC"))
    accounts = pd.DataFrame(
        {
            "hashed_account_id": list(owner_of) + list(out_of_perimeter),
            "bank_id": [owner_of[a] for a in owner_of] + [UNKNOWN_BANK] * len(out_of_perimeter),
            "opened_at": [earliest] * (len(owner_of) + len(out_of_perimeter)),
        }
    )
    transfers = pd.DataFrame(
        [
            {
                "transfer_id": t.transfer_id,
                "from_hashed_account_id": t.from_hashed_account_id,
                "to_hashed_account_id": t.to_hashed_account_id,
                "amount": t.amount,
                "currency": t.currency,
                "timestamp": t.timestamp,
                "channel": t.channel,
            }
            for t in deduped
        ]
    )

    tables: dict[str, pd.DataFrame] = {
        "accounts": accounts,
        "transfers": transfers,
        **_empty_link_tables(),
    }

    started = time.perf_counter()
    # cutoff=None -> the full window: a judge's upload is already a small,
    # bounded sample, not something to time-slice the way the fixed
    # published dataset's horizon picker does.
    features = build_features(MlConfig.from_env(), tables=tables, cutoff=None)
    load_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    model_c = _get_model_c()
    cb_scores = model_score(model_c, features) if model_c is not None else None
    if cb_scores is not None:
        cb_flag = cb_scores >= model_c.metadata["threshold_at_1pct_fpr"]
        detector = "model_c"
        model_c_performance = model_c.metadata.get("validated_performance")
    else:
        cb_flag = velocity_decision(features, "cb_")
        detector = "rule_fallback"
        model_c_performance = None
    sb_flag = velocity_decision(features, "sb_", require_bank_spread=False)
    # The shipped velocity rule on the JOINED (cross-bank) features, reported
    # alongside Model C rather than instead of it: Model C is weak on the short
    # histories a small upload has, the rule is what the product ships on.
    rule_cb_flag = velocity_decision(features, "cb_")
    score_ms = (time.perf_counter() - started) * 1000.0

    features = features.copy()
    features["flag_cb"] = cb_flag
    features["flag_sb"] = sb_flag
    features["flag_rule"] = rule_cb_flag
    bank_of = accounts.set_index("hashed_account_id")["bank_id"]
    features["display_bank_id"] = features["hashed_account_id"].map(bank_of)

    # Computed over every account (owned + out-of-perimeter, for correct
    # cross-bank breadth), but only OWNED accounts are ever returned --
    # the same "computed over everything, displayed is a filtered subset"
    # split agents/subset.py's bank filter already uses.
    shown = features.loc[features["hashed_account_id"].isin(owner_of)]
    flagged = shown.loc[shown["flag_cb"] | shown["flag_sb"] | shown["flag_rule"]].sort_values(
        "cb_in_degree", ascending=False
    )
    truncated = len(flagged) > MAX_FLAGGED_ACCOUNTS_RETURNED
    flagged_accounts = [
        {
            "hashed_account_id": str(row.hashed_account_id),
            "bank_id": str(row.display_bank_id),
            "caught_cross_bank": bool(row.flag_cb),
            "caught_single_bank": bool(row.flag_sb),
            "caught_by_rule": bool(row.flag_rule),
            **{name: _jsonable(getattr(row, f"cb_{name}")) for name in _EVIDENCE_COLUMNS},
        }
        for row in flagged.head(MAX_FLAGGED_ACCOUNTS_RETURNED).itertuples()
    ]

    return {
        "banks": names,
        "transfers_considered": len(deduped),
        "duplicate_transfers_removed": duplicates_removed,
        "accounts_in_scope": len(owner_of),
        "out_of_perimeter_accounts": len(out_of_perimeter),
        "ownership_ambiguous_accounts": ownership_ambiguous_count,
        "cross_bank_flagged": int(shown["flag_cb"].sum()),
        "single_bank_flagged": int(shown["flag_sb"].sum()),
        "rule_cross_bank_flagged": int(shown["flag_rule"].sum()),
        "flagged_accounts": flagged_accounts,
        "flagged_accounts_truncated": truncated,
        "detector": detector,
        "model_c_validated_performance": model_c_performance,
        "load_ms": int(round(load_ms)),
        "score_ms": int(round(score_ms)),
        "total_ms": int(round(load_ms + score_ms)),
    }


def _jsonable(value: Any) -> Any:
    """int64/float64 -> plain Python, so FastAPI can serialise it."""
    if hasattr(value, "item"):
        return value.item()
    return value
