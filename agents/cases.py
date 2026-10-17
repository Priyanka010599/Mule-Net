"""The ``cases`` producer: one case per ring, written only on verification.

Nothing produced ``cases`` before this. The table existed in the SQL
allowlist and in ``shared/ddl.sql``, and ``agents/`` served an empty
frame so ``FROM cases`` would not error -- which meant every query
against it quietly returned nothing.

**A case is a record that a narrative was checked, not that one was
written.** :func:`record_verified_investigation` writes only when
:func:`agents.loop.investigate_verified` comes back verified. A withheld
result writes nothing at all, and that is the enforcement rather than a
convention: ``loop.py`` drops the claim list along with the withheld
narrative, so a withheld outcome has ``verified_claim_count == 0``, and
the contract's ``verified_claim_count >= 1`` makes that shape
unconstructible. The producer cannot write an unverified case even if a
future edit forgets to check, because the model refuses to be built.

**One case per ring, forever.** ``case_id`` is
:func:`~shared.contracts.case_id_for_ring`, a uuid5 of the ring id, so a
second verified investigation of the same ring produces the same id and
updates the same row. Re-running the producer is idempotent instead of
minting duplicates.

Output mirrors ``ml/out/``: a parquet file plus a ``manifest.json``
recording row counts and what validated them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.config import AgentsConfig
from shared.contracts import Investigation, InvestigationStatus, case_id_for_ring

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from agents.loop import VerifiedInvestigation

#: Column order for ``cases.parquet``.
CASE_COLUMNS: tuple[str, ...] = (
    "case_id",
    "ring_id",
    "verified_at",
    "verification_attempts",
    "verified_claim_count",
    "analyst_notes",
    "status",
    "opened_at",
    "closed_at",
)


#: Where the producer writes, mirroring ``ml/out/``.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"


def cases_dir(config: AgentsConfig | None = None) -> Path:
    """Directory ``cases.parquet`` and its manifest live in."""
    config = config or AgentsConfig.from_env()
    return getattr(config, "out_dir", None) or DEFAULT_OUT_DIR


def cases_path(config: AgentsConfig | None = None) -> Path:
    """Path to ``agents/out/cases.parquet``."""
    return cases_dir(config) / "cases.parquet"


def manifest_path(config: AgentsConfig | None = None) -> Path:
    """Path to ``agents/out/manifest.json``."""
    return cases_dir(config) / "manifest.json"


def load_cases(config: AgentsConfig | None = None) -> pd.DataFrame:
    """Read the cases table, or an empty frame shaped like it."""
    import pandas as pd

    path = cases_path(config)
    if path.is_file():
        return pd.read_parquet(path)
    return empty_cases_frame()


def empty_cases_frame() -> pd.DataFrame:
    """An empty frame with the real dtypes.

    Shaped rather than bare so ``FROM cases`` returns no rows instead of
    failing on a missing column, and so a comparison against
    ``verified_at`` type-checks on an empty table.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "case_id": pd.Series(dtype="string"),
            "ring_id": pd.Series(dtype="string"),
            "verified_at": pd.Series(dtype="datetime64[us, UTC]"),
            "verification_attempts": pd.Series(dtype="int64"),
            "verified_claim_count": pd.Series(dtype="int64"),
            "analyst_notes": pd.Series(dtype="string"),
            "status": pd.Series(dtype="string"),
            "opened_at": pd.Series(dtype="datetime64[us, UTC]"),
            "closed_at": pd.Series(dtype="datetime64[us, UTC]"),
        }
    )


def build_case(
    ring_id: str,
    outcome: VerifiedInvestigation,
    verified_at: datetime | None = None,
    analyst_notes: str = "",
) -> Investigation:
    """Build the case for one verified investigation.

    Args:
        ring_id: The ring investigated.
        outcome: A **verified** outcome. An unverified one is refused.
        verified_at: When the verifier passed. Defaults to now, in UTC.
        analyst_notes: Optional notes to carry on the case.

    Returns:
        A validated :class:`~shared.contracts.Investigation`.

    Raises:
        ValueError: If ``outcome`` did not verify, or if the contract
            rejects the row.
    """
    if not outcome.verified:
        raise ValueError(
            f"refusing to build a case for {ring_id!r}: the investigation did "
            "not verify. A case records that a narrative was checked."
        )
    return Investigation(
        case_id=case_id_for_ring(ring_id),
        ring_id=ring_id,
        verified_at=verified_at or datetime.now(UTC),
        verification_attempts=outcome.attempts,
        verified_claim_count=len(outcome.claims),
        analyst_notes=analyst_notes,
        status=InvestigationStatus.OPEN,
        closed_at=None,
    )


def upsert_cases(
    cases: list[Investigation],
    config: AgentsConfig | None = None,
) -> dict[str, int]:
    """Merge cases into ``cases.parquet`` by ``case_id``, then write it.

    A ring already holding a **closed** case is left untouched: closing
    is an analyst's decision, and a later re-verification must not
    silently reopen it.

    Args:
        cases: Validated cases to merge.
        config: Locates the output directory.

    Returns:
        Counts of what happened: ``inserted``, ``updated``,
        ``skipped_closed`` and the resulting ``total``.
    """
    import pandas as pd

    existing = load_cases(config)
    by_id: dict[str, dict[str, Any]] = {
        str(row["case_id"]): dict(row) for _, row in existing.iterrows()
    }

    counts = {"inserted": 0, "updated": 0, "skipped_closed": 0}
    for case in cases:
        current = by_id.get(case.case_id)
        if current is not None and current.get("status") == (
            InvestigationStatus.CLOSED.value
        ):
            counts["skipped_closed"] += 1
            continue
        row = case.model_dump(mode="python")
        if current is not None:
            # Preserve the original opening time: the case is the same
            # case, re-verified, not a new one.
            row["opened_at"] = current["opened_at"]
            counts["updated"] += 1
        else:
            counts["inserted"] += 1
        row["status"] = case.status.value
        by_id[case.case_id] = row

    frame = pd.DataFrame(list(by_id.values()), columns=list(CASE_COLUMNS))
    frame = frame.sort_values("case_id", kind="stable").reset_index(drop=True)
    _write(frame, config)
    counts["total"] = len(frame)
    return counts


def _write(frame: pd.DataFrame, config: AgentsConfig | None = None) -> None:
    """Write the parquet and refresh the manifest beside it."""
    directory = cases_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cases_path(config), index=False)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "producer": "agents.cases",
        "tables": {
            "cases": {
                "rows": int(len(frame)),
                "validated_against": "shared.contracts.Investigation",
                "columns": list(CASE_COLUMNS),
                "key": "case_id = case_id_for_ring(ring_id), uuid5",
                "written_only_when": "investigate_verified returned verified",
            }
        },
    }
    manifest_path(config).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def record_verified_investigation(
    ring_id: str,
    outcome: VerifiedInvestigation,
    config: AgentsConfig | None = None,
    analyst_notes: str = "",
) -> Investigation | None:
    """Write a case for one investigation, if it verified.

    This is the whole public entry point. Call it with any outcome; it
    decides.

    Returns:
        The case written, or ``None`` when the investigation did not
        verify and nothing was written.
    """
    if not outcome.verified:
        return None
    case = build_case(ring_id, outcome, analyst_notes=analyst_notes)
    upsert_cases([case], config)
    return case
