"""Live comparison: the BigQuery-backed ``DataStore`` path against parquet.

The one thing a mock cannot tell us is whether
``agents.data.DataStore._load_from_bigquery`` actually reproduces what
``pd.read_parquet`` produces today -- same columns, same dtypes, same
values -- which is the whole point of the new path being *additive*:
``agents/data.py``'s module docstring and ``docs/reviewer_notes.md`` both
say parquet stays the default and this stays untried-in-production until a
human deploys it. So this module runs against real BigQuery, not a stub.

Needs, both real:

* ``simulator/out/*.parquet`` on disk (``python -m simulator.generator``;
  the default seed reproduces the exact dataset already loaded into
  BigQuery -- 572 890 ``transfers`` rows, matching
  ``mulenet.mulenet.transfers``'s live row count).
* Real BigQuery credentials and the ``mulenet.mulenet`` dataset populated
  by ``infra/load_bigquery.py``.

Both are unavailable/expensive in CI, so the whole module is opt-in via
``MULENET_BQ_LIVE=1`` -- the exact gate ``ui/test_bigquery_rings.py`` uses
for its own live BigQuery comparison. Skipped (not failed) otherwise, same
as every other BigQuery-touching test in this repo.

Only ``SIMULATOR_TABLES`` (``accounts``, ``transfers``, ``devices``,
``kyc_documents``, ``kyc_submissions``, ``ground_truth``) get the
byte-for-byte comparison: this worktree has no ``ml/out/*.parquet`` to
diff the ``ML_TABLES`` against (it is a gitignored build artifact of
running ``ml/``, not committed). The BigQuery load path is the same
generic function for both groups (``_load_from_bigquery`` is driven
entirely by ``shared.bq_schemas.TABLE_SCHEMAS``, not special-cased per
table), and :func:`test_bigquery_typed_reads_agree_with_the_contract`
below exercises it against the ``ML_TABLES``' real, already-published
BigQuery data by round-tripping every typed accessor that reads nested
RECORD/REPEATED columns (``ring()``, ``ring_explanation()``,
``account_explanation()``, ``account_evidence()``) -- not a parquet diff,
but a real, live, non-mocked check that the nested columns survive the
BigQuery round trip in the shape the contract requires.
"""

from __future__ import annotations

import dataclasses
import os

import pandas as pd
import pytest

from agents.config import AgentsConfig
from agents.data import BIGQUERY_BACKED_TABLES, DataStore

pytestmark = pytest.mark.skipif(
    os.environ.get("MULENET_BQ_LIVE", "") not in {"1", "true", "yes"},
    reason="set MULENET_BQ_LIVE=1 to run the live BigQuery-vs-parquet "
    "comparison (needs credentials and scans the mulenet.mulenet dataset)",
)

PARQUET_CONFIG = AgentsConfig.from_env()
BIGQUERY_CONFIG = dataclasses.replace(PARQUET_CONFIG, data_source="bigquery")

#: name -> the column that uniquely identifies a row, for sorting before
#: comparison. Row order is the one thing allowed to differ between the
#: two paths: parquet preserves the generator's write order, BigQuery's
#: plain ``SELECT`` makes no ordering guarantee. Nothing downstream
#: relies on a specific order -- everywhere order matters,
#: ``agents/data.py`` already sorts explicitly (see ``transfers_frame``).
SIMULATOR_PRIMARY_KEY: dict[str, str] = {
    "accounts": "hashed_account_id",
    "transfers": "transfer_id",
    "devices": "device_id",
    "kyc_documents": "doc_id",
    "kyc_submissions": "submission_id",
    "ground_truth": "hashed_account_id",
}

_needs_parquet = pytest.mark.skipif(
    not (AgentsConfig.from_env().data_dir / "accounts.parquet").is_file(),
    reason="run `python -m simulator.generator` first",
)


@_needs_parquet
@pytest.mark.parametrize("name", sorted(SIMULATOR_PRIMARY_KEY))
def test_bigquery_matches_parquet_exactly(name: str) -> None:
    """Same columns, same dtypes, same values, for every simulator table."""
    key = SIMULATOR_PRIMARY_KEY[name]
    parquet_store = DataStore(PARQUET_CONFIG)
    bigquery_store = DataStore(BIGQUERY_CONFIG)

    from_parquet = (
        parquet_store.table(name).sort_values(key).reset_index(drop=True)
    )
    from_bigquery = (
        bigquery_store.table(name).sort_values(key).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(from_parquet, from_bigquery)


def test_missing_tables_finds_every_bigquery_backed_table() -> None:
    """The readiness check must recognise the BigQuery path, not just stat
    a local file that bigquery mode never reads."""
    store = DataStore(BIGQUERY_CONFIG)
    missing = set(store.missing_tables())
    assert not (missing & BIGQUERY_BACKED_TABLES)


def test_bigquery_typed_reads_agree_with_the_contract() -> None:
    """Every typed accessor that reads a nested RECORD/REPEATED column
    still builds a valid ``shared.contracts`` object from real,
    already-published BigQuery data (``rings``, ``scores``,
    ``ring_explanations``, ``account_explanations``,
    ``account_evidence``) -- the group this module cannot diff against a
    local parquet file (see the module docstring).
    """
    store = DataStore(BIGQUERY_CONFIG)

    rings = store.table("rings")
    assert len(rings) > 0
    ring_id = str(rings.iloc[0]["ring_id"])
    ring = store.ring(ring_id)
    assert ring.ring_id == ring_id
    assert ring.members

    explanation = store.ring_explanation(ring_id)
    assert explanation.ring_id == ring_id
    assert explanation.narrative

    scores = store.table("scores")
    assert len(scores) > 0
    account_id = str(scores.iloc[0]["hashed_account_id"])
    score = store.risk_score(account_id)
    assert score is not None
    assert score.hashed_account_id == account_id

    account_explanations = store.table("account_explanations")
    assert len(account_explanations) > 0
    explained_account = str(account_explanations.iloc[0]["hashed_account_id"])
    account_explanation = store.account_explanation(explained_account)
    assert account_explanation.hashed_account_id == explained_account

    evidence_accounts = store.table("account_evidence")["hashed_account_id"]
    assert len(evidence_accounts) > 0
    evidence = store.account_evidence(str(evidence_accounts.iloc[0]))
    assert evidence

    ring_members = store.table("ring_members")
    assert len(ring_members) >= len(ring.members)
