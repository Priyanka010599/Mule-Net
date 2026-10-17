"""Ask never exposes the simulator's hidden labels.

The fixture dataset (``test_chat_claims.tiny``) writes a real
``ground_truth.parquet`` with ``is_mule``, ``role``, ``cohort`` and
``pattern_type``, so these tests prove the labels are unreachable while they
are sitting on disk, not merely absent.
"""

from __future__ import annotations

import pytest

from agents import adk_agents, hidden_labels, tools, verify
from agents.sql_guard import SqlValidationError, validate_sql
from agents.test_chat_claims import Tiny, tiny  # noqa: F401 - tiny is a fixture

LABEL_QUERIES = [
    "SELECT * FROM ground_truth",
    "SELECT is_mule FROM ground_truth",
    "SELECT COUNT(*) FROM ground_truth WHERE is_mule = 1",
    "SELECT * FROM GROUND_TRUTH",
    'SELECT * FROM "ground_truth"',
    "SELECT * FROM ground_truth AS g",
    "WITH g AS (SELECT * FROM ground_truth) SELECT * FROM g",
    "SELECT * FROM accounts WHERE hashed_account_id IN (SELECT hashed_account_id "
    "FROM ground_truth WHERE is_mule)",
    "SELECT a.* FROM accounts a JOIN ground_truth g ON g.hashed_account_id = a.hashed_account_id",
    "SELECT * FROM ring_id_map",
    # Column names alone, on an allowed table: refused before execution.
    "SELECT is_mule FROM accounts",
    'SELECT "is_mule" FROM accounts',
    "SELECT a.role FROM accounts a",
    "SELECT COUNT(*) FROM accounts WHERE cohort = 'mule_ring'",
    "SELECT x FROM (SELECT pattern_type AS x FROM accounts)",
    "SELECT true_ring_id FROM rings",
]


@pytest.mark.parametrize("query", LABEL_QUERIES)
def test_label_tables_and_columns_fail_validation(query: str) -> None:
    with pytest.raises(SqlValidationError):
        validate_sql(query)


@pytest.mark.parametrize("query", LABEL_QUERIES)
def test_run_sql_cannot_read_labels_even_with_the_file_on_disk(
    tiny: Tiny, query: str  # noqa: F811
) -> None:
    with pytest.raises(ValueError):
        tools.run_sql(query)


def test_the_analyst_qa_tool_returns_an_error_value_that_leaks_nothing(
    tiny: Tiny,  # noqa: F811
) -> None:
    run_sql = adk_agents._model_tool(
        adk_agents._restricted_run_sql(adk_agents.ANALYST_QA_DENIED_TABLES), sink := set()
    )
    result = run_sql("SELECT COUNT(*) AS really_mules FROM ground_truth WHERE is_mule")
    assert "error" in result
    text = str(result)
    for label in ("is_mule", "cohort", "pattern_type", "mule_ring"):
        assert label not in text.replace("hidden simulator label", ""), label
    # The schema handed back to help the model recover lists no label.
    assert "ground_truth" not in result["error"]["allowed_tables"]
    assert sink == {"run_sql"}


def test_describe_tables_lists_no_label_table_or_column(tiny: Tiny) -> None:  # noqa: F811
    described = tools.describe_tables()["tables"]
    assert "ground_truth" not in described
    for columns in described.values():
        assert not [c for c in columns if hidden_labels.is_denied_column(c["name"])]
    with pytest.raises(ValueError):
        tools.describe_tables("ground_truth")


def test_a_tool_result_is_scrubbed_of_label_keys(tiny: Tiny) -> None:  # noqa: F811
    def leaky() -> dict:
        return {"n": 3, "rows": [{"hashed_account_id": "a", "is_mule": True, "role": "mule"}]}

    assert adk_agents._model_tool(leaky)() == {"n": 3, "rows": [{"hashed_account_id": "a"}]}


def test_a_scratch_database_is_built_without_label_columns(tiny: Tiny) -> None:  # noqa: F811
    """Defence in depth: a published table that grew a label column leaks nothing."""
    import sqlite3

    from agents.sql_guard import _load_table

    store = verify.default_store()
    frame = store.table("ground_truth")  # has is_mule / role / cohort
    connection = sqlite3.connect(":memory:")
    _load_table(connection, "gt", frame)
    columns = {row[1] for row in connection.execute('PRAGMA table_info("gt")')}
    assert "hashed_account_id" in columns
    assert not columns & hidden_labels.DENIED_COLUMNS


def test_a_claim_cannot_count_labels(tiny: Tiny) -> None:  # noqa: F811
    for kind, subject in (
        ("filtered_row_count", "accounts.is_mule=True"),
        ("distinct_value_count", "accounts.role"),
        ("table_row_count", "ground_truth"),
    ):
        claim = verify.Claim(id="c1", kind=kind, subject=subject, value=3, text="")
        assert verify.check_claim(claim) is not None


def test_permitted_queries_still_work(tiny: Tiny) -> None:  # noqa: F811
    rows = tools.run_sql("SELECT COUNT(*) AS n FROM accounts WHERE bank_id = 'BANK_A'")
    assert rows == [{"n": tiny.bank_a_count}]
    assert tools.run_sql("SELECT ring_id FROM rings")
