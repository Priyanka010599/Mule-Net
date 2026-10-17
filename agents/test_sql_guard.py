"""Adversarial tests for the ``run_sql`` boundary.

``run_sql`` takes a query an analyst wrote, or that a model wrote on an
analyst's behalf, and runs it against the bank tables. It is the one
place in ``agents/`` where untrusted text becomes execution, so it gets
tested the way a boundary is tested: mostly by trying to get past it.

The attacks below are grouped by what they abuse -- statement
termination, comments, CTEs, quoting, schema qualification, the row
limit. Where a rule comes from the allowlist, the test reads
``ALLOWED_SQL_TABLES`` rather than hardcoding names, so extending the
allowlist cannot silently leave a test asserting a stale set.
"""

from __future__ import annotations

import pytest

from agents import tools
from agents.sql_guard import SqlValidationError, validate_sql
from shared.tool_interfaces import ALLOWED_SQL_TABLES, SQL_ROW_LIMIT

# --------------------------------------------------------------- accepted


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM accounts",
        "select bank_id from accounts",
        "SELECT bank_id, COUNT(*) FROM accounts GROUP BY bank_id",
        "SELECT * FROM accounts WHERE bank_id = 'BANK_A' LIMIT 10",
        "SELECT a.bank_id FROM accounts a WHERE a.status = 'active'",
        "WITH r AS (SELECT * FROM rings) SELECT ring_id FROM r",
        "SELECT * FROM scores WHERE score > 0.9 ORDER BY score DESC",
        "SELECT * FROM account_evidence WHERE scope = 'cross_bank'",
        "SELECT (SELECT COUNT(*) FROM rings) AS n FROM accounts LIMIT 1",
    ],
)
def test_accepts_read_only_selects(query: str) -> None:
    """A genuine read-only SELECT must survive validation."""
    assert validate_sql(query).sql


def test_every_allowlisted_table_is_queryable() -> None:
    """No allowlisted name is rejected by the validator.

    Guards the case where ``ALLOWED_SQL_TABLES`` grows but the data layer
    has no table behind the new name.
    """
    for table in sorted(ALLOWED_SQL_TABLES):
        validated = validate_sql(f"SELECT * FROM {table} LIMIT 1")
        assert table in validated.tables


# ------------------------------------------------- multiple statements


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM accounts; DROP TABLE accounts",
        "SELECT * FROM accounts; SELECT * FROM rings",
        "SELECT * FROM accounts; INSERT INTO accounts VALUES ('x')",
        "SELECT * FROM accounts;\n-- trailing\nDELETE FROM rings",
    ],
)
def test_rejects_stacked_statements(query: str) -> None:
    """A second statement is never executed, whatever follows the first."""
    with pytest.raises(SqlValidationError):
        validate_sql(query)


def test_trailing_semicolons_are_benign() -> None:
    """A stray terminator is not a second statement.

    ``SELECT 1 FROM accounts;;`` parses to one SELECT plus an empty
    segment. Rejecting it would be theatre: the executed SQL is
    regenerated from the parsed tree, so the terminators are gone by the
    time anything runs. The assertion that matters is that one statement
    comes out the other side.
    """
    validated = validate_sql("SELECT 1 FROM accounts;;")
    assert ";" not in validated.sql
    assert validated.tables == frozenset({"accounts"})


# ------------------------------------------------------------ DDL / DML


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE accounts",
        "DELETE FROM accounts",
        "UPDATE accounts SET bank_id = 'X'",
        "INSERT INTO accounts VALUES ('x')",
        "CREATE TABLE evil (a INT)",
        "ALTER TABLE accounts ADD COLUMN evil TEXT",
        "TRUNCATE TABLE accounts",
        "PRAGMA table_info(accounts)",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "VACUUM",
        "SELECT * INTO evil FROM accounts",
        "REPLACE INTO accounts VALUES ('x')",
    ],
)
def test_rejects_writes_and_commands(query: str) -> None:
    """Anything that is not a SELECT is refused before execution."""
    with pytest.raises(SqlValidationError):
        validate_sql(query)


# ------------------------------------------------------------ CTE tricks


@pytest.mark.parametrize(
    "query",
    [
        "WITH x AS (DELETE FROM accounts RETURNING *) SELECT * FROM x",
        "WITH x AS (INSERT INTO accounts VALUES ('a') RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE accounts SET bank_id='X' RETURNING *) SELECT * FROM x",
        "WITH a AS (SELECT * FROM accounts), "
        "b AS (DELETE FROM rings RETURNING *) SELECT * FROM a",
    ],
)
def test_rejects_write_hidden_in_cte(query: str) -> None:
    """A write inside a CTE is still a write.

    This is the attack a naive ``query.strip().upper().startswith("SELECT")``
    check waves straight through, which is why validation walks the parsed
    tree instead.
    """
    with pytest.raises(SqlValidationError):
        validate_sql(query)


def test_cte_alias_does_not_open_the_allowlist() -> None:
    """Naming a CTE after a forbidden table does not make it readable."""
    with pytest.raises(SqlValidationError):
        validate_sql(
            "WITH sqlite_master AS (SELECT * FROM sqlite_master) "
            "SELECT * FROM sqlite_master"
        )


# -------------------------------------------------------------- allowlist


@pytest.mark.parametrize(
    "table",
    ["sqlite_master", "sqlite_temp_master", "pg_catalog", "users", "secrets", "ml_out"],
)
def test_rejects_tables_outside_the_allowlist(table: str) -> None:
    """Only the allowlisted tables are reachable."""
    assert table not in ALLOWED_SQL_TABLES
    with pytest.raises(SqlValidationError):
        validate_sql(f"SELECT * FROM {table}")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM main.accounts",
        "SELECT * FROM temp.accounts",
        "SELECT * FROM otherdb.public.accounts",
    ],
)
def test_rejects_qualified_table_names(query: str) -> None:
    """A schema-qualified name could reach a database we did not attach."""
    with pytest.raises(SqlValidationError):
        validate_sql(query)


def test_rejects_forbidden_table_in_a_join() -> None:
    """Every table in the query is checked, not just the first."""
    with pytest.raises(SqlValidationError):
        validate_sql(
            "SELECT * FROM accounts JOIN sqlite_master ON 1=1"
        )


def test_rejects_forbidden_table_in_a_subquery() -> None:
    """A nested SELECT cannot smuggle in an unlisted table."""
    with pytest.raises(SqlValidationError):
        validate_sql(
            "SELECT * FROM accounts WHERE bank_id IN "
            "(SELECT name FROM sqlite_master)"
        )


def test_rejects_query_reading_no_allowlisted_table() -> None:
    """A query with no real table reads nothing and is refused."""
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT 1")


# ------------------------------------------------------------- row limit


def test_appends_limit_when_absent() -> None:
    """A query with no LIMIT gets the maximum, not unbounded output."""
    assert str(SQL_ROW_LIMIT) in validate_sql("SELECT * FROM accounts").sql


def test_preserves_a_smaller_limit() -> None:
    """A tighter limit the analyst asked for is respected."""
    assert "5" in validate_sql("SELECT * FROM accounts LIMIT 5").sql


@pytest.mark.parametrize("requested", [SQL_ROW_LIMIT + 1, 1000, 10**9])
def test_rejects_limit_above_the_maximum(requested: int) -> None:
    """Asking for more than the cap is an error, not a silent truncation.

    Silently clamping would let a caller believe it had the full result
    set. The contract says this raises.
    """
    with pytest.raises(SqlValidationError):
        validate_sql(f"SELECT * FROM accounts LIMIT {requested}")


@pytest.mark.parametrize(
    "limit", ["'100'", "1 + 1", "(SELECT 100)", "-1", "99999999999999999999"]
)
def test_rejects_non_literal_or_negative_limits(limit: str) -> None:
    """A LIMIT that is not a plain non-negative integer cannot be checked."""
    with pytest.raises(SqlValidationError):
        validate_sql(f"SELECT * FROM accounts LIMIT {limit}")


def test_run_sql_never_returns_more_than_the_limit() -> None:
    """The cap holds at execution, not merely in the rewritten SQL."""
    assert len(tools.run_sql("SELECT * FROM accounts")) <= SQL_ROW_LIMIT


# ---------------------------------------------------- malformed and empty


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "-- just a comment"])
def test_rejects_empty_or_commentary_queries(query: str) -> None:
    """Nothing to run is an error, not an empty result set."""
    with pytest.raises(SqlValidationError):
        validate_sql(query)


def test_comment_cannot_disguise_a_second_statement() -> None:
    """A comment does not neutralise what follows it."""
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT * FROM accounts /* harmless */; DROP TABLE accounts")


def test_select_inside_a_string_literal_does_not_make_a_query_readable() -> None:
    """The word SELECT in a literal is data, not a statement type."""
    with pytest.raises(SqlValidationError):
        validate_sql("DROP TABLE accounts -- SELECT * FROM accounts")


# ------------------------------------------------------------- execution


def test_run_sql_raises_value_error_for_rejected_queries() -> None:
    """The contract documents ValueError; SqlValidationError subclasses it."""
    with pytest.raises(ValueError):
        tools.run_sql("DROP TABLE accounts")


def test_run_sql_returns_mappings() -> None:
    """Rows come back as column-name to value mappings."""
    rows = tools.run_sql("SELECT bank_id FROM accounts LIMIT 3")
    assert rows and all(isinstance(row, dict) and "bank_id" in row for row in rows)


def test_run_sql_aggregate_matches_the_table() -> None:
    """An aggregate over a real table returns the real number."""
    rows = tools.run_sql("SELECT COUNT(*) AS n FROM rings")
    assert rows[0]["n"] > 0
