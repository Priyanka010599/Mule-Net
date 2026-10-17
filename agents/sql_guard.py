"""Validation and execution for :func:`agents.tools.run_sql`.

This is a security boundary: the query text arrives from an analyst (or
from an LLM relaying an analyst) and must not be able to write, escalate
to an unlisted table, or return an unbounded result set.

Validation is done on a parsed abstract syntax tree, never on the query
string. String matching is not defensible here -- ``SELECT`` appears
inside ``/*SELECT*/ DROP TABLE accounts``, inside string literals, and
inside ``WITH x AS (DELETE ... RETURNING *) SELECT * FROM x``.

The four rules, in the order they are applied:

1. The text parses to exactly one statement, and that statement is a
   ``SELECT``. Anything else -- extra statements, ``UNION``, ``INSERT``,
   ``PRAGMA``, a bare comment -- is rejected.
2. No node anywhere in the tree is a write, a DDL, or an out-of-band
   command. This is what catches write CTEs and ``SELECT ... INTO``.
3. Every table reference resolves to an unqualified name in
   :data:`~shared.tool_interfaces.ALLOWED_SQL_TABLES`, or to a CTE
   defined in the same query. Catalog- and schema-qualified names are
   rejected outright so ``main.sqlite_master`` cannot be reached.
4. A ``LIMIT`` of at most :data:`~shared.tool_interfaces.SQL_ROW_LIMIT`
   is present; a larger one is an error and a missing one is added.

Execution runs the validated SQL against an in-memory SQLite database
opened read-only over copies of just the tables the query named, so even
a validator bypass has nothing durable to write to.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import expressions as exp

from agents import hidden_labels
from shared.tool_interfaces import ALLOWED_SQL_TABLES, SQL_ROW_LIMIT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agents.data import DataStore

#: SQL dialect the analyst query is parsed and executed as.
DIALECT = "sqlite"

#: Expression types that write, change structure, or step outside the
#: query language. Finding any of these anywhere in the tree is fatal,
#: which is what stops a write hidden inside a CTE or a subquery.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Command,  # ATTACH, PRAGMA, VACUUM, and anything else unparsed
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Set,
    exp.Into,  # SELECT ... INTO <table>
)


class SqlValidationError(ValueError):
    """Raised when a query fails validation.

    A subclass of ``ValueError`` because
    :func:`shared.tool_interfaces.run_sql` documents ``ValueError`` for
    every rejection reason.
    """


@dataclass(frozen=True)
class ValidatedQuery:
    """A query that passed validation, with its limit already enforced."""

    sql: str
    tables: frozenset[str]


def validate_sql(query: str) -> ValidatedQuery:
    """Validate an analyst query and return it with ``LIMIT`` enforced.

    Args:
        query: The candidate ``SELECT`` statement.

    Returns:
        The rewritten, safe-to-execute query and the allowlisted tables
        it reads.

    Raises:
        SqlValidationError: On any rule violation. The message names the
            rule that failed.
    """
    if not query or not query.strip():
        raise SqlValidationError("query is empty")

    try:
        statements = sqlglot.parse(query, dialect=DIALECT)
    except Exception as exc:  # sqlglot raises several parse error types
        raise SqlValidationError(f"query does not parse: {exc}") from exc

    # Rule 1: exactly one statement, and it is a SELECT.
    real = [s for s in statements if s is not None]
    if len(real) != 1:
        raise SqlValidationError(
            f"expected exactly one statement, found {len(real)}"
        )
    statement = real[0]
    if not isinstance(statement, exp.Select):
        raise SqlValidationError(
            f"only SELECT is allowed, got {type(statement).__name__.upper()}"
        )

    # Rule 2: no write, DDL, or command node anywhere in the tree.
    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlValidationError(
                f"{type(node).__name__.upper()} is not allowed in a read-only query"
            )

    # Rule 3: every table reference is allowlisted or a CTE of this query.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        if table.catalog or table.db:
            raise SqlValidationError(
                f"qualified table name {table.sql(dialect=DIALECT)!r} is not allowed"
            )
        name = table.name.lower()
        if name in cte_names:
            continue
        if name in hidden_labels.DENIED_TABLES:
            raise SqlValidationError(
                f"table {name!r} holds hidden simulator labels and cannot be queried"
            )
        if name not in ALLOWED_SQL_TABLES:
            raise SqlValidationError(
                f"table {name!r} is not in the allowlist: {sorted(ALLOWED_SQL_TABLES)}"
            )
        tables.add(name)
    if not tables:
        raise SqlValidationError("query reads no allowlisted table")

    # Rule 3b: no hidden simulator label, by column name, wherever it
    # appears in the tree (select list, WHERE, a subquery, a quoted name).
    for column in statement.find_all(exp.Column):
        if hidden_labels.is_denied_column(column.name):
            raise SqlValidationError(
                f"column {column.name!r} is a hidden simulator label and cannot be queried"
            )

    # Rule 4: enforce the row limit.
    safe = _apply_limit(statement)
    return ValidatedQuery(sql=safe.sql(dialect=DIALECT), tables=frozenset(tables))


def _apply_limit(statement: exp.Select) -> exp.Select:
    """Return the statement with a literal ``LIMIT`` of at most 100."""
    limit = statement.args.get("limit")
    if limit is None:
        return statement.limit(SQL_ROW_LIMIT)

    expression = limit.expression
    if not isinstance(expression, exp.Literal) or not expression.is_int:
        raise SqlValidationError("LIMIT must be an integer literal")
    requested = int(expression.name)
    if requested < 0:
        raise SqlValidationError("LIMIT must not be negative")
    if requested > SQL_ROW_LIMIT:
        raise SqlValidationError(
            f"LIMIT {requested} exceeds the maximum of {SQL_ROW_LIMIT}"
        )
    return statement


def execute_select(store: DataStore, validated: ValidatedQuery) -> list[dict[str, Any]]:
    """Run a validated query against an in-memory copy of its tables.

    Args:
        store: Source of the table frames.
        validated: Output of :func:`validate_sql`. Never pass raw text.

    Returns:
        Up to :data:`SQL_ROW_LIMIT` rows as column-name to value mappings.

    Raises:
        RuntimeError: If the query is valid but fails to execute.
    """
    connection = _scratch_database(store, validated.tables)
    try:
        connection.row_factory = sqlite3.Row
        # Nothing may write even if a table were somehow reachable.
        connection.execute("PRAGMA query_only = ON")
        try:
            cursor = connection.execute(validated.sql)
            rows = cursor.fetchmany(SQL_ROW_LIMIT)
        except sqlite3.Error as exc:
            raise RuntimeError(f"query failed to execute: {exc}") from exc
        return [dict(row) for row in rows]
    finally:
        connection.close()


#: Loaded scratch databases, as SQLite images, keyed by the tables they hold.
#:
#: Copying a DataFrame into SQLite cell by cell cost ~2.8s for the 30,000-row
#: ``accounts`` table on **every** query, which was most of the non-model time
#: of a typical Ask question. The image is built once per (store, data
#: version, tables) and each query deserialises its own private copy, so a
#: query still cannot see or change another's.
_IMAGES: OrderedDict[tuple[Any, ...], tuple[Any, bytes]] = OrderedDict()
_IMAGES_LOCK = threading.Lock()
_MAX_IMAGES = 8
#: An image larger than this is rebuilt per query instead of held (``transfers``).
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


def _scratch_database(store: DataStore, tables: frozenset[str]) -> sqlite3.Connection:
    """An in-memory database holding exactly ``tables``, loaded or restored."""
    from agents import tool_cache  # noqa: PLC0415 - avoids a cycle

    names = tuple(sorted(tables))
    try:
        key: tuple[Any, ...] | None = (tool_cache.data_key(store), names)
    except Exception:  # noqa: BLE001 - an unkeyable store is built fresh each time
        key = None
    if key is not None and tool_cache.enabled():
        with _IMAGES_LOCK:
            entry = _IMAGES.get(key)
            if entry is not None and entry[0] is store:
                _IMAGES.move_to_end(key)
                connection = sqlite3.connect(":memory:")
                connection.deserialize(entry[1])
                return connection

    connection = sqlite3.connect(":memory:")
    try:
        for name in names:
            _load_table(connection, name, store.table(name))
    except BaseException:
        connection.close()
        raise
    if key is not None and tool_cache.enabled():
        image = connection.serialize()
        if len(image) <= _MAX_IMAGE_BYTES:
            with _IMAGES_LOCK:
                _IMAGES[key] = (store, image)
                while len(_IMAGES) > _MAX_IMAGES:
                    _IMAGES.popitem(last=False)
    return connection


def _load_table(connection: sqlite3.Connection, name: str, frame: Any) -> None:
    """Copy one DataFrame into the scratch database as a SQLite table."""
    # Defence in depth: a label column never reaches the scratch database,
    # even if a published table were to carry one.
    flat = frame.drop(
        columns=[c for c in frame.columns if hidden_labels.is_denied_column(c)]
    )
    for column in flat.columns:
        flat[column] = flat[column].map(_to_sqlite_value)
    quoted = ", ".join(f'"{c}"' for c in flat.columns)
    placeholders = ", ".join("?" for _ in flat.columns)
    connection.execute(f'CREATE TABLE "{name}" ({quoted})')
    connection.executemany(
        f'INSERT INTO "{name}" ({quoted}) VALUES ({placeholders})',
        flat.itertuples(index=False, name=None),
    )


def _to_sqlite_value(value: Any) -> Any:
    """Coerce a pandas cell to something SQLite can store.

    Lists become JSON so array columns stay queryable with ``LIKE``;
    ``Decimal`` amounts become strings so no cents are lost to a float;
    timestamps become ISO-8601 text so string comparison orders them
    correctly.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    # Sequences first: a numpy array is not None and comparing it to
    # itself yields an array, so it must never reach the NaN check.
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__len__") and not isinstance(value, (str, bytes))
    ):
        return json.dumps([_to_sqlite_value(item) for item in value])
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bytes)):
        return value
    if hasattr(value, "tolist"):  # numpy scalar
        return value.tolist()
    if value != value:  # NaN
        return None
    return str(value)
