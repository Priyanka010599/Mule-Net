"""Data access for the MuleNet agent tools.

Reads the real parquet tables when they are on disk and falls back to
mock tables built from the ``shared/`` models otherwise, so the agents
are never blocked on another module finishing. Switching to real data is
a config flip (``MULENET_ML_OUT_DIR``), not a rewrite.

A second config flip, ``MULENET_AGENTS_DATA_SOURCE=bigquery`` (default
``parquet``), reads the same tables from BigQuery instead -- additive,
tested (``agents/test_data_bigquery.py``), and **not used by any
deployed instance**; see ``docs/reviewer_notes.md``. ``cases`` is never
BigQuery-backed; see :data:`BIGQUERY_BACKED_TABLES`.

Two source directories:

``simulator/out``
    ``accounts``, ``transfers``, ``devices``, ``kyc_documents``,
    ``kyc_submissions``, ``ground_truth``. Written by ``simulator/``.
``ml/out``
    ``rings``, ``scores``, ``ring_explanations``, ``account_evidence``.
    Written by ``ml/``; see ``ml/out/manifest.json``. Each is mocked
    independently if it is missing, so a half-published ``ml/out`` still
    works -- **in ``mock`` mode only**. On the Vertex path a missing
    table raises :class:`MissingTableError`, because there a mock is not
    a convenience but a fluent wrong answer carrying real provenance.
    :func:`required_tables` and :meth:`DataStore.missing_tables` are what
    the HTTP service checks before it agrees to be ready, and they are
    derived from the SQL allowlist rather than typed out, so the
    readiness check cannot be narrower than the fallback.

As of this writing every table above is present and real; only ``cases``
is mocked, because no module produces it yet. Check
:meth:`DataStore.provenance` rather than assuming.

The mock fallbacks are deliberately *not* label oracles except for
``rings``:

* ``scores`` is computed from transfer velocity and structure only --
  pass-through ratio, fan-in, and cross-bank fan-out. No ground-truth
  column is read.
* ``account_evidence`` is computed from genuinely shared devices and KYC
  attributes. It is attached to a subject, never used to score it:
  identity reuse is a weak signal (thousands of legitimate accounts
  share documents) and must not drive a decision.
* ``rings`` *is* derived from ``ground_truth.ring_id`` because ring
  clustering is ``ml/``'s job. Any eval number measured against mock
  rings therefore bounds the agent plumbing, not detection quality.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.config import AgentsConfig
from shared.contracts import (
    Account,
    AccountEvidence,
    AccountExplanation,
    AccountStatus,
    EvidenceScope,
    ExplanationFactor,
    KycDocument,
    MembershipBasis,
    Ring,
    RingExplanation,
    RingMember,
    RingStatus,
    RiskScore,
    Transfer,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Tables expected under ``simulator/out``.
SIMULATOR_TABLES: dict[str, str] = {
    "accounts": "accounts.parquet",
    "transfers": "transfers.parquet",
    "devices": "devices.parquet",
    "kyc_documents": "kyc_documents.parquet",
    "kyc_submissions": "kyc_submissions.parquet",
    "ground_truth": "ground_truth.parquet",
}

#: Tables expected under ``ml/out``, named as ``ml/out/manifest.json``
#: names them. Mocked independently when absent.
ML_TABLES: dict[str, str] = {
    "rings": "rings.parquet",
    "scores": "scores.parquet",
    "ring_explanations": "ring_explanations.parquet",
    "account_explanations": "account_explanations.parquet",
    "account_evidence": "account_evidence.parquet",
}

#: ``ALLOWED_SQL_TABLES`` uses shorter names than ``ml/out`` does. Map the
#: allowlist spelling onto the real table so an analyst can write
#: ``FROM explanations`` without knowing the file layout.
TABLE_ALIASES: dict[str, str] = {
    "explanations": "ring_explanations",
}

#: Tables :meth:`DataStore.table` can load from BigQuery instead of
#: parquet, when ``MULENET_AGENTS_DATA_SOURCE=bigquery`` (see
#: ``agents/config.py``). Every one of these has a schema in
#: ``shared/bq_schemas.py`` and real published data loaded by
#: ``infra/load_bigquery.py`` -- verified live, see
#: ``docs/reviewer_notes.md``.
#:
#: ``cases`` is deliberately **not** here even though it also has a BQ
#: schema: it is written locally by this very service
#: (``agents/cases.py``) as investigations are verified, so reading it
#: back from a BigQuery snapshot would serve a stale copy of a table the
#: process itself just appended to. It stays on the parquet path in
#: every mode, same as today.
BIGQUERY_BACKED_TABLES: frozenset[str] = frozenset({*SIMULATOR_TABLES, *ML_TABLES})

#: Model version stamped on mock scores so real and mock rows are never
#: confused downstream.
MOCK_MODEL_VERSION = "mock-velocity-structure-0.1"

#: Last-resort timestamp for a mock evidence row whose accounts have no
#: KYC submission on file. Fixed, so mock tables are byte-stable.
_MOCK_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

#: A credit followed by a debit within this many seconds counts as
#: pass-through. Matches ``DEFAULT_BURST_GAP_SECONDS`` in ``ml/config.py``.
PASSTHROUGH_WINDOW_SECONDS = 1800


@dataclass(frozen=True)
class TableStatus:
    """Where one logical table came from."""

    name: str
    path: Path | None
    is_mock: bool


class MissingTableError(FileNotFoundError):
    """A table has no parquet and the mock fallback is not permitted.

    A ``FileNotFoundError`` subclass so the callers that already treat a
    missing table as a missing file keep working.
    """


def required_tables() -> list[str]:
    """Every table the tools can read and therefore must be able to reach.

    Derived, per CLAUDE.md, from its sources of truth rather than typed
    out: ``shared.tool_interfaces.ALLOWED_SQL_TABLES`` -- everything
    ``run_sql`` will accept -- unioned with :meth:`DataStore.known_tables`,
    which is everything the typed tool reads open. A table added to the
    allowlist is therefore checked for readiness without anyone
    remembering to add it here.
    """
    from shared.tool_interfaces import (  # noqa: PLC0415 - avoids a cycle
        ALLOWED_SQL_TABLES,
    )

    return sorted(set(ALLOWED_SQL_TABLES) | set(DataStore.known_tables()))


class DataStore:
    """Lazily-loaded access to every table the agent tools need.

    Loading is per-table and cached, so a query that only touches
    ``accounts`` never pays for the 570k-row transfers table.
    """

    def __init__(self, config: AgentsConfig | None = None) -> None:
        self.config = config or AgentsConfig.from_env()
        self._cache: dict[str, Any] = {}
        self._status: dict[str, TableStatus] = {}
        self._bq_client_obj: Any = None
        self._bq_table_names_cache: frozenset[str] | None = None

    # ---------------------------------------------------------------- tables

    def table(self, name: str) -> pd.DataFrame:
        """Return one logical table as a DataFrame, loading it on first use."""
        if name in self._cache:
            return self._cache[name]
        frame = self._load(name)
        self._cache[name] = frame
        return frame

    def status(self, name: str) -> TableStatus:
        """Return provenance for a table, loading it if necessary."""
        self.table(name)
        return self._status[name]

    def provenance(self) -> dict[str, bool]:
        """Map every known table to whether it is currently mocked."""
        return {name: self.status(name).is_mock for name in self.known_tables()}

    def source_path(self, name: str) -> Path:
        """Where ``name`` would be read from, without reading it.

        Follows aliases and derivations to the file that actually has to
        exist: ``explanations`` resolves to ``ring_explanations.parquet``
        and ``ring_members`` to ``rings.parquet``, because neither has a
        file of its own. Used by the readiness check, which must stat the
        tables rather than load them -- ``transfers`` alone is 570k rows.
        """
        if name in TABLE_ALIASES:
            return self.source_path(TABLE_ALIASES[name])
        if name == "ring_members":
            # Membership is published nested inside rings.parquet; see _load.
            return self.source_path("rings")
        bq_name = self._resolve_bigquery_table(name)
        if bq_name is not None:
            # A display-only stand-in, never read as a real filesystem
            # path: Path() normalises "//" to "/" (and to "\" on
            # Windows), so this is NOT parsed back out anywhere -- see
            # _table_reachable, which resolves BigQuery-backed tables
            # through _resolve_bigquery_table directly instead.
            return Path(
                f"bigquery://{self.config.bq_project}.{self.config.bq_dataset}/{bq_name}"
            )
        if name in SIMULATOR_TABLES:
            return self.config.data_dir / SIMULATOR_TABLES[name]
        if name in ML_TABLES:
            return self.config.ml_out_dir / ML_TABLES[name]
        if name == "cases":
            from agents.cases import cases_path  # noqa: PLC0415 - avoids a cycle

            return cases_path(self.config)
        raise KeyError(f"unknown table {name!r}; known: {self.known_tables()}")

    def _resolve_bigquery_table(self, name: str) -> str | None:
        """The real BigQuery table ``name`` would read from under the
        current config, or ``None`` when it would not: either
        :attr:`AgentsConfig.data_source` is not ``"bigquery"``, or
        ``name`` resolves to a table that is never BigQuery-backed
        (``cases``; see :data:`BIGQUERY_BACKED_TABLES`).
        """
        if self.config.data_source != "bigquery":
            return None
        if name in TABLE_ALIASES:
            return self._resolve_bigquery_table(TABLE_ALIASES[name])
        if name == "ring_members":
            return self._resolve_bigquery_table("rings")
        if name in BIGQUERY_BACKED_TABLES:
            return name
        return None

    def missing_tables(self) -> list[str]:
        """Names of :func:`required_tables` that cannot currently be read.

        Empty means every table the tools can reach is real. Anything in
        it would be served from a mock or an empty frame, which is why
        the HTTP service refuses to become ready while it is non-empty.

        On the parquet path (default) this stats a local file. On the
        BigQuery path it instead checks the table exists in
        ``bq_project.bq_dataset``.
        """
        return [name for name in required_tables() if not self._table_reachable(name)]

    def _table_reachable(self, name: str) -> bool:
        bq_name = self._resolve_bigquery_table(name)
        if bq_name is not None:
            return bq_name in self._bigquery_table_names()
        return self.source_path(name).is_file()

    def _bq_client(self) -> Any:
        if self._bq_client_obj is None:
            from google.cloud import bigquery  # noqa: PLC0415 - avoids a cycle

            self._bq_client_obj = bigquery.Client(project=self.config.bq_project)
        return self._bq_client_obj

    def _bigquery_table_names(self) -> frozenset[str]:
        """Table ids actually present in ``bq_project.bq_dataset``, cached.

        One ``list_tables`` call rather than one ``get_table`` per
        required table -- :meth:`missing_tables` is evaluated once at
        startup (``agents/service.py``), but there is no reason to make
        that one evaluation cost more RPCs than it needs to.
        """
        if self._bq_table_names_cache is None:
            dataset_ref = f"{self.config.bq_project}.{self.config.bq_dataset}"
            self._bq_table_names_cache = frozenset(
                t.table_id for t in self._bq_client().list_tables(dataset_ref)
            )
        return self._bq_table_names_cache

    def _bigquery_table_fingerprint(self, name: str) -> str:
        """A short string that changes whenever ``name`` is reloaded in
        BigQuery; see :meth:`data_version`."""
        table = self._bq_client().get_table(
            f"{self.config.bq_project}.{self.config.bq_dataset}.{name}"
        )
        return f"{table.etag}:{table.num_rows}"

    def data_version(
        self,
        names: tuple[str, ...] = (
            "accounts",
            "transfers",
            "devices",
            "kyc_submissions",
            "kyc_documents",
        ),
    ) -> str:
        """A short content hash of the named tables' source files.

        What a derived result (a cached feature build, a served artifact) is
        keyed by, so it can never outlive the data it was computed from: a
        re-published parquet changes the hash, and nothing keyed on the old
        one is reused. Computed once per process and cached; ``"mock"`` for
        a table with no file, so a mock run never shares a key with a real one.

        On the BigQuery path, hashing 572,890 ``transfers`` rows byte for
        byte on every cold start would cost a full table scan for the one
        property the parquet hash gets for free: changing when the data
        does. BigQuery's own ``etag`` already changes on every reload
        (``infra/load_bigquery.py`` TRUNCATEs and re-appends), so that
        plus the row count stands in for the file hash there.
        """
        import hashlib

        key = f"_data_version:{','.join(names)}"
        if key in self._cache:
            return self._cache[key]
        digest = hashlib.sha256()
        for name in names:
            digest.update(name.encode())
            bq_name = self._resolve_bigquery_table(name)
            if bq_name is not None:
                digest.update(self._bigquery_table_fingerprint(bq_name).encode())
                continue
            path = self.source_path(name)
            if not path.is_file():
                digest.update(b"mock")
                continue
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        self._cache[key] = digest.hexdigest()[:16]
        return self._cache[key]

    def _require_mock_allowed(self, name: str, path: Path) -> None:
        """Refuse a mock fallback unless the config permits one.

        The fallback is right locally -- it kept the agents working while
        ``ml/`` was still being written -- and dangerous deployed: Gemini
        answers fluently over invented numbers while the reply still says
        ``backend: "gemini"`` and carries real-looking ``sources``. A
        wrong number with authentic provenance is worse than an outage,
        because an outage is obvious. So on the Vertex path it is a hard
        error at the point of the read, not only at readiness: readiness
        is checked once at startup, and this closes the window where a
        table disappears afterwards or a caller builds its own store.
        """
        if self.config.allow_mock_tables:
            return
        raise MissingTableError(
            f"table {name!r} is missing at {path} and the mock fallback is "
            f"disabled in {self.config.mode!r} mode. Refusing to answer over "
            "invented numbers: ship the published parquet into the image, or "
            "set MULENET_AGENT_MODE=mock for local work."
        )

    @staticmethod
    def known_tables() -> list[str]:
        """Every table name :meth:`table` accepts."""
        return sorted({*SIMULATOR_TABLES, *ML_TABLES, *TABLE_ALIASES, "cases", "ring_members"})

    def _load(self, name: str) -> pd.DataFrame:
        import pandas as pd

        if name in TABLE_ALIASES:
            target = TABLE_ALIASES[name]
            frame = self.table(target)
            self._status[name] = self._status[target]
            return frame

        if name in SIMULATOR_TABLES:
            if self.config.data_source == "bigquery":
                frame = self._load_from_bigquery(name)
                self._status[name] = TableStatus(
                    name, self.source_path(name), is_mock=False
                )
                return frame
            path = self.config.data_dir / SIMULATOR_TABLES[name]
            if not path.is_file():
                raise FileNotFoundError(
                    f"simulator table {name!r} not found at {path}; "
                    "run simulator/generator.py first"
                )
            self._status[name] = TableStatus(name, path, is_mock=False)
            return pd.read_parquet(path)

        if name in ML_TABLES:
            if self.config.data_source == "bigquery":
                frame = self._load_from_bigquery(name)
                self._status[name] = TableStatus(
                    name, self.source_path(name), is_mock=False
                )
                return frame
            path = self.config.ml_out_dir / ML_TABLES[name]
            if path.is_file():
                self._status[name] = TableStatus(name, path, is_mock=False)
                return pd.read_parquet(path)
            self._require_mock_allowed(name, path)
            self._status[name] = TableStatus(name, None, is_mock=True)
            return _MOCK_BUILDERS[name](self)

        if name == "ring_members":
            # Allowlisted for SQL and specified in shared/ddl.sql as a
            # child table, but ml/ publishes membership nested inside
            # rings.parquet. Exploded here so `FROM ring_members` works
            # rather than passing validation and then failing to execute.
            rings = self.table("rings")
            self._status[name] = TableStatus(
                name, self._status["rings"].path, self._status["rings"].is_mock
            )
            return pd.DataFrame.from_records(
                [
                    {"ring_id": str(row.ring_id), **dict(member)}
                    for row in rings.itertuples()
                    for member in list(row.members)
                ],
                columns=[
                    "ring_id",
                    "hashed_account_id",
                    "membership_basis",
                    "alertable",
                    "link_scope",
                    "linked_from_hashed_account_id",
                    "link_evidence_type",
                    "reason",
                ],
            )

        if name == "cases":
            # Produced by agents/cases.py from verified investigations.
            # Falls back to a correctly shaped empty frame so `FROM cases`
            # returns no rows rather than failing on a missing column.
            from agents.cases import (  # noqa: PLC0415 - avoids a cycle
                cases_path,
                empty_cases_frame,
            )

            path = cases_path(self.config)
            if path.is_file():
                self._status[name] = TableStatus(name, path, is_mock=False)
                return pd.read_parquet(path)
            self._require_mock_allowed(name, path)
            self._status[name] = TableStatus(name, None, is_mock=True)
            return empty_cases_frame()

        raise KeyError(f"unknown table {name!r}; known: {self.known_tables()}")

    def _load_from_bigquery(self, name: str) -> pd.DataFrame:
        """Load one table from BigQuery, shaped exactly like
        ``pd.read_parquet`` shapes the same table.

        Column order comes from ``shared.bq_schemas.TABLE_SCHEMAS`` -- the
        same contract ``infra/load_bigquery.py`` loaded the table
        against -- so ``SELECT <those columns>`` reproduces the parquet
        column order exactly; nothing here guesses at a column list.

        The one dtype difference measured live (see
        ``agents/test_data_bigquery.py``): BigQuery's pandas conversion
        gives nullable extension dtypes (``boolean``, ``Int64``) for
        BOOL/INT64 columns, while ``pd.read_parquet`` gives the plain
        numpy dtype whenever the contract says the column is REQUIRED and
        therefore never null. STRING, TIMESTAMP and NUMERIC already come
        back byte-for-byte identical to the parquet read and need no
        cast. A NULLABLE BOOL/INT64 column would still need the nullable
        dtype to represent an actual null, so the cast below is scoped to
        REQUIRED fields only -- none of the tables this reads have a
        nullable column of either type today (``features`` does, but it
        is intentionally not in :data:`BIGQUERY_BACKED_TABLES`).
        """
        from shared.bq_schemas import TABLE_SCHEMAS  # noqa: PLC0415 - avoids a cycle

        schema = TABLE_SCHEMAS[name]
        columns = [field.name for field in schema]
        table_ref = f"{self.config.bq_project}.{self.config.bq_dataset}.{name}"
        query = f"SELECT {', '.join(columns)} FROM `{table_ref}`"
        frame = self._bq_client().query(query).to_dataframe()
        for field in schema:
            if field.mode != "REQUIRED":
                continue
            if field.field_type == "BOOL":
                frame[field.name] = frame[field.name].astype(bool)
            elif field.field_type in ("INT64", "INTEGER"):
                frame[field.name] = frame[field.name].astype("int64")
        return frame

    # ------------------------------------------------------------ typed reads

    def account(self, hashed_account_id: str) -> Account:
        """Return one account as a ``shared.contracts.Account``."""
        rows = self.table("accounts")
        hit = rows[rows["hashed_account_id"] == hashed_account_id]
        if hit.empty:
            raise KeyError(f"unknown account {hashed_account_id!r}")
        return _to_account(next(hit.itertuples()))

    def accounts(self, hashed_account_ids: list[str]) -> list[Account]:
        """Return several accounts, skipping ids that do not exist."""
        rows = self.table("accounts")
        hit = rows[rows["hashed_account_id"].isin(set(hashed_account_ids))]
        return [_to_account(row) for row in hit.itertuples()]

    def ring(self, ring_id: str) -> Ring:
        """Return one ring as a ``shared.contracts.Ring``."""
        rows = self.table("rings")
        hit = rows[rows["ring_id"] == ring_id]
        if hit.empty:
            raise KeyError(f"unknown ring {ring_id!r}")
        row = hit.iloc[0]
        return Ring(
            ring_id=str(row["ring_id"]),
            # Members arrive as a numpy array of dicts; each is validated
            # through RingMember so the basis cannot detach from the
            # account it describes.
            members=[
                RingMember.model_validate(dict(member))
                for member in list(row["members"])
            ],
            banks_involved=list(row["banks_involved"]),
            detected_at=row["detected_at"].to_pydatetime(),
            status=RingStatus(row["status"]),
        )

    def risk_score(self, hashed_account_id: str) -> RiskScore | None:
        """Return the latest risk score for an account, or ``None``."""
        rows = self.table("scores")
        hit = rows[rows["hashed_account_id"] == hashed_account_id]
        if hit.empty:
            return None
        row = hit.sort_values("scored_at").iloc[-1]
        return RiskScore(
            hashed_account_id=str(row["hashed_account_id"]),
            score=float(row["score"]),
            model_version=str(row["model_version"]),
            scored_at=row["scored_at"].to_pydatetime(),
            top_factors=list(row["top_factors"]),
        )

    def ring_explanation(self, ring_id: str) -> RingExplanation:
        """Return the stored explanation for a ring."""
        rows = self.table("ring_explanations")
        hit = rows[rows["ring_id"] == ring_id]
        if hit.empty:
            raise KeyError(f"no explanation for ring {ring_id!r}")
        row = hit.iloc[0]
        return RingExplanation(
            ring_id=ring_id,
            top_factors=list(row["top_factors"]),
            narrative=str(row["narrative"]),
            generated_at=row["generated_at"].to_pydatetime(),
        )

    def account_explanation(self, hashed_account_id: str) -> AccountExplanation:
        """Return the stored per-account explanation.

        ``score`` and ``model_version`` come from the explanation row
        itself rather than from ``scores``, because an explanation shown
        beside a score it was not computed from is a wrong explanation.
        """
        rows = self.table("account_explanations")
        hit = rows[rows["hashed_account_id"] == hashed_account_id]
        if hit.empty:
            raise KeyError(f"no explanation for account {hashed_account_id!r}")
        row = hit.iloc[0]
        return AccountExplanation(
            hashed_account_id=str(row["hashed_account_id"]),
            score=float(row["score"]),
            model_version=str(row["model_version"]),
            top_factors=[
                ExplanationFactor.model_validate(dict(factor))
                for factor in list(row["top_factors"])
            ],
            narrative=str(row["narrative"]),
            generated_at=row["generated_at"].to_pydatetime(),
        )

    def rings_for_account(self, hashed_account_id: str) -> list[str]:
        """Return the ids of every ring the account belongs to.

        Membership uses *all* members, so a context member is not told
        it is outside the ring it is in. What the membership means is a
        separate question: see :meth:`ring_membership`.
        """
        rows = self.table("rings")
        return sorted(
            str(row.ring_id)
            for row in rows.itertuples()
            if any(
                member["hashed_account_id"] == hashed_account_id
                for member in list(row.members)
            )
        )

    def ring_membership(
        self, ring_id: str, hashed_account_id: str
    ) -> RingMember | None:
        """Return how one account came to be in a ring, or ``None``.

        Callers that describe an account must read this rather than
        assume: a ``velocity`` member is a finding about its own
        behaviour, an ``identity_link`` member is only a lead.
        """
        for member in self.ring(ring_id).members:
            if member.hashed_account_id == hashed_account_id:
                return member
        return None

    def kyc_documents(self, hashed_account_id: str) -> list[KycDocument]:
        """Return the KYC documents filed for one account."""
        rows = self.table("kyc_documents")
        hit = rows[rows["hashed_account_id"] == hashed_account_id]
        return [_to_kyc_document(row) for row in hit.itertuples()]

    def account_evidence(self, hashed_account_id: str) -> list[AccountEvidence]:
        """Return identity evidence rows attached to an account.

        Evidence corroborates a finding that velocity and structure
        already support. It is never a ground of suspicion on its own,
        and a ``single_bank`` row is weaker still. Every row is parsed
        through the shared contract, so a malformed parquet row fails
        here rather than reaching a report.
        """
        rows = self.table("account_evidence")
        hit = rows[rows["hashed_account_id"] == hashed_account_id]
        return [to_account_evidence(row) for row in hit.to_dict(orient="records")]

    def bank_index(self) -> dict[str, str]:
        """Map every hashed account id to its bank id, cached."""
        cached = self._cache.get("_bank_index")
        if cached is None:
            accounts = self.table("accounts")
            cached = dict(
                zip(accounts["hashed_account_id"], accounts["bank_id"], strict=True)
            )
            self._cache["_bank_index"] = cached
        return cached

    def submission_times(self) -> dict[str, datetime]:
        """Map every hashed account id to its KYC submission time, cached."""
        cached = self._cache.get("_submission_times")
        if cached is None:
            subs = self.table("kyc_submissions")
            cached = {
                row.hashed_account_id: row.submitted_at.to_pydatetime()
                for row in subs.itertuples()
            }
            self._cache["_submission_times"] = cached
        return cached

    def transfers_frame(self) -> pd.DataFrame:
        """Return the transfers table sorted by timestamp, cached."""
        cached = self._cache.get("_transfers_sorted")
        if cached is None:
            cached = self.table("transfers").sort_values("timestamp", kind="stable")
            self._cache["_transfers_sorted"] = cached
        return cached


# ------------------------------------------------------------ row conversion


def _to_account(row: Any) -> Account:
    return Account(
        hashed_account_id=str(row.hashed_account_id),
        bank_id=str(row.bank_id),
        opened_at=row.opened_at.to_pydatetime(),
        account_type=str(row.account_type),
        status=AccountStatus(row.status),
    )


def _to_kyc_document(row: Any) -> KycDocument:
    ref = getattr(row, "photo_embedding_ref", None)
    if ref is None or ref != ref:  # NaN is the only value unequal to itself
        ref = None
    return KycDocument(
        doc_id=str(row.doc_id),
        hashed_account_id=str(row.hashed_account_id),
        doc_type=str(row.doc_type),
        hashed_id_number=str(row.hashed_id_number),
        normalized_address=str(row.normalized_address),
        photo_embedding_ref=ref,
    )


def to_account_evidence(row: dict[str, Any]) -> AccountEvidence:
    """Parse one evidence parquet row through the shared contract."""
    return AccountEvidence(
        hashed_account_id=str(row["hashed_account_id"]),
        evidence_type=str(row["evidence_type"]),
        scope=EvidenceScope(row["scope"]),
        detail=str(row["detail"]),
        counterparty_hashed_account_ids=list(row["counterparty_hashed_account_ids"]),
        counterparty_bank_ids=list(row["counterparty_bank_ids"]),
        observed_at=row["observed_at"].to_pydatetime(),
        source_table=str(row["source_table"]),
        detector_version=(
            str(row["detector_version"]) if row.get("detector_version") else None
        ),
    )


def to_transfer(row: Any) -> Transfer:
    """Convert a transfers DataFrame row to a ``shared.contracts.Transfer``."""
    return Transfer(
        transfer_id=str(row.transfer_id),
        from_hashed_account_id=str(row.from_hashed_account_id),
        to_hashed_account_id=str(row.to_hashed_account_id),
        amount=row.amount,
        currency=str(row.currency),
        timestamp=row.timestamp.to_pydatetime(),
        channel=str(row.channel),
    )


# --------------------------------------------------------------- mock builders


def _mock_rings(store: DataStore) -> pd.DataFrame:
    """Stand in for ``ml/out/rings.parquet`` using ground-truth clusters."""
    import pandas as pd

    gt = store.table("ground_truth")
    labelled = gt[gt["ring_id"].astype("string").fillna("") != ""]
    records = [
        {
            "ring_id": str(ring_id),
            # Every mock member is a velocity member: stage-2 identity
            # propagation is ml/'s work, and inventing identity links
            # here would fabricate leads against real accounts. Built
            # through RingMember so a mock ring cannot differ in shape
            # from a published one.
            "members": [
                RingMember(
                    hashed_account_id=account,
                    membership_basis=MembershipBasis.VELOCITY,
                    alertable=True,
                ).model_dump(mode="python")
                for account in sorted(group["hashed_account_id"].tolist())
            ],
            "banks_involved": sorted(set(group["bank_id"].tolist())),
            "detected_at": group["opened_at"].max(),
            "status": RingStatus.CANDIDATE.value,
        }
        for ring_id, group in labelled.groupby("ring_id", sort=True)
    ]
    return pd.DataFrame.from_records(
        records,
        columns=["ring_id", "members", "banks_involved", "detected_at", "status"],
    )


def _passthrough_counts(transfers: pd.DataFrame) -> pd.Series:
    """Per-account count of credits followed by a debit inside the window.

    One ``merge_asof`` per direction: for every credit, find that
    account's next debit and keep the pair if the gap is within
    :data:`PASSTHROUGH_WINDOW_SECONDS`.
    """
    import pandas as pd

    credits = (
        transfers[["to_hashed_account_id", "timestamp"]]
        .rename(columns={"to_hashed_account_id": "acct", "timestamp": "credit_at"})
        .sort_values("credit_at", kind="stable")
    )
    debits = (
        transfers[["from_hashed_account_id", "timestamp"]]
        .rename(columns={"from_hashed_account_id": "acct", "timestamp": "debit_at"})
        .sort_values("debit_at", kind="stable")
    )
    paired = pd.merge_asof(
        credits,
        debits,
        left_on="credit_at",
        right_on="debit_at",
        by="acct",
        direction="forward",
        tolerance=pd.Timedelta(seconds=PASSTHROUGH_WINDOW_SECONDS),
    )
    matched = paired.dropna(subset=["debit_at"])
    return matched.groupby("acct").size()


def _velocity_structure_features(store: DataStore) -> pd.DataFrame:
    """Per-account velocity and structure counts, computed without labels."""
    transfers = store.transfers_frame()
    accounts = store.table("accounts")[["hashed_account_id", "bank_id"]]
    bank_of = dict(zip(accounts["hashed_account_id"], accounts["bank_id"], strict=True))

    labelled = transfers.assign(
        src_bank=transfers["from_hashed_account_id"].map(bank_of),
        dst_bank=transfers["to_hashed_account_id"].map(bank_of),
    )
    cross = labelled[labelled["src_bank"] != labelled["dst_bank"]]

    feat = accounts.set_index("hashed_account_id").join(
        [
            transfers.groupby("to_hashed_account_id").size().rename("in_count"),
            transfers.groupby("from_hashed_account_id").size().rename("out_count"),
            transfers.groupby("to_hashed_account_id")["from_hashed_account_id"]
            .nunique()
            .rename("distinct_senders"),
            transfers.groupby("from_hashed_account_id")["to_hashed_account_id"]
            .nunique()
            .rename("distinct_recipients"),
            cross.groupby("from_hashed_account_id").size().rename("cross_bank_out"),
            _passthrough_counts(transfers).rename("passthrough_count"),
        ]
    )
    counts = [
        "in_count",
        "out_count",
        "distinct_senders",
        "distinct_recipients",
        "cross_bank_out",
        "passthrough_count",
    ]
    feat[counts] = feat[counts].fillna(0).astype(float)
    feat["passthrough_ratio"] = feat["passthrough_count"] / feat["in_count"].clip(lower=1)
    return feat.reset_index()


def _mock_scores(store: DataStore) -> pd.DataFrame:
    """Stand in for ``ml/out/scores.parquet``. Velocity + structure only."""
    import numpy as np
    import pandas as pd

    feat = _velocity_structure_features(store)
    logit = (
        2.2 * feat["passthrough_ratio"]
        + 0.35 * np.log1p(feat["distinct_senders"])
        + 0.30 * np.log1p(feat["cross_bank_out"])
        + 0.10 * np.log1p(feat["out_count"])
        - 3.0
    )
    feat["score"] = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "hashed_account_id": feat["hashed_account_id"],
            "score": feat["score"].astype(float),
            "model_version": MOCK_MODEL_VERSION,
            "scored_at": pd.Timestamp("2026-01-01", tz="UTC"),
            "top_factors": [_score_factors(row) for row in feat.itertuples()],
        }
    )


def _score_factors(row: Any) -> list[str]:
    """Name the velocity and structure factors that pushed a score up.

    Identity reuse never appears here: it is evidence, not a driver.
    """
    factors: list[str] = []
    if row.passthrough_ratio >= 0.5:
        factors.append(f"pass-through ratio {row.passthrough_ratio:.2f}")
    if row.distinct_senders >= 5:
        factors.append(f"fan-in from {int(row.distinct_senders)} distinct senders")
    if row.cross_bank_out >= 3:
        factors.append(f"{int(row.cross_bank_out)} cross-bank outbound transfers")
    if row.out_count >= 20:
        factors.append(f"{int(row.out_count)} outbound transfers")
    return factors or ["no velocity or structure factor above threshold"]


def _mock_ring_explanations(store: DataStore) -> pd.DataFrame:
    """Stand in for ``ml/out/ring_explanations.parquet``.

    Columns match ``shared.contracts.RingExplanation`` exactly, as the
    real table does: ``ring_id``, ``top_factors``, ``narrative``,
    ``generated_at``.
    """
    import pandas as pd

    rings = store.table("rings")
    scores = store.table("scores").set_index("hashed_account_id")
    records = []
    for row in rings.itertuples():
        # The frame carries member structs, not a flat list of ids.
        records_in = [dict(member) for member in list(row.members)]
        members = [member["hashed_account_id"] for member in records_in]
        alertable = [
            member["hashed_account_id"] for member in records_in if member["alertable"]
        ]
        member_scores = scores.reindex(alertable or members)["score"].dropna()
        factors: list[str] = []
        for member in members:
            if member in scores.index:
                factors.extend(scores.loc[member, "top_factors"])
        top = list(dict.fromkeys(factors))[:5] or [
            "no velocity or structure factor above threshold"
        ]
        mean_score = float(member_scores.mean()) if len(member_scores) else 0.0
        records.append(
            {
                "ring_id": str(row.ring_id),
                "top_factors": top,
                # The bare count is the alertable one. The total appears
                # only in the clause that says what the extra members are.
                "narrative": (
                    f"Ring {row.ring_id} groups {len(alertable)} alertable "
                    f"accounts across {len(row.banks_involved)} bank(s) "
                    f"({', '.join(row.banks_involved)})"
                    + (
                        f", plus {len(members) - len(alertable)} same-bank "
                        "context member(s) linked for review and not alerted on"
                        if len(members) > len(alertable)
                        else ""
                    )
                    + f". Mean alertable member risk score {mean_score:.3f}. "
                    "Flagged on transfer velocity and graph structure; any "
                    "shared identity attributes are attached as corroborating "
                    "evidence only."
                ),
                "generated_at": pd.Timestamp("2026-01-01", tz="UTC"),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["ring_id", "top_factors", "narrative", "generated_at"]
    )


def _mock_account_evidence(store: DataStore) -> pd.DataFrame:
    """Stand in for ``ml/out/account_evidence.parquet``.

    Columns match ``ml.publish.AccountEvidence``. Only *cross-bank*
    reuse is emitted, as the real table does: within one bank a shared
    document is far too common to be worth citing.
    """
    import pandas as pd

    frames = [
        _shared_groups(store, store.table("devices"), "device_fingerprint",
                       "cross_bank_shared_device", "devices"),
        _shared_groups(store, store.table("kyc_documents"), "hashed_id_number",
                       "cross_bank_reused_id_number", "kyc_documents"),
        _shared_groups(store, store.table("kyc_documents"), "photo_embedding_ref",
                       "cross_bank_reused_photo", "kyc_documents"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["hashed_account_id", "evidence_type"], kind="stable"
    ).reset_index(drop=True)


def _shared_groups(
    store: DataStore,
    frame: pd.DataFrame,
    column: str,
    evidence_type: str,
    source_table: str,
) -> pd.DataFrame:
    """One AccountEvidence row per account sharing ``column`` across banks."""
    import pandas as pd

    accounts = store.table("accounts")[["hashed_account_id", "bank_id"]]
    bank_of = dict(zip(accounts["hashed_account_id"], accounts["bank_id"], strict=True))

    sub = frame[["hashed_account_id", column]].dropna().drop_duplicates()
    sub = sub.assign(bank_id=sub["hashed_account_id"].map(bank_of)).dropna()

    submitted = store.submission_times()
    records = []
    for value, group in sub.groupby(column, sort=True):
        if group["bank_id"].nunique() < 2:
            continue
        for row in group.itertuples():
            others = group[group["hashed_account_id"] != row.hashed_account_id]
            cross = others[others["bank_id"] != row.bank_id]
            if cross.empty:
                continue
            banks = sorted(set(cross["bank_id"]))
            counterparties = sorted(cross["hashed_account_id"].tolist())
            # The later of the two sightings: the instant the link became
            # citable. Required by the contract and never defaulted to now.
            seen = [
                submitted[a]
                for a in [row.hashed_account_id, *counterparties]
                if a in submitted
            ]
            # Built through the shared model so a mock row cannot differ in
            # shape from a real one.
            evidence = AccountEvidence(
                hashed_account_id=row.hashed_account_id,
                evidence_type=evidence_type,
                scope=EvidenceScope.CROSS_BANK,
                detail=(
                    f"{column} {_short_hash(value)} is also held by "
                    f"{len(cross)} account(s) at {', '.join(banks)}. "
                    "Shared identifiers also arise legitimately; this "
                    "corroborates and does not by itself establish suspicion."
                ),
                counterparty_hashed_account_ids=counterparties,
                counterparty_bank_ids=banks,
                observed_at=max(seen) if seen else _MOCK_EPOCH,
                source_table=source_table,
                detector_version=MOCK_MODEL_VERSION,
            )
            records.append(evidence.model_dump(mode="python"))
    return pd.DataFrame.from_records(
        records,
        columns=[
            "hashed_account_id",
            "evidence_type",
            "scope",
            "detail",
            "counterparty_hashed_account_ids",
            "counterparty_bank_ids",
            "observed_at",
            "source_table",
            "detector_version",
        ],
    )


def _short_hash(value: object) -> str:
    """Stable 16-hex-char digest, so no raw attribute value is exposed."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


_MOCK_BUILDERS = {
    "rings": _mock_rings,
    "scores": _mock_scores,
    "ring_explanations": _mock_ring_explanations,
    "account_evidence": _mock_account_evidence,
}


@functools.lru_cache(maxsize=1)
def default_store() -> DataStore:
    """Process-wide default store, so tools share one set of loaded tables."""
    return DataStore()
