"""BigQuery table schemas for MuleNet.

Mirrors the Pydantic models in :mod:`shared.contracts`. Every module
(``ingest/``, ``ml/``, ``docai/``, ``ui/``) must import these schemas
rather than redefining BigQuery tables locally (see CLAUDE.md).

Account identifiers are always hashed; no raw identifiers appear here.
"""

from __future__ import annotations

from google.cloud.bigquery import SchemaField

#: Analytical copy of :class:`shared.contracts.Account`.
ACCOUNTS_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("bank_id", "STRING", mode="REQUIRED"),
    SchemaField("opened_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("account_type", "STRING", mode="REQUIRED"),
    SchemaField("status", "STRING", mode="REQUIRED"),
]

#: Analytical copy of :class:`shared.contracts.Device`.
DEVICES_SCHEMA: list[SchemaField] = [
    SchemaField("device_id", "STRING", mode="REQUIRED"),
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("device_fingerprint", "STRING", mode="REQUIRED"),
    SchemaField("first_seen", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("last_seen", "TIMESTAMP", mode="REQUIRED"),
]

#: Analytical copy of :class:`shared.contracts.KycDocument`.
#:
#: ``photo_embedding_ref`` is NULLABLE, matching ``Optional[str] = None``
#: on the contract: most accounts have no rendered image (see
#: ``simulator/README.md``), so most rows carry a null here.
KYC_DOCUMENTS_SCHEMA: list[SchemaField] = [
    SchemaField("doc_id", "STRING", mode="REQUIRED"),
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("doc_type", "STRING", mode="REQUIRED"),
    SchemaField("hashed_id_number", "STRING", mode="REQUIRED"),
    SchemaField("normalized_address", "STRING", mode="REQUIRED"),
    SchemaField("photo_embedding_ref", "STRING", mode="NULLABLE"),
]

#: Analytical copy of :class:`shared.contracts.KycSubmission`, published in
#: its ``simulator/out/kyc_submissions.parquet`` shape: ``doc_ids`` is a
#: repeated id column rather than nested ``documents``, matching the file
#: on disk; join to ``kyc_documents`` on ``doc_id`` to rebuild the nested
#: form (see ``simulator/README.md``).
KYC_SUBMISSIONS_SCHEMA: list[SchemaField] = [
    SchemaField("submission_id", "STRING", mode="REQUIRED"),
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("bank_id", "STRING", mode="REQUIRED"),
    SchemaField("submitted_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("doc_ids", "STRING", mode="REPEATED"),
]

#: Analytical copy of ``simulator/out/ground_truth.parquet``. No pydantic
#: model backs this one -- it is the simulator's hidden label table, not a
#: contract any production module is meant to read (see
#: ``agents/hidden_labels.py``: ``ground_truth`` is denied to every agent
#: tool regardless of this schema existing). ``ring_id``, ``role`` and
#: ``pattern_type`` are REQUIRED, not NULLABLE: a non-mule row carries ``""``
#: rather than null, matching the published Parquet exactly (zero nulls in
#: any column) and the comparison ``agents/data.py`` already does against
#: these columns.
GROUND_TRUTH_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("bank_id", "STRING", mode="REQUIRED"),
    SchemaField("is_mule", "BOOL", mode="REQUIRED"),
    SchemaField("ring_id", "STRING", mode="REQUIRED"),
    SchemaField("role", "STRING", mode="REQUIRED"),
    SchemaField("pattern_type", "STRING", mode="REQUIRED"),
    SchemaField("cohort", "STRING", mode="REQUIRED"),
    SchemaField("is_hard_negative", "BOOL", mode="REQUIRED"),
    SchemaField("ring_bank_count", "INT64", mode="REQUIRED"),
    SchemaField("opened_at", "TIMESTAMP", mode="REQUIRED"),
]

#: Analytical copy of :class:`shared.contracts.Transfer`.
TRANSFERS_SCHEMA: list[SchemaField] = [
    SchemaField("transfer_id", "STRING", mode="REQUIRED"),
    SchemaField("from_hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("to_hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("amount", "NUMERIC", mode="REQUIRED"),
    SchemaField("currency", "STRING", mode="REQUIRED"),
    SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("channel", "STRING", mode="REQUIRED"),
]

#: Feature store for ``ml/``. One row per (hashed_account_id, as_of, scope).
#: ``scope`` distinguishes features computed from a single bank's own
#: transactions from features computed over the merged cross-bank graph;
#: that comparison is the project's headline result.
FEATURES_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    SchemaField(
        "scope",
        "STRING",
        mode="REQUIRED",
        description="single_bank or cross_bank",
    ),
    SchemaField("in_degree", "INT64", mode="NULLABLE"),
    SchemaField("out_degree", "INT64", mode="NULLABLE"),
    SchemaField("unique_counterparties", "INT64", mode="NULLABLE"),
    SchemaField("distinct_counterparty_banks", "INT64", mode="NULLABLE"),
    SchemaField(
        "passthrough_ratio",
        "FLOAT64",
        mode="NULLABLE",
        description="amount_out_total / amount_in_total over the window",
    ),
    SchemaField(
        "passthrough_speed_seconds",
        "FLOAT64",
        mode="NULLABLE",
        description="median seconds between a credit and the matching debit",
    ),
    SchemaField("account_age_days", "FLOAT64", mode="NULLABLE"),
    SchemaField(
        "shared_device_count",
        "INT64",
        mode="NULLABLE",
        description="accounts sharing a device fingerprint with this account",
    ),
    SchemaField("amount_in_total", "NUMERIC", mode="NULLABLE"),
    SchemaField("amount_out_total", "NUMERIC", mode="NULLABLE"),
    SchemaField(
        "round_amount_ratio",
        "FLOAT64",
        mode="NULLABLE",
        description="share of transfers whose amount is a round number",
    ),
    SchemaField(
        "document_reuse_count",
        "INT64",
        mode="NULLABLE",
        description="KYC documents from docai/ reused across accounts",
    ),
]

#: Union of :class:`shared.contracts.RiskScore` and
#: :class:`shared.contracts.GnnScore`; ``model_type`` says which produced it.
SCORES_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField("score", "FLOAT64", mode="REQUIRED"),
    SchemaField("model_version", "STRING", mode="REQUIRED"),
    SchemaField(
        "model_type",
        "STRING",
        mode="REQUIRED",
        description="xgboost or gnn",
    ),
    SchemaField("scored_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField("top_factors", "STRING", mode="REPEATED"),
    SchemaField(
        "embedding_ref",
        "STRING",
        mode="NULLABLE",
        description="GnnScore.embedding_ref; null for the XGBoost baseline",
    ),
]

#: Analytical copy of :class:`shared.contracts.Ring`.
#:
#: ``members`` is a repeated RECORD, not the repeated STRING
#: ``member_hashed_account_ids`` it replaces — that column is gone. A
#: member now arrives by one of two routes and the route is part of the
#: claim: ``velocity`` means the account itself forwarded money fast in
#: a cross-bank structure, ``identity_link`` means only that it shares a
#: device or hashed id number with an account that did. A bare string
#: cannot carry that, and a parallel repeated STRING of bases would be
#: index-aligned by convention only — a misalignment there labels a
#: person a mule. Same pattern as ``account_explanations.top_factors``.
#:
#: ``alertable`` is materialised rather than derived. Stage-2 alerting
#: is cross-bank only; deriving that in SQL means joining
#: ``account_evidence`` and comparing banks, which is the derivation a
#: query will get wrong. An aggregate that counts ``members`` without
#: filtering on ``alertable`` is reporting a wrong alert count.
RINGS_SCHEMA: list[SchemaField] = [
    SchemaField("ring_id", "STRING", mode="REQUIRED"),
    SchemaField(
        "members",
        "RECORD",
        mode="REPEATED",
        fields=[
            SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
            SchemaField(
                "membership_basis",
                "STRING",
                mode="REQUIRED",
                description="velocity (stage-1 detection) or identity_link (stage-2 lead)",
            ),
            SchemaField(
                "alertable",
                "BOOL",
                mode="REQUIRED",
                description=(
                    "may be published as an alert: always true for velocity, true "
                    "for identity_link only when link_scope is cross_bank"
                ),
            ),
            SchemaField(
                "link_scope",
                "STRING",
                mode="NULLABLE",
                description="single_bank or cross_bank; null for a velocity member",
            ),
            SchemaField(
                "linked_from_hashed_account_id",
                "STRING",
                mode="NULLABLE",
                description="ring member at the other end of the identity edge",
            ),
            SchemaField(
                "link_evidence_type",
                "STRING",
                mode="NULLABLE",
                description="joins to account_evidence.evidence_type for the citable row",
            ),
            SchemaField(
                "reason",
                "STRING",
                mode="NULLABLE",
                description=(
                    "analyst-readable reason, required for identity_link; must read "
                    "as a lead ('linked, needs review'), never as a finding"
                ),
            ),
        ],
    ),
    SchemaField("banks_involved", "STRING", mode="REPEATED"),
    SchemaField("detected_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField(
        "status",
        "STRING",
        mode="REQUIRED",
        description="candidate, confirmed or dismissed",
    ),
]

#: Analytical copy of :class:`shared.contracts.RingExplanation`.
#:
#: Ring and account explanations are two tables, not one polymorphic
#: ``explanations`` table with a ``subject_type`` discriminator. They
#: have different keys (``ring_id`` against ``hashed_account_id``) and
#: different payloads, so a single table forces every column NULLABLE
#: and makes every correct query carry a ``WHERE subject_type = ...``
#: that is silently wrong when forgotten.
RING_EXPLANATIONS_SCHEMA: list[SchemaField] = [
    SchemaField("ring_id", "STRING", mode="REQUIRED"),
    SchemaField("top_factors", "STRING", mode="REPEATED"),
    SchemaField("narrative", "STRING", mode="REQUIRED"),
    SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
]

#: Analytical copy of :class:`shared.contracts.AccountExplanation`.
#:
#: ``top_factors`` is a repeated RECORD rather than a repeated STRING:
#: an analyst needs the *direction* of each factor, and a bare string
#: cannot carry the sign of the contribution. ``description`` is the
#: analyst-facing text; ``feature_name`` is the raw column, kept for
#: reproducibility and not for display.
ACCOUNT_EXPLANATIONS_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField(
        "score",
        "FLOAT64",
        mode="REQUIRED",
        description="the score this explanation explains",
    ),
    SchemaField(
        "model_version",
        "STRING",
        mode="REQUIRED",
        description="must match scores.model_version for the same account",
    ),
    SchemaField(
        "top_factors",
        "RECORD",
        mode="REPEATED",
        fields=[
            SchemaField(
                "description",
                "STRING",
                mode="REQUIRED",
                description="human-meaningful reason, never a raw column name",
            ),
            SchemaField(
                "contribution",
                "FLOAT64",
                mode="REQUIRED",
                description="signed contribution to the score; sign is load-bearing",
            ),
            SchemaField(
                "feature_name",
                "STRING",
                mode="NULLABLE",
                description="underlying feature column; null for rule-derived factors",
            ),
        ],
    ),
    SchemaField("narrative", "STRING", mode="REQUIRED"),
    SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
]

#: Analytical copy of :class:`shared.contracts.Investigation`.
#:
#: ``cases`` has a table in ``shared/ddl.sql`` *and* a schema here, the
#: same dual role ``str_reports`` has, and for the same reason: the row
#: is operational (an analyst moves ``status`` and sets ``closed_at``)
#: but the table is also read as an aggregate. The questions it answers
#: — how many detected rings ended in a verified case, and what share of
#: those passed on the first attempt rather than the retry — are scans
#: over every row, which is what BigQuery is for. Until now ``cases``
#: was in :data:`shared.tool_interfaces.ALLOWED_SQL_TABLES` with no
#: schema here, so the one tool allowed to query it had nothing to query.
#:
#: The three verification columns are ``REQUIRED``: a case exists only
#: where a narrative was re-derived from tool data and matched, and an
#: exported row must carry that fact without consulting another source
#: — the same rule as ``str_reports.citations_verified``.
CASES_SCHEMA: list[SchemaField] = [
    SchemaField("case_id", "STRING", mode="REQUIRED"),
    SchemaField(
        "ring_id",
        "STRING",
        mode="REQUIRED",
        description="unique: one case per ring; case_id is derived from it",
    ),
    SchemaField(
        "verified_at",
        "TIMESTAMP",
        mode="REQUIRED",
        description="when the verifier passed the investigation this case rests on",
    ),
    SchemaField(
        "verification_attempts",
        "INT64",
        mode="REQUIRED",
        description="1 (passed first draft) or 2 (passed on the single retry)",
    ),
    SchemaField(
        "verified_claim_count",
        "INT64",
        mode="REQUIRED",
        description="claims the verifier re-derived and confirmed; always at least 1",
    ),
    SchemaField("analyst_notes", "STRING", mode="REQUIRED"),
    SchemaField(
        "status",
        "STRING",
        mode="REQUIRED",
        description="open, in_review or closed",
    ),
    SchemaField("opened_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField(
        "closed_at",
        "TIMESTAMP",
        mode="NULLABLE",
        description="set if and only if status is closed",
    ),
]

#: Analytical copy of :class:`shared.contracts.StrReport`.
#:
#: ``citations_verified`` travels with the row. A consumer that reads a
#: report out of BigQuery and renders it must not have to consult any
#: other source to learn that its citations are synthetic placeholders.
#:
#: This table is deliberately **absent** from
#: :data:`shared.tool_interfaces.ALLOWED_SQL_TABLES` -- see the note
#: there. It exists to be persisted and displayed, not to be retrieved
#: as evidence by the agent that writes it.
STR_REPORTS_SCHEMA: list[SchemaField] = [
    SchemaField("report_id", "STRING", mode="REQUIRED"),
    SchemaField("case_id", "STRING", mode="REQUIRED"),
    SchemaField("narrative", "STRING", mode="REQUIRED"),
    SchemaField("regulation_citations", "STRING", mode="REPEATED"),
    SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
    SchemaField(
        "status",
        "STRING",
        mode="REQUIRED",
        description="draft, filed or rejected",
    ),
    SchemaField(
        "citations_verified",
        "BOOL",
        mode="REQUIRED",
        description=(
            "true only when every citation resolved against a real indexed "
            "regulation; false while the placeholder corpus is in use"
        ),
    ),
]

#: Analytical copy of :class:`shared.contracts.AccountEvidence`.
#:
#: Evidence, not a detector: nothing in this table participates in the
#: firing decision, which is velocity and structure only. ``scope`` is
#: the load-bearing column — same-bank identity reuse is mostly noise,
#: cross-bank reuse is the citable fact — so a query that aggregates
#: these rows without grouping by ``scope`` is reporting a wrong number.
ACCOUNT_EVIDENCE_SCHEMA: list[SchemaField] = [
    SchemaField("hashed_account_id", "STRING", mode="REQUIRED"),
    SchemaField(
        "evidence_type",
        "STRING",
        mode="REQUIRED",
        description=(
            "cross_bank_shared_device, cross_bank_reused_id_number, "
            "cross_bank_reused_photo, or their same_bank_ counterparts"
        ),
    ),
    SchemaField(
        "scope",
        "STRING",
        mode="REQUIRED",
        description="single_bank or cross_bank",
    ),
    SchemaField(
        "detail",
        "STRING",
        mode="REQUIRED",
        description="ready-to-cite sentence naming the counterparty banks",
    ),
    SchemaField("counterparty_hashed_account_ids", "STRING", mode="REPEATED"),
    SchemaField("counterparty_bank_ids", "STRING", mode="REPEATED"),
    SchemaField(
        "observed_at",
        "TIMESTAMP",
        mode="REQUIRED",
        description="when the link became citable: the later of the two sightings",
    ),
    SchemaField(
        "source_table",
        "STRING",
        mode="REQUIRED",
        description="devices or kyc_documents",
    ),
    SchemaField(
        "detector_version",
        "STRING",
        mode="NULLABLE",
        description="detector whose flagged accounts this evidence was attached to",
    ),
]

#: Table name -> schema. Used by ``infra/`` to create tables and by
#: writers to validate rows before load.
TABLE_SCHEMAS: dict[str, list[SchemaField]] = {
    "accounts": ACCOUNTS_SCHEMA,
    "devices": DEVICES_SCHEMA,
    "kyc_documents": KYC_DOCUMENTS_SCHEMA,
    "kyc_submissions": KYC_SUBMISSIONS_SCHEMA,
    "ground_truth": GROUND_TRUTH_SCHEMA,
    "transfers": TRANSFERS_SCHEMA,
    "features": FEATURES_SCHEMA,
    "scores": SCORES_SCHEMA,
    "rings": RINGS_SCHEMA,
    "ring_explanations": RING_EXPLANATIONS_SCHEMA,
    "account_explanations": ACCOUNT_EXPLANATIONS_SCHEMA,
    "account_evidence": ACCOUNT_EVIDENCE_SCHEMA,
    "cases": CASES_SCHEMA,
    "str_reports": STR_REPORTS_SCHEMA,
}
