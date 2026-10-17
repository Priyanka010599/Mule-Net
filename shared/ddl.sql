-- MuleNet AlloyDB (PostgreSQL-compatible) DDL.
--
-- Source of truth for field names and types is shared/contracts.py.
-- Every other folder must code against these tables rather than
-- redefining schemas locally (see CLAUDE.md).
--
-- Account identifiers are ALWAYS hashed. No raw account numbers,
-- names, or document numbers are stored in any column here. The
-- account-identifier column is named hashed_account_id everywhere;
-- directional or collection variants keep that name as their root
-- (from_hashed_account_id, to_hashed_account_id,
-- member_hashed_account_ids). Ring membership lives in the ring_members
-- child table, one row per account per ring, carrying the basis on
-- which that account joined; see the comment on that table.
--
-- WHAT IS DELIBERATELY NOT HERE.
-- features, scores, ring_explanations and account_explanations have
-- BigQuery schemas in shared/bq_schemas.py and no table in this file.
-- That is a decision, not an omission. This file is the OPERATIONAL
-- store: entities written by the transaction path or edited by a human
-- (accounts, transfers, devices, KYC, rings, cases, str_reports,
-- account_evidence), read row-by-row by id, and updated in place.
-- The four listed above are MODEL-RUN OUTPUT: recomputed wholesale on
-- every ml/ run, never updated in place, keyed by (account, as_of) or
-- by run rather than by a stable business key, and queried as scans and
-- aggregates. Putting them in AlloyDB would buy nothing and would
-- invite an UPDATE against a table whose only correct write is a full
-- replace. If an operational read of a model output is ever needed, the
-- fix is a materialized copy of the latest run, not moving the table.

CREATE TABLE IF NOT EXISTS accounts (
    hashed_account_id TEXT PRIMARY KEY,
    bank_id           TEXT        NOT NULL,
    opened_at         TIMESTAMPTZ NOT NULL,
    account_type      TEXT        NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'dormant', 'frozen', 'closed'))
);

CREATE INDEX IF NOT EXISTS idx_accounts_bank_id ON accounts (bank_id);
CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts (status);
CREATE INDEX IF NOT EXISTS idx_accounts_opened_at ON accounts (opened_at);

CREATE TABLE IF NOT EXISTS transfers (
    transfer_id            TEXT PRIMARY KEY,
    from_hashed_account_id TEXT           NOT NULL REFERENCES accounts (hashed_account_id),
    to_hashed_account_id   TEXT           NOT NULL REFERENCES accounts (hashed_account_id),
    amount                 NUMERIC(15, 2) NOT NULL,
    currency               TEXT           NOT NULL DEFAULT 'USD',
    "timestamp"            TIMESTAMPTZ    NOT NULL,
    channel                TEXT           NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transfers_from_hashed_account_id
    ON transfers (from_hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_transfers_to_hashed_account_id
    ON transfers (to_hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_transfers_timestamp ON transfers ("timestamp");
CREATE INDEX IF NOT EXISTS idx_transfers_from_ts
    ON transfers (from_hashed_account_id, "timestamp");
CREATE INDEX IF NOT EXISTS idx_transfers_to_ts
    ON transfers (to_hashed_account_id, "timestamp");

CREATE TABLE IF NOT EXISTS devices (
    device_id          TEXT PRIMARY KEY,
    hashed_account_id  TEXT        NOT NULL REFERENCES accounts (hashed_account_id),
    device_fingerprint TEXT        NOT NULL,
    first_seen         TIMESTAMPTZ NOT NULL,
    last_seen          TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_devices_hashed_account_id ON devices (hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_devices_fingerprint ON devices (device_fingerprint);

CREATE TABLE IF NOT EXISTS kyc_submissions (
    submission_id     TEXT PRIMARY KEY,
    hashed_account_id TEXT        NOT NULL REFERENCES accounts (hashed_account_id),
    bank_id           TEXT        NOT NULL,
    submitted_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kyc_submissions_hashed_account_id
    ON kyc_submissions (hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_kyc_submissions_bank_id ON kyc_submissions (bank_id);

CREATE TABLE IF NOT EXISTS kyc_documents (
    doc_id              TEXT PRIMARY KEY,
    submission_id       TEXT NOT NULL REFERENCES kyc_submissions (submission_id),
    hashed_account_id   TEXT NOT NULL REFERENCES accounts (hashed_account_id),
    doc_type            TEXT NOT NULL,
    hashed_id_number    TEXT NOT NULL,
    normalized_address  TEXT NOT NULL,
    photo_embedding_ref TEXT
);

CREATE INDEX IF NOT EXISTS idx_kyc_documents_submission_id ON kyc_documents (submission_id);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_hashed_account_id
    ON kyc_documents (hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_hashed_id_number ON kyc_documents (hashed_id_number);
CREATE INDEX IF NOT EXISTS idx_kyc_documents_normalized_address ON kyc_documents (normalized_address);

CREATE TABLE IF NOT EXISTS rings (
    ring_id        TEXT PRIMARY KEY,
    banks_involved TEXT[]      NOT NULL,
    detected_at    TIMESTAMPTZ NOT NULL,
    status         TEXT        NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'confirmed', 'dismissed'))
);

CREATE INDEX IF NOT EXISTS idx_rings_status ON rings (status);
CREATE INDEX IF NOT EXISTS idx_rings_detected_at ON rings (detected_at);
CREATE INDEX IF NOT EXISTS idx_rings_banks ON rings USING GIN (banks_involved);

-- Ring membership (shared.contracts.RingMember), one row per account
-- per ring. This replaces the former rings.member_hashed_account_ids
-- TEXT[] column, which is gone.
--
-- WHY A CHILD TABLE AND NOT JSONB. A TEXT[] cannot hold records, so the
-- choice was a JSONB column on rings or this table. Three things decide
-- it. (1) membership_basis and alertable are *filters*, not payload:
-- the whole point of this change is that a consumer can write
-- "WHERE alertable" instead of re-deriving it, and the agent SQL tool
-- can too. JSONB pushes that into containment operators that are easy
-- to get subtly wrong, and a wrong filter here publishes an alert
-- against an innocent account. (2) The reverse lookup "which rings
-- contain this account" -- get_account_profile and the UI both start
-- there -- becomes a plain btree index instead of the GIN index the
-- array needed. (3) The CHECK constraints below mirror
-- RingMember._link_fields_match_basis, so the database refuses a
-- mislabelled row by UPDATE just as the model refuses it on
-- construction; JSONB can enforce none of that. The cost is one table
-- and a join on read, which is the same join ui/ and agents/ already do
-- against scores and account_evidence.
--
-- MEMBERSHIP IS NOT ONE CLAIM. A velocity row says the account itself
-- forwarded money fast in a cross-bank structure. An identity_link row
-- says only that it shares a device or a hashed id number with an
-- account that did -- a lead, to be described as "linked, needs review"
-- and never as a mule. alertable draws the further line: stage-2
-- alerting is CROSS-BANK ONLY, and same-bank identity links are
-- published as context. Same-bank identity reuse is dominated by
-- legitimate cover (~1,442 legitimate accounts against ~292 mules), so
-- a same-bank link is something to show an analyst, never something to
-- accuse anyone with.
CREATE TABLE IF NOT EXISTS ring_members (
    ring_id                      TEXT    NOT NULL REFERENCES rings (ring_id),
    hashed_account_id            TEXT    NOT NULL REFERENCES accounts (hashed_account_id),
    membership_basis             TEXT    NOT NULL
        CHECK (membership_basis IN ('velocity', 'identity_link')),
    alertable                    BOOLEAN NOT NULL,
    link_scope                   TEXT
        CHECK (link_scope IN ('single_bank', 'cross_bank')),
    linked_from_hashed_account_id TEXT   REFERENCES accounts (hashed_account_id),
    link_evidence_type           TEXT,
    reason                       TEXT,
    PRIMARY KEY (ring_id, hashed_account_id),
    -- A velocity member carries no identity link and is always alertable.
    CHECK (
        membership_basis <> 'velocity'
        OR (
            link_scope IS NULL
            AND linked_from_hashed_account_id IS NULL
            AND link_evidence_type IS NULL
            AND alertable
        )
    ),
    -- An identity_link member must name its link and carry its reason.
    CHECK (
        membership_basis <> 'identity_link'
        OR (
            link_scope IS NOT NULL
            AND linked_from_hashed_account_id IS NOT NULL
            AND link_evidence_type IS NOT NULL
            AND reason IS NOT NULL
        )
    ),
    -- Stage-2 alerting is cross-bank only; same-bank links are context.
    CHECK (
        membership_basis <> 'identity_link'
        OR alertable = (link_scope = 'cross_bank')
    )
);

-- The primary key already covers "members of this ring". These cover
-- the reverse lookup ("which rings contain this account" -- where
-- get_account_profile and the UI account view both start) and the two
-- filters every consumer applies.
CREATE INDEX IF NOT EXISTS idx_ring_members_hashed_account_id
    ON ring_members (hashed_account_id);
CREATE INDEX IF NOT EXISTS idx_ring_members_basis ON ring_members (membership_basis);
CREATE INDEX IF NOT EXISTS idx_ring_members_alertable ON ring_members (alertable);

-- Analyst cases (shared.contracts.Investigation). Produced by agents/:
-- one row per ring whose investigation PASSED VERIFICATION, published as
-- agents/out/cases.parquet, which infra/ loads here.
--
-- A CASE MEANS A NARRATIVE THAT WAS CHECKED. The investigator drafts a
-- narrative with structured claims; a deterministic Python verifier
-- re-derives every claim from tool data; on failure there is one retry,
-- and if that also fails the narrative is WITHHELD and only raw data
-- plus a warning is returned. A withheld investigation produces NO ROW
-- here. There is no 'withheld' status on purpose: a case an analyst has
-- to re-check before trusting is not worth having a table for.
--
-- The three verification columns are what make that enforceable rather
-- than merely stated, mirroring
-- Investigation._rests_on_a_passed_verification in contracts.py.
-- verified_claim_count >= 1 is the load-bearing one: a withheld
-- investigation carries an empty claim list, so its shape cannot be
-- written here without inventing a number.
--
-- ONE CASE PER RING, enforced by UNIQUE (ring_id) and by deriving
-- case_id from ring_id (contracts.case_id_for_ring). A ring
-- re-investigated and verified again UPDATES this row -- verified_at
-- advances, the attempt and claim counts are replaced -- rather than
-- opening a second case. A second case per ring would make
-- str_reports.case_id ambiguous about which case a draft belongs to,
-- and "several goes at the same ring" already has a home: one case
-- carries many STR drafts. A case that has reached 'closed' is a
-- compliance record and is left untouched by a later run; reopening it
-- is a human action, not an agent one.
CREATE TABLE IF NOT EXISTS cases (
    case_id               TEXT PRIMARY KEY,
    -- UNIQUE, not just a foreign key: one ring, one case.
    ring_id               TEXT        NOT NULL UNIQUE REFERENCES rings (ring_id),
    verified_at           TIMESTAMPTZ NOT NULL,
    verification_attempts INTEGER     NOT NULL
        CHECK (verification_attempts BETWEEN 1 AND 2),
    verified_claim_count  INTEGER     NOT NULL
        CHECK (verified_claim_count >= 1),
    analyst_notes         TEXT        NOT NULL DEFAULT '',
    status                TEXT        NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'in_review', 'closed')),
    opened_at             TIMESTAMPTZ NOT NULL,
    closed_at             TIMESTAMPTZ,
    -- closed_at is set if and only if the case is closed; mirrors
    -- Investigation._closed_at_matches_status in contracts.py.
    CHECK ((status = 'closed') = (closed_at IS NOT NULL))
);

-- UNIQUE (ring_id) already indexes the ring lookup.
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);

-- Drafted suspicious transaction reports (shared.contracts.StrReport).
-- Produced by the report agent in agents/, one or more per case.
--
-- citations_verified is FALSE until every citation on the report
-- resolves against a real indexed regulation document. Today every
-- citation comes from docai/regulations/placeholder_corpus.json, which
-- is synthetic, so every row is FALSE. The CHECK below makes an
-- unverified report *unfileable* rather than merely labelled: a draft
-- cannot become a filing by an UPDATE that forgets the warning.
CREATE TABLE IF NOT EXISTS str_reports (
    report_id            TEXT PRIMARY KEY,
    case_id              TEXT        NOT NULL REFERENCES cases (case_id),
    narrative            TEXT        NOT NULL,
    regulation_citations TEXT[]      NOT NULL DEFAULT '{}',
    generated_at         TIMESTAMPTZ NOT NULL,
    status               TEXT        NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'filed', 'rejected')),
    citations_verified   BOOLEAN     NOT NULL DEFAULT FALSE,
    CHECK (status <> 'filed' OR citations_verified)
);

CREATE INDEX IF NOT EXISTS idx_str_reports_case_id ON str_reports (case_id);
CREATE INDEX IF NOT EXISTS idx_str_reports_status ON str_reports (status);
CREATE INDEX IF NOT EXISTS idx_str_reports_generated_at ON str_reports (generated_at);

-- Identity-reuse evidence attached to an account (shared.contracts.AccountEvidence).
--
-- EVIDENCE, NOT A DETECTOR. Nothing in this table takes part in the
-- firing decision, which is velocity and structure only. Rows exist so
-- an analyst and the report-writing agent can cite who, which bank and
-- when. Identity reuse is weak on its own -- legitimate customers reuse
-- documents within a bank in far greater numbers than mules do, and
-- dual-bank customers legitimately reuse an id across two banks -- so a
-- row here never implies guilt and must always be read with its scope.
CREATE TABLE IF NOT EXISTS account_evidence (
    hashed_account_id               TEXT        NOT NULL REFERENCES accounts (hashed_account_id),
    evidence_type                   TEXT        NOT NULL,
    scope                           TEXT        NOT NULL
        CHECK (scope IN ('single_bank', 'cross_bank')),
    detail                          TEXT        NOT NULL,
    counterparty_hashed_account_ids TEXT[]      NOT NULL DEFAULT '{}',
    counterparty_bank_ids           TEXT[]      NOT NULL DEFAULT '{}',
    observed_at                     TIMESTAMPTZ NOT NULL,
    source_table                    TEXT        NOT NULL,
    detector_version                TEXT,
    PRIMARY KEY (hashed_account_id, evidence_type, scope)
);

-- The primary key already covers lookup by account (the badge and the
-- STR narrative both start from one account), so no separate index on
-- hashed_account_id is needed. These cover the cross-account filters the
-- UI badge and the report agent apply, and the reverse lookup "which
-- accounts name this one as a counterparty".
CREATE INDEX IF NOT EXISTS idx_account_evidence_scope ON account_evidence (scope);
CREATE INDEX IF NOT EXISTS idx_account_evidence_type ON account_evidence (evidence_type);
CREATE INDEX IF NOT EXISTS idx_account_evidence_observed_at ON account_evidence (observed_at);
CREATE INDEX IF NOT EXISTS idx_account_evidence_counterparties
    ON account_evidence USING GIN (counterparty_hashed_account_ids);
