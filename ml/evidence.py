"""Identity-reuse evidence attached to accounts the velocity rule flags.

Why this module exists
----------------------
The identity branch of the old compound rule was measured and found
**strictly dominated**: over the full window it added 0 true positives
the velocity branch had not already caught, and 146 false positives. A
clause that only ever costs precision has no business inside a firing
condition.

But the signal itself is not worthless — it is worthless *as a
detector*. "This account shares a device fingerprint with an account at
a different bank" is exactly the sentence a compliance analyst wants in
an STR narrative, and exactly the sentence a report-writing agent can
cite. So the design was inverted: **velocity decides, identity is
evidence.** Nothing in this module can cause an account to be flagged;
it only annotates accounts the detector already chose.

Three evidence types, all cross-bank
------------------------------------
``cross_bank_shared_device``
    Another account at a *different* bank was seen on the same
    ``devices.device_fingerprint``.

``cross_bank_reused_id_number``
    Another account at a different bank filed a KYC document carrying
    the same ``kyc_documents.hashed_id_number``.

``cross_bank_reused_photo``
    Another account at a different bank filed a document whose
    ``kyc_documents.photo_embedding_ref`` matches. This key is **not**
    used by ``features.identity_links`` and therefore does not touch any
    ``sb_``/``cb_`` feature value; it exists here because a matching face
    across two banks is the most quotable single fact in a report.

The ``same_bank_`` counterparts are the single-bank analogue: the only
identity reuse one bank can observe on its own. They are what makes the
cross-bank contrast legible rather than asserted.

``same_bank_*`` rows are now also **written** to
``ml/out/account_evidence.parquet``, not merely computed as flags. Rings
publish same-bank identity links as CONTEXT members
(``RingMember.alertable is False``), and each such member names its
``link_evidence_type``; without the matching row a consumer that follows
that join renders a blank panel in place of the qualifier that says the
account is not accused of anything. The spellings are the same ones
``docai/`` emits — ``same_bank_shared_device``,
``same_bank_reused_id_number``, ``same_bank_reused_photo`` — because the
field is a join key across both producers.

Point-in-time discipline
------------------------
Identical to ``features.py``: a pair counts only when **both**
observations land at or before the *observing* account's cutoff. An
analyst cannot cite a fingerprint sighting that has not happened yet, so
neither can this table. ``photo_embedding_ref`` inherits the filing
account's ``kyc_submissions.submitted_at``, exactly as
``hashed_id_number`` does.
"""

from __future__ import annotations

import pandas as pd

from ml.features import identity_links
from shared.contracts import EvidenceScope

#: Link key -> evidence name. Order fixes the column order downstream.
EVIDENCE_KEYS: dict[str, str] = {
    "device_fingerprint": "shared_device",
    "hashed_id_number": "reused_id_number",
    "photo_embedding_ref": "reused_photo",
}

#: Evidence columns for the cross-bank scope, in report order.
CROSS_BANK_COLUMNS: tuple[str, ...] = tuple(f"cross_bank_{n}" for n in EVIDENCE_KEYS.values())

#: The single-bank analogue of the same three signals.
SAME_BANK_COLUMNS: tuple[str, ...] = tuple(f"same_bank_{n}" for n in EVIDENCE_KEYS.values())


def evidence_links(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The three identity link tables, each stamped with its observation time.

    Device and ID-number links are reused verbatim from
    ``features.identity_links`` so the evidence table and the feature
    table can never disagree about what was observed. The photo link is
    added here only, and inherits ``submitted_at`` from the filing
    account's KYC submission.
    """
    links = dict(identity_links(tables))
    submitted = tables["kyc_submissions"].groupby("hashed_account_id")["submitted_at"].min()
    photos = tables["kyc_documents"][["hashed_account_id", "photo_embedding_ref"]].copy()
    photos = photos.dropna(subset=["photo_embedding_ref"])
    photos["observed_at"] = photos["hashed_account_id"].map(submitted)
    links["photo_embedding_ref"] = photos.dropna(subset=["observed_at"])
    return links


def _partner_flags(
    links: pd.DataFrame, key: str, bank: pd.Series, index: pd.Index, cutoff: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """(same-bank partner, other-bank partner) per account for one link key."""
    pairs = links.groupby(["hashed_account_id", key], sort=False)["observed_at"].min().reset_index()
    joined = pairs.merge(pairs, on=key, suffixes=("", "_other"))
    joined = joined.loc[joined["hashed_account_id"] != joined["hashed_account_id_other"]]
    own_cutoff = joined["hashed_account_id"].map(cutoff)
    joined = joined.loc[
        (joined["observed_at"] <= own_cutoff) & (joined["observed_at_other"] <= own_cutoff)
    ]
    cross = joined["hashed_account_id"].map(bank) != joined["hashed_account_id_other"].map(bank)
    same = pd.Series(index.isin(joined.loc[~cross, "hashed_account_id"]), index=index)
    other = pd.Series(index.isin(joined.loc[cross, "hashed_account_id"]), index=index)
    return same, other


def evidence_flags(tables: dict[str, pd.DataFrame], cutoff: pd.Series) -> pd.DataFrame:
    """Per account, which identity-reuse evidence is citable at its cutoff.

    Returns a boolean frame indexed by ``hashed_account_id`` with the
    three ``cross_bank_*`` columns and their three ``same_bank_*``
    counterparts. Never consulted by any firing condition.
    """
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    index = pd.Index(tables["accounts"]["hashed_account_id"], name="hashed_account_id")
    links = evidence_links(tables)

    columns: dict[str, pd.Series] = {}
    for key, name in EVIDENCE_KEYS.items():
        same, other = _partner_flags(links[key], key, bank, index, cutoff.reindex(index))
        columns[f"same_bank_{name}"] = same
        columns[f"cross_bank_{name}"] = other
    frame = pd.DataFrame(columns, index=index)
    return frame[list(CROSS_BANK_COLUMNS) + list(SAME_BANK_COLUMNS)]


def attachment_rates(evidence: pd.DataFrame, flagged: pd.Series) -> dict[str, float]:
    """Share of flagged accounts carrying each evidence type, plus any/none.

    This is the number the UI badge and the STR narrative are built on:
    of the accounts the detector hands an analyst, how many arrive with a
    citable identity fact attached.
    """
    subset = evidence.loc[flagged.reindex(evidence.index).fillna(False).to_numpy()]
    n = len(subset)
    out: dict[str, float] = {"flagged": n}
    for column in CROSS_BANK_COLUMNS + SAME_BANK_COLUMNS:
        out[column] = float(subset[column].mean()) if n else 0.0
    any_cross = subset[list(CROSS_BANK_COLUMNS)].any(axis=1) if n else pd.Series(dtype=bool)
    out["any_cross_bank_evidence"] = float(any_cross.mean()) if n else 0.0
    out["no_cross_bank_evidence"] = 1.0 - out["any_cross_bank_evidence"] if n else 0.0
    return out


#: Evidence name -> the simulator table the fact is quotable from.
SOURCE_TABLES: dict[str, str] = {
    "shared_device": "devices",
    "reused_id_number": "kyc_documents",
    "reused_photo": "kyc_documents",
}

#: Scope -> evidence name -> the sentence an analyst or the STR agent can
#: cite. The two scopes get different sentences on purpose. A cross-bank
#: row is the citable fact; a same-bank row is **context**, and its own
#: text has to say so, because the sentence travels into an STR draft
#: and a UI panel without its ``scope`` column attached.
DETAIL_PHRASES: dict[EvidenceScope, dict[str, str]] = {
    EvidenceScope.CROSS_BANK: {
        "shared_device": "device fingerprint shared with {n} account(s) at bank(s) {banks}",
        "reused_id_number": "KYC ID number reused by {n} account(s) at bank(s) {banks}",
        "reused_photo": "KYC photo matches {n} account(s) at bank(s) {banks}",
    },
    EvidenceScope.SINGLE_BANK: {
        "shared_device": (
            "context only: device fingerprint also seen on {n} account(s) at the same "
            "bank ({banks}); same-bank device sharing is common among legitimate "
            "customers and is not evidence of a mule"
        ),
        "reused_id_number": (
            "context only: KYC ID number also filed by {n} account(s) at the same "
            "bank ({banks}); same-bank document reuse is ordinary and is not evidence "
            "of a mule"
        ),
        "reused_photo": (
            "context only: KYC photo also matches {n} account(s) at the same bank "
            "({banks}); same-bank photo reuse is ordinary and is not evidence of a mule"
        ),
    },
}

#: Scope -> the ``evidence_type`` prefix. ``EvidenceScope.SINGLE_BANK`` is
#: spelled ``single_bank`` but the evidence taxonomy says ``same_bank_``,
#: and ``docai/`` already emits those exact keys. ``evidence_type`` is a
#: join key, so the two spellings are mapped here rather than derived
#: from ``scope.value``, which would silently invent ``single_bank_*``.
SCOPE_PREFIXES: dict[EvidenceScope, str] = {
    EvidenceScope.CROSS_BANK: "cross_bank",
    EvidenceScope.SINGLE_BANK: "same_bank",
}


def evidence_records(
    tables: dict[str, pd.DataFrame],
    cutoff: pd.Series,
    accounts: pd.Index | None = None,
    scope: EvidenceScope = EvidenceScope.CROSS_BANK,
) -> pd.DataFrame:
    """Long-format, citable evidence rows: one per (account, evidence type).

    Unlike ``evidence_flags`` this carries *who* and *when* — the partner
    accounts, the banks they sit at, and the instant the link became
    citable — because an STR narrative that says "shares a device" without
    naming the counterparty bank is not worth filing.

    ``observed_at`` is the **later** of the two sightings: a link is
    citable only once both sides have been seen, so that is the moment
    the bank could first have acted on it. It is returned tz-aware UTC.

    ``accounts`` restricts output to the flagged accounts; omitted, every
    account with any evidence is returned.

    ``scope`` selects which side of the bank boundary is emitted.
    ``CROSS_BANK`` (the default, and the only thing this function used to
    do) keeps partners at a *different* bank and emits ``cross_bank_*``
    types. ``SINGLE_BANK`` keeps partners at the *same* bank and emits
    ``same_bank_*`` types. The second scope exists because ``ml/`` now
    publishes same-bank identity links as ring **context** members, and a
    member whose ``link_evidence_type`` joins to nothing renders blank in
    a UI that follows the join. Emitting the row does not promote it:
    these rows are context, they carry ``single_bank`` scope, and nothing
    in the firing condition reads either scope.
    """
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    links = evidence_links(tables)
    prefix = SCOPE_PREFIXES[scope]
    phrases = DETAIL_PHRASES[scope]
    rows: list[pd.DataFrame] = []

    for key, name in EVIDENCE_KEYS.items():
        pairs = (
            links[key].groupby(["hashed_account_id", key], sort=False)["observed_at"]
            .min()
            .reset_index()
        )
        joined = pairs.merge(pairs, on=key, suffixes=("", "_other"))
        joined = joined.loc[joined["hashed_account_id"] != joined["hashed_account_id_other"]]
        own_cutoff = joined["hashed_account_id"].map(cutoff)
        joined = joined.loc[
            (joined["observed_at"] <= own_cutoff) & (joined["observed_at_other"] <= own_cutoff)
        ]
        own_bank = joined["hashed_account_id"].map(bank)
        other_bank = joined["hashed_account_id_other"].map(bank)
        if scope is EvidenceScope.CROSS_BANK:
            joined = joined.loc[own_bank != other_bank]
        else:
            joined = joined.loc[own_bank == other_bank]
        if accounts is not None:
            joined = joined.loc[joined["hashed_account_id"].isin(accounts)]
        if joined.empty:
            continue
        joined = joined.assign(
            counterparty_bank=other_bank.loc[joined.index],
            # Citable only once both sides have been seen.
            citable_at=joined[["observed_at", "observed_at_other"]].max(axis=1),
        )
        grouped = joined.groupby("hashed_account_id", sort=False).agg(
            counterparty_hashed_account_ids=("hashed_account_id_other", lambda s: sorted(set(s))),
            counterparty_bank_ids=("counterparty_bank", lambda s: sorted(set(s))),
            observed_at=("citable_at", "min"),
        )
        grouped = grouped.reset_index()
        grouped["evidence_type"] = f"{prefix}_{name}"
        grouped["scope"] = scope.value
        grouped["source_table"] = SOURCE_TABLES[name]
        grouped["detail"] = [
            phrases[name].format(n=len(ids), banks=", ".join(banks))
            for ids, banks in zip(
                grouped["counterparty_hashed_account_ids"],
                grouped["counterparty_bank_ids"],
                strict=True,
            )
        ]
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=EVIDENCE_RECORD_COLUMNS)
    frame = pd.concat(rows, ignore_index=True)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame[list(EVIDENCE_RECORD_COLUMNS)].sort_values(
        ["hashed_account_id", "evidence_type"], ignore_index=True
    )


#: Column order of the long-format evidence table. This is the schema
#: proposed to the architect as a shared ``AccountEvidence`` contract.
EVIDENCE_RECORD_COLUMNS: tuple[str, ...] = (
    "hashed_account_id",
    "evidence_type",
    "scope",
    "detail",
    "counterparty_hashed_account_ids",
    "counterparty_bank_ids",
    "observed_at",
    "source_table",
)


def redundancy_check(evidence: pd.DataFrame) -> dict[str, float]:
    """Pairwise Jaccard between the cross-bank evidence columns.

    Measured because an analyst shown two badges will read them as two
    independent corroborations. In **this dataset** that reading is
    wrong: ``cross_bank_reused_photo`` and ``cross_bank_reused_id_number``
    fire on exactly the same 649 accounts, Jaccard 1.000. The simulator
    attaches a reused ``photo_embedding_ref`` to precisely the documents
    that reuse ``hashed_id_number``, so the two are one fact wearing two
    labels.

    Verdict on that 1.000: **(b) artifact** — of the generator, not of
    the model and not leakage. Nothing about the pipeline forces the two
    keys to coincide; real banks routinely see a matched face under a
    different ID number and vice versa, which is why both keys are
    carried separately here. Until the generator decouples them, an STR
    narrative should cite identity reuse **once**, and any UI must not
    render these as two independent corroborations.
    """
    out: dict[str, float] = {}
    for i, left in enumerate(CROSS_BANK_COLUMNS):
        for right in CROSS_BANK_COLUMNS[i + 1 :]:
            a = evidence[left].to_numpy()
            b = evidence[right].to_numpy()
            union = int((a | b).sum())
            out[f"jaccard:{left}|{right}"] = float((a & b).sum() / union) if union else 0.0
    return out


def top_factors(row: pd.Series) -> list[str]:
    """Human-readable evidence strings for one account, for ``top_factors``."""
    phrases = {
        "cross_bank_shared_device": "device fingerprint shared with an account at another bank",
        "cross_bank_reused_id_number": "KYC ID number reused at another bank",
        "cross_bank_reused_photo": "KYC photo matches an account at another bank",
    }
    return [text for column, text in phrases.items() if bool(row.get(column, False))]
