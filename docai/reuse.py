"""Cross-bank KYC document reuse detection.

Three independent match types, all emitted as document-level *pairs*:

``hashed_id_number``
    Exact equality on the hashed id number.
``normalized_address``
    Exact equality on the normalized address.
``photo_embedding_ref``
    Near-duplicate images, via :mod:`docai.embeddings`. Two accounts
    pointing at the same file match at similarity 1.0; two different
    files match when their embeddings clear the configured threshold.

What this is worth
------------------
Reuse on its own is a *weak* signal and this module does not pretend
otherwise. Legitimate accounts reuse documents constantly in the
simulated population -- joint accounts, business owners, families,
people with two accounts at one bank -- and 400 people legitimately
present one identity at two banks. The discriminating part is
``cross_bank``: reuse spanning two banks is the thing no single bank can
see on its own. Every pair therefore carries both bank ids and a sorted
``bank_pair`` so downstream scoring can weight the bank combination
instead of the bare fact of reuse.

``ReusePair`` is defined here rather than in ``shared/contracts.py``
because shared/ has no reuse-pair model today. It is a proposed addition
to the shared contracts -- see ``docai/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from docai.config import DocaiConfig
from docai.embeddings import ImageEmbedder, build_embedder, embed_refs, find_near_duplicates
from docai.parsing import parse_kyc_documents
from shared.contracts import AccountEvidence, EvidenceScope, KycDocument

#: Attribute a pair matched on. These strings match the keys documented
#: for ``search`` of ``get_document_reuse`` in ``shared/tool_interfaces.py``.
MatchType = Literal["hashed_id_number", "normalized_address", "photo_embedding_ref"]

EXACT_MATCH_TYPES: tuple[MatchType, ...] = ("hashed_id_number", "normalized_address")

#: Match type -> (``AccountEvidence.evidence_type`` suffix, detail subject).
#: The suffix is prefixed with ``cross_bank_`` or ``same_bank_`` at emit
#: time. ``reused_id_number`` and ``reused_photo`` are the keys ``ml/``
#: already writes; ``reused_address`` is the one docai/ adds, which the
#: open taxonomy on ``AccountEvidence.evidence_type`` anticipates.
EVIDENCE_TYPE_BY_MATCH: dict[MatchType, tuple[str, str]] = {
    "hashed_id_number": ("reused_id_number", "id number"),
    "normalized_address": ("reused_address", "address"),
    "photo_embedding_ref": ("reused_photo", "document photo"),
}

#: Columns written to ``account_evidence.parquet``, in shared-contract order.
ACCOUNT_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "hashed_account_id",
    "evidence_type",
    "scope",
    "detail",
    "counterparty_hashed_account_ids",
    "counterparty_bank_ids",
    "observed_at",
    "source_table",
    "detector_version",
)

#: Columns written to ``reuse_pairs.parquet``.
REUSE_PAIR_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "left_hashed_account_id",
    "right_hashed_account_id",
    "left_bank_id",
    "right_bank_id",
    "bank_pair",
    "cross_bank",
    "match_type",
    "match_value",
    "similarity",
    "left_doc_id",
    "right_doc_id",
    "doc_type",
    "backend",
    "detected_at",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ReusePair(BaseModel):
    """Two accounts that share one KYC attribute.

    Account ids are sorted lexically, so a pair appears once and never
    in both orders; ``left_bank_id`` and ``right_bank_id`` stay aligned
    with that order. Only hashed account ids are stored.
    """

    #: Stable digest of (match_type, left_doc_id, right_doc_id). Lets the
    #: table be rewritten idempotently and joined without a surrogate key.
    pair_id: str
    left_hashed_account_id: str
    right_hashed_account_id: str
    left_bank_id: str
    right_bank_id: str
    #: Both bank ids sorted and joined with "|", e.g. "BANK_A|BANK_B".
    #: A same-bank pair repeats the bank, e.g. "BANK_A|BANK_A".
    bank_pair: str
    #: True when the two accounts sit at different banks. This is the
    #: signal; same-bank reuse is mostly legitimate noise.
    cross_bank: bool
    match_type: MatchType
    #: The shared value: the hashed id number, the normalized address, or
    #: the left document's image ref. Never a raw identifier.
    match_value: str
    #: 1.0 for an exact match or an identical image file; otherwise the
    #: embedding similarity, in [0, 1].
    similarity: float
    left_doc_id: str
    right_doc_id: str
    #: Document type both sides shared, e.g. "id_card" or "address_proof".
    doc_type: str
    #: What produced the match: "exact", "dhash" or "vertex-multimodal".
    backend: str
    detected_at: datetime = Field(default_factory=_utcnow)


def _pair_id(match_type: str, left_doc_id: str, right_doc_id: str) -> str:
    digest = hashlib.sha256(f"{match_type}:{left_doc_id}:{right_doc_id}".encode())
    return digest.hexdigest()[:32]


def load_bank_map(config: DocaiConfig) -> dict[str, str]:
    """Map hashed account id -> bank id, from ``kyc_submissions.parquet``."""
    frame = pd.read_parquet(
        config.submissions_path, columns=["hashed_account_id", "bank_id"]
    )
    return dict(
        zip(
            frame["hashed_account_id"].astype(str),
            frame["bank_id"].astype(str),
            strict=True,
        )
    )


def _make_pair(
    left: KycDocument,
    right: KycDocument,
    bank_by_account: Mapping[str, str],
    *,
    match_type: MatchType,
    match_value: str,
    similarity: float,
    backend: str,
    detected_at: datetime,
) -> ReusePair:
    """Build one pair with its accounts (and banks) in lexical id order."""
    if left.hashed_account_id > right.hashed_account_id:
        left, right = right, left
    try:
        left_bank = bank_by_account[left.hashed_account_id]
        right_bank = bank_by_account[right.hashed_account_id]
    except KeyError as exc:
        raise KeyError(f"no KYC submission on file for account {exc.args[0]}") from exc
    return ReusePair(
        pair_id=_pair_id(match_type, left.doc_id, right.doc_id),
        left_hashed_account_id=left.hashed_account_id,
        right_hashed_account_id=right.hashed_account_id,
        left_bank_id=left_bank,
        right_bank_id=right_bank,
        bank_pair="|".join(sorted((left_bank, right_bank))),
        cross_bank=left_bank != right_bank,
        match_type=match_type,
        match_value=match_value,
        similarity=similarity,
        left_doc_id=left.doc_id,
        right_doc_id=right.doc_id,
        doc_type=left.doc_type,
        backend=backend,
        detected_at=detected_at,
    )


def exact_reuse_pairs(
    documents: Sequence[KycDocument],
    bank_by_account: Mapping[str, str],
    match_type: MatchType,
    detected_at: datetime,
) -> list[ReusePair]:
    """Pairs sharing an exact attribute value, grouped within a document type."""
    groups: dict[tuple[str, str], list[KycDocument]] = defaultdict(list)
    for doc in documents:
        value = getattr(doc, match_type)
        if not value:
            continue
        groups[(doc.doc_type, value)].append(doc)

    pairs: list[ReusePair] = []
    for (_, value), members in groups.items():
        if len(members) < 2:
            continue
        for left, right in combinations(members, 2):
            if left.hashed_account_id == right.hashed_account_id:
                continue
            pairs.append(
                _make_pair(
                    left,
                    right,
                    bank_by_account,
                    match_type=match_type,
                    match_value=value,
                    similarity=1.0,
                    backend="exact",
                    detected_at=detected_at,
                )
            )
    return pairs


def photo_reuse_pairs(
    documents: Sequence[KycDocument],
    bank_by_account: Mapping[str, str],
    config: DocaiConfig,
    embedder: ImageEmbedder,
    detected_at: datetime,
) -> list[ReusePair]:
    """Pairs whose photos are the same file or near-duplicate images."""
    by_ref: dict[str, list[KycDocument]] = defaultdict(list)
    for doc in documents:
        if doc.photo_embedding_ref:
            by_ref[doc.photo_embedding_ref].append(doc)
    if not by_ref:
        return []

    pairs: list[ReusePair] = []

    # Same file, two accounts: how the simulator models a recycled identity.
    for ref, members in by_ref.items():
        for left, right in combinations(members, 2):
            if left.hashed_account_id == right.hashed_account_id:
                continue
            if left.doc_type != right.doc_type:
                continue
            pairs.append(
                _make_pair(
                    left,
                    right,
                    bank_by_account,
                    match_type="photo_embedding_ref",
                    match_value=ref,
                    similarity=1.0,
                    backend=embedder.backend,
                    detected_at=detected_at,
                )
            )

    # Different files that still look alike.
    embeddings = embed_refs(by_ref, embedder, config.images_dir)
    for match in find_near_duplicates(
        embeddings, config.photo_similarity_threshold, embedder.metric
    ):
        for left in by_ref[match.left_ref]:
            for right in by_ref[match.right_ref]:
                if left.hashed_account_id == right.hashed_account_id:
                    continue
                if left.doc_type != right.doc_type:
                    continue
                pairs.append(
                    _make_pair(
                        left,
                        right,
                        bank_by_account,
                        match_type="photo_embedding_ref",
                        match_value=match.left_ref,
                        similarity=match.similarity,
                        backend=embedder.backend,
                        detected_at=detected_at,
                    )
                )
    return pairs


def detect_reuse(
    documents: Sequence[KycDocument],
    bank_by_account: Mapping[str, str],
    config: DocaiConfig,
    embedder: ImageEmbedder | None = None,
) -> list[ReusePair]:
    """Find every reused-document pair in a parsed KYC corpus.

    Args:
        documents: Parsed documents, as produced by :mod:`docai.parsing`.
        bank_by_account: Hashed account id -> bank id, from
            :func:`load_bank_map`.
        config: Resolved module config; supplies the image root and the
            photo similarity threshold.
        embedder: Override the image backend, mostly for tests. Defaults
            to the one matching ``config.mode``.

    Returns:
        Pairs sorted by match type then account ids. Two accounts sharing
        an identity typically yield several pairs -- one per attribute per
        document type -- because each is separate evidence.

    Raises:
        KeyError: If a document's account has no KYC submission on file.
    """
    active = embedder if embedder is not None else build_embedder(config)
    detected_at = _utcnow()

    pairs: list[ReusePair] = []
    for match_type in EXACT_MATCH_TYPES:
        pairs.extend(exact_reuse_pairs(documents, bank_by_account, match_type, detected_at))
    pairs.extend(photo_reuse_pairs(documents, bank_by_account, config, active, detected_at))

    pairs.sort(
        key=lambda p: (
            p.match_type,
            p.left_hashed_account_id,
            p.right_hashed_account_id,
            p.doc_type,
        )
    )
    return pairs


def write_reuse_pairs(pairs: Iterable[ReusePair], path: Path) -> Path:
    """Write reuse pairs to parquet with :data:`REUSE_PAIR_COLUMNS` in order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [pair.model_dump() for pair in pairs], columns=list(REUSE_PAIR_COLUMNS)
    )
    frame.to_parquet(path, index=False)
    return path


def load_submitted_at_map(config: DocaiConfig) -> dict[str, datetime]:
    """Map hashed account id -> KYC submission time.

    Supplies ``AccountEvidence.observed_at``, which the shared contract
    requires and deliberately does not default: a reuse link only becomes
    citable once both sides have been submitted.
    """
    frame = pd.read_parquet(
        config.submissions_path, columns=["hashed_account_id", "submitted_at"]
    )
    stamps = pd.to_datetime(frame["submitted_at"], utc=True)
    return dict(
        zip(frame["hashed_account_id"].astype(str), stamps.dt.to_pydatetime(), strict=True)
    )


def evidence_from_pairs(
    pairs: Sequence[ReusePair],
    submitted_at_by_account: Mapping[str, datetime],
) -> list[AccountEvidence]:
    """Reshape reuse pairs into ``shared.contracts.AccountEvidence`` rows.

    One row per ``(hashed_account_id, evidence_type, scope)``, matching
    the shape ``ml/`` writes to ``ml/out/account_evidence.parquet``. Each
    pair contributes a row to *both* of its accounts, each naming the
    other as the counterparty.

    ``scope`` comes straight from whether the two accounts' banks differ.
    A ``single_bank`` row is not incriminating and must not be rendered
    as though it were: same-bank reuse is overwhelmingly legitimate in
    this population. ``source_table`` is always ``"kyc_documents"`` and
    ``detector_version`` is always ``None``, because this evidence is
    produced independently of any detector run.

    Args:
        pairs: Document-level pairs from :func:`detect_reuse`.
        submitted_at_by_account: Hashed account id -> submission time,
            from :func:`load_submitted_at_map`.

    Returns:
        Evidence rows sorted by account id then evidence type.

    Raises:
        KeyError: If an account in ``pairs`` has no submission time.
    """
    # (account, evidence_type, scope) -> counterparty account -> bank
    grouped: dict[tuple[str, str, EvidenceScope], dict[str, str]] = defaultdict(dict)
    observed: dict[tuple[str, str, EvidenceScope], datetime] = {}
    subjects: dict[tuple[str, str, EvidenceScope], str] = {}

    for pair in pairs:
        scope = EvidenceScope.CROSS_BANK if pair.cross_bank else EvidenceScope.SINGLE_BANK
        prefix = "cross_bank" if pair.cross_bank else "same_bank"
        suffix, subject = EVIDENCE_TYPE_BY_MATCH[pair.match_type]
        evidence_type = f"{prefix}_{suffix}"
        sides = (
            (pair.left_hashed_account_id, pair.right_hashed_account_id, pair.right_bank_id),
            (pair.right_hashed_account_id, pair.left_hashed_account_id, pair.left_bank_id),
        )
        try:
            # The later of the two sightings: a bank cannot act on a pair
            # before both sides exist.
            citable_at = max(
                submitted_at_by_account[pair.left_hashed_account_id],
                submitted_at_by_account[pair.right_hashed_account_id],
            )
        except KeyError as exc:
            raise KeyError(f"no KYC submission time for account {exc.args[0]}") from exc
        for account, counterparty, counterparty_bank in sides:
            key = (account, evidence_type, scope)
            grouped[key][counterparty] = counterparty_bank
            subjects[key] = subject
            previous = observed.get(key)
            if previous is None or citable_at > previous:
                observed[key] = citable_at

    evidence: list[AccountEvidence] = []
    for (account, evidence_type, scope), counterparties in grouped.items():
        accounts = sorted(counterparties)
        banks = sorted(set(counterparties.values()))
        evidence.append(
            AccountEvidence(
                hashed_account_id=account,
                evidence_type=evidence_type,
                scope=scope,
                detail=(
                    f"{subjects[(account, evidence_type, scope)]} reused by "
                    f"{len(accounts)} account(s) at bank(s) {', '.join(banks)}"
                ),
                counterparty_hashed_account_ids=accounts,
                counterparty_bank_ids=banks,
                observed_at=observed[(account, evidence_type, scope)],
                source_table="kyc_documents",
                detector_version=None,
            )
        )
    evidence.sort(key=lambda e: (e.hashed_account_id, e.evidence_type))
    return evidence


def write_account_evidence(evidence: Iterable[AccountEvidence], path: Path) -> Path:
    """Write ``AccountEvidence`` rows to parquet in shared-contract field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in evidence:
        row = item.model_dump()
        # EvidenceScope is a str enum; store its plain value so the file
        # round-trips through pyarrow as a string column, like ml/ does.
        row["scope"] = item.scope.value
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(ACCOUNT_EVIDENCE_COLUMNS))
    frame.to_parquet(path, index=False)
    return path


def account_pair_summary(pairs: Sequence[ReusePair]) -> pd.DataFrame:
    """Collapse document-level pairs to one row per account pair.

    Returns a frame with one row per (left account, right account) and
    the columns ``left_hashed_account_id``, ``right_hashed_account_id``,
    ``bank_pair``, ``cross_bank``, ``match_types`` (sorted, comma
    joined), ``max_similarity`` and ``n_pairs``.
    """
    if not pairs:
        return pd.DataFrame(
            columns=[
                "left_hashed_account_id",
                "right_hashed_account_id",
                "bank_pair",
                "cross_bank",
                "match_types",
                "max_similarity",
                "n_pairs",
            ]
        )
    frame = pd.DataFrame([pair.model_dump() for pair in pairs])
    grouped = frame.groupby(
        ["left_hashed_account_id", "right_hashed_account_id", "bank_pair", "cross_bank"],
        as_index=False,
    ).agg(
        match_types=("match_type", lambda s: ",".join(sorted(set(s)))),
        max_similarity=("similarity", "max"),
        n_pairs=("pair_id", "size"),
    )
    return grouped


def main(argv: list[str] | None = None) -> int:
    """Parse the corpus, detect reuse and write ``reuse_pairs.parquet``."""
    parser = argparse.ArgumentParser(description="Detect MuleNet KYC document reuse.")
    parser.add_argument("--limit", type=int, default=None, help="parse only N documents")
    args = parser.parse_args(argv)

    config = DocaiConfig.from_env()
    documents = parse_kyc_documents(config, limit=args.limit)
    pairs = detect_reuse(documents, load_bank_map(config), config)
    out = write_reuse_pairs(pairs, config.reuse_pairs_path)

    evidence = evidence_from_pairs(pairs, load_submitted_at_map(config))
    evidence_out = write_account_evidence(evidence, config.account_evidence_path)

    summary = account_pair_summary(pairs)
    cross = int(summary["cross_bank"].sum()) if len(summary) else 0
    print(
        f"mode={config.mode} documents={len(documents)} pairs={len(pairs)} "
        f"account_pairs={len(summary)} cross_bank_account_pairs={cross} -> {out}"
    )
    print(f"evidence_rows={len(evidence)} -> {evidence_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
