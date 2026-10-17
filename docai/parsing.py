"""KYC document parsing into :class:`shared.contracts.KycDocument`.

Two interchangeable backends behind one interface:

``MockDocumentParser`` (mock)
    Offline. Fields come from the simulator's ``kyc_documents.parquet``
    intake record, and the referenced image is inspected only far enough
    to record whether it is present and readable. No OCR.

``DocumentAiParser`` (cloud)
    Sends the rendered PNG to a Document AI processor (Identity Document
    or Form Parser) and reads the holder name, document number and
    address off the returned entities.

In both modes ``hashed_id_number`` comes from the intake record, never
from the document: the hash is assigned at intake and Document AI only
ever sees the rendered ``SYN-xxxx-xxxx-xxxx`` placeholder. What the
cloud path contributes is the *address*, which is parsed from the
document and then normalized here.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from docai.config import DocaiConfig
from shared.contracts import KycDocument

#: Columns ``parse_kyc_documents`` needs from ``kyc_documents.parquet``.
DOCUMENT_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "hashed_account_id",
    "doc_type",
    "hashed_id_number",
    "normalized_address",
    "photo_embedding_ref",
)

_WHITESPACE = re.compile(r"\s+")

#: Document AI entity types that carry a postal address, lowercased.
_ADDRESS_HINTS: tuple[str, ...] = ("address",)

#: Field labels the simulator prints on its two document layouts, in the
#: order they are drawn (``simulator/kyc.py``: ``render_id_card`` and
#: ``render_address_proof``). The Form Parser's own key/value pairing is
#: unreliable on these cards -- measured on live responses, an ID card
#: came back with the address filed under ``TYPE`` and an address proof
#: came back with no form fields at all -- but the OCR *text* is exact
#: and keeps label and value on consecutive lines. So fields are read off
#: the text by label.
FIELD_LABELS: tuple[str, ...] = (
    "NAME",
    "ID NUMBER",
    "ADDRESS",
    "TYPE",
    "ACCOUNT HOLDER",
    "SERVICE ADDRESS",
    "STATEMENT PERIOD",
    "PRESENTED TO",
    "AMOUNT DUE",
)

#: Decorative strings drawn inside the card layout that OCR interleaves
#: between a label and its value. ``ADDRESS`` on an ID card is followed
#: by ``SYNTHETIC RECORD`` before the address itself.
_DECORATIONS: frozenset[str] = frozenset({"NO PHOTO", "SYNTHETIC RECORD"})

#: Labels that carry the holder's address, most specific first.
ADDRESS_LABELS: tuple[str, ...] = ("SERVICE ADDRESS", "ADDRESS")

#: Labels that carry the holder's name, most specific first.
NAME_LABELS: tuple[str, ...] = ("ACCOUNT HOLDER", "NAME")

#: Label carrying the rendered ``SYN-xxxx-xxxx-xxxx`` document number.
ID_NUMBER_LABEL = "ID NUMBER"


def extract_labelled_fields(text: str) -> dict[str, str]:
    """Read ``LABEL`` / value pairs out of a Document AI text layer.

    Each label in :data:`FIELD_LABELS` takes the first following line
    that is neither another label nor layout decoration. Returns only
    the labels actually present.
    """
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        if line not in FIELD_LABELS or line in found:
            continue
        for candidate in lines[index + 1 :]:
            if candidate in FIELD_LABELS or candidate in _DECORATIONS:
                continue
            found[line] = candidate
            break
    return found


def first_labelled(fields: Mapping[str, str], labels: Iterable[str]) -> str | None:
    """Return the first present label's value, or ``None``."""
    for label in labels:
        value = fields.get(label)
        if value:
            return value
    return None


def normalize_address(text: str) -> str:
    """Normalize an address to the form stored in ``normalized_address``.

    Uppercased, whitespace collapsed, surrounding punctuation stripped.
    Matching addresses is an exact-equality test downstream, so both the
    parsed and the intake value must pass through here.
    """
    return _WHITESPACE.sub(" ", text).strip(" ,.\t\n").upper()


@dataclass(frozen=True)
class DocumentRecord:
    """One intake row from ``kyc_documents.parquet``."""

    doc_id: str
    hashed_account_id: str
    doc_type: str
    hashed_id_number: str
    normalized_address: str
    photo_embedding_ref: str | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> DocumentRecord:
        """Build a record from a parquet row, coercing missing refs to ``None``."""
        ref = row["photo_embedding_ref"]
        return cls(
            doc_id=str(row["doc_id"]),
            hashed_account_id=str(row["hashed_account_id"]),
            doc_type=str(row["doc_type"]),
            hashed_id_number=str(row["hashed_id_number"]),
            normalized_address=str(row["normalized_address"]),
            photo_embedding_ref=None if ref is None or pd.isna(ref) else str(ref),
        )


@runtime_checkable
class DocumentParser(Protocol):
    """Turns one intake record plus its image into a ``KycDocument``."""

    source: str

    def parse(self, record: DocumentRecord, images_dir: Path) -> KycDocument:
        """Parse one document."""
        ...


class MockDocumentParser:
    """Offline parser: fields from the intake record, image checked for presence."""

    source = "mock"

    def parse(self, record: DocumentRecord, images_dir: Path) -> KycDocument:
        """Build a ``KycDocument`` without calling Document AI.

        The image reference is kept only when the file actually exists, so
        a half-rendered corpus cannot make reuse detection compare against
        images that are not there.
        """
        ref = record.photo_embedding_ref
        if ref is not None and not (images_dir / ref).is_file():
            ref = None
        return KycDocument(
            doc_id=record.doc_id,
            hashed_account_id=record.hashed_account_id,
            doc_type=record.doc_type,
            hashed_id_number=record.hashed_id_number,
            normalized_address=normalize_address(record.normalized_address),
            photo_embedding_ref=ref,
        )


class DocumentAiParser:
    """Cloud parser backed by a Document AI processor.

    Unverified in this environment -- no Google Cloud credentials are
    configured here, so this path has never been executed. The request
    shape is checked against the installed ``google-cloud-documentai``
    3.15.0 (``DocumentProcessorServiceClient.process_document`` with a
    ``RawDocument``).

    Processor choice: the Form Parser. Read off ``processorTypes`` for
    this project on 2026-09-27, the identity-document options offered are
    ``ID_PROOFING_PROCESSOR`` and ``US_DRIVER_LICENSE_PROCESSOR``; there
    is no generic identity parser. ID Proofing returns fraud signals
    rather than fields, and these are not US driver licences, so neither
    extracts a name or an address from a specimen card. The Form Parser
    does, it is a tenth of their price ($0.03/page against $0.10/doc),
    and it is offered in ``asia-south1``, which they are not.
    """

    source = "documentai"

    def __init__(self, config: DocaiConfig) -> None:
        config.require_cloud_ids(processor=True)
        self._config = config
        self._client = None

    def _processor_name(self) -> str:
        cfg = self._config
        return (
            f"projects/{cfg.project_id}/locations/{cfg.docai_location}"
            f"/processors/{cfg.processor_id}"
        )

    def _get_client(self):  # pragma: no cover - requires credentials
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai_v1

            self._client = documentai_v1.DocumentProcessorServiceClient(
                client_options=ClientOptions(
                    api_endpoint=f"{self._config.docai_location}-documentai.googleapis.com"
                )
            )
        return self._client

    def process(self, path: Path) -> Any:  # pragma: no cover
        """Send one rendered document to Document AI and return its ``Document``.

        Separated from :meth:`parse` so an evaluation can read every
        extracted field off one response instead of paying for a second
        call. One call, one page, one billed unit.
        """
        from google.cloud import documentai_v1

        response = self._get_client().process_document(
            request=documentai_v1.ProcessRequest(
                name=self._processor_name(),
                raw_document=documentai_v1.RawDocument(
                    content=path.read_bytes(), mime_type="image/png"
                ),
            )
        )
        return response.document

    def parse(self, record: DocumentRecord, images_dir: Path) -> KycDocument:  # pragma: no cover
        """Send the rendered document to Document AI and map its fields."""
        ref = record.photo_embedding_ref
        address = record.normalized_address
        if ref is not None and (images_dir / ref).is_file():
            parsed_address = _extract_address(self.process(images_dir / ref))
            if parsed_address:
                address = parsed_address
        else:
            ref = None
        return KycDocument(
            doc_id=record.doc_id,
            hashed_account_id=record.hashed_account_id,
            doc_type=record.doc_type,
            # The hash is assigned at intake; Document AI never sees it.
            hashed_id_number=record.hashed_id_number,
            normalized_address=normalize_address(address),
            photo_embedding_ref=ref,
        )


def _extract_address(document: Any) -> str | None:
    """Pull an address out of a Document AI response.

    The labelled text layer first, because it is what actually works on
    these layouts; then typed entities and the Form Parser's key/value
    pairs, which is what a real identity or form document would give.
    """
    labelled = first_labelled(
        extract_labelled_fields(getattr(document, "text", "") or ""), ADDRESS_LABELS
    )
    if labelled:
        return labelled
    for entity in getattr(document, "entities", ()) or ():
        entity_type = (getattr(entity, "type_", "") or "").lower()
        if any(hint in entity_type for hint in _ADDRESS_HINTS):
            text = (getattr(entity, "mention_text", "") or "").strip()
            if text:
                return text
    for page in getattr(document, "pages", ()) or ():
        for field in getattr(page, "form_fields", ()) or ():
            name = _layout_text(document, getattr(field, "field_name", None)).lower()
            if any(hint in name for hint in _ADDRESS_HINTS):
                value = _layout_text(document, getattr(field, "field_value", None))
                if value:
                    return value
    return None


def _layout_text(document: Any, layout: Any) -> str:
    """Resolve a Document AI layout's text anchor against the full document text."""
    if layout is None:
        return ""
    text_anchor = getattr(layout, "text_anchor", None)
    if text_anchor is None:
        return ""
    full_text = getattr(document, "text", "") or ""
    parts = [
        full_text[int(segment.start_index) : int(segment.end_index)]
        for segment in getattr(text_anchor, "text_segments", ()) or ()
    ]
    return "".join(parts).strip()


def build_parser(config: DocaiConfig) -> DocumentParser:
    """Return the parser matching ``config.mode``."""
    if config.mode == "cloud":
        return DocumentAiParser(config)
    return MockDocumentParser()


def load_document_records(
    config: DocaiConfig, limit: int | None = None
) -> Iterator[DocumentRecord]:
    """Stream intake records from ``kyc_documents.parquet``."""
    frame = pd.read_parquet(config.documents_path, columns=list(DOCUMENT_COLUMNS))
    if limit is not None:
        frame = frame.head(limit)
    for row in frame.to_dict("records"):
        yield DocumentRecord.from_row(row)


def parse_kyc_documents(
    config: DocaiConfig,
    limit: int | None = None,
    parser: DocumentParser | None = None,
) -> list[KycDocument]:
    """Parse the KYC corpus into ``KycDocument`` models.

    Args:
        config: Resolved module config; its ``mode`` picks the backend.
        limit: Parse only the first ``limit`` documents. Useful against
            the cloud path, where every document costs an API call.
        parser: Override the backend, mostly for tests.

    Returns:
        One ``KycDocument`` per intake row, in file order.
    """
    active = parser if parser is not None else build_parser(config)
    images_dir = config.images_dir
    return [active.parse(record, images_dir) for record in load_document_records(config, limit)]


def write_parsed_documents(documents: list[KycDocument], path: Path) -> Path:
    """Write parsed documents to parquet, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([doc.model_dump() for doc in documents]).to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    """Parse the corpus and write ``parsed_documents.parquet``."""
    argparser = argparse.ArgumentParser(description="Parse MuleNet KYC documents.")
    argparser.add_argument("--limit", type=int, default=None)
    args = argparser.parse_args(argv)

    config = DocaiConfig.from_env()
    documents = parse_kyc_documents(config, limit=args.limit)
    out = write_parsed_documents(documents, config.parsed_documents_path)
    print(f"mode={config.mode} parsed={len(documents)} -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
