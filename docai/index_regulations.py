"""Load public AML/KYC regulation PDFs into Vertex AI Search.

Usage::

    # offline: check what would be indexed, and validate the placeholders
    python -m docai.index_regulations

    # cloud: upload the corpus to GCS and import it into the data store
    set GOOGLE_CLOUD_PROJECT=mulenet
    set MULENET_REGULATIONS_DATASTORE=mulenet-aml-regulations
    set MULENET_REGULATIONS_GCS_URI=gs://mulenet-mulenet-data/regulations
    set MULENET_REGULATIONS_PDF_DIR=D:\\LLM\\MuleNet\\docai\\regulations_corpus
    python -m docai.index_regulations --mode cloud

**The data store and the search app are created once, by hand**; this
script only uploads and (re)imports. The exact commands are in
``docai/README.md`` under "Standing the index up".

What gets imported is not the bare PDFs. Each PDF is paired with its
entry in ``regulations/corpus_manifest.json`` and imported as a JSONL
metadata record, so every indexed document carries the title and the
issuing authority's URL that a citation must quote. The ``gs://`` object
is not a citable source; the RBI or FIU-IND URL is.

Only *public* regulation documents belong here. The repository is
public: do not commit PDFs whose licence does not permit redistribution;
point ``MULENET_REGULATIONS_PDF_DIR`` at a local directory instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from docai.config import DocaiConfig, DocaiMode
from docai.regulations import (
    CorpusDocument,
    load_corpus_manifest,
    load_placeholder_corpus,
)

#: Object name the import metadata JSONL is written to, inside the
#: bucket that ``MULENET_REGULATIONS_GCS_URI`` names.
METADATA_OBJECT = "regulations_metadata/regulations_metadata.jsonl"


def build_import_records(
    documents: list[CorpusDocument], pdf_uris: dict[str, str]
) -> list[dict[str, Any]]:
    """Pair each manifest entry with its uploaded PDF, as import records.

    The ``structData`` block is where a returned passage's title and
    source come from, so it is written here from the manifest rather
    than left to whatever the PDF's own metadata happens to say.

    Raises:
        RuntimeError: If a manifest document was not uploaded.
    """
    records: list[dict[str, Any]] = []
    for document in documents:
        uri = pdf_uris.get(document.filename)
        if uri is None:
            raise RuntimeError(
                f"{document.filename} is in the manifest but was not uploaded"
            )
        records.append(
            {
                "id": document.doc_id,
                "structData": {
                    "title": document.title,
                    "short_title": document.short_title,
                    "authority": document.authority,
                    "version": document.version,
                    "source_url": document.source_url,
                },
                "content": {"mimeType": "application/pdf", "uri": uri},
            }
        )
    return records


def find_pdfs(pdf_dir: Path | None) -> list[Path]:
    """List the regulation PDFs available for indexing."""
    if pdf_dir is None or not pdf_dir.is_dir():
        return []
    return sorted(pdf_dir.glob("*.pdf"))


def upload_pdfs_to_gcs(pdfs: list[Path], gcs_uri: str) -> list[str]:  # pragma: no cover
    """Upload each PDF under ``gcs_uri`` and return the resulting object URIs.

    Unverified: requires credentials and a bucket.
    """
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"MULENET_REGULATIONS_GCS_URI must start with gs://, got {gcs_uri!r}")
    bucket_name, _, prefix = gcs_uri[len("gs://") :].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uris: list[str] = []
    for pdf in pdfs:
        blob_name = f"{prefix.rstrip('/')}/{pdf.name}" if prefix else pdf.name
        bucket.blob(blob_name).upload_from_filename(str(pdf), content_type="application/pdf")
        uris.append(f"gs://{bucket_name}/{blob_name}")
    return uris


def upload_metadata(records: list[dict[str, Any]], gcs_uri: str) -> str:  # pragma: no cover
    """Write the import records as JSONL to GCS and return its URI."""
    from google.cloud import storage

    bucket_name, _, _ = gcs_uri[len("gs://") :].partition("/")
    body = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    client = storage.Client()
    client.bucket(bucket_name).blob(METADATA_OBJECT).upload_from_string(
        body, content_type="application/jsonl"
    )
    return f"gs://{bucket_name}/{METADATA_OBJECT}"


def import_to_datastore(config: DocaiConfig, metadata_uri: str) -> str:  # pragma: no cover
    """Start the Discovery Engine import and return the operation name.

    REST rather than ``google-cloud-discoveryengine``, which is not in
    ``pyproject.toml`` and which this module may not add.

    ``dataSchema="document"`` (not ``"content"``) is what makes Discovery
    Engine read the JSONL's ``structData`` while still parsing the PDF
    each record points at.

    Import is asynchronous. The first import of this corpus took 10m59s
    to land four documents, and the long-running operation reported no
    progress whatsoever until they appeared -- ``done`` stayed unset and
    ``updateTime`` never moved off its creation value. It then completed
    honestly, with ``successCount`` 4 of 4. **A 200 on this call is not
    evidence that anything was indexed.** Poll the operation *and* list
    the branch's documents before believing it worked.
    """
    import google.auth
    import requests
    from google.auth.transport.requests import Request

    config.require_cloud_ids(datastore=True)
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    parent = (
        f"projects/{config.project_id}/locations/{config.search_location}"
        f"/collections/default_collection/dataStores/{config.search_datastore_id}"
        "/branches/default_branch"
    )
    prefix = "" if config.search_location == "global" else f"{config.search_location}-"
    response = requests.post(
        f"https://{prefix}discoveryengine.googleapis.com/v1/{parent}/documents:import",
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "X-Goog-User-Project": str(config.project_id),
        },
        json={
            "gcsSource": {"inputUris": [metadata_uri], "dataSchema": "document"},
            "reconciliationMode": "INCREMENTAL",
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"import failed: HTTP {response.status_code} {response.text[:400]}")
    return str(response.json().get("name", ""))


def _report_mock(config: DocaiConfig, pdfs: list[Path]) -> None:
    passages = load_placeholder_corpus(config.regulations_dir)
    documents = load_corpus_manifest(config.regulations_dir)
    print(
        f"mode=mock placeholder_passages={len(passages)} "
        f"manifest_documents={len(documents)}"
    )
    print(
        "NOTE: no real AML/KYC regulation PDFs are indexed. Mock search runs "
        "over clearly-labelled PLACEHOLDER passages with invented citations; "
        "nothing drafted from them may be filed or shown to a regulator."
    )
    if pdfs:
        print(f"{len(pdfs)} PDF(s) found and NOT indexed (mock mode):")
        for pdf in pdfs:
            print(f"  {pdf}")
        print("Re-run with --mode cloud once credentials and a data store exist.")
    else:
        print(
            "Set MULENET_REGULATIONS_PDF_DIR to a directory of public AML/KYC "
            "regulation PDFs to index them."
        )


def main(argv: list[str] | None = None) -> int:
    """Index regulation PDFs, or report what is missing in mock mode."""
    parser = argparse.ArgumentParser(description="Index AML/KYC regulations.")
    parser.add_argument("--mode", choices=("mock", "cloud"), default=None)
    args = parser.parse_args(argv)

    config = DocaiConfig.from_env()
    mode: DocaiMode = args.mode or config.mode
    pdfs = find_pdfs(config.regulations_pdf_dir)

    if mode == "mock":
        _report_mock(config, pdfs)
        return 0

    if not config.regulations_gcs_uri:
        raise SystemExit("cloud mode requires MULENET_REGULATIONS_GCS_URI")
    if not pdfs:
        raise SystemExit(
            "no PDFs found; set MULENET_REGULATIONS_PDF_DIR to a directory of "
            "public AML/KYC regulation PDFs"
        )
    documents = load_corpus_manifest(config.regulations_dir)
    uris = upload_pdfs_to_gcs(pdfs, config.regulations_gcs_uri)
    records = build_import_records(documents, {Path(uri).name: uri for uri in uris})
    metadata_uri = upload_metadata(records, config.regulations_gcs_uri)
    operation = import_to_datastore(config, metadata_uri)
    print(f"mode=cloud uploaded={len(uris)} metadata={metadata_uri}")
    print(f"import operation started: {operation}")
    print(
        "This is asynchronous and took 10m59s on the first run. Poll the "
        "operation and list the branch documents before believing it worked."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
