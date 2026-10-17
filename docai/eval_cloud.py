"""Measure the cloud path against the offline one on a bounded sample.

Answers two questions with numbers rather than hope:

1. **Extraction.** Does Document AI read the holder name, document number
   and address off a rendered KYC document correctly? Ground truth is the
   simulator's own renderer -- ``simulator.kyc`` is imported rather than
   reimplemented, so the expected values are the ones actually drawn.
2. **Reuse detection.** Do Vertex AI multimodal embeddings separate two
   *unrelated* documents that the offline difference hash cannot tell
   apart? The offline hash is a layout hash and the synthetic cards share
   a layout: one pair of unrelated address proofs hashes identically,
   producing 2 false cross-bank photo pairs out of 1,792 on the full
   corpus. That pair is always included in the sample, because a sample
   that excluded it could not answer the question.

Every live call costs money, so the sample is capped and the run prints
what it spent. Usage is counted exactly; the rates are list prices read
from the Cloud Billing Catalog API for this project on 2026-09-27.

Usage::

    set MULENET_DOCAI_MODE=cloud
    set GOOGLE_CLOUD_PROJECT=mulenet
    set MULENET_DOCAI_PROCESSOR_ID=...
    python -m docai.eval_cloud --limit 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from docai.config import DocaiConfig
from docai.embeddings import (
    COSINE,
    DifferenceHashEmbedder,
    ImageEmbedding,
    VertexMultimodalEmbedder,
    find_near_duplicates,
)
from docai.parsing import (
    ADDRESS_LABELS,
    ID_NUMBER_LABEL,
    NAME_LABELS,
    DocumentAiParser,
    DocumentRecord,
    extract_labelled_fields,
    first_labelled,
    load_document_records,
    normalize_address,
)

#: USD per page for the Form Parser. Billing Catalog SKU "Number of
#: 'general' type pages processed", first 1M pages.
FORM_PARSER_USD_PER_PAGE = 0.03

#: USD per image for ``multimodalembedding``. Billing Catalog SKU
#: "Embeddings for multimodal - Image (input)".
MULTIMODAL_EMBEDDING_USD_PER_IMAGE = 0.0001

#: The two unrelated address proofs whose 32-grid difference hashes are
#: identical on the seed-42 corpus. Measured, not assumed: they are the
#: only false photo pair the offline backend produces, and they are the
#: whole reason to ask whether real embeddings do better. Always sampled.
DHASH_COLLIDING_REFS: tuple[str, str] = (
    "kyc_images/05a77b37192440b1_address_proof.png",
    "kyc_images/eb2f3bf3e3880d8a_address_proof.png",
)


@dataclass(frozen=True)
class ExtractionResult:
    """One document's extracted fields against what the renderer drew."""

    ref: str
    doc_type: str
    latency_s: float
    name_ok: bool
    id_number_ok: bool
    address_ok: bool
    extracted_name: str | None
    extracted_id_number: str | None
    extracted_address: str | None
    expected_name: str
    expected_id_number: str | None
    expected_address: str


def expected_fields(record: DocumentRecord) -> tuple[str, str | None, str]:
    """The name, document number and address the simulator drew on this document.

    Imported from ``simulator.kyc`` rather than reimplemented: the
    renderer is the source of truth for what is on the image, and a
    hand-copied expectation would test this module against itself.

    The document number is rendered only on ID cards, so it is ``None``
    for an address proof. The identity key is recoverable from the image
    filename, which the simulator names ``<identity_key[:16]>_<type>.png``.
    """
    from simulator.kyc import display_id_number, holder_label

    ref = record.photo_embedding_ref or ""
    identity_key = Path(ref).stem.split("_")[0]
    id_number = (
        display_id_number(record.hashed_id_number) if record.doc_type == "id_card" else None
    )
    return holder_label(identity_key), id_number, normalize_address(record.normalized_address)


def select_sample(config: DocaiConfig, limit: int) -> list[DocumentRecord]:
    """Pick ``limit`` documents deterministically, one per distinct image.

    The two colliding refs come first so they survive any ``limit``; the
    rest are taken in sorted ref order, which interleaves the two
    document types because both share an identity key prefix.

    Raises:
        ValueError: If ``limit`` is smaller than the mandatory refs.
    """
    if limit < len(DHASH_COLLIDING_REFS):
        raise ValueError(f"limit must be at least {len(DHASH_COLLIDING_REFS)}, got {limit}")

    by_ref: dict[str, DocumentRecord] = {}
    for record in load_document_records(config):
        ref = record.photo_embedding_ref
        if ref and ref not in by_ref and (config.images_dir / ref).is_file():
            by_ref[ref] = record

    missing = [ref for ref in DHASH_COLLIDING_REFS if ref not in by_ref]
    if missing:
        raise RuntimeError(f"expected colliding refs are not in the corpus: {missing}")

    ordered = list(DHASH_COLLIDING_REFS) + [
        ref for ref in sorted(by_ref) if ref not in DHASH_COLLIDING_REFS
    ]
    return [by_ref[ref] for ref in ordered[:limit]]


def evaluate_extraction(
    records: Sequence[DocumentRecord], parser: DocumentAiParser, images_dir: Path
) -> list[ExtractionResult]:
    """Run each document through Document AI and score its fields."""
    results: list[ExtractionResult] = []
    for record in records:
        ref = record.photo_embedding_ref or ""
        started = time.perf_counter()
        document = parser.process(images_dir / ref)
        latency = time.perf_counter() - started

        fields = extract_labelled_fields(getattr(document, "text", "") or "")
        name = first_labelled(fields, NAME_LABELS)
        id_number = fields.get(ID_NUMBER_LABEL)
        address = first_labelled(fields, ADDRESS_LABELS)
        want_name, want_id, want_address = expected_fields(record)

        results.append(
            ExtractionResult(
                ref=ref,
                doc_type=record.doc_type,
                latency_s=latency,
                name_ok=name == want_name,
                # An address proof carries no document number, so "correct"
                # means the parser did not invent one.
                id_number_ok=(id_number or None) == want_id,
                address_ok=normalize_address(address or "") == want_address,
                extracted_name=name,
                extracted_id_number=id_number,
                extracted_address=address,
                expected_name=want_name,
                expected_id_number=want_id,
                expected_address=want_address,
            )
        )
    return results


def _similarity(left: ImageEmbedding, right: ImageEmbedding, metric: str) -> float:
    """Cosine or Hamming similarity between two embeddings."""
    a = np.asarray(left.vector, dtype=np.float64)
    b = np.asarray(right.vector, dtype=np.float64)
    if metric == COSINE:
        norm = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(a @ b / norm)
    return float((a == b).mean())


def compare_reuse_backends(
    refs: Sequence[str], images_dir: Path, config: DocaiConfig
) -> dict[str, object]:
    """Score both photo backends over every pair of a distinct-image sample.

    Every ref is a different file belonging to a different identity, so
    **every pair here is a true negative**: any pair a backend reports as
    a near-duplicate is a false positive. That makes the false-pair rate
    directly comparable with the 2-in-1,792 (0.11%) the offline hash
    produces on the full corpus.

    The same-file case is checked separately by embedding one sampled
    image twice: recycled identities in this simulator point at literally
    the same file, so a backend that does not return 1.0 for a file
    against itself would miss the reuse that matters.
    """
    dhash = DifferenceHashEmbedder(config.hash_size)
    vertex = VertexMultimodalEmbedder(config)

    dhash_embeddings = [dhash.embed(images_dir / ref, ref) for ref in refs]
    vertex_embeddings = [vertex.embed(images_dir / ref, ref) for ref in refs]

    repeat = vertex.embed(images_dir / refs[0], refs[0])
    same_file_similarity = _similarity(vertex_embeddings[0], repeat, COSINE)

    by_ref = {embedding.ref: embedding for embedding in vertex_embeddings}
    left, right = DHASH_COLLIDING_REFS

    pairs = list(combinations(refs, 2))
    dhash_by_ref = {embedding.ref: embedding for embedding in dhash_embeddings}
    dhash_sims = [_similarity(dhash_by_ref[a], dhash_by_ref[b], "hamming") for a, b in pairs]
    vertex_sims = [_similarity(by_ref[a], by_ref[b], COSINE) for a, b in pairs]

    dhash_false = find_near_duplicates(
        dhash_embeddings, config.photo_similarity_threshold, dhash.metric
    )
    return {
        "n_images": len(refs),
        "n_pairs": len(pairs),
        "dhash": {
            "false_pairs_at_1.0": len(dhash_false),
            "colliding_pair_similarity": _similarity(
                dhash_by_ref[left], dhash_by_ref[right], "hamming"
            ),
            "max_pair_similarity": max(dhash_sims),
            "median_pair_similarity": statistics.median(dhash_sims),
        },
        "vertex": {
            "embedding_dimension": len(vertex_embeddings[0].vector),
            "same_file_similarity": same_file_similarity,
            "colliding_pair_similarity": _similarity(by_ref[left], by_ref[right], COSINE),
            "max_pair_similarity": max(vertex_sims),
            "median_pair_similarity": statistics.median(vertex_sims),
            "false_pairs_at_0.99": sum(1 for s in vertex_sims if s >= 0.99),
            "false_pairs_at_0.95": sum(1 for s in vertex_sims if s >= 0.95),
        },
    }


def summarize(
    extraction: Sequence[ExtractionResult], reuse: dict[str, object]
) -> dict[str, object]:
    """Roll the per-document results up into the reported numbers."""
    total = len(extraction)
    latencies = sorted(result.latency_s for result in extraction)
    id_cards = [result for result in extraction if result.doc_type == "id_card"]
    docai_usd = total * FORM_PARSER_USD_PER_PAGE
    # One embedding per image, plus the repeat used for the same-file check.
    embedding_calls = total + 1
    embedding_usd = embedding_calls * MULTIMODAL_EMBEDDING_USD_PER_IMAGE
    return {
        "documents": total,
        "extraction": {
            "name_accuracy": sum(r.name_ok for r in extraction) / total,
            "address_accuracy": sum(r.address_ok for r in extraction) / total,
            "id_number_accuracy": sum(r.id_number_ok for r in extraction) / total,
            "id_number_accuracy_id_cards_only": (
                sum(r.id_number_ok for r in id_cards) / len(id_cards) if id_cards else None
            ),
            "all_fields_correct": sum(
                r.name_ok and r.address_ok and r.id_number_ok for r in extraction
            )
            / total,
        },
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": latencies[0],
            "max": latencies[-1],
            "total": sum(latencies),
        },
        "cost_usd": {
            "document_ai_pages": total,
            "document_ai": docai_usd,
            "embedding_calls": embedding_calls,
            "embeddings": embedding_usd,
            "total": docai_usd + embedding_usd,
            "per_document": (docai_usd + embedding_usd) / total,
        },
        "reuse": reuse,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the bounded cloud evaluation and print a JSON report."""
    parser = argparse.ArgumentParser(description="Evaluate the docai cloud path.")
    parser.add_argument("--limit", type=int, default=50, help="documents to process")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    config = DocaiConfig.from_env()
    if config.mode != "cloud":
        raise SystemExit("this evaluation only runs with MULENET_DOCAI_MODE=cloud")

    records = select_sample(config, args.limit)
    refs = [record.photo_embedding_ref or "" for record in records]
    extraction = evaluate_extraction(records, DocumentAiParser(config), config.images_dir)
    reuse = compare_reuse_backends(refs, config.images_dir, config)

    report = summarize(extraction, reuse)
    report["per_document"] = [asdict(result) for result in extraction]
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
