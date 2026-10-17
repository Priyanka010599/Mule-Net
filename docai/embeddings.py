"""Image embeddings for KYC photo near-duplicate detection.

Two interchangeable backends behind one interface:

``DifferenceHashEmbedder`` (mock)
    A deterministic difference hash computed locally with Pillow. Two
    accounts that submitted *literally the same file* -- which is how
    the simulator models a recycled identity -- produce the same hash,
    so reuse is found exactly and offline.

``VertexMultimodalEmbedder`` (cloud)
    Vertex AI ``multimodalembedding`` vectors, compared by cosine
    similarity.

Near-duplicate search is a single chunked matrix product, so the whole
corpus can be compared without a vector database. At the scale the
simulator produces (a few thousand distinct images) that is cheaper and
far easier to verify than standing up Vector Search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from docai.config import DocaiConfig

#: Similarity metrics a backend can declare.
HAMMING = "hamming"
COSINE = "cosine"


@dataclass(frozen=True)
class ImageEmbedding:
    """One embedded image, keyed by its ``photo_embedding_ref``."""

    ref: str
    vector: tuple[float, ...]
    backend: str


@dataclass(frozen=True)
class PhotoMatch:
    """A near-duplicate pair of image refs with its similarity score."""

    left_ref: str
    right_ref: str
    similarity: float


@runtime_checkable
class ImageEmbedder(Protocol):
    """Embeds an image file into a fixed-width vector."""

    backend: str
    metric: str

    def embed(self, path: Path, ref: str) -> ImageEmbedding:
        """Embed the image at ``path``, labelled with its corpus ``ref``."""
        ...


def difference_hash_bits(path: Path, hash_size: int) -> tuple[float, ...]:
    """Compute a difference hash of the image as a tuple of 0.0/1.0 bits.

    The image is reduced to greyscale on a ``(hash_size + 1) x hash_size``
    grid; each bit records whether a pixel is brighter than its right-hand
    neighbour. Identical files always give identical bits.
    """
    if hash_size < 2:
        raise ValueError(f"hash_size must be at least 2, got {hash_size}")
    with Image.open(path) as img:
        small = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        raw = small.tobytes()
    stride = hash_size + 1
    bits: list[float] = []
    for row in range(hash_size):
        line = raw[row * stride : (row + 1) * stride]
        bits.extend(1.0 if line[col] > line[col + 1] else 0.0 for col in range(hash_size))
    return tuple(bits)


def bits_to_hex(bits: Sequence[float]) -> str:
    """Render hash bits as a hex string, for logging and fixtures."""
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    return f"{value:0{(len(bits) + 3) // 4}x}"


class DifferenceHashEmbedder:
    """Offline embedder: a Pillow difference hash, compared by Hamming distance."""

    backend = "dhash"
    metric = HAMMING

    def __init__(self, hash_size: int) -> None:
        self.hash_size = hash_size

    def embed(self, path: Path, ref: str) -> ImageEmbedding:
        """Hash the image at ``path``."""
        return ImageEmbedding(
            ref=ref,
            vector=difference_hash_bits(path, self.hash_size),
            backend=self.backend,
        )


class VertexMultimodalEmbedder:
    """Cloud embedder: Vertex AI multimodal image embeddings.

    Unverified in this environment -- no Google Cloud credentials are
    configured here, so this path has never been executed. The API shape
    is checked against the installed ``google-cloud-aiplatform`` 2.2.0
    (``vertexai.vision_models.MultiModalEmbeddingModel.get_embeddings``).
    """

    backend = "vertex-multimodal"
    metric = COSINE

    def __init__(self, config: DocaiConfig) -> None:
        config.require_cloud_ids()
        self._config = config
        self._model = None

    def _load_model(self):  # pragma: no cover - requires credentials
        if self._model is None:
            import vertexai
            from vertexai.vision_models import MultiModalEmbeddingModel

            vertexai.init(
                project=self._config.project_id,
                location=self._config.vertex_location,
            )
            self._model = MultiModalEmbeddingModel.from_pretrained(
                self._config.embedding_model
            )
        return self._model

    def embed(self, path: Path, ref: str) -> ImageEmbedding:  # pragma: no cover
        """Embed the image at ``path`` with the Vertex multimodal model."""
        from vertexai.vision_models import Image as VertexImage

        model = self._load_model()
        response = model.get_embeddings(
            image=VertexImage.load_from_file(str(path)),
            dimension=self._config.embedding_dimension,
        )
        if response.image_embedding is None:
            raise RuntimeError(f"Vertex returned no image embedding for {ref}")
        return ImageEmbedding(
            ref=ref,
            vector=tuple(float(v) for v in response.image_embedding),
            backend=self.backend,
        )


def build_embedder(config: DocaiConfig) -> ImageEmbedder:
    """Return the embedder matching ``config.mode``."""
    if config.mode == "cloud":
        return VertexMultimodalEmbedder(config)
    return DifferenceHashEmbedder(config.hash_size)


def embed_refs(
    refs: Iterable[str],
    embedder: ImageEmbedder,
    images_dir: Path,
) -> list[ImageEmbedding]:
    """Embed each distinct ref once, skipping refs whose file is missing.

    Refs are relative paths as stored in ``photo_embedding_ref``; they are
    resolved against ``images_dir``.
    """
    seen: set[str] = set()
    out: list[ImageEmbedding] = []
    for ref in refs:
        if not ref or ref in seen:
            continue
        seen.add(ref)
        path = images_dir / ref
        if not path.is_file():
            continue
        out.append(embedder.embed(path, ref))
    return out


def _similarity_block(
    block: np.ndarray,
    matrix: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Similarity of every row in ``block`` against every row in ``matrix``."""
    if metric == COSINE:
        return block @ matrix.T
    # Hamming similarity on 0/1 vectors: agreements / width, expanded so it
    # needs one matrix product instead of a per-pair XOR.
    width = matrix.shape[1]
    gram = block @ matrix.T
    block_sum = block.sum(axis=1, keepdims=True)
    matrix_sum = matrix.sum(axis=1, keepdims=True).T
    return (2.0 * gram - block_sum - matrix_sum + width) / width


def find_near_duplicates(
    embeddings: Sequence[ImageEmbedding],
    threshold: float,
    metric: str,
    *,
    block_size: int = 512,
) -> list[PhotoMatch]:
    """Return every pair of refs whose similarity is at least ``threshold``.

    Args:
        embeddings: Distinct image embeddings, one per ref.
        threshold: Minimum similarity to report. With the difference-hash
            backend on the synthetic corpus this should stay at 1.0; the
            placeholder cards share a layout, so unrelated cards already
            sit around 0.95.
        metric: :data:`HAMMING` or :data:`COSINE`.
        block_size: Rows compared per matrix product. Bounds peak memory.

    Returns:
        Matches ordered by descending similarity, each ref pair once with
        ``left_ref < right_ref``.
    """
    if metric not in (HAMMING, COSINE):
        raise ValueError(f"unknown metric {metric!r}")
    if len(embeddings) < 2:
        return []

    matrix = np.asarray([e.vector for e in embeddings], dtype=np.float32)
    if metric == COSINE:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

    refs = [e.ref for e in embeddings]
    matches: list[PhotoMatch] = []
    for start in range(0, len(refs), block_size):
        stop = min(start + block_size, len(refs))
        sims = _similarity_block(matrix[start:stop], matrix, metric)
        # Only the strict upper triangle, so each pair is emitted once.
        rows, cols = np.nonzero(sims >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            left = start + row
            if left >= col:
                continue
            first, second = sorted((refs[left], refs[col]))
            matches.append(
                PhotoMatch(
                    left_ref=first,
                    right_ref=second,
                    similarity=float(sims[row, col]),
                )
            )
    matches.sort(key=lambda m: (-m.similarity, m.left_ref, m.right_ref))
    return matches
