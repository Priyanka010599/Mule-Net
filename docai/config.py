"""Environment-driven configuration for the MuleNet ``docai/`` module.

Every path, mode switch and Google Cloud identifier that ``parsing.py``,
``embeddings.py``, ``reuse.py`` and ``regulations.py`` need is resolved
here (see CLAUDE.md: "Config via env vars in one config.py per module").
Nothing in this module reads data or talks to Google Cloud.

Modes
-----
``mock``
    Fully offline. Document AI parsing is derived from the simulator
    metadata, image embeddings are a deterministic local difference
    hash, and regulation search runs over the bundled placeholder
    corpus. This is the default whenever credentials are absent.
``cloud``
    Document AI, Vertex AI multimodal embeddings and Vertex AI Search.
    Selected **only** by an explicit ``MULENET_DOCAI_MODE=cloud``; it is
    never inferred from the presence of credentials. See
    :func:`resolve_mode`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_DOCAI_DIR = Path(__file__).resolve().parent

#: Parquet input written by ``simulator/generator.py``.
DEFAULT_DATA_DIR = _DOCAI_DIR.parent / "simulator" / "out"

#: Where ``reuse_pairs.parquet`` and ``parsed_documents.parquet`` land.
DEFAULT_OUT_DIR = _DOCAI_DIR / "out"

#: Bundled placeholder regulation passages used by mock search.
DEFAULT_REGULATIONS_DIR = _DOCAI_DIR / "regulations"

#: Difference-hash grid edge for the mock image embedder, in pixels.
#: 32 means a 1024-bit hash. Measured on the seed-42 corpus: a 32-grid
#: hash is unique across all 2,991 distinct synthetic ID cards, while an
#: 8-grid hash collides on roughly two thirds of them -- the placeholder
#: cards share a layout and differ only in drawn text and accent colour,
#: so a coarse hash cannot tell them apart.
DEFAULT_HASH_SIZE = 32

#: Minimum similarity for two photos to count as near-duplicates.
#: 1.0 (an exact hash match) is the default in mock mode because the
#: synthetic cards are structurally identical: the median similarity
#: between two *unrelated* cards is already ~0.95, so anything looser
#: would flag most of the corpus. Real Vertex embeddings discriminate on
#: content and tolerate a lower threshold.
DEFAULT_PHOTO_SIMILARITY_THRESHOLD = 1.0

#: Document AI processor location; not a Vertex region.
#: ``asia-south1`` (Mumbai) is used because it is one of the locations
#: ``FORM_PARSER_PROCESSOR`` is offered in -- read off ``processorTypes``
#: for this project on 2026-09-27, which lists asia-south1, us, eu,
#: europe-west3, europe-west2, australia-southeast1, us-east7 and
#: northamerica-northeast1 for that type. It puts document parsing in the
#: same region as Gemini (see CLAUDE.md and docs/residency.md).
#: Note that not every processor type is offered there:
#: ``ID_PROOFING_PROCESSOR`` and ``LAYOUT_PARSER_PROCESSOR`` are us/eu only.
DEFAULT_DOCAI_LOCATION = "asia-south1"

#: Vertex AI region for the multimodal embedding model.
DEFAULT_VERTEX_LOCATION = "us-central1"

#: Vertex AI multimodal embedding model. Verify against current docs
#: before a real run; model ids are versioned and do get retired.
DEFAULT_EMBEDDING_MODEL = "multimodalembedding@001"

#: Embedding width requested from the multimodal model (128/256/408/1408).
DEFAULT_EMBEDDING_DIMENSION = 1408

#: Vertex AI Search (Discovery Engine) serving location. Discovery Engine
#: offers only ``global``, ``us`` and ``eu`` -- there is no asia-south1
#: option for a data store, so the regulation index cannot sit in India.
#: ``global`` is Google's recommended default and carries the full
#: feature set. The corpus is public Government-of-India text, so this is
#: a latency and feature choice, not a residency one.
DEFAULT_SEARCH_LOCATION = "global"

DocaiMode = Literal["mock", "cloud"]

_VALID_MODES: frozenset[str] = frozenset({"mock", "cloud"})


def credentials_available() -> bool:
    """Return whether Application Default Credentials resolve locally.

    Used to *diagnose* an explicitly requested cloud run (see
    :meth:`DocaiConfig.require_cloud_ids`), never to choose the mode --
    see :func:`resolve_mode`. A ``True`` here does not promise that the
    credentials carry the Document AI or Vertex AI roles the cloud path
    needs.

    **Keep this off hot paths.** When ADC is configured,
    ``google.auth.default()`` performs a project lookup that measured
    ~2.3s per call on this machine. It used to run on every
    ``DocaiConfig.from_env()``, which meant merely having ADC installed
    made every config construction seconds long -- and
    ``search_regulations`` builds a config per query. Call it once, when
    a cloud run is already known to be wanted.
    """
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        return Path(explicit).is_file()
    try:
        import google.auth
    except ImportError:
        return False
    try:
        google.auth.default()
    except Exception:
        # Any auth failure means "no usable credentials here".
        return False
    return True


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def resolve_mode() -> DocaiMode:
    """Pick the execution mode from the environment.

    Mock unless ``MULENET_DOCAI_MODE=cloud`` is set explicitly. **Cloud
    is opt-in and is never inferred.**

    In particular the presence of Application Default Credentials does
    not select cloud mode. Credentials being *available* is not the same
    as the operator *intending* to spend money on an API call: inferring
    intent from ambient credentials makes the same command behave
    differently on different machines, so a suite that passes on a
    laptop fails the moment someone runs ``gcloud auth login``. Mode is
    an explicit choice.
    """
    raw = os.environ.get("MULENET_DOCAI_MODE", "").strip().lower()
    if not raw:
        return "mock"
    if raw not in _VALID_MODES:
        raise ValueError(
            f"MULENET_DOCAI_MODE must be one of {sorted(_VALID_MODES)}, got {raw!r}"
        )
    return "cloud" if raw == "cloud" else "mock"


@dataclass(frozen=True)
class DocaiConfig:
    """Resolved settings for KYC parsing, reuse detection and regulations."""

    mode: DocaiMode
    data_dir: Path
    out_dir: Path
    regulations_dir: Path
    hash_size: int
    photo_similarity_threshold: float
    project_id: str | None
    docai_location: str
    processor_id: str | None
    vertex_location: str
    embedding_model: str
    embedding_dimension: int
    search_location: str
    search_datastore_id: str | None
    search_engine_id: str | None
    regulations_pdf_dir: Path | None
    regulations_gcs_uri: str | None

    @classmethod
    def from_env(cls) -> DocaiConfig:
        """Build a config from the process environment, applying defaults."""
        pdf_dir = os.environ.get("MULENET_REGULATIONS_PDF_DIR")
        return cls(
            mode=resolve_mode(),
            data_dir=_env_path("MULENET_DATA_DIR", DEFAULT_DATA_DIR),
            out_dir=_env_path("MULENET_DOCAI_OUT_DIR", DEFAULT_OUT_DIR),
            regulations_dir=_env_path("MULENET_REGULATIONS_DIR", DEFAULT_REGULATIONS_DIR),
            hash_size=_env_int("MULENET_DOCAI_HASH_SIZE", DEFAULT_HASH_SIZE),
            photo_similarity_threshold=_env_float(
                "MULENET_PHOTO_SIMILARITY_THRESHOLD", DEFAULT_PHOTO_SIMILARITY_THRESHOLD
            ),
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            docai_location=os.environ.get("MULENET_DOCAI_LOCATION", DEFAULT_DOCAI_LOCATION),
            processor_id=os.environ.get("MULENET_DOCAI_PROCESSOR_ID") or None,
            vertex_location=os.environ.get("MULENET_VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
            embedding_model=os.environ.get("MULENET_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimension=_env_int(
                "MULENET_EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION
            ),
            search_location=os.environ.get("MULENET_SEARCH_LOCATION", DEFAULT_SEARCH_LOCATION),
            search_datastore_id=os.environ.get("MULENET_REGULATIONS_DATASTORE") or None,
            search_engine_id=os.environ.get("MULENET_REGULATIONS_ENGINE") or None,
            regulations_pdf_dir=Path(pdf_dir).expanduser().resolve() if pdf_dir else None,
            regulations_gcs_uri=os.environ.get("MULENET_REGULATIONS_GCS_URI") or None,
        )

    @property
    def documents_path(self) -> Path:
        """``kyc_documents.parquet`` from the simulator."""
        return self.data_dir / "kyc_documents.parquet"

    @property
    def submissions_path(self) -> Path:
        """``kyc_submissions.parquet`` from the simulator (carries ``bank_id``)."""
        return self.data_dir / "kyc_submissions.parquet"

    @property
    def images_dir(self) -> Path:
        """Root that ``photo_embedding_ref`` values are relative to."""
        return self.data_dir

    @property
    def reuse_pairs_path(self) -> Path:
        """Output of ``reuse.py``."""
        return self.out_dir / "reuse_pairs.parquet"

    @property
    def parsed_documents_path(self) -> Path:
        """Output of ``parsing.py``."""
        return self.out_dir / "parsed_documents.parquet"

    @property
    def account_evidence_path(self) -> Path:
        """``shared.contracts.AccountEvidence`` rows derived from reuse pairs.

        Same schema as ``ml/out/account_evidence.parquet``; this file is
        the ``source_table="kyc_documents"`` half of it.
        """
        return self.out_dir / "account_evidence.parquet"

    def require_cloud_ids(
        self, *, processor: bool = False, datastore: bool = False, engine: bool = False
    ) -> None:
        """Fail fast when a cloud call is missing something it needs.

        Raises:
            ValueError: If a required identifier is unset.
            RuntimeError: If cloud mode was requested but Application
                Default Credentials do not resolve. This is the one place
                credentials are checked -- the operator has already said
                ``cloud``, so an unusable environment is a loud failure
                rather than a silent fall back to mock.
        """
        missing: list[str] = []
        if not self.project_id:
            missing.append("GOOGLE_CLOUD_PROJECT")
        if processor and not self.processor_id:
            missing.append("MULENET_DOCAI_PROCESSOR_ID")
        if datastore and not self.search_datastore_id:
            missing.append("MULENET_REGULATIONS_DATASTORE")
        if engine and not self.search_engine_id:
            missing.append("MULENET_REGULATIONS_ENGINE")
        if missing:
            raise ValueError(
                "cloud mode requires these environment variables: " + ", ".join(missing)
            )
        if not credentials_available():
            raise RuntimeError(
                "MULENET_DOCAI_MODE=cloud was requested but Application Default "
                "Credentials do not resolve. Run `gcloud auth application-default "
                "login`, or set GOOGLE_APPLICATION_CREDENTIALS, or unset "
                "MULENET_DOCAI_MODE to run offline in mock mode."
            )
