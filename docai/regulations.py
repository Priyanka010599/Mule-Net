"""AML/KYC regulation search.

Implements the ``search_regulations`` tool stub declared in
``shared/tool_interfaces.py``: same signature, same result keys, same
exceptions.

Backends
--------
``MockRegulationSearch`` (mock)
    Ranks the bundled placeholder passages by term overlap. Offline.
``VertexRegulationSearch`` (cloud)
    Queries a Vertex AI Search (Discovery Engine) data store built by
    ``docai/index_regulations.py`` over the four official AML/KYC PDFs
    listed in ``regulations/corpus_manifest.json``.

**Mock is the default**, and every bundled mock passage is a
clearly-labelled placeholder with an invented citation; nothing drafted
from them may be filed or shown to a regulator.

Provenance is the whole point of the cloud path. A live report agent
once emitted five citations that all pointed at a URL the tool had never
returned, so the rule here is mechanical: a citation string is only ever
built from a document the API actually returned *and* that appears in
the corpus manifest. Nothing is composed from the query, and a result
that cannot be attributed to a manifest document is dropped rather than
rendered.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from docai.config import DocaiConfig

#: Filename of the bundled placeholder corpus inside ``regulations_dir``.
PLACEHOLDER_CORPUS_FILE = "placeholder_corpus.json"

#: Filename of the real-corpus manifest inside ``regulations_dir``.
CORPUS_MANIFEST_FILE = "corpus_manifest.json"

#: Marker present in every bundled citation, so a caller can tell at a
#: glance that a result is not a real regulation.
PLACEHOLDER_MARKER = "PLACEHOLDER"

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words too common in AML prose to carry ranking signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
        "were", "which", "with", "should", "shall", "may", "placeholder",
        "text", "testing", "only", "passage", "synthetic", "not", "real",
    }
)


@dataclass(frozen=True)
class RegulationPassage:
    """One indexed passage of a regulation document."""

    doc_name: str
    section: str
    citation: str
    text: str
    source_uri: str

    def as_result(self) -> dict[str, str]:
        """Render as a ``search_regulations`` result row.

        ``citation``, ``excerpt`` and ``source_uri`` are the keys the
        shared tool interface documents. ``doc_name`` and ``section`` are
        added alongside them because the docai brief asks for the
        document name and section to be returned separately -- see the
        contract note in ``docai/README.md``.
        """
        return {
            "citation": self.citation,
            "excerpt": self.text,
            "source_uri": self.source_uri,
            "doc_name": self.doc_name,
            "section": self.section,
        }


@dataclass(frozen=True)
class CorpusDocument:
    """One official regulation PDF indexed into Vertex AI Search.

    The metadata that travels with the PDF. ``title`` and ``source_url``
    are what a citation must carry: the ``gs://`` object is not a
    citable source, the issuing authority's URL is.
    """

    doc_id: str
    filename: str
    title: str
    short_title: str
    authority: str
    version: str
    source_url: str
    size_bytes: int


def load_corpus_manifest(regulations_dir: Path) -> list[CorpusDocument]:
    """Read the official-corpus manifest from disk.

    This is the closed set of documents a citation may name. It is read
    from the repo, not from the index, so an unexpected document in the
    data store cannot widen what a report is allowed to cite.

    Raises:
        RuntimeError: If the manifest is missing, malformed or empty.
    """
    path = regulations_dir / CORPUS_MANIFEST_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"regulation corpus manifest not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"regulation corpus manifest at {path} is not valid JSON") from exc
    try:
        documents = [CorpusDocument(**entry) for entry in raw["documents"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"regulation corpus manifest at {path} has an unexpected shape"
        ) from exc
    if not documents:
        raise RuntimeError(f"regulation corpus manifest at {path} lists no documents")
    return documents


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def load_placeholder_corpus(regulations_dir: Path) -> list[RegulationPassage]:
    """Read the bundled placeholder passages from disk.

    Raises:
        RuntimeError: If the bundled corpus file is missing or malformed.
    """
    path = regulations_dir / PLACEHOLDER_CORPUS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"placeholder regulation corpus not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"placeholder regulation corpus at {path} is not valid JSON") from exc
    try:
        return [RegulationPassage(**passage) for passage in raw["passages"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"placeholder regulation corpus at {path} has an unexpected shape")\
            from exc


@runtime_checkable
class RegulationSearch(Protocol):
    """Ranks regulation passages against a natural-language query."""

    backend: str

    def search(self, query: str, top_k: int) -> list[RegulationPassage]:
        """Return up to ``top_k`` passages, most relevant first."""
        ...


class MockRegulationSearch:
    """Offline search over the bundled placeholder passages.

    Ranking is inverse-document-frequency weighted term overlap,
    length-normalized. It is deliberately simple: its job is to return
    plausibly ordered passages in the right shape, not to stand in for a
    real retrieval system.
    """

    backend = "placeholder"

    def __init__(self, passages: list[RegulationPassage]) -> None:
        if not passages:
            raise RuntimeError("regulation corpus is empty")
        self._passages = passages
        self._tokens = [Counter(_tokenize(f"{p.section} {p.text}")) for p in passages]
        document_frequency: Counter[str] = Counter()
        for counts in self._tokens:
            document_frequency.update(counts.keys())
        total = len(passages)
        self._idf = {
            term: math.log(1.0 + total / (1 + freq)) for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[RegulationPassage]:
        """Rank the corpus by weighted term overlap with ``query``."""
        terms = _tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._tokens):
            length = sum(counts.values()) or 1
            score = sum(
                self._idf.get(term, 0.0) * counts[term] for term in terms if term in counts
            )
            if score > 0.0:
                scored.append((score / math.sqrt(length), index))
        # Descending score, then corpus order, so results are deterministic.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._passages[index] for _, index in scored[:top_k]]


class VertexRegulationSearch:
    """Vertex AI Search (Discovery Engine) backend.

    Queries the ``mulenet-aml-regulations`` data store built by
    ``docai/index_regulations.py`` over the Discovery Engine REST API,
    using Application Default Credentials. REST rather than
    ``google-cloud-discoveryengine`` because that client is not in
    ``pyproject.toml`` and this module may not add it; ``requests`` and
    ``google-auth`` are already dependencies.

    Every returned passage is checked against
    :func:`load_corpus_manifest` before it is handed back. A result whose
    document id is not in the manifest is dropped, so the corpus a
    caller can cite is a closed, four-document set rather than whatever
    the index happens to contain.
    """

    backend = "vertex-ai-search"

    #: Discovery Engine REST host. Only ``global``, ``us`` and ``eu``
    #: exist; the regional hosts are prefixed, e.g. ``us-``.
    _HOST = "discoveryengine.googleapis.com"

    def __init__(self, config: DocaiConfig) -> None:
        # The engine, not the data store: a search against a data store
        # serving config is refused with HTTP 400 "Cannot use enterprise
        # edition features (... extractive answers/segments ...) in a
        # standard edition search engine", because Enterprise edition is
        # only enabled at the app level. Extractive segments are what
        # carry the page number, so there is no usable data-store path.
        config.require_cloud_ids(engine=True)
        self._config = config
        self._documents = {
            doc.doc_id: doc for doc in load_corpus_manifest(config.regulations_dir)
        }
        self._credentials = None

    def _serving_config(self) -> str:
        """The search app's (engine's) default serving config."""
        cfg = self._config
        return (
            f"projects/{cfg.project_id}/locations/{cfg.search_location}"
            f"/collections/default_collection/engines/{cfg.search_engine_id}"
            "/servingConfigs/default_config"
        )

    def _endpoint(self) -> str:
        location = self._config.search_location
        prefix = "" if location == "global" else f"{location}-"
        return f"https://{prefix}{self._HOST}/v1/{self._serving_config()}:search"

    def _token(self) -> str:  # pragma: no cover - requires credentials
        import google.auth
        from google.auth.transport.requests import Request

        if self._credentials is None:
            self._credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return str(self._credentials.token)

    def search(self, query: str, top_k: int) -> list[RegulationPassage]:  # pragma: no cover
        """Query the data store and map each result to a passage."""
        import requests

        try:
            response = requests.post(
                self._endpoint(),
                headers={
                    "Authorization": f"Bearer {self._token()}",
                    "X-Goog-User-Project": str(self._config.project_id),
                },
                json={
                    "query": query,
                    "pageSize": top_k,
                    "contentSearchSpec": {
                        "extractiveContentSpec": {
                            "maxExtractiveSegmentCount": 1,
                            "returnExtractiveSegmentScore": True,
                        }
                    },
                },
                timeout=60,
            )
        except Exception as exc:
            raise RuntimeError(f"Vertex AI Search query failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"Vertex AI Search query failed: HTTP {response.status_code} "
                f"{response.text[:400]}"
            )
        results = response.json().get("results", [])
        passages = [
            passage
            for passage in (
                passage_from_search_result(result, self._documents) for result in results
            )
            if passage is not None
        ]
        return passages[:top_k]


def passage_from_search_result(
    result: Any, documents: Mapping[str, CorpusDocument]
) -> RegulationPassage | None:
    """Map one Discovery Engine REST result onto a :class:`RegulationPassage`.

    Returns ``None`` when the result cannot be attributed to a document
    in the manifest, or when it carries no extractive text. Both are
    provenance failures: a passage with no identified source document,
    or no quotable text, must not reach a report as a citation.

    Every field of the returned passage comes from the API response or
    from the manifest entry it resolved to. Nothing is composed from the
    query or from a model, so a citation string exists only if a real
    retrieval produced it.
    """
    document = (result or {}).get("document") or {}
    doc_id = str(document.get("id") or "")
    entry = documents.get(doc_id)
    if entry is None:
        return None

    derived = document.get("derivedStructData") or {}
    segments = derived.get("extractive_segments") or derived.get("extractiveSegments") or []
    if not segments:
        return None
    first = segments[0]
    excerpt = str(first.get("content") or "").strip()
    if not excerpt:
        return None
    page = str(first.get("pageNumber") or first.get("page_number") or "").strip()

    section = f"p. {page}" if page else "page not reported"
    return RegulationPassage(
        doc_name=entry.title,
        section=section,
        citation=f"{entry.short_title}, {section}",
        text=excerpt,
        source_uri=entry.source_url,
    )


def build_search(config: DocaiConfig) -> RegulationSearch:
    """Return the search backend matching ``config.mode``."""
    if config.mode == "cloud":
        return VertexRegulationSearch(config)
    return MockRegulationSearch(load_placeholder_corpus(config.regulations_dir))


def search_regulations(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Search the AML regulation corpus for passages to cite in a report.

    Implements the tool stub in ``shared/tool_interfaces.py``. In mock
    mode the corpus is the bundled placeholder set: every result is
    clearly marked and none of it is a real regulation.

    Args:
        query: Natural-language description of the suspicious behaviour.
        top_k: Maximum number of passages to return.

    Returns:
        A list of results ordered by relevance, each with the keys
        ``"citation"`` (regulation and section reference), ``"excerpt"``
        (the matching text) and ``"source_uri"``, plus ``"doc_name"``
        and ``"section"`` broken out separately.

    Raises:
        ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        RuntimeError: If the search backend is unavailable.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    backend = build_search(DocaiConfig.from_env())
    return [passage.as_result() for passage in backend.search(query, top_k)]
