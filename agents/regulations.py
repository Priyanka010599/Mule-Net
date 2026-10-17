"""Regulation passage retrieval for the report agent.

``docai/`` owns the real Vertex AI Search index. Until it exposes a
callable, this module serves the same interface from a small built-in
corpus so the report agent, the verifier and the eval harness all run
offline. :func:`search` prefers ``docai`` whenever it becomes importable,
so adopting the real index needs no change here or in ``agents/tools.py``.

The mock corpus below is for that offline path only -- tests, eval, and
a deployment that never set ``MULENET_DOCAI_MODE=cloud`` in the first
place. Once cloud mode is requested, a failure to reach the real index
(``docai`` not importable, or it raising) is never silently absorbed
into a mock answer; see :class:`RegulationSearchUnavailable`. A real
production bug motivated this: ``docai/`` was simply missing from the
agents Docker image, so every cloud-mode call quietly answered from
this corpus's US/FATF passages instead of the real RBI/PMLA ones, with
no error anywhere -- the search "succeeded".

The corpus is paraphrased and deliberately short. It exists so every
ground of suspicion in a drafted report can carry a citation; it is not
legal advice and is not a substitute for the real index.

One passage exists specifically to constrain the report agent:
``FATF-R10-CDD-LIMITS`` states that a shared identifier is a trigger for
further scrutiny and not evidence of wrongdoing. Identity reuse is a
weak signal in this dataset -- far more legitimate accounts share
documents than mule accounts do -- so a report that leans on it must
carry that caveat.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)

#: Marker on every mock ``source_uri`` so a reader can tell at a glance
#: that a citation did not come from the real index.
MOCK_SOURCE_PREFIX = "mock://aml-corpus/"


class RegulationSearchUnavailable(RuntimeError):
    """Cloud mode was requested and the real index could not answer.

    Raised instead of silently returning this module's built-in mock
    corpus. The gap that motivated this: ``docai`` not being
    importable (or failing to initialise -- an unset
    ``MULENET_REGULATIONS_ENGINE``, missing credentials) looked
    identical to "the real backend ran and found nothing", because
    both paths used to return a normal result list -- the mock
    corpus's. A deployment that declared ``MULENET_DOCAI_MODE=cloud``
    has no honest reading of that as "search succeeded".
    """

#: The built-in corpus. Each entry becomes one search result.
CORPUS: list[dict[str, str]] = [
    {
        "citation": "31 CFR 1020.320(a)(2) - Suspicious transaction reporting",
        "excerpt": (
            "A bank shall file a report of any suspicious transaction conducted "
            "or attempted by, at, or through the bank that involves or aggregates "
            "at least $5,000 in funds and where the bank knows, suspects, or has "
            "reason to suspect that the transaction has no business or apparent "
            "lawful purpose, or is not the sort in which the particular customer "
            "would normally be expected to engage."
        ),
        "keywords": (
            "suspicious transaction report sar filing obligation "
            "threshold no lawful purpose"
        ),
    },
    {
        "citation": "31 CFR 1010.314 - Structuring prohibited",
        "excerpt": (
            "No person shall structure or assist in structuring any transaction "
            "with one or more domestic financial institutions for the purpose of "
            "evading the currency transaction reporting requirements, including "
            "by breaking a single sum into multiple smaller transfers."
        ),
        "keywords": "structuring smurfing split amounts below threshold evade reporting ctr",
    },
    {
        "citation": "FATF Recommendation 20 - Reporting of suspicious transactions",
        "excerpt": (
            "If a financial institution suspects or has reasonable grounds to "
            "suspect that funds are the proceeds of criminal activity, it should "
            "be required to report promptly its suspicions to the financial "
            "intelligence unit. Grounds for suspicion should be documented and "
            "based on the institution's own analysis of the transaction pattern."
        ),
        "keywords": "reasonable grounds suspicion proceeds criminal report fiu document grounds",
    },
    {
        "citation": "FATF Recommendation 16 - Wire transfers and intermediaries",
        "excerpt": (
            "Financial institutions should monitor wire transfers for the purpose "
            "of detecting those which lack required originator or beneficiary "
            "information, and should apply enhanced scrutiny where funds move "
            "rapidly through an intermediary account to a further institution "
            "without an evident commercial rationale."
        ),
        "keywords": (
            "wire transfer intermediary pass-through passthrough rapid movement "
            "cross-bank cross bank layering onward remittance velocity"
        ),
    },
    {
        "citation": (
            "FATF Recommendation 10 - Customer due diligence, "
            "limits of identifier matching"
        ),
        "excerpt": (
            "Where a customer identifier, address or device is shared with other "
            "customers, the institution should treat the match as a trigger for "
            "further enquiry rather than as a conclusion. Shared identifiers arise "
            "legitimately among family members, shared residences, joint business "
            "operations and customers holding accounts at more than one "
            "institution, and must be corroborated by transaction evidence before "
            "any determination of suspicion is made."
        ),
        "keywords": (
            "shared identifier document reuse device address family joint household "
            "dual bank legitimate corroborate not conclusive weak signal identity"
        ),
    },
    {
        "citation": "BSA/AML Examination Manual - Funnel and pass-through accounts",
        "excerpt": (
            "A funnel account receives deposits from multiple unrelated parties, "
            "often in geographically dispersed locations, which are then withdrawn "
            "or transferred onward in a short period. High fan-in combined with "
            "prompt onward transfer and a balance that returns to near zero is a "
            "recognised indicator warranting investigation."
        ),
        "keywords": (
            "funnel account fan-in fan in multiple senders unrelated deposits "
            "onward transfer zero balance collector aggregation"
        ),
    },
    {
        "citation": "BSA/AML Examination Manual - Money mule typologies",
        "excerpt": (
            "Money mule accounts typically show a short interval between account "
            "opening and first high-value activity, transaction volumes "
            "inconsistent with the stated customer profile, and coordinated timing "
            "with other accounts. Networks frequently span multiple institutions "
            "so that no single institution observes the complete pattern."
        ),
        "keywords": (
            "money mule ring network newly opened account inconsistent profile "
            "coordinated timing multiple institutions partial view velocity"
        ),
    },
    {
        "citation": "31 CFR 1010.610 - Correspondent accounts and enhanced due diligence",
        "excerpt": (
            "A covered financial institution shall apply enhanced scrutiny to "
            "accounts used as conduits for funds whose origin or destination lies "
            "outside the institution's own customer relationships, including "
            "reasonable steps to ascertain the identity of the ultimate "
            "beneficiaries of the funds."
        ),
        "keywords": (
            "conduit ultimate beneficiary enhanced due diligence "
            "origin destination cross institution"
        ),
    },
    {
        "citation": "FATF Recommendation 11 - Record keeping",
        "excerpt": (
            "Financial institutions should maintain, for at least five years, all "
            "necessary records on transactions and the results of any analysis "
            "undertaken, so that the evidence underlying a suspicion can be "
            "reconstructed and made available to competent authorities."
        ),
        "keywords": "record keeping retention evidence reconstruct analysis audit trail five years",
    },
    {
        "citation": "31 CFR 1020.220 - Customer identification programme",
        "excerpt": (
            "A bank's customer identification programme must include procedures "
            "for verifying the identity of each customer to the extent reasonable "
            "and practicable, and for responding to circumstances in which the "
            "bank cannot form a reasonable belief that it knows the true identity "
            "of a customer."
        ),
        "keywords": (
            "customer identification programme cip verify identity "
            "kyc onboarding true identity"
        ),
    },
]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _relevance(query_tokens: set[str], entry: dict[str, str]) -> float:
    """Token-overlap score, weighting the curated keywords over the body."""
    keyword_tokens = set(_tokens(entry["keywords"]))
    body_tokens = set(_tokens(entry["excerpt"] + " " + entry["citation"]))
    return 2.0 * len(query_tokens & keyword_tokens) + len(query_tokens & body_tokens)


#: Substrings that mark a citation as not a real regulation. ``docai``'s
#: placeholder corpus stamps these into every citation it returns, and a
#: report must repeat the warning rather than pass them off as genuine.
SYNTHETIC_MARKERS: tuple[str, ...] = ("PLACEHOLDER", "SYNTHETIC", "mock://")


#: Lazily built ``doc title -> issuing authority`` map, for enriching a
#: citation card. ``None`` until first use; ``{}`` if the manifest
#: cannot be read (mock mode, or docai missing), cached either way so a
#: card-heavy chat session does not re-read the manifest file per card.
_AUTHORITY_CACHE: dict[str, str] | None = None


def authority_for(doc_name: str) -> str:
    """The issuing body for a real citation's document title, or ``""``.

    Best-effort and never raises: this enriches a citation card for
    display, it does not decide whether a citation is real (that is
    :func:`verify_citation`'s job, against the retrieval ledger, not
    this lookup).
    """
    global _AUTHORITY_CACHE
    if _AUTHORITY_CACHE is None:
        _AUTHORITY_CACHE = {}
        try:
            from docai.config import DocaiConfig  # noqa: PLC0415
            from docai.regulations import load_corpus_manifest  # noqa: PLC0415

            config = DocaiConfig.from_env()
            for doc in load_corpus_manifest(config.regulations_dir):
                _AUTHORITY_CACHE[doc.title] = doc.authority
        except Exception:  # noqa: BLE001 - display enrichment, never fatal
            pass
    return _AUTHORITY_CACHE.get(doc_name, "")


def _docai_search() -> Callable[..., Any] | None:
    """Return ``docai``'s implementation if it exists, else ``None``.

    ``docai.regulations`` owns the corpus and the Vertex AI Search path.
    Probing for the callable on each call means adopting the real index
    needs no change here.
    """
    try:
        from docai.regulations import (  # noqa: PLC0415 - deliberate optional import
            search_regulations,
        )
    except ImportError:
        return None
    return search_regulations if callable(search_regulations) else None


def docai_available() -> bool:
    """True when ``docai.regulations`` exposes a ``search_regulations``."""
    return _docai_search() is not None


#: Citations this process has actually seen the corpus return, newest
#: last. :func:`search` appends to it and :func:`verify_citation` checks
#: it, so a citation is verified by *identity* -- it is byte-for-byte one
#: the retrieval layer handed out -- rather than by asking the index to
#: rank it back up again.
#:
#: Why not a by-id fetch from Vertex AI Search? Measured against the
#: live ``mulenet-aml-regulations`` data store on 2026-09-27:
#:
#: * ``documents.get`` exists in the GA ``v1`` surface
#:   (``GET .../branches/{branch}/documents/{id}``) and answers by id:
#:   ``pmla_act_2002`` returns 200 with ``structData.short_title`` =
#:   ``"PMLA, 2002"`` and ``indexTime`` set, and an invented id returns
#:   404 NOT_FOUND. But a ``Document`` carries only metadata and a
#:   ``gs://`` pointer -- no page text -- so it can confirm the document
#:   half of a citation and never the page half. On its own it would
#:   pass a fabricated ``"PMLA, 2002, p. 9999"``.
#: * ``chunks.get`` / ``chunks.list`` are the only by-id path to passage
#:   *text*, and they exist solely in ``v1alpha``. Called on this data
#:   store they return 400 FAILED_PRECONDITION: "ListChunks is not
#:   available for datastores that don't have chunking_config set." The
#:   store is built with ``defaultParsingConfig.digitalParsingConfig``
#:   and no chunking config, because the page number in a citation comes
#:   from extractive segments, which are a document-mode feature.
#:
#: So there is no by-id fetch of the cited text. This ledger is the
#: identity check that is available: it records what retrieval actually
#: returned, costs nothing, and does not depend on ranking.
_RETRIEVED: dict[str, None] = {}

#: Ledger cap. Ample for a report (a few passages per ground) while
#: keeping a long-lived service from growing without bound.
_RETRIEVED_LIMIT = 4096


def _record_retrieved(results: list[dict[str, str]]) -> None:
    """Remember the citations ``results`` carries, evicting the oldest."""
    for result in results:
        citation = str(result.get("citation", "")).strip()
        if not citation:
            continue
        _RETRIEVED.pop(citation, None)
        _RETRIEVED[citation] = None
    while len(_RETRIEVED) > _RETRIEVED_LIMIT:
        _RETRIEVED.pop(next(iter(_RETRIEVED)))


def reset_retrieved() -> None:
    """Forget every recorded citation. For tests and for process reuse."""
    _RETRIEVED.clear()


def verify_citation(citation: str, query: str = "", top_k: int = 25) -> bool:
    """True when ``citation`` is one the corpus actually returned.

    A model asked to cite will happily produce something that *looks*
    like a citation. A live report agent returned five grounds all
    sourced to ``https://www.fincen.gov/regulations/statutes/...`` -- a
    real URL, correctly formatted, and never returned by
    :func:`search`. Checking only for placeholder markers passed it,
    because an invented citation carries no marker: it is not labelled
    synthetic, it is simply made up.

    So a citation is verified against what the corpus returns, never
    against how it looks. Anything else is the report agent grading its
    own homework.

    **The check is identity first.** Every citation :func:`search` hands
    out is recorded in ``_RETRIEVED``, and a citation found there was
    provably produced by retrieval. Re-searching to confirm something a
    search already returned is circular, and on the real 825-page corpus
    it did not even work: ``"PMLA, 2002, p. 14"`` carries almost no
    semantic signal about what is on page 14, so using the citation
    string as its own query put the passage nowhere near the top 25 and
    every genuine ground was dropped. See ``_RETRIEVED`` for why a
    by-id fetch of the cited text is not available from this data store.

    The search probe is kept as a second chance for a citation this
    process did not retrieve itself -- one read back from a file, say.
    It can only ever add a pass, and only on an exact match, so it
    cannot admit a fabrication; it simply cannot be relied on to find a
    real one.

    Args:
        citation: The citation string to check.
        query: Optional context to search alongside the citation itself,
            improving the chance the passage is in the candidate set.
        top_k: How many candidates to consider.

    Returns:
        Whether the corpus returned exactly this citation.
    """
    if not citation.strip():
        return False
    wanted = citation.strip()
    if wanted in _RETRIEVED:
        return True
    for probe in filter(None, (citation, query)):
        try:
            results = search(probe, top_k)
        except (ValueError, RuntimeError):
            continue
        if any(result.get("citation", "").strip() == wanted for result in results):
            return True
    return False


def is_synthetic(result: dict[str, str]) -> bool:
    """True when a search result is a placeholder rather than a real passage.

    Both backends currently return synthetic passages: ``docai``'s
    corpus is explicitly labelled, and this module's fallback corpus is
    paraphrased. Callers that render citations must surface this.
    """
    citation = result.get("citation", "") + result.get("source_uri", "")
    return any(marker in citation for marker in SYNTHETIC_MARKERS)


def search(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Return regulation passages relevant to ``query``, best match first.

    Delegates to ``docai.search_regulations`` when that exists, otherwise
    searches the built-in corpus.

    Args:
        query: Natural-language description of the suspicious behaviour.
        top_k: Maximum number of passages to return.

    Returns:
        Up to ``top_k`` results, each with ``"citation"``, ``"excerpt"``
        and ``"source_uri"`` keys. Passages with no token overlap at all
        are omitted rather than padded.

    Raises:
        ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        RegulationSearchUnavailable: If ``MULENET_DOCAI_MODE=cloud`` was
            requested and the real index could not be reached --
            ``docai`` not importable, or ``docai.regulations.search_
            regulations`` itself raised (missing ``MULENET_REGULATIONS_
            ENGINE``, no credentials, a failed HTTP call). Never falls
            back to this module's mock corpus in that case.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    cloud_requested = os.environ.get("MULENET_DOCAI_MODE", "").strip().lower() == "cloud"
    delegate = _docai_search()
    if delegate is not None:
        try:
            results = list(delegate(query, top_k))
        except Exception as exc:
            if cloud_requested:
                _log.error(
                    "regulation search cloud mode failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                raise RegulationSearchUnavailable(
                    "regulation lookup is unavailable right now"
                ) from exc
            raise
        _record_retrieved(results)
        return results
    if cloud_requested:
        _log.error(
            "MULENET_DOCAI_MODE=cloud is set but docai.regulations is not "
            "importable in this process"
        )
        raise RegulationSearchUnavailable("regulation lookup is unavailable right now")

    query_tokens = set(_tokens(query))
    scored = [
        (_relevance(query_tokens, entry), index, entry)
        for index, entry in enumerate(CORPUS)
    ]
    # Sort by score descending, then corpus order, so results are stable.
    scored.sort(key=lambda item: (-item[0], item[1]))
    results = [
        {
            "citation": entry["citation"],
            "excerpt": entry["excerpt"],
            "source_uri": MOCK_SOURCE_PREFIX + _slug(entry["citation"]),
        }
        for score, _, entry in scored[:top_k]
        if score > 0
    ]
    _record_retrieved(results)
    return results


def _slug(citation: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", citation.lower()).strip("-")
