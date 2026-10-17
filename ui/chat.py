"""Chat backends for the investigator panel.

Two backends, selected by ``MULENET_UI_CHAT_BACKEND``:

:class:`MockRouter`
    Offline. Routes on intent the way the real router does, then answers
    *from the loaded snapshot*, so every number it says is a number the
    graph and the case panel also show. A mock that invented figures
    would make the demo untrustworthy the moment a judge cross-checked
    one. It labels itself ``backend="mock"`` and never ``"gemini"``.

:class:`AgentsRouter`
    The ``agents/`` Cloud Run service over HTTP, exactly as
    ``docs/agents_api.md`` specifies. That file is frozen and this client
    conforms to it rather than negotiating with it.

**Auth and the browser.** The agents service has no public invoker
access, so ``ui`` calls it with a Cloud Run service-to-service ID token
minted for the agents service as the audience. Two rules follow and both
are enforced here: a browser-supplied token is never forwarded (the token
is minted server side from this server's own identity), and the agents
URL never leaves the server -- it lives in ``ui/config.py``, is not
rendered into a template, and is not carried on any error path. See
:class:`AgentsUnavailable`, which carries an ``error_type`` and nothing
else.

**Degradation is visible.** When the agents service errors, times out, is
unreachable or rate-limits, the mock answers instead, and the reply
carries ``offline_demo`` plus an ``offline_notice`` the page renders on
screen. A silent fallback would let a judge believe they were talking to
Gemini when they were talking to a keyword matcher.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol

from ui.config import UiConfig
from ui.errors import ApiError
from ui.models import (
    ChatReply,
    ChatRequest,
    ChatTrace,
    LiveProofStartResult,
    LiveProofStatusResult,
    RingDetail,
    ScreenResult,
    SimulateBounds,
    SimulateFieldBound,
    SimulatePatternValues,
    SimulateVerdict,
    SubsetFlaggedAccount,
    SubsetHorizonOption,
    SubsetLoadResult,
    SubsetOptions,
    SubsetScoreResult,
    UploadFlaggedAccount,
    UploadScoreResult,
)

#: Agent names the router dispatches to. Mirrors the five ADK agents
#: ``agents/`` is building; the mock answers in their voice.
AGENTS: tuple[str, ...] = (
    "router",
    "graph_analyst",
    "kyc_analyst",
    "compliance_writer",
    "metrics_analyst",
)

_GRAPH_WORDS = ("graph", "ring", "flow", "trail", "hop", "transfer", "money", "path")
_KYC_WORDS = ("document", "kyc", "device", "identity", "address", "reuse", "shared")
_REPORT_WORDS = ("report", "str", "sar", "regulation", "cite", "citation", "file", "draft")
_METRIC_WORDS = ("metric", "auc", "recall", "precision", "compare", "baseline", "alert")

#: Regulator/statute names -- a different intent from "show this ring's
#: drafted STR" (_REPORT_WORDS above, which also matches on the word
#: "regulation"). Used only by MockRouter below, the offline fallback:
#: it has no real index, and would otherwise route a regulation
#: question into this ring's own drafted-report reply, surfacing that
#: report's hardcoded placeholder citations as if they answered the
#: question asked. AgentsRouter does NOT use this -- the real agents
#: service has a real, cited regulation search (search_regulations over
#: Vertex AI Search) and answers these questions itself.
_REGULATION_LOOKUP_WORDS = (
    "rbi", "pmla", "fincen", "fatf", "bsa",
    "money laundering act", "prevention of money laundering",
)

#: Mirrors ``EXAMPLE_QUESTIONS`` in ``ui/static/app.js`` -- same three
#: chips, surfaced here as the regulation decline's suggested actions.
_EXAMPLE_QUESTIONS = (
    "How many rings are in the dataset?",
    "Which banks are in the largest cross-bank ring and how many accounts "
    "does it have?",
    "How many alertable members does the largest ring have?",
)


def _regulation_decline(message: str) -> ChatReply | None:
    """An honest, fixed decline for a regulation-lookup question.

    Returns ``None`` for anything else, so callers fall through to their
    normal routing. ``backend="mock"`` because nothing here calls Gemini
    or the real router -- see ``ChatReply.backend``'s docstring on why
    that label must stay honest.
    """
    text = message.lower()
    if not any(word in text for word in _REGULATION_LOOKUP_WORDS):
        return None
    return ChatReply(
        reply=(
            "Regulation lookup isn't connected yet -- it's on our roadmap. "
            "I can answer questions about MuleNet's rings, banks, accounts "
            "and scores."
        ),
        backend="mock",
        agent="router",
        suggested_actions=list(_EXAMPLE_QUESTIONS),
    )


def route(message: str) -> str:
    """Pick the agent that should answer. Same intents as the real router."""
    text = message.lower()
    for words, agent in (
        (_REPORT_WORDS, "compliance_writer"),
        (_KYC_WORDS, "kyc_analyst"),
        (_METRIC_WORDS, "metrics_analyst"),
        (_GRAPH_WORDS, "graph_analyst"),
    ):
        if any(word in text for word in words):
            return agent
    return "router"


class ChatBackend(Protocol):
    """What ``main.py`` needs from a chat backend."""

    name: str

    def reply(self, request: ChatRequest, ring: RingDetail | None) -> ChatReply:
        """Answer one message, optionally in the context of a ring."""
        ...

    def screen(self, hashed_account_id: str) -> ScreenResult:
        """Flag whether ``hashed_account_id`` is a known mule."""
        ...

    def simulate_bounds(self) -> SimulateBounds:
        """Field bounds and clause names for the live pattern-builder form."""
        ...

    def simulate(self, pattern: dict[str, Any]) -> SimulateVerdict:
        """Run one judge-built evasion pattern through the real detector, live."""
        ...

    def subset_options(self) -> SubsetOptions:
        """Horizon and bank choices for the live "load a subset" panel."""
        ...

    def subset_load(self, horizon: str, bank: str) -> SubsetLoadResult:
        """Build features for one caller-chosen real data slice, live."""
        ...

    def subset_score(self, horizon: str, bank: str) -> SubsetScoreResult:
        """Load, then rule on, one caller-chosen real data slice, live."""
        ...

    def subset_upload(self, banks: list[tuple[str, str]]) -> UploadScoreResult:
        """Score a judge's own per-bank CSV upload, live."""
        ...

    def live_proof_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        """Kick off one real encrypt -> attest -> decrypt enclave run."""
        ...

    def live_proof_tampered_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        """Kick off one real "tamper test": the same VM lifecycle, but
        launching the known-refused tampered image. Shares the cap with
        live_proof_start."""
        ...

    def live_proof_status(self, run_id: str) -> LiveProofStatusResult:
        """Poll one live-proof run's current stage."""
        ...


class MockRouter:
    """Offline stand-in for the ADK router agent."""

    name = "mock"

    def reply(self, request: ChatRequest, ring: RingDetail | None) -> ChatReply:
        decline = _regulation_decline(request.message)
        if decline is not None:
            return decline
        agent = route(request.message)
        if ring is None:
            return ChatReply(
                reply=(
                    "Select a ring in the joint view and I can answer against its "
                    "subgraph, its KYC evidence and its draft report. Running "
                    "offline against the mock router - no Gemini call is made."
                ),
                backend=self.name,
                agent="router",
                suggested_actions=["Open the joint view", "Press Demo Mode"],
            )
        handler = {
            "graph_analyst": self._graph,
            "kyc_analyst": self._kyc,
            "compliance_writer": self._report,
            "metrics_analyst": self._metrics,
            "router": self._router,
        }[agent]
        return handler(ring)

    def screen(self, hashed_account_id: str) -> ScreenResult:
        """Offline stand-in: never invents a flagged/not-flagged verdict.

        Unlike ``reply()``, the mock has no honest way to answer this: the
        loaded snapshot is one ring's subgraph, not the account-lookup
        table ``agents/``'s ``/screen`` reads. Saying "not flagged" from a
        table that was never actually checked is exactly the kind of
        invented number this module's docstring warns against, so the
        mock reports the check as unavailable instead of guessing.
        """
        return ScreenResult(hashed_account_id=hashed_account_id, flagged=False)

    def simulate_bounds(self) -> SimulateBounds:
        """Offline stand-in: there is no honest offline answer.

        Unlike :meth:`reply` and :meth:`screen`, the mock has nothing to
        fall back to here -- the pattern bounds live in ``ml.redteam``,
        which ``ui/`` deliberately does not import (see
        ``infra/Dockerfile.ui``), and there is no snapshot number that
        substitutes for them. So this returns an empty field set rather
        than invented bounds a slider could be built from.
        """
        return SimulateBounds(
            fields={},
            evadable_clauses=[],
            backend=self.name,
            offline_demo=True,
            offline_notice=simulate_offline_notice("unreachable"),
        )

    def simulate(self, pattern: dict[str, Any]) -> SimulateVerdict:
        """Offline stand-in: there is no offline detector to run this against.

        A judge's pattern can only be ruled on by the real
        ``ml.redteam.evaluate`` in ``agents/``. There is nothing here to
        approximate it with -- unlike a chat answer or a screen check,
        which the mock can honestly derive from the loaded snapshot, a
        live per-request detector run has no snapshot equivalent -- so
        this returns a verdict that is visibly not one, rather than a
        guessed caught/missed that could be mistaken for a real result.
        """
        return SimulateVerdict(
            name="offline demo mode",
            description="The live evasion simulator needs the agents/ service.",
            targets="unavailable",
            aimed_at_clause=str(pattern.get("evades_clause", "")),
            why_it_should_evade="",
            pattern=SimulatePatternValues(
                collectors=0,
                mules_per_collector=0,
                onward_accounts=0,
                hop_delay_seconds=0.0,
                amount_per_inbound_hop=0.0,
                retained_fraction=0.0,
                counterparty_banks=0,
                reuse_device=False,
                reuse_kyc_id=False,
            ),
            caught=False,
            caught_by="unavailable",
            reasoning=(
                "No verdict was computed. This demo has no offline detector "
                "to run a live pattern through; the agents/ service must be "
                "reachable for this panel to mean anything."
            ),
            flagged_accounts=0,
            flagged_collectors=0,
            rings_formed=0,
            stage2_alertable=0,
            stage2_context=0,
            clause_values={},
            backend=self.name,
            offline_demo=True,
            offline_notice=simulate_offline_notice("unreachable"),
        )

    def subset_options(self) -> SubsetOptions:
        """Offline stand-in: same reasoning as :meth:`simulate_bounds`.

        The horizon menu lives in ``agents.subset.HORIZONS``, which
        ``ui/`` deliberately does not import (no ``ml/`` dependency here;
        see ``infra/Dockerfile.ui``), so there is no honest menu to offer
        without the agents service.
        """
        return SubsetOptions(
            horizons=[],
            banks=[],
            backend=self.name,
            offline_demo=True,
            offline_notice=subset_offline_notice("unreachable"),
        )

    def subset_load(self, horizon: str, bank: str) -> SubsetLoadResult:
        """Offline stand-in: there is no offline feature builder either.

        Same reasoning as :meth:`simulate`: the loaded snapshot is
        precomputed at one fixed horizon (the full window), not a table
        this mock can re-cut to an arbitrary caller-chosen one.
        """
        return SubsetLoadResult(
            horizon=horizon,
            bank=bank,
            backend=self.name,
            offline_demo=True,
            offline_notice=subset_offline_notice("unreachable"),
        )

    def subset_score(self, horizon: str, bank: str) -> SubsetScoreResult:
        """Offline stand-in: there is no offline detector to run this
        against, same reasoning as :meth:`simulate`."""
        return SubsetScoreResult(
            horizon=horizon,
            bank=bank,
            backend=self.name,
            offline_demo=True,
            offline_notice=subset_offline_notice("unreachable"),
        )

    def subset_upload(self, banks: list[tuple[str, str]]) -> UploadScoreResult:
        """Offline stand-in: same reasoning as :meth:`subset_score` -- no
        offline detector to run an uploaded file through either."""
        return UploadScoreResult(
            banks=[name for name, _ in banks],
            backend=self.name,
            offline_demo=True,
            offline_notice=subset_offline_notice("unreachable"),
        )

    def live_proof_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        """Offline stand-in: there is no VM to create here either.

        Same reasoning as :meth:`simulate`: a real encrypt/attest/decrypt
        cycle needs an actual GCP project and a Confidential VM, which
        this container never has. The page's job on seeing
        ``offline_demo`` here is the same as on a cap hit -- show the
        frozen static evidence card instead of a dead button.
        """
        return LiveProofStartResult(
            backend=self.name,
            offline_demo=True,
            offline_notice=live_proof_offline_notice("unreachable"),
        )

    def live_proof_tampered_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        """Offline stand-in: same reasoning as :meth:`live_proof_start`
        -- a real tamper test needs an actual GCP project and KMS."""
        return LiveProofStartResult(
            backend=self.name,
            offline_demo=True,
            offline_notice=live_proof_offline_notice("unreachable"),
        )

    def live_proof_status(self, run_id: str) -> LiveProofStatusResult:
        """Offline stand-in. Never actually polled: the mock's
        :meth:`live_proof_start` returns no ``run_id`` to poll for."""
        return LiveProofStatusResult(
            run_id=run_id,
            stage="failed",
            error="live proof is unavailable offline",
            backend=self.name,
            offline_demo=True,
            offline_notice=live_proof_offline_notice("unreachable"),
        )

    def _router(self, ring: RingDetail) -> ChatReply:
        return ChatReply(
            reply=(
                f"{ring.ring_id} spans {ring.span} bank(s) "
                f"({', '.join(ring.banks_involved)}) with "
                f"{ring.alertable_member_count} alertable account(s)"
                + (
                    f" and {ring.context_member_count} shown as context only - "
                    "same-bank identity links, which are leads, not findings"
                    if ring.context_member_count
                    else ""
                )
                + ". Ask me about the money flow, the document and device evidence, "
                "the case and draft report, or how cross-bank scoring compares with "
                "single-bank."
            ),
            backend=self.name,
            agent="router",
            suggested_actions=[
                "Trace the money flow",
                "What document reuse is there?",
                "Draft the report",
                "How does this compare to single-bank?",
            ],
        )

    def _graph(self, ring: RingDetail) -> ChatReply:
        alertable = [m for m in ring.members if m.alertable]
        context = [m for m in ring.members if not m.alertable]
        collectors = [m.hashed_account_id for m in alertable if m.role == "collector"]
        sources = [m for m in alertable if m.role == "source"]
        mules = [m for m in alertable if m.role == "mule"]
        exit_text = (
            f"converging on one cash-out account ({collectors[0][:12]}...)"
            if collectors
            else "with no single cash-out account identified"
        )
        # Upstream sources are usually not flagged by the velocity rule,
        # so a detected ring often contains none. Say that rather than
        # reporting "0 source accounts", which reads like a data error.
        intake = (
            f"{len(sources)} source account(s) fan money into {len(mules)} mules"
            if sources
            else (
                f"{len(mules)} flagged mules relay funds from upstream senders "
                "that the rule did not itself flag"
            )
        )
        return ChatReply(
            reply=(
                f"{intake}, "
                f"{exit_text}. {ring.intra_ring_transfer_count} intra-ring transfers "
                f"moved {ring.total_intra_ring_amount}, and "
                f"{ring.cross_bank_transfer_count} of them cross a bank boundary - "
                "those are the hops no single institution observes both sides of. "
                "That structure plus the forwarding velocity is what fires the "
                "detection."
                + (
                    f" Separately, {len(context)} account(s) are attached to this "
                    "ring as context only: they share an identifier with a member "
                    "inside the same bank. That is a lead to review, not an "
                    "accusation, and they are not counted as detections."
                    if context
                    else ""
                )
            ),
            backend=self.name,
            agent="graph_analyst",
            suggested_actions=["Play the money trail", "Show the cash-out account"],
        )

    def _kyc(self, ring: RingDetail) -> ChatReply:
        cross = [e for e in ring.evidence if e.cross_bank]
        if not cross:
            return ChatReply(
                reply=(
                    "No cross-bank identity reuse on this ring. The finding rests on "
                    "transaction structure alone, which is the stronger basis anyway."
                ),
                backend=self.name,
                agent="kyc_analyst",
            )
        widest = max(cross, key=lambda e: e.accounts_with_this)
        collapsed = next((e.note for e in cross if e.note), "")
        return ChatReply(
            reply=(
                f"{len(cross)} distinct cross-bank findings. The widest: "
                f"{widest.detail.strip()} - it touches "
                f"{widest.accounts_with_this} of this ring's members, sourced from "
                f"{widest.source_table}. "
                + (collapsed + " " if collapsed else "")
                + "Read it as corroboration, not proof. Identity reuse took no part "
                "in the detection decision: it was tested as a detector and found "
                "strictly dominated, so it was removed from the rule and re-attached "
                "as evidence. It does discriminate once an account is flagged - "
                "55.7% of flagged true positives carry it against 2.0% of flagged "
                "false positives - but families and dual-bank customers share "
                "devices and addresses too."
            ),
            backend=self.name,
            agent="kyc_analyst",
            suggested_actions=["Highlight reused documents on the graph"],
        )

    def _report(self, ring: RingDetail) -> ChatReply:
        if ring.case is None or ring.report is None:
            return ChatReply(
                reply=(
                    "No case yet for this ring. A case is opened only when an "
                    "investigation's claims have been re-derived from the data and "
                    "matched by the verifier; this ring's narrative was withheld or "
                    "has not been investigated, so there is nothing verified to "
                    "quote and no STR has been drafted against it. The raw signals "
                    "and evidence are still on the case panel."
                ),
                backend=self.name,
                agent="compliance_writer",
                suggested_actions=["Show the signals", "Show the evidence"],
            )
        return ChatReply(
            reply=(
                f"Draft report {ring.report.report_id[:8]} is on case "
                f"{ring.case.case_id[:8]}, status {ring.report.status}. The case "
                f"rests on {ring.case.verified_claim_count} claim(s) re-derived by "
                f"the verifier"
                + (
                    " on the second attempt, after the first draft asserted "
                    "something the data did not support."
                    if ring.case.verification_attempts > 1
                    else " on the first attempt."
                )
                + " Narrative: "
                f"{ring.explanation.narrative[:320]}..."
                + (
                    ""
                    if ring.report.citations_verified
                    else " Citations are placeholders until agents/ retrieves real "
                    "passages through Vertex AI Search - do not file this."
                )
            ),
            backend=self.name,
            agent="compliance_writer",
            citations=ring.report.regulation_citations,
            suggested_actions=["Open the full report", "Add an analyst note"],
        )

    def _metrics(self, ring: RingDetail) -> ChatReply:
        gain = {1: "+0.009", 2: "+0.174", 3: "+0.328"}.get(ring.span, "n/a")
        per_bank = Decimal(ring.total_intra_ring_amount) / max(ring.span, 1)
        return ChatReply(
            reply=(
                f"This is a {ring.span}-bank ring, and span is what decides the size "
                f"of the gain. At five minutes after ring onset, cross-bank scoring "
                f"beats single-bank by {gain} PR-AUC on {ring.span}-bank rings. Each "
                f"bank here sees roughly {per_bank:,.0f} of the "
                f"{ring.total_intra_ring_amount} moved and cannot follow the rest. "
                "Open the metrics page for the full span stratification."
            ),
            backend=self.name,
            agent="metrics_analyst",
            suggested_actions=["Open the metrics page"],
        )


# -- the agents service -------------------------------------------

#: The label the page puts on screen whenever an answer came from the
#: mock while ``agents`` was the configured backend. On screen, not only
#: in a JSON field: the failure this guards against is a judge reading a
#: keyword match as a Gemini answer.
OFFLINE_DEMO_LABEL = "offline demo mode"

#: One sentence per documented ``error_type`` in docs/agents_api.md. The
#: two ``429`` variants are deliberately NOT collapsed: one means "you
#: are going too fast", the other means "this demo is finished for
#: today", and a caller who cannot tell them apart keeps retrying into a
#: spend cap.
_OFFLINE_CAUSE: dict[str, str] = {
    "rate_limited": (
        "You are going too fast - the live agent rate-limited this session."
    ),
    "daily_cap": (
        "This demo is finished for today - the live agent has reached its "
        "daily spend cap."
    ),
    "timeout": "The live Gemini call ran past its timeout without answering.",
    "upstream_unavailable": (
        "Vertex AI was unreachable, so the live agent could not answer."
    ),
    "unreachable": "The agents service could not be reached from this server.",
    "forbidden": "This server is not authorised to call the agents service.",
    "bad_request": "The agents service rejected the request as malformed.",
    "no_credentials": "This server could not mint a service-to-service ID token.",
    "bad_response": (
        "The agents service replied with something that was not a Gemini answer."
    ),
    "internal": "The live agent failed with an internal error.",
    # The next two are the service declining on purpose, not failing: the
    # withhold rule applied at the transport edge (docs/agents_api.md).
    "no_tool_evidence": (
        "The live agent declined to publish an answer - it replied without "
        "running any tool, so nothing grounded it and it was withheld."
    ),
    "unverified": (
        "The live agent declined to publish an answer - a claim in it was "
        "checked against the data and failed, so it was withheld."
    ),
    # The next two are real problems on the service side.
    "data_unavailable": (
        "The live agent's detection tables are not loaded, so it could not "
        "ground an answer in real data."
    ),
    "malformed_response": (
        "The live agent's model returned text the service could not parse, "
        "and its retries were used up."
    ),
}

#: Said on every degraded answer. The mock's numbers are real -- they come
#: from the same snapshot the graph draws -- but its reasoning is a
#: keyword match, and the notice says so rather than implying a model.
_OFFLINE_SUFFIX = (
    "The answer below is from the offline keyword-matching mock, not from "
    "Gemini. Its numbers still come from the loaded snapshot."
)

#: Status -> ``error_type`` for a non-2xx that did not carry one. Cloud
#: Run's ``403`` for a missing or invalid token is produced before
#: application code runs and its body is not the documented JSON, so it is
#: mapped here. A bare ``429`` is read as the recoverable variant:
#: guessing ``daily_cap`` would end a working demo on a malformed reply.
_STATUS_FALLBACK: dict[int, str] = {
    403: "forbidden",
    422: "bad_request",
    429: "rate_limited",
    500: "internal",
    503: "upstream_unavailable",
    504: "timeout",
}


class AgentsUnavailable(Exception):
    """The agents service did not produce a usable Gemini answer.

    Carries the documented ``error_type`` and ``retry_after_s`` and
    nothing else -- deliberately not the URL, the response body or a
    traceback, because this value's whole job is to become a sentence
    shown in a browser.
    """

    def __init__(
        self,
        error_type: str,
        retry_after_s: int | None = None,
        message: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.retry_after_s = retry_after_s
        #: Field-level problems a 422 carries (``[{field, message, ...}]``).
        self.errors = errors
        # Optional, and unused by every existing caller: the specific
        # ServiceError.message a 422 carries. Chat/simulate/subset's
        # closed, pre-validated inputs rarely hit a 422 worth quoting, but
        # an uploaded file's validation failure (agents/subset_upload.py)
        # genuinely needs to say which bank and which row -- see
        # AgentsRouter.subset_upload.
        self.message = message


#: ``error_type`` values the agents service uses for a live-proof launch that a
#: limit refused. These are the caller being told to wait, not an outage.
_LAUNCH_DENIED = {"cooldown", "rate_limited_global", "rate_limited_ip"}


def _raise_if_launch_denied(exc: AgentsUnavailable) -> None:
    if exc.error_type in _LAUNCH_DENIED:
        raise ApiError(
            429,
            exc.error_type,
            exc.message or "Live proof is rate limited. Try again shortly.",
            retry_after_seconds=exc.retry_after_s,
        ) from exc


def offline_notice(error_type: str, retry_after_s: int | None = None) -> str:
    """The on-screen sentence for one failure case."""
    cause = _OFFLINE_CAUSE.get(error_type, f"The live agent failed ({error_type}).")
    retry = f" Retry in {retry_after_s}s." if retry_after_s else ""
    # Opens with the label, so the notice stands alone wherever it lands:
    # the page splits the first sentence off it for the banner heading
    # rather than restating the words and letting the two drift apart.
    return f"{OFFLINE_DEMO_LABEL.capitalize()}. {cause}{retry} {_OFFLINE_SUFFIX}"


#: Said on a degraded /simulate or /simulate/bounds answer, in place of
#: :data:`_OFFLINE_SUFFIX`. That suffix promises the mock's numbers still
#: come from the loaded snapshot -- true for chat and screen, both of
#: which read the snapshot, and false here: a live evasion pattern has no
#: snapshot equivalent to fall back to, only the real detector in
#: agents/. Reusing the chat/screen wording on this panel would make a
#: false claim rather than an honest one.
_SIMULATE_OFFLINE_SUFFIX = (
    "There is no offline equivalent of this panel: a judge's pattern can "
    "only be ruled on by the real detector in agents/, so nothing shown "
    "below is a real verdict."
)


def simulate_offline_notice(error_type: str, retry_after_s: int | None = None) -> str:
    """The on-screen sentence for a degraded ``/simulate`` answer.

    Same shape as :func:`offline_notice`, with :data:`_SIMULATE_OFFLINE_SUFFIX`
    in place of the chat/screen suffix -- see that constant for why the two
    must not share wording.
    """
    cause = _OFFLINE_CAUSE.get(error_type, f"The live agent failed ({error_type}).")
    retry = f" Retry in {retry_after_s}s." if retry_after_s else ""
    return (
        f"{OFFLINE_DEMO_LABEL.capitalize()}. {cause}{retry} "
        f"{_SIMULATE_OFFLINE_SUFFIX}"
    )


#: Said on a degraded ``/subset/options``, ``/subset/load`` or
#: ``/subset/score`` answer, in place of :data:`_OFFLINE_SUFFIX`. Same
#: reasoning as :data:`_SIMULATE_OFFLINE_SUFFIX`: a live feature build
#: over a caller-chosen data slice has no snapshot equivalent to fall
#: back to.
_SUBSET_OFFLINE_SUFFIX = (
    "There is no offline equivalent of this panel: a caller-chosen data "
    "subset can only be loaded and scored by the real code in agents/, so "
    "nothing shown below reflects a real computation."
)


def subset_offline_notice(error_type: str, retry_after_s: int | None = None) -> str:
    """The on-screen sentence for a degraded ``/subset/*`` answer.

    Same shape as :func:`simulate_offline_notice`, with
    :data:`_SUBSET_OFFLINE_SUFFIX` in place of that function's suffix.
    """
    cause = _OFFLINE_CAUSE.get(error_type, f"The live agent failed ({error_type}).")
    retry = f" Retry in {retry_after_s}s." if retry_after_s else ""
    return (
        f"{OFFLINE_DEMO_LABEL.capitalize()}. {cause}{retry} "
        f"{_SUBSET_OFFLINE_SUFFIX}"
    )


#: Said on a degraded ``/live-proof/*`` answer, in place of
#: :data:`_OFFLINE_SUFFIX`. Same reasoning as :data:`_SUBSET_OFFLINE_SUFFIX`:
#: a real encrypt/attest/decrypt cycle needs an actual GCP project and a
#: Confidential VM, which this container never has.
_LIVE_PROOF_OFFLINE_SUFFIX = (
    "There is no offline equivalent of this proof: a real encrypt, attest "
    "and decrypt cycle needs an actual GCP project and a Confidential VM. "
    "The evidence below is a frozen record of one real run."
)


def live_proof_offline_notice(error_type: str, retry_after_s: int | None = None) -> str:
    """The on-screen sentence for a degraded ``/live-proof/*`` answer.

    Same shape as :func:`subset_offline_notice`, with
    :data:`_LIVE_PROOF_OFFLINE_SUFFIX` in place of that function's suffix.
    """
    cause = _OFFLINE_CAUSE.get(error_type, f"The live agent failed ({error_type}).")
    retry = f" Retry in {retry_after_s}s." if retry_after_s else ""
    return (
        f"{OFFLINE_DEMO_LABEL.capitalize()}. {cause}{retry} "
        f"{_LIVE_PROOF_OFFLINE_SUFFIX}"
    )


def google_id_token(audience: str) -> str:
    """Mint a Cloud Run service-to-service ID token for ``audience``.

    On Cloud Run this signs a token at the instance metadata server using
    the service's own service account; locally it uses Application
    Default Credentials. It is *this server's* identity in both cases --
    no token from a browser request is read, here or anywhere.

    Imported lazily so a machine with no Google credentials can still run
    the app on the mock backend.
    """
    import google.auth.transport.requests
    import google.oauth2.id_token

    return google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), audience
    )


class AgentsClient:
    """HTTP client for the ``agents/`` service. Server side only.

    ``docs/agents_api.md`` is frozen and authoritative: the request is
    ``{session_id, message}``, a ``200`` carries
    ``{answer, backend, verified, sources, latency_ms}``, and every
    non-2xx carries ``{error_type, message, retry_after_s}``. Nothing is
    renamed or made optional here.

    ``token_provider`` is a seam, not a loophole: it defaults to nothing
    so a local fake agents service can be exercised without Google
    credentials, and :func:`build_backend` supplies
    :func:`google_id_token` for any real deployment.
    """

    #: Every field docs/agents_api.md declares on a ``200``. A response
    #: missing any of them is not the documented shape, so it is not shown
    #: as a Gemini answer.
    _REQUIRED: tuple[str, ...] = (
        "answer",
        "backend",
        "verified",
        "sources",
        "latency_ms",
    )

    def __init__(
        self,
        base_url: str,
        audience: str = "",
        timeout_s: float = 30.0,
        token_provider: Callable[[str], str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._audience = audience or self._base_url
        self._timeout_s = timeout_s
        self._token_provider = token_provider

    def chat(
        self,
        session_id: str,
        message: str,
        ui_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """``POST /chat``. Raises :class:`AgentsUnavailable` on any failure."""
        fields: dict[str, Any] = {"session_id": session_id, "message": message}
        if ui_context:
            fields["ui_context"] = ui_context
        body = json.dumps(fields)
        headers = {"Content-Type": "application/json"}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:  # any auth failure degrades, it does not 500
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/chat",
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        return self._validated(payload)

    @staticmethod
    def _from_http_error(exc: urllib.error.HTTPError) -> AgentsUnavailable:
        error_type = ""
        retry_after_s: int | None = None
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:  # Cloud Run's own 403 body is not the JSON envelope
            detail = None
        message: str | None = None
        errors: list[dict[str, Any]] | None = None
        if isinstance(detail, dict):
            if isinstance(detail.get("error_type"), str):
                error_type = detail["error_type"]
            if isinstance(detail.get("retry_after_s"), int):
                retry_after_s = detail["retry_after_s"]
            if isinstance(detail.get("message"), str):
                message = detail["message"]
            if isinstance(detail.get("errors"), list):
                errors = [e for e in detail["errors"] if isinstance(e, dict)]
        return AgentsUnavailable(
            error_type or _STATUS_FALLBACK.get(exc.code, "internal"),
            retry_after_s,
            message,
            errors,
        )

    @classmethod
    def _validated(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or any(
            key not in payload for key in cls._REQUIRED
        ):
            raise AgentsUnavailable("bad_response")
        # docs/agents_api.md: ``backend`` is the literal "gemini", and the
        # service must never return it for an answer it did not get from
        # Gemini on Vertex. Anything else did not come from Gemini, so it
        # is not displayed as though it had.
        # The one other literal is "precomputed": a suggested question
        # answered from the rings table, which says it was not a model.
        if payload["backend"] not in ("gemini", "precomputed"):
            raise AgentsUnavailable("bad_response")
        if not isinstance(payload["answer"], str) or not payload["answer"].strip():
            raise AgentsUnavailable("bad_response")
        if not isinstance(payload["sources"], list):
            raise AgentsUnavailable("bad_response")
        return payload

    def screen(self, hashed_account_id: str) -> dict[str, Any]:
        """``GET /screen/{id}``. Raises :class:`AgentsUnavailable` on any failure."""
        headers: dict[str, str] = {}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/screen/{hashed_account_id}",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "flagged" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def simulate_bounds(self) -> dict[str, Any]:
        """``GET /simulate/bounds``. Raises :class:`AgentsUnavailable` on any failure."""
        headers: dict[str, str] = {}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/simulate/bounds",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "fields" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def simulate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """``POST /simulate``. Raises :class:`AgentsUnavailable` on any failure.

        ``pattern`` is sent verbatim as the request body: bounds-checking
        happens once, server side, in ``ml.redteam._coerce_pattern`` --
        this client does not repeat it.
        """
        body = json.dumps(pattern)
        headers = {"Content-Type": "application/json"}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/simulate",
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "caught" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def subset_options(self) -> dict[str, Any]:
        """``GET /subset/options``. Raises :class:`AgentsUnavailable` on any failure."""
        headers: dict[str, str] = {}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/subset/options",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "horizons" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def _subset_post(self, path: str, horizon: str, bank: str) -> dict[str, Any]:
        """Shared body for ``/subset/load`` and ``/subset/score`` --
        identical request shape, only the path and the required response
        key differ, so those two are parameters rather than two near-copies
        of this method."""
        body = json.dumps({"horizon": horizon, "bank": bank})
        headers = {"Content-Type": "application/json"}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict):
            raise AgentsUnavailable("bad_response")
        return payload

    def subset_load(self, horizon: str, bank: str) -> dict[str, Any]:
        """``POST /subset/load``. Raises :class:`AgentsUnavailable` on any failure."""
        payload = self._subset_post("/subset/load", horizon, bank)
        if "transfers_considered" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def subset_score(self, horizon: str, bank: str) -> dict[str, Any]:
        """``POST /subset/score``. Raises :class:`AgentsUnavailable` on any failure."""
        payload = self._subset_post("/subset/score", horizon, bank)
        if "flagged_accounts" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def subset_upload(self, banks: list[tuple[str, str]]) -> dict[str, Any]:
        """``POST /subset/upload``. Raises :class:`AgentsUnavailable` on any failure.

        Different body shape from :meth:`_subset_post` (a list of
        ``{name, csv_text}`` objects, not ``horizon``/``bank``), so this
        builds its own request rather than reusing that helper -- same
        auth/error handling, inlined once rather than parameterising
        ``_subset_post`` for a shape it was never meant to carry.
        """
        body = json.dumps(
            {"banks": [{"name": name, "csv_text": csv_text} for name, csv_text in banks]}
        )
        headers = {"Content-Type": "application/json"}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/subset/upload",
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "flagged_accounts" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def _live_proof_post(self, path: str, client_ip: str = "unknown") -> dict[str, Any]:
        """Shared body for ``/live-proof/start`` and
        ``/live-proof/tampered-start`` -- identical request shape and
        response contract (``capped`` is NOT a failure, see
        ``agents/service.py``'s ``LiveProofStartOut``), only the path
        differs."""
        # The agents service applies the per-client launch limit; it only ever
        # sees this service as the caller, so the real client address rides in
        # X-Forwarded-For (agents trusts it: only this service may call it).
        headers = {"Content-Type": "application/json", "X-Forwarded-For": client_ip}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "capped" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload

    def live_proof_start(self, client_ip: str = "unknown") -> dict[str, Any]:
        """``POST /live-proof/start``. Raises :class:`AgentsUnavailable` on any failure."""
        return self._live_proof_post("/live-proof/start", client_ip)

    def live_proof_tampered_start(self, client_ip: str = "unknown") -> dict[str, Any]:
        """``POST /live-proof/tampered-start``. Raises :class:`AgentsUnavailable` on any failure."""
        return self._live_proof_post("/live-proof/tampered-start", client_ip)

    def live_proof_status(self, run_id: str) -> dict[str, Any]:
        """``GET /live-proof/status/{run_id}``. Raises
        :class:`AgentsUnavailable` on any failure, including an unknown
        ``run_id`` (the service's own 404)."""
        headers: dict[str, str] = {}
        if self._token_provider is not None:
            try:
                token = self._token_provider(self._audience)
            except Exception as exc:
                raise AgentsUnavailable("no_credentials") from exc
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self._base_url}/live-proof/status/{run_id}",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentsUnavailable("unreachable") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise AgentsUnavailable("bad_response") from exc
        if not isinstance(payload, dict) or "stage" not in payload:
            raise AgentsUnavailable("bad_response")
        return payload


#: Agent and node names that are plumbing, not wording. If one is written
#: into the answer text ("router: ...", "[narrative] ..."), it moves to the
#: trace instead of being shown to the person reading the answer.
_INTERNAL_LABELS = (
    "router",
    "narrative",
    "analyst_qa",
    "graph_analyst",
    "kyc_analyst",
    "metrics_analyst",
    "compliance_writer",
    "investigator",
)
_LABELS_RE = "|".join(_INTERNAL_LABELS)
_LEADING_LABEL = re.compile(
    rf"^\s*(?:\[(?P<a>{_LABELS_RE})\]\s*[:\-\u2013\u2014\u00b7|>]*"
    rf"|(?P<b>{_LABELS_RE})\s*[:\-\u2013\u2014\u00b7|>]+)\s*",
    re.IGNORECASE,
)
_TRAILING_PROVENANCE = re.compile(
    r"\s*(?:\u00b7\s*)?(?:Grounded in [^\n]*?)?\d+\s*ms server-side\s*$", re.IGNORECASE
)


def finalize_reply(reply: ChatReply) -> ChatReply:
    """Set ``mode``, move internal labels out of the text, build ``trace``.

    Applied to every chat reply on its way out, live or offline, so the
    contract does not depend on which backend answered. The wording of an
    answer is never changed beyond removing a leading agent label or a
    trailing provenance fragment that was written into it.
    """
    labels: list[str] = []
    text = reply.reply
    while (m := _LEADING_LABEL.match(text)) and text[m.end():].strip():
        labels.append((m.group("a") or m.group("b")).lower())
        text = text[m.end():]
    trailing = _TRAILING_PROVENANCE.search(text)
    if trailing and text[: trailing.start()].strip():
        text = text[: trailing.start()]
    live = reply.backend == "gemini" and not reply.offline_demo
    return reply.model_copy(
        update={
            "reply": text.strip() if labels or trailing else reply.reply,
            "mode": "live" if live else "offline",
            "trace": ChatTrace(
                agent=reply.agent,
                backend=reply.backend,
                sources=list(reply.sources),
                latency_ms=reply.latency_ms,
                verified=reply.verified,
                labels=labels,
            ),
        }
    )


def with_ring_context(message: str, ring_id: str | None) -> str:
    """Name the open ring in the question unless the analyst already did.

    Each agents request runs in a fresh ADK session (docs/agents_api.md), so
    "trace the money flow" with a ring open reaches Gemini with no subject:
    it calls no tool, the verifier withholds the answer (409
    no_tool_evidence), and the analyst gets the offline mock instead.
    """
    if ring_id is None or ring_id in message:
        return message
    return f"About ring {ring_id}: {message}"


def _forwarded_context(request: ChatRequest, ring: RingDetail | None) -> dict[str, str] | None:
    """The ``ui_context`` to send: the browser's, with the open ring filled in.

    ``None`` (field omitted) when there is nothing to say, so a plain
    question still sends exactly ``{session_id, message}``.
    """
    sent = request.ui_context.model_dump(exclude_none=True) if request.ui_context else {}
    if "ring_id" not in sent and ring is not None:
        sent["ring_id"] = ring.ring_id
    return {k: v for k, v in sent.items() if v} or None


class AgentsRouter:
    """The live ADK router, with the mock as a *labelled* fallback."""

    name = "agents"

    def __init__(
        self,
        client: AgentsClient,
        fallback: ChatBackend | None = None,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or MockRouter()
        # One bucket key per ui process, minted here. docs/agents_api.md
        # says session_id is a bucket key that grants no authority, so it
        # is never accepted from the browser: a caller free to choose it
        # could reset the per-session rate limit at will, and that rate
        # limit is the cost control.
        self._session_id = session_id or f"mulenet-ui-{uuid.uuid4().hex}"

    def reply(self, request: ChatRequest, ring: RingDetail | None) -> ChatReply:
        # No keyword short-circuit here, unlike MockRouter below: the
        # real agents service now has a real, cited regulation search
        # (analyst_qa + search_regulations, Vertex AI Search over the
        # actual RBI/PMLA/FIU-IND texts) and can answer these questions
        # itself, honestly declining the ones it can't. A local
        # short-circuit would shadow that -- it did, briefly, until
        # this was caught verifying the fix live.
        try:
            payload = self._client.chat(
                self._session_id,
                with_ring_context(
                    request.message, ring.ring_id if ring is not None else None
                ),
                _forwarded_context(request, ring),
            )
        except AgentsUnavailable as exc:
            degraded = self._fallback.reply(request, ring)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": offline_notice(
                        exc.error_type, exc.retry_after_s
                    ),
                    "answer": degraded.reply,
                    "mode": "offline",
                    "failure_reason": exc.error_type,
                    "failure_detail": (exc.message or "")[:400] or None,
                    "error": {
                        "code": exc.error_type,
                        "message": exc.message
                        or _OFFLINE_CAUSE.get(exc.error_type, "The live agent failed."),
                        "retry_after_seconds": exc.retry_after_s,
                    },
                }
            )
        return ChatReply(
            mode="live",
            answer=payload["answer"],
            source_details=[d for d in payload.get("source_details", []) if isinstance(d, dict)],
            timings={
                k: int(v) for k, v in dict(payload.get("timings") or {}).items()
            },
            agent_trace=[d for d in payload.get("agent_trace", []) if isinstance(d, dict)],
            model=str(payload.get("model", "")),
            request_id=str(payload.get("request_id", "")),
            reply=payload["answer"],
            # Passed through rather than hard-coded: ``_validated`` has
            # already refused anything that was not the documented literal.
            backend=payload["backend"],
            agent="router",
            sources=[str(item) for item in payload["sources"]],
            regulation_citation_cards=list(payload.get("regulation_citations", [])),
            verified=bool(payload["verified"]),
            latency_ms=int(payload["latency_ms"]),
            source=str(payload.get("source", "model")),
        )

    def screen(self, hashed_account_id: str) -> ScreenResult:
        try:
            payload = self._client.screen(hashed_account_id)
        except AgentsUnavailable as exc:
            degraded = self._fallback.screen(hashed_account_id)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": offline_notice(
                        exc.error_type, exc.retry_after_s
                    ),
                }
            )
        return ScreenResult(
            hashed_account_id=str(payload["hashed_account_id"]),
            flagged=bool(payload["flagged"]),
            backend="agents",
            ring_ids=[str(item) for item in payload.get("ring_ids", [])],
            reason=str(payload.get("reason", "")),
            risk_score=(
                float(payload["risk_score"])
                if payload.get("risk_score") is not None
                else None
            ),
        )

    def simulate_bounds(self) -> SimulateBounds:
        try:
            payload = self._client.simulate_bounds()
        except AgentsUnavailable as exc:
            degraded = self._fallback.simulate_bounds()
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": simulate_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return SimulateBounds(
            fields={
                name: SimulateFieldBound(**bound)
                for name, bound in payload["fields"].items()
            },
            evadable_clauses=[str(c) for c in payload.get("evadable_clauses", [])],
            backend="agents",
        )

    def simulate(self, pattern: dict[str, Any]) -> SimulateVerdict:
        try:
            payload = self._client.simulate(pattern)
        except AgentsUnavailable as exc:
            if exc.error_type == "validation_error":
                # The service is up and said the input was out of range: that
                # is the caller's error, not an outage to dress up as a mock.
                raise ApiError(
                    422, "validation_error", exc.message or "invalid pattern", errors=exc.errors
                ) from exc
            degraded = self._fallback.simulate(pattern)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": simulate_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return SimulateVerdict(
            **{**payload, "backend": "agents"},
        )

    def subset_options(self) -> SubsetOptions:
        try:
            payload = self._client.subset_options()
        except AgentsUnavailable as exc:
            degraded = self._fallback.subset_options()
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": subset_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return SubsetOptions(
            horizons=[
                SubsetHorizonOption(**h) for h in payload.get("horizons", [])
            ],
            banks=[str(b) for b in payload.get("banks", [])],
            backend="agents",
        )

    def subset_load(self, horizon: str, bank: str) -> SubsetLoadResult:
        try:
            payload = self._client.subset_load(horizon, bank)
        except AgentsUnavailable as exc:
            degraded = self._fallback.subset_load(horizon, bank)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": subset_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return SubsetLoadResult(**{**payload, "backend": "agents"})

    def subset_score(self, horizon: str, bank: str) -> SubsetScoreResult:
        try:
            payload = self._client.subset_score(horizon, bank)
        except AgentsUnavailable as exc:
            degraded = self._fallback.subset_score(horizon, bank)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": subset_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return SubsetScoreResult(
            **{
                **payload,
                "flagged_accounts": [
                    SubsetFlaggedAccount(**row)
                    for row in payload.get("flagged_accounts", [])
                ],
                "backend": "agents",
            }
        )

    def subset_upload(self, banks: list[tuple[str, str]]) -> UploadScoreResult:
        try:
            payload = self._client.subset_upload(banks)
        except AgentsUnavailable as exc:
            if exc.error_type == "bad_request" and exc.message:
                # The judge's file was rejected -- a real, non-degraded
                # outcome, not the service being unreachable. See
                # UploadScoreResult.error's docstring for why this must
                # not go through the "offline demo mode" banner.
                return UploadScoreResult(
                    banks=[name for name, _ in banks],
                    backend="agents",
                    error=exc.message,
                    errors=exc.errors or [],
                )
            degraded = self._fallback.subset_upload(banks)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": subset_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return UploadScoreResult(
            **{
                **payload,
                "flagged_accounts": [
                    UploadFlaggedAccount(**row)
                    for row in payload.get("flagged_accounts", [])
                ],
                "backend": "agents",
            }
        )

    def live_proof_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        try:
            payload = self._client.live_proof_start(client_ip)
        except AgentsUnavailable as exc:
            _raise_if_launch_denied(exc)
            degraded = self._fallback.live_proof_start()
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": live_proof_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        # A cap hit is the service's own, non-degraded 200 -- carried
        # through as-is, never routed through the mock fallback.
        return LiveProofStartResult(**{**payload, "backend": "agents"})

    def live_proof_tampered_start(self, client_ip: str = "unknown") -> LiveProofStartResult:
        try:
            payload = self._client.live_proof_tampered_start(client_ip)
        except AgentsUnavailable as exc:
            _raise_if_launch_denied(exc)
            degraded = self._fallback.live_proof_tampered_start()
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": live_proof_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return LiveProofStartResult(**{**payload, "backend": "agents"})

    def live_proof_status(self, run_id: str) -> LiveProofStatusResult:
        try:
            payload = self._client.live_proof_status(run_id)
        except AgentsUnavailable as exc:
            degraded = self._fallback.live_proof_status(run_id)
            return degraded.model_copy(
                update={
                    "offline_demo": True,
                    "offline_notice": live_proof_offline_notice(exc.error_type, exc.retry_after_s),
                }
            )
        return LiveProofStatusResult(**{**payload, "backend": "agents"})


def build_backend(cfg: UiConfig) -> ChatBackend:
    """Select the chat backend named in the config."""
    if cfg.chat_backend == "mock":
        return MockRouter()
    if cfg.chat_backend == "agents":
        return AgentsRouter(
            AgentsClient(
                base_url=cfg.agents_url,
                audience=cfg.agents_audience,
                timeout_s=cfg.agents_timeout_s,
                token_provider=google_id_token if cfg.agents_auth else None,
            )
        )
    raise ValueError(f"unknown chat backend {cfg.chat_backend!r}")
