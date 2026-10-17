"""FastAPI application for the MuleNet investigator console.

Serves a single-page shell plus a small JSON API. Every endpoint reads
from :class:`~ui.datasource.DataSource`, so pointing the app at real
``ml/`` output is a config change rather than a rewrite.

Run locally::

    python -m ui.build_snapshot          # once, builds the demo data
    python -m uvicorn ui.main:app --reload --port 8080
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ui.chat import build_backend, finalize_reply
from ui.config import UiConfig
from ui.datasource import DataSource, SnapshotMissingError
from ui.errors import ApiError, api_error_response, client_ip, error_body, register_handlers
from ui.ops import (
    PROCESS_START,
    configure_logging,
    install_middleware,
    probe_agents,
    scrub_model,
)
from ui.provenance import CACHE_MAX_AGE_S, Provenance, etag_matches
from ui.ratelimit import SlidingWindow
from ui.replay_index import ReplayIndex, build_replay_index
from ui.models import (
    BankTile,
    ChatReply,
    ChatRequest,
    CoalitionPayload,
    DemoState,
    EnclavePayload,
    LiveProofStartResult,
    LiveProofStatusResult,
    MetricsPayload,
    Overview,
    RedteamPayload,
    ReplayPayload,
    RingDetail,
    ScreenResult,
    SimulateBounds,
    SimulateVerdict,
    SubsetLoadResult,
    SubsetOptions,
    SubsetScoreResult,
    UploadScoreResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Resolve config and warm the snapshot cache before serving."""
    configure_logging()
    cfg = UiConfig.from_env()
    app.state.cfg = cfg
    app.state.data = DataSource(cfg)
    app.state.chat = build_backend(cfg)
    app.state.provenance = Provenance(cfg)
    app.state.agents_probe = None
    app.state.limiters = {
        "simulate": SlidingWindow(cfg.simulate_rate_per_minute, 60.0),
        "subset": SlidingWindow(cfg.subset_rate_per_minute, 60.0),
        # Per-client brake at the public edge, in front of agents' own
        # session/IP/daily ceilings (a fresh agents call costs real money).
        "chat": SlidingWindow(cfg.chat_rate_per_minute, 60.0),
        # Only a loose per-minute brake on button-mashing: the real launch
        # limits (cooldown, hourly, daily, single flight) live in agents.
        "live_proof": SlidingWindow(10, 60.0),
    }
    loading = time.monotonic()
    try:
        _ = app.state.data.snapshot
    except SnapshotMissingError:
        # Not fatal: /health and the shell still serve, and the banner
        # tells the operator to run the builder.
        pass
    snapshot_load_ms = round((time.monotonic() - loading) * 1000, 1)
    app.state.provenance.log_startup_check()
    app.state.startup_ms = round((time.monotonic() - PROCESS_START) * 1000, 1)
    logging.getLogger("mulenet.ui.startup").info(
        "cold start complete",
        extra={"startup_ms": app.state.startup_ms, "snapshot_load_ms": snapshot_load_ms},
    )
    yield


app = FastAPI(
    title="MuleNet investigator console",
    description="Cross-bank money mule detection: bank tiles, joint graph, metrics.",
    version="0.1.0",
    lifespan=lifespan,
)

_cfg = UiConfig.from_env()
register_handlers(app)
install_middleware(app, _cfg)
app.mount("/static", StaticFiles(directory=_cfg.static_dir), name="static")
_templates = Jinja2Templates(directory=str(_cfg.templates_dir))


def get_data(request: Request) -> DataSource:
    """Request-scoped handle on the data layer."""
    return request.app.state.data


#: Injected data layer. Declared as an ``Annotated`` alias rather than a
#: ``Depends`` default so the dependency is part of the type, which is
#: both the current FastAPI idiom and free of the mutable-default trap
#: that bugbear's B008 warns about.
Data = Annotated[DataSource, Depends(get_data)]


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return api_error_response(request, exc)


def _enforce(request: Request, limiter: str) -> None:
    """Spend one hit of ``limiter`` for this client, or raise a structured 429."""
    retry = request.app.state.limiters[limiter].check(client_ip(request))
    if retry is not None:
        raise ApiError(
            429,
            "rate_limited",
            "Too many requests from this address. Try again shortly.",
            retry_after_seconds=retry,
        )


@app.exception_handler(SnapshotMissingError)
async def _snapshot_missing(request: Request, exc: SnapshotMissingError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=error_body(
            "snapshot_missing", str(exc), request_id=getattr(request.state, "request_id", None)
        ),
    )


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: is the snapshot loaded, and is the agents service reachable?

    Distinct from ``/health`` (liveness, always cheap). Calls the agents
    ``/health`` only -- never Gemini -- and caches the answer for 10 s so a
    probe loop cannot become traffic. A down agents service is ``degraded``
    (the page falls back to its offline demo) and still 200; a missing
    snapshot is 503.
    """
    state = request.app.state
    snapshot_loaded = state.data._snapshot is not None
    now = time.monotonic()
    cached = getattr(state, "agents_probe", None)
    if cached is None or now - cached[0] > 10.0:
        cached = (now, probe_agents(state.cfg))
        state.agents_probe = cached
    agents = cached[1]
    ok = snapshot_loaded
    status = "ready" if ok and agents["reachable"] is not False else "degraded"
    if not ok:
        status = "not_ready"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": status,
            "snapshot_loaded": snapshot_loaded,
            "agents": agents,
            "startup_ms": state.startup_ms,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Cloud Run.

    **Not** ``/healthz``. Cloud Run's front end reserves paths ending in
    ``z`` and answers them with its own HTML 404, so ``/healthz`` never
    reaches the container on a ``*.run.app`` host and any uptime check
    pointed at it reports permanently unhealthy. The name is load-bearing.
    """
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """The single-page shell. All data arrives over the JSON API."""
    return _templates.TemplateResponse(
        request,
        "index.html",
        {"chat_backend": request.app.state.cfg.chat_backend},
    )


@app.get("/api/overview", response_model=Overview)
def overview(data: Data) -> Overview:
    """Dataset scale, graph bounding, ring index and the featured ring."""
    return data.overview()


@app.get("/api/banks", response_model=list[BankTile])
def banks(
    data: Data,
    request: Request,
    ring_id: Annotated[
        str | None, Query(description="ring to fragment by bank")
    ] = None,
    reveal_labels: Annotated[
        bool | None,
        Query(
            description=(
                "1 includes the simulator's ground-truth labels, 0 withholds them "
                "(null, with labels_revealed=false on each tile). Omitted: the "
                "server default, MULENET_UI_LABELS_DEFAULT_REVEALED."
            )
        ),
    ] = None,
) -> list[BankTile]:
    """The three bank dashboards. Each holds only its own bank's data."""
    if reveal_labels is None:
        reveal_labels = request.app.state.cfg.labels_default_revealed
    try:
        return data.banks(ring_id, reveal_labels)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ring {exc.args[0]!r}") from exc


@app.get("/api/rings", response_model=Overview)
def rings(data: Data) -> Overview:
    """Ring index (same payload as /api/overview; kept for clarity)."""
    return data.overview()


@app.get("/api/rings/{ring_id}", response_model=RingDetail)
def ring(ring_id: str, data: Data) -> RingDetail:
    """One ring: bounded subgraph, evidence, explanation, case and report."""
    try:
        return data.ring(ring_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ring {exc.args[0]!r}") from exc


@app.get("/api/replay/{ring_id}", response_model=ReplayPayload)
def replay(ring_id: str, data: Data) -> ReplayPayload:
    """Replay timeline for one detected cluster.

    A cluster with no replay is a 200 carrying ``available: false`` and a
    reason, not a 404: "there is no ground truth for this false-positive
    cluster" is an answer the panel must render, and an error status would
    make it indistinguishable from a broken request. Only an id the
    detector never published is a 404.
    """
    try:
        return data.replay(ring_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ring {exc.args[0]!r}") from exc


def _not_modified(request: Request, response: Response, *artifacts: str) -> Response | None:
    """ETag + Cache-Control keyed by the data version of the artifacts read.

    Returns a bare 304 when the browser already holds this version, else sets
    the validators on ``response`` and returns ``None``.
    """
    etag = f'"{request.app.state.provenance.version(*artifacts)}"'
    headers = {"ETag": etag, "Cache-Control": f"public, max-age={CACHE_MAX_AGE_S}"}
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return None


@app.get("/api/meta")
def meta(request: Request) -> dict:
    """Provenance of every artifact served: hash, seed, generation time, source,
    model/rule versions, and whether it predates the data it describes."""
    return request.app.state.provenance.meta()


@app.get("/api/replay-index", response_model=ReplayIndex)
def replay_index(data: Data) -> ReplayIndex:
    """Per ring: first-flag horizon and minutes before exit for each scope, whether
    cross-bank flagged earlier (and by how many minutes), the ring's bank span, and
    which rings flag at the same horizon. Summarises the snapshot's replays only."""
    return build_replay_index(data.snapshot)


@app.get("/api/metrics", response_model=MetricsPayload)
def metrics(data: Data, request: Request, response: Response) -> MetricsPayload | Response:
    """Single-bank vs cross-bank by early-detection horizon and ring span."""
    return _not_modified(request, response, "metrics", "lead_time") or data.metrics()


@app.get("/api/coalition", response_model=CoalitionPayload)
def coalition(
    data: Data, request: Request, response: Response
) -> CoalitionPayload | Response:
    """Ring recall as a function of how many banks joined.

    Read-only, and it reads a file: every point was measured by
    ``ml/coalition.py`` before this process started. The sweep is 22
    feature builds over the full transfer graph and takes about 50
    seconds, so recomputing it per request was never an option -- and a
    deployed container has no ``simulator/out/`` to recompute it from,
    which would have left the endpoint inventing the number it was meant
    to prove.
    """
    return _not_modified(request, response, "coalition") or data.coalition()


@app.get("/api/redteam", response_model=RedteamPayload)
def redteam(data: Data, request: Request, response: Response) -> RedteamPayload | Response:
    """Gemini-designed evasion strategies, run through the real detector.

    Read-only, and it reads a file: the run was made once, live against
    Gemini, and is served exactly as measured. ``ml/redteam.py`` is the one
    script in this repo with no rate limit on its model call, so this
    endpoint deliberately never re-runs it -- there is no "run again"
    control anywhere this payload is consumed, only this precomputed
    result.
    """
    return _not_modified(request, response, "redteam") or data.redteam()


@app.get("/api/enclave", response_model=EnclavePayload)
def enclave(data: Data, request: Request, response: Response) -> EnclavePayload | Response:
    """One real Confidential Space run's evidence (docs/enclave.md, Stage 3).

    Read-only, and it reads a file: a real, credentialed run against
    Google Cloud, captured once and served as-is. Unlike every other
    endpoint in this module there is no live path this could fall back
    from -- reproducing it needs an actual GCP project, KMS and a
    Confidential VM, none of which this container can reach.
    """
    return _not_modified(request, response, "enclave") or data.enclave()


@app.post("/api/chat", response_model=ChatReply)
def chat(
    body: ChatRequest,
    request: Request,
    data: Data,
) -> ChatReply:
    """Ask the router agent. Mock-backed until ``agents/`` is ready."""
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")
    _enforce(request, "chat")
    context: RingDetail | None = None
    ring_id = (body.ui_context.ring_id if body.ui_context else None) or body.ring_id
    if ring_id:
        try:
            context = data.ring(ring_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown ring {exc.args[0]!r}"
            ) from exc
    return finalize_reply(request.app.state.chat.reply(body, context))


@app.get("/api/screen/{hashed_account_id}", response_model=ScreenResult)
def screen(hashed_account_id: str, request: Request) -> ScreenResult:
    """Check one account before a payment to it, the way a bank's own
    transaction path would call this ahead of a transfer. Mock-backed
    until ``agents/`` is ready, the same as chat.
    """
    return request.app.state.chat.screen(hashed_account_id)


@app.get("/api/simulate/bounds", response_model=SimulateBounds)
def simulate_bounds(request: Request) -> SimulateBounds:
    """Field bounds for the live "build your own evasion attempt" form.

    Proxied to ``agents/`` (docs/agents_api.md's ``GET /simulate/bounds``)
    the same way chat and screen are: this server never imports ``ml/``
    (see ``infra/Dockerfile.ui``), so the numbers a slider is built from
    have to come from the service that actually enforces them.
    """
    return request.app.state.chat.simulate_bounds()


@app.post("/api/simulate", response_model=SimulateVerdict)
def simulate(body: dict, request: Request) -> SimulateVerdict:
    """Run one judge-built evasion pattern through the real detector, live.

    ``body`` is forwarded to ``agents/`` verbatim -- validation and
    bounds-clamping happen exactly once, server side, in
    ``ml.redteam._coerce_pattern``. Mock-backed until ``agents/`` is
    configured, the same as chat and screen, except the mock has nothing
    honest to answer with here: see ``ui.chat.MockRouter.simulate``.

    Out-of-range fields come back as a 422 with a field-level ``errors``
    list, produced by ``agents/`` (the one place the ranges live) and passed
    through; a client over its per-minute budget gets a structured 429.
    """
    _enforce(request, "simulate")
    return request.app.state.chat.simulate(body)


@app.get("/api/subset/options", response_model=SubsetOptions)
def subset_options(request: Request) -> SubsetOptions:
    """Horizon and bank choices for the live "load a subset" panel.

    Proxied to ``agents/`` (``GET /subset/options``) the same way
    ``simulate_bounds`` above is: this server never imports ``ml/`` (see
    ``infra/Dockerfile.ui``), so the menu a dropdown is built from has to
    come from the service that actually enforces it.
    """
    return request.app.state.chat.subset_options()


@app.post("/api/subset/load", response_model=SubsetLoadResult)
def subset_load(body: dict, request: Request) -> SubsetLoadResult:
    """Build features for one caller-chosen real data slice, live.

    ``body`` (``{horizon, bank}``) is forwarded to ``agents/`` verbatim --
    the fixed-menu validation happens exactly once, server side. Runs no
    rule and flags nothing; see ``/api/subset/score`` below for that.
    Mock-backed until ``agents/`` is configured, the same as ``simulate``,
    with the same "nothing honest to answer with" caveat -- see
    ``ui.chat.MockRouter.subset_load``.
    """
    _enforce(request, "subset")
    horizon = str(body.get("horizon", "24h"))
    bank = str(body.get("bank", "all"))
    return request.app.state.chat.subset_load(horizon, bank)


@app.post("/api/subset/score", response_model=SubsetScoreResult)
def subset_score(body: dict, request: Request) -> SubsetScoreResult:
    """Load, then rule on, one caller-chosen real data slice, live -- no
    caching of the result across different subset choices.

    Same request shape as ``/api/subset/load`` above, and the same
    mock-backed degradation.
    """
    _enforce(request, "subset")
    horizon = str(body.get("horizon", "24h"))
    bank = str(body.get("bank", "all"))
    return request.app.state.chat.subset_score(horizon, bank)


@app.post("/api/live-proof/start", response_model=LiveProofStartResult)
def live_proof_start(request: Request) -> LiveProofStartResult:
    """Kick off one real encrypt -> attest -> decrypt enclave run.

    A cap hit comes back as ``capped: true``, not an error (see
    ``agents/service.py``'s ``LiveProofStartOut``); the page falls back
    to the frozen static evidence card rather than a dead button.
    Mock-backed until ``agents/`` is configured, with the same
    "nothing honest to answer with" caveat as ``simulate`` --
    see ``ui.chat.MockRouter.live_proof_start``.
    """
    _enforce(request, "live_proof")
    return scrub_model(request.app.state.chat.live_proof_start(client_ip(request)))


@app.post("/api/live-proof/tampered-start", response_model=LiveProofStartResult)
def live_proof_tampered_start(request: Request) -> LiveProofStartResult:
    """Kick off one real "tamper test": the same VM lifecycle as
    ``/api/live-proof/start``, launching the known-refused tampered
    image instead of the approved one. Shares the SAME 10/day cap.
    Refused with 404 unless MULENET_UI_LIVE_PROOF_TAMPER is on.
    """
    if not request.app.state.cfg.live_proof_tamper_enabled:
        raise HTTPException(status_code=404, detail="not found")
    _enforce(request, "live_proof")
    return scrub_model(request.app.state.chat.live_proof_tampered_start(client_ip(request)))


@app.get("/api/live-proof/runs/{run_id}", response_model=LiveProofStatusResult)
@app.get("/api/live-proof/status/{run_id}", response_model=LiveProofStatusResult)
def live_proof_status(run_id: str, request: Request) -> LiveProofStatusResult:
    """Poll one live-proof run's current stage. Never blocks -- the page
    polls this every few seconds while the real ~2-4 minute cycle runs."""
    return scrub_model(request.app.state.chat.live_proof_status(run_id))


#: Largest request body ``/api/subset/upload`` will forward: MAX_BANKS files of
#: MAX_FILE_BYTES each plus JSON framing. Mirrors ``agents.subset_upload``
#: (this image does not ship ``agents/``); ``ui/test_upload_edge.py`` asserts
#: the two stay equal.
MAX_UPLOAD_BODY_BYTES = 5 * 512 * 1024 + 64 * 1024


@app.post("/api/subset/upload", response_model=UploadScoreResult)
def subset_upload_route(
    body: dict,
    request: Request,
    http_errors: Annotated[
        bool,
        Query(description="answer a rejected file with HTTP 422 instead of 200 + `error`"),
    ] = False,
) -> UploadScoreResult:
    """Score a judge's own per-bank CSV upload, live.

    ``body`` is ``{banks: [{name, csv_text}, ...]}``, forwarded to
    ``agents/`` verbatim -- the real validation (schema, size, row count,
    ownership) happens exactly once, server side, in
    ``agents/subset_upload.py``. Same proxy shape as ``/api/subset/score``
    above, with one difference worth naming: a rejected upload comes back
    as a normal ``200`` carrying ``UploadScoreResult.error``, not an
    ``offline_demo`` degradation -- the service is not down, the file was
    invalid, and those must not look the same on screen.
    """
    _enforce(request, "subset")
    raw_banks = body.get("banks", [])
    if not isinstance(raw_banks, list) or any(not isinstance(b, dict) for b in raw_banks):
        raise ApiError(422, "validation_error", "banks must be a list of {name, csv_text}")
    banks = [(str(b.get("name", "")), str(b.get("csv_text", ""))) for b in raw_banks]
    if sum(len(name) + len(text) for name, text in banks) > MAX_UPLOAD_BODY_BYTES:
        raise ApiError(413, "payload_too_large", "The upload is larger than the allowed total size.")
    result = request.app.state.chat.subset_upload(banks)
    if http_errors and result.error:
        raise ApiError(422, "validation_error", result.error, errors=result.errors)
    return result


@app.post("/api/demo/reset", response_model=DemoState)
def demo_reset(data: Data) -> DemoState:
    """Reset demo state, then hand back the ring that makes the argument."""
    return data.demo_reset()
