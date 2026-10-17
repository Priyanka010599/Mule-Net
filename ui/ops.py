"""Operability for the public service: request ids, safe logs, size cap, CORS, readiness.

Kept out of ``ui/main.py`` so the routes there stay readable. Nothing in this
module logs a request body, a query string, an account id or a question: the
access line carries the route *template* (``/api/screen/{hashed_account_id}``),
never the concrete path, and a redaction filter backstops every handler it is
attached to (including uvicorn's own access log, which does print raw paths).
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ui.config import UiConfig
from ui.errors import error_body

#: Monotonic time this module was imported: the start of cold-start timing.
PROCESS_START = time.monotonic()

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

#: An inbound ``X-Request-Id`` is honoured only if it is plainly an id.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

#: Paths that take a larger body and enforce their own cap in the handler.
UPLOAD_PATH = "/api/subset/upload"

_HEALTH_PATHS = frozenset({"/health", "/ready"})

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/api/screen/[^\s\"?]+"), "/api/screen/[id]"),
    (re.compile(r"\b[0-9a-fA-F]{24,}\b"), "[id]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
)

#: ``extra=`` keys a log line may carry. Anything else is dropped, not printed.
_FIELDS = (
    "request_id", "method", "route", "status", "duration_ms", "first_request",
    "since_process_start_ms", "startup_ms", "snapshot_load_ms", "client_error",
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactFilter(logging.Filter):
    """Scrub account ids, screen paths and e-mail addresses from the message."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with a Cloud Logging ``severity``."""

    def format(self, record: logging.LogRecord) -> str:
        body: dict[str, Any] = {
            "severity": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key in _FIELDS:
            value = getattr(record, key, None)
            if value is not None and key not in body:
                body[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            # The type only: an exception message can quote the row that broke it.
            body["exception"] = record.exc_info[0].__name__
        return json.dumps(body, default=str)


def configure_logging() -> None:
    """JSON logs on stdout for ``mulenet.*``; redaction on uvicorn's access log."""
    root = logging.getLogger("mulenet")
    if not any(getattr(h, "_mulenet_json", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._mulenet_json = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonFormatter())
        handler.addFilter(RedactFilter())
        root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactFilter) for f in access.filters):
        access.addFilter(RedactFilter())


access_log = logging.getLogger("mulenet.ui.access")


class OpsMiddleware:
    """Request id, one access line, first-request timing and the body cap (ASGI)."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self._first_done = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        inbound = headers.get("x-request-id", "")
        request_id = inbound if _SAFE_REQUEST_ID.match(inbound) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        started = time.monotonic()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode("latin-1"))
                )
            await send(message)

        try:
            if scope["method"] == "POST" and scope["path"] != UPLOAD_PATH:
                declared = headers.get("content-length", "")
                if declared.isdigit() and int(declared) > self.max_body_bytes:
                    await self._too_large(scope, receive, send_wrapper, request_id)
                    return
                receive = self._counting(receive)
            await self.app(scope, receive, send_wrapper)
        finally:
            ms = round((time.monotonic() - started) * 1000, 1)
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            first = not self._first_done
            self._first_done = True
            extra: dict[str, Any] = {
                "request_id": request_id,
                "method": scope["method"],
                "route": route,
                "status": status,
                "duration_ms": ms,
            }
            if first:
                extra["first_request"] = True
                extra["since_process_start_ms"] = round(
                    (time.monotonic() - PROCESS_START) * 1000, 1
                )
            level = logging.DEBUG if scope["path"] in _HEALTH_PATHS else logging.INFO
            access_log.log(level, "request", extra=extra)
            request_id_var.reset(token)

    def _counting(self, receive: Receive) -> Receive:
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_body_bytes:
                    raise HTTPException(413, "The request body is larger than the allowed size.")
            return message

        return counted

    async def _too_large(self, scope: Scope, receive: Receive, send: Send, rid: str) -> None:
        body = json.dumps(
            error_body(
                "payload_too_large",
                "The request body is larger than the allowed size.",
                request_id=rid,
            )
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class StaticCacheMiddleware:
    """``StaticFiles`` sets no ``Cache-Control``, so a browser's heuristic
    caching can keep serving yesterday's ``app.js`` after a deploy with no
    way to know it's stale -- this has silently hidden more than one real
    fix behind a browser cache this project shipped. ``no-cache`` (not
    ``no-store``) still lets the browser keep the bytes; it just makes it
    revalidate via ETag/Last-Modified first, which StaticFiles already sets."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/static/"):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"cache-control", b"no-cache"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def install_middleware(app: FastAPI, cfg: UiConfig) -> None:
    """CORS limited to ``cfg.cors_allow_origins`` (none by default), then ops."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-Id"],
        expose_headers=["X-Request-Id", "ETag"],
    )
    app.add_middleware(OpsMiddleware, max_body_bytes=cfg.max_body_bytes)
    app.add_middleware(StaticCacheMiddleware)


# -- scrubbing what the agents service may relay --------------------------------

_INTERNAL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"projects/[\w.-]+(?:/[\w.%-]+)+"), "[cloud resource]"),
    (re.compile(r"[\w.-]+@[\w-]+\.iam\.gserviceaccount\.com"), "[service account]"),
    (re.compile(r"gs://[^\s'\"]+"), "[bucket path]"),
    (re.compile(r"https?://[\w.-]+\.run\.app\S*"), "[internal service]"),
)


def scrub_internal(value: Any) -> Any:
    """Replace cloud resource names, service accounts, bucket paths and service URLs
    in every string of a JSON-like value. A Google error message quoting the KMS key
    it refused is the case this exists for."""
    if isinstance(value, str):
        for pattern, replacement in _INTERNAL:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {k: scrub_internal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_internal(v) for v in value]
    return value


def scrub_model(model: Any) -> Any:
    """``scrub_internal`` over a pydantic model, returning the same model type."""
    return type(model).model_validate(scrub_internal(model.model_dump()))


# -- readiness ----------------------------------------------------------


def probe_agents(cfg: UiConfig) -> dict[str, Any]:
    """``GET /health`` on the agents service. No model call, and no URL in the result."""
    if cfg.chat_backend != "agents":
        return {"backend": cfg.chat_backend, "configured": False, "reachable": None}
    headers: dict[str, str] = {}
    if cfg.agents_auth:
        try:
            from ui.chat import google_id_token

            headers["Authorization"] = f"Bearer {google_id_token(cfg.agents_audience)}"
        except Exception:  # noqa: BLE001 - any credential failure is "not reachable"
            return {"backend": "agents", "configured": True, "reachable": False,
                    "reason": "no_credentials"}
    request = urllib.request.Request(f"{cfg.agents_url}/health", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=min(cfg.agents_timeout_s, 3.0)) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        return {"backend": "agents", "configured": True, "reachable": False,
                "status_code": exc.code}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"backend": "agents", "configured": True, "reachable": False,
                "reason": "unreachable"}
    return {"backend": "agents", "configured": True, "reachable": 200 <= code < 300,
            "status_code": code}
