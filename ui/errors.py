"""One error envelope for the public API, and the client-address helper.

Every error this app raises on purpose comes back as::

    {"code": "rate_limited", "message": "...", "retry_after_seconds": 12,
     "errors": [{"field": ..., "message": ...}], "request_id": "..."}

``errors`` is present only for field-level problems (422), and
``retry_after_seconds`` only when waiting helps (it is also sent as the
``Retry-After`` header). ``detail`` carries ``message`` as well, because the
page's existing error handling reads FastAPI's ``detail`` and must keep
working until the frontend pass moves to ``code``/``message``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """An expected failure with a stable machine-readable ``code``."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        retry_after_seconds: int | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.errors = errors


def error_body(
    code: str,
    message: str,
    *,
    retry_after_seconds: int | None = None,
    errors: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
    detail: Any = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "detail": message if detail is None else detail,
    }
    if retry_after_seconds is not None:
        body["retry_after_seconds"] = retry_after_seconds
    if errors:
        body["errors"] = errors
    if request_id:
        body["request_id"] = request_id
    return body


def api_error_response(request: Request, exc: ApiError) -> JSONResponse:
    headers = (
        {"Retry-After": str(exc.retry_after_seconds)}
        if exc.retry_after_seconds is not None
        else None
    )
    return JSONResponse(
        status_code=exc.status,
        content=error_body(
            exc.code,
            exc.message,
            retry_after_seconds=exc.retry_after_seconds,
            errors=exc.errors,
            request_id=getattr(request.state, "request_id", None),
        ),
        headers=headers,
    )


#: HTTP status -> stable ``code`` for errors raised as plain ``HTTPException``.
_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    503: "unavailable",
}


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(
            _STATUS_CODES.get(exc.status_code, "http_error"),
            message,
            request_id=_rid(request),
            detail=exc.detail,
        ),
        headers=getattr(exc, "headers", None),
    )


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's own entries echo the offending ``input`` (an uploaded row, a
    # question); neither the envelope nor ``detail`` repeats it.
    entries = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
        for e in entries
        if e["loc"]
    ]
    return JSONResponse(
        status_code=422,
        content=error_body(
            "validation_error",
            "The request did not match the expected shape.",
            errors=errors,
            request_id=_rid(request),
            detail=entries,
        ),
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # The type only, in the log (see ui.ops.JsonFormatter); never the message.
    logging.getLogger("mulenet.ui.errors").error(
        "unhandled %s", type(exc).__name__, exc_info=(type(exc), None, None)
    )
    return JSONResponse(
        status_code=500,
        content=error_body("internal_error", "Something went wrong.", request_id=_rid(request)),
    )


def register_handlers(app: FastAPI) -> None:
    """One envelope for every error the app can produce, not only ``ApiError``."""
    app.add_exception_handler(StarletteHTTPException, _http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled)


def client_ip(request: Request) -> str:
    """The caller's address, for per-client limits only (never logged).

    On Cloud Run the front end APPENDS the real client address to any
    ``X-Forwarded-For`` the caller sent, so the rightmost entry is the one
    that cannot be forged and the leftmost is whatever the client claimed.
    A limiter keyed on the leftmost value would be bypassed by sending a
    fresh fake address each time.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip() or "unknown"
    return request.client.host if request.client else "unknown"
