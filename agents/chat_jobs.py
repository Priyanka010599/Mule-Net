"""Asynchronous Ask: ``POST /chat/jobs`` and ``GET /chat/jobs/{request_id}``.

Why this exists: ``POST /chat`` holds one HTTP request open for the whole
router + tools + narrative round trip, which is 6-10s with a live model.
A browser sees nothing for that long, and anything between it and the
service with a shorter idle timeout turns a correct answer into an error.
Here the POST returns at once with ``{request_id, status: "working"}`` and
the browser polls the GET, which reports a ``stage``
(``routing`` -> ``querying`` -> ``writing``) until ``status`` is ``done``
or ``failed``.

**It adds no second code path for the answer.** The job runs the very
coroutine ``POST /chat`` runs (``service._chat``: rate limit, wall-clock
timeout, error mapping, ``ChatOut``), so a job's ``answer``, ``verified``,
``sources`` and error envelope cannot drift from the synchronous route's.
The only addition is :func:`stage`, which the answer thread calls through a
context variable; with no job in context (every ``POST /chat``) it is a
no-op, so the synchronous route is unchanged.

Jobs are in memory (one Cloud Run instance, one process; a lost job is a
lost poll, the caller re-asks), kept for a TTL after they finish, and
there is a hard cap on jobs working at once: a job that cannot start is a
``429 rate_limited`` at submit time, not a queue that grows behind a
model that is the bottleneck.

A request the synchronous route would refuse up front (rate limit,
validation) is refused by the ``POST`` too, with the same status and
envelope, rather than as a ``failed`` job: the coroutine is given one
loop turn to run its synchronous checks before the ``POST`` returns.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request

from agents.grounding import RunSink

STAGES = ("routing", "querying", "writing")

DEFAULT_TTL_S = 600.0
DEFAULT_MAX_CONCURRENT = 6
#: Finished jobs kept at most, whatever their age: bounds memory if polls stop.
MAX_RETAINED = 500

_current: contextvars.ContextVar[Job | None] = contextvars.ContextVar(
    "mulenet_chat_job", default=None
)


class JobCapExceeded(Exception):
    """Too many jobs are working at once."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"{limit} chat jobs are already working")


@dataclass
class Job:
    request_id: str
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None
    status: str = "working"
    stage: str = STAGES[0]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def view(self) -> dict[str, Any]:
        """The ``GET`` body. Result fields are flattened beside ``status``."""
        end = self.finished if self.finished is not None else time.monotonic()
        body: dict[str, Any] = {
            "request_id": self.request_id,
            "status": self.status,
            "stage": self.stage,
            "elapsed_ms": int((end - self.created) * 1000),
            "error": self.error,
        }
        if self.result is not None:
            body.update(self.result)
        return body


def stage(name: str) -> None:
    """Report progress from the answer thread. No-op outside a job."""
    job = _current.get()
    if job is not None and job.status == "working":
        job.stage = name


class StageSink(RunSink):
    """The ``sources`` sink, which also tells a job what the tools are doing.

    ADK runs tools on its own thread, where the context variable is not
    visible, so the job is bound here at construction and reached through
    the same closure the sink already travels in.
    """

    def __init__(self, job: Job) -> None:
        super().__init__()
        self._job = job

    def add(self, element: Any) -> None:
        super().add(element)
        if self._job.status == "working":
            self._job.stage = "querying"

    def tool_done(self) -> None:  # noqa: D102 - see RunSink
        """A tool finished: the model is composing, or about to call another."""
        if self._job.status == "working":
            self._job.stage = "writing"


def new_sink() -> RunSink:
    """A fresh provenance sink: a :class:`StageSink` inside a job, else a plain one."""
    job = _current.get()
    return StageSink(job) if job is not None else RunSink()


class JobStore:
    """In-memory jobs with a TTL and a cap on those working at once."""

    def __init__(
        self,
        ttl_s: float | None = None,
        max_concurrent: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s if ttl_s is not None else _env_float(
            "MULENET_AGENTS_JOB_TTL_S", DEFAULT_TTL_S
        )
        self.max_concurrent = max_concurrent if max_concurrent is not None else int(
            _env_float("MULENET_AGENTS_MAX_JOBS", DEFAULT_MAX_CONCURRENT)
        )
        self._clock = clock
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = self._clock()
        done = [j for j in self._jobs.values() if j.finished is not None]
        for job in done:
            if now - (job.finished or now) > self.ttl_s:
                del self._jobs[job.request_id]
        extra = len(self._jobs) - MAX_RETAINED
        if extra > 0:
            oldest = sorted(
                (j for j in self._jobs.values() if j.finished is not None),
                key=lambda j: j.finished or 0.0,
            )
            for job in oldest[:extra]:
                del self._jobs[job.request_id]

    def working(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "working")

    def create(self) -> Job:
        """Reserve a slot and register a job, or raise :class:`JobCapExceeded`."""
        with self._lock:
            self._purge()
            if sum(1 for j in self._jobs.values() if j.status == "working") >= self.max_concurrent:
                raise JobCapExceeded(self.max_concurrent)
            job = Job(request_id=uuid.uuid4().hex, created=self._clock())
            self._jobs[job.request_id] = job
            return job

    def discard(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.request_id, None)

    def get(self, request_id: str) -> Job | None:
        with self._lock:
            self._purge()
            return self._jobs.get(request_id)

    def finish(
        self,
        job: Job,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            job.result = result
            job.error = error
            job.status = "failed" if error is not None else "done"
            job.finished = self._clock()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


async def _error_body(app: FastAPI, request: Request, exc: BaseException) -> dict[str, Any]:
    """The envelope the app's own exception handlers would have produced."""
    for klass in type(exc).__mro__:
        handler = app.exception_handlers.get(klass)
        if handler is not None and klass is not Exception:
            response = await handler(request, exc)
            body = json.loads(bytes(response.body))
            if {"error_type", "message", "retry_after_s"} <= body.keys():
                return {k: body[k] for k in ("error_type", "message", "retry_after_s")}
            break
    detail = getattr(exc, "detail", None)
    if getattr(exc, "status_code", None) == 422:
        return {"error_type": "bad_request", "message": str(detail), "retry_after_s": None}
    return {
        "error_type": "internal",
        "message": f"The agent call failed and no answer was produced ({type(exc).__name__}).",
        "retry_after_s": None,
    }


def register(
    app: FastAPI,
    jobs: JobStore,
    payload_model: type,
    run_chat: Callable[[Any, Request], Awaitable[Any]],
    service_error: Callable[..., Exception],
) -> None:
    """Add the two routes to ``app``.

    ``run_chat`` is the synchronous route's own coroutine (readiness check
    and ``_chat``), so both entry points answer identically.
    ``service_error(status, error_type, message, retry_after_s)`` builds the
    exception the app renders as the pinned envelope.
    """

    async def start_chat_job(payload: Any, request: Request) -> dict[str, Any]:
        try:
            job = jobs.create()
        except JobCapExceeded as exc:
            raise service_error(
                429,
                "rate_limited",
                f"{exc.limit} questions are already being answered. Try again in a moment.",
                5,
            ) from exc

        async def run() -> None:
            token = _current.set(job)
            try:
                out = await run_chat(payload, request)
                jobs.finish(job, result=out.model_dump())
            except asyncio.CancelledError:
                jobs.finish(job, error=_cancelled())
                raise
            except Exception as exc:  # noqa: BLE001 - every failure becomes an envelope
                jobs.finish(job, error=await _error_body(app, request, exc))
                if _refused_up_front(exc):
                    job_errors[job.request_id] = exc
            finally:
                _current.reset(token)

        job_errors: dict[str, BaseException] = {}
        job.task = asyncio.ensure_future(run())
        # One loop turn: lets the route's synchronous checks (rate limit,
        # readiness) run, so a refusal is the POST's own response.
        await asyncio.sleep(0)
        if job.status == "failed" and job.request_id in job_errors:
            exc = job_errors.pop(job.request_id)
            jobs.discard(job)
            raise exc
        return {"request_id": job.request_id, "status": "working", "stage": job.stage}

    # ``payload_model`` is only known at call time, and this module's
    # postponed annotations would leave it an unresolvable string.
    start_chat_job.__annotations__["payload"] = payload_model
    app.post("/chat/jobs")(start_chat_job)

    @app.get("/chat/jobs/{request_id}")
    async def get_chat_job(request_id: str) -> dict[str, Any]:
        job = jobs.get(request_id)
        if job is None:
            raise service_error(
                404, "not_found", f"No chat job {request_id!r}; it may have expired."
            )
        return job.view()


def _refused_up_front(exc: BaseException) -> bool:
    """Refusals the route raises before it awaits anything: rate limits,
    validation, readiness. Answered by the POST itself, not by a failed job."""
    return type(exc).__name__ in {"LimitExceeded", "HTTPException"} or (
        getattr(exc, "error_type", None) == "data_unavailable"
    )


def _cancelled() -> dict[str, Any]:
    return {
        "error_type": "internal",
        "message": "The job was cancelled before it finished.",
        "retry_after_s": None,
    }
