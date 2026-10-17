"""The Gemini-on-Vertex path: five ADK agents over the seven tools.

**Status: written against google-adk 2.9.2, imported, but never
executed.** This environment has no Google Cloud credentials and no
Vertex access, so no call in this module has run against a live model.
The ADK API surface used here was checked by introspecting the installed
package rather than from memory; the shapes below match
``google.adk`` 2.9.2:

* ``LlmAgent(name=, model=, instruction=, description=, tools=,
  output_key=)``
* ``FunctionTool(func)``
* ``Runner(app_name=, agent=, session_service=)``
* ``InMemorySessionService().create_session(app_name=, user_id=,
  session_id=)``
* ``runner.run(user_id=, session_id=, new_message=types.Content(...),
  run_config=)`` yielding ``Event`` objects
* ``RunConfig(max_llm_calls=)`` from ``google.adk.agents.run_config``
  (ADK's own default is 500)

**Three retry budgets sit on top of each other. They compose by
multiplication, so they are stated here rather than discovered.**

1. ``RunConfig(max_llm_calls)`` -- **12** by default -- bounds model
   calls *inside one ADK run*. Retries never raise it.
2. The **transport retry** in :func:`_run_agent` re-runs the whole ADK
   run on a retryable status (:data:`RETRYABLE_STATUSES`): up to 3
   retries, backing off 2s, 8s, 20s.
3. The **parse-retry ledger** in :func:`_parsed_with_retries` re-runs
   :func:`_run_agent` when the reply cannot be parsed: at most
   :data:`MAX_ATTEMPTS` attempts, one per failure mode.

Worst case for one answer is therefore 3 x 4 x 12 = 144 model calls,
which is not a budget so much as an arithmetic accident. Two things stop
it being reached:

* The layers fire on **disjoint causes**. A transport retry happens when
  no reply arrived; a ledger attempt happens only after one did. A run
  that exhausts both had a transport failure *and* an unparseable reply,
  in that order, repeatedly.
* ``_run_agent`` holds a **deadline** of ``config.request_timeout_s``
  and will not start a backoff it cannot afford. The HTTP layer already
  enforces the same number as a wall clock; this gives the in-process
  callers -- the eval harness and the case producer, which have no wall
  clock -- the same bound.

The practical ceiling is that deadline: 90s of transport retrying per
``_run_agent`` call, and the ledger only ever reaches attempt 3 when
attempts 1 and 2 failed for *different* reasons.

One ADK behaviour below is load-bearing and was verified by reading
``google.adk.runners``: the synchronous ``Runner.run`` executes the
agent on a bare ``threading.Thread`` (``create_thread`` then
``asyncio.run``), re-yielding events through a queue. A plain thread
starts with an **empty context**, so a ``ContextVar`` set by the caller
is invisible inside every tool call. Provenance is therefore bound into
each tool wrapper as a closure, never looked up from context; see
:func:`_model_tool`.

What remains unverified is behavioural, not structural: whether Gemini
returns JSON matching the schemas the prompts ask for, how reliably it
picks the right tool, and what the verifier pass rate looks like on real
model output. Those need credentials to answer, and no number in this
module's docstrings should be read as measured.

ADK is an optional import. Every entry point imports it lazily so the
rest of ``agents/`` stays importable and testable with ADK absent.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import threading
import time
from dataclasses import fields, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agents import ask_scope, hidden_labels, regulations, tools, untrusted
from agents.config import AgentsConfig, load_prompt
from agents.mock_agents import Ground, QaAnswer, ReportDraft, RouterDecision
from agents.verify import Claim, InvestigationResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from google.adk.agents import LlmAgent

#: ADK application name, used for session scoping.
APP_NAME = "mulenet"

#: Tools each agent is given. The router gets none on purpose: routing on
#: data would let a lookup decide the destination, and a router that can
#: read the tables will answer the question instead of routing it.
#: Tables ``analyst_qa`` may not query directly.
#:
#: ``transfers`` is 101 MB and roughly 570k rows. Asked for a ring's total
#: intra-ring amount, the model wrote a three-way join across it and
#: ``ring_members`` twice, and the request died on the 90-second wall
#: clock. ``get_ring_transfer_summary`` answers the same question exactly,
#: off the adjacency index, in a few hundred bytes.
#:
#: **This is not a repo-wide policy change.** ``ALLOWED_SQL_TABLES`` in
#: ``shared/tool_interfaces.py`` is frozen and untouched; ``transfers``
#: remains allowlisted there, and the investigator and report agents keep
#: their access. This restriction is ``analyst_qa``'s alone, because
#: ``analyst_qa`` is the one agent on the public request path where
#: latency and token exposure are a cost and liveness property.
ANALYST_QA_DENIED_TABLES = frozenset({"transfers"})


def _restricted_run_sql(denied: frozenset[str]) -> Any:
    """``run_sql`` that refuses ``denied`` tables, by raising.

    Raises rather than returning the error, so it travels the same path
    as every other tool failure: :func:`_model_tool` turns it into the
    structured, schema-carrying error the model can read and recover
    from. A tool that returned a bespoke error dict here would bypass
    that and the model would see two different failure shapes.

    ``functools.wraps`` keeps ``__name__ == "run_sql"``, which matters
    three times over: the declaration ADK generates is unchanged, the
    provenance sink records the same name, and the grounding check in
    ``service.py`` -- which matches against names derived from
    ``tools.ALL_TOOLS`` -- still counts it.
    """

    @functools.wraps(tools.run_sql)
    def run_sql(query: str) -> Any:
        from agents.sql_guard import validate_sql  # noqa: PLC0415 - avoids a cycle

        # Validated first, so the tables checked are the ones the guard
        # resolved from the parse tree rather than names matched in the
        # query text. `FROM transfers` and a quoted or aliased spelling
        # of it must fail alike.
        named = validate_sql(query).tables
        blocked = sorted(named & denied)
        if blocked:
            raise ValueError(
                f"this agent may not query {', '.join(blocked)} directly: the "
                "table is large enough that a join across it times out the "
                "request. For money moved inside a ring, call "
                "get_ring_transfer_summary(ring_id), which returns exact "
                "totals, cross-bank counts and the time span. For a specific "
                "payment chain, call trace_money_flow."
            )
        rows = tools.run_sql(query)
        ask_scope.check_rows(rows, tools._store())
        return rows

    return run_sql


AGENT_TOOLS: dict[str, tuple[Any, ...]] = {
    "router": (),
    # analyst_qa answers on the public request path, so it gets the
    # aggregate view of a ring and not the row-level one: no
    # get_ring_subgraph (~113k prompt tokens on a live investigation) and
    # no direct transfers access.
    "analyst_qa": (
        tools.describe_tables,
        _restricted_run_sql(ANALYST_QA_DENIED_TABLES),
        ask_scope.scoped_account_tool(tools.get_account_profile, "hashed_account_id"),
        tools.get_ring_summary,
        tools.get_ring_transfer_summary,
        tools.get_ring_race,
        tools.describe_metrics,
        tools.get_metric,
        ask_scope.scoped_account_tool(
            tools.get_explanation,
            "subject_id",
            only_when=lambda args: args.get("subject_type") == "account",
        ),
        tools.search_regulations,
    ),
    # get_ring_summary leads deliberately: the subgraph is ~50x larger
    # and made the investigator the slowest agent in the system.
    # ALL_TOOLS already lists get_ring_summary ahead of the subgraph.
    # Prepending it again declared the same function twice and Vertex
    # rejected the whole request with 400 "Duplicate function
    # declaration".
    "investigator": tools.ALL_TOOLS,
    "report": (
        # Summary first, for the same reason as the investigator: a live
        # report drafted from the full subgraph cost 687,170 prompt
        # tokens over 7 calls and took 127 seconds. The summary carries
        # the counts and behaviour a narrative actually quotes.
        tools.get_ring_summary,
        tools.get_ring_subgraph,
        tools.get_account_profile,
        tools.get_document_reuse,
        tools.get_explanation,
        tools.search_regulations,
    ),
    # No tools, deliberately. The red-team agent is told the detector's
    # thresholds in its prompt and asked to design patterns that clear
    # them; it must not be able to read the detection tables, because a
    # strategy fitted to the 24 rings already in ml/out/ is a strategy
    # that evades this dataset rather than this rule. It never sees an
    # account id either -- ml/redteam.py synthesises every account the
    # strategies imply.
    "redteam": (),
}

_DESCRIPTIONS = {
    "router": "Routes an analyst request to exactly one specialist agent.",
    "analyst_qa": "Answers short factual questions from the detection tables.",
    "investigator": "Explains why a ring or account was flagged, with checkable claims.",
    "report": "Drafts a regulation-cited suspicious transaction report.",
    "redteam": "Designs transfer patterns built to stay under the detector's thresholds.",
}


def _json_safe(value: Any) -> Any:
    """Coerce a tool result into something ``json.dumps`` accepts.

    ADK serialises every tool result into the *next* request's contents,
    and google-genai does that with a plain ``json.dumps``. The tools in
    ``agents/tools.py`` return pydantic models carrying ``datetime``
    fields (``Ring.detected_at``, ``Transfer.timestamp``,
    ``AccountEvidence.observed_at``), so without this the second turn of
    any investigation dies with::

        TypeError: Object of type datetime is not JSON serializable

    This converts at the ADK boundary rather than changing the tool
    signatures, because those signatures are the contract in
    ``shared/tool_interfaces.py`` and the mock path and verifier call the
    same functions expecting real model objects.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    if is_dataclass(value) and not isinstance(value, type):
        # RingSubgraph. Walked field by field rather than with
        # dataclasses.asdict, which deep-copies and would not recurse
        # through the pydantic models inside. Serialising it as a dict is
        # also what puts `truncated` and the true totals in front of the
        # model -- as a bare object it fell through untouched and ADK hit
        # the raw Decimal inside Transfer.amount.
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    # numpy scalars out of pandas (run_sql) expose .item(); nothing else
    # reaching here does.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


#: Tools whose failures should carry the queryable schema back to the
#: model, because the likely cause is a wrong table or column name.
_SCHEMA_AWARE_TOOLS = frozenset({"run_sql", "describe_tables"})


def _sql_error_payload() -> dict[str, Any]:
    """The schema a model needs to rewrite a failed query."""
    from shared.tool_interfaces import (  # noqa: PLC0415 - avoids a cycle
        ALLOWED_SQL_TABLES,
        SQL_ROW_LIMIT,
    )

    return {
        "allowed_tables": sorted(ALLOWED_SQL_TABLES),
        "row_limit": SQL_ROW_LIMIT,
        "schema": tools.describe_tables()["tables"],
    }


def _error_payload(name: str, exc: Exception) -> dict[str, Any]:
    """Describe a tool failure in a form the model can act on."""
    payload: dict[str, Any] = {
        "error": {
            "tool": name,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "recoverable": True,
        }
    }
    if name in _SCHEMA_AWARE_TOOLS:
        # The live failure this exists for: the model wrote
        # `WHERE num_banks = 3`, a column that does not exist. Telling it
        # only "query failed" leaves it to guess again; handing back the
        # real columns lets it fix the query on the next turn.
        payload["error"]["hint"] = (
            "The query named something that does not exist. Use the schema "
            "below and reissue the query."
        )
        payload["error"].update(_sql_error_payload())
    return payload


#: Every tool execution observed in this process, newest last.
#:
#: **A second, independent observer, on purpose.** This is not how
#: ``sources`` is built; it is how a claim about ``sources`` gets checked.
#: The first version of this module collected provenance through a
#: ``ContextVar`` and returned an empty list on every live request -- and
#: its own instrumentation could not reveal that, because the
#: instrumentation *was* the broken thing. A collector that records
#: nothing and a run that called nothing produce the same empty list.
#:
#: So this is written by a different mechanism -- a module-level list
#: under a lock, which crosses threads -- at the moment the tool function
#: is entered. If the two ever disagree, the disagreement is the bug
#: report.
#:
#: Bounded, because a long-lived Cloud Run instance must not accumulate
#: one entry per tool call for the life of the process.
TOOL_CALLS: list[dict[str, Any]] = []

#: Entries kept in :data:`TOOL_CALLS` before the oldest are dropped.
_TOOL_LOG_LIMIT = 500

_TOOL_LOG_LOCK = threading.Lock()

_log = logging.getLogger("agents.adk_agents")


def reset_tool_log() -> None:
    """Clear :data:`TOOL_CALLS`."""
    with _TOOL_LOG_LOCK:
        TOOL_CALLS.clear()


def _log_tool_execution(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Record one tool execution, to the log and to :data:`TOOL_CALLS`.

    Fires whether or not anything asked for provenance, and whichever
    thread ADK runs the tool on. Arguments are truncated: one of them can
    be a whole SQL statement.
    """
    entry = {
        "tool": name,
        "args": [str(value)[:200] for value in args],
        "kwargs": {key: str(value)[:200] for key, value in kwargs.items()},
        "at": datetime.now(UTC).isoformat(),
        "thread": threading.current_thread().name,
    }
    with _TOOL_LOG_LOCK:
        TOOL_CALLS.append(entry)
        if len(TOOL_CALLS) > _TOOL_LOG_LIMIT:
            del TOOL_CALLS[: len(TOOL_CALLS) - _TOOL_LOG_LIMIT]
    _log.info(
        "TOOL EXECUTED %s args=%s kwargs=%s thread=%s",
        name,
        entry["args"],
        entry["kwargs"],
        entry["thread"],
    )


def _note_sql_tables(
    sink: set[str], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Record the tables a ``run_sql`` call actually named.

    The query is re-parsed with the same validator that gates execution,
    so the names recorded are the ones the guard resolved -- not whatever
    the model claimed to be querying.
    """
    query = kwargs.get("query") or (args[0] if args else "")
    if not isinstance(query, str):
        return
    from agents.sql_guard import validate_sql  # noqa: PLC0415 - avoids a cycle

    try:
        tables = validate_sql(query).tables
    except Exception:  # noqa: BLE001 - an invalid query has no tables to name
        return
    sink.update(tables)


def _model_tool(
    func: Any, sink: set[str] | None = None, citations: list[dict[str, str]] | None = None
) -> Any:
    """Wrap a tool so ADK receives a JSON-serialisable result, or an error.

    Three jobs.

    *Serialisation*: results contain ``datetime`` and ``Decimal``, which
    ADK cannot encode.

    *Error containment*: a raising tool is fatal. When a tool raised, the
    exception propagated out through the Runner and killed the whole
    request -- the caller got ``None`` back and, worse, **the model never
    saw the error**, so it could not correct itself. An agent that cannot
    observe its own failures cannot recover from them, which makes the
    retry loop above it decorative. Every exception therefore becomes a
    value the model reads and can act on.

    *Provenance*: the tool name, and for ``run_sql`` the tables the guard
    resolved, are added to ``sink``.

    ``sink`` is bound **here, at wrap time**, rather than looked up when
    the tool runs. That is the fix for the empty-``sources`` bug: ADK's
    synchronous ``Runner.run`` executes the agent on a bare
    ``threading.Thread``, and a plain thread starts with an empty
    context, so a ``ContextVar`` set by the caller cannot be seen inside
    any tool call. A closure over a set does not care which thread
    dereferences it.

    ``functools.wraps`` keeps ``__name__``, ``__doc__`` and
    ``__wrapped__``, so the declaration ADK generates from the signature
    is unchanged and the model sees the same tool it always did.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Logged and recorded before the call, not after: a tool that
        # failed was still executed. Hiding that would make provenance
        # disagree with what the model was told, and would let a run
        # whose only tool call raised look like a run that never queried.
        _log_tool_execution(func.__name__, args, kwargs)
        if sink is not None:
            sink.add(func.__name__)
            if func.__name__ == "run_sql":
                _note_sql_tables(sink, args, kwargs)
        try:
            result = hidden_labels.scrub(_json_safe(func(*args, **kwargs)))
            if func.__name__ != "search_regulations":
                # The regulation corpus is ours; everything else carries free
                # text from data we did not write. See agents.untrusted.
                result = untrusted.wrap_tool_result(result, func.__name__)
            record_result = getattr(sink, "record_result", None)
            if record_result is not None:
                # Kept for the grounding check: every number an answer
                # states must be in one of these (agents.grounding).
                record_result(func.__name__, result)
            if func.__name__ == "search_regulations":
                # Low-volume tool (unlike run_sql, which this would
                # spam for). Diagnostic for why the model declined a
                # regulation question despite a real index existing:
                # was the call even made with a sane query, and what,
                # if anything, came back.
                _log.warning(
                    "search_regulations args=%s kwargs=%s -> %d result(s): %s",
                    args,
                    kwargs,
                    len(result) if isinstance(result, list) else -1,
                    [r.get("citation") for r in result] if isinstance(result, list) else result,
                )
                if citations is not None and isinstance(result, list):
                    # Every passage actually returned, not just the one
                    # (if any) the model's own prose went on to quote --
                    # the caller renders these as citation cards
                    # regardless of what the final answer says.
                    citations.extend(result)
            return result
        except Exception as exc:  # noqa: BLE001 - a raising tool kills the request
            # The model sees this error and can act on it (that is the
            # whole point of _error_payload), but until now nothing did:
            # an operator reading Cloud Logging for why a tool kept
            # failing found nothing, because the only trace of the
            # exception was inside the model's own context window.
            _log.warning("TOOL RAISED %s: %s: %s", func.__name__, type(exc).__name__, exc)
            return _error_payload(func.__name__, exc)
        finally:
            # A job's sink (agents.chat_jobs.StageSink) learns the tool is
            # done; a plain set has no such hook.
            tool_done = getattr(sink, "tool_done", None)
            if tool_done is not None:
                tool_done()

    return wrapper


#: Backoff schedule for a rate-limited model call, in seconds.
_BACKOFF_SECONDS = (2.0, 8.0, 20.0)

#: Token usage accumulated since :func:`reset_usage`, per agent.
#:
#: Thinking tokens are tracked separately because they dominate: one
#: live investigator turn spent 538 thinking tokens against 27 of
#: output, so a total that hides the split makes the bill look
#: inexplicable.
LAST_USAGE: dict[str, dict[str, int]] = {}


def reset_usage() -> None:
    """Clear accumulated token usage."""
    LAST_USAGE.clear()


_capture = threading.local()


def capture_usage() -> dict[str, int]:
    """Start counting this thread's token usage for one request.

    :data:`LAST_USAGE` is process-wide, so two concurrent requests would
    mix their counts there. A request that runs its agents on one worker
    thread calls this first and reads the returned dict afterwards:
    ``prompt``, ``output``, ``thinking``, ``total`` and ``calls``.
    """
    mine = {"prompt": 0, "output": 0, "thinking": 0, "total": 0, "calls": 0}
    _capture.usage = mine
    return mine


def _accumulate_usage(event: Any) -> None:
    """Add one event's token usage to :data:`LAST_USAGE`."""
    usage = getattr(event, "usage_metadata", None)
    if usage is None:
        return
    mine = getattr(_capture, "usage", None)
    if mine is not None:
        mine["prompt"] += getattr(usage, "prompt_token_count", 0) or 0
        mine["output"] += getattr(usage, "candidates_token_count", 0) or 0
        mine["thinking"] += getattr(usage, "thoughts_token_count", 0) or 0
        mine["total"] += getattr(usage, "total_token_count", 0) or 0
        mine["calls"] += 1
    bucket = LAST_USAGE.setdefault(
        getattr(event, "author", "unknown"),
        {"prompt": 0, "output": 0, "thinking": 0, "total": 0, "calls": 0},
    )
    bucket["prompt"] += getattr(usage, "prompt_token_count", 0) or 0
    bucket["output"] += getattr(usage, "candidates_token_count", 0) or 0
    bucket["thinking"] += getattr(usage, "thoughts_token_count", 0) or 0
    bucket["total"] += getattr(usage, "total_token_count", 0) or 0
    bucket["calls"] += 1


#: Statuses where the identical request may simply succeed later.
#:
#: ``429`` is rate limiting. The ``5xx`` codes are Vertex failing on its
#: own side -- a live 25-question eval died on
#: ``ServerError: 504 DEADLINE_EXCEEDED`` from the ``global`` endpoint,
#: mid-run, with nothing wrong with the request. Retrying those is the
#: difference between an unattended eval finishing and an unattended eval
#: being a coin toss.
#:
#: ``400`` is deliberately absent: it means the request itself is wrong,
#: usually the token limit, and repeating it changes nothing.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _status_code(exc: Exception) -> int | None:
    """Best-effort HTTP status for a google-genai error."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    text = str(exc)
    # Longest-lived first: a message mentioning 504 must not be read as a
    # 500 because 500 happened to be scanned earlier.
    for code in (429, 400, 500, 502, 503, 504):
        if str(code) in text:
            return code
    return None


#: Thinking-token budget per agent.
#:
#: Thinking dominated the first live run -- one investigator turn spent
#: 538 thinking tokens to emit 27 of output, and the router took 12-31
#: seconds to produce roughly 60 tokens of JSON. The budgets below are
#: set by how much genuine deliberation each job needs:
#:
#: * ``router`` picks one of three labels from a sentence. That is
#:   classification, not reasoning, and it was the single slowest step.
#: * ``analyst_qa`` chooses a tool and reads one number back.
#: * ``investigator`` composes a narrative and must choose claims that
#:   will survive verification, so it keeps a real budget -- just a
#:   bounded one rather than the unbounded default.
#: * ``report`` is given a moderate budget; the hard
#:   checks run in Python either way.
#:
#: ``0`` disables thinking. Raise a budget rather than remove the cap if
#: an agent starts producing thin answers.
THINKING_BUDGETS: dict[str, int] = {
    "router": 0,
    "analyst_qa": 128,
    "investigator": 512,
    "report": 512,
    # The largest budget of any agent, and the only one justified by the
    # task rather than the output length: every other agent looks facts
    # up, while this one has to hold five ANDed thresholds in mind and
    # find a pattern that fails exactly one of them.
    "redteam": 1024,
}


#: Ceiling on generated tokens per call, per agent.
#:
#: Output is billed roughly ten times input, so an unbounded response is
#: the most expensive thing this service can do, and "the model rambled"
#: becomes a cost event rather than a nuisance. Every generate call
#: therefore carries a limit.
#:
#: Sized off what the agents actually emit, with room to spare: the
#: router returns a three-field JSON object, ``analyst_qa`` a number and
#: its basis, and one live investigator turn was observed emitting 27
#: output tokens. The narrative agents keep large budgets because
#: truncated JSON fails to parse and costs a whole retry, which is more
#: expensive than the tokens saved.
#:
#: Per-agent rather than one number because the four differ by an order
#: of magnitude in what they legitimately produce.
#: ``AgentsConfig.max_output_tokens``, when set, caps all of them.
MAX_OUTPUT_TOKENS: dict[str, int] = {
    "router": 256,
    "analyst_qa": 1024,
    "investigator": 4096,
    "report": 8192,
    # Three strategies, each a description plus a ten-field pattern
    # block. Measured against the prompt's own example, that is ~900
    # tokens; the ceiling is generous because a reply truncated
    # mid-object costs a whole retry to discover.
    "redteam": 4096,
}


def _content_config(types: Any, name: str, config: AgentsConfig) -> Any:
    """Build the per-agent generation config.

    Carries three cost and liveness controls onto every Gemini call: the
    thinking budget, the output-token ceiling, and an HTTP timeout. The
    timeout is here rather than only around the request because a single
    hung call is what holds a request open, and ADK makes several calls
    per run.
    """
    budget = MAX_OUTPUT_TOKENS.get(name, 1024)
    if config.max_output_tokens is not None:
        budget = min(budget, config.max_output_tokens)
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=THINKING_BUDGETS.get(name, 256)
        ),
        max_output_tokens=budget,
        # google-genai takes this in milliseconds.
        http_options=types.HttpOptions(
            timeout=int(config.model_timeout_s * 1000),
        ),
    )


class AdkUnavailableError(RuntimeError):
    """Raised when the Gemini path is requested but ADK cannot be imported."""


#: Model failures worth retrying, and telling apart afterwards.
#:
#: Both are non-deterministic on identical input -- measured live, the
#: same question produced each of them and a clean answer across five
#: attempts -- so a retry earns its call. They end differently, which is
#: why they are named rather than counted: see
#: :class:`ModelUnusableError`.
RETRYABLE_MODES = ("empty", "unreadable")

#: Attempts for one agent call: the first, plus one retry per mode.
#:
#: Derived from the mode list rather than written as ``3``, so adding a
#: third retryable mode cannot silently leave a cap that no longer
#: matches the rule in ``docs/agents_api.md``.
MAX_ATTEMPTS = 1 + len(RETRYABLE_MODES)


class ModelUnusableError(RuntimeError):
    """Every attempt produced something unusable.

    Carries the modes in the order they were seen, because the caller
    classifies on them: a run that ended empty produced nothing to stand
    behind, and a run that ended unreadable produced text we could not
    parse. Those are a ``409`` and a ``502`` respectively, and collapsing
    them would tell a caller "we broke" for a model that merely rambled.
    """

    def __init__(self, agent: str, modes: list[str]) -> None:
        self.agent = agent
        self.modes = list(modes)
        self.trail = " -> ".join(modes) if modes else "none"
        super().__init__(f"{agent} produced no usable response ({self.trail})")


class CallBudgetExceeded(RuntimeError):
    """The agent used every model call it was allowed and never converged.

    Its own class because it is **our** limit, not an upstream fault, and
    the two need different reactions. ADK raises
    ``LlmCallsLimitExceededError`` from
    ``google.adk.agents.invocation_context``; left unclassified it would
    fall through :func:`agents.service._upstream`, whose test is "did this
    come from a ``google.*`` module", and be reported to the caller as
    ``503 upstream_unavailable``. Exhausting a budget we set ourselves is
    not Vertex being down, and saying so sends whoever is on call to read
    the wrong status page.

    Observed live: ``gemini-3.5-flash`` @ ``asia-south1`` blew through
    ``max_llm_calls=12`` on an eval question that ``gemini-2.5-flash`` and
    ``gemini-3.8-flash`` both answer in about two calls. That is a fact
    about the model, and raising the cap to accommodate it would raise the
    worst-case bill for every other model too -- so it is surfaced rather
    than absorbed.
    """

    def __init__(self, agent: str, limit: int) -> None:
        self.agent = agent
        self.limit = limit
        super().__init__(
            f"{agent} used its whole budget of {limit} model calls without "
            "producing an answer"
        )


class EmptyModelResponse(RuntimeError):
    """The model returned no text at all.

    Its own error class because ``docs/agents_api.md`` treats an empty
    completion as an expected model outcome rather than a defect in this
    service: retry once, then refuse with ``409 no_tool_evidence``.
    Folded in with parse failures it surfaced as ``500 internal``, which
    told the caller this service was broken when in fact Gemini had
    produced an empty completion -- as it did, live, on the one question
    that could not be answered without querying.
    """


@functools.lru_cache(maxsize=1)
def _llm_calls_limit_error() -> type[BaseException]:
    """ADK's call-limit exception, or a sentinel when ADK is absent.

    Looked up rather than imported at module scope because ADK is an
    optional dependency here. The fallback is a private exception class
    that nothing raises, so the ``except`` clause is simply never taken
    when ADK is missing.
    """
    try:
        from google.adk.agents.invocation_context import (  # noqa: PLC0415
            LlmCallsLimitExceededError,
        )
    except ImportError:  # pragma: no cover - needs ADK absent
        class _NeverRaised(Exception):
            pass

        return _NeverRaised
    return LlmCallsLimitExceededError


def _require_adk() -> Any:
    """Import ADK, or fail with an actionable message."""
    try:
        from google.adk.agents import LlmAgent  # noqa: PLC0415 - optional dependency
        from google.adk.agents.run_config import RunConfig  # noqa: PLC0415
        from google.adk.runners import Runner  # noqa: PLC0415
        from google.adk.sessions import InMemorySessionService  # noqa: PLC0415
        from google.adk.tools import FunctionTool  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - needs ADK absent
        raise AdkUnavailableError(
            "google-adk is not installed, so MULENET_AGENT_MODE=vertex cannot "
            "run. Install it, or set MULENET_AGENT_MODE=mock for the offline "
            "path."
        ) from exc
    return LlmAgent, Runner, InMemorySessionService, FunctionTool, types, RunConfig


def build_agent(
    name: str,
    config: AgentsConfig | None = None,
    sink: set[str] | None = None,
    citations: list[dict[str, str]] | None = None,
) -> LlmAgent:
    """Build one ADK agent from its prompt file and tool set.

    Args:
        name: One of ``router``, ``analyst_qa``, ``investigator``,
            ``report``.
        config: Supplies the model id. Defaults to the environment.
        sink: Collects the names of tools this agent actually executes,
            and the tables its SQL named. Bound into each tool wrapper,
            so it fills whichever thread ADK runs the tool on.
        citations: Collects every passage ``search_regulations`` actually
            returned this run, regardless of what the model's own prose
            goes on to quote. See :func:`_model_tool`.

    Returns:
        A configured ``LlmAgent``.

    Raises:
        AdkUnavailableError: If ADK is not installed.
        KeyError: If ``name`` is not one of the four agents.
        FileNotFoundError: If the prompt file is missing.
    """
    if name not in AGENT_TOOLS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENT_TOOLS)}")

    config = config or AgentsConfig.from_env()
    LlmAgent, _, _, FunctionTool, types, _ = _require_adk()

    return LlmAgent(
        name=name,
        model=config.model,
        description=_DESCRIPTIONS[name],
        generate_content_config=_content_config(types, name, config),
        # Prompts live in agents/prompts/*.md, never inline, so they can be
        # reviewed and diffed without reading Python.
        instruction=load_prompt(name),
        tools=[
            FunctionTool(_model_tool(func, sink, citations)) for func in AGENT_TOOLS[name]
        ],
        output_key=f"{name}_output",
    )


def build_all(config: AgentsConfig | None = None) -> dict[str, LlmAgent]:
    """Build all four agents."""
    config = config or AgentsConfig.from_env()
    return {name: build_agent(name, config) for name in AGENT_TOOLS}


def _run_agent(
    name: str,
    message: str,
    config: AgentsConfig | None = None,
    sink: set[str] | None = None,
    citations: list[dict[str, str]] | None = None,
) -> str:
    """Run one agent to completion and return its final text.

    Vertex is selected through the environment variables the google-genai
    client reads (``GOOGLE_GENAI_USE_VERTEXAI``, ``GOOGLE_CLOUD_PROJECT``,
    ``GOOGLE_CLOUD_LOCATION``); ``AgentsConfig`` surfaces them so a
    misconfiguration fails here with a clear message rather than as an
    authentication error several layers down.
    """
    config = config or AgentsConfig.from_env()
    if config.use_vertex and not config.project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is unset but GOOGLE_GENAI_USE_VERTEXAI is on; "
            "the Vertex path needs a project id."
        )

    _, Runner, InMemorySessionService, _, types, RunConfig = _require_adk()

    agent = build_agent(name, config, sink, citations)
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(
        app_name=APP_NAME, user_id="analyst", session_id=f"{name}-session"
    )
    runner = Runner(
        app_name=APP_NAME, agent=agent, session_service=session_service
    )

    content = types.Content(role="user", parts=[types.Part(text=message)])

    # ADK's default is 500 model calls per run. For one HTTP request that
    # is an unbounded bill, so it is always overridden: a tool-calling
    # loop that has not converged in a dozen turns is not going to, and
    # the cost of letting it try is linear.
    budget = config.max_llm_calls_for(name)
    run_config = RunConfig(max_llm_calls=budget)

    # The transport retry is bounded twice over: by the number of backoff
    # steps, and by a deadline. The deadline matters because this budget
    # multiplies with the parse-retry ledger above it -- see the module
    # docstring -- and because the in-process callers (the eval harness,
    # the case producer) have no wall clock of their own. The HTTP path
    # has one, and reusing its value keeps the two from disagreeing.
    deadline = time.monotonic() + config.request_timeout_s

    for attempt, pause in enumerate((*_BACKOFF_SECONDS, None)):
        try:
            return _drain(runner, session.id, content, run_config)
        except _llm_calls_limit_error() as exc:
            # Never retried: the next attempt gets the same budget and
            # spends it the same way. Re-raised as ours so it cannot be
            # mistaken for an upstream outage.
            _log.warning(
                "%s exhausted its %d-call budget without converging",
                name,
                budget,
            )
            raise CallBudgetExceeded(name, budget) from exc
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            code = _status_code(exc)
            if code in RETRYABLE_STATUSES and pause is not None:
                remaining = deadline - time.monotonic()
                if remaining <= pause:
                    # Sleeping would spend the whole budget waiting and
                    # leave nothing to make the call with. Give up now
                    # and report the real cause rather than a timeout.
                    _log.warning(
                        "%s: %s from Vertex, out of retry budget after %d "
                        "attempt(s)",
                        name,
                        code,
                        attempt + 1,
                    )
                    raise
                _log.warning(
                    "%s: %s from Vertex, retrying in %.0fs (attempt %d)",
                    name,
                    code,
                    pause,
                    attempt + 1,
                )
                time.sleep(pause)
                continue
            if code in (400, 429):
                # 400 is usually the token limit, which a retry cannot
                # fix. Return it as a value so the caller reports a
                # useful failure rather than an exception trace, and so
                # a partial result is not mistaken for a crash.
                return json.dumps(
                    {
                        "error": {
                            "agent": name,
                            "error_type": type(exc).__name__,
                            "status": code,
                            "message": str(exc)[:2000],
                            "recoverable": False,
                            "hint": (
                                "Request too large or rejected. Narrow the "
                                "query: call get_ring_subgraph with max_hops=1 "
                                "and check the truncation flag."
                            ),
                        }
                    }
                )
            raise
        finally:
            _ = attempt
    return ""  # pragma: no cover - the loop always returns


def _drain(
    runner: Any, session_id: str, content: Any, run_config: Any = None
) -> str:
    """Run the agent to completion, recording token usage as it goes."""
    final_text = ""
    last_finish: Any = None
    for event in runner.run(
        user_id="analyst",
        session_id=session_id,
        new_message=content,
        run_config=run_config,
    ):
        _accumulate_usage(event)
        last_finish = getattr(event, "finish_reason", None) or last_finish
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(
                part.text for part in event.content.parts if part.text
            )
    if not final_text:
        # An empty completion has several very different causes and the
        # caller cannot tell them apart from the empty string alone.
        # MAX_TOKENS means the output ceiling cut the answer off and the
        # fix is a bigger budget; STOP with no text means the model chose
        # to say nothing; SAFETY means it refused. Logging the reason is
        # the difference between tuning the right knob and guessing.
        _log.warning(
            "empty completion from the model: finish_reason=%s", last_finish
        )
    return final_text


def _balanced_object(text: str) -> str | None:
    """The first ``{...}`` in ``text`` whose braces balance, string-aware.

    Taking first-brace to last-brace (as :func:`_parse_json` first does)
    swallows any prose that follows the object and happens to contain a
    brace; this stops at the object's own closing brace. ``None`` when the
    object never closes, which is what a reply cut off by the output limit
    looks like.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw newlines and tabs that sit inside JSON string literals.

    A model often writes a multi-line ``answer`` with real line breaks in
    it, which JSON forbids. Only characters inside a string are touched.
    """
    out: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            elif ch == "\n":
                out.append("\\n")
                continue
            elif ch == "\r":
                out.append("\\r")
                continue
            elif ch == "\t":
                out.append("\\t")
                continue
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _repair_json(text: str) -> dict[str, Any] | None:
    """Mend the structural slips a model makes around an otherwise good object.

    Three, and only three: prose or a fence around the object, a trailing
    comma, and raw line breaks inside a string. Every one changes how the
    object is written, never what it says. Prose with no object in it, and
    an object that never closes (cut off by the output limit, so some of
    the answer is missing), are NOT repaired: they stay unreadable and go
    to the retry, because publishing half an answer, or wrapping prose in
    an envelope, is the thing ``docs/agents_api.md`` rules out.

    Returns the object, or ``None`` when these repairs do not yield one.
    """
    candidate = _balanced_object(text)
    if candidate is None:
        return None
    for attempt in (
        candidate,
        re.sub(r",(\s*[}\]])", r"\1", candidate),
        re.sub(r",(\s*[}\]])", r"\1", _escape_control_chars_in_strings(candidate)),
        _escape_control_chars_in_strings(candidate),
    ):
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    Models fence JSON in markdown more often than not, so strip a fence
    before parsing rather than failing on it. A reply that is almost JSON
    gets one structural repair pass (:func:`_repair_json`) before it counts
    as unreadable.

    Raises:
        EmptyModelResponse: If there is no text at all.
        ValueError: If there is text but no JSON object in it.
    """
    cleaned = text.strip()
    if not cleaned:
        raise EmptyModelResponse(
            "the model returned an empty completion: no text at all"
        )
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except ValueError:
        repaired = _repair_json(cleaned)
        if repaired is None:
            raise
        _log.info("repaired a structurally invalid JSON reply (%d chars)", len(text))
        return repaired


#: Added to the retry after an unreadable reply. The retry runs in a fresh
#: session, so without it the model has no idea its last answer was refused.
_FORMAT_REMINDER = (
    "\n\n(Your previous reply could not be read as JSON. Answer again from "
    "the tool results, and reply with ONLY the JSON object the instructions "
    "describe: no prose before or after it, no markdown fence.)"
)


def _parsed_with_retries(
    agent: str,
    message: str,
    config: AgentsConfig | None,
    sink: set[str] | None,
    citations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one agent until it returns parseable JSON, or give up.

    One retry per failure mode, at most :data:`MAX_ATTEMPTS` attempts, per
    ``docs/agents_api.md``. A mode seen twice has spent its retry, so the
    loop stops rather than burning the remaining budget on a failure that
    has already repeated.

    **Every agent entry point goes through this.** ``route`` and
    ``answer_question`` did first; ``investigate_with_gemini`` and
    ``draft_report`` were left calling ``_parse_json`` directly, and a
    live eval died on
    ``JSONDecodeError: Expecting ',' delimiter`` from the investigator --
    the same class of failure the ledger exists to absorb, on the one
    path that had not been wired to it. A retry rule applied to three of
    four callers is a rule with a hole in it.

    Lives here rather than in ``agents/service.py`` because both callers
    need it. It was in the HTTP layer first, and running the eval found
    the consequence: :func:`agents.loop.handle` called :func:`route`
    directly, a router replied in prose instead of JSON, and the
    ``ValueError`` killed the whole eval run. A rule that only holds on
    one of two paths into the same model is not a rule.

    ``sink`` is cleared before each attempt, so the provenance returned
    belongs to the attempt that actually answered rather than accumulating
    across failures.

    Raises:
        ModelUnusableError: If the attempts are exhausted.
    """
    seen: list[str] = []
    prompt = message
    while len(seen) < MAX_ATTEMPTS:
        if sink is not None:
            sink.clear()
        if citations is not None:
            citations.clear()
        # Run and parse are deliberately separate statements. Only the
        # parse is classified: an exception out of _run_agent is the call
        # itself failing -- auth, transport, our own bug -- and wrapping
        # that as "unreadable" would report a genuine 500 as a model
        # rambling. A test caught exactly that: a ValueError raised
        # inside the agent call was being answered with 502.
        text = _run_agent(agent, prompt, config, sink, citations)
        try:
            return _parse_json(text)
        except EmptyModelResponse:
            mode = "empty"
        except ValueError:
            mode = "unreadable"
            prompt = message + _FORMAT_REMINDER
        seen.append(mode)
        _log.warning(
            "%s attempt %d: %s completion (modes so far: %s)",
            agent,
            len(seen),
            mode,
            " -> ".join(seen),
        )
        if seen.count(mode) > 1:
            break
    raise ModelUnusableError(agent, seen)


def route(
    request: str,
    config: AgentsConfig | None = None,
    sink: set[str] | None = None,
) -> RouterDecision:
    """Route a request using the Gemini router.

    The router has no tools, so ``sink`` normally stays empty. It is
    accepted so a caller can prove that rather than assume it.

    Raises:
        ModelUnusableError: If the router never returned parseable JSON.
    """
    payload = _parsed_with_retries("router", request, config, sink)
    return RouterDecision(
        destination=payload["destination"],
        subject_id=payload.get("subject_id"),
        reason=payload.get("reason", ""),
    )


def _claims_from_payload(
    payload: dict[str, Any],
    default_subject: str = "",
) -> list[Claim]:
    """Read the ``claims`` array a model returned into typed claims.

    Shared by the investigator and ``analyst_qa`` so the two cannot parse
    the same array differently -- the verifier is one function and the
    claims reaching it have to be built one way.

    A malformed entry is kept rather than dropped, with whatever it gave:
    an unknown ``kind`` or an empty ``subject`` fails the verifier, which
    is the outcome it should have. Silently discarding it would turn a
    broken claim into no claim at all, and no claim reads as "nothing was
    claimed", which is a different and much friendlier verdict than the
    model earned.

    Args:
        payload: The parsed model reply.
        default_subject: Used when an entry names no subject. The
            investigator is always about one subject and can supply it;
            ``analyst_qa`` cannot, so an entry with no subject there is
            left empty and fails.
    """
    items = payload.get("claims", [])
    if not isinstance(items, list):
        _log.warning("claims was %s, not a list; treating as none", type(items).__name__)
        return []
    return [
        Claim(
            id=str(item.get("id", f"c{index}")),
            kind=str(item.get("kind", "")),
            subject=str(item.get("subject", default_subject)),
            value=item.get("value"),
            text=str(item.get("text", "")),
        )
        for index, item in enumerate(items, start=1)
        if isinstance(item, dict)
    ]


def answer_question(
    question: str,
    config: AgentsConfig | None = None,
    sink: set[str] | None = None,
) -> QaAnswer:
    """Answer a factual question using the Gemini analyst_qa agent.

    Args:
        question: The analyst's question.
        config: Defaults to the environment.
        sink: Collects the tools actually executed. The caller uses its
            size to decide whether the answer is grounded at all; see
            ``docs/agents_api.md`` on ``no_tool_evidence``.

    A question that exhausts the call budget is **declined, not failed**.
    An agent still circling after its whole budget has demonstrated it
    cannot answer from the allowed tables, which is what the decline says
    -- and a decline is a correct outcome here, where a crash is not.
    Observed live: ``gemini-3.5-flash`` looped on two questions it should
    have refused, turning a right judgement into a dead run.

    The string is :data:`agents.mock_agents.CANNOT_ANSWER`, the same
    constant the offline path and the eval set use, so the two ways of
    saying "out of scope" cannot drift apart.

    Raises:
        ModelUnusableError: If no attempt returned parseable JSON.
    """
    from agents.mock_agents import CANNOT_ANSWER  # noqa: PLC0415 - avoids a cycle

    raw_citations: list[dict[str, str]] = []
    try:
        payload = _parsed_with_retries("analyst_qa", question, config, sink, raw_citations)
    except CallBudgetExceeded as exc:
        _log.info("analyst_qa declining after exhausting its budget: %s", exc)
        return QaAnswer(
            answer=CANNOT_ANSWER,
            basis="",
            unit="",
            regulation_citations=_dedup_citations(raw_citations),
        )

    answer = str(payload.get("answer", ""))
    basis = str(payload.get("basis", ""))
    # search_regulations is the one tool here whose result the model
    # quotes rather than counts, so ClaimKind (built for re-derivable
    # table facts) cannot check it. Same gate draft_report() uses: a
    # citation not found in this turn's own retrieval -- and not
    # re-findable by a live probe -- is a fabrication risk, not a
    # borderline case, and the whole answer is withheld rather than
    # shipped with an unverifiable quote attached.
    if sink is not None and "search_regulations" in sink and answer != CANNOT_ANSWER:
        if not basis or not regulations.verify_citation(basis, question):
            _log.info(
                "analyst_qa declining: regulation answer's basis %r did not "
                "match a passage search_regulations actually returned",
                basis,
            )
            return QaAnswer(
                answer=CANNOT_ANSWER,
                basis="",
                unit="",
                regulation_citations=_dedup_citations(raw_citations),
            )

    return QaAnswer(
        answer=answer,
        basis=basis,
        unit=str(payload.get("unit", "")),
        # Claims come from the model's own ``claims`` array and from
        # nowhere else. Nothing here reads ``answer`` to guess what was
        # claimed: a kind inferred from the wording of a number would let
        # a coincidence set ``verified: true``, and that field is the one
        # thing in this service a reader is asked to trust.
        claims=_claims_from_payload(payload),
        # Every passage search_regulations actually returned this turn,
        # independent of whether `answer`'s prose quotes it -- the UI
        # renders these as citation cards regardless. Not re-verified
        # here: they ARE the retrieval, not a claim about it.
        regulation_citations=_dedup_citations(raw_citations),
    )


def _dedup_citations(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Unique by citation string, in first-seen order, capped at 5.

    A question that tries several phrasings can retrieve the same
    passage more than once; a judge-facing card list should not repeat
    itself. 5 is generous for a chat answer without dumping an entire
    multi-query session's worth of passages onto one reply.
    """
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in raw:
        key = item.get("citation", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append({**item, "authority": regulations.authority_for(item.get("doc_name", ""))})
    return deduped[:5]


def investigate_with_gemini(
    subject_id: str,
    feedback: str = "",
    config: AgentsConfig | None = None,
) -> InvestigationResult:
    """Investigate a subject using the Gemini investigator.

    Args:
        subject_id: Ring id or hashed account id.
        feedback: Unsupported claims from a failed verification. Appended
            to the request so the retry sees exactly what failed. This is
            the one retry the loop allows.
        config: Defaults to the environment.
    """
    message = f"Investigate {subject_id}."
    if feedback:
        message = f"{message}\n\n{feedback}"

    payload = _parsed_with_retries("investigator", message, config, None)
    if "error" in payload and "narrative" not in payload:
        # The agent call itself failed -- a 400, or a rejected request.
        # Surface it as the failure it is rather than as an empty
        # investigation, which downstream would read as "nothing to
        # report" instead of "nothing was produced".
        raise RuntimeError(
            f"investigator agent failed: {json.dumps(payload['error'])[:500]}"
        )
    return InvestigationResult(
        narrative=str(payload.get("narrative", "")),
        claims=_claims_from_payload(payload, default_subject=subject_id),
        evidence=[str(line) for line in payload.get("evidence", [])],
        raw={"model_payload": payload, "retry_feedback": feedback},
    )


def draft_report(subject_id: str, config: AgentsConfig | None = None) -> ReportDraft:
    """Draft an STR using the Gemini report agent.

    The citation rule is enforced here as well as in the prompt: a ground
    the model returned without a citation is dropped, not trusted. A
    prompt is guidance; this is the gate.
    """
    payload = _parsed_with_retries(
        "report",
        f"Draft a suspicious transaction report for {subject_id}.",
        config,
        None,
    )

    grounds: list[Ground] = []
    warnings: list[str] = []
    for item in payload.get("grounds", []):
        citation = str(item.get("citation", "")).strip()
        if not citation:
            warnings.append(
                f"ground dropped, model returned no citation: {item.get('ground', '')}"
            )
            continue
        ground_text = str(item.get("ground", ""))
        # "No citation, no ground" means a citation the corpus actually
        # returned. A fabricated one is not a weaker citation, it is not
        # a citation, and the prompt cannot enforce that -- this can.
        if not regulations.verify_citation(citation, ground_text):
            warnings.append(
                f"ground dropped, citation is not from the corpus and appears "
                f"fabricated: {citation!r} for {ground_text[:80]!r}"
            )
            continue
        grounds.append(
            Ground(
                ground=ground_text,
                citation=citation,
                source_uri=str(item.get("source_uri", "")),
            )
        )

    synthetic = any(
        regulations.is_synthetic({"citation": g.citation, "source_uri": g.source_uri})
        for g in grounds
    )
    if synthetic:
        warnings.append(
            "citations are synthetic placeholders from a development corpus"
        )
    narrative = str(payload.get("narrative", ""))
    dropped = len(payload.get("grounds", [])) - len(grounds)
    withheld = ""

    if dropped:
        # The prose was written in one pass, so it asserts every ground
        # the model originally produced -- including the ones whose
        # citations turned out to be fabricated. Removing a ground from
        # the structured list does not remove it from the text, and the
        # text is the part a human reads.
        #
        # So the narrative is withheld whole, exactly as
        # investigate_verified withholds an unverified one. The
        # alternative -- cutting the offending sentences -- would publish
        # prose that no model wrote and no human reviewed. The surviving
        # grounds and their verified citations are still returned below,
        # which is the same trade the investigator loop makes: keep the
        # checked data, drop the unchecked narrative.
        withheld = narrative
        narrative = ""
        warnings.insert(
            0,
            f"narrative withheld: {dropped} of {len(payload.get('grounds', []))} "
            "ground(s) cited a regulation passage that is not in the corpus, and "
            "the prose asserts them alongside the verified grounds. The text "
            "cannot be published without republishing an unsupported claim. "
            f"{len(grounds)} ground(s) survived and are listed with their "
            "citations.",
        )

    return ReportDraft(
        narrative=narrative,
        citations=[g.citation for g in grounds],
        grounds=grounds,
        warnings=warnings,
        citations_verified=bool(grounds) and not synthetic,
        withheld_narrative=withheld,
    )


def generate_evasion_strategies(
    count: int = 3,
    config: AgentsConfig | None = None,
) -> list[dict[str, Any]]:
    """Ask Gemini for evasion strategies against MuleNet's own thresholds.

    Returns the ``strategies`` array from the model's reply, untyped and
    unvalidated. Typing it is :mod:`ml.redteam`'s job, and the split is
    deliberate: the ranges a pattern field has to sit in are properties
    of the *generator* that executes it, and a bound enforced here would
    be a second copy of them that could disagree with the one that does
    the clamping.

    What this function does not do is judge the strategies. A reply that
    names a threshold and then picks a value on the wrong side of it is
    passed through and run anyway -- the detector's verdict is the whole
    output of the exercise, and filtering out the ones that look wrong
    here would be grading the model's homework with the answer key.

    Args:
        count: How many strategies to ask for. The prompt asks for three
            and the generator does not care how many come back; this is
            appended to the request so the number is stated in one place
            the reader can see rather than only in the prompt file.
        config: Defaults to the environment.

    Raises:
        ModelUnusableError: If the attempts are exhausted.
        RuntimeError: If the reply carries no ``strategies`` array. That
            is a contract break rather than a bad answer, and the caller
            cannot do anything useful with it.
    """
    payload = _parsed_with_retries(
        "redteam",
        f"Design {count} distinct evasion strategies against the detector "
        "described in your instructions. Reply with the JSON object only.",
        config,
        None,
    )
    strategies = payload.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise RuntimeError(
            "redteam agent returned no strategies array: "
            f"{json.dumps(payload)[:500]}"
        )
    return [item for item in strategies if isinstance(item, dict)]
