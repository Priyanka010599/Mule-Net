# The agents HTTP API

The contract between the `agents/` service and its callers. It exists because
two folders are built against it in parallel and neither may guess what the
other meant.

**This file is frozen once written.** Neither `agents/` nor `ui/` may edit it.
A lane that needs a change stops and reports it, so the change is agreed once
rather than diverging twice.

`shared/contracts.py` remains authoritative for the domain models. This
document defines transport only: paths, request and response shapes, status
codes and auth. Where a field carries a domain model, it carries the shape
`shared/` already defines.

## Service identity

One Cloud Run service, separate from `mulenet-ui` and `mulenet-ingest`. It
holds the only Gemini-on-Vertex code path a deployed caller can reach.

## Authentication

**Cloud Run service-to-service ID token.** The service is deployed without
public invoker access. `mulenet-ui` calls it with an ID token minted for the
agents service as the audience, presented as `Authorization: Bearer <token>`.

Consequences both lanes must honour:

- The service is **not** a public URL. A browser cannot call it directly. The
  UI must never forward a browser-supplied token, and must never expose the
  agents URL to client-side code. The browser talks to `ui`; `ui` talks to
  `agents`.
- Rate limits and the daily cap are still required. Auth bounds *who* may
  call, not *how much* they may spend, and the caller is a public
  unauthenticated web page fanning requests through. The limits are the cost
  control; auth is not a substitute for them.
- A missing or invalid token is Cloud Run's `403`, produced before application
  code runs. The service does not implement its own auth.

## `GET /health`

**Not `/healthz`.** Cloud Run's front end reserves paths ending in `z` and
answers them with its own HTML 404, so `/healthz` never reaches the
container on a `*.run.app` host — confirmed on the live service. An uptime
check pointed at `/healthz` reports permanently unhealthy no matter how
well the service is running. The name is load-bearing.


No model call, no data load.

```json
{"status": "ok"}
```

`200` when this instance can serve real answers. `503 data_unavailable` when
the detection tables are not loaded.

It answers "can this container answer truthfully", not "is Vertex reachable" —
a readiness probe that called Gemini would bill for every health check and
would fail the service whenever the model was briefly unavailable.

The data check is not fussiness. `agents/data.py` degrades per table to a
small built-in mock when a parquet file is absent, which is right locally and
dangerous deployed: Gemini would answer fluently over invented numbers and the
reply would still say `backend: "gemini"`. An instance that cannot reach the
real tables must never become healthy, so Cloud Run rolls the revision back
instead of serving confident fiction.

## `POST /chat`

Router plus `analyst_qa`, against live Gemini on Vertex AI.

**Request**

```json
{"session_id": "str", "message": "str"}
```

`session_id` is caller-supplied and scopes the per-session rate limit. It is
not a secret and grants no authority: treat it as a bucket key, never as
identity. A malformed or absent body is `422`.

**Response `200`**

```json
{
  "answer": "str",
  "backend": "gemini",
  "verified": true,
  "sources": ["str"],
  "regulation_citations": [
    {
      "citation": "str",
      "excerpt": "str",
      "source_uri": "str",
      "doc_name": "str",
      "section": "str",
      "authority": "str"
    }
  ],
  "latency_ms": 0
}
```

**`backend`** is the literal `"gemini"`. The field exists so a caller can
prove which path produced the answer. The service must never return
`"gemini"` for an answer it did not obtain from Gemini on Vertex. If it cannot
reach Vertex it returns an error, not a degraded answer under this label. The
mock path does not live in this service.

**`verified`** means: *every quoted claim in `answer` was re-derived from tool
data by the deterministic verifier in `agents/verify.py`, and all of them
checked out.* It is `true` only on that basis.

`false` does **not** mean the answer is wrong. It means nothing in the answer
was of a kind the verifier can re-derive — a question with no numeric claim
has nothing to check. Both lanes must treat `false` as *"not
machine-checked"*, never as *"unverified, therefore suspect"*. A UI that
renders `false` as a warning would teach an analyst to distrust correct
answers.

If a claim is checked and **fails**, the answer does not ship with
`verified: false`. That path returns an error, because the established
contract is that failed verification withholds the output rather than
publishing it with a flag.

**`sources`** are the tool or table names the answer actually drew on, e.g.
`["rings", "account_evidence"]` — derived from what the run consulted, not
from the model's own account of itself. An empty list is valid. This is
provenance, not citation.

**`regulation_citations`** is every passage `search_regulations` actually
returned on this turn — not a claim about them, the retrieval itself — so a
UI can render citation cards regardless of whether `answer`'s own prose
quotes a passage in full. Deduplicated, capped at 5, empty when the turn
never called that tool. Added when the real Vertex AI Search index (RBI KYC
Master Direction, PMLA 2002, PML Rules 2005, FIU-IND Reporting Format) was
wired into this endpoint; supersedes this doc's earlier "regulation
citations belong to reports, not here."

**`latency_ms`** is server-side wall time for the whole request, integer
milliseconds.

### Additive `/chat` fields (all optional for clients; none replace an old field)

Request: `ui_context` `{tab, ring_id, horizon, bank}`, every key optional.
`tab` is one of `banks|joint|metrics|simulate|subset`; `horizon` and `bank`
come from the `agents/subset.py` menus; `ring_id` must exist (else `422
bad_request`). A bad `tab`/`horizon`/`bank` is FastAPI's own `422`. When
present it is put in the prompt as a delimited trusted block and the model is
told to use that ring instead of choosing one; when absent the question is
answered generically.

Response:

| Field | Meaning |
|---|---|
| `answer` | Prose only. Internal labels (`Router:`, `Narrative:`, `Grounded in ...`) are stripped. |
| `sources` | Unchanged: plain list of tool and table names. |
| `source_details` | `[{tool, args_summary}]`; `args_summary` names tables only, never values. |
| `timings` | `{total_ms, router_ms, answer_ms}` (model answers); `{total_ms}` for precomputed. |
| `agent_trace` | `[{agent, tool, ms}]`, one per agent run. `tool` lists what that run executed; per-tool ms is not measured. |
| `model` | Model that answered; `""` for precomputed. |
| `request_id` | Same as the `X-Request-Id` header (sent on every response, errors included; a sane incoming one is reused). |
| `source` | `"precomputed"` or `"model"`. |

**Suggested questions are precomputed.** The three one-click prompts (ring
count; banks and size of the largest ring; alertable members of the largest
ring) are matched on normalised text, answered from the rings table with no
model call, served before the rate-limit guard, cached by
`DataStore.data_version()`, and returned with `source: "precomputed"` and
`backend: "precomputed"` (not `"gemini"`: no model wrote it). "Largest" means
most members, ties broken by ring id, and the answer says so.

**Limits.** Question length: 1000 characters (`MULENET_AGENT_MAX_MESSAGE_CHARS`).
Model work per question is bounded by the call budget (`analyst_qa` 6 model
turns, `DEFAULT_MAX_LLM_CALLS` 12 otherwise), which is what bounds tool calls.
Wall clock: `request_timeout_s` (90 s) then `504 timeout`, below Cloud Run's
`--timeout=120` (`infra/deploy.sh`). Daily: `MULENET_AGENT_DAILY_CAP` requests
and `MULENET_AGENT_DAILY_TOKEN_CAP` Gemini tokens (default 2,000,000, charged
after each answer); either returns `429 daily_cap`. Both counters are in
memory, per instance, and reset on restart or at 00:00 UTC; the deployment
pins `--max-instances=1` so they mean what they say. The UI also limits
`/api/chat` per client (`MULENET_UI_CHAT_RATE_PER_MIN`, default 10).

**Logging.** One line per request, `chat_request {json}`: `request_id`,
`status`, `error_type`, `latency_ms`, `question_chars`, `source`, `model`,
`tools`, `tokens` `{prompt,output,thinking,total,calls}`. Never the question
text, ring or account ids, or rows.

## `GET /cases/{ring_id}`

The verified case for a ring. **No model call, and no verification at request
time.** It serves what the deterministic verifier already produced, published
at `agents/out/cases.parquet`.

**Response `200`** — the case as published, carrying that file's columns:

```json
{
  "case_id": "str",
  "ring_id": "str",
  "verified_at": "2026-09-26T06:02:20.205175+00:00",
  "verification_attempts": 1,
  "verified_claim_count": 4,
  "analyst_notes": "str",
  "status": "open",
  "opened_at": "2026-09-26T06:02:20.205187+00:00",
  "closed_at": null
}
```

Timestamps are tz-aware UTC, serialised ISO-8601. `case_id` must equal
`case_id_for_ring(ring_id)` from `shared/contracts.py`.

**Response `404`**

```json
{"error_type": "no_case"}
```

Returned when no verified case exists for that ring — including when the
investigation was **withheld** because verification failed. A ring without a
verified case gets `404`. It never gets a placeholder, an empty case, or a
case assembled at request time. 45 rings have cases; the rest legitimately
have none, and an unknown `ring_id` is the same `404`.

## `GET /simulate/bounds`

Field bounds and clause names for the live "build your own evasion attempt"
panel. **No model call, no data read** — a plain dict off
`ml.redteam.PATTERN_BOUNDS` and `ml.redteam.EVADABLE_CLAUSES`, so a bound
tightened there reaches the form without a second place to edit. Answers
even on an instance the readiness check would refuse `/chat` from: this
route needs no published table, only the detector code.

**Response `200`**

```json
{
  "fields": {
    "collectors": {"low": 1, "high": 4, "default": 1, "integer": true}
  },
  "evadable_clauses": ["in_degree", "out_degree", "..."]
}
```

## `POST /simulate`

Clamp, synthesise and rule on one caller-submitted evasion pattern, **live,
on this request**. This is `ml.redteam.evaluate()` — the same function that
rules on a Gemini-designed strategy above — called synchronously with no
caching and no precomputation. **No model call anywhere in this path**: see
`agents/simulate.py` for why that is what makes it safe to expose without
the cost ceilings `/chat` needs.

**Request** — every field optional; an omitted one takes
`ml.redteam.PATTERN_BOUNDS`'s default, exactly as an unset field does for a
Gemini strategy. Bounds are clamped, not rejected: an out-of-range value
comes back inside the response's `pattern.clamped`.

```json
{
  "collectors": 2,
  "mules_per_collector": 20,
  "onward_accounts": 4,
  "hop_delay_seconds": 600,
  "amount_per_inbound_hop": 5000,
  "retained_fraction": 0.05,
  "counterparty_banks": 2,
  "reuse_device": false,
  "reuse_kyc_id": false,
  "evades_clause": "in_degree"
}
```

**Response `200`** — shaped like one entry of the report `ml.redteam.as_json()`
writes (`ml/experiments/redteam/metrics.json`):

```json
{
  "name": "judge-built pattern",
  "description": "str",
  "targets": "judge-specified pattern",
  "aimed_at_clause": "in_degree",
  "why_it_should_evade": "str",
  "pattern": {"collectors": 2, "...": "...", "clamped": []},
  "caught": true,
  "caught_by": "velocity_ring",
  "reasoning": "str",
  "flagged_accounts": 2,
  "flagged_collectors": 2,
  "rings_formed": 1,
  "stage2_alertable": 0,
  "stage2_context": 0,
  "clause_values": {"<collector hashed id>": {"in_degree": {"value": 20, "passes": true, "requirement": "x >= 16"}}}
}
```

`evades_clause` is never validated against the known clause names: kept
verbatim, same treatment `ml.redteam.parse_strategies` gives a model's claim,
because what was *aimed at* is part of the record even when it names the
wrong clause.

Validation failures (a malformed body) are FastAPI's default `422` shape, not
the error envelope below — same as every other route.

## `GET /subset/options`

Horizon and bank choices for the live "load a subset, score it live" panel.
**No model call, no data read** — a plain dict off `agents.subset.HORIZONS`
and `agents.subset.BANK_CHOICES`. Answers even on an instance the readiness
check would refuse `/chat`, `/subset/load` or `/subset/score` from.

**Response `200`**

```json
{
  "horizons": [
    {"key": "30m", "label": "30 minutes after the first transfer", "minutes": 30}
  ],
  "banks": ["all", "BANK_A", "BANK_B", "BANK_C"]
}
```

## `POST /subset/load`

Build features for one caller-chosen, **real** (not synthetic) slice of the
published dataset, live, on this request — the "load" half of the two-step
load-then-score structure. Calls `ml.features.build_features` over
`transfers` filtered to `timestamp <= (first transfer + horizon)`. Evaluates
no rule and flags nothing; see `POST /subset/score` for that.

**Request** — both fields are closed enumerations; anything else is a `422`:

```json
{"horizon": "24h", "bank": "all"}
```

`horizon` — one of `30m`, `2h`, `6h`, `24h`, `3d`, `7d` (default `24h`).
`bank` — one of `all`, `BANK_A`, `BANK_B`, `BANK_C` (default `all`); narrows
which accounts are *shown*, never what is computed — see
`agents/subset.py`'s module docstring on why the bank filter cannot be
applied before feature-building without corrupting
`cb_distinct_counterparty_banks`.

**Response `200`**

```json
{
  "horizon": "24h",
  "horizon_label": "1 day after the first transfer",
  "bank": "all",
  "cutoff": "2025-01-02T00:00:13+00:00",
  "transfers_considered": 14481,
  "accounts_in_scope": 30000,
  "load_ms": 991
}
```

## `POST /subset/score`

Load, then rule on, the same caller-chosen subset — live, on this request,
**no caching of the result across different subset choices**. Same request
shape as `POST /subset/load` above (and reruns its feature build rather than
sharing state with a prior `/subset/load` call — see
`agents/subset.py`'s module docstring). Calls `ml.rule.velocity_decision`
twice, unmodified: once over `cb_` features (the shipped rule), once over
`sb_` features with `require_bank_spread=False` (the single-bank variant
`ml/rule.py` itself reports).

**Response `200`**

```json
{
  "horizon": "7d",
  "horizon_label": "7 days after the first transfer",
  "bank": "BANK_B",
  "cutoff": "2025-01-08T00:00:13+00:00",
  "transfers_considered": 118406,
  "accounts_in_scope": 10000,
  "cross_bank_flagged": 10,
  "single_bank_flagged": 0,
  "flagged_accounts": [
    {
      "hashed_account_id": "str",
      "bank_id": "BANK_B",
      "caught_cross_bank": true,
      "caught_single_bank": false,
      "in_degree": 33,
      "out_degree": 11,
      "distinct_counterparty_banks": 3,
      "passthrough_ratio": 1.10,
      "passthrough_speed_seconds": 1260.0
    }
  ],
  "flagged_accounts_truncated": false,
  "load_ms": 2028,
  "score_ms": 2,
  "total_ms": 2030
}
```

`flagged_accounts` is capped at `agents.subset.MAX_FLAGGED_ACCOUNTS_RETURNED`
(50), sorted by cross-bank in-degree descending; `flagged_accounts_truncated`
says whether more were flagged than shown. Both `/subset/load` and
`/subset/score` share the readiness gate `/chat` uses: `503
data_unavailable` when `accounts`/`transfers` are not on disk.

## Additions since the original contract

Added at the owner's request, after the contract above was frozen. Nothing
above is changed or removed; each item below is additive, and a client written
against the original contract keeps working. (The "Additive `/chat` fields"
subsection under `POST /chat` is part of the same history.)

### `POST /chat/jobs` and `GET /chat/jobs/{request_id}`

`POST /chat` holds one request open for the whole router, tools and narrative
round trip (6 to 10 seconds with a live model). The job routes run the **same**
coroutine, so an answer, its `verified`, `sources` and any error envelope cannot
differ from the synchronous route's.

`POST /chat/jobs` takes the same body as `POST /chat` (including `ui_context`)
and returns at once:

```json
{"request_id": "str", "status": "working", "stage": "routing"}
```

A request the synchronous route would refuse up front (rate limit, validation,
data not ready) is refused by this `POST` itself, with the same status and
envelope. A `429 rate_limited` is also returned when too many jobs are already
working (`MULENET_AGENTS_MAX_JOBS`, default 6).

`GET /chat/jobs/{request_id}` polls:

```json
{
  "request_id": "str",
  "status": "working | done | failed",
  "stage": "routing | querying | writing",
  "elapsed_ms": 0,
  "error": null
}
```

When `status` is `done` the whole `ChatOut` body (`answer`, `backend`,
`verified`, `sources`, ... see above) is flattened beside those keys. When it is
`failed`, `error` is the standard envelope `{error_type, message,
retry_after_s}`. An unknown or expired id is `404 not_found`. Jobs are in memory
on one instance and kept for `MULENET_AGENTS_JOB_TTL_S` (default 600) after they
finish.

### `ui_context.chart`

`ui_context` gains an optional `chart`, from a fixed menu
(`shared/metric_catalog.py` `CHART_MENU`): `false_alerts_per_catch_t5`,
`pr_auc_t5`, `ring_recall_single_vs_joined`, `warning_coverage_by_span`,
`lead_time`, `coalition_recall_by_banks_joined`, `redteam_results`,
`ibm_aml_benchmark` and `race_to_exit`. A value outside the menu is FastAPI's own
`422` with a field-level error and no model call. When present it goes into the
same delimited trusted block as the other keys, with a sentence naming the tool
that answers it (`get_ring_race` for `race_to_exit`, `get_metric` otherwise).

### Tools available to `analyst_qa`

Read-only, like the others. Unknown input is the standard structured tool error;
none returns simulator ground truth (true ring ids, `is_mule`, roles, cohorts).

| Tool | Answers | Reads |
|---|---|---|
| `get_ring_race(ring_id)` | When single-bank and cross-bank each first flagged a DETECTED ring, minutes before or after the money reached the exit account, `cross_bank_earlier`, `earlier_by_minutes`, `joining_helps`. A cluster that mixes several simulated rings is reported `ambiguous` with no race. | `agents/out/ring_race.json` (written by `python -m ui.build_snapshot`) |
| `describe_metrics()` | The charts and claim ids `get_metric` accepts, and the withdrawn claims that must not be quoted. | `agents/out/metric_catalog.json` (written by `python -m ml.claims`) |
| `get_metric(name)` | The value(s), qualifier, claim id, source file and key path for a chart or a claim, as `docs/claims.md` states them. May carry a `documentation_discrepancy` to report, not resolve. | `agents/out/metric_catalog.json` |

A missing, unreadable or old-schema artifact is a tool error ("race data
unavailable" or "metrics unavailable"), not a service failure. Both files are in
`infra/agents_fetch.py`'s fetch list, which the deploy checks on disk.

`POST /chat` also answers the race card's "Explain this" question without a model
call (`source: "precomputed"`, `backend: "precomputed"`, `sources:
["get_ring_race"]`) when `ui_context.ring_id` is a ring with an unambiguous race.
Otherwise the question goes to the model, which has `get_ring_race`.

## Errors

Every non-2xx application response carries:

```json
{"error_type": "str", "message": "str", "retry_after_s": null}
```

`retry_after_s` is an integer when the caller should wait, `null` otherwise.

| Status | `error_type` | When |
|---|---|---|
| `404` | `no_case` | `GET /cases/{ring_id}` with no verified case |
| `422` | — | Request validation (FastAPI's default shape) |
| `429` | `rate_limited` | Per-session or per-IP limit exceeded. `retry_after_s` set. |
| `429` | `daily_cap` | Service-wide daily cap reached. `retry_after_s` = seconds to reset. |
| `409` | `no_tool_evidence` | The answering agent produced text without executing a single tool, so nothing grounded it. See below. |
| `409` | `unverified` | A claim was checked and failed, so the answer was withheld. |
| `504` | `timeout` | A Gemini call exceeded its timeout |
| `503` | `upstream_unavailable` | Vertex unreachable or refused |
| `503` | `data_unavailable` | The detection tables are not loaded, so no answer would be grounded in real data. Also returned by `/health`. |
| `502` | `malformed_response` | The model returned text this service cannot parse — prose where the prompt requires JSON — and the retries were exhausted. |
| `500` | `internal` | Anything else. Message is displayable and carries no stack trace. |

The two `409`s are the service refusing to publish something it produced.
They are not failures of the request and not crashes: they are the withhold
rule applied at the transport edge. `500 internal` must not be used for
either — a caller cannot distinguish "we would not stand behind this" from
"we broke" if both arrive as `500`.

### Why `no_tool_evidence` exists

`prompts/analyst_qa.md` requires every answer to come from a tool call. An
answer produced with no tool executed is ungrounded no matter how plausible it
reads, and small factual questions are exactly where a model bluffs
convincingly. Returning it as a `200` under `backend: "gemini"` would publish
a guess with the authority of a query.

So the rule is mechanical, not advisory: **zero executed tool calls means no
answer is returned.** It is enforced on the count of tool executions observed
at the call path, never on the model's own account of what it used.

### Retries, and the two model failures worth distinguishing

Both of these are non-deterministic on identical input, so a retry earns its
call rather than being a formality:

- **empty completion** — no text at all. Nothing was produced to stand behind.
- **unreadable completion** — text, but not the JSON the prompt requires.
  Observed live as correct prose with no envelope, on a question the model had
  genuinely queried the data to answer.

**One retry per failure mode, at most three attempts per request.** So an
empty first attempt followed by an unreadable second gets a third, while
nothing retries without bound. Log which modes were seen, in order.

When the attempts are exhausted: an empty completion returns `409
no_tool_evidence`; an unreadable one returns `502 malformed_response`. Neither
is `500` — an empty or malformed completion is an expected model outcome, not a
defect in this service, and a caller must be able to tell "the model gave us
nothing usable" from "we crashed".

Unparsed prose is never published as an `answer`. It may equally be a refusal,
and labelling unvalidated text as a grounded answer is the thing this contract
exists to prevent.

The two `429` variants have distinct `error_type` values because they need
different wording: one means "you are going too fast", the other means "this
demo is finished for today".

## Rules that hold inside the service

**Tool errors are returned, never raised.** An exception inside a tool reaches
the model as a structured error it can recover from — carrying the allowed
tables and the schema where relevant — rather than killing the request. This
property already exists in `agents/tools.py` and must survive being placed
behind HTTP.

**No stack trace reaches a client, on any path.**

**Every Gemini call has a timeout.** A hung model call must not hold a request
open indefinitely.

**Max output tokens is set on every generate call.**

**Cost limits are configured through `agents/config.py` env vars, with
defaults safe for a deployed service rather than convenient for a friendly
caller.**

## What a caller must not assume

- That `/chat` is cheap. Every call bills Vertex.
- That `session_id` is trustworthy. It is a bucket key.
- That `verified: false` is a problem. See above.
- That a `404` from `/cases` is an error to work around. It is the answer.
