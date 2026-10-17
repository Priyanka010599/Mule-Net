# agents

ADK agents on Gemini (Vertex AI), their tools, the verification gate, and the
evaluation harness. Owned by the **agents-engineer** agent. Tool signatures and
contracts come from `shared/`.

## Four agents, and a verifier that is not one of them

| Agent | Job | Tools |
|---|---|---|
| `router` | Routes a request to exactly one specialist. No data tools — a router that can read the tables answers the question instead of routing it. | none |
| `analyst_qa` | Short factual answers, every one from a tool call, each carrying the claims that back it. | `describe_tables`, restricted `run_sql`, profile, ring summary, ring transfer summary, explanation |
| `investigator` | Explains why a ring or account was flagged, emitting checkable claims. | all nine |
| `report` | Drafts a regulation-cited STR. | subgraph, profile, reuse, explanation, `search_regulations` |

**Verification is deterministic Python, not a fifth agent.** `agents/verify.py`
re-derives every claim from the tools and compares. There is no verifier
`LlmAgent`: one was built and never invoked, and it was removed rather than left
in place to inflate the agent count. Asking a model to check another model's
arithmetic is strictly weaker than recomputing it, and the recomputation is
cheap.

What `verify()` enforces:

- **Claim re-derivation.** Each claim kind has one function that recomputes it
  from tool data. Subject-scoped: `ring_member_count` (all members),
  `ring_alertable_member_count` (members that may be alerted on),
  `ring_bank_count`, `ring_banks`, `account_score`, `account_in_ring`,
  `account_bank`, `cross_bank_evidence_count`. Populations, for the questions
  chat actually asks: `table_row_count`, `filtered_row_count`,
  `distinct_value_count`, `column_distinct_values`, `ring_bank_span_count`,
  `top_risk_account`. A claim whose kind is not in that set is unsupported by
  definition — it cannot be checked, so it cannot pass.

  For a population kind the model names *what* to count — a table, a
  `table.column`, a `table.column=value` — and the verifier decides *how*: it
  resolves the table against `ALLOWED_SQL_TABLES`, checks the column against the
  frame, and counts with pandas. **No model-supplied SQL is executed as part of
  a check.** Re-running the model's own query and comparing the result to itself
  would verify nothing, and a claim must not be able to reach `ground_truth`
  through a table name either.
- **Counting.** An unqualified count must equal the *alertable* count. The
  all-members total may appear only in a sentence naming the context members it
  includes. A reader who sees "this ring has 34 accounts" takes all 34 as
  alerts; if six are same-bank context members, that sentence has accused six
  people the system declined to alert on.
- **Identity-link wording.** An `identity_link` member is "linked, needs
  review", never a mule, suspect, launderer or fraudster — and the word "mule"
  may not appear in an identity-link sentence **even inside a negation**, because
  a truncated sentence keeps the noun and drops the "not".
- **Reuse as cause.** Detection fires on velocity and structure. A sentence
  crediting identity reuse with causing a detection describes a detector this
  system does not have. Checked per sentence, not per character window. This
  is about the shipped rule/stage-2 pipeline's firing condition, not about
  Model A's trained feature weights -- identity-derived features are a large
  contributor there (see `ml/experiments/ablation/README.md`); this rule
  still applies because no report narrative cites Model A's score as "why"
  an account was flagged.
- **Nothing to verify is a failure.** An empty narrative with no claims does not
  pass. "No claims failed" is not "the claims held".

### Chat answers: `verify_answer`

`verify()` gates an investigation, which is a narrative. `verify_answer()` gates
a chat answer, which is a value, and `POST /chat`'s `verified` field is its
verdict. Four outcomes, and the eval records which one each question got:

| outcome | `verified` | what the caller gets |
|---|---|---|
| `verified` | `true` | every claim re-derived, and one of them is the answer |
| `nothing_checkable` | `false` | the answer, with no claims to check — a refusal, or a question no claim kind covers |
| `unbacked` | `false` | the answer; the claims held but none of them states it, so the answer itself was not checked |
| `withheld` | — | **no answer at all**: `409 unverified`. A claim was checked and failed. |

Two properties hold this together, and both are tested:

- **Claims are never inferred from answer text.** They arrive from the model as
  structure or not at all. A kind guessed from the wording of a number would let
  a coincidence of phrasing produce `verified: true`.
- **A claim must state the answer.** Otherwise a model could satisfy `verified`
  with any easy true fact while answering something else. `states_claim()` runs
  the other way round from inference: given an emitted claim, it asks whether the
  answer quotes it, and with no claims there is nothing for it to match.

## What `analyst_qa` may not reach, and why

`analyst_qa` answers on the public request path, so it gets the aggregate
view of a ring rather than the row-level one:

- **No `get_ring_subgraph`.** It ran to ~113k prompt tokens on a live
  investigation. Replacing it with `get_ring_summary` dropped a live chat
  question from ~34,000 prompt tokens to ~5,300.
- **No direct `transfers` access.** Asked for a ring's total intra-ring
  amount, the model wrote a three-way join across `transfers` (101 MB,
  ~570k rows) and `ring_members` twice, and the request died on the
  90-second wall clock. `get_ring_transfer_summary` answers the same
  question exactly — counts, totals, cross-bank split, time span — off the
  adjacency index, in a few hundred bytes. Live, the same question then
  returned `3500.80 USD` in 26s.

Both are enforced in `agents/`, not by editing `shared/`:
`ALLOWED_SQL_TABLES` is frozen and still lists `transfers`, and the
investigator and report agents keep their access. The restricted `run_sql`
**raises**, so the refusal travels the normal tool-error path and reaches
the model as a structured error naming the aggregate tool to use instead —
and it keeps `__name__ == "run_sql"`, so provenance and the grounding
count still see it.

An earlier live answer to that question, summed by the model from a
1-hop subgraph, was `4392.34` — it had included boundary transfers. The
aggregate tool defines intra-ring as *both* endpoints in the ring and
reports `boundary_transfer_count` separately, so the excluded edges are
visible rather than silently folded in.

## The verification loop

`investigator → verify()`. On failure, **one** retry with the failed claims fed
back (`MAX_RETRIES` from `shared/contracts`). If it still fails, the narrative is
**withheld** and the raw tool data is returned with a warning. A plausible
paragraph the tools contradict is worse than a table of numbers, because it reads
as though someone checked it.

The report path applies the same principle to citations. Every ground must carry
a passage that `search_regulations` actually returned — checked with
`regulations.verify_citation()`, because a model asked to cite will produce
something that *looks* like a citation. If any ground is dropped, the narrative
is withheld whole: the prose is written in one pass and asserts every ground the
model produced, so it cannot be published once one of them is unsupported. The
surviving grounds and their citations are still returned.

## Layout

```
config.py      one env-driven config; MULENET_AGENT_MODE=mock|vertex
data.py        real ml/out + simulator/out tables, per-table mock fallback
tools.py       the nine tools
sql_guard.py   AST-based read-only SQL validation, then sandboxed execution
regulations.py regulation search; delegates to docai, falls back to a corpus
verify.py      claim types and the deterministic verifier
mock_agents.py the four agents, offline and deterministic
adk_agents.py  the four agents, Gemini on Vertex
loop.py        the verification loop and handle()
cases.py       the cases producer: one case per ring, verified only
service.py     the HTTP entrypoint; contract in docs/agents_api.md
limits.py      per-session, per-IP and per-day cost ceilings
prompts/       router.md, analyst_qa.md, investigator.md, report.md
eval/          25 questions plus a scorer for accuracy and verifier pass rate
out/           cases.parquet + manifest.json
```

## The HTTP service

`docs/agents_api.md` is the frozen contract; `service.py` implements it and
does not restate it. Three paths: `GET /health`, `POST /chat` (router then
`analyst_qa`, on live Gemini) and `GET /cases/{ring_id}` (the published
parquet, no model call).

```
uvicorn agents.service:app --host 0.0.0.0 --port ${PORT:-8080}
```

Two things about it are easy to undo by accident and are enforced in code
rather than left to a comment:

**There is no mock path in the service.** `backend` is the literal
`"gemini"`, so returning it for the offline keyword matcher would make the
field a lie. `/chat` calls Vertex whatever `MULENET_AGENT_MODE` says, and an
unreachable Vertex is a `503` rather than a quiet downgrade. The mock path
still exists for the eval harness; it just cannot be served.

**The ceilings are not optional.** Cloud Run auth bounds who may call;
the caller is a public page, so the limits are what bound the bill. Every
one is an env var with a safe default — see `config.ServiceLimits`.

**An answer with zero executed tool calls is not returned.** `409
no_tool_evidence`, decided on the count of tool executions observed at the
wrapper — never on the model's account of what it used. A small factual
question is exactly where a model bluffs convincingly, and publishing that
under `backend: "gemini"` would give a guess the authority of a query.

**It refuses to serve without the published tables.** `agents/data.py`
degrades per table to a built-in mock when a parquet file is absent, which
is right locally and catastrophic deployed: the result is Gemini answering
correctly over invented numbers, labelled `backend: "gemini"` and carrying
real-looking `sources`. A wrong number with authentic provenance is worse
than an outage, because an outage is obvious. So the lifespan hook raises
and the process does not serve, and `/health` returns `503
data_unavailable` rather than `ok`. The check is in the lifespan, not in
`create_app`, because no parquet is committed — failing at import would
take out the offline suite on every fresh clone.

The check is **per table**, matching the fallback, and the table list is
derived in `data.required_tables()` from `shared/ALLOWED_SQL_TABLES` plus
what the typed tools open. A directory-level check was the earlier
version and it was the wrong granularity: a partial fetch left `ml/out`
non-empty, so the instance reported itself healthy and served mocks for
whatever had not arrived. On the Vertex path the fallback is also gone at
the point of the read — `data.MissingTableError` — so no store built
after startup can reopen the hole. The mock stays for `mock` mode, which
is local work and the eval harness.

**Retries: one per failure mode, at most three attempts.** The two modes
are an *empty* completion and an *unreadable* one (text, but not the JSON
the prompt requires). Both are non-deterministic on identical input —
measured live, one compound question produced each of them and a clean
answer across five attempts — so a retry earns its call. Exhausted, they
end differently: empty is `409 no_tool_evidence`, unreadable is `502
malformed_response`. Neither is `500`, because a caller has to tell "the
model gave us nothing usable" from "we crashed". The ledger lives in
`adk_agents`, not in `service.py`, so the in-process callers get it too —
that gap is what killed a 25-question eval run on its first question.

### Why provenance is a closure and not a ContextVar

The first version collected `sources` in a `ContextVar` and returned `[]`
on **every** live request while the whole offline suite passed. ADK's
synchronous `Runner.run` executes the agent on a bare `threading.Thread`
(`create_thread`, then `asyncio.run`), and a plain thread starts with an
empty context — so every tool call wrote into a collector the request
could not see. The offline tests called the tools inline, on the same
thread, and so never reproduced it.

Two things came out of that and should not be undone:

- The sink is **bound into each tool wrapper at wrap time**. A closure
  over a set does not care which thread dereferences it.
- `adk_agents.TOOL_CALLS` is a **second, independent observer**, written
  by a different mechanism. A collector cannot be its own witness: a
  collector that records nothing and a run that called nothing produce
  the same empty list. `test_sources_survive_tools_running_on_a_bare_thread`
  runs the tools on a bare thread on purpose.

Not ADK's `get_fast_api_app`: these agents are built programmatically with
no `root_agent` module for its `agents_dir` loader to discover, and it
serves ADK's own `/run` protocol rather than the three paths the contract
fixes. Checked against <https://adk.dev/deploy/cloud-run/> on 2026-09-26
against google-adk 2.9.2. Note `google-adk` is **unpinned** in
`pyproject.toml`; 2.9.2 is what is installed, not what is required.

## Running it

```
MULENET_AGENT_MODE=mock python -m agents.eval.run_eval    # offline, no cloud
python -m pytest agents/ -q
python -m ruff check agents/
```

Mock mode's language step is a keyword matcher, not a model. Its eval score is a
regression guard on the tool, data and verification path — not a Gemini score.

**The eval measures accuracy and verification separately**, and does not mix
them. Accuracy grades the answer; `chat_verification` counts the four outcomes
above across the 25 questions. A question can be answered correctly and carry no
claim, and a question can verify while the number is the wrong one for what was
asked, so collapsing the two would make a before/after comparison meaningless.
The one combination that is always a defect gets named rather than averaged:
`withheld_but_correct` lists any question graded correct whose answer was
withheld. That means the claim emission or the verifier is wrong — never the
answer — and it is an invitation to investigate, not to relax the check.

**The eval runs in-process**, through `agents/loop.handle`, not over HTTP. So
the rate limits, the daily cap and the `no_tool_evidence` grounding refusal do
**not** apply to it: 25 eval questions are not evidence that any of those work.
They are covered by `agents/test_service.py` instead.

**Every eval run writes its evidence as it goes**, to
`agents/out/eval/<model>__<location>__<stamp>.jsonl` — one JSON line per
question and per investigation, flushed and `fsync`ed immediately. Each question
record carries its `verification`, `claim_count` and any `unsupported` messages,
so the interesting question — the correct answer that was withheld — can be
found in the records rather than inferred from a total. A run
killed at question 20 leaves 19 questions of evidence behind.

That is not tidiness. Live model calls cost money, and two runs were lost
mid-measurement — one to an account rate limit, one to a crash — each time
taking the paid-for numbers with them. A measurement that cannot survive its
own process is one you have to buy twice. Each record carries the model id
**and the region**, because the same id behaves differently by region: 
`gemini-3.5-flash` serves in `asia-south1` and 404s in `us-central1`, so a
result without its region cannot be compared to anything.

`agents/eval/record.py` also reads runs back, tolerating the torn final line a
killed process leaves.

## The model id

`config.DEFAULT_MODEL` is the only definition, and
`test_no_source_file_hardcodes_a_model_id_besides_config` keeps it that way —
a second literal is a second thing to forget at a retirement date.
`MULENET_GEMINI_MODEL` overrides it.

`gemini-2.5-flash` retires 16–20 October 2026. **The Gemini 3 line is not
served from `us-central1`**: probed live against project `mulenet`, every
`gemini-3.x` id returns `404 ... not found or your project does not have
access ... ensure the model is available in the specified region`, while the
same ids all answer on the `global` endpoint. So the successor is not a
one-variable change — it moves the region too, which is a data-residency
decision for a BFSI entry, not an implementation detail. See the handover
notes; unresolved at time of writing.
