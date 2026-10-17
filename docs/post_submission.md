# Post-submission backlog

`shared/` contracts are **frozen** until after submission. Change them only for a
bug that breaks the demo or a test. Everything else lands here.

Each item records what was noticed, why it matters, and who owns it, so the
reasoning does not have to be rediscovered later.

## Contract changes deferred

### 1. Nothing forces a tool to call the rounding validator
`assert_score_fields_rounded()` is opt-in. A new tool returning a `RiskScore`
without calling `round_score_fields()` breaks the precision promise silently.
A validator on `RiskScore` itself would be wrong — stored rows are legitimately
unrounded — so the enforcement point is the tool wrapper in `agents/`, e.g. a
decorator. Needs a ruling on where enforcement belongs.
Owner: architect + agents-engineer.

### 2. `MAX_TRACE_PATHS = 50` is still agents-local
`trace_money_flow` silently returns at most 50 paths. Same category as
`SQL_ROW_LIMIT`, `MAX_SUBGRAPH_ACCOUNTS`, `MAX_SUBGRAPH_TRANSFERS`, which all
live in `shared/tool_interfaces.py`: a cap a caller cannot read the answer
correctly without knowing about.
Owner: architect.

### 3. No `Features` pydantic model
`passthrough_ratio`, `round_amount_ratio`, `passthrough_speed_seconds` and
`account_age_days` exist only as `FEATURES_SCHEMA` columns, so nothing
type-checks a feature row. If one is added, the rounding decision must be made
field by field — and for model *inputs* the answer is almost certainly "never
round", since rounding a feature changes the score it produces.
Owner: architect + ml-engineer.

### 4. `MoneyAmount` has no mechanical guard beyond `Transfer.amount`
`amount_in_total` / `amount_out_total` are `NUMERIC` in `FEATURES_SCHEMA` with
no model and no quantization promise. Same silent-corruption risk that
`MoneyAmount` exists to prevent.
Owner: architect.

### 5. `Ring` carries no producer version
No `detector_version` / `stage2_version` on `Ring`, so a consumer cannot tell
which pipeline produced a ring or whether stage 2 was on. It is in
`manifest.json` only, which does not survive a BigQuery read. Purely additive.
Owner: architect.

### 6. `RingExplanation` does not distinguish the two populations
`top_factors` and `narrative` describe a single undifferentiated ring. Now that
membership is mixed (velocity vs identity_link, alertable vs context), a ring
explanation that does not say "N detected, M linked for review" understates the
qualification.
Owner: architect + agents-engineer.

### 7. `GnnScore` may be dead weight
The GNN was evaluated and not shipped (no real gain, and it declared a
transductive advantage). `SCORES_SCHEMA.model_type` still documents
`"xgboost or gnn"` with a nullable `embedding_ref`. If the GNN is permanently
abandoned, `scores` could be flattened.
Owner: architect + ml-engineer.

### 8. `str_reports` allowlist, revisit when human sign-off exists
Deliberately excluded from `ALLOWED_SQL_TABLES`: an agent that can `SELECT`
prior reports could cite an unverified draft as corroboration, and the second
report never touches the regulation corpus, so `citations_verified` is bypassed
rather than propagated. Revisit only when reports carry human sign-off, and then
admit `status = 'filed'` rows only.
Owner: architect.

### 9. `ReusePair` promotion to `shared/`
Defined locally in `docai/reuse.py`. It is the document-level evidence beneath
`AccountEvidence` — an analyst needs to see *which two documents* matched.
15 fields incl. `bank_pair`, `cross_bank`, `match_type`, `similarity`, `backend`.
Owner: architect + docai-engineer.

### 10. `search_regulations` return-key widening
The stub documents `citation` / `excerpt` / `source_uri`; `docai/` also returns
`doc_name` and `section`. Confirm the widening is acceptable or fold them into
`citation`.
Owner: architect + docai-engineer.

## Non-contract items

### 11. `ui/build_snapshot.py:531` rounds score to 6 decimals
A different path from the agent tool boundary (ml parquet -> UI snapshot), so it
is not governed by `SCORE_DECIMALS`. But it is an independent, undocumented
rounding of the same field to a different precision: if a UI-displayed score is
ever compared against an agent-quoted one they disagree in the fourth decimal.
Needs a decision on whether the two are ever shown side by side.
Owner: frontend-engineer.

### 12. `agents/out/` has no `manifest.json` convention
`ml/out/` has one. Now that `agents/` publishes `cases.parquet` (and
`str_reports.parquet` per spec), the two folders' publishing conventions should
match.
Owner: agents-engineer + platform-engineer.

### 13. `run_eval.py` string-compare path assumes `.3f`
`_normalise` compares formatted numbers as text before the numeric fallback. A
score formatted to a different number of places than `SCORE_DECIMALS` fails the
string compare and then rides on the tolerance path. The scorer should format
with `SCORE_DECIMALS`.
Owner: agents-engineer.

### 14. `google-cloud-discoveryengine` missing from `pyproject.toml`
Vertex AI Search cannot run without it. The cloud path raises an actionable
`RuntimeError` rather than degrading silently, so this is not urgent while the
regulation corpus is placeholders.
Owner: platform-engineer.

### 15. Ring recall is capped by clustering, not detection
87.8% account-level recall collapses to 44% ring-level recall, because a flagged
mule only joins a ring when its counterparty was also flagged. The fix is not a
threshold: it is clustering over *candidate* edges rather than requiring both
endpoints flagged. A design change needing agreement before it is attempted.
Owner: ml-engineer.

### 16. `fast_passthrough_only` sources are unreachable by stage 2
0 of 41 recovered, and it is a structural ceiling rather than a tuning failure —
stage 2 can only follow an identity edge, and a ring that reuses no credentials
leaks none. Reaching those needs a third signal (funding-side structure) that
does not exist yet.
Owner: ml-engineer.

## Known artefacts of the synthetic data

Not bugs; recorded so nobody "fixes" them or quotes them as production numbers.

- **`cross_bank_reused_photo` and `cross_bank_reused_id_number` are Jaccard
  1.000** — the generator attaches a reused `photo_embedding_ref` to precisely
  the documents that reuse `hashed_id_number`. An STR must cite identity reuse
  **once**; the UI must not render them as two independent corroborations. The
  same now holds for the `same_bank_` pair.
- **XGBoost scores are bimodal** (p97.5 = 0.0021, p98 = 0.9999), so every
  threshold between 0.003 and 0.999 returns the same 712 accounts. This is why
  rings are cut by the velocity rule and not by a score threshold: a score-based
  ring table would reconstruct ground truth exactly and make the demo a
  tautology.
- **Stage-2 addition precision of 87.4% is partly a generator artefact** — the
  simulator wires ring members' devices and documents tightly together, so a
  mule's identity neighbourhood is unusually pure. Real households are messier.
  Not a production estimate.
- **Mock-mode eval scores (25/25, 20/20) are a regression guard, not a model
  score.** Mock mode's language step is a keyword matcher, not Gemini.
- **The offline dHash is a layout hash** and the synthetic documents share
  layouts: one pair of unrelated address proofs collides, giving 2 false
  cross-bank photo pairs out of 1,792 (0.11%). Real Vertex embeddings are the
  intended fix.

## `ui/` — contract and data requests (frontend-engineer)

None of these are blocking. The UI works against the frozen contracts as
they stand; each is a place where the frontend has to compute or caveat
something a producer could state.

### Contract requests (post-submission, not for `shared/` now)

- **No single-bank score column.** `ml/out/scores.parquet` carries only
  `mulenet-xgboost-cb-oof-v1` — the *cross-bank* model. The bank tiles are
  the "before" half of the demo's argument, but the only score available to
  rank a bank's own queue is the cross-bank one restricted to that bank's
  accounts. That is not what a bank would see alone, so each tile carries an
  explicit `queue_caveat` saying so and points at the metrics page for the
  real comparison. A published `score_single_bank` column (or a second
  `scores` table keyed by scope) would let the tiles make the fragmentation
  argument with a number instead of a disclaimer.
- **`AccountEvidence` has no ring scope.** Evidence rows are per-account, so
  to caption a ring finding the UI groups rows by `(kind, scope)` and picks
  one row's `detail` as an example. A ring-level evidence view, or a
  `ring_id` on the row, would remove that aggregation from the presentation
  layer — where an error becomes a mislabelled accusation.
- **`RingMember.reason` is absent for `velocity` members** (optional by
  contract, and `ml/` leaves it unset). The UI therefore falls back to the
  account's `top_factors` for a detected member and shows `reason` only for
  identity links. Consistent per-member reasons would let one rendering path
  serve both.
- **`Investigation` has no link to the narrative it verified.** The case row
  proves *that* four claims were re-derived, but not *which*. Surfacing the
  verified claim texts would let the case panel show what was checked rather
  than only the count.

### Observed data discrepancy (worth resolving before the demo)

- The brief circulated to `ui/` quoted **T=5m cross-bank PR-AUC 0.643 vs
  single-bank 0.382**. No file under `ml/experiments/` contains that pair.
  `early_detection/metrics.json` gives **0.6220 / 0.4282**, and
  `gnn/metrics.json` (GraphSAGE) gives **0.6366 / 0.4392**.
- `ui/` reads `early_detection`, which is the right source: its span
  gradient is **+0.009 / +0.174 / +0.328** and its burden pair is
  **6.23 → 1.97**, both matching the brief exactly. The `gnn` run does not —
  its span-1 delta is **−0.086**, which would invert the dose-response claim
  that a one-bank ring has nearly nothing to gain.
- Nothing is hardcoded: `/api/metrics` transcribes whatever
  `MULENET_UI_METRICS` points at, and a test asserts the served values equal
  that file's values rather than pinning literals. Someone should confirm
  which run is canonical for the submission.

## An LLM judge as a second verification pass (agents/)

The verifier `LlmAgent` was removed before submission: it was built, never
invoked, and its core job — re-deriving claim values — is strictly better done
deterministically in `agents/verify.py`. `_run_agent` was only ever called for
the router, analyst_qa, investigator and report agents.

There is, however, one thing the deterministic verifier genuinely cannot do, and
it is worth building later: judging whether a narrative is **misleading** without
tripping any keyword rule. Two examples the current checks pass:

- "These accounts behaved exactly as a criminal network would." No banned word,
  no claim to re-derive, and it implies a conclusion the data does not support.
- An identity-link sentence phrased accusatorially while avoiding every term in
  the accusation list.

A second pass would need: the surviving narrative plus the claim set as input, a
rubric rather than free judgement, its verdict treated as advisory (it must not
be able to *pass* something the deterministic gate rejected, only flag
something the gate allowed), and a measured false-positive rate before it gates
anything. Until that exists, the honest description is four agents and a
deterministic verifier.

## Two `ml/out/` tables are published uncontracted (ml/)

`ml/out/ring_detection_timeline.parquet` (one row per ring x horizon x scope,
from `ml/timeline.py`) and `ml/out/ring_id_map.parquet` (detector ring id ->
ground-truth ring id, from `ml/ring_map.py`) are the only tables in `ml/out/`
with `validated_against: null` in the manifest. Every other table is built by
importing its `shared/contracts.py` model, so a contract change breaks the
writer loudly; this one has no model to import because `shared/` was frozen
when it was added, and redefining a schema locally is the thing the house rules
forbid. They should get `RingDetectionTimeline` and `RingIdMap` contracts after
submission, at which point `ml/timeline.py` and `ml/ring_map.py` should build
their rows through them like `ml/publish.py` does.

`RingIdMap` is the more urgent of the two: the UI renders `true_ring_id` to a
judge as "the ring being replayed", and the pairing is best-member-overlap,
which is a heuristic. A contract is the natural place to make
`match_is_ambiguous` impossible to ignore — 7 of 45 detected rings span more
than one true ring, so the flag is not an edge case.

Two smaller things found while adding it:

- `.gitignore`'s `*.parquet` is unanchored, so it also catches
  `ml/experiments/early_detection/oof/*.parquet` — the persisted out-of-fold
  scores the timeline is derived from. Those are *derived model outputs over
  synthetic data*, not transaction data, and they are what makes the published
  per-ring numbers reproducible without a refit. They are currently
  untrackable. Worth an explicit negation rather than loosening the blanket
  rule, which exists to keep real data out of a public repo.
- `ml/out/manifest.json` is written wholesale by `ml/publish.py` and
  read-modify-written by `ml/timeline.py`, so `ml.timeline` must run after
  `ml.publish` or its entry is dropped. `ml/publish_all.py` now fixes the
  sequence and is the only supported way to regenerate `ml/out/`, but two
  writers for one file remains the underlying defect: the manifest should be
  assembled by one module that both steps feed their entries to.

  `ml.publish_all` has since been run end to end against real data on a
  scratch `MULENET_ML_DIR`, and all six tables were compared value-by-value
  against the published ones: identical everywhere except the two
  `generated_at` stamps, with `score` and all 120,000 SHAP contributions
  bit-identical (max |delta| exactly 0). So the pipeline is reproducible and
  the ordering fix is sound.

  What remains: `ml.publish` stamps `generated_at` from `datetime.now(UTC)`
  into `account_explanations` and `ring_explanations`, which is the only
  reason a re-run cannot be compared byte-for-byte and the only reason
  regenerating in place is risky at all. Those two should derive from the
  data, as `detected_at` already does.

## `shared/tool_interfaces.py` needs a signature for `get_ring_transfer_summary`

Logged by the **agents-engineer**; `shared/` was frozen, so the tool was
added in `agents/tools.py` alone.

`agents/tools.get_ring_transfer_summary(ring_id)` returns aggregates for the
money moving inside one ring — transfer count, exact total, cross-bank split,
distinct member pairs, boundary count and time span. It is registered in
`tools.ALL_TOOLS` but has no declaration in `shared/tool_interfaces.py`,
which is where every other tool's signature lives.

Why it exists: asked for a ring's total intra-ring amount, the model wrote a
three-way join across `transfers` (101 MB, ~570k rows) and `ring_members`
twice, and the request died on the 90-second wall clock. The aggregate
answers the same question exactly, off the adjacency index already built for
the other tools, in a few hundred bytes. Live, the question then returned
`3500.80 USD` in 26 seconds.

It reads only tables the contract already exposes, so nothing in `shared/`
has to change for it to be *correct* — but the signature belongs there for
the same reason the other nine do.

Two related items, both currently enforced in `agents/` because `shared/` is
frozen:

- **`analyst_qa` is denied direct `transfers` access**, via a wrapped
  `run_sql` in `agents/adk_agents.py`. `ALLOWED_SQL_TABLES` is untouched and
  still lists `transfers`; the investigator and report agents keep their
  access. If per-agent table policy is worth having as a first-class concept,
  it belongs beside the allowlist rather than in one agent's tool wiring.
- **`agents/prompts/analyst_qa.md` needs its JSON-envelope instruction
  hardened.** Measured on the 25-question eval against live Gemini 2.5 Flash:
  4 of 25 questions (16%) failed because the model answered in prose instead
  of the required JSON, twice in a row. Several were questions it *declined*
  correctly in substance. On the HTTP path those become `502
  malformed_response`. This is the single largest source of eval failure and
  is a prompt fix, not a code fix.

## Gemini reaches for `google:python_interpreter`, a tool it was never given

Logged by the **agents-engineer**, seen live during the 25-question eval on
`gemini-2.5-flash` @ us-central1, 2026-09-27. Not chased; recorded so the
next person does not rediscover it.

ADK logged:

```
Tool 'google:python_interpreter' not found.
Available tools: describe_tables, run_sql, get_account_profile,
                 get_ring_summary, get_ring_transfer_summary, get_explanation

Possible causes:
  1. LLM hallucinated the function name - review agent instruction clarity
  2. Tool not registered - verify agent.tools list
  3. Name mismatch - check for typos
```

It is not a typo or a missing registration: `google:python_interpreter` is a
Gemini **built-in** tool, not one of ours. The model decided the easiest way
to answer was to compute the answer itself rather than query the tables.

Not fatal, and the run continued — ADK returns the failure to the model along
with the real tool list, which is the same error-containment property
`agents/adk_agents._model_tool` provides for our own tools, so the model can
correct itself on the next turn. Two things worth noting anyway:

- It costs a whole turn against `max_llm_calls`, and a turn is the scarce
  resource on the request path.
- A model that would rather compute than query is exactly the failure the
  `no_tool_evidence` rule exists to catch. Here the guard holds — a
  hallucinated tool executes nothing, so it adds nothing to the provenance
  sink and cannot count as grounding. Worth re-checking if built-in tools are
  ever enabled deliberately, because a built-in that *does* execute would
  count as a tool call while grounding nothing in our data.

## `gemini-3.5-flash` does not converge inside `max_llm_calls=12`

Logged by the **agents-engineer**, 2026-09-27, from a live eval of
`gemini-3.5-flash` @ `asia-south1` (candidate A). Written down before the
fix because the run that produced it could not be cheaply repeated.

The eval died with:

```
google.adk.agents.invocation_context.LlmCallsLimitExceededError:
Max number of llm calls limit of `12` exceeded
```

raised from `run_accuracy` -> `handle(question.question, config)`
(`agents/eval/run_eval.py:137`), during the **accuracy** phase, not the
verifier phase. The same error appeared 3 times in the log.

**This was not rate limiting.** Retry counters for the run: `429/5xx
retries: 0`, `parse-mode failures: 0`. `asia-south1` did not throttle us
once across 25 questions. The single `429 RESOURCE_EXHAUSTED` seen earlier
on a chat question did not reproduce: a clean re-run of all four chat
questions returned 4/4 `200` with zero retries.

**It is a model-behaviour difference, and it exposed two defects of ours.**
`gemini-2.5-flash` and `gemini-3.8-flash` both answer these questions in
about **2** model calls. `gemini-3.5-flash` burned more than **12** on at
least one of them without converging.

1. `LlmCallsLimitExceededError` is handled nowhere. It is not a retryable
   status, not an `EmptyModelResponse`, and not a parse `ValueError`, so it
   propagates out of `_run_agent` and kills the caller — which for an
   unattended eval means losing the whole run.
2. Worse, over HTTP it would be **misclassified**. Its module is
   `google.adk.agents.invocation_context`, which starts with `google.`, so
   `agents/service._upstream()` returns true and the caller is told `503
   upstream_unavailable`. Exhausting *our own* deliberately-set call budget
   is not a Vertex outage, and reporting it as one sends whoever is on call
   to check Google's status page.

Both are fixed in `agents/`. Recorded here because the conclusion outlives
the fix: **a model that needs more turns than the budget allows is a
deployment fact about that model**, and raising `max_llm_calls` to
accommodate it would raise the worst-case bill for every other model too.


## `features` removed from the SQL allowlist

Logged by the **agents-engineer**, 2026-09-27, under an explicit freeze
exception for `shared/` as a correctness bug. One line of `shared/` changed:
`features` is gone from `shared.tool_interfaces.ALLOWED_SQL_TABLES`, with the
reasoning left in a comment where the name used to be.

**The defect.** `features` was allowlisted, so `run_sql` accepted
`SELECT ... FROM features` and `describe_tables` advertised it with the
contract's columns. Nothing publishes it. `agents/data.py` read
`ml/features.parquet` — a path nothing uploads into the image — and fell back
to a one-column empty frame, so the query succeeded and returned no rows. The
BigQuery `features` table is 0 rows for the same reason. **The deployed
service was in that state.** An empty result reads to an agent as "no such
account", not "no such table": the failure is silent in the one direction that
matters.

**Publish or remove.** Publishing was the option worth checking, and the shape
mismatch reported by the platform-engineer is real — verified here against the
file rather than taken on report:

| | rows | columns |
|---|---|---|
| `ml/features.parquet` | one per account | 34, `hashed_account_id`, `as_of`, then `sb_*` and `cb_*` pairs |
| `FEATURES_SCHEMA` (`shared/bq_schemas.py`) | one per (account, `as_of`, `scope`) | 15, with a `scope` column of `single_bank`/`cross_bank` |

13 of the contract's 15 columns are absent from the file. The ML frame is
**wide** — each feature twice, once per scope, in the column name — where the
contract is **long**, with the scope as a value. That is not a rename: it is an
unpivot, plus `document_reuse_count` to reconcile and the `sb_*`/`cb_*` extras
(`unique_in_counterparties`, `device_count`, `document_reuse_flag`) to drop.
`infra/load_bigquery.restate()` does `table.select(schema.names).cast(schema)`,
so adding this file to `SOURCES` as-is raises on the first absent column —
the breakage the platform-engineer predicted, confirmed by reading the code.

So removing it is the fix. An allowlisted table that cannot be read is a
promise the system does not keep, and the cheap way to keep the promise is to
stop making it. Writing the unpivot would mean `agents/` publishing a table
`ml/` owns, to a schema neither of them currently produces, to serve a query
nobody has asked.

**Blast radius, from a repo-wide grep.** Nothing breaks.

- `shared/bq_schemas.py` still holds `FEATURES_SCHEMA` and still lists
  `features` in `TABLE_SCHEMAS`, so `infra/load_bigquery.py` keeps creating the
  empty BigQuery table exactly as before — it is in `TABLE_SCHEMAS - SOURCES`,
  which is the "part of the contract, nothing publishes it yet" branch. The
  allowlist governs the agent's SQL tool, not the dataset. Untouched.
- `agents/data.py` no longer carries a `features` branch or `FEATURES_FILE`;
  `known_tables()` drops the name. Those existed only to serve this table.
- `agents/tools.describe_tables` and `agents/adk_agents`'s error payload both
  derive their table list from the allowlist, so the name disappears from what
  the model is told without either being edited.
- No other folder assumes it is queryable. `ml/features.py`,
  `ml/config.features_path` and the `ml/` tests are about the ML feature
  frame, which is unaffected — it is still built and still feeds
  `ml/baseline.py`.
- `docs/deployment.md` has a row reading `` `features` | 0 | created empty;
  nothing publishes it yet ``. Still true, and now worth one clause saying it
  is also no longer reachable from the agent's SQL tool. **For the
  platform-engineer** — `docs/` is not the agents lane.

Re-admit `features` when something publishes rows in the contract's shape.
