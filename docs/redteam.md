# Red-teaming the detector with Gemini

**Code:** `ml/redteam.py`, `agents/prompts/redteam.md`, `ml/test_redteam.py`

Gemini is told MuleNet's own detection thresholds and asked to design
transfer patterns that move the same money while staying outside them. The
patterns it returns are then **executed** against the real detector, and the
catch-or-miss verdict is reported as it comes out.

```
MULENET_AGENT_MODE=vertex python -m ml.redteam
```

No simulator or `ml/out/` artefact is needed — the graph is built in memory.
Vertex credentials are needed for the model call; `ml/test_redteam.py` runs
the whole thing offline with the model replaced.

## What this proves that nothing else in the repo does

Every other number here measures the detector against the attack patterns
`simulator/generator.py` was written to produce. Those patterns and the rule
that catches them share an author, which is the oldest problem in detection
work: you cannot tell a robust rule from a generator that only proposes
attacks the rule already stops.

Gemini is not a better adversary than a real launderer. It is an
**independent** one. It produces patterns nobody on this project chose, and
it does so without ever seeing the detection tables — the `redteam` agent is
registered with **no tools** (`agents/adk_agents.AGENT_TOOLS`), because a
strategy fitted to the 24 rings already in `ml/out/` would evade *this
dataset* rather than *this rule*.

**A miss is a valid result, and the more useful one.** A harness that reports
a clean sweep has either found a robust rule or a toothless adversary, and the
two look identical from outside.

## What it actually runs

The model returns, per strategy, a name, a description, the clause it is
aiming at, and a ten-field `pattern` block: how many collectors, fan-in per
collector, fan-out, hop delay, amount, retained fraction, bank spread, and
whether devices or KYC ids are reused. `ml/redteam.py` turns those numbers
into accounts and transfers and runs them through **the real code**, in the
order `ml/publish.py` runs it:

| stage | function |
| --- | --- |
| features | `ml.features.build_features` |
| stage-1 decision | `ml.rule.velocity_decision` |
| ring clustering | `ml.publish.high_risk_components` |
| stage-2 propagation | `ml.stage2.propagate` |

Nothing is reimplemented, and the five thresholds are **imported** from
`ml.rule`, never retyped, so retuning the rule moves this harness with it.
`ml/test_redteam.py` asserts that: drift a threshold and the test fails.

Every synthetic id is an HMAC-SHA256 digest of a loop counter, 64 hex
characters, shaped exactly like a real hashed account id so nothing
downstream can special-case it. The key is a literal in `ml/redteam.py` and
is deliberately **not** read from the environment: nobody can point this
harness at a real hashing key and have it emit ids that collide with
production ones.

## The three verdicts

| verdict | meaning |
| --- | --- |
| `velocity_ring` | The rule fired on a collector **and** a ring formed. |
| `velocity_alert_no_ring` | The rule fired, but nothing the collector transacts with was flagged too, so clustering had nothing to join it to. A real account-level alert; the ring view does not show it. |
| `missed` | No clause fired. |

The first two are kept apart because MuleNet keeps them apart — *"a component
of one is an alert, not a ring"* (`ml.publish.high_risk_components`).
Collapsing them would either overstate the detector (calling an alert a ring)
or understate it (calling a fired rule a miss). Both were written while
building this, and both read wrong.

There is deliberately **no** "caught by stage 2" verdict. Stage 2 cannot
create a ring — that is the safety property `ml/stage2.py` is built around —
so on a pattern that clears the velocity rule, propagation has nothing to
propagate out of. Stage 2 can only widen a catch stage 1 already made, and
the `stage2_alertable` / `stage2_context` counts report that widening.
`stage2_context` is counted and deliberately excluded from `caught`: a
same-bank identity link is downgraded to context by the ruling, and counting
it as a catch would claim credit for an accusation MuleNet refuses to make.

That property is now measurable from the outside, which is a side benefit of
this harness. Two strategies sharing one device fingerprint, identical but
for the bank spread: the cross-bank one forms a ring and stage 2 pulls in 42
alertable accounts; the single-bank one forms no ring and stage 2 adds
**nothing**, despite the same shared credential. Identity reuse alone
originates no accusation, demonstrated rather than asserted.

## The honest limitation: each strategy runs as an isolated graph

The only transfers in the feature table are the ones the strategy itself
creates. **This cuts both ways, and the two directions are not symmetric.**

**`caught` is solid evidence.** `in_degree`, `out_degree` and
`distinct_counterparty_banks` are lower-bound clauses over monotone counts.
Embedding the same pattern in a bank's real traffic can only raise those
counts, so a pattern caught in isolation is caught at least as easily in
company.

**`missed` does not transfer.** It means the pattern clears the thresholds
*on its own transactions*. A real collector also carries ordinary traffic
that feeds the same degree counts — a mule holding fan-in at 15 to duck a
floor of 16 is one legitimate salary credit away from firing. Background
traffic also shifts `passthrough_speed_seconds` (a median over bursts) and
`passthrough_ratio` in directions this harness cannot predict.

So the honest reading of a miss is **"a hole in the rule as written, worth a
look"** — never "this evades MuleNet". Closing that gap means re-running the
surviving strategies with the pattern injected into `simulator/out/` instead
of standing alone. That needs the simulator tables and is not what this
module does.

Two further limits, stated rather than discovered later:

- **The XGBoost scorer is not in this loop.** Rings come from the velocity
  rule, so the rule is what is attacked. An account this harness reports as
  missed may still score high in the alert queue.
- **The synthetic accounts have no credential history** beyond what
  `reuse_device` / `reuse_kyc_id` create, so the stage-2 result is about
  propagation mechanics, not about how dense legitimate credential sharing
  is. In the real data 4,102 of the 4,773 accounts carrying an identity edge
  are not complicit (`ml/stage2.py`); nothing here reproduces that.

## Reading a run

The JSON written to `ml/experiments/redteam/metrics.json` carries the
thresholds it graded against and both caveats above, inline. That is on
purpose: a `missed` row lifted out of the file and pasted into a slide is
exactly the misreading this page is written to prevent, so the qualifier
travels with the number.

Two things in the output are worth reading carefully.

**`clamped`.** The prompt states a range for every pattern field and
`ml/redteam.py` enforces it. A value outside its range is clamped *and
named*, because a verdict on a silently altered pattern is a verdict about
the wrong thing. A field the model simply omitted takes its default and is
**not** listed as clamped — declining to choose is not a decision the harness
overruled.

**"this miss is not the designed one".** Each strategy names the clause it is
aiming at, and the reasoning line names the clause that actually failed. When
they differ, the line says so. A strategy that evades by accident is a weaker
result than one that evades as designed, and the difference is invisible
unless it is written down. The model's claim is never corrected or filtered
either — a strategy that names the fan-in floor and then picks a fan-in above
it is run anyway and reported as caught. Filtering those out would be grading
the model's homework with the answer key.

## The live version: a judge builds the pattern, not Gemini

**Code:** `agents/simulate.py`, `POST /simulate` and `GET /simulate/bounds` in
`agents/service.py`, the "4 · Build your own evasion attempt" panel in
`ui/`.

Everything above this section is about Gemini's strategies. This is the
same execution path — `ml.redteam._coerce_pattern` then
`ml.redteam.evaluate` — pointed at a pattern a person typed into a form
instead of one a model proposed. A judge fills in the same ten fields
`PATTERN_BOUNDS` bounds (collectors, mules per collector, onward accounts,
hop delay, amount, retained fraction, bank spread, device/KYC reuse),
presses Run, and an actual HTTP request computes the verdict **on that
request** — no caching, no precomputation, no canned response. The UI
shows a brief "computing…" state because there is a real, if sub-second,
round trip behind it.

**No model call anywhere in this path.** `_coerce_pattern` only clamps
numbers and `evaluate` is pure deterministic computation, which is what
makes this safe to expose without the cost ceilings `/chat` needs — see
`agents/limits.py`'s docstring on why those exist at all, and why they do
not apply here: they bound Gemini spend, and nothing here calls Gemini.
`PATTERN_BOUNDS` already keeps the synthetic graph tiny (a few hundred
rows at the extreme), which is the only guard this endpoint needs.

**Why it lives in `agents/`, not `ui/`.** The deployed `ui` container is
deliberately dependency-light — no pandas, no `ml/` (see
`infra/Dockerfile.ui`'s own comment) — so it has nothing to recompute
this with. `agents/` already has `pandas`/`pyarrow`; running a judge's
pattern through `ml.redteam.evaluate` additionally requires `scipy`,
`scikit-learn` and `xgboost`, because `ml.rule` imports `ml.evaluate` for
its threshold-tuning helpers (which imports `xgboost` at module level)
and `ml.metrics` (which imports `scikit-learn`) — **even though the
velocity rule itself never calls either.** That was checked rather than
assumed; see `infra/Dockerfile.agents`'s comment for the exact import
chain. `torch`/`torch_geometric` (`ml/gnn.py`) are confirmed **not** in
this chain and are not installed. Pillow is also deliberately absent:
`agents/simulate.py` inlines the two things `ml.redteam._generator()`
would otherwise reach for in `simulator.generator` (the bank ids and the
id-hashing convention) specifically so this live request path never
imports the module that pulls Pillow in to render KYC document images.
The `ui` → `agents` request goes over the same authenticated
service-to-service path chat and screen already use (`ui/chat.py`'s
`AgentsClient`), degrading visibly — never silently — when `agents/` is
unreachable, exactly like those two.

**What this proves that a Gemini strategy does not.** A judge picks the
numbers with full knowledge of the thresholds, which Gemini also has, so
this is not a stronger adversary in that sense. What it adds is
*interactivity*: a skeptical judge can test their own hypothesis about
the rule directly — "what if I keep fan-in at exactly the floor" — and
watch the real detector answer in front of them, rather than reading a
verdict somebody else generated earlier.

**What it does not prove — same caveat as the section above, restated
because it is easy to miss on a live demo.** Every pattern here still
runs as an **isolated synthetic graph**. A CAUGHT verdict holds at least
as strongly with real background traffic (the degree clauses are lower
bounds). A MISSED verdict is **not** evidence the pattern survives in
production: it means the pattern clears the thresholds on its own
transactions, and a real collector's ordinary traffic would raise the
same counts. The UI states this next to every result; do not quote a
live MISSED as "this evades MuleNet" any more than a Gemini one.

## A second live panel: load a real subset, score it live

**Code:** `agents/subset.py`, `GET /subset/options`, `POST /subset/load` and
`POST /subset/score` in `agents/service.py`, the "5 · Load a subset, score it
live" panel in `ui/`.

Everything above this section runs a pattern — Gemini-designed or judge-built
— as an **isolated synthetic graph**. This panel is a different kind of
live: it reads a bounded slice of the **actual** 572,890-transfer,
30,000-account dataset the rest of the app is built from, and runs it
through the same unmodified `ml.features.build_features` and
`ml.rule.velocity_decision` the metrics tab's precomputed grid was built
from — live, per request, no caching of the result across different
choices.

**What "subset" means, and why.** `ml/early.py` is this repo's headline
result: scoring every account from only what its bank could see by a
cutoff instant, swept across a grid of horizons. That grid is precomputed,
behind `GET /api/metrics`. This panel exposes the same underlying idea — a
time cutoff that bounds what the feature builder may see — as something a
caller picks and gets scored on the spot, from a small fixed menu (`30m`,
`2h`, `6h`, `24h`, `3d`, `7d`) rather than reading a number someone else
already computed. It is a deliberately simpler cut of that idea, not a
restatement of it: `ml.early` cuts each account off relative to *its own
ring's onset* (or a resampled reference time for a non-ring account), which
needs `ground_truth` — the very labels a live detector would not have. This
panel instead uses one **absolute** calendar cutoff (the same instant for
every account), which reads no label at all — the flags it reports come from
exactly the code path a live system would run, unlabelled data in, a
decision out. A caller may also narrow the *display* to one bank; that
filter is applied only after scoring, never before, because narrowing the
account table itself would silently undercount `cb_distinct_counterparty_banks`
for exactly the accounts a cross-bank rule cares about most — see
`agents/subset.py`'s module docstring.

**Why the horizon menu stops at 7 days.** Measured, not guessed: a
full-window feature build over the entire 572,890-transfer graph took
~8.5s on this machine, well outside "a few seconds". Every entry in the
fixed menu (`agents.subset.HORIZONS`) was measured directly and tops out
around 1.7s at `7d` (118,406 transfers). The full 30-day window is already
shown, precomputed, on the metrics tab; this panel's job is the live,
fast end that tab does not cover. The menu being fixed and small is also
the entire abuse guard — the same shape `PATTERN_BOUNDS` gives `/simulate`
— so no second rate limiter earns its keep here either.

**Two real HTTP round trips, not one.** The brief asked for "loading" to be
a visible step, separate from "scoring" — not two labels painted on one
spinner. `ml.features.build_features` *is* the load step; the vectorised
comparisons in `ml.rule.velocity_decision` *are* the score step and are
effectively free next to it. So `POST /subset/load` runs only the first and
`POST /subset/score` runs both, meaning a full click-through (load, then
score) builds features twice. That duplication is a deliberate trade for
genuine two-step visibility with no server-side session state holding a
half-finished computation between two requests.

**What this proves that the metrics tab does not.** The metrics tab's
grid is real, but it is one thing, computed once, read many times. This
panel proves the same detection code runs correctly, live, against a
caller's own choice of slice — the "loading data as its own moment,
separate from training" the panel exists to make visible, distinct from
the ~13-minute full model retrain (out of scope; see the README's
Reproduce section) and from `/simulate`'s synthetic graphs.

**What this does not prove.** A `cross_bank_flagged` count at a short
horizon reads low or zero for an honest reason, not a bug: most rings have
not yet formed their fan-in pattern that early (`ml.early`'s own onset
distribution is why its grid runs to 180 minutes). A low count at `30m` is
not a claim the detector missed anything — it is the same claim
`ml.early.CUTOFF_MINUTES`'s own early rows make. And the bank filter is
cosmetic, as above: it never changes what was computed, only what is shown.

## Status

The detection half is exercised end to end by `ml/test_redteam.py` (35 tests,
offline, no artefacts). The Gemini half is exercised with
`agents.adk_agents._run_agent` replaced, matching `agents/test_service.py` —
so the prompt registration, the JSON parse, the shared retry ledger and the
strategy typing all run through their real code with no network and no bill.

As with the rest of `agents/`, **no call in the Vertex path has run against a
live model** from this environment: there are no credentials here. No
`metrics.json` is committed for that reason — a file produced from a
hand-written reply would present authored strategies as model output. Run it
with credentials and the numbers are yours.
