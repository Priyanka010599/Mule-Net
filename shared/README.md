# shared/

Contracts for MuleNet. These files are the single source of truth for
every schema in the project.

**The rule (CLAUDE.md):** code against `shared/` contracts; never
redefine schemas locally. If a contract needs to change, stop and report
what you need — the architect changes it here and lists every affected
folder.

**Hashed account ids:** only hashed account identifiers appear anywhere
in these contracts, and therefore anywhere in the system. No raw account
numbers, customer names, or raw document numbers are stored, published,
logged, or passed to a tool. KYC identity fields follow the same rule:
`hashed_id_number` is a hash, and `normalized_address` is a normalized
form kept for reuse matching. This is a public repo — no real data.

## Conventions

### The account identifier is always `hashed_account_id`

Every model, column, BigQuery field, and tool parameter that carries an
account identifier is named `hashed_account_id`. There is exactly one
spelling. Fields that hold a *direction* or a *collection* of account ids
keep `hashed_account_id` as their root:

| Where | Name |
| --- | --- |
| Any single account reference | `hashed_account_id` |
| `Transfer` sender / receiver | `from_hashed_account_id` / `to_hashed_account_id` |
| `Ring` members | `RingMember.hashed_account_id`, on `Ring.members` |

`Ring.member_hashed_account_ids` is **gone**; see "Ring membership is
two different claims" below. The root name is unchanged — it is now the
`hashed_account_id` of a `RingMember` record.

*Rationale.* The contracts previously used three spellings for the same
thing — `hashed_account_id` on `Account`, bare `account_id` on `Device`,
`KycDocument`, `KycSubmission`, `RiskScore`, `GnnScore` and
`GnnExplanation`, and `from_account` / `to_account` on `Transfer`. That
made joins hand-written per table, made it easy to write a `USING
(account_id)` join that silently missed, and — worse for a public BFSI
repo — the bare name `account_id` reads as if it could hold a real
account number. One name that says "hashed" at every boundary makes the
project's central privacy rule self-documenting and makes joins
mechanical. The directional prefix form was chosen over alternatives
like `sender_hashed_account_id` because `from_`/`to_` preserves the
original semantics of the columns being replaced and sorts adjacent in
column listings.

### Money is `Decimal`, never `float`

`contracts.py` exports `MoneyAmount`, an `Annotated[Decimal, ...]` type
used for every monetary field (today: `Transfer.amount`). It

* coerces input through `Decimal(str(value))` and quantizes to two
  decimal places with `ROUND_HALF_UP`, so a float never becomes a
  binary-rounded balance; and
* serializes to JSON as a **string with exactly two decimals**
  (`PlainSerializer(..., when_used="json")`), so Pub/Sub round-trips
  amounts losslessly — `decode_transfer(encode_transfer(t, b)).amount`
  is the same `Decimal` that went in.

It lines up with `NUMERIC(15,2)` in `ddl.sql` and `NUMERIC` in
`bq_schemas.py`.

### Datetimes are timezone-aware UTC

All `default_factory` datetimes use `datetime.now(timezone.utc)`, never
the deprecated `datetime.utcnow()`. `utcnow()` returns a *naive*
datetime, which mismatches the `TIMESTAMPTZ` columns in `ddl.sql` and
the `TIMESTAMP` fields in `bq_schemas.py`, and is deprecated on
Python 3.12+.

### Explanations are two tables, never one polymorphic table

A ring explanation and an account explanation live in
`ring_explanations` and `account_explanations`. There is no
`explanations` table and no `subject_type` discriminator.

*Rationale.* The two have different keys (`ring_id` against
`hashed_account_id`) and different payloads. Unioning them forces every
column `NULLABLE`, so the schema stops expressing which fields are
actually required, and makes every correct query carry a
`WHERE subject_type = ...` that is silently wrong — not an error, a
wrong number — when an analyst forgets it. Two narrow tables also match
what `ml/` already publishes (`ml/out/ring_explanations.parquet`) and
retire the `TABLE_ALIASES` bridge in `agents/data.py` that existed only
because the allowlist and the files disagreed. The cost is one extra
schema constant; the benefit is that a forgotten filter becomes an
unknown-column error instead of a mixed result set.

### AlloyDB is operational, BigQuery is model output

`features`, `scores`, `ring_explanations` and `account_explanations`
have BigQuery schemas in `bq_schemas.py` and **no table in `ddl.sql`**.
That asymmetry is a decision, not an oversight.

`ddl.sql` holds entities written by the transaction path or edited by a
human — `accounts`, `transfers`, `devices`, `kyc_submissions`,
`kyc_documents`, `rings`, `cases`, `str_reports`, `account_evidence`.
They are read row-by-row by a stable business key and updated in place.

The four BigQuery-only tables are model-run output: recomputed wholesale
on every `ml/` run, never updated in place, keyed by `(account, as_of)`
or by run rather than by a stable business key, and queried as scans and
aggregates. An AlloyDB copy would buy nothing and would invite an
`UPDATE` against a table whose only correct write is a full replace. If
an operational read of a model output is ever needed, the fix is a
materialized copy of the latest run, not moving the table.

### A draft STR can never be mistaken for a filing

`StrReport.citations_verified` defaults to `False` and is `True` only
when every citation resolved against a real indexed regulation document.
Every citation in the system today comes from
`docai/regulations/placeholder_corpus.json` — synthetic placeholders —
so every report is currently `False`.

The flag lives on the report rather than only in a UI banner because a
report gets exported, copied and re-rendered, and a warning that lives
outside the row does not survive that. It is enforced twice: a
`model_validator` on `StrReport` refuses `status="filed"` while
`citations_verified` is `False`, and a `CHECK (status <> 'filed' OR
citations_verified)` on the `str_reports` table refuses the same by
`UPDATE`. An unverified report is therefore *unfileable*, not merely
labelled.

For the related reason that `str_reports` is kept out of
`ALLOWED_SQL_TABLES`, see the `tool_interfaces.py` section.

### A case means a narrative that was checked

Nothing but a **verified** investigation may become a row in `cases`.
`agents/` drafts a narrative with structured claims, a deterministic
Python verifier re-derives every claim from tool data, and on failure
there is exactly one retry with the unsupported claims fed back. If that
also fails the narrative is **withheld** and only raw data plus a
warning is returned — in the live run, ring `00002`'s first
investigation was withheld exactly this way. A withheld investigation
produces **no case row at all**. There is deliberately no `withheld`
status: a case an analyst must re-check before trusting is not worth
having a table for, and an analyst opening `cases` may take it that
every row rests on claims that were re-derived and matched.

This is enforced by **shape**, not by a label — the difference from
`StrReport.citations_verified`. A draft STR legitimately exists
unverified, so it needs a flag that can be `False`. An unverified case
does not legitimately exist, so a `verified: bool` here would be a
constant `True`: a claim about the row rather than a constraint on it.
Instead `Investigation` carries `verified_at` (required, no default, so
a case cannot come into being by omission), `verification_attempts`
(1 or 2 — the investigator gets one draft and one retry, so any other
number describes a run that did not happen) and `verified_claim_count`
(at least 1). The last is the load-bearing one: `agents/loop.py` drops
the claim list along with the withheld narrative, so the withheld shape
cannot be written here without fabricating a number. A `model_validator`
and matching `CHECK` constraints on the `cases` table enforce all three,
the same validator-plus-`CHECK` pairing used for `citations_verified`.

**One case per ring.** `UNIQUE (ring_id)` on `cases`, and `case_id` is
derived from the ring via `contracts.case_id_for_ring` rather than
minted at random, so re-running the producer is idempotent. A ring
re-investigated and verified again **updates** that row — `verified_at`
advances, the attempt and claim counts are replaced — rather than
opening a second case. A second case per ring would make
`str_reports.case_id` ambiguous about which case a draft belongs to, and
"several goes at the same ring" already has a home: one case carries
many STR drafts. A case that has reached `closed` is a compliance record
and is left untouched by a later run; reopening it is a human action,
not an agent one.

### Score rounding and its comparison tolerance are one fact

`shared/tool_interfaces.py` defines `SCORE_DECIMALS = 3` and then
**derives** the comparison tolerance from it:

```python
SCORE_DECIMALS = 3
SCORE_TOLERANCE = 0.5 * 10**-SCORE_DECIMALS   # == 0.0005
```

`SCORE_DECIMALS` is the number of decimal places every score-like float
is rounded to **at the tool boundary**, before it reaches an agent. It
applies to exactly three fields: `RiskScore.score`,
`AccountExplanation.score`, and `ExplanationFactor.contribution`. It is
*not* a storage precision — `ml/` writes full-precision floats and both
`ddl.sql` and `bq_schemas.py` keep them as `FLOAT64`. It exists because
the raw float is what leaks: a live answer came back as
`0.99999809265136719` where `1.000` was expected and failed a `1e-6`
comparison by `1.9e-6`.

`SCORE_TOLERANCE` is half of one unit in the last retained place, which
is exactly the error the rounding can introduce: a value rounded to `d`
places sits at most `0.5 * 10**-d` from what it was rounded from. Two
numbers agreeing inside that bound are the same number at two
precisions; two that do not, differ for some other reason.

**Why derived and not a second literal.** Until now `agents/tools.py`
held `SCORE_DECIMALS = 3` and `agents/verify.py` held
`SCORE_TOLERANCE = 0.0005`, with nothing connecting them — two folders
agreeing on a number by coincidence. A frozen literal drifts wrong in
**either** direction as soon as the rounding moves:

* Round to fewer places (2) and genuine rounding differences reach
  `0.005`; a frozen `0.0005` is ten times too tight and rejects correct
  answers as discrepancies.
* Round to more places (4) and rounding differences never exceed
  `0.00005`; a frozen `0.0005` is ten times too loose and **accepts a
  real 0.0004 discrepancy as a rounding artefact** — the invisible
  failure, because everything still passes.

**No consumer may define its own tolerance for these values.** A module
that writes `0.0005`, `1e-6`, or any other literal has reintroduced the
bug this derivation exists to make impossible. Import the name. If a
tolerance is needed at some other precision, derive it in
`tool_interfaces.py` and name it there.

**The rounding preserves sign.** `contribution` is signed and the sign
is load-bearing — it is the whole difference between a mitigating factor
and an incriminating one. `round` never moves a value across zero, so a
negative contribution stays negative (a magnitude under half a unit in
the last place becomes `-0.0`, which still compares `<= 0`). Any
implementation must keep that; one that took an absolute value, clamped
at zero, or normalised `-0.0` to `0.0` would silently reclassify
factors. The one thing rounding does remove is direction at tiny
magnitudes: a contribution rounding to zero means "did not measurably
move the score", not a direction.

**The rounding is mechanical, not remembered.** `SCORE_DECIMALS` used to
be a promise `agents/tools.py` kept by convention: a new tool returning
a `RiskScore` without calling its private `_round_score` broke it
silently. `tool_interfaces.py` now carries two functions instead:

```python
round_score_fields(model)          # -> a copy with score fields rounded
assert_score_fields_rounded(model) # -> None, or raises ValueError
```

They are deliberately two, not one. The rounder makes a value correct;
the validator says whether it already was, so a tool that rounds by hand
can still fail loudly at the boundary rather than hand an agent a raw
float that some downstream comparison rejects later for reasons nobody
can trace — which is exactly the q16 failure.

**Fields are chosen by an explicit registry, not by name convention.**
`_ROUNDED_FIELDS` maps model class to field names:

| model | field |
| --- | --- |
| `RiskScore` | `score` |
| `GnnScore` | `score` |
| `AccountExplanation` | `score` |
| `ExplanationFactor` | `contribution` |

A suffix rule (`*_score`, `*_ratio`) would be automatic and would cover
a new field for free. It was rejected because the two failure modes are
not symmetric. A **missed** field is the bug that already exists, and
the validator now catches it loudly. A **wrong match** is silent and
permanent: a value that needed its precision is quietly degraded and
nothing downstream can tell. The obvious wrong match is money —
`Transfer.amount` is a `Decimal` quantized to 2dp and specified
`NUMERIC(15, 2)` — and "the amount is off by a tenth of a cent" is not a
bug anyone finds by reading a narrative. `Transfer` is therefore simply
not a key in the registry, and `round_score_fields` additionally raises
`TypeError` on any registered field that is not a `float`, so
registering money by mistake is an error rather than a rounding.

`GnnScore.score` is registered although no tool returns it today: it is
the same quantity as `RiskScore.score` under a different model family,
lands in the same `scores.score` `FLOAT64` column with
`model_type = 'gnn'`, and is a documented swap-in for the XGBoost
baseline. Registering it now means shipping the GNN does not also
require remembering this.

The ratio-shaped values `passthrough_ratio` and `round_amount_ratio` are
**not** registered and cannot be: they are `FEATURES_SCHEMA` columns in
`bq_schemas.py` with no Pydantic model behind them, so nothing typed
carries them across this boundary. They must also stay unrounded on
principle — they are model *inputs*, and rounding a feature changes the
score it produces.

**Nesting follows the typed structure.** Every field holding a
`BaseModel`, and every `BaseModel` inside a `list` field, is visited, so
`AccountExplanation.top_factors` (`list[ExplanationFactor]`) gets its
contributions rounded. A model with no registry entry comes back
unchanged, which is what makes blanket recursion safe. No contract model
refers to itself or an ancestor, so the graph is a finite tree — today
at most two levels (`AccountExplanation` → `ExplanationFactor`, `Ring` →
`RingMember`, `KycSubmission` → `KycDocument`) — and the recursion is
written against the structure rather than that number.

**Neither helper mutates.** `round_score_fields` returns a `model_copy`,
and the same instance when nothing needed rounding. Mutating in place
would be wrong rather than merely impure: `agents/tools.py` hands out
models read from a shared in-memory store, so in-place rounding would
round the *stored* row, and the verifier re-deriving a claim would then
compare a rounded value against a rounded value and find them equal by
construction — making `SCORE_TOLERANCE` vacuous. Rounding is a property
of the call, never of the record.

**`-0.0` survives both.** `round(-0.0, d)` is `-0.0` and nothing is done
to the result afterwards — no `abs`, no clamp, no `+ 0.0`. The validator
compares with `!=`, and since `-0.0 == 0.0` in Python it cannot tell the
two apart and therefore cannot normalise one into the other. A validator
that "fixed" `-0.0` to `0.0` would reclassify a mitigating factor as
neutral, which is the failure the whole promise exists to prevent.
`agents/test_live_fixes.py:311` depends on this.

**The validator raises `ValueError`, naming the field and both values:**

```
AccountExplanation.top_factors[2].contribution is not rounded to 3
places: found -0.1234567, expected -0.123. Pass the value through
shared.tool_interfaces.round_score_fields() before it crosses the tool
boundary.
```

The full path is there because a tool author needs to know *which*
field broke the promise, not merely that one did. `ValueError` and not
`assert`: this is a contract violation of the same class as the model
validators in `contracts.py`, and `assert` vanishes under `python -O`,
which would silently remove the check from the production run it guards.

`RiskScore`, `AccountExplanation` and `ExplanationFactor` in
`contracts.py` carry a pointer to this in their docstrings, so a reader
of the model knows the value arrives rounded. The constants themselves
stay in `tool_interfaces.py`, not `contracts.py`, for the same reason
`RingSubgraph` does: `contracts.py` describes persisted entities and the
stored rows are *not* rounded, so stating the promise on the record
would make the model say something untrue of the data it models. This is
a property of the call — the same kind of fact as `SQL_ROW_LIMIT`.

### Identity signals are evidence, never a detection

Detection fires on **velocity and structure only**. Identity reuse —
shared device fingerprints, reused KYC id numbers, matching photos — is
deliberately outside the firing condition: it was measured and found
strictly dominated as a detector (0 extra true positives, +146 false
positives). It is attached afterwards, as citable evidence for a
compliance analyst and for the report-writing agent.

That separation is enforced by the contracts. `AccountEvidence` is a
separate model from `RiskScore`; nothing on it feeds a score, and no
column of `account_evidence` appears in any rule.

**`scope` is load-bearing.** Identity reuse is a *weak* signal on its
own — roughly 1,442 legitimate accounts share a document within a bank
against ~292 mules, and 400 legitimate dual-bank customers reuse an id
across two banks. So `single_bank` reuse is mostly noise and
`cross_bank` reuse is the fact worth citing. Every consumer must read a
row together with its `scope` and its `counterparty_bank_ids`; an
aggregate that does not group by `scope`, or a UI badge that renders a
`single_bank` row as incriminating, is reporting a wrong number. A row
states what was observed. It never states guilt.

**This "weak signal" finding is about identity reuse as a standalone
rule trigger, not about identity-derived features inside a trained
model.** Those are different claims and both are true: firing an alert
on "these two accounts share a device" alone is weak, as measured
above. But `device_count`/`shared_device_count`/`document_reuse_*` as
*inputs* XGBoost learns to weigh jointly with transfer-graph features
are a different, much stronger finding — dropping all four from
Model A collapses its T=5m PR-AUC from 0.622 to 0.245, and dropping
them plus `account_age_days` collapses it to 0.071 (see
`ml/experiments/ablation/README.md`). A reader should not treat either
finding as contradicting the other.

### Ring membership is two different claims

A ring's members arrive by two routes, and the route is part of the
claim made about each account.

* **`velocity`** — stage 1. Fan-in, fast pass-through, cross-bank
  structure. A *finding* about the account's own behaviour.
* **`identity_link`** — stage 2. The account was pulled into an
  *already-detected* ring because it shares a device fingerprint or a
  hashed KYC id number with a member. A *lead*, never a finding. Every
  consumer must describe it as **"linked, needs review"** and never as a
  mule; the verifier rejects a narrative that does.

Stage 2 never creates a ring, only grows one stage 1 already found. It
recovers 58 of 187 `source` accounts (previously 0) and takes rings with
every member recovered from 0/120 to 34/120, for 13 false positives.

**Alerting is cross-bank only.** `RingMember.alertable` is `True` for
every `velocity` member, and for an `identity_link` member only when its
`link_scope` is `cross_bank`. Same-bank identity links are published as
**context**, not alerts: same-bank identity reuse is dominated by
legitimate cover (~1,442 legitimate accounts against ~292 mules), so it
is something to show an analyst, never something to accuse anyone with.
Cross-bank-only propagation is also sharper on its own terms — 35
sources for 1 false positive.

*Why this is in the contract and not a UI convention.* "Forwarded money
fast" and "shares a phone with someone who did" are not the same
accusation. If the distinction can be dropped by a consumer that forgets
to read a side-channel, an innocent family member ends up rendered as a
detected mule. So it is carried three ways that all have to agree: the
basis is a field *on the member record*, `alertable` is **materialised**
on the row rather than derived, and both a `model_validator` and three
`CHECK` constraints in `ddl.sql` refuse an inconsistent row.

`alertable` is stored, not derived, on purpose. Deriving it means
joining `account_evidence` and comparing `bank_id` across counterparties
— exactly the derivation a consumer will get wrong, and getting it wrong
publishes an alert against a relative. The validator keeps it pinned to
`link_scope`, so it states an invariant rather than being a second
source of truth that can disagree.

## contracts.py

`MoneyAmount` (see Conventions), Pydantic models and status enums: `Account`, `Transfer`, `Device`,
`KycDocument`, `KycSubmission`, `RiskScore`, `RingMember`, `Ring`,
`RingExplanation`, `GnnScore`, `ExplanationFactor`,
`AccountExplanation`, `AccountEvidence`, `Investigation`, `StrReport`,
plus `AccountStatus`, `RingStatus`, `InvestigationStatus`,
`StrReportStatus`, `EvidenceScope`, `MembershipBasis`.

`Ring.member_hashed_account_ids: list[str]` is **replaced** by
`Ring.members: list[RingMember]`, where `RingMember` is
`hashed_account_id`, `membership_basis`, `alertable`, `link_scope`,
`linked_from_hashed_account_id`, `link_evidence_type`, `reason`. See
"Ring membership is two different claims" above for why.

* The basis lives **on the member record**, not in a parallel
  `list[str]` aligned by index. A parallel array can be sliced,
  filtered or sorted out of alignment, and a misaligned basis renders an
  innocent relative as a detected mule — an invariant a validator can
  check at construction but cannot stop a consumer violating later. A
  record cannot be separated from its own basis. The price is a one-time
  migration of every consumer, which is the right price for an
  invariant about an accusation attached to a person.
* The flat list is **not kept alongside** as a compatibility alias. That
  would leave every existing consumer reading a list in which a linked
  relative is indistinguishable from a detected mule — the exact
  failure this change prevents, preserved under an old name. Removing it
  turns each call site into an `AttributeError`: a loud migration
  instead of a silent mislabelling. The replacements are deliberately
  named so a caller must say which population it means —
  `all_member_ids` (subgraph expansion), `velocity_member_ids`,
  `identity_link_member_ids`, `alertable_member_ids` (anything that
  counts or accuses), `context_member_ids`.
* `membership_basis` is a **closed `MembershipBasis` enum**, unlike
  `AccountEvidence.evidence_type`. That field is an open taxonomy of
  *observations* and grows; this one names the *accusation class*, and
  every consumer branches on it exhaustively. An open `str` would let a
  producer emit a third value that every `if basis == "identity_link"`
  branch falls straight through, silently rendering a lead as a mule. A
  stage-3 mechanism should be a deliberate contract change that forces
  every consumer to decide how to describe it.
* `link_scope` **reuses `EvidenceScope`** rather than introducing a
  second two-valued enum: same distinction, same underlying facts
  (`devices`, `kyc_documents`), one spelling.
* `reason` is a **denormalised** analyst-readable sentence, required for
  `identity_link`. `ui/` is required to show the reason, and a UI that
  must resolve a join before it can render will sooner or later render
  the member with the reason missing — the flat list again, an
  accusation with its qualifier dropped. It does not duplicate
  `account_evidence`: it states the *ring-scoped* edge (this member,
  this ring, that member), which no evidence row carries.
  `link_evidence_type` + `linked_from_hashed_account_id` + `link_scope`
  remain the join key into `account_evidence` for the full citable row.

`AccountExplanation` is the per-account counterpart of
`RingExplanation`: `hashed_account_id`, `score`, `model_version`,
`top_factors: list[ExplanationFactor]`, `narrative`, `generated_at`.

* It replaces **`GnnExplanation`, which is removed.** That model was
  named after a model family that was evaluated and *not shipped*, and
  its required `subgraph_ref` field only means anything for a GNN. The
  shipped scorer is XGBoost, so keeping it would have meant either a
  fabricated `subgraph_ref` or a second per-account explanation model
  that nothing produces — and `get_explanation` returning a union of
  two account shapes, only one of which ever exists. If a GNN is ever
  shipped, its explanation is still "top factors plus a narrative for
  one account": it reuses `AccountExplanation` and adds an *optional*
  `subgraph_ref` at that point, which is an additive change.
* `score` and `model_version` are carried on the explanation itself
  rather than joined from `scores`, so an explanation rendered beside a
  score it was not computed from is detectable instead of plausible.
* `top_factors` is `list[ExplanationFactor]`, not `list[str]` as on
  `RingExplanation`. A factor is `description` (analyst-readable text —
  "KYC ID number reused at another bank", never
  `cb_passthrough_speed_seconds`), `contribution` (signed, SHAP-style;
  the sign is load-bearing, since a list that drops it lets a
  *mitigating* factor read as incriminating) and optional
  `feature_name` (the raw column, for reproducibility, not for
  display). `RingExplanation.top_factors` stays `list[str]`: its
  strings already quote the ring's own measured values against the rule
  thresholds, and it has no per-factor attribution to carry.

`Investigation` is a case opened against a ring by a **verified**
investigation: `case_id`, `ring_id`, `verified_at`,
`verification_attempts`, `verified_claim_count`, `analyst_notes`,
`status`, `opened_at`, `closed_at`. See "A case means a narrative that
was checked" above for why those three verification fields exist and
why there is no `verified: bool`. `case_id` is no longer defaulted to a
random UUID: it must be `case_id_for_ring(ring_id)`, which is what makes
"one case per ring" a property of the id rather than a convention.
`MAX_VERIFICATION_ATTEMPTS` is **derived, never a literal** — the same
rule as `SCORE_TOLERANCE`, and for the same reason. `contracts.py` now
owns the retry budget too:

```python
MAX_RETRIES = 1                              # re-attempts AFTER the first draft
MAX_VERIFICATION_ATTEMPTS = MAX_RETRIES + 1  # == 2
```

Attempts are first draft + retries. The loop is investigator → verifier:
on a rejected claim the investigator gets exactly one more go with the
unsupported claims fed back, and if that also fails the narrative is
withheld. `MAX_VERIFICATION_ATTEMPTS` still evaluates to `2`, so nothing
changes behaviourally today.

It had to become a derivation because a frozen `2` was correct only by
coincidence with `agents/loop.py`'s own `MAX_RETRIES = 1`, and drifts
wrong in **either** direction as soon as the loop changes: raise the
retries to 2 and the validator rejects a legitimate three-attempt case;
lower them to 0 and a frozen `2` silently **accepts a row claiming a
retry that never happened** — the invisible failure, since an analyst
reads `verification_attempts = 2` as "passed on the retry".

`MAX_RETRIES` lives in `contracts.py`, not in `tool_interfaces.py`
beside `SQL_ROW_LIMIT` and `MAX_SUBGRAPH_ACCOUNTS`, for two reasons that
agree. Those are caps on what one *tool call* returns; this is a cap on
the agent orchestration loop, and what depends on it here is a
record-level constraint — the `Investigation` validator refuses a row
claiming an attempt the loop cannot make. And the import direction
forbids the alternative outright: `tool_interfaces.py` already imports
from `contracts.py`, so putting `MAX_RETRIES` there and importing it
back into `contracts.py` would be a circular import. `agents/loop.py`
imports the name instead of declaring its own.

`verification_attempts` is worth showing an analyst: in the live run
rings `00018` and `00001` passed on the first attempt and ring `00002`
only on the second.

`closed_at` is `None` unless `status` is `closed`, enforced by a
validator and by a `CHECK` in `ddl.sql`. `Investigation` deliberately
has **no `report_id`** — the link is `StrReport.case_id` and points the
way the cardinality runs, one case to many drafts; a back-pointer would
have to name one draft and the two sides could then disagree about
which is current. It deliberately has **no `assignee`** — MuleNet has no
user or identity model, so any value there would be invented.

`StrReport` gains `citations_verified` (see Conventions).

`AccountEvidence` is one citable identity-reuse fact attached to an
account: `hashed_account_id`, `evidence_type`, `scope`, `detail`,
`counterparty_hashed_account_ids`, `counterparty_bank_ids`,
`observed_at`, `source_table`, `detector_version`. One row per
`(hashed_account_id, evidence_type, scope)`.

* `scope` is an `EvidenceScope` enum (`single_bank` / `cross_bank`) —
  closed, because the entire reading of a row turns on it, and it
  matches the `scope` values on the `features` BigQuery table.
* `evidence_type` is a plain `str`, *not* an enum, unlike
  `AccountStatus` and `RingStatus`. Those are closed lifecycles; the
  evidence taxonomy is open and still growing — `ml/` emits
  `cross_bank_shared_device`, `cross_bank_reused_id_number` and
  `cross_bank_reused_photo`, and `docai/` also detects
  `normalized_address` reuse. A closed enum would force a `shared/`
  contract change for every new reuse key.
* `observed_at` is **required with no default**, unlike `scored_at` /
  `detected_at` / `generated_at` which default to `_utcnow()`. It means
  "when the link first became citable" — the later of the two sightings
  — which is a point-in-time claim an analyst may quote in a filing.
  Defaulting it to wall-clock now would fabricate that claim.
* `detector_version` is the only `Optional` field: evidence produced by
  `docai/` independently of a detector run has no detector version.

Produced by: `simulator/` (accounts, transfers, devices, KYC
submissions), `docai/` (parsed `KycDocument`, and single-bank /
cross-bank document-reuse `AccountEvidence`), `ml/` (`RiskScore`,
`Ring`, `RingExplanation`, `AccountExplanation`, cross-bank
`AccountEvidence`), `agents/` (`Investigation`, `StrReport`).
Consumed by: every folder. All other files in `shared/` are derived from
these models.

## ddl.sql

AlloyDB (PostgreSQL-compatible) DDL for the operational store:
`accounts`, `transfers`, `devices`, `kyc_submissions`, `kyc_documents`,
`rings`, `ring_members`, `cases`, `str_reports`, `account_evidence`. See
"AlloyDB is operational, BigQuery is model output" above for what is
deliberately *not* here. Field names and types mirror
`contracts.py`; `TIMESTAMPTZ` for datetimes, `NUMERIC(15,2)` for
amounts, `TEXT[]` for the ring and counterparty arrays. Foreign keys tie
transfers and devices to accounts, KYC documents to their submission and
account, cases to rings, and evidence to accounts. Indexes cover the
lookup columns the agents and UI filter on.

`rings` loses its `member_hashed_account_ids TEXT[]` column and its GIN
index; membership moves to the **`ring_members`** child table, one row
per `(ring_id, hashed_account_id)` with FKs to both. A `TEXT[]` cannot
hold records, so the real choice was a `JSONB` column on `rings` or this
table. The child table wins on three counts. (1) `membership_basis` and
`alertable` are *filters*, not payload — the whole point is that a
consumer, and `run_sql`, can write `WHERE alertable` instead of
re-deriving it; `JSONB` pushes that into containment operators that are
easy to get subtly wrong, and a wrong filter here alerts on an innocent
account. (2) The reverse lookup "which rings contain this account" —
where `get_account_profile` and the UI account view both start —
becomes a plain btree index. (3) Three `CHECK` constraints mirror
`RingMember._link_fields_match_basis`, so the database refuses a
mislabelled row by `UPDATE` just as the model refuses it on
construction; `JSONB` can enforce none of that. The cost is one table
and a join on read — the same join `ui/` and `agents/` already do
against `scores` and `account_evidence`.

`account_evidence` has no surrogate key: its primary key is the natural
`(hashed_account_id, evidence_type, scope)`, which already serves lookup
by account, so only `scope`, `evidence_type` and `observed_at` carry
their own indexes, plus a GIN index on
`counterparty_hashed_account_ids` for the reverse "who names this
account" lookup. `scope` carries a `CHECK` constraint mirroring
`EvidenceScope`, in the same style as the `status` columns;
`evidence_type` deliberately does not, matching the open taxonomy in
`contracts.py`.

`cases` is produced by `agents/`: one row per ring whose investigation
**passed verification**, published as `agents/out/cases.parquet`,
mirroring the `ml/out/` convention, which `infra/` loads. `ring_id` is
`UNIQUE` and the three verification columns carry `CHECK` constraints
mirroring the model validator; see "A case means a narrative that was
checked" above. `infra/` must load it as an upsert on `case_id` that
refreshes `verified_at`, `verification_attempts`,
`verified_claim_count` and `analyst_notes` and leaves `status` /
`closed_at` alone, skipping rows whose existing case is already
`closed`. `str_reports` is
produced by the report agent in the same folder
(`agents/out/str_reports.parquet`), one or more rows per case, FK to
`cases`, and carries `citations_verified` with the `CHECK` described
above.

Produced by: applied by `infra/` setup scripts.
Consumed by: `ingest/` (Cloud Run consumer writes), `docai/` (KYC
writes), `ml/` (ring writes), `agents/` (`cases` and `str_reports`
writes), `agents/` and `ui/` (reads).

## bq_schemas.py

BigQuery schemas as `google.cloud.bigquery.SchemaField` lists for
`transfers`, `features`, `scores`, `rings`, `ring_explanations`,
`account_explanations`, `account_evidence`, `cases`, `str_reports`,
exported individually and through the `TABLE_SCHEMAS` dict.

`cases` now has a schema here **as well as** a table in `ddl.sql`, the
same dual role `str_reports` has and for the same reason: the row is
operational (an analyst moves `status` and sets `closed_at`) but the
table is also read as an aggregate — how many detected rings ended in a
verified case, and what share passed on the first attempt rather than
the retry. Those are scans over every row. Until now `cases` was in
`ALLOWED_SQL_TABLES` with no schema here, so the one tool allowed to
query it had nothing to query; that gap is closed. The three
verification columns are `REQUIRED`, so an exported row carries the
fact that it was verified without consulting another source — the same
rule as `str_reports.citations_verified`.

`rings.members` is a **repeated RECORD** (`hashed_account_id`,
`membership_basis`, `alertable`, `link_scope`,
`linked_from_hashed_account_id`, `link_evidence_type`, `reason`),
replacing the repeated STRING `member_hashed_account_ids`, which is
gone. Same reasoning as `account_explanations.top_factors`: a bare
string cannot carry the per-member attribute, and a parallel repeated
STRING would be index-aligned by convention only. BigQuery nests where
AlloyDB uses a child table, which is the existing asymmetry — BigQuery
is scanned and aggregated, AlloyDB is read row-by-row and filtered. An
aggregate over `members` that does not filter on `alertable` is
reporting a wrong alert count.

`account_evidence` mirrors `AccountEvidence`, with `REPEATED` mode for
`counterparty_hashed_account_ids` and `counterparty_bank_ids` and
`NULLABLE` for `detector_version`.

`features` is the `ml/` feature store, one row per
`(hashed_account_id, as_of, scope)`. `scope` is `single_bank` or `cross_bank` —
that comparison is the project's headline result. It is published as
`ml/out/features.parquet`, beside every other published table, and
listed in `ml/out/manifest.json`. `scores` still unions the XGBoost and
GNN score variants from `contracts.py`, discriminated by `model_type`;
explanations no longer union anything (see Conventions).

`account_explanations.top_factors` is a **repeated RECORD**
(`description`, `contribution`, `feature_name`), not a repeated STRING,
because the signed contribution cannot be carried by a bare string.

`str_reports` mirrors `StrReport`, with `citations_verified` as a
`REQUIRED BOOL` so the flag travels with every exported row.

Produced by: `infra/` creates the tables; `ingest/` writes `transfers`;
`ml/` writes `features`, `scores`, `rings`, `ring_explanations`,
`account_explanations`; `agents/` writes `str_reports`.
Consumed by: `ml/` (training and evaluation), `agents/` and `ui/`
(analyst queries, including `run_sql` — except `str_reports`, which is
off the allowlist).

## pubsub_schema.py

The Pub/Sub envelope for transfer events: `TransferMessage`
(`message_id`, `published_at`, `bank_id`, `payload: Transfer`), which
reuses `Transfer` from `contracts.py` rather than restating its fields.
Also holds the topic, subscription, and dead-letter naming constants
used by `infra/`, and `encode_message` / `decode_message` /
`encode_transfer` / `decode_transfer` helpers.

Produced by: `simulator/` and the `ingest/` publisher.
Consumed by: the `ingest/` Cloud Run consumer; `infra/` for resource
names.

## tool_interfaces.py

Signatures and Google-style docstrings only (bodies raise
`NotImplementedError`) for the seven tools the ADK agents call:
`get_ring_subgraph`, `get_explanation`, `trace_money_flow`,
`get_account_profile`, `get_document_reuse`, `search_regulations`,
`run_sql`. Return types are `contracts.py` models.

`get_explanation` returns `RingExplanation | AccountExplanation` and is
backed by the two separate tables; the `subject_type="account"` branch
is now producible rather than always raising `KeyError`.

`get_ring_subgraph` returns a **`RingSubgraph`**, not the
`tuple[Ring, list[Account], list[Transfer]]` it used to, and `max_hops`
defaults to **1**, not 2. The unbounded version killed a live request:
the model called the tool without `max_hops`, took the old default of 2
and got 29,582 accounts and 185,011 transfers back — 49,001,023 tokens
against a 1,048,576 limit, 47× over. The same ring at one hop is 1,289
accounts and 2,199 transfers, because every mule account is a hub and
the second hop pulls in the counterparties of every counterparty.

* `RingSubgraph.__iter__` yields `(ring, accounts, transfers)`, so all
  fifteen existing `ring, accounts, transfers = ...` unpack sites keep
  working. The truncation data sits on **attributes**, not in the tuple,
  precisely so that adding it does not break every caller in order to
  tell them something only the model needs.
* **Ring members are never truncated; only counterparties are.** This is
  a correctness property, not a convenience. Every claim the verifier
  re-derives — member count, alertable member count, bank count, banks,
  account-in-ring — is about the member population, so a capped member
  list would not make the answer smaller, it would make it **wrong**,
  turning a size limit into a false statement about how many accounts
  were implicated. `MAX_SUBGRAPH_ACCOUNTS = 200` therefore caps only the
  counterparties; `MAX_SUBGRAPH_TRANSFERS = 400` keeps the transfers
  that move the most money.
* `truncated`, `total_accounts`, `total_transfers` and the
  plain-language `note` exist so the model knows data was cut rather
  than reasoning confidently over a partial graph and stating a total it
  derived from `len()` of a capped list. Silent truncation turns a size
  problem into a correctness problem.
* The caps live here rather than in `agents/` for the same reason
  `SQL_ROW_LIMIT` does: a caller cannot read the answer correctly
  without knowing the answer may be capped.
* `RingSubgraph` is a frozen `dataclass` in **`tool_interfaces.py`, not
  a Pydantic model in `contracts.py`**. It is a per-call tool response,
  not a persisted entity: every model in `contracts.py` has a row in
  `ddl.sql` or `bq_schemas.py` behind it, and this has neither and never
  will. It is assembled from already-validated `Ring`, `Account` and
  `Transfer` values rather than parsed from untrusted input, so
  validation would buy nothing — and a `BaseModel` would make the
  `__iter__` override actively harmful, since Pydantic's own `__iter__`
  yields `(field_name, value)` pairs and backs `dict(model)`. `ui/`
  imports it from `shared.tool_interfaces` like any other contract.

`SCORE_DECIMALS = 3` and `SCORE_TOLERANCE = 0.5 * 10**-SCORE_DECIMALS`
(`== 0.0005`) also live here, together with the two helpers that make
the rounding mechanical rather than remembered:
`round_score_fields(model)` returns a copy with every registered score
field rounded, and `assert_score_fields_rounded(model)` raises
`ValueError` naming the field and both values when one is not. Scores
and signed contributions returned by `get_account_profile` and
`get_explanation` are rounded to `SCORE_DECIMALS` places; anything
comparing a quoted score to a recomputed one uses `SCORE_TOLERANCE` and
never its own literal. See "Score rounding and its comparison tolerance
are one fact" above for the registry, the money exclusion and the
`-0.0` rule.

`run_sql` is deliberately constrained: a single read-only `SELECT`,
restricted to the tables in `ALLOWED_SQL_TABLES`, with `LIMIT 100`
(`SQL_ROW_LIMIT`) enforced.

**`str_reports` is deliberately NOT in `ALLOWED_SQL_TABLES`**, even
though it has both a DDL table and a BigQuery schema. It holds
generated narratives whose citations are currently synthetic
placeholders. An agent that can `SELECT` prior reports can retrieve one
and cite it as corroboration, so an unverified draft becomes apparent
evidence for the next draft — a laundering-of-provenance loop in which
a placeholder citation eventually reads as an established prior
finding, and the `citations_verified` flag is bypassed because the
second report never touched the corpus at all. Every fact in an STR
must trace to a primary table (`transfers`, `rings`, `account_evidence`,
`scores`), never to another report. `ui/` still reads `str_reports`
directly — the restriction is on the agent's SQL tool, not on the data.
Revisit only when reports carry a human sign-off, and then admit
`status = 'filed'` rows alone.

`ring_members` is in `ALLOWED_SQL_TABLES`: the basis is the difference
between "this account forwarded money fast" and "this account shares a
phone with someone who did", and an agent that can query `rings` but not
`ring_members` can only reach member accounts through a flat list that
has lost the distinction. Any agent query that counts alerts must filter
on `alertable`.

`account_evidence` is in that allowlist so
the report-writing agent can pull the facts it cites; it is a read-only
evidence table containing only hashed ids, and without it the agent
would have no path to the counterparty banks a narrative must name.

`get_document_reuse` returns `list[AccountEvidence]`. It previously
returned `dict[str, list[KycDocument]]`, which could carry neither the
counterparty bank nor the `scope` — the two things that decide whether
a reuse fact is citable or noise.

Produced by: implemented in `agents/`.
Consumed by: the ADK/Gemini agents and, where useful, `ui/`.
