# ml

Feature engineering, the mule detector, ring clustering, and per-account
explanations. Owned by the **ml-engineer** agent. Reads and writes the
contracts defined in `shared/`.

## What ships

**The detector is a velocity rule, not a model.** It fires on
pass-through speed, flow ratio, fan-in, fan-out and counterparty-bank
spread — and on nothing else. Identity reuse (shared devices, reused KYC
IDs, matching photos) was measured as a detection clause, found strictly
dominated, and moved out of the firing condition into **evidence**
attached to each flagged account for the analyst and the STR agent.

## Modules

| module | what it does |
|---|---|
| `config.py` | env-driven paths and knobs |
| `features.py` | paired `sb_`/`cb_` per-account features, cutoff-aware — **canonical** |
| `bq_features.py` | the same features again, as BigQuery SQL over `transfers`; a check, not a source |
| `evaluate.py` | grouped 5-fold CV over rings, span strata, XGBoost baseline |
| `metrics.py` | PR-AUC, ROC-AUC, recall@FPR, false-alerts-per-catch |
| `rule.py` | the shipped velocity rule + the record of why identity was retired |
| `tune.py` | **honest tuning**: grid search inside each training fold, scored out-of-fold |
| `evidence.py` | identity-reuse evidence, long-format and citable |
| `early.py` | early detection at T = 5/10/20/30/60 min |
| `gnn.py` | GraphSAGE over the account graph, judged at T = 5/10/30 min |
| `publish.py` | writes the five `ml/out/` tables, all validated against `shared/contracts.py` |
| `compare.py` | unified table over `experiments/**/metrics.json` |
| `vertex/gate.py` | bit-exact comparator, cloud vs local: every JSON leaf plus md5 |
| `vertex/claims_gate.py` | claims-level comparator: directions, span ordering, values within the seed-spread tolerance |

## Running things

```bash
python -m ml.features    # -> ml/features.parquet
python -m ml.tune        # -> held-out threshold tuning
python -m ml.rule        # -> the shipped rule + evidence attachment rates
python -m ml.publish_all # -> ml/out/*.parquet  (see ml/out/README.md)
python -m ml.early       # -> early-detection sweep + the OOF store
python -m ml.ring_map    # -> detector <-> ground-truth ring id map
python -m ml.leadtime    # -> warning time before cash-out, per ring
python -m ml.claims      # -> docs/claims.md, every quotable number + provenance
python -m ml.gnn         # -> GraphSAGE  (needs torch + torch_geometric)
python -m ml.compare     # -> one table across every experiment

python -m ml.bq_features           # -> BigQuery/pandas parity, both cutoff scenarios
python -m ml.bq_features --leakage # -> the point-in-time leakage check
```

Outputs consumed by other agents live in **`ml/out/`** — read
[`ml/out/README.md`](out/README.md) before using them; it documents each
table, which model produced it, and the two caveats that matter
(ring-level coverage is weaker than account-level recall, and two of the
three evidence types are the same fact in this dataset).

`ml.publish_all` — not `ml.publish` — is the supported way to rebuild
`ml/out/`. Both `ml.publish` and `ml.timeline` write `manifest.json`, the
first wholesale and the second by appending, so running `ml.publish`
alone leaves the manifest describing five tables while six sit on disk,
with no error. `ml.publish_all` runs them in the right order.

## Two ways to run the training, and which one is authoritative

**The local run is the source of truth. Every published number comes
from it.** The cloud job is a reproduction of it, not a replacement.

### The local canonical run

```bash
python -m ml.early      # ~20 min: metrics.json + the 50-file OOF store
python -m ml.leadtime   # ~15 s : lead-time metrics.json + per_ring.parquet
```

This is what `docs/claims.md` resolves against, what `ml/test_claims.py`
re-resolves, and what `ml.publish_all` builds `ml/out/` from. It is the
only run whose output is committed to, and it is reproducible: two
separate local runs on 2026-09-27 produced **4316 + 1079 identical JSON
leaves with 0 differing**, all **50** OOF parquet files byte-identical,
and `per_ring.parquet` byte-identical. The only thing that moved between
them was the `generated_at` wall clock in each metrics file.

### The Vertex AI custom job

```bash
RUN=run-$(date -u +%Y%m%dt%H%M%S)
python -m ml.vertex.submit build    --run-id "$RUN" --context-dir /tmp/ctx
python -m ml.vertex.submit run      --run-id "$RUN" --seed 42
python -m ml.vertex.submit download --run-id "$RUN" --dest /tmp/cloud
python -m ml.vertex.submit register --run-id "$RUN"
python -m ml.vertex.submit log      --run-id "$RUN" --artifacts /tmp/cloud

# guarantee 1 -- bit-exact. The right check for a SAME-platform re-run.
python -m ml.vertex.gate \
  --pair early     ml/experiments/early_detection/metrics.json \
                   /tmp/cloud/experiments/early_detection/metrics.json \
  --pair lead_time ml/experiments/lead_time/metrics.json \
                   /tmp/cloud/experiments/lead_time/metrics.json \
  --bytes oof ml/experiments/early_detection/oof \
              /tmp/cloud/experiments/early_detection/oof '*.parquet'

# guarantee 2 -- the conclusions. The right check ACROSS platforms.
python -m ml.vertex.claims_gate \
  --local-experiments ml/experiments \
  --cloud-experiments /tmp/cloud/experiments
```

| | |
|---|---|
| what it runs | `ml.early` then `ml.leadtime`, unmodified, via `ml/vertex/task.py` |
| where | Vertex AI custom job, **`asia-south1`**, `n1-standard-4`, 1 replica, CPU only |
| seed | 42 (`ml.config.DEFAULT_SEED`; all three of `ml.evaluate.SEEDS` are swept) |
| input | `gs://mulenet-mulenet-data-asia/simulator/out` — the six tables, md5-verified identical to `simulator/out/` |
| output | `gs://mulenet-mulenet-data-asia/ml/vertex_runs/<run-id>/` |
| registry | Vertex AI Model Registry, `asia-south1`, `mulenet-xgboost-cb-oof-v1`, model id `1756698523790737408` |
| experiments | Vertex AI Experiments, `asia-south1`, `mulenet-early-detection` |

**Why `asia-south1`.** Vertex AI custom jobs and Model Registry are
regional, so a region has to be chosen rather than inherited. MuleNet's
other Vertex usage — Gemini — is already there (`docs/residency.md`), the
`mulenet` Artifact Registry repo exists there, and the training inputs
are mirrored there. **There is no cross-region read**: the job reads the
`asia-south1` mirror, not the `us-central1` original. The published
tables and BigQuery stay in `us-central1`; this job does not touch them.

**What it is for.** Reviewability and lineage — a job spec, an image
digest, a Model Registry version and an Experiments run that together say
exactly what produced the numbers — plus a standing check that the result
is not an artefact of one laptop. **It is not what the deck quotes.**

**What is registered, and what it is not.** `ml.early` fits models and
keeps only their pooled out-of-fold predictions; it never writes a
booster. So `ml/vertex/export_model.py` re-fits exactly the folds behind
the published scores (scope `cb_`, full window, seed 42 — what
`ml/out/manifest.json` calls `mulenet-xgboost-cb-oof-v1`) and **asserts
the re-fitted out-of-fold scores are bit-identical** to the stored OOF
file before anything is registered. Five per-fold boosters are not one
servable predictor, so the registry entry is a **lineage and version
record, not a deployable endpoint**. `metadata.json` beside the boosters
says so in `servable: false`.

## Two reproducibility guarantees, and the second is weaker

This project claims **two** different things about reproducibility, and
they are not the same strength. Both are runnable; keeping them separate
is the point, because collapsing them would let the weaker one be quoted
as if it were the stronger.

| | guarantee 1 | guarantee 2 |
|---|---|---|
| claim | **bit-exact** output | the **conclusions** hold |
| holds | on the **same platform** | **across platforms** |
| comparator | `ml/vertex/gate.py` | `ml/vertex/claims_gate.py` |
| resolution | one bit | the local run's own seed-to-seed spread |
| evidence | local re-runs: 0 differing leaves of 5395, 50/50 OOF files byte-identical | cloud `run-20260927t124500`, 79/79 checks pass |
| verdict today | **PASSES** locally, **FAILS** local-vs-cloud | **PASSES** |

**Guarantee 2 is strictly weaker than guarantee 1**, and it is weaker in
a specific way: it cannot see any effect smaller than the seed noise it
uses as its resolution. A cloud run could pass it and still be a
materially different model. It is accepted here because guarantee 1 is
**not achievable** across platforms — see "one thing cannot be pinned"
below — and refusing to state anything at all would be worse than
stating the weaker thing precisely. What is *not* accepted is quoting
cloud numbers: **every published number still comes from the local run.**

### Guarantee 1 — bit-exact, on the same platform

**Measured, on 2026-09-27: separate local runs of `python -m ml.early`
and `python -m ml.leadtime` produced 4316 + 1079 identical JSON leaves
with 0 differing, all 50 out-of-fold parquet files byte-identical, and
`per_ring.parquet` byte-identical.** The only thing that moved between
them was the `generated_at` wall clock in each metrics
file. That is the whole guarantee: on one machine, the canonical training
is bit-reproducible, and `ml/vertex/gate.py` is the comparator that says
so.

**This comparator is kept, not replaced.** It is the right check for a
same-platform re-run — after a refactor, a dependency bump, or a second
run on the same box — and it is the only check that can prove a
refactor changed nothing. It is the wrong check for a cross-platform run,
which is the next section.

What the comparator asserts, whichever two runs it is pointed at:

* **`metrics.json`, both files: every JSON leaf equal, by `repr`, with no
  tolerance — except the single top-level `generated_at` in each.** That
  one exception is forced: `ml.early` and `ml.leadtime` both write
  `datetime.now(UTC).isoformat()` into the file, so a byte-for-byte gate
  on `metrics.json` cannot pass by construction. `ml/vertex/gate.py`
  `IGNORED_PATHS` is the whole exception list and holds that one path.
* **The OOF store and `per_ring.parquet`: byte-for-byte identical**, md5,
  no exception list at all — those files carry no wall clock.

So across the whole output, exactly two timestamp strings are not
compared bit-for-bit. A float differing in its last bit fails the gate
and is reported with old value, new value and delta.

### Guarantee 1 does NOT hold across platforms, and this is why

Run `run-20260927t124500`, job
`projects/449407000058/locations/asia-south1/customJobs/208890991480930304`,
`JOB_STATE_SUCCEEDED`:

| comparison | compared | differing | verdict |
|---|---|---|---|
| `early_detection/metrics.json` | 4316 leaves | **2374** | FAIL |
| `lead_time/metrics.json` | 1079 leaves | **342** | FAIL |
| `oof/*.parquet` (bytes) | 50 files | **50** | FAIL |
| `lead_time/per_ring.parquet` (bytes) | 1 file | **1** | FAIL |

**The data path is not the cause. The model is.** Splitting the differing
leaves by what they depend on isolates it exactly:

| leaf group | leaves | differing |
|---|---|---|
| `sanity_check` — per-account cutoffs, cutoff quantiles, median degrees | 325 | **0** |
| exit/onset structure — `rings_money_not_yet_at_exit` | 68 | **0** |
| `positives`, `rows`, `n_features` | 554 | **0** |
| metric means — every one model-dependent | 444 | **405** |

So the six staged inputs, the per-account cutoffs, the resampled
reference instants for non-ring accounts, the fold-independent feature
values and the ring/exit structure are all **identical**. In the OOF
store the account ids and `as_of` instants match row for row; only
`score` moves — 29,972 of 30,000 rows, max |delta| 0.339, mean 4.3e-05,
and the 1%-FPR threshold shifts from 0.0004371683 to 0.0004143202.

Different scores mean different flags, so 211 integer counts in
`early_detection` and 52 in `lead_time` changed too — e.g.
`horizons[0].ring_catch_per_seed.cross_bank.42.rings_caught` 102 -> 108.
Only 17 of the 2374 early differences are inside 1e-12; the median
|delta| is 6.6e-03. **This is not float dust, it is a different fit.**

The remaining unpinned variable is the one `requirements-train.txt`
already names as unpinnable: xgboost 3.4.1 as a `win_amd64` wheel locally
versus a `manylinux_2_28` x86_64 wheel in the container, two different
compiled binaries of `tree_method="hist"`. A secondary suspect is the
interpreter patch level — the image resolved `python:3.14-slim` to
**3.14.7** against the local **3.14.5**.

**Consequence: the cloud job is not a reproduction of the local run
today, and nothing published may be taken from it.** The Model Registry
entry is therefore the export from the **local** run, whose boosters were
asserted bit-identical to the published OOF scores — not the cloud run's.
The Experiments run carries its own `gate_verdict` param saying FAILED,
so the entry cannot be read as authoritative on its own.

**This is not the end of the story, it is the reason guarantee 2 exists.**
The failure is confined to the fit, and the question "did the conclusions
survive?" is a different question with a different answer — below.

**A gate on `ml/out/` was considered and rejected.** `ml.publish` writes
`generated_at` from `datetime.now(UTC)` into two columns
(`account_explanations`, `ring_explanations`), so a byte gate there
cannot pass either — and `ml.publish_all` does no training at all, it
reshapes what `ml.early` already produced. Gating the thing that fits the
models, on the file the claims resolve from, is both the stronger gate
and the one that matches the word "training".

### Guarantee 2 — the conclusions reproduce across platforms

`ml/vertex/claims_gate.py`. Since bit-identical XGBoost across two
compiled wheels is not a reasonable thing to demand, the cross-platform
question is a different one: **does the cloud run support the same
statements `docs/claims.md` publishes?** Three checks, all on the cloud
outputs of `run-20260927t124500`:

```bash
python -m ml.vertex.claims_gate \
  --local-experiments ml/experiments \
  --cloud-experiments /tmp/cloud/experiments
```

1. **Direction at every horizon.** Every scope comparison in
   `docs/claims.md` keeps the sign it has locally at **all nine**
   horizons, not just the headline one. The comparison set is *derived*
   from `ml.claims.CLAIMS` by path identity — a claim added or retired
   there changes what the gate checks, with no second list to maintain.
2. **Span ordering.** At T=5m: span-1 near zero, spans 2 and 3 positive.
   Plus span3 > span2 > span1 at every one of the nine horizons, in both
   runs.
3. **Headline values** within a derived tolerance of the local canonical
   run, for every claim in `docs/claims.md`.

**Result: PASSED, 79 of 79 checks** — 30 direction checks, 24 span
checks, 25 value checks.

#### The tolerance is the local run's own seed-to-seed spread

Every published number is a mean over the three seeds in
`ml.evaluate.SEEDS`, and this project already treats the seed as a
nuisance parameter that changes no conclusion — it quotes the mean and
reports the spread beside it. So **`max - min` across those three seeds
is the measured size of a difference the project has already decided it
reads nothing into**, and it is the yardstick for "the cloud landed in
the same place". It is not a knob: nothing here was chosen to make the
run pass.

For each claim the gate maps the `*_mean` leaf back to the per-seed
records it was averaged from (`runs[]`, `ring_catch_per_seed`,
`per_seed`), takes `max - min`, and **pools that over the claim's
siblings at the same level of aggregation** — the other scope, the other
spans, the other horizons — taking the maximum. Pooling is deliberate:
three draws estimate a spread badly, and for the full-window horizon
(one seed) or a quantity that lands identically on all three it is
`0.0`, which would silently restore a bit-exact gate. Pooling stays
*within* a level of aggregation, so a summary-level claim is not handed
the noise of a thin span-3 stratum — worth 1.04 versus 7.63 on
`false_alerts_per_catch`, i.e. the difference between a tolerance smaller
than the claim and one larger than it. The report prints the pooled
tolerance **and** the claim's own spread, so how much came from pooling
is visible per row.

The headline numbers and their allowances:

| claim | local | cloud | delta | tolerance | own spread |
|---|---|---|---|---|---|
| `pr_auc_t5_cross_bank` | 0.6220 | 0.6190 | −0.0029 | 0.0172 | 0.0089 |
| `pr_auc_t5_single_bank` | 0.4282 | 0.4283 | +0.0001 | 0.0172 | 0.0063 |
| `alert_burden_t5_cross_bank` | 1.9678 | 2.0257 | +0.0579 | 1.0439 | 0.2439 |
| `ring_recall_t180_single_bank` | 0.9250 | 0.9278 | +0.0028 | 0.0417 | 0.0167 |
| `coverage_span2_cross_bank` | 1.0000 | 1.0000 | 0.0000 | 0.0833 | 0.0000 |
| `lead_median_minutes` | 0.0 | 0.0 | 0.0 | **0.0000** | 0.0000 |
| `exit_minutes_p50` | 89.4 | 89.4 | 0.0 | **0.0000** | 0.0000 |

The two rows with tolerance `0.0000` are the tightest checks in the gate
and they are **exact matches, not allowances**. `exit_distribution` has
no seed dimension anywhere — it is a property of the generated data, not
of any fit — so the gate demands exactness, and the bit-exact diagnosis
is what licenses that: the data path *is* reproducible across platforms,
so this is a live check that would catch a data-path regression.
`lead_median_minutes` lands at exactly 0 on all three seeds because 90 of
112 rings tie, so it too is required to match exactly, and did.

#### "Near zero" for the span-1 gradient means 0.0349, and here is why

Span-1's cross-bank advantage at T=5m is **+0.0085 locally with a
seed-to-seed spread of 0.0349**, and the three per-seed deltas are
**−0.0034, −0.0025, +0.0314** — two of three *negative*. A quantity whose
sign is not consistent across the seeds it was averaged over is not a
positive finding. So the gate's near-zero band is that quantity's own
spread at that horizon, **0.0349**, and the assertion for span 1 is
`|delta| <= 0.0349`, not `delta > 0`. Spans 2 and 3 must clear the same
bar *and* be positive on every individual seed, which is strictly
stronger.

The band is deliberately the quantity's **own** spread rather than the
pooled tolerance (0.0602) the value check uses: "near zero" is a
statement about one quantity at one horizon, so the resolution it was
measured at is the honest threshold, and it is the tighter of the two.

**This is where the cross-platform shift actually showed up, and the
gate's shape is what absorbed it.** In the cloud the span-1 gradient is
**+0.0001** — it did not flip sign, but it collapsed to essentially
nothing, per-seed **−0.0107, +0.0031, +0.0078**, again mixed. Had the
gate asserted `span1 > 0`, that claim would have been one seed away from
failing, and it would have been the *gate* that was wrong, not the run:
a claim of "positive" was never supportable at +0.0085 with a spread four
times its size. Spans 2 and 3 moved by +0.0016 and −0.0002.

#### What this gate does not promise

* **Not that the cloud reproduced the local run.** It says the cloud run
  contradicts no published claim, at a resolution set by the local run's
  own seed noise.
* **Any real effect below that resolution is invisible to it.** The
  span-1 gradient is exactly such an effect, which is why it is reported
  as near zero rather than as an advantage.
* **A comparison whose gap is inside the tolerance is reported as
  `TIE`, not as a pass.** The cloud must tie in the same place. Three
  rows do this and all three agree: `ring_recall` and `alert_burden` at
  the full window, where cross-bank and single-bank both saturate, and
  `coverage_span1` — span-1 rings show no cross-bank coverage advantage in
  either run, as they should not.
* **It does not license quoting a cloud number.** Guarantee 2 is about
  conclusions; `docs/claims.md` still resolves against the local file.

### Reproducibility is pinned, and one thing cannot be

`ml/vertex/requirements-train.txt` pins the interpreter and every numeric
library to the exact versions the local run used (Python 3.14,
xgboost 3.4.1, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0,
scipy 1.18.0, pyarrow 24.0.0). Note that this is 3.14, **not** the 3.11
`pyproject.toml` names — that pin is for the Cloud Run service images;
here the interpreter has to match the machine the canonical file was
produced on.

What cannot be pinned is the compiled binary: locally these are
`win_amd64` wheels, in the container they are `manylinux_2_28` x86_64
wheels of the same version, built by a different compiler for a different
CPU baseline. XGBoost's `hist` method also sums gradients per thread, so
`ml.evaluate.XGB_PARAMS`'s `n_jobs=4` is part of the numerical result —
which is why the machine type has 4 vCPUs and is recorded above. That
residual risk is exactly why there is a gate instead of an assumption.

## Quoting a number

Take it from [`docs/claims.md`](../docs/claims.md), never from a metrics
file by hand and never from another document. That file is generated by
`python -m ml.claims`, and every entry carries the metrics file and the
exact key path it was resolved from. `ml/test_claims.py` re-resolves all of
them and fails if any has drifted, so a stale number is a red test rather
than a wrong slide.

Regenerate it after any run that moves a metric:

```bash
python -m ml.early && python -m ml.leadtime && python -m ml.claims
```

It also records what was **retired** and why, so a metric that was dropped
for being degenerate does not quietly reappear on a slide because someone
found it in a JSON file.

## Read `caught_before_exit` with the horizon grid in hand

`ring_catch.share_caught_before_exit` is **not** a detector-quality number
on its own, and the extended grid makes that obvious. A ring only counts
as caught-before-exit if the horizon precedes its cash-out, so the metric
is bounded by `rings_money_not_yet_at_exit` — the rings whose money is
still in flight at that horizon:

| T | rings still in flight | cb caught before exit |
|---|---|---|
| 5 | 119 / 120 | 104.3 |
| 60 | 93 / 120 | 90.0 |
| 90 | 60 / 120 | 58.7 |
| 120 | 21 / 120 | 18.7 |
| 180 | **0 / 120** | **0.0** |

At T=5 the qualifier does almost nothing: cross-bank caught-before-exit
(0.8694) equals cross-bank `ring_recall` (0.8694) exactly, because only
one ring has already cashed out. At T=180 every exit has happened, so the
share is **structurally zero** — the same as the full window, and for the
same reason. The metric therefore peaks at the shortest horizon and
vanishes at the horizon where detection is actually strongest
(cb PR-AUC 0.9876 at T=180).

So do not read it as "how often we beat the money out the door". Use
**`ml/experiments/lead_time/`** for that: it measures, per ring, the
first horizon at which the ring was flagged and the signed minutes
between that flag and the cash-out. It keeps three kinds of missing apart
(`flagged_in_grid`, `flagged_full_window_only`, `never_flagged`) and never
averages an unflagged ring in as zero, and it reports both the
**conditional** median (over flagged rings, n stated) and the
**unconditional** share of all rings with positive warning.

The headline from it: cross-bank's advantage is **coverage, not speed**.
Among rings both scopes flag inside the grid the median lead is 0 minutes
(90 of 112 tie), but cross-bank flags 120/120 rings within the grid versus
single-bank's 112.3, and those rings single-bank misses are only ever
caught at the full window — days after the money left. Unconditional share
with positive warning: **0.961 cross-bank vs 0.847 single-bank**, and the
gap is entirely in the 2- and 3-bank rings (span 1 shows no cross-bank
advantage at all, as it should not).

## Two ring id spaces, and the map between them

`ml/out/rings.parquet` names rings after the **detector**
(`ring-mulenet-velocity-rule-v2-00007`, 45 of them).
`ml/out/ring_detection_timeline.parquet` and
`simulator/out/ground_truth.parquet` name them after the **generator**
(`ring_003`, 120 of them). **The two id spaces do not intersect at all** —
nothing in `rings.parquet` references a true ring, so a consumer holding a
detected ring cannot reach its ground truth unaided.

`ml/out/ring_id_map.parquet` (from `ml/ring_map.py`) is that join: one row
per detected ring, pairing it with the true ring it shares the most
members with, plus the overlap count and both ring sizes so a consumer can
judge the pairing instead of trusting it.

**Read `match_is_ambiguous` before showing a pairing to anyone.** Best
overlap is a heuristic, not identity. Detector clusters are connected
components over flagged accounts, so one cluster can swallow several true
rings that transact with each other — on this data 7 of 45 detected rings
overlap more than one true ring, and one spans four. For those, naming a
single true ring describes a minority of what the analyst is looking at.
One detected ring overlaps no true ring at all (a false-positive cluster);
it keeps its row with a NULL `true_ring_id`.

Pairing ties are broken by the lexicographically smallest `true_ring_id`,
so the published pairing never depends on row or dict order, and
`overlap_tied` marks where that rule had to decide.

## The out-of-fold score store

`python -m ml.early` writes 50 parquet files (~109 MB) to
**`ml/experiments/early_detection/oof/`** — one per (horizon, scope,
seed), each holding every account's out-of-fold score, the observation
instant it was computed at, and that run's 1%-FPR threshold.

It exists because `ml.evaluate.strip_scores` drops the score arrays
before `metrics.json` is written. Without the store, no per-account or
per-ring question about the early-detection results can be answered
without refitting, and `ml/out/ring_detection_timeline.parquet` — the
per-ring catch-before-cash-out table — is derived from it and from
nothing else. The published per-ring numbers therefore depend on these
files being present and current.

`.gitignore` excludes them (`*.parquet`, unanchored), so a fresh clone
has the code and `metrics.json` but not the evidence. Regenerate with:

```bash
python -m ml.early        # rebuilds the OOF store and metrics.json
python -m ml.publish_all  # rebuilds ml/out/, including the timeline
```

Tests that read these artefacts **skip** when they are missing, so a
clone stays green. That is the wrong default for CI, where a missing
artefact means the pipeline did not run and a skip is a silent pass — so
CI must set:

```bash
MULENET_REQUIRE_OOF=1 python -m pytest ml/
```

which turns every one of those skips into a failure.

## The same features in BigQuery, and what that is for

`ml/bq_features.py` computes twelve of the sixteen per-account features a
second time, as BigQuery SQL over the 572,890-row `transfers` table, for
both scopes. **`ml/features.py` stays canonical.** Every published number
— `docs/claims.md`, `ml/out/`, the deck — comes from the pandas path. The
SQL exists to show the definitions are portable and to disagree loudly if
they ever drift; nothing reads its output, and if the two ever disagree
the pandas answer is the one that ships.

The account dimension (`hashed_account_id`, `bank_id`, `opened_at`, and
the per-account `as_of`) travels as **query parameters**, three parallel
30,000-element arrays, not as a table. BigQuery holds only `transfers`:
`infra/load_bigquery.py`'s `SOURCES` loads no `accounts`, `devices`,
`kyc_submissions` or `kyc_documents`, and `shared/bq_schemas.py` defines
no schema for them.

### What reproduces, measured on all 30,000 accounts

| | |
|---|---|
| exact, zero delta | all six count features, `passthrough_speed_seconds`, `round_amount_ratio`, `account_age_days` — both scopes |
| exact in SQL, inexact in pandas | `amount_in_total`, `amount_out_total` |
| float tolerance, derived | `passthrough_ratio`, max abs delta 1.14e-13 |
| not reproducible at all | `device_count`, `shared_device_count`, `document_reuse_count`, `document_reuse_flag` |

Two cutoff scenarios are run, because one of them proves nothing on its
own: with the full-window cutoff every transfer is visible and the
point-in-time filter is a no-op. The second gives accounts 24 different
cutoffs, so only 288,420 of 572,890 transfers are visible, and every
count still matches exactly.

`passthrough_speed_seconds` and `account_age_days` are bit-exact because
the SQL reproduces pandas' *arithmetic* rather than an improved version
of it. `Timedelta.total_seconds()` is int64 nanoseconds divided by 1e9,
and at five years the nanosecond count exceeds 2**53, so that conversion
rounds; dividing integer seconds by 86400 instead would be more accurate
and would differ from the canonical path in the last few ulps. The SQL
takes the nanosecond route on purpose.

### The money columns: the canonical path is the inexact one

`SUM(amount)` over a BigQuery `NUMERIC` column and a `Decimal` sum are
both exact, so a 6e-11 disagreement means a float got involved. It did,
and on the pandas side: `ml.features._edge_events` does
`transfers["amount"].astype("float64")` before summing, so every
`NUMERIC(15,2)` value is rounded to a double first. The error is in the
per-value cast, not in the summation order — `math.fsum` over the same
cast floats reproduces the pandas answer, while the exact `Decimal` sum
does not.

Located by a third, all-`Decimal` aggregation
(`bq_features.exact_decimal_totals`), which never touches a float:

| | mismatched accounts | max abs error |
|---|---|---|
| BigQuery `SUM(NUMERIC)` vs exact `Decimal` | **0** of 30,000, all four columns, both scenarios | 0 |
| pandas `float64` vs exact `Decimal` | 2,673 – 6,171 | $5.8e-11 |

This is **not** fixed here and the canonical path is **not** changed. A
$6e-11 error on a cent-quantised column changes no decision — one cent is
the smallest money difference that can — and rewriting `features.py` to
match a second implementation would move published numbers to chase a
check. It is recorded instead, and `ml/test_bq_features.py` asserts the
size of it, so the day the cast goes the test says so.

### The leakage check matters more than the equality check

A point-in-time leak does not appear as a mismatch. It appears as a
*better* feature, and if both implementations read the same future rows
they agree with each other while both being wrong. So
`python -m ml.bq_features --leakage` holds the query fixed and changes
the data: with every cutoff at the midpoint of the window, it runs the
same SQL against `transfers` unmodified, against `transfers` with every
post-cutoff row deleted, and against `transfers` with every post-cutoff
row rewritten (amount x7, direction reversed) *and* duplicated. All three
must produce identical feature rows. A control run then moves the cutoffs
to the end of the window and must produce *different* rows — without it,
the two equalities would also hold for a query that never read
`transfers`.

### Cost and latency, measured

Each query scans 127,181,580 bytes (121.29 MiB) of the table — which
BigQuery reports as 133,370,235 logical bytes across 572,890 rows; the
scan is smaller because only six of the seven columns are read — and is
billed 127,926,272 bytes, so **$0.00073 per query** at the on-demand
$6.25/TiB rate — and the first 1 TiB each month is free. The job itself
takes **92–116 s**, with another ~90 s of wall clock for the 3 MB
parameter upload and the 30,000-row download. That is far above the 1–3 s
a scan-and-aggregate of this size would take: the query is not scan-bound
but shuffle-bound. It joins 572,890 rows twice against an unindexed
30,000-element `UNNEST`, unions the result into 1.15 M endpoint rows, and
then runs two `OVER (PARTITION BY account ORDER BY ...)` windows, four
exact `COUNT(DISTINCT ...)`s and a `PERCENTILE_CONT` over them —
about 1,000 slot-seconds. Caching is disabled on purpose: a cached query
measures nothing.

The leakage variants are far heavier — 503 s and 562 s, 10.2 M and 19.4 M
slot-ms — because filtering the source on a subquery over the parameter
array defeats partition-free pruning and the corrupted variant unions
three passes over `transfers` before the aggregation. Bytes processed and
billed stay at the same 127,181,580 / 127,926,272 for all four, so cost
does not move with that: **$0.0029 for the whole leakage check**. The full
verification to date — two parity runs and one leakage run, twelve
queries — is under a cent.

### Why the output is not written to the `features` table

`features` exists in `TABLE_SCHEMAS` and is created empty. It is not
written to, and the mismatch is shape as much as content:

- `FEATURES_SCHEMA` is **long** — one row per `(hashed_account_id, as_of,
  scope)` — while `features.parquet` is **wide**, both scopes side by side
  on one row, because that is what the model consumes. Publishing needs a
  melt, which is a transformation of the artefact rather than a load of it.
- `FEATURES_SCHEMA` has no column for `unique_in_counterparties`,
  `unique_out_counterparties`, `device_count` or `document_reuse_flag`, so
  a load either drops four features silently or fails.

Both are `shared/` decisions, so nothing is forced here.

## Standing discipline

- Per-account cutoffs everywhere; point-in-time device and KYC links.
- `account_age_days` is measured against the cutoff, never the horizon.
- `ring_bank_count` is stratification and reporting **only**; it must
  never enter a feature, and `test_leakage.py` asserts it.
- Any metric above 0.99 gets an explicit verdict: (a) genuine,
  (b) artifact, or (c) possible leakage.
- pandas is canonical. A second implementation is a check on it, never a
  replacement for it, and never a reason to change a published number.
