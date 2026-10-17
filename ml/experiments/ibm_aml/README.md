# Lane D2: IBM AML HI-Small public benchmark

**Offline only.** Data lives in the gitignored `ml/experiments/public_benchmark/`
folder, never committed. Nothing here cites or reproduces any row from
that folder; all numbers below are aggregated metrics computed from it.

**Licence:** IBM Transactions for Anti-Money Laundering (AML), HI-Small,
CDLA-Sharing-1.0. Computational use is unrestricted (CDLA-Sharing-1.0
s3.5); publishing the Data or Enhanced Data triggers share-alike (s3).
This repo publishes computational Results only -- no IBM data, account
IDs, or account-level labels appear in any commit.

Dataset: IBM Transactions for Anti-Money Laundering (AML), HI-Small, by
Erik Altman (IBM),
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml ,
CDLA-Sharing-1.0. See the dataset page for the citation of the
accompanying paper.

## What this is

A public, synthetic transaction dataset released by IBM (`HI-Small`:
transactions, accounts, and a `patterns.txt` describing the laundering
rings the simulator planted), used here as an external check on
MuleNet's transfer-graph features and grouped-CV methodology — not as
a benchmark we are trying to beat, and not part of the submission's
claims.

**What differs from our own simulator:**
- 515,088 accounts across 30,470 "banks" (median 4 accounts/bank) vs.
  our 30,000 accounts across 3 banks — a completely different topology,
  closer to routing-code fragmentation than our 3-big-bank design.
- 15 currencies including Bitcoin, vs. our single currency.
- 18 days of transactions (Sep 1–18, 2022), 5,078,345 rows, base rate
  0.102% of transactions / 1.23% of accounts positive — close to our
  own 2.37% account-level rate, so the label isn't the issue here.
- No device or KYC data at all — `accounts.csv` has only
  `Bank Name, Bank ID, Account Number, Entity ID, Entity Name`. Every
  identity-link feature we use elsewhere in MuleNet is structurally
  absent from this dataset.
- No account-open timestamp — `account_age_days` has no source field.

## Time-ordered holdout: dropped

Accounts are active from day 1: median first activity is
**2022-09-01 00:28**, and only **1 of 515,088** accounts first appears
after Sep 15. A time-ordered holdout like our simulator's isn't
meaningful on this data — any cutoff either leaves the test side empty
or forces nearly everything into train (checked directly: a Sep 11
cutoff gives 21 wholly-test components against 90 forced-to-train
straddlers; Sep 13 gives zero wholly-test components). No cutoff was
substituted. The grouped-CV run below is the only split used.

## Split

Groups = connected components over all positive (`Is Laundering=1`)
transactions, nodes keyed by `(bank, account)`; accounts with no
positive transaction are each their own group. Committed in
`split.json` before any training. 1,843 positive components;
6,357 of 515,088 accounts are positive.

## Features used (and why some were dropped)

Scale-free, transfer-only, point-in-time (identical cutoff logic for
every account, no leakage across the cutoff):

`in_degree, out_degree, unique_in_counterparties, unique_out_counterparties, unique_counterparties, distinct_counterparty_banks, passthrough_ratio, passthrough_speed_seconds`

**Dropped by decision, not by necessity alone:**
- `amount_in_total` / `amount_out_total` — dropped from the main run.
  15 currencies make a raw cross-currency sum meaningless, and this
  isn't just the 1.2% of rows with an actual FX conversion: even
  same-unit face values of US Dollar, Yen and Bitcoin are not
  comparable magnitudes at all, so summing them is wrong for the other
  98.8% too. Kept only in the sensitivity run below.
- `round_amount_ratio` — dropped. "Multiple of 100" is a unit-dependent
  artefact across 15 currencies of wildly different scale; the feature
  silently changes meaning depending which currency an account mostly
  transacts in.
- `account_age_days` — dropped. No source field in `accounts.csv`.

No feature re-engineering was done after seeing results; this set was
fixed before training and used identically across all runs below.

## Results (3 seeds: 42, 43, 44, mean ± std)

1% FPR threshold set from the training part only (inner grouped CV for
trained models; training-fold negatives for the two unlabeled-graph
baselines). `realized_fpr` is what the held-out fold actually got at
that threshold — reported because a train-set threshold doesn't
reproduce exactly 1% out of fold.

| model | PR-AUC | recall @ ~1% FPR | realized FPR | false alerts / catch @80% recall |
|---|---|---|---|---|
| `xgboost_full` | 0.081 ± 0.001 | 0.111 ± 0.001 | 0.0104 | 25.1 ± 0.3 |
| `xgboost_first3` (first 3 days only) | 0.032 ± 0.0001 | 0.053 ± 0.002 | 0.0101 | 52.6 ± 0.1 |
| `xgboost_full` + raw amounts (sensitivity) | 0.090 ± 0.0004 | 0.128 ± 0.002 | 0.0107 | 23.7 ± 0.2 |
| `logistic_regression_full` | 0.051 ± 0.0003 | 0.087 ± 0.001 | 0.0097 | 37.3 ± 0.6 |
| `logistic_regression_first3` | 0.024 ± 0.0002 | 0.029 ± 0.0004 | 0.0098 | 54.1 ± 0.1 |
| `component_size_baseline` (full unlabeled graph) | 0.017 ± 0 | 0.998 ± 0 | **0.719** | 47.6 ± 0 |
| `degree_threshold_baseline` (full unlabeled graph) | 0.026 ± 0 | 0.036 ± 0.0001 | 0.0102 | 46.9 ± 0 |

**Sensitivity: including the raw, currency-mixed amount sums
(`xgboost_full` + raw amounts) moves PR-AUC from 0.081 to 0.090 and
recall from 0.111 to 0.128 — a real but modest improvement, not a
qualitative change.** The currency-mixing problem costs some signal
but doesn't dominate the result either way.

## The giant-component finding

The full transaction graph (all 515,088 accounts, no labels) collapses
into just **15 connected components**, one of which holds **72.2%** of
all accounts (372,089 of 515,088). Degree is heavily skewed: median 6,
99th percentile 119, max 169,756 (one extreme hub account). This is
why `component_size_baseline`'s 99.8% recall is **not a meaningful
result** — it reaches that recall by flagging almost everyone (71.9%
realized FPR): a dense real-world-scale transaction graph has one
dominant component almost by construction, so "is this account in a
big component" carries essentially no information at this density.
`degree_threshold_baseline` (flag by raw degree instead of component
membership) avoids that collapse and lands close to the logistic
regression baseline.

## Not comparable to published GNN numbers on this dataset

Papers using this dataset typically use edge/graph-level GNN models
with the full feature set (amounts, timestamps at native resolution,
multi-hop structure) and their own split protocol. This run uses a
reduced, scale-free, account-level feature set by explicit choice (see
above), a different grouped-CV split, and a different task framing
(account-level "ever touches laundering" rather than edge-level
classification). **Nothing here should be read against a published
number on this dataset** — the setups aren't the same experiment.

## Comparison to our own synthetic results

| | IBM AML HI-Small (this experiment) | MuleNet's own simulator |
|---|---|---|
| best PR-AUC | 0.090 (raw-amount sensitivity) / 0.081 (main features) | 0.62 (T=5m) – 0.99 (T=180m) |
| recall @ ~1% FPR | 0.11–0.13 | 0.62 (T=5m) – 0.99 (T=180m) |

The gap is large and expected, not a sign our own numbers are
suspect: our simulator's rings are generated by the same rule-evasion
logic our features are built to detect, at much lower volume (30k
accounts, 3 banks) with device/KYC identity features available. IBM's
data is a different, much larger, much sparser-labelled, identity-free
problem. This experiment is a sanity check that our feature family
produces *some* signal (PR-AUC 5-9x the 0.0102 base rate) on an
independent dataset we had no hand in generating — not a claim that
MuleNet's shipped numbers transfer to it.

## What this does NOT show

- Not a claim about MuleNet's production detector or any number in
  `docs/claims.md` — this is a separate, offline sanity check on a
  different dataset.
- Not comparable to this dataset's own published benchmarks (different
  features, split, and task framing — see above).
- The component-size baseline's headline recall (99.8%) is explicitly
  **not** evidence of a working detector — see the giant-component
  finding.
- Amount-based features were dropped for the main run by an explicit
  currency-comparability decision, not because they carry no signal
  (the sensitivity run shows they help some).
- No multi-bank coalition test was run (skipped by decision; noted as
  roadmap below), despite 38% of entities spanning more than one bank
  in this data.
- Licence is CDLA-Sharing-1.0 (confirmed, see top of this file); for the
  accompanying paper's citation, see the dataset page linked there.

## Roadmap (not pursued in this round)

- Multi-bank coalition test: 63,362 of 166,207 entities (38%) span
  more than one of the 30,470 banks via shared `Entity ID` — a
  genuine cross-bank identity link, structurally different from our
  3-bank coalition sweep (no natural "N banks" lever here; the natural
  test is binary, single-bank-visible vs. entity-linked features).
  Skipped this round by decision.
