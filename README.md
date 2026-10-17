# MuleNet: catch money-mule rings across banks, without sharing customer data

## In plain words

Criminals move stolen money through ordinary bank accounts (called **money
mules**) so it is hard to trace. A group of linked mules is a **ring**, and
rings are often spread across several banks. Each bank sees only its own small
piece, so nothing looks wrong to any single bank.

MuleNet lets the banks' views be joined so the whole ring shows up, while no
bank, and not even us as the operator, can read another bank's customers. The
joining runs inside a sealed Google Cloud environment (Confidential Space).

Our main data is **synthetic** (we generated it, no real customers). We also
**tested on IBM's public AML dataset (HI-Small)**, which we did not generate:
see [Also tested on IBM's public dataset](#also-tested-on-ibms-public-dataset).

## At a glance

| | |
|---|---|
| **What it does** | Finds mule rings from transfers, explains why, drafts a compliance report |
| **Why joining helps** | Rings split across banks: ring recall 92.5% alone, 100% joined (at 3 hours) |
| **Fewer wasted reviews** | 6.23 false alerts per real mule caught alone, 1.97 joined (at 5 minutes) |
| **Privacy** | Banks share only who-paid-whom links across banks, inside a sealed enclave |
| **AI used** | Gemini on Vertex AI (explanations, reports, Q&A) and XGBoost (scoring) |
| **Data** | 30,000 accounts, 573,000 transfers, 120 hidden rings, 3 banks |
| **Independent check** | Also tested on IBM's public AML dataset (HI-Small, 515,088 accounts): PR-AUC 0.0813, about 5 to 9 times the base rate |
| **Honest limit** | 87.8% of mule accounts flagged, but rings found end to end: 44% (see below) |
| **Runs in** | Google Cloud, `asia-south1` (Mumbai) |

## Quick tour (5 minutes, for everyone)

1. **One bank's view** (tab 1). A single bank sees only a fragment of a ring.
2. **The ring appears** (tab 2). Join the three banks and the full ring shows:
   source, mules, cash-out. Press *Play replay* to watch the money move.
3. **What joining buys** (tab 3). Share of rings the rule catches: 60.3% with
   2 banks, 100% with all 3.
4. **Run live proof** (tab 3, enclave card). Starts a real sealed VM. Google's
   key service hands over the key only after the VM proves what code it runs.
   A wrong key fails, the right key works. About 2 to 3 minutes.
5. **Try tampered code** (same card). Change one line of the code and Google
   refuses to release the key. Banks approve one exact version and the cloud
   enforces it, even against us.
6. **Ask Gemini.** Click a ring, then *Report* for a drafted suspicious-activity
   report that cites real RBI, PMLA and FIU-IND pages (claims with no evidence
   are removed). Use *Ask* for plain-English questions about the data or about
   Indian AML rules.
7. **Score your own data** (tab 5). Upload one CSV per bank
   (`from_account,to_account,amount,timestamp,channel`) and it is scored live.

Good to know: the live proof is limited to 10 runs per 24 hours (each is a real
paid VM). After that the card shows the latest verified run.

## How it works, step by step

1. **Each bank looks alone.** It sees only its own customers and transfers.
2. **The banks are joined.** Who paid whom, shared devices, and the same ID
   document used twice are combined, so scattered pieces become one ring.
3. **Two checks decide who is suspicious.** A simple **speed rule** flags an
   account that many accounts pay into and that passes the money on quickly.
   A trained **model** (XGBoost) scores every account from similar patterns.
   Connected flagged accounts are grouped into rings, with reasons listed.
   (Exact features per model and the rule's thresholds:
   [`docs/technical_overview.md`](docs/technical_overview.md#model-features-and-detection-rule).)
4. **Identity is evidence, never the trigger.** A shared phone or reused ID
   never raises an alert by itself, because families share phones. It only
   strengthens a ring the transfers already pointed at.
5. **Gemini explains, a checker verifies.** Gemini writes the explanation and
   report. Plain Python code then recomputes every number in it. If a claim does
   not check out, the whole answer is withheld.
6. **Privacy while joining.** Banks send only cross-bank links, encrypted to the
   sealed enclave. The key is released only to the exact approved code, and each
   bank gets alerts about its own accounts only.

## What we found

- **Joining helps exactly where rings are split.** For rings inside one bank it
  helps nothing (slightly negative, which is the sanity check). For rings split
  over 2 or 3 banks it helps clearly. That pattern is why we believe the result.
- **The gain is about coverage, not speed.** Joined banks catch rings a single
  bank never catches in time, rather than the same rings sooner.
- **Every number is checked.** The figures live in
  [`docs/claims.md`](docs/claims.md), generated from the experiment outputs, and a
  test fails if any of them drifts.

## Also tested on IBM's public dataset

> **Our own data is synthetic, but we did not stop there.** We ran the same
> kind of test on **IBM's public AML dataset (HI-Small)**, a different dataset
> built by someone else, to check that our method finds real signal and is not
> just fitting our own simulator.

- **How different it is:** 515,088 accounts spread over about 30,000 tiny
  "banks" (not our 3 big ones), 15 currencies, 18 days of transfers, and **no
  device or ID-document data at all**. We could only use transfer-pattern
  features.
- **Result:** the reduced transfer-only model scored **PR-AUC 0.0813**, about
  **5 to 9 times the base rate** of laundering accounts (3 seeds, cross-validation
  that keeps whole connected groups of accounts together so nothing leaks).
- **A useful negative finding:** "is this account in a big connected cluster?"
  is not a detector there. 72% of all accounts sit in one cluster, so cluster
  membership alone says almost nothing.
- **How to read it honestly:** this is a sanity check, not a score to compare
  with other models on that dataset (different features, split and task), and
  it does not show the result carries over to real banks. The IBM data is not
  in this repo (licence); only the aggregate results are.

Details: the Metrics tab (tab 3) in the app, [`docs/claims.md`](docs/claims.md)
(`ibm_aml_xgboost_pr_auc`) and `ml/experiments/ibm_aml/README.md`.

## Words used in the app

- **Single-bank / cross-bank:** a bank scoring alone vs banks joined.
- **PR-AUC:** one score from 0 to 1 for how well real mules are ranked at the top
  of the suspicion list. Good for rare events like mules. The Metrics tab
  explains it with an example.
- **False alerts per catch:** innocent accounts flagged for each real mule found.
- **Ring recall:** share of true rings flagged.
- **Enclave:** a sealed, attested cloud computer nobody can look inside.

## What is real and what is not

We state limits plainly:

- **The data is synthetic.** Our main numbers describe a simulator we wrote, not
  real banks. The IBM test above is also synthetic data (IBM's). Real-world
  accuracy is unknown until a pilot with real banks.
- **Rings found are fewer than accounts found.** 87.8% of mule accounts are
  flagged, but the rings built from them reach only 53 of 120 true rings (44%),
  because a flagged account joins a ring only when the account it pays on is
  also flagged.
- **Fraudsters can adapt.** Gemini designed 3 patterns that beat the rule and
  both trained models. Retraining against them was inconclusive, so it is not
  shipped.
- **The detectors are supervised.** They learn this dataset's mule pattern. An
  unsupervised Isolation Forest was tried on one stress pattern, trailed XGBoost,
  and is not deployed.
- **Chat answers are not fact-checked.** The checker verifies investigations and
  reports, not free-form chat.
- **The regulation index has a gap.** It is Indian statute text, so it has
  nothing on transfer-pattern typologies. Reports drop grounds they cannot cite.
- **Not production-ready.** Next steps include retiring the shared demo hashing
  key and a real-data pilot.

## Built on Google Cloud

**Live and on the serving path:** Gemini on Vertex AI, Vertex AI Search
(RBI/PMLA/FIU-IND regulation index), BigQuery (the agents service's data
source as of 2026-10-15 -- `accounts`, `transfers`, `rings` and 8 other
tables, behind `MULENET_AGENTS_DATA_SOURCE=bigquery`; a parquet fallback
still exists), Pub/Sub, Cloud Run, Confidential Space, Cloud KMS, Secret
Manager, Cloud Storage, Cloud Build, Artifact Registry.
**Live, but not yet on the serving path:** Vertex AI Model Registry (a
lineage record only, the model is not servable) and BigQuery's own
`features` table specifically (still not published, unlike the 9 tables
above).
**Built, not yet connected to the live app:** Document AI KYC parser
(validated offline on 50 synthetic documents).

## Run it yourself

Python 3.11. Everything is seeded.

```bash
pip install -e ".[dev]"
python -m simulator.generator      # make the synthetic data
python -m ml.publish_all           # build the model outputs
python -m ml.claims                # regenerate docs/claims.md
python -m pytest                   # whole test suite, offline and free
```

Deploying: `infra/deploy.sh` (see [`docs/deployment.md`](docs/deployment.md)).

## Where things are

| folder | what is in it |
|---|---|
| `simulator/` | the synthetic banks and data |
| `ml/` | features, models, rings, metrics |
| `agents/` | the Gemini agents, checker and API |
| `ui/` | the web app |
| `enclave/` | the sealed cross-bank join |
| `docai/` | KYC parsing, regulation search |
| `ingest/`, `infra/` | data pipeline, deployment |
| `docs/` | claims ledger, deployment, enclave and privacy design |

## Go deeper

- [`docs/technical_overview.md`](docs/technical_overview.md): the detailed README
  (architecture diagram, every Google Cloud service, full limitations, lessons)
- [`docs/claims.md`](docs/claims.md): every quotable number and its source
- [`docs/data_sharing.md`](docs/data_sharing.md) and [`docs/enclave.md`](docs/enclave.md): how banks share without sharing
- [`docs/residency.md`](docs/residency.md): why everything runs in Mumbai

No secrets, no keys, no real data. Account identifiers are hashed everywhere,
including in logs. Licence: [Apache 2.0](LICENSE).
