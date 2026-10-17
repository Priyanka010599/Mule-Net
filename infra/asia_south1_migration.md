# asia-south1 migration — parallel stack

Goal: move MuleNet's data and services to `asia-south1` (Mumbai) without
breaking the running `us-central1` demo. Strategy: build a parallel
`asia-south1` stack, verify it end to end, then (separately, after a
soak period and explicit sign-off) retire the `us-central1` one.

This file records what the inventory found and what the new stack
actually is, so the move is reproducible from a clean checkout rather
than only from shell history.

## Inventory: what was already in asia-south1

Before this migration, most of the stack was **already** there:

| Resource | Location |
|---|---|
| Enclave VM zone | `asia-south1-a` |
| KMS keyring `mulenet-enclave` | `asia-south1` |
| Enclave buckets (inbox/outbox/keys/evidence) | `asia-south1` (all 4) |
| Workload Identity Pool | `global` (required by Confidential Space) |
| `mulenet-agents` Cloud Run | `asia-south1` |
| Agents data mirror bucket (`mulenet-mulenet-data-asia`) | `asia-south1` |
| Artifact Registry `mulenet` (agents images) + `mulenet-enclave` | `asia-south1` |
| Vertex AI Model Registry (model_a, model_b, xgboost-cb-oof-v1) | `asia-south1` |
| Secret Manager secrets | automatic/global replication, not region-pinned |
| Pub/Sub topics | global resource; only the push *endpoint* is regional |
| Vertex AI Search engine | `global`, not regional |

**Nothing in that list needed to move.** The KMS key location, the
digest-pinned IAM bindings, and the tampered-image negative test were
never touched by this migration, because the enclave was never going to
move — it was already home.

## What actually moved: `infra/deploy_asia_south1.sh`

Still-`us-central1` resources, and what the script does about each:

| Resource (us-central1) | New resource (asia-south1) | How |
|---|---|---|
| GCS bucket `mulenet-mulenet-data` | `mulenet-mulenet-data-as1` | New bucket (bucket location is immutable — this is a create, not a move); holds only `ui/out/demo_snapshot.json`, the one object the `ui` service fetches |
| BigQuery dataset `mulenet.mulenet` | `mulenet.mulenet_as1` | New dataset (also immutable location), loaded via `infra/load_bigquery.py` with `BQ_DATASET=mulenet_as1 REGION=asia-south1` |
| Cloud Run `mulenet-ui` (us-central1) | `mulenet-ui` (asia-south1) | Same service *name*, different region — Cloud Run keys services by (name, region), so this is a genuinely separate resource, not a rename |
| Cloud Run `mulenet-ingest` (us-central1) | `mulenet-ingest` (asia-south1) | Same |
| Artifact Registry `mulenet` repo (us-central1, ingest/ui images) | existing `mulenet` repo in `asia-south1` (already created for the agents image) | `gcloud builds submit --region=asia-south1` |
| Pub/Sub push subscription `mulenet-transfers-sub` | repointed, not duplicated | Pub/Sub subscriptions are global — there is no way to have both regions receiving live push traffic off one subscription at once, so this is a repoint to the new ingest URL, not a second subscription |

**IAM:** `mulenet-ui`'s service account needed one new grant
(`roles/storage.objectViewer` on the new bucket). `mulenet-ingest`,
`mulenet-ui` and `mulenet-agents` already hold `roles/bigquery.dataEditor`
/ `dataViewer` / `jobUser` at the **project** level (`infra/setup.sh`),
which covers any dataset in the project including the new one — no new
BigQuery grant was needed.

**`mulenet-agents` was not redeployed.** The new `mulenet-ui` (asia-south1)
points at the *existing* `mulenet-agents` service, which was already in
`asia-south1` — so this call becomes same-region instead of cross-region,
a latency improvement that came free with the move.

## Live-proof tamper testing

`MULENET_UI_LIVE_PROOF_TAMPER` defaults to `"0"` in both
`infra/deploy.sh` and `infra/deploy_asia_south1.sh` (nobody had set
`LIVE_PROOF_TAMPER=1` on either stack as of this migration). To exercise
the tamper-test verification step, it was enabled on the new stack via
`gcloud run services update mulenet-ui --region=asia-south1
--update-env-vars=MULENET_UI_LIVE_PROOF_TAMPER=1`. This is a flag on the
public `ui` service, not a change to the enclave or its IAM policy.

## Running it

```bash
PROJECT_ID=mulenet ./infra/deploy_asia_south1.sh
```

Safe to re-run (every step is describe-or-create). Does not touch
`infra/deploy.sh`'s resources, `mulenet-agents`, or anything
enclave-related.

## New resource names (asia-south1)

- Bucket: `mulenet-mulenet-data-as1`
- BigQuery dataset: `mulenet.mulenet_as1`
- Cloud Run: `mulenet-ui`, `mulenet-ingest` (region `asia-south1`)
- Artifact Registry: existing `mulenet` repo, `asia-south1`
