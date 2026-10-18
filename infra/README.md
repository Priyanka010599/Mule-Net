# infra

Google Cloud setup and deploy scripts, least-privilege IAM, container
builds, and the Cloud SQL script that is kept ready but unprovisioned.
Owned by the **platform-engineer** agent. Everything is parameterized by
`PROJECT_ID` and `REGION`; no project IDs, keys, or secrets are ever
committed here.

`setup.sh` also grants the IAM roles the rest of the stack depends on:
**Secret Manager** secret access (read-only, per service account), the
**Cloud KMS** and **Compute Engine** permissions the live-proof enclave
needs to create/delete **Confidential Space** VMs and release keys, and
the **Cloud Logging** read access that makes a run's own audit trail
checkable from the live app.

| File | What it is |
|---|---|
| `setup.sh` | One-time project setup: 10 APIs, Artifact Registry repo, one least-privilege service account per service with IAM. Already run. |
| `deploy.sh` | The deploy. GCS bucket, BigQuery load, Pub/Sub topic + dead-letter + push subscription, Cloud Run for `ingest` and `ui`. Idempotent. |
| `cloudsql_deploy.sh` | Would provision Cloud SQL, apply `shared/ddl.sql` and load it. **Never executed** — see `docs/deployment.md` §2. |
| `load_bigquery.py` | Creates the dataset and tables from `shared/bq_schemas.py` `TABLE_SCHEMAS` and loads the published Parquet. Restates each file against the contract first. |
| `fetch_data.py` | Container-start helper: pulls a service's data files out of the bucket and verifies each md5. |
| `cloudbuild.yaml` | Builds one service image, parameterized by `_SERVICE`. |
| `cloudbuild.ignore` | Keeps the ~175 MB of generated data out of the build upload. |
| `Dockerfile.ingest`, `Dockerfile.ui` | Runtime images, Python 3.11. |

## Usage

```bash
PROJECT_ID=your-project-id REGION=us-central1 ./infra/setup.sh   # once
PROJECT_ID=your-project-id REGION=us-central1 ./infra/deploy.sh  # every deploy
```

`SKIP_DATA=1` skips the Cloud Storage upload and the BigQuery load, which
is what you want when only the code changed.

## Why the Dockerfiles live here and not in each service folder

Every service image needs its own folder **plus** `shared/`, so the build
context has to be the repo root either way. Putting the three Dockerfiles
together in `infra/` keeps that decision in one place, keeps the service
folders owned by their own engineers, and avoids three near-identical
Dockerfiles at the repo root. `cloudbuild.yaml` selects one with
`-f infra/Dockerfile.${_SERVICE}`.

Images are built by Cloud Build, not locally, so no Docker daemon is
needed on the operator's machine.

## No Dockerfile for agents

`agents/` has no HTTP entrypoint — no FastAPI app, no `uvicorn` call, no
`__main__` outside the eval runner — so there is nothing for a Cloud Run
service to listen with, and `ui/chat.py` cannot call one anyway. Writing a
Dockerfile for it would be speculative. See `docs/deployment.md` §4.
