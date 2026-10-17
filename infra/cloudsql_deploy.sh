#!/usr/bin/env bash
#
# MuleNet -- provision the OPERATIONAL store on Cloud SQL for PostgreSQL.
#
# THIS SCRIPT HAS NEVER BEEN RUN. It is kept ready and unprovisioned on
# purpose: the submitted stack is Pub/Sub + BigQuery only, because an
# always-on Cloud SQL instance is the single largest idle cost in this
# project and nothing in the demo path needs a row-by-row operational read.
# See docs/deployment.md for the full argument and for what switching over
# would cost in engineering, not just in dollars.
#
# What it would do:
#   1. create a Cloud SQL for PostgreSQL instance, database and user, with
#      the password generated into Secret Manager and never printed
#   2. apply shared/ddl.sql -- the operational schema (accounts, transfers,
#      devices, KYC, rings, ring_members, cases, str_reports,
#      account_evidence). Model-run output stays in BigQuery by design; see
#      the "WHAT IS DELIBERATELY NOT HERE" comment at the top of that file
#   3. load accounts with the existing idempotent loader, so replayed
#      transfers satisfy their foreign keys
#   4. point the deployed ingest service at it
#
# Nothing here is novel SQL: step 3 reuses ingest/local_dev.py bootstrap-db,
# and ingest writes transfers through ingest/sinks.py PostgresSink, whose
# statement is "INSERT ... ON CONFLICT (transfer_id) DO NOTHING". That is
# what makes a re-run and a Pub/Sub redelivery both safe.
#
# Usage:
#   PROJECT_ID=your-project-id REGION=us-central1 ./infra/cloudsql_deploy.sh
#
# Required environment variables:
#   PROJECT_ID  Google Cloud project to provision in.
#   REGION      Region for the instance. Keep it equal to the Cloud Run
#               region: a cross-region hop is charged and slow.
#
# Optional:
#   INSTANCE    Instance name.  Defaults to mulenet-pg.
#   DB_NAME     Database name.  Defaults to mulenet.
#   DB_USER     Database user.  Defaults to mulenet.
#   TIER        Machine tier.   Defaults to db-g1-small.
#   BUCKET      Bucket used to stage ddl.sql for import. Defaults to
#               ${PROJECT_ID}-mulenet-data, the bucket infra/deploy.sh makes.
#
# Safe to re-run: describe-or-create throughout, the DDL is
# CREATE TABLE IF NOT EXISTS, and the account load is an upsert.

set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID is not set. Run: PROJECT_ID=your-project-id REGION=us-central1 ./infra/cloudsql_deploy.sh}"
: "${REGION:?REGION is not set. Run: PROJECT_ID=your-project-id REGION=us-central1 ./infra/cloudsql_deploy.sh}"

INSTANCE="${INSTANCE:-mulenet-pg}"
DB_NAME="${DB_NAME:-mulenet}"
DB_USER="${DB_USER:-mulenet}"
TIER="${TIER:-db-g1-small}"
BUCKET="${BUCKET:-${PROJECT_ID}-mulenet-data}"

SECRET_PASSWORD="mulenet-pg-password"
SECRET_DSN="mulenet-pg-dsn"
SA_INGEST="mulenet-ingest"
SVC_INGEST="mulenet-ingest"

sa_email() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

echo "==> MuleNet Cloud SQL: project=${PROJECT_ID} region=${REGION} instance=${INSTANCE}"

# ---------------------------------------------------------------------------
# 0. API
# ---------------------------------------------------------------------------
echo "==> Enabling sqladmin.googleapis.com..."
gcloud services enable sqladmin.googleapis.com --project="${PROJECT_ID}"

# ---------------------------------------------------------------------------
# 1. Password -- generated into Secret Manager, never echoed or passed on argv
# ---------------------------------------------------------------------------
# A password on a gcloud command line lands in shell history and in the
# audit log's request metadata, so it is generated locally, stored, and from
# then on only ever read back into a variable.
echo "==> Ensuring secret ${SECRET_PASSWORD}..."
if gcloud secrets describe "${SECRET_PASSWORD}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, reusing."
else
  gcloud secrets create "${SECRET_PASSWORD}" \
    --replication-policy=automatic --project="${PROJECT_ID}"
  python -c "import secrets; print(secrets.token_urlsafe(32), end='')" \
    | gcloud secrets versions add "${SECRET_PASSWORD}" \
        --data-file=- --project="${PROJECT_ID}"
  echo "    Created and seeded."
fi
DB_PASSWORD="$(gcloud secrets versions access latest \
  --secret="${SECRET_PASSWORD}" --project="${PROJECT_ID}")"

# ---------------------------------------------------------------------------
# 2. Instance, database, user
# ---------------------------------------------------------------------------
# Public IP with no authorized networks: nothing on the internet can reach
# it. Cloud Run connects over the Cloud SQL connector, which does not need an
# authorized network, so the instance stays unreachable from outside Google.
echo "==> Ensuring instance ${INSTANCE} (this takes ~10 minutes on create)..."
if gcloud sql instances describe "${INSTANCE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud sql instances create "${INSTANCE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --database-version=POSTGRES_15 \
    --tier="${TIER}" \
    --storage-size=10GB \
    --storage-auto-increase \
    --availability-type=ZONAL \
    --backup-start-time=03:00
  echo "    Created."
fi

CONNECTION_NAME="$(gcloud sql instances describe "${INSTANCE}" \
  --project="${PROJECT_ID}" --format='value(connectionName)')"
echo "    connection name: ${CONNECTION_NAME}"

echo "==> Ensuring database ${DB_NAME}..."
if gcloud sql databases describe "${DB_NAME}" --instance="${INSTANCE}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud sql databases create "${DB_NAME}" \
    --instance="${INSTANCE}" --project="${PROJECT_ID}"
  echo "    Created."
fi

echo "==> Ensuring user ${DB_USER}..."
if gcloud sql users list --instance="${INSTANCE}" --project="${PROJECT_ID}" \
     --format='value(name)' | grep -qx "${DB_USER}"; then
  echo "    Already exists, resetting password to the stored secret."
  gcloud sql users set-password "${DB_USER}" \
    --instance="${INSTANCE}" --project="${PROJECT_ID}" \
    --password="${DB_PASSWORD}"
else
  gcloud sql users create "${DB_USER}" \
    --instance="${INSTANCE}" --project="${PROJECT_ID}" \
    --password="${DB_PASSWORD}"
  echo "    Created."
fi

# ---------------------------------------------------------------------------
# 3. Schema -- shared/ddl.sql, imported from the bucket
# ---------------------------------------------------------------------------
# Imported rather than piped through `gcloud sql connect`, which needs an
# interactive psql and an authorized network. The instance's own service
# account must be able to read the staged object.
echo "==> Staging shared/ddl.sql and importing it into ${DB_NAME}..."
gcloud storage cp shared/ddl.sql "gs://${BUCKET}/infra/ddl.sql" --project="${PROJECT_ID}"

INSTANCE_SA="$(gcloud sql instances describe "${INSTANCE}" \
  --project="${PROJECT_ID}" --format='value(serviceAccountEmailAddress)')"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${INSTANCE_SA}" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT_ID}" --quiet >/dev/null

gcloud sql import sql "${INSTANCE}" "gs://${BUCKET}/infra/ddl.sql" \
  --database="${DB_NAME}" --project="${PROJECT_ID}" --quiet
echo "    Schema applied."

# ---------------------------------------------------------------------------
# 4. Accounts -- the existing idempotent loader
# ---------------------------------------------------------------------------
# transfers has a foreign key onto accounts in shared/ddl.sql, so the
# accounts must land before any transfer can be written. This reuses
# ingest/local_dev.py bootstrap-db, which applies the DDL (a no-op now) and
# upserts simulator/out/accounts.parquet. Needs the Cloud SQL Auth Proxy
# running locally, because the loader connects with psycopg over a DSN:
#
#   cloud-sql-proxy --port 5432 "${CONNECTION_NAME}" &
#
# Left as an explicit operator step rather than backgrounded here: a proxy
# this script starts and kills would hide a connection failure as a load
# failure, and a half-loaded accounts table is the confusing state to be in.
DSN="postgresql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}"
echo "==> Loading accounts through the idempotent upsert loader..."
if python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',5432))==0 else 1)"; then
  python -m ingest.local_dev bootstrap-db \
    --dsn "${DSN}" \
    --accounts simulator/out/accounts.parquet
else
  echo "    SKIPPED: nothing listening on 127.0.0.1:5432."
  echo "    Start the proxy and re-run this script:"
  echo "      cloud-sql-proxy --port 5432 ${CONNECTION_NAME}"
fi

# ---------------------------------------------------------------------------
# 5. Store the DSN and point ingest at the instance
# ---------------------------------------------------------------------------
# The DSN carries the password, so it lives in Secret Manager and reaches the
# container as a mounted secret -- never as a --set-env-vars value, which is
# readable by anyone who can describe the service.
echo "==> Ensuring secret ${SECRET_DSN}..."
if ! gcloud secrets describe "${SECRET_DSN}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud secrets create "${SECRET_DSN}" \
    --replication-policy=automatic --project="${PROJECT_ID}"
fi
# Over the Cloud SQL connector the host is the unix socket Cloud Run mounts.
printf '%s' \
  "postgresql://${DB_USER}:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${CONNECTION_NAME}" \
  | gcloud secrets versions add "${SECRET_DSN}" \
      --data-file=- --project="${PROJECT_ID}"

echo "==> Granting the ingest SA access to the DSN secret and to Cloud SQL..."
gcloud secrets add-iam-policy-binding "${SECRET_DSN}" \
  --member="serviceAccount:$(sa_email "${SA_INGEST}")" \
  --role="roles/secretmanager.secretAccessor" \
  --project="${PROJECT_ID}" --quiet >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:$(sa_email "${SA_INGEST}")" \
  --role="roles/cloudsql.client" \
  --condition=None --quiet >/dev/null

# This is the switch-over: naming the postgres sink explicitly overrides the
# "noop" default that ingest/config.py applies in the cloud.
echo "==> Repointing ${SVC_INGEST} at Cloud SQL..."
gcloud run services update "${SVC_INGEST}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --add-cloudsql-instances="${CONNECTION_NAME}" \
  --update-env-vars="MULENET_INGEST_OPERATIONAL_SINK=postgres" \
  --update-secrets="MULENET_INGEST_PG_DSN=${SECRET_DSN}:latest"

echo
echo "==> Done."
echo "    instance:    ${INSTANCE} (${CONNECTION_NAME})"
echo "    database:    ${DB_NAME}"
echo "    dsn secret:  ${SECRET_DSN}"
echo
echo "    NOTE: the ingest image built by infra/cloudbuild.yaml does NOT"
echo "    install psycopg, because the deployed stack has no Postgres."
echo "    Add psycopg[binary] to infra/Dockerfile.ingest and rebuild before"
echo "    running this, or the postgres sink will fail on its lazy import."
