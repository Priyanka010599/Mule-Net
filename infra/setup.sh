#!/usr/bin/env bash
#
# MuleNet — one-time Google Cloud project setup.
#
# Enables the APIs the project needs, creates one least-privilege service
# account per Cloud Run service (ingest, ui, agents), binds minimal IAM roles,
# and creates the Artifact Registry repo that Cloud Build pushes images to.
#
# Usage:
#   PROJECT_ID=your-project-id REGION=us-central1 ./infra/setup.sh
#
# Required environment variables:
#   PROJECT_ID  Google Cloud project ID to configure.
#   REGION      Region for Artifact Registry and the Cloud Run services
#               (must be a region where Vertex AI and AlloyDB are available).
#
# Safe to re-run: every step tolerates already-enabled APIs and existing
# resources. No project IDs, regions, keys, or secrets are hardcoded here.

set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID is not set. Run: PROJECT_ID=your-project-id REGION=us-central1 ./infra/setup.sh}"
: "${REGION:?REGION is not set. Run: PROJECT_ID=your-project-id REGION=us-central1 ./infra/setup.sh}"

REPO_NAME="mulenet"

SA_INGEST="mulenet-ingest"
SA_UI="mulenet-ui"
SA_AGENTS="mulenet-agents"

sa_email() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

echo "==> MuleNet setup: project=${PROJECT_ID} region=${REGION}"

# ---------------------------------------------------------------------------
# 1. APIs
# ---------------------------------------------------------------------------
# Enabling an already-enabled service is a no-op, so this is idempotent.
#
# Not enabled here, add when the matching feature lands:
#   discoveryengine.googleapis.com      Vertex AI Search (AML regulation corpus)
#   servicenetworking.googleapis.com    AlloyDB private-IP VPC peering
#   compute.googleapis.com              the VPC that peering attaches to
#   confidentialcomputing.googleapis.com  Confidential Space (stretch, enclave/)
APIS=(
  pubsub.googleapis.com            # transaction event bus
  bigquery.googleapis.com          # analytics store for scores and rings
  aiplatform.googleapis.com        # Vertex AI: Gemini, training, endpoints
  run.googleapis.com               # ingest / ui / agents services
  cloudbuild.googleapis.com        # build and deploy pipeline
  artifactregistry.googleapis.com  # container images for Cloud Run
  alloydb.googleapis.com           # operational graph store
  documentai.googleapis.com        # KYC document parsing
  cloudkms.googleapis.com          # keys for the Confidential Space stretch
  secretmanager.googleapis.com     # the only place secrets live
)

echo "==> Enabling ${#APIS[@]} APIs (this can take a couple of minutes)..."
gcloud services enable "${APIS[@]}" --project="${PROJECT_ID}"
echo "    APIs enabled."

# ---------------------------------------------------------------------------
# 2. Artifact Registry
# ---------------------------------------------------------------------------
echo "==> Ensuring Artifact Registry repo '${REPO_NAME}' in ${REGION}..."
if gcloud artifacts repositories describe "${REPO_NAME}" \
     --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Repo already exists, skipping."
else
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="MuleNet Cloud Run images" \
    --project="${PROJECT_ID}"
  echo "    Repo created."
fi

# ---------------------------------------------------------------------------
# 3. Service accounts
# ---------------------------------------------------------------------------
ensure_sa() {
  local account_id="$1" display_name="$2"
  echo "==> Ensuring service account '${account_id}'..."
  if gcloud iam service-accounts describe "$(sa_email "${account_id}")" \
       --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    Already exists, skipping."
  else
    gcloud iam service-accounts create "${account_id}" \
      --display-name="${display_name}" \
      --project="${PROJECT_ID}"
    echo "    Created."
  fi
}

# add-iam-policy-binding is idempotent; --condition=None keeps it
# non-interactive on policies that already contain conditional bindings.
bind_role() {
  local account_id="$1" role="$2"
  echo "    binding ${role}"
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:$(sa_email "${account_id}")" \
    --role="${role}" \
    --condition=None \
    --quiet >/dev/null
}

ensure_sa "${SA_INGEST}" "MuleNet ingest (Cloud Run consumer)"
ensure_sa "${SA_UI}"     "MuleNet UI (Cloud Run FastAPI + frontend)"
ensure_sa "${SA_AGENTS}" "MuleNet agents (Cloud Run ADK agents)"

# ---------------------------------------------------------------------------
# 4. IAM — one identity per service, smallest role that does the job
# ---------------------------------------------------------------------------
# NOTE: these bind at the project level because the BigQuery dataset and
# AlloyDB cluster do not exist yet. Once shared/ddl.sql has been applied,
# narrow the BigQuery roles to dataset-level bindings.

echo "==> Binding IAM roles for ${SA_INGEST}..."
# Consumes the transaction subscription. Subscriber can pull and ack but
# cannot publish, so a compromised consumer cannot forge transactions.
bind_role "${SA_INGEST}" "roles/pubsub.subscriber"
# Writes rows into the MuleNet dataset. dataEditor can insert and update table
# data but cannot delete the dataset or change IAM on it.
bind_role "${SA_INGEST}" "roles/bigquery.dataEditor"
# Required to run load jobs; dataEditor alone grants no job-creation right.
bind_role "${SA_INGEST}" "roles/bigquery.jobUser"
# Connects to AlloyDB through the connector with IAM auth. No admin rights,
# so it cannot create, resize, or delete clusters.
bind_role "${SA_INGEST}" "roles/alloydb.client"
# The AlloyDB connector calls the Service Usage API on the caller's behalf.
bind_role "${SA_INGEST}" "roles/serviceusage.serviceUsageConsumer"
# Custom runtime service accounts get no logging right by default.
bind_role "${SA_INGEST}" "roles/logging.logWriter"

echo "==> Binding IAM roles for ${SA_UI}..."
# Read-only on the analytics store: the UI shows scores and rings, never writes.
bind_role "${SA_UI}" "roles/bigquery.dataViewer"
bind_role "${SA_UI}" "roles/bigquery.jobUser"
# Reads secret values only. Cannot create, update, or destroy secrets.
bind_role "${SA_UI}" "roles/secretmanager.secretAccessor"
# Calls the private agents Cloud Run service. Invoker cannot deploy or modify it.
bind_role "${SA_UI}" "roles/run.invoker"
bind_role "${SA_UI}" "roles/logging.logWriter"

echo "==> Binding IAM roles for ${SA_AGENTS}..."
# Gemini and Vertex AI endpoint predictions. aiplatform.user can call models
# but cannot delete models, endpoints, or datasets.
bind_role "${SA_AGENTS}" "roles/aiplatform.user"
# Agents read evidence to build reports; they never write analytics tables.
bind_role "${SA_AGENTS}" "roles/bigquery.dataViewer"
bind_role "${SA_AGENTS}" "roles/bigquery.jobUser"
# Reads the operational graph for ring lookups.
bind_role "${SA_AGENTS}" "roles/alloydb.client"
bind_role "${SA_AGENTS}" "roles/serviceusage.serviceUsageConsumer"
bind_role "${SA_AGENTS}" "roles/secretmanager.secretAccessor"
bind_role "${SA_AGENTS}" "roles/logging.logWriter"

echo
echo "==> Done."
echo "    ingest SA: $(sa_email "${SA_INGEST}")"
echo "    ui     SA: $(sa_email "${SA_UI}")"
echo "    agents SA: $(sa_email "${SA_AGENTS}")"
echo "    images:    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
echo
echo "    Next: create the AlloyDB cluster, the BigQuery dataset from"
echo "    shared/ddl.sql, and the Pub/Sub topic + subscription."
