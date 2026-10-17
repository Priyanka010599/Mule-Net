#!/usr/bin/env bash
#
# MuleNet -- parallel asia-south1 stack for the ui/ingest/data move.
#
# Deliberately NOT a parameterisation of infra/deploy.sh: that script
# also builds and redeploys mulenet-agents and the enclave-adjacent env
# vars every run, and the whole point of this migration is to leave
# mulenet-agents (already in asia-south1) and the enclave (already
# entirely in asia-south1: KMS keyring, VM zone, all four buckets) alone.
# This script touches only what is still in us-central1: the UI's data
# bucket and the BigQuery dataset, plus the ui and ingest Cloud Run
# services themselves.
#
# Additive only. Nothing here deletes or reconfigures the existing
# mulenet-ui / mulenet-ingest / mulenet-mulenet-data / mulenet.mulenet
# resources in us-central1 -- those keep serving traffic unchanged. The
# one shared resource this script DOES modify is the existing Pub/Sub
# push subscription, repointed to the new ingest URL per the migration
# plan (Pub/Sub subscriptions are global, not regional, so there is no
# way to run both ingest regions live off one subscription at once).
#
# Usage:
#   PROJECT_ID=mulenet ./infra/deploy_asia_south1.sh
#
# Safe to re-run: every step is describe-or-create.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

: "${PROJECT_ID:?PROJECT_ID is not set. Usage: PROJECT_ID=mulenet ./infra/deploy_asia_south1.sh}"

REGION="asia-south1"
AGENTS_REGION="asia-south1"
REPO_NAME="mulenet"

# New names, distinct from both the us-central1 primary bucket
# (mulenet-mulenet-data) and the existing agents-only mirror
# (mulenet-mulenet-data-asia) -- this bucket is the UI's own primary
# copy in the new region, not a mirror of anything.
BUCKET="${BUCKET:-${PROJECT_ID}-mulenet-data-as1}"
BQ_DATASET="${BQ_DATASET:-mulenet_as1}"

SA_UI="mulenet-ui"
SA_INGEST="mulenet-ingest"
SVC_UI="mulenet-ui"
SVC_INGEST="mulenet-ingest"

sa_email() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

echo "==> MuleNet asia-south1 parallel stack: project=${PROJECT_ID}"
echo "    region: ${REGION}   bucket: gs://${BUCKET}   dataset: ${BQ_DATASET}"
echo "    us-central1 stack is NOT touched by this script."

IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

# ---------------------------------------------------------------------------
# 1. New GCS bucket -- holds only what the ui service fetches
#    (ui/out/demo_snapshot.json). ingest does not read GCS at all: it is a
#    Pub/Sub consumer that writes to BigQuery, so it needs no bucket grant.
# ---------------------------------------------------------------------------
echo "==> Ensuring bucket gs://${BUCKET}..."
if gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="${PROJECT_ID}" --location="${REGION}" --uniform-bucket-level-access
  echo "    Created."
fi
echo "    granting objectViewer on the bucket to ${SA_UI}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:$(sa_email "${SA_UI}")" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT_ID}" --quiet >/dev/null

# ---------------------------------------------------------------------------
# 2. Rebuild and publish the UI snapshot, md5-verified (same pattern as
#    infra/deploy.sh section 7 -- never deploy ui against an unverified copy).
# ---------------------------------------------------------------------------
echo "==> Rebuilding the UI snapshot (python -m ui.build_snapshot)..."
python -m ui.build_snapshot

SNAPSHOT_LOCAL="ui/out/demo_snapshot.json"
SNAPSHOT_URI="gs://${BUCKET}/ui/out/demo_snapshot.json"
echo "==> Publishing ${SNAPSHOT_LOCAL} to ${SNAPSHOT_URI}..."
gcloud storage cp "${SNAPSHOT_LOCAL}" "${SNAPSHOT_URI}" --project="${PROJECT_ID}"
SNAPSHOT_MD5_LOCAL="$(python -c 'import sys; from pathlib import Path; from infra.fetch_data import local_md5_b64; print(local_md5_b64(Path(sys.argv[1])))' "${SNAPSHOT_LOCAL}")"
SNAPSHOT_MD5_REMOTE="$(gcloud storage objects describe "${SNAPSHOT_URI}" \
  --project="${PROJECT_ID}" --format='value(md5_hash)')"
if [[ -z "${SNAPSHOT_MD5_REMOTE}" || "${SNAPSHOT_MD5_REMOTE}" != "${SNAPSHOT_MD5_LOCAL}" ]]; then
  echo "ERROR: snapshot md5 mismatch: local ${SNAPSHOT_MD5_LOCAL}, ${SNAPSHOT_URI} '${SNAPSHOT_MD5_REMOTE}'." >&2
  exit 1
fi
echo "    md5 ok: ${SNAPSHOT_MD5_LOCAL}"

# ---------------------------------------------------------------------------
# 3. New BigQuery dataset, loaded the same way as the us-central1 one.
#    SA_INGEST / SA_UI / SA_AGENTS already hold roles/bigquery.dataEditor,
#    dataViewer and jobUser at the PROJECT level (infra/setup.sh), which
#    cover every dataset in the project -- no new IAM binding is needed
#    for this dataset specifically.
# ---------------------------------------------------------------------------
echo "==> Loading BigQuery dataset ${BQ_DATASET} in ${REGION} from shared/bq_schemas.py..."
PROJECT_ID="${PROJECT_ID}" BUCKET="${BUCKET}" BQ_DATASET="${BQ_DATASET}" REGION="${REGION}" \
  python -m infra.load_bigquery

# ---------------------------------------------------------------------------
# 4. Build ui + ingest images in asia-south1, into the Artifact Registry
#    repo that already exists there (created for the agents image).
# ---------------------------------------------------------------------------
build_image() {
  local service="$1"
  echo "==> Building ${service} image in ${REGION}..."
  gcloud builds submit . \
    --project="${PROJECT_ID}" --region="${REGION}" \
    --config=infra/cloudbuild.yaml --ignore-file=infra/cloudbuild.ignore \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/$(sa_email mulenet-build)" \
    --substitutions="_SERVICE=${service},_REPO=${REPO_NAME},_REGION=${REGION},_TAG=${IMAGE_TAG}"
}
build_image ingest
build_image ui

# ---------------------------------------------------------------------------
# 5. Cloud Run -- ingest, in asia-south1. Same config as the us-central1
#    deploy, pointed at the new dataset.
# ---------------------------------------------------------------------------
echo "==> Deploying ${SVC_INGEST} to Cloud Run in ${REGION}..."
gcloud run deploy "${SVC_INGEST}" \
  --project="${PROJECT_ID}" --region="${REGION}" \
  --image="${IMAGE_BASE}/ingest:${IMAGE_TAG}" \
  --service-account="$(sa_email "${SA_INGEST}")" \
  --no-allow-unauthenticated --min-instances=0 --max-instances=2 \
  --set-env-vars="MULENET_GCP_PROJECT=${PROJECT_ID},MULENET_INGEST_ANALYTICS_SINK=bigquery,MULENET_BQ_DATASET=${BQ_DATASET},MULENET_INGEST_DLQ_SINK=pubsub"

INGEST_URL="$(gcloud run services describe "${SVC_INGEST}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "    ingest URL: ${INGEST_URL}"

echo "==> Granting the ingest SA invoker on ${SVC_INGEST}..."
gcloud run services add-iam-policy-binding "${SVC_INGEST}" \
  --member="serviceAccount:$(sa_email "${SA_INGEST}")" \
  --role="roles/run.invoker" \
  --project="${PROJECT_ID}" --region="${REGION}" --quiet >/dev/null

# ---------------------------------------------------------------------------
# 6. Repoint the EXISTING Pub/Sub push subscription at the new ingest URL.
#    Pub/Sub subscriptions are global, not regional: there is no way to
#    have both regions receive live push traffic at once off one
#    subscription, which is why this is a repoint, not a second
#    subscription, per the migration plan.
# ---------------------------------------------------------------------------
read_pubsub_name() { python -c "import shared.pubsub_schema as s; print(s.$1)"; }
TRANSFERS_SUB="$(read_pubsub_name TRANSFERS_SUBSCRIPTION)"
DLQ_TOPIC="$(read_pubsub_name TRANSFERS_DEAD_LETTER_TOPIC)"
MAX_DELIVERY="$(read_pubsub_name MAX_DELIVERY_ATTEMPTS)"

echo "==> Repointing ${TRANSFERS_SUB} to ${INGEST_URL}/pubsub/push..."
gcloud pubsub subscriptions update "${TRANSFERS_SUB}" \
  --project="${PROJECT_ID}" \
  --push-endpoint="${INGEST_URL}/pubsub/push" \
  --push-auth-service-account="$(sa_email "${SA_INGEST}")" \
  --dead-letter-topic="${DLQ_TOPIC}" \
  --max-delivery-attempts="${MAX_DELIVERY}"

# ---------------------------------------------------------------------------
# 7. Cloud Run -- ui, in asia-south1. Points at the new bucket/dataset and
#    at the EXISTING mulenet-agents service, which was already in
#    asia-south1 -- so this call becomes same-region instead of
#    cross-region, a free latency win from the move.
#
#    MULENET_UI_LIVE_PROOF_TAMPER is "1" here, not "0" as in
#    infra/deploy.sh: it is the ONE flag (ui/config.py -> ui/main.py)
#    that gates the tamper-test button and its proxy route. There is no
#    separate agents-side flag -- agents/live_proof_service.py always
#    knows how to run a tamper test; only the UI's own route 404s
#    without this. Deliberately on here so Step 3's tamper-test check
#    can be re-run from a clean deploy without a manual patch after.
# ---------------------------------------------------------------------------
AGENTS_URL="$(gcloud run services describe mulenet-agents \
  --project="${PROJECT_ID}" --region="${AGENTS_REGION}" --format='value(status.url)')"

ENV_FILE="$(mktemp)"
trap 'rm -f "${ENV_FILE}"' EXIT
cat > "${ENV_FILE}" <<YAML
MULENET_DATA_BUCKET: "${BUCKET}"
MULENET_FETCH: "ui/out/demo_snapshot.json"
MULENET_UI_SNAPSHOT: "/app/ui/out/demo_snapshot.json"
MULENET_UI_METRICS: "/app/ui/data/metrics_fallback.json"
MULENET_UI_CHAT_BACKEND: "agents"
MULENET_UI_AGENTS_URL: "${AGENTS_URL}"
MULENET_UI_AGENTS_AUDIENCE: "${AGENTS_URL}"
MULENET_UI_AGENTS_TIMEOUT_S: "75"
MULENET_UI_LIVE_PROOF_TAMPER: "1"
YAML

echo "==> Deploying ${SVC_UI} to Cloud Run in ${REGION}..."
gcloud run deploy "${SVC_UI}" \
  --project="${PROJECT_ID}" --region="${REGION}" \
  --image="${IMAGE_BASE}/ui:${IMAGE_TAG}" \
  --service-account="$(sa_email "${SA_UI}")" \
  --allow-unauthenticated --min-instances=0 --max-instances=2 --memory=1Gi \
  --env-vars-file="${ENV_FILE}"

UI_URL="$(gcloud run services describe "${SVC_UI}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"

echo
echo "==> Done."
echo "    UI (public):   ${UI_URL}                 [${REGION}]"
echo "    ingest:        ${INGEST_URL}  (private, Pub/Sub push only)  [${REGION}]"
echo "    agents (reused, unchanged): ${AGENTS_URL}  [${AGENTS_REGION}]"
echo "    data bucket:   gs://${BUCKET}"
echo "    BigQuery:      ${PROJECT_ID}.${BQ_DATASET}"
echo "    Pub/Sub push repointed to: ${INGEST_URL}/pubsub/push"
echo
echo "    us-central1 stack (mulenet-ui, mulenet-ingest, mulenet-mulenet-data,"
echo "    ${PROJECT_ID}.mulenet) is UNCHANGED and still serving its own traffic."
echo "    The enclave (KMS, VM, all 4 buckets) was already in ${REGION} and"
echo "    nothing here touched it."
