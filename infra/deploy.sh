#!/usr/bin/env bash
#
# MuleNet -- deploy the running stack to Google Cloud.
#
# Assumes infra/setup.sh has already run (APIs, Artifact Registry repo, and
# the three least-privilege service accounts). This script adds only what
# setup.sh deliberately left out, because it did not exist yet:
#
#   1. a Cloud Storage bucket holding the published data
#   2. a BigQuery dataset loaded from shared/bq_schemas.py TABLE_SCHEMAS
#   3. the Pub/Sub topic, dead-letter topic and authenticated push
#      subscription, named from shared/pubsub_schema.py
#   4. Cloud Run services for ingest, ui and agents
#
# NO Cloud SQL and NO AlloyDB is provisioned. The deployed stack is
# Pub/Sub + BigQuery only; see docs/deployment.md for why, and
# infra/cloudsql_deploy.sh for the script that would provision it.
#
# ---------------------------------------------------------------------------
# TWO REGIONS, NAMED SEPARATELY AND ON PURPOSE
# ---------------------------------------------------------------------------
# This script used to take one REGION and hand it to everything. It cannot
# any more, because the stack is genuinely split:
#
#   REGION         (e.g. us-central1) data and the services that read it --
#                  the GCS bucket, the BigQuery dataset, ingest and ui.
#   AGENTS_REGION  (e.g. asia-south1) the agents service, because that is
#                  where Gemini is called from. us-central1 serves no
#                  Gemini 3.x at all. asia-south1 serves gemini-3.5-flash
#                  (agents.config.DEFAULT_MODEL) region-pinned; every
#                  newer 3.x tier is global-endpoint-only as of
#                  2026-10-03. See docs/residency.md.
#
# BOTH are required, with no default. A default would be the whole bug: one
# REGION silently putting the agents service in us-central1 is a deploy that
# succeeds, serves traffic, and is on the retiring region -- the failure only
# shows up as a model that stops existing. AGENTS_REGION is additionally
# checked against agents.config.DEFAULT_LOCATION below, which is the source of
# truth for where the agents call Gemini; the deploy refuses if they disagree
# rather than quietly deploying a service into a region its own config does
# not point at.
#
# Usage:
#   PROJECT_ID=your-project-id REGION=us-central1 AGENTS_REGION=asia-south1 \
#     ./infra/deploy.sh
#
# Required environment variables:
#   PROJECT_ID     Google Cloud project to deploy into. Never inherited from
#                  `gcloud config`: see the guard below.
#   REGION         Region for GCS, BigQuery, and the ingest and ui services.
#   AGENTS_REGION  Region for the agents service. Must match
#                  agents.config.DEFAULT_LOCATION.
#
# Optional:
#   BUCKET         Data bucket name. Defaults to ${PROJECT_ID}-mulenet-data.
#   BUCKET_AGENTS  Same-region mirror the agents service reads. Defaults to
#                  ${BUCKET}-asia. See section 1b for why it exists.
#   BQ_DATASET     BigQuery dataset. Defaults to mulenet.
#   SKIP_DATA      Set to 1 to skip the GCS upload and BigQuery load.
#
# Safe to re-run: every step is describe-or-create, uploads are rsynced,
# and the BigQuery loader truncates before appending. No project IDs,
# regions, keys or secrets are hardcoded.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

_USAGE="Run: PROJECT_ID=your-project-id REGION=us-central1 AGENTS_REGION=asia-south1 ./infra/deploy.sh"
: "${PROJECT_ID:?PROJECT_ID is not set. ${_USAGE}}"
: "${REGION:?REGION is not set. ${_USAGE}}"
: "${AGENTS_REGION:?AGENTS_REGION is not set. The agents service runs in a DIFFERENT region from the data (see the header). ${_USAGE}}"

# Every gcloud call below passes --project explicitly, so this guard changes
# nothing that runs. It exists because the failure it catches is silent and
# expensive: a machine whose `gcloud config` points at an unrelated project
# will happily create service accounts, IAM bindings, Artifact Registry repos
# and Cloud Run services THERE the moment one call is added without the flag,
# and a stray IAM binding in someone else's project is easy to miss and
# unpleasant to unpick. Warn loudly rather than fail: the default being
# different is normal and fine, relying on it is not.
_GCLOUD_DEFAULT="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -n "${_GCLOUD_DEFAULT}" && "${_GCLOUD_DEFAULT}" != "${PROJECT_ID}" ]]; then
  echo "NOTE: gcloud's default project is '${_GCLOUD_DEFAULT}', not '${PROJECT_ID}'."
  echo "      Every command here passes --project=${PROJECT_ID} explicitly, so this"
  echo "      deploy is safe. Do NOT 'gcloud config set project' to silence it:"
  echo "      that changes a global default for everything else on this machine."
fi

BUCKET="${BUCKET:-${PROJECT_ID}-mulenet-data}"
BUCKET_AGENTS="${BUCKET_AGENTS:-${BUCKET}-asia}"
BQ_DATASET="${BQ_DATASET:-mulenet}"
SKIP_DATA="${SKIP_DATA:-0}"
REPO_NAME="mulenet"

SA_INGEST="mulenet-ingest"
SA_UI="mulenet-ui"
SA_AGENTS="mulenet-agents"
SA_BUILD="mulenet-build"

SVC_INGEST="mulenet-ingest"
SVC_UI="mulenet-ui"
SVC_AGENTS="mulenet-agents"

# Where the agents call Gemini is decided in agents/config.py, not here.
# Read it rather than retyping it, so AGENTS_REGION cannot drift away from
# the region the service's own config will use at runtime (CLAUDE.md:
# "Derive lists from their source of truth; never duplicate them by hand").
AGENTS_DEFAULT_LOCATION="$(python -c 'import agents.config as c; print(c.DEFAULT_LOCATION)')"
if [[ "${AGENTS_REGION}" != "${AGENTS_DEFAULT_LOCATION}" ]]; then
  echo "ERROR: AGENTS_REGION=${AGENTS_REGION} but agents.config.DEFAULT_LOCATION" >&2
  echo "       is ${AGENTS_DEFAULT_LOCATION}. Deploying the service to one region" >&2
  echo "       while it calls Gemini in another is a cross-region call on every" >&2
  echo "       model turn, and is never what was meant. Fix one of them." >&2
  exit 1
fi

#: The parquet the agents service reads, as object names in the data bucket.
#:
#: Derived from agents/data.py's own table->file maps plus the case producer's
#: output, so this list cannot drift from what the service actually opens. It
#: is deliberately NOT "everything in the bucket": simulator/out also holds
#: ~173 MB of KYC PNGs that only docai/ reads, and copying those would triple
#: the cold start for files no agent opens.
#:
#: Fetching the COMPLETE set matters for correctness, not just coverage.
#: agents/data.py degrades PER TABLE to a small built-in mock when a parquet
#: is absent. agents/service.py's readiness check now matches that
#: granularity: it stats EVERY table in agents.data.required_tables()
#: (ALLOWED_SQL_TABLES | DataStore.known_tables()) individually, where it
#: used to ask only whether each DIRECTORY held at least one parquet. A
#: partial fetch therefore fails startup now instead of producing a service
#: that reports {"status": "ok"}, answers under backend "gemini", carries
#: real-looking sources -- and invents the numbers for whichever table was
#: left out.
#:
#: Anything added to agents/data.py's table maps must reach
#: infra/agents_fetch.py too; a name added to ALLOWED_SQL_TABLES that
#: DataStore cannot resolve to a file fails readiness with a KeyError rather
#: than being quietly unfetched -- see that module's own "uncovered" check.
#:
#: Built from infra/agents_fetch.py, not inline here, so the same list that
#: this script uses (both for MULENET_FETCH below and the mirror-upload loop
#: further down) is also what infra/test_agents_fetch.py imports and checks
#: -- a model added to ml/model_runtime.py without its files reaching this
#: list now fails a test instead of silently never reaching the deployed
#: service. See that module's docstring for the incident that motivated it
#: (Model C shipped without its files here, caught only by inspection).
AGENTS_FETCH="$(python -m infra.agents_fetch)"

# Resource names come from shared/pubsub_schema.py, which is the contract.
# Read them from the module instead of retyping them, so a rename there can
# never silently disagree with what this script creates.
read_pubsub_name() {
  python -c "import shared.pubsub_schema as s; print(s.$1)"
}
TRANSFERS_TOPIC="$(read_pubsub_name TRANSFERS_TOPIC)"
TRANSFERS_SUB="$(read_pubsub_name TRANSFERS_SUBSCRIPTION)"
DLQ_TOPIC="$(read_pubsub_name TRANSFERS_DEAD_LETTER_TOPIC)"
DLQ_SUB="$(read_pubsub_name TRANSFERS_DEAD_LETTER_SUBSCRIPTION)"
MAX_DELIVERY="$(read_pubsub_name MAX_DELIVERY_ATTEMPTS)"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
PUBSUB_AGENT="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
# A second Artifact Registry repo, in the agents region. Cloud Run can pull
# cross-region, but then every cold start drags the image across the Pacific
# as well as the data. Same-region is one repo and no egress.
IMAGE_BASE_AGENTS="${AGENTS_REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

# Immutable tag alongside :latest, so a revision can be traced to a commit.
# Falls back to a timestamp outside a git checkout.
IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)"

sa_email() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

echo "==> MuleNet deploy: project=${PROJECT_ID}"
echo "    data + ingest + ui region: ${REGION}   bucket gs://${BUCKET}"
echo "    agents region:             ${AGENTS_REGION}   mirror gs://${BUCKET_AGENTS}"
echo "    pubsub names from shared/pubsub_schema.py:"
echo "      topic=${TRANSFERS_TOPIC} sub=${TRANSFERS_SUB}"
echo "      dlq=${DLQ_TOPIC} dlq-sub=${DLQ_SUB} max-delivery=${MAX_DELIVERY}"

# ---------------------------------------------------------------------------
# 1. Cloud Storage -- the one published copy of the data
# ---------------------------------------------------------------------------
# Uniform bucket-level access: no per-object ACLs, so read access is exactly
# the IAM bindings below and nothing else. The bucket stays private; the UI
# reads it with its own service account, never the browser.
echo "==> Ensuring bucket gs://${BUCKET}..."
if gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
  echo "    Created."
fi

# The UI and the agents read the published snapshot; neither ever writes it.
# objectViewer is read-only and scoped to this bucket, not the project.
for account in "${SA_UI}" "${SA_AGENTS}"; do
  echo "    granting objectViewer on the bucket to ${account}"
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:$(sa_email "${account}")" \
    --role="roles/storage.objectViewer" \
    --project="${PROJECT_ID}" --quiet >/dev/null
done

if [[ "${SKIP_DATA}" != "1" ]]; then
  echo "==> Publishing data to gs://${BUCKET} (~175 MB, first run is slow)..."
  # rsync so a re-run only moves what changed.
  gcloud storage rsync --recursive simulator/out "gs://${BUCKET}/simulator/out" --project="${PROJECT_ID}"
  gcloud storage rsync --recursive ml/out "gs://${BUCKET}/ml/out" --project="${PROJECT_ID}"
  gcloud storage rsync --recursive agents/out "gs://${BUCKET}/agents/out" --project="${PROJECT_ID}"
  # ui/out/demo_snapshot.json is NOT uploaded here: it is rebuilt in section
  # 2b and published in section 7 on every deploy, even under SKIP_DATA=1.
  echo "    Uploaded."

  # -------------------------------------------------------------------------
  # 1b. The agents mirror -- the same bytes, in the agents' own region
  # -------------------------------------------------------------------------
  # The data lives in ${REGION} and the agents service runs in
  # ${AGENTS_REGION}, so its cold start would pull ~136 MB of parquet across
  # regions before it can serve. That is not a theoretical cost; both halves
  # were measured on this project:
  #
  #   cold start reading gs://${BUCKET} (us-central1 -> asia-south1)   49 s
  #     of which fetch_data                                            40 s
  #   cold start reading the same-region mirror                         9 s
  #     of which fetch_data                                           2.6 s
  #
  # 5x on total cold start, 15x on the fetch. Most of the cross-region cost
  # was not bandwidth but per-object round trips -- the 106 MB transfers file
  # took 11 s while the eleven small files took 2-4 s EACH.
  #
  # The mirror is written here, from the same local files and in the same run
  # as the primary upload above, so the two copies have no way to drift apart
  # independently; and infra/fetch_data.py verifies each downloaded file's md5
  # against the object it came from, so a partial replication fails the
  # container rather than quietly serving stale numbers. Only the parquet the
  # agents actually read is mirrored -- see AGENTS_FETCH.
  #
  # The alternative considered and rejected was baking the tables into the
  # image. It would also remove the cross-region pull, but it puts 136 MB of
  # generated data into every Cloud Build upload and couples a data refresh to
  # an image rebuild, and it would drop the md5-against-the-published-object
  # check that is currently the thing proving ui and agents read identical
  # bytes.
  echo "==> Ensuring agents mirror bucket gs://${BUCKET_AGENTS} in ${AGENTS_REGION}..."
  if gcloud storage buckets describe "gs://${BUCKET_AGENTS}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    Already exists, skipping."
  else
    gcloud storage buckets create "gs://${BUCKET_AGENTS}" \
      --project="${PROJECT_ID}" \
      --location="${AGENTS_REGION}" \
      --uniform-bucket-level-access
    echo "    Created."
  fi

  echo "    granting objectViewer on the mirror to ${SA_AGENTS}"
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_AGENTS}" \
    --member="serviceAccount:$(sa_email "${SA_AGENTS}")" \
    --role="roles/storage.objectViewer" \
    --project="${PROJECT_ID}" --quiet >/dev/null

  echo "==> Mirroring the agents' tables into ${AGENTS_REGION} (~136 MB)..."
  # Copied object by object rather than rsyncing the directories, because the
  # directories hold more than the agents read: simulator/out also carries
  # ~173 MB of KYC PNGs that only docai/ touches.
  ( IFS=','; for obj in ${AGENTS_FETCH}; do
      gcloud storage cp "gs://${BUCKET}/${obj}" "gs://${BUCKET_AGENTS}/${obj}" \
        --project="${PROJECT_ID}" >/dev/null
    done )
  echo "    Mirrored."

  # -------------------------------------------------------------------------
  # 2. BigQuery -- schemas come from the contract, never from inference
  # -------------------------------------------------------------------------
  # Run as Python rather than 'bq load': a Parquet load auto-detects its
  # schema, and WRITE_TRUNCATE deletes the table schema outright, so either
  # would quietly replace TABLE_SCHEMAS with whatever pandas wrote --
  # including the REQUIRED/NULLABLE modes and the column descriptions.
  # infra/load_bigquery.py creates each table from the contract, TRUNCATEs
  # (which keeps the schema) and appends. See its module docstring.
  echo "==> Loading BigQuery dataset ${BQ_DATASET} from shared/bq_schemas.py..."
  PROJECT_ID="${PROJECT_ID}" BUCKET="${BUCKET}" \
    BQ_DATASET="${BQ_DATASET}" REGION="${REGION}" \
    python -m infra.load_bigquery
else
  echo "==> SKIP_DATA=1: leaving Cloud Storage and BigQuery untouched."
fi

# ---------------------------------------------------------------------------
# 2b. Rebuild the UI snapshot from this checkout
# ---------------------------------------------------------------------------
# The ui image and ui/out/demo_snapshot.json must ship together: the image's
# pydantic models validate the snapshot on every ring page, so new ui code
# over a snapshot built by old code fails at request time. (PR #2 added
# evidence[].detector_versions; every ring-detail page then raised a
# ValidationError until the snapshot was rebuilt.) So the snapshot is rebuilt
# on every deploy, independent of SKIP_DATA, and published in section 7 just
# before the ui service is deployed.
#
# Rebuilt BEFORE the image builds because the builder also refreshes
# ui/data/metrics_fallback.json and lead_time_fallback.json, which ARE baked
# into the ui image. If those change here, that is real content: commit it.
echo "==> Rebuilding the UI snapshot (python -m ui.build_snapshot)..."
python -m ui.build_snapshot

# ---------------------------------------------------------------------------
# 3. Build images
# ---------------------------------------------------------------------------
# Built by Cloud Build, not locally: no Docker daemon is needed on the
# operator machine and the image is produced by the same steps every time.
# --ignore-file keeps the generated data out of the upload.
#
# Builds run as their own service account. Since mid-2024 the default is the
# Compute Engine default service account, which on a new project has no
# project-wide Editor role, so an unmodified `gcloud builds submit` fails on
# its own source upload:
#
#   ...-compute@developer.gserviceaccount.com does not have
#   storage.objects.get access to ... /buckets/PROJECT_cloudbuild/objects/...
#
# Google's guidance is that a user-specified service account is the more
# secure fix, rather than granting roles to a default account that every
# other service in the project also runs as. This one gets three narrow
# roles and nothing else.
echo "==> Ensuring build service account '${SA_BUILD}'..."
if gcloud iam service-accounts describe "$(sa_email "${SA_BUILD}")" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud iam service-accounts create "${SA_BUILD}" \
    --display-name="MuleNet Cloud Build" --project="${PROJECT_ID}"
  echo "    Created."
fi

# Reads the source tarball gcloud uploads for it.
gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}_cloudbuild" \
  --member="serviceAccount:$(sa_email "${SA_BUILD}")" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT_ID}" --quiet >/dev/null
# Pushes the built image. writer can push and pull but cannot delete a repo.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:$(sa_email "${SA_BUILD}")" \
  --role="roles/artifactregistry.writer" \
  --condition=None --quiet >/dev/null
# cloudbuild.yaml sets logging: CLOUD_LOGGING_ONLY, which needs this.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:$(sa_email "${SA_BUILD}")" \
  --role="roles/logging.logWriter" \
  --condition=None --quiet >/dev/null

# The build region is a parameter, not ${REGION}. It decides BOTH where the
# build runs and which Artifact Registry repo the image is pushed to, so
# defaulting it would push the agents image into the data region and put the
# cross-region pull back on every cold start by the back door.
build_image() {
  local service="$1"
  local build_region="$2"
  echo "==> Building ${service} image in ${build_region}..."
  gcloud builds submit . \
    --project="${PROJECT_ID}" \
    --region="${build_region}" \
    --config=infra/cloudbuild.yaml \
    --ignore-file=infra/cloudbuild.ignore \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/$(sa_email "${SA_BUILD}")" \
    --substitutions="_SERVICE=${service},_REPO=${REPO_NAME},_REGION=${build_region},_TAG=${IMAGE_TAG}"
}

# The agents image needs a repo in its own region. setup.sh created only the
# one in ${REGION}, because the agents service did not exist when it ran.
echo "==> Ensuring Artifact Registry repo '${REPO_NAME}' in ${AGENTS_REGION}..."
if gcloud artifacts repositories describe "${REPO_NAME}" \
     --project="${PROJECT_ID}" --location="${AGENTS_REGION}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud artifacts repositories create "${REPO_NAME}" \
    --project="${PROJECT_ID}" --location="${AGENTS_REGION}" \
    --repository-format=docker \
    --description="MuleNet container images (${AGENTS_REGION})"
  echo "    Created."
fi

build_image ingest "${REGION}"
build_image ui "${REGION}"
build_image agents "${AGENTS_REGION}"

# ---------------------------------------------------------------------------
# 4. Cloud Run -- ingest
# ---------------------------------------------------------------------------
# --no-allow-unauthenticated: only the Pub/Sub push identity may call it.
# MULENET_INGEST_OPERATIONAL_SINK is NOT set: ingest/config.py already
# defaults it to "noop" when a GCP project is configured and no Pub/Sub
# emulator is in play, which is exactly this environment. Postgres stays off
# without the deploy having to name it.
echo "==> Deploying ${SVC_INGEST} to Cloud Run..."
gcloud run deploy "${SVC_INGEST}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_BASE}/ingest:${IMAGE_TAG}" \
  --service-account="$(sa_email "${SA_INGEST}")" \
  --no-allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --set-env-vars="MULENET_GCP_PROJECT=${PROJECT_ID},MULENET_INGEST_ANALYTICS_SINK=bigquery,MULENET_BQ_DATASET=${BQ_DATASET},MULENET_INGEST_DLQ_SINK=pubsub"

INGEST_URL="$(gcloud run services describe "${SVC_INGEST}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "    ingest URL: ${INGEST_URL}"

# ---------------------------------------------------------------------------
# 5. Pub/Sub -- topic, dead-letter topic, authenticated push subscription
# ---------------------------------------------------------------------------
ensure_topic() {
  local topic="$1"
  echo "==> Ensuring topic ${topic}..."
  if gcloud pubsub topics describe "${topic}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    Already exists, skipping."
  else
    gcloud pubsub topics create "${topic}" --project="${PROJECT_ID}"
    echo "    Created."
  fi
}

ensure_topic "${TRANSFERS_TOPIC}"
ensure_topic "${DLQ_TOPIC}"

# Pub/Sub mints the OIDC token for authenticated push using the ingest
# service account, so its own service agent needs permission to create
# tokens for that account, and the account needs invoker on the service.
echo "==> Granting the Pub/Sub service agent token-creator on the ingest SA..."
gcloud iam service-accounts add-iam-policy-binding "$(sa_email "${SA_INGEST}")" \
  --member="serviceAccount:${PUBSUB_AGENT}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project="${PROJECT_ID}" --quiet >/dev/null

echo "==> Granting the ingest SA invoker on ${SVC_INGEST}..."
gcloud run services add-iam-policy-binding "${SVC_INGEST}" \
  --member="serviceAccount:$(sa_email "${SA_INGEST}")" \
  --role="roles/run.invoker" \
  --project="${PROJECT_ID}" --region="${REGION}" --quiet >/dev/null

# Dead-lettering needs the Pub/Sub service agent to publish to the DLQ topic
# and to acknowledge on the subscription it forwards from. Both grants must
# come after the dead-letter topic exists.
echo "==> Granting the Pub/Sub service agent publisher on ${DLQ_TOPIC}..."
gcloud pubsub topics add-iam-policy-binding "${DLQ_TOPIC}" \
  --member="serviceAccount:${PUBSUB_AGENT}" \
  --role="roles/pubsub.publisher" \
  --project="${PROJECT_ID}" --quiet >/dev/null

echo "==> Ensuring push subscription ${TRANSFERS_SUB}..."
if gcloud pubsub subscriptions describe "${TRANSFERS_SUB}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, updating push endpoint and dead-letter policy."
  gcloud pubsub subscriptions update "${TRANSFERS_SUB}" \
    --project="${PROJECT_ID}" \
    --push-endpoint="${INGEST_URL}/pubsub/push" \
    --push-auth-service-account="$(sa_email "${SA_INGEST}")" \
    --dead-letter-topic="${DLQ_TOPIC}" \
    --max-delivery-attempts="${MAX_DELIVERY}"
else
  gcloud pubsub subscriptions create "${TRANSFERS_SUB}" \
    --project="${PROJECT_ID}" \
    --topic="${TRANSFERS_TOPIC}" \
    --push-endpoint="${INGEST_URL}/pubsub/push" \
    --push-auth-service-account="$(sa_email "${SA_INGEST}")" \
    --dead-letter-topic="${DLQ_TOPIC}" \
    --max-delivery-attempts="${MAX_DELIVERY}" \
    --ack-deadline=60
  echo "    Created."
fi

echo "==> Granting the Pub/Sub service agent subscriber on ${TRANSFERS_SUB}..."
gcloud pubsub subscriptions add-iam-policy-binding "${TRANSFERS_SUB}" \
  --member="serviceAccount:${PUBSUB_AGENT}" \
  --role="roles/pubsub.subscriber" \
  --project="${PROJECT_ID}" --quiet >/dev/null

# A plain pull subscription on the dead-letter topic, so parked messages can
# be inspected instead of vanishing.
echo "==> Ensuring dead-letter subscription ${DLQ_SUB}..."
if gcloud pubsub subscriptions describe "${DLQ_SUB}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud pubsub subscriptions create "${DLQ_SUB}" \
    --project="${PROJECT_ID}" --topic="${DLQ_TOPIC}"
  echo "    Created."
fi

# ---------------------------------------------------------------------------
# 6. Cloud Run -- agents (a DIFFERENT region from everything above)
# ---------------------------------------------------------------------------
# This is the only deployed service that calls Gemini on Vertex AI, and the
# only one a public page can cause to spend money. Both facts drive what
# follows.
#
# Least privilege, narrower than setup.sh left it. setup.sh gave
# ${SA_AGENTS} seven project-wide roles, mirroring what ingest and ui got.
# The service needs exactly two things -- call Vertex, and read the published
# tables -- so the rest are removed here rather than left lying around on the
# one identity with a spend attached to it. Removing is safe to re-run:
# remove-iam-policy-binding on an absent binding is a no-op.
#
# Deliberately NOT restored, and why each one was there:
#   alloydb.client                 no database is provisioned at all.
#   secretmanager.secretAccessor   the two live-proof keys exist, but access
#                                  is granted per-secret (gcloud secrets
#                                  add-iam-policy-binding on
#                                  mulenet-live-proof-key-bank-a/-b, done
#                                  out-of-band, not in this script) rather
#                                  than project-wide, so this broader role
#                                  stays removed.
#   serviceusage.serviceUsageConsumer  already implied by aiplatform.user.
#   logging.logWriter              VERIFIED UNNECESSARY on this project:
#                                  container stdout still reaches Cloud
#                                  Logging without it, because Cloud Run's
#                                  platform collects it rather than the
#                                  runtime identity writing it. Checked by
#                                  reading this service's own startup logs
#                                  after the role was removed.
echo "==> Tightening ${SA_AGENTS} to least privilege..."
for role in \
  roles/alloydb.client \
  roles/logging.logWriter \
  roles/secretmanager.secretAccessor \
  roles/serviceusage.serviceUsageConsumer
do
  gcloud projects remove-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:$(sa_email "${SA_AGENTS}")" \
    --role="${role}" --condition=None --quiet >/dev/null 2>&1 || true
done
# bigquery.dataViewer / jobUser are deliberately NOT in the removal loop above
# (they used to be, with the rationale "agents read parquet, not BigQuery" --
# true when this comment was written, no longer true). agents/data.py now has
# an optional BigQuery-backed DataStore path (MULENET_AGENTS_DATA_SOURCE=bigquery,
# default stays parquet; see agents/config.py and docs/reviewer_notes.md). Both
# roles are granted out-of-band, the same way secretmanager.secretAccessor
# above is handled per-secret rather than from this script -- re-adding them to
# the removal loop would silently break that path on the next full deploy.sh
# run without anyone noticing until a live request failed.
# The one project-level role it keeps. Re-asserted rather than assumed, so
# this script produces the intended policy from any starting state.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:$(sa_email "${SA_AGENTS}")" \
  --role="roles/aiplatform.user" \
  --condition=None --quiet >/dev/null

# Regulation search needs exactly one permission --
# discoveryengine.servingConfigs.search -- to query the one
# mulenet-aml-regulations engine. Neither predefined Discovery Engine
# role fits: roles/discoveryengine.viewer is 106 read-only permissions
# across the whole product (agents, sessions, connectors, analytics
# dashboards); roles/discoveryengine.user is narrower in name only and
# actually grants real write/execute actions (session create/delete/
# upload, data-connector token/action execution, agent create/delete/
# update) this service has no business holding. A custom role with the
# one permission is the same pattern as the existing
# muleNetLiveProofVmOperator role.
echo "==> Ensuring custom role 'muleNetRegulationSearcher'..."
if gcloud iam roles describe muleNetRegulationSearcher --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Already exists, skipping."
else
  gcloud iam roles create muleNetRegulationSearcher \
    --project="${PROJECT_ID}" \
    --title="MuleNet Regulation Searcher" \
    --description="Search the one mulenet-aml-regulations Discovery Engine data store for cited regulation passages. Nothing broader: no admin, no write, no other Discovery Engine product surface (agents, sessions, data connectors)." \
    --permissions="discoveryengine.servingConfigs.search" \
    --stage=GA
  echo "    Created."
fi
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:$(sa_email "${SA_AGENTS}")" \
  --role="projects/${PROJECT_ID}/roles/muleNetRegulationSearcher" \
  --condition=None --quiet >/dev/null

echo "    ${SA_AGENTS}: roles/aiplatform.user, the custom regulation-search"
echo "    role, and objectViewer on the two data buckets -- nothing else."

# --max-instances=1 IS NOT A TUNING CHOICE. It is the cost control.
#
# agents/limits.py holds the per-session limit, the per-IP limit and the
# service-wide daily cap in PROCESS MEMORY. Each Cloud Run instance therefore
# enforces its own private copy of all three, so with N instances the daily
# cap is N x DEFAULT_DAILY_CAP, not DEFAULT_DAILY_CAP. Raising max-instances
# does not loosen the limit, it MULTIPLIES it -- silently, with no error and
# no log line, on a service whose caller is a public web page. At two
# instances the cap this service advertises is simply false.
#
# So: one instance, or move the counters to something shared. Until the
# second exists, this flag is the first.
#
# --no-allow-unauthenticated: docs/agents_api.md, section "Authentication".
# Only ${SA_UI} may call it, with an ID token minted for this service.
#
# memory/cpu: transfers.parquet is 101 MB and agents/tools.py builds a
# transfer adjacency index over ~573k rows in memory, so 4Gi is sizing for
# the working set rather than generosity. --cpu-boost is for startup: without
# it the container gets throttled CPU while fetch_data is hashing 136 MB.
#
# --timeout=120 sits just above the 90 s request budget in
# agents/config.py DEFAULT_REQUEST_TIMEOUT_S, so the application's own
# timeout fires first and returns the documented 504 envelope instead of
# Cloud Run severing the connection with no error_type at all.
# Pulls in the enclave's own resource names (buckets, KMS keys, workload
# SA, region/zone, bank ids) so the live-proof env vars below are derived
# from the one place those names already live, not retyped here.
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
LIVE_PROOF_DIGEST="$(cat "${REPO_ROOT}/.enclave-digest-v1")"
LIVE_PROOF_PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
LIVE_PROOF_WIP_AUDIENCE="//iam.googleapis.com/projects/${LIVE_PROOF_PROJECT_NUMBER}/locations/${ENCLAVE_WIP_LOCATION}/workloadIdentityPools/${ENCLAVE_WIP}/providers/${ENCLAVE_WIP_PROVIDER}"

# Which accounts to flag, per bank. In a full deployment this would travel
# inside the sealed inbox bundle; here it is env-var metadata instead
# (enclave/workload.py's _flagged_from_env owns the real explanation).
# Read from the same .enclave-flagged-<BANK>.csv files the stage3v2
# production run itself was driven from, rather than retyped here --
# observed live: a run with this omitted completes cleanly, attests, and
# gets its KMS key, and still has nothing to decrypt, because ring-finding
# then has an empty flagged set to cluster.
flagged_csv() {
  tail -n +2 "${REPO_ROOT}/.enclave-flagged-$1.csv" | paste -sd, -
}
LIVE_PROOF_FLAGGED_BANK_A="$(flagged_csv BANK_A)"
LIVE_PROOF_FLAGGED_BANK_B="$(flagged_csv BANK_B)"
LIVE_PROOF_FLAGGED_BANK_C="$(flagged_csv BANK_C)"

echo "==> Deploying ${SVC_AGENTS} to Cloud Run in ${AGENTS_REGION}..."
AGENTS_ENV_FILE="$(mktemp)"
trap 'rm -f "${AGENTS_ENV_FILE}"' EXIT
cat > "${AGENTS_ENV_FILE}" <<YAML
MULENET_AGENT_MODE: "vertex"
GOOGLE_CLOUD_PROJECT: "${PROJECT_ID}"
GOOGLE_CLOUD_LOCATION: "${AGENTS_REGION}"
GOOGLE_GENAI_USE_VERTEXAI: "1"
MULENET_AGENT_REQUIRE_DATA: "1"
# Real Vertex AI Search over the actual RBI/PMLA/FIU-IND texts (built
# and populated 2026-09-27 -- see docai/index_regulations.py), rather
# than docai/regulations.py's paraphrased mock corpus. "global" because
# that is the only location Discovery Engine's Enterprise search tier
# (needed for extractive answers with page numbers) is offered in.
MULENET_DOCAI_MODE: "cloud"
MULENET_REGULATIONS_ENGINE: "mulenet-aml-regulations-app"
MULENET_SEARCH_LOCATION: "global"
MULENET_DATA_BUCKET: "${BUCKET_AGENTS}"
MULENET_FETCH: "${AGENTS_FETCH}"
MULENET_ENCLAVE_PROJECT: "${ENCLAVE_PROJECT}"
MULENET_ENCLAVE_REGION: "${ENCLAVE_REGION}"
MULENET_ENCLAVE_ZONE: "${ENCLAVE_ZONE}"
MULENET_ENCLAVE_IMAGE_DIGEST: "${LIVE_PROOF_DIGEST}"
MULENET_ENCLAVE_WORKLOAD_SA: "${ENCLAVE_WORKLOAD_SA_EMAIL}"
MULENET_ENCLAVE_INBOX_BUCKET: "${ENCLAVE_INBOX_BUCKET}"
MULENET_ENCLAVE_OUTBOX_BUCKET: "${ENCLAVE_OUTBOX_BUCKET}"
MULENET_ENCLAVE_EVIDENCE_BUCKET: "${ENCLAVE_EVIDENCE_BUCKET}"
MULENET_ENCLAVE_KMS_KEY: "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}"
MULENET_ENCLAVE_HASH_KMS_KEY: "${ENCLAVE_KMS_HASH_KEY_RESOURCE}"
MULENET_ENCLAVE_HASH_KEY_URI: "gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin"
MULENET_ENCLAVE_WIP_AUDIENCE: "${LIVE_PROOF_WIP_AUDIENCE}"
MULENET_ENCLAVE_BANK_IDS: "${BANK_IDS}"
MULENET_ENCLAVE_LIVE_PROOF_INBOX_RUN_ID: "stage3v2"
MULENET_LIVE_PROOF_CORRECT_BANK: "bank-a"
MULENET_LIVE_PROOF_WRONG_BANK: "bank-b"
# Public keys, not secret (see enclave/outbox_crypto.py's module docstring):
# derived from the three banks' existing .enclave-outbox-key-<bank>.bin
# private keys via X25519's public_key(), the same ones
# infra/enclave_run.sh has always passed as tee-env-* metadata for every
# non-live-proof run. Required by enclave.config.EnclaveConfig.from_env()
# for every bank in BANK_IDS -- omitting even one is the ONE failure path
# in enclave/workload.py's main() that writes no evidence object at all,
# observed live: a VM that boots, runs and cleanly self-terminates while
# producing nothing, with no way to tell why from outside (the production
# image does not redirect container stdout by design).
MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A: "J2V0K6KzWKRnGKa3qz6ZUbYdOR/59ZPoLD0/Y8Skbyc="
MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_B: "22ws31NbjbvQO8hCy5myVZIxqT61y2C8xYIblU1xXUo="
MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_C: "uZ8WmcwTjwATW9pYj8JkI1wwX0VMp5zjelseqqF3BTk="
MULENET_ENCLAVE_FLAGGED_BANK_A: "${LIVE_PROOF_FLAGGED_BANK_A}"
MULENET_ENCLAVE_FLAGGED_BANK_B: "${LIVE_PROOF_FLAGGED_BANK_B}"
MULENET_ENCLAVE_FLAGGED_BANK_C: "${LIVE_PROOF_FLAGGED_BANK_C}"
# The rolling-24h cap's durable counter (agents/live_proof_service.py).
# Read/written under mulenet-agents' own identity, never impersonated:
# it is not compute or decrypt material, so it does not belong behind
# the narrowly-scoped mulenet-live-proof SA. Reuses the evidence bucket
# rather than provisioning a new one -- one small object under cap/,
# nothing else in that bucket is touched.
MULENET_LIVE_PROOF_CAP_BUCKET: "${ENCLAVE_EVIDENCE_BUCKET}"
YAML

# --no-cpu-throttling: agents/live_proof_service.py starts a ~2-4 min
# background asyncio task (the live-proof VM create -> poll -> decrypt
# cycle) and returns from POST /live-proof/start immediately. Cloud Run's
# default CPU throttling freezes CPU almost entirely between requests, so
# without this flag that background task would stall the moment the HTTP
# response is sent and only inch forward on each incoming poll.
#
# --set-secrets kept separate from AGENTS_ENV_FILE: the two secrets hold
# the banks' real private key material, and this way it is never written
# to any file this script leaves on disk.
gcloud run deploy "${SVC_AGENTS}" \
  --project="${PROJECT_ID}" \
  --region="${AGENTS_REGION}" \
  --image="${IMAGE_BASE_AGENTS}/agents:${IMAGE_TAG}" \
  --service-account="$(sa_email "${SA_AGENTS}")" \
  --no-allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --memory=4Gi \
  --cpu=2 \
  --cpu-boost \
  --no-cpu-throttling \
  --timeout=120 \
  --concurrency=8 \
  --env-vars-file="${AGENTS_ENV_FILE}" \
  --set-secrets="MULENET_LIVE_PROOF_CORRECT_KEY_B64=mulenet-live-proof-key-bank-a:latest,MULENET_LIVE_PROOF_WRONG_KEY_B64=mulenet-live-proof-key-bank-b:latest"

AGENTS_URL="$(gcloud run services describe "${SVC_AGENTS}" \
  --project="${PROJECT_ID}" --region="${AGENTS_REGION}" --format='value(status.url)')"
echo "    agents URL: ${AGENTS_URL}  (private)"

# The UI calls the agents service as itself. ${SA_UI} already holds a
# PROJECT-WIDE roles/run.invoker from setup.sh, which is broader than this
# needs -- it lets the UI invoke every Cloud Run service in the project. This
# service-scoped binding is the one that should exist, and it is added here so
# the wiring survives the project-wide grant being tightened later.
echo "==> Granting ${SA_UI} invoker on ${SVC_AGENTS}..."
gcloud run services add-iam-policy-binding "${SVC_AGENTS}" \
  --member="serviceAccount:$(sa_email "${SA_UI}")" \
  --role="roles/run.invoker" \
  --project="${PROJECT_ID}" --region="${AGENTS_REGION}" --quiet >/dev/null

# ---------------------------------------------------------------------------
# 7. Cloud Run -- ui
# ---------------------------------------------------------------------------
# Publicly reachable: this is the demo console a judge opens.
# MULENET_FETCH pulls the published snapshot from the bucket at start, so the
# UI serves the same bytes that were loaded into BigQuery. MULENET_UI_METRICS
# points at the copy bundled in the image, because ml/experiments/ is not
# published to the bucket.
#
# Passed as a YAML file rather than --set-env-vars. Git Bash / MSYS on Windows
# rewrites command-line arguments that look like Unix paths into Windows ones,
# and it mangled the container-side paths here: MULENET_UI_SNAPSHOT was
# delivered as "C:/Program Files/Git/app/ui/out/demo_snapshot.json", which
# ui/config.py then resolved against the container's working directory into
# "/app/C:/Program Files/Git/app/...". The conversion applies to the whole
# argument string, so escaping one value does not help. Values inside a file
# are never seen by the shell, which makes this correct on every platform.
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
MULENET_UI_LIVE_PROOF_TAMPER: "${LIVE_PROOF_TAMPER:-0}"
YAML

# On the three env vars above:
#
# MULENET_UI_CHAT_BACKEND=agents replaces the MockRouter the deployed demo
# used to serve. ui/config.py refuses to start with this set and no
# MULENET_UI_AGENTS_URL, so the two cannot be half-applied.
#
# MULENET_UI_AGENTS_AUDIENCE is set to the same value even though
# ui/config.py defaults it to the URL. The ID token's audience must match
# the service EXACTLY, a Cloud Run service answers on more than one hostname,
# and a mismatch fails as a Cloud Run 403 that the UI renders as a plausible
# "offline demo mode" banner rather than as a configuration error. Naming it
# means the value is visible in the service description instead of implied.
#
# MULENET_UI_AGENTS_TIMEOUT_S is raised from ui/config.py's 30 s default.
# Measured answers on the live service run 2.3-10.5 s, but the agents service
# gives itself 90 s (agents/config.py DEFAULT_REQUEST_TIMEOUT_S) and a
# multi-tool question can use it. At 30 s the UI would abandon a request the
# agents service was still going to answer and show the offline mock instead
# -- a Gemini answer silently replaced by a keyword match, which is exactly
# the confusion the offline label exists to prevent. 75 s leaves the agents
# service's own timeout to fire first and return the documented 504.

# Publish the snapshot rebuilt in section 2b to the exact object the ui
# service fetches at start (MULENET_DATA_BUCKET + MULENET_FETCH above), then
# prove the bytes landed: the object's md5 in GCS must equal the local
# file's. The exit code of `cp` is not proof; the md5 is.
SNAPSHOT_LOCAL="ui/out/demo_snapshot.json"
SNAPSHOT_URI="gs://${BUCKET}/ui/out/demo_snapshot.json"
echo "==> Publishing ${SNAPSHOT_LOCAL} to ${SNAPSHOT_URI}..."
gcloud storage cp "${SNAPSHOT_LOCAL}" "${SNAPSHOT_URI}" --project="${PROJECT_ID}"
SNAPSHOT_MD5_LOCAL="$(python -c 'import sys; from pathlib import Path; from infra.fetch_data import local_md5_b64; print(local_md5_b64(Path(sys.argv[1])))' "${SNAPSHOT_LOCAL}")"
SNAPSHOT_MD5_REMOTE="$(gcloud storage objects describe "${SNAPSHOT_URI}" \
  --project="${PROJECT_ID}" --format='value(md5_hash)')"
if [[ -z "${SNAPSHOT_MD5_REMOTE}" || "${SNAPSHOT_MD5_REMOTE}" != "${SNAPSHOT_MD5_LOCAL}" ]]; then
  echo "ERROR: snapshot md5 mismatch: local ${SNAPSHOT_MD5_LOCAL}, ${SNAPSHOT_URI} '${SNAPSHOT_MD5_REMOTE}'." >&2
  echo "       Refusing to deploy ${SVC_UI} against a snapshot it did not verify." >&2
  exit 1
fi
echo "    md5 ok: ${SNAPSHOT_MD5_LOCAL}"

echo "==> Deploying ${SVC_UI} to Cloud Run..."
gcloud run deploy "${SVC_UI}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_BASE}/ui:${IMAGE_TAG}" \
  --service-account="$(sa_email "${SA_UI}")" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --memory=1Gi \
  --env-vars-file="${ENV_FILE}"

UI_URL="$(gcloud run services describe "${SVC_UI}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"

echo
echo "==> Done."
echo "    UI (public):  ${UI_URL}                 [${REGION}]"
echo "    ingest:       ${INGEST_URL}  (private, Pub/Sub push only)  [${REGION}]"
echo "    agents:       ${AGENTS_URL}  (private, ${SA_UI} only)  [${AGENTS_REGION}]"
echo "    data bucket:  gs://${BUCKET}"
echo "    agents data:  gs://${BUCKET_AGENTS}  (same-region mirror)"
echo "    BigQuery:     ${PROJECT_ID}.${BQ_DATASET}"
echo "    topic:        ${TRANSFERS_TOPIC} -> ${TRANSFERS_SUB} -> ingest"
echo "    dead letter:  ${DLQ_TOPIC} (after ${MAX_DELIVERY} failed deliveries)"
echo
echo "    The deployed chat now calls Gemini on Vertex AI in ${AGENTS_REGION}."
echo "    ${SVC_AGENTS} runs at --max-instances=1 ON PURPOSE: the rate limits"
echo "    and the daily cap in agents/limits.py are in-memory and per-instance,"
echo "    so a second instance doubles every cap without saying so."
echo
echo "    NOT provisioned: Cloud SQL / AlloyDB. See infra/cloudsql_deploy.sh."
