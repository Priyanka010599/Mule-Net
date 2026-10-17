#!/usr/bin/env bash
#
# MuleNet -- IAM for the live-proof durable cap (agents/live_proof_service.py).
#
# Idempotent. Grants mulenet-agents objectAdmin on the evidence bucket,
# conditioned to objects under cap/ ONLY. objectAdmin is needed because the
# cap is overwritten with if_generation_match, which requires
# storage.objects.delete; objectCreator alone allows the first write but
# every later reservation fails with 403 (observed live in production).
#
# Same condition style as the bank-sender bindings on the inbox bucket.
#
# Usage: PROJECT_ID=mulenet ./infra/live_proof_iam.sh

set -euo pipefail
: "${PROJECT_ID:?PROJECT_ID is not set}"

BUCKET="${PROJECT_ID}-enclave-evidence"
SA="mulenet-agents@${PROJECT_ID}.iam.gserviceaccount.com"
PREFIX="projects/_/buckets/${BUCKET}/objects/cap/"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/storage.objectAdmin" \
  --condition="expression=resource.name.startsWith(\"${PREFIX}\"),title=live-proof-cap-prefix-only,description=Read write and overwrite only the live-proof cap counter" \
  >/dev/null
echo "granted objectAdmin on gs://${BUCKET}/cap/ to ${SA}"
