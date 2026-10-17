#!/usr/bin/env bash
#
# MuleNet enclave — STAGE 2: give each bank an identity-only project and prove
# it can deliver ciphertext to the operator's inbox.
#
# Usage:
#   ./infra/enclave_bank_identity.sh bank-a mulenet-bank-a
#   ./infra/enclave_bank_identity.sh bank-c mulenet-bank-c     # the unbilled one
#
# ---------------------------------------------------------------------------
# THE FIRST THING THIS SCRIPT DOES IS THE THING NOBODY HAS VERIFIED
# ---------------------------------------------------------------------------
# The design ("option c") rests on an assumption: a service account living in an
# UNBILLED project can be granted write access to a bucket in another project
# and can authenticate. The trial billing account is capped at five linked
# projects and is full, mulenet-bank-c is deliberately unbilled, and if this
# assumption is false the identity-only bank model does not work.
#
# So this script proves it end to end, with a real object write, and STOPS if it
# cannot. It does NOT add billing to bank-c as a workaround -- that decision is
# the owner's, not this script's.
#
# Authentication is proven by IMPERSONATION, not by a service-account key:
# key creation is blocked by org policy (iam.disableServiceAccountKeyCreation),
# and a downloaded key in a public repo's workflow would be the wrong answer
# even if it were allowed. The operator is granted
# roles/iam.serviceAccountTokenCreator on the bank's service account, mints a
# short-lived token for it, and writes as that identity.
#
# UNEXECUTED: not run; this session had no Google Cloud credentials. The
# assumption above therefore remains UNVERIFIED -- see docs/enclave.md.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

BANK_ID="${1:?usage: $0 <bank-id> <bank-project-id>}"
BANK_PROJECT="${2:?usage: $0 <bank-id> <bank-project-id>}"
BANK_SA_ID="${BANK_SA_ID:-mulenet-${BANK_ID}-sender}"
BANK_SA_EMAIL="${BANK_SA_ID}@${BANK_PROJECT}.iam.gserviceaccount.com"

info "bank id=${BANK_ID} project=${BANK_PROJECT}"

step "0. Billing state of ${BANK_PROJECT} (for the record, not a gate)"
BILLED="$(project_billing_enabled "${BANK_PROJECT}" || true)"
case "${BILLED}" in
  True)  ok "${BANK_PROJECT} is billed" ;;
  False) warn "${BANK_PROJECT} is UNBILLED — this run is the real test of the identity-only model" ;;
  *)     warn "${BANK_PROJECT} billing state unreadable ('${BILLED:-unknown}'); continuing, the write test below is what decides" ;;
esac

step "1. APIs on the bank project"
# Only what an identity needs. Deliberately NOT compute, storage-admin, kms
# admin or anything billable: a bank project holds an identity and nothing else.
# iamcredentials is what mints the impersonated token; sts backs federation.
# cloudkms is enabled here for USAGE ATTRIBUTION ONLY, not capability: it
# grants nothing by itself -- the bank's sender can still only encrypt under
# the one key mulenet's own IAM binding names (roles/cloudkms.cryptoKeyEncrypter,
# step 4 below). Observed live: without it, an encrypt call authenticated as
# this bank's impersonated identity, with no quota project set client-side,
# gets attributed by KMS to the CALLER's own project for consumer/usage
# checks -- refused as "KMS API has not been used in project <this project>"
# even though the key and its IAM policy live entirely in mulenet. This bit
# both banks whose project happened not to have it already enabled; the third
# had it enabled incidentally and never hit the refusal.
# Enabling an API on an unbilled project is itself part of what is unverified,
# so a failure here is reported as the STOP condition rather than swallowed.
if ! ensure_api_enabled "${BANK_PROJECT}" iamcredentials.googleapis.com sts.googleapis.com cloudkms.googleapis.com; then
  fatal "could not enable iamcredentials/sts/cloudkms on ${BANK_PROJECT}. If this is because the project is unbilled, STOP: the identity-only bank model does not hold, and the design needs revisiting. Do NOT add billing as a workaround without the owner's decision."
fi

step "2. The bank's sender service account"
ensure_service_account "${BANK_PROJECT}" "${BANK_SA_ID}" \
  "MuleNet ${BANK_ID} cross-bank edge sender" >/dev/null

step "3. Grant the bank write-only access to its own inbox prefix"
BANK_MEMBER="serviceAccount:${BANK_SA_EMAIL}"
PREFIX="inbox/${BANK_ID}/"

# objectCreator, not objectAdmin or objectUser. The difference is load-bearing:
#   - objectCreator can write a new object and cannot read or overwrite,
#   - objectAdmin could overwrite ANOTHER bank's bundle, silently removing that
#     bank's accounts from the joint graph -- which would let a bank hide its
#     own mules from the scoring.
# The IAM condition restricts it further to the bank's own prefix, so bank-a
# cannot write into bank-b's slot and impersonate its contribution.
CONDITION_EXPR="resource.name.startsWith(\"projects/_/buckets/${ENCLAVE_INBOX_BUCKET}/objects/${PREFIX}\")"
info "granting storage.objectCreator on gs://${ENCLAVE_INBOX_BUCKET}/${PREFIX} to ${BANK_SA_EMAIL}"
gcloud storage buckets add-iam-policy-binding "gs://${ENCLAVE_INBOX_BUCKET}" \
  --project="${ENCLAVE_PROJECT}" \
  --member="${BANK_MEMBER}" \
  --role="roles/storage.objectCreator" \
  --condition="expression=${CONDITION_EXPR},title=${BANK_ID}-own-prefix-only,description=Write only into this bank's inbox prefix" \
  --quiet >/dev/null
record_binding "bucket:${ENCLAVE_INBOX_BUCKET} (${PREFIX})" "${BANK_MEMBER}" \
  "roles/storage.objectCreator [conditional]"

step "4. Grant the bank encrypt-only access to the enclave's KMS key"
# cryptoKeyEncrypter, never Decrypter. A bank can wrap a data key for the
# enclave and cannot unwrap anyone's -- including its own, once sent. That
# asymmetry is what stops a bank from reading another bank's bundle out of the
# inbox even though the objects sit in a bucket it can write to.
gcloud kms keys add-iam-policy-binding "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" \
  --project="${ENCLAVE_PROJECT}" \
  --member="${BANK_MEMBER}" \
  --role="roles/cloudkms.cryptoKeyEncrypter" \
  --condition=None --quiet >/dev/null
record_binding "kms:edge-wrap" "${BANK_MEMBER}" "roles/cloudkms.cryptoKeyEncrypter"

# Verified against the live policy, not the exit code above.
KMS_MEMBERS="$(gcloud kms keys get-iam-policy "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" \
  --project="${ENCLAVE_PROJECT}" \
  --flatten="bindings[].members[]" \
  --filter="bindings.role:roles/cloudkms.cryptoKeyEncrypter" \
  --format="value(bindings.members)" 2>/dev/null || true)"
if [[ "${KMS_MEMBERS}" == *"${BANK_SA_EMAIL}"* ]]; then
  ok "KMS policy names ${BANK_SA_EMAIL} as an encrypter"
else
  fatal "the encrypter binding for ${BANK_SA_EMAIL} is not in the live KMS policy"
fi

# And the negative half: the bank must NOT be able to decrypt.
DECRYPTERS="$(gcloud kms keys get-iam-policy "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" \
  --project="${ENCLAVE_PROJECT}" \
  --flatten="bindings[].members[]" \
  --filter="bindings.role:roles/cloudkms.cryptoKeyDecrypter" \
  --format="value(bindings.members)" 2>/dev/null || true)"
if [[ "${DECRYPTERS}" == *"${BANK_SA_EMAIL}"* ]]; then
  fatal "${BANK_SA_EMAIL} holds cryptoKeyDecrypter. A bank that can decrypt can read every other bank's bundle; remove that binding before continuing."
fi
ok "${BANK_SA_EMAIL} cannot decrypt (only attested image digests can)"

step "5. THE UNVERIFIED ASSUMPTION: can this SA authenticate and write?"
# Allow the current operator to mint tokens for the bank SA. This is the
# key-free substitute for a downloaded credential, and it is scoped to this one
# service account rather than granted project-wide.
OPERATOR_ACCOUNT="$(api_value account auth list --filter=status:ACTIVE)"
[[ -n "${OPERATOR_ACCOUNT}" ]] || fatal "no active gcloud credential"

info "granting roles/iam.serviceAccountTokenCreator on ${BANK_SA_EMAIL} to ${OPERATOR_ACCOUNT}"
gcloud iam service-accounts add-iam-policy-binding "${BANK_SA_EMAIL}" \
  --project="${BANK_PROJECT}" \
  --member="user:${OPERATOR_ACCOUNT}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet >/dev/null
record_binding "serviceAccount:${BANK_SA_EMAIL}" "user:${OPERATOR_ACCOUNT}" \
  "roles/iam.serviceAccountTokenCreator"

PROBE_FILE="$(mktemp)"
PROBE_OBJECT="${PREFIX}_auth_probe_$(date -u +%s).json"
echo '{"probe":"identity-only bank write test","payload":"not real data"}' > "${PROBE_FILE}"
trap 'rm -f "${PROBE_FILE}"' EXIT

info "writing gs://${ENCLAVE_INBOX_BUCKET}/${PROBE_OBJECT} AS ${BANK_SA_EMAIL}"
# `gcloud storage cp` is NOT used here, even impersonated: observed live, it
# 403'd with "does not have storage.objects.get access" on a fresh object
# this same identity had just created -- the CLI's upload path checks for an
# existing object first, which needs read, and this identity correctly has
# none (roles/storage.objectCreator only, by design: an identity that could
# read what it just wrote could read every other bank's ciphertext in the
# same bucket). A raw create-only insert, matching what
# infra/enclave_send_edges.py's blob.upload_from_string() actually does in
# production, has no such requirement. This is the real write path being
# tested, not a weaker substitute for it.
#
# IAM propagation is eventually consistent; a single immediate attempt can fail
# for timing reasons and would be misread as "unbilled projects cannot write".
# No success flag is kept from this loop on purpose: the authority on whether
# the write happened is the object-existence read below, not this command's
# status. A flag here would invite branching on the weaker signal.
for attempt in 1 2 3 4 5; do
  if python3 - "${BANK_SA_EMAIL}" "${ENCLAVE_PROJECT}" "${ENCLAVE_INBOX_BUCKET}" \
       "${PROBE_OBJECT}" "${PROBE_FILE}" 2>"${PROBE_FILE}.err" <<'PY'
import sys
import google.auth
from google.auth import impersonated_credentials
from google.cloud import storage

sa_email, project, bucket, object_name, payload_path = sys.argv[1:6]
source_credentials, _ = google.auth.default()
credentials = impersonated_credentials.Credentials(
    source_credentials=source_credentials,
    target_principal=sa_email,
    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = storage.Client(project=project, credentials=credentials)
blob = client.bucket(bucket).blob(object_name)
with open(payload_path, encoding="utf-8") as fh:
    blob.upload_from_string(fh.read(), content_type="application/json")
PY
  then
    break
  fi
  warn "attempt ${attempt} failed; waiting for IAM propagation"
  sleep $((attempt * 5))
done

# The decisive check is the OBJECT'S EXISTENCE read back from the API, not the
# exit status of the copy. A cp whose status came from somewhere else, or which
# wrote to a different path, would otherwise look like a pass.
PROBE_STATE="$(api_value name storage objects describe \
  "gs://${ENCLAVE_INBOX_BUCKET}/${PROBE_OBJECT}" --project="${ENCLAVE_PROJECT}")"

if [[ -n "${PROBE_STATE}" ]]; then
  ok "VERIFIED: a service account in ${BANK_PROJECT} (billingEnabled=${BILLED:-unknown}) authenticated and wrote gs://${ENCLAVE_INBOX_BUCKET}/${PROBE_OBJECT}"
  ok "the identity-only bank model holds for ${BANK_PROJECT}"
  gcloud storage rm "gs://${ENCLAVE_INBOX_BUCKET}/${PROBE_OBJECT}" \
    --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || \
    warn "could not clean up the probe object; remove it by hand"
else
  err "the probe object does not exist in the bucket. The write did NOT succeed."
  if [[ -s "${PROBE_FILE}.err" ]]; then
    err "gcloud said:"
    sed 's/^/        /' "${PROBE_FILE}.err" >&2
  fi
  rm -f "${PROBE_FILE}.err"
  cat >&2 <<'STOP'

    STOP. This is the assumption the design was resting on, and it failed.

    A service account in this project could not authenticate and write to the
    operator's inbox bucket. Do NOT add billing to the project to work around
    it -- report the failure and let the owner decide. The alternatives are:

      * host each bank's sender identity in a BILLED project (costs one of the
        five billing slots, which are full),
      * have the operator write on each bank's behalf, which weakens the story
        because the operator then handles plaintext before it is sealed,
      * drop to two banks for the demo and say so.

STOP
  exit 1
fi
rm -f "${PROBE_FILE}.err"

# Negative half of the isolation claim: the bank must not be able to write into
# ANOTHER bank's prefix. Without this, the conditional binding above is
# untested, and a bank could impersonate another's contribution.
step "6. Confirm the bank cannot write outside its own prefix"
FOREIGN_OBJECT="inbox/not-${BANK_ID}/_should_not_exist_$(date -u +%s).json"
# Same raw create-only insert as step 5, not gcloud storage cp -- see the
# comment there. Using the CLI here would make this test pass whether or not
# the IAM condition actually restricts the prefix, since the CLI's own
# missing-get requirement would deny the write regardless of destination.
if python3 - "${BANK_SA_EMAIL}" "${ENCLAVE_PROJECT}" "${ENCLAVE_INBOX_BUCKET}" \
     "${FOREIGN_OBJECT}" "${PROBE_FILE}" >/dev/null 2>&1 <<'PY'
import sys
import google.auth
from google.auth import impersonated_credentials
from google.cloud import storage

sa_email, project, bucket, object_name, payload_path = sys.argv[1:6]
source_credentials, _ = google.auth.default()
credentials = impersonated_credentials.Credentials(
    source_credentials=source_credentials,
    target_principal=sa_email,
    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = storage.Client(project=project, credentials=credentials)
blob = client.bucket(bucket).blob(object_name)
with open(payload_path, encoding="utf-8") as fh:
    blob.upload_from_string(fh.read(), content_type="application/json")
PY
then
  # Verify by state whether it really landed before declaring a leak.
  FOREIGN_STATE="$(api_value name storage objects describe \
    "gs://${ENCLAVE_INBOX_BUCKET}/${FOREIGN_OBJECT}" --project="${ENCLAVE_PROJECT}")"
  if [[ -n "${FOREIGN_STATE}" ]]; then
    gcloud storage rm "gs://${ENCLAVE_INBOX_BUCKET}/${FOREIGN_OBJECT}" \
      --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || true
    fatal "${BANK_SA_EMAIL} wrote OUTSIDE its prefix (${FOREIGN_OBJECT}). The IAM condition is not restricting it; fix before continuing."
  fi
fi
ok "${BANK_SA_EMAIL} cannot write outside inbox/${BANK_ID}/"

print_bindings

step "Stage 2 ready for ${BANK_ID}"
cat <<SUMMARY
    sender SA : ${BANK_SA_EMAIL}
    may write : gs://${ENCLAVE_INBOX_BUCKET}/${PREFIX}  (objectCreator, conditional)
    may wrap  : ${ENCLAVE_KMS_EDGE_KEY_RESOURCE}        (cryptoKeyEncrypter only)
    may NOT   : decrypt anything, read the inbox, or write another bank's prefix

    Next: seal and upload this bank's edges with
        ./infra/enclave_send_edges.py --bank-id ${BANK_ID} --bank-project ${BANK_PROJECT} ...
SUMMARY
