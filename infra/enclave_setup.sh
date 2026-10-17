#!/usr/bin/env bash
#
# MuleNet enclave — STAGE 1 infrastructure: attested key release.
#
# Usage:
#   ./infra/enclave_preflight.sh && ./infra/enclave_setup.sh
#
# Creates, idempotently:
#   - the APIs the operator project needs
#   - a KMS keyring and two keys: one wrapping the banks' per-bundle DEKs, one
#     wrapping the in-enclave HMAC key for Stage 4
#   - a workload identity pool and an OIDC provider that validates Confidential
#     Space attestation tokens
#   - the workload service account
#   - the Artifact Registry repo, and the inbox / outbox / keys buckets
#
# It does NOT grant decrypt on the KMS keys. That grant names a specific
# container image digest and is made by infra/enclave_pin_digest.sh after the
# image is built, because until an image exists there is no digest to pin and a
# broader grant would be a key released to "anything in Confidential Space".
#
# ---------------------------------------------------------------------------
# How key release is actually gated
# ---------------------------------------------------------------------------
#
# Two independent gates, on purpose:
#
#   1. The WIP provider's --attribute-condition rejects any token that is not a
#      genuine Confidential Space attestation with Secure Boot on. This stops a
#      non-enclave caller before a credential is ever minted.
#
#   2. The KMS IAM binding names the image digest:
#
#        principalSet://iam.googleapis.com/projects/<NUM>/locations/global/
#          workloadIdentityPools/<POOL>/attribute.image_digest/sha256:<DIGEST>
#
#      This is what makes "KMS releases the key ONLY to that attested image"
#      literally true, and it is what makes the negative test produce a real
#      KMS 403 rather than a generic auth error: a rebuilt image passes gate 1
#      and fails gate 2, so the refusal comes from KMS, naming the key.
#
# Attestation claim names below are taken from Google's own published sample
# (codelabs/health_data_analysis_codelab/src/uwear/, which validates
# submods.container.image_digest, hwmodel, swname and secboot) rather than from
# memory.
#
# UNEXECUTED: this script has not been run. The session that wrote it had no
# Google Cloud credentials. See docs/enclave.md.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

: > "${IAM_LOG_FILE}"

info "operator project=${ENCLAVE_PROJECT} region=${ENCLAVE_REGION} kms=${ENCLAVE_KMS_LOCATION}"

step "1. APIs"
ensure_api_enabled "${ENCLAVE_PROJECT}" "${ENCLAVE_APIS[@]}"

step "2. KMS keyring and keys"
ensure_kms_keyring "${ENCLAVE_PROJECT}" "${ENCLAVE_KMS_LOCATION}" "${ENCLAVE_KEYRING}"
ensure_kms_key "${ENCLAVE_PROJECT}" "${ENCLAVE_KMS_LOCATION}" "${ENCLAVE_KEYRING}" \
  "${ENCLAVE_EDGE_KEY}" encryption
ensure_kms_key "${ENCLAVE_PROJECT}" "${ENCLAVE_KMS_LOCATION}" "${ENCLAVE_KEYRING}" \
  "${ENCLAVE_HASH_KEY}" encryption

step "3. Workload service account"
# The identity the attestation federates into. It holds no key material: its
# only privileges are to fetch the image, write logs, read the inbox and write
# the outbox. Decrypt is granted to the ATTESTED IMAGE, not to this account, so
# possessing this account is not enough to read a bank's data.
ensure_service_account "${ENCLAVE_PROJECT}" "${ENCLAVE_WORKLOAD_SA}" \
  "MuleNet enclave workload (Confidential Space)" >/dev/null

bind_project_role() {
  local project="$1" member="$2" role="$3"
  info "granting ${role} on project ${project} to ${member}"
  gcloud projects add-iam-policy-binding "${project}" \
    --member="${member}" --role="${role}" --condition=None --quiet >/dev/null
  record_binding "project:${project}" "${member}" "${role}"
}

WORKLOAD_MEMBER="serviceAccount:${ENCLAVE_WORKLOAD_SA_EMAIL}"

# Required for the VM to obtain an attestation token at all.
bind_project_role "${ENCLAVE_PROJECT}" "${WORKLOAD_MEMBER}" \
  "roles/confidentialcomputing.workloadUser"
# A custom runtime service account gets no logging right by default, and without
# logs the negative test has nowhere to read the refusal from.
bind_project_role "${ENCLAVE_PROJECT}" "${WORKLOAD_MEMBER}" "roles/logging.logWriter"

step "4. Artifact Registry"
ensure_artifact_repo "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}" "${ENCLAVE_ARTIFACT_REPO}"
info "granting artifactregistry.reader on the repo to the workload SA"
gcloud artifacts repositories add-iam-policy-binding "${ENCLAVE_ARTIFACT_REPO}" \
  --location="${ENCLAVE_REGION}" --project="${ENCLAVE_PROJECT}" \
  --member="${WORKLOAD_MEMBER}" --role="roles/artifactregistry.reader" --quiet >/dev/null
record_binding "artifact-repo:${ENCLAVE_ARTIFACT_REPO}" "${WORKLOAD_MEMBER}" \
  "roles/artifactregistry.reader"

# The VM is created with --no-address, so it reaches Google APIs (image pull,
# KMS, GCS, logging) only through Private Google Access. It is off by default;
# observed live: without it the VM booted, attested, and could reach nothing.
info "enabling Private Google Access on the ${ENCLAVE_REGION} default subnet"
gcloud compute networks subnets update default --region="${ENCLAVE_REGION}"   --enable-private-ip-google-access --project="${ENCLAVE_PROJECT}" --quiet >/dev/null
require_state "private google access" "True" "privateIpGoogleAccess"   compute networks subnets describe default --region="${ENCLAVE_REGION}"   --project="${ENCLAVE_PROJECT}"

step "5. Buckets"
ensure_bucket "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}" "${ENCLAVE_INBOX_BUCKET}"
ensure_bucket "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}" "${ENCLAVE_OUTBOX_BUCKET}"
ensure_bucket "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}" "${ENCLAVE_KEYS_BUCKET}"
ensure_bucket "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}" "${ENCLAVE_EVIDENCE_BUCKET}"

bind_bucket_role() {
  local bucket="$1" member="$2" role="$3"
  info "granting ${role} on gs://${bucket} to ${member}"
  gcloud storage buckets add-iam-policy-binding "gs://${bucket}" \
    --member="${member}" --role="${role}" --project="${ENCLAVE_PROJECT}" --quiet >/dev/null
  record_binding "bucket:${bucket}" "${member}" "${role}"
}

# The workload reads what the banks sealed and writes per-bank alerts.
# objectViewer on the inbox, not objectAdmin: the enclave has no reason to
# delete a bank's bundle, and if it could, a bug could destroy the evidence.
bind_bucket_role "${ENCLAVE_INBOX_BUCKET}"  "${WORKLOAD_MEMBER}" "roles/storage.objectViewer"
bind_bucket_role "${ENCLAVE_OUTBOX_BUCKET}" "${WORKLOAD_MEMBER}" "roles/storage.objectCreator"
# The wrapped HMAC key lives here; the enclave reads it and never writes it.
bind_bucket_role "${ENCLAVE_KEYS_BUCKET}"   "${WORKLOAD_MEMBER}" "roles/storage.objectViewer"

# Evidence. objectCreator, NOT objectAdmin or objectUser: the workload must be
# able to add a run's result and must NOT be able to overwrite or delete an
# earlier one. An enclave that could rewrite its own audit trail is not an audit
# trail. (The workload also passes if_generation_match=0, so an accidental
# overwrite fails at the API even if the role were widened.)
#
# This binding is on the VM's ATTACHED service account, and it is deliberately
# independent of attestation: that is what lets a run refused the KMS key still
# write down why it was refused. The trade-off is stated in docs/enclave.md —
# an evidence object proves a VM ran, not that it was attested; the object's
# CONTENT is what proves that.
bind_bucket_role "${ENCLAVE_EVIDENCE_BUCKET}" "${WORKLOAD_MEMBER}" "roles/storage.objectCreator"

step "6. Workload identity pool and attestation provider"
ensure_workload_identity_pool "${ENCLAVE_PROJECT}" "${ENCLAVE_WIP_LOCATION}" "${ENCLAVE_WIP}"

# Maps the attestation claims this project makes decisions on. image_digest is
# the one the KMS binding keys off; swname and secboot back the provider
# condition; image_reference is mapped for auditing, so a log shows which tag a
# digest came from.
ATTRIBUTE_MAPPING="google.subject=assertion.sub"
ATTRIBUTE_MAPPING+=",attribute.image_digest=assertion.submods.container.image_digest"
ATTRIBUTE_MAPPING+=",attribute.image_reference=assertion.submods.container.image_reference"
ATTRIBUTE_MAPPING+=",attribute.swname=assertion.swname"
ATTRIBUTE_MAPPING+=",attribute.hwmodel=assertion.hwmodel"
ATTRIBUTE_MAPPING+=",attribute.dbgstat=assertion.dbgstat"

# Gate 1. Rejects any attestation that is not a genuine, NON-DEBUG, stable
# Confidential Space image, before a credential exists.
#
# Each term earns its place:
#
#   swname       it is really Confidential Space and not some other COS image.
#
#   dbgstat      THE ONE THAT MATTERS MOST HERE. The debug image family
#                (confidential-space-debug) lets the OPERATOR SSH INTO THE VM
#                and read its memory. Without this term, KMS would release the
#                bank-data key to a VM the operator can read — breaking the
#                design's central promise that the operator only ever holds
#                ciphertext it cannot decrypt. Documented values: the production
#                image reports 'disabled-since-boot', the debug image 'enable'.
#
#   STABLE       Defence in depth on the same property, from the other side.
#                Google's own image database marks a hardened (production) entry
#                [LATEST, STABLE, USABLE] and a debug entry with NO labels, so a
#                debug image cannot carry STABLE. Isolated in
#                ENCLAVE_SUPPORT_ATTR_CHECK because its CEL shape is the one
#                thing here that could not be verified offline — see
#                infra/enclave_config.sh for the alternatives to try.
#
#   secboot      Without Secure Boot the measured boot chain that the
#                image_digest claim rests on is not enforced, so the digest
#                binding would be resting on nothing.
#
#   hwmodel      A real GCP confidential platform (GCP_AMD_SEV, GCP_INTEL_TDX...).
ATTRIBUTE_CONDITION="assertion.swname == 'CONFIDENTIAL_SPACE'"
ATTRIBUTE_CONDITION+=" && assertion.dbgstat == 'disabled-since-boot'"
ATTRIBUTE_CONDITION+=" && ${ENCLAVE_SUPPORT_ATTR_CHECK}"
ATTRIBUTE_CONDITION+=" && assertion.secboot"
ATTRIBUTE_CONDITION+=" && assertion.hwmodel.startsWith('GCP_')"

echo "    condition to install:"
echo "        ${ATTRIBUTE_CONDITION}"

PROVIDER_PATH="${ENCLAVE_WIP_PROVIDER}"
if resource_exists "/providers/${ENCLAVE_WIP_PROVIDER}" "name" \
     iam workload-identity-pools providers describe "${PROVIDER_PATH}" \
     --workload-identity-pool="${ENCLAVE_WIP}" --location="${ENCLAVE_WIP_LOCATION}" \
     --project="${ENCLAVE_PROJECT}"; then
  ok "attestation provider exists: ${ENCLAVE_WIP_PROVIDER} (updating mapping and condition)"
  gcloud iam workload-identity-pools providers update-oidc "${PROVIDER_PATH}" \
    --workload-identity-pool="${ENCLAVE_WIP}" \
    --location="${ENCLAVE_WIP_LOCATION}" \
    --project="${ENCLAVE_PROJECT}" \
    --issuer-uri="https://confidentialcomputing.googleapis.com/" \
    --allowed-audiences="https://sts.googleapis.com" \
    --attribute-mapping="${ATTRIBUTE_MAPPING}" \
    --attribute-condition="${ATTRIBUTE_CONDITION}"
else
  info "creating attestation provider ${ENCLAVE_WIP_PROVIDER}"
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_PATH}" \
    --workload-identity-pool="${ENCLAVE_WIP}" \
    --location="${ENCLAVE_WIP_LOCATION}" \
    --project="${ENCLAVE_PROJECT}" \
    --issuer-uri="https://confidentialcomputing.googleapis.com/" \
    --allowed-audiences="https://sts.googleapis.com" \
    --attribute-mapping="${ATTRIBUTE_MAPPING}" \
    --attribute-condition="${ATTRIBUTE_CONDITION}"
fi

require_state "attestation provider" "/providers/${ENCLAVE_WIP_PROVIDER}" "name" \
  iam workload-identity-pools providers describe "${PROVIDER_PATH}" \
  --workload-identity-pool="${ENCLAVE_WIP}" --location="${ENCLAVE_WIP_LOCATION}" \
  --project="${ENCLAVE_PROJECT}" \
  || fatal "attestation provider was not created"

# Read the condition back from the API and show it. A provider that exists with
# the wrong condition is the failure that silently turns the whole attestation
# story into theatre, so it is printed for a human to read, not just asserted.
ACTUAL_CONDITION="$(api_value attributeCondition \
  iam workload-identity-pools providers describe "${PROVIDER_PATH}" \
  --workload-identity-pool="${ENCLAVE_WIP}" --location="${ENCLAVE_WIP_LOCATION}" \
  --project="${ENCLAVE_PROJECT}")"
echo "    attributeCondition now: ${ACTUAL_CONDITION}"

if [[ "${ACTUAL_CONDITION}" != *"CONFIDENTIAL_SPACE"* ]]; then
  fatal "the provider's attribute condition does not mention CONFIDENTIAL_SPACE; refusing to continue"
fi

# The non-debug check is verified separately and by reading the LIVE condition,
# because this is the term whose absence is silently catastrophic: everything
# would still work, the demo would still pass, and the key would be released to
# a VM the operator can SSH into. A missing CONFIDENTIAL_SPACE term at least
# breaks loudly; a missing dbgstat term breaks nothing and protects nothing.
if [[ "${ACTUAL_CONDITION}" != *"${ENCLAVE_REQUIRED_CONDITION_MARKER}"* ]]; then
  fatal "the provider's attribute condition does not contain the non-debug check \
('${ENCLAVE_REQUIRED_CONDITION_MARKER}'). Without it KMS would release the bank-data \
key to a confidential-space-debug VM, which the operator can SSH into and read \
memory from. Refusing to continue."
fi
ok "condition includes the non-debug check (${ENCLAVE_REQUIRED_CONDITION_MARKER})"

# The STABLE term is reported rather than enforced: its CEL shape is the one
# unverified piece, and if the API rejected the expression the create above would
# already have failed. Saying so keeps a silently-dropped term from passing as a
# clean run.
if [[ "${ACTUAL_CONDITION}" == *"support_attributes"* ]]; then
  ok "condition includes the STABLE support-attribute check"
else
  warn "the live condition has no support_attributes term. dbgstat still blocks \
the debug image, so this is not fatal, but the defence-in-depth check is absent \
— try the alternative CEL forms listed in infra/enclave_config.sh."
fi

step "7. Generate and wrap the in-enclave HMAC key (Stage 4)"
# Generated locally, encrypted to the KMS key, and uploaded as ciphertext. The
# plaintext never touches the bucket and is shredded immediately: an operator
# who can read the keys bucket holds only something KMS will not decrypt for it.
HASH_KEY_PLAIN="$(mktemp)"; HASH_KEY_WRAPPED="$(mktemp)"
trap 'shred -u "${HASH_KEY_PLAIN}" 2>/dev/null || rm -f "${HASH_KEY_PLAIN}"; rm -f "${HASH_KEY_WRAPPED}"' EXIT

if resource_exists "hash-rekey.bin" "name" \
     storage objects describe "gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin" \
     --project="${ENCLAVE_PROJECT}"; then
  ok "wrapped HMAC key already present; leaving it alone (regenerating would orphan every previously computed enclave id)"
else
  head -c 32 /dev/urandom > "${HASH_KEY_PLAIN}"
  gcloud kms encrypt \
    --project="${ENCLAVE_PROJECT}" \
    --location="${ENCLAVE_KMS_LOCATION}" \
    --keyring="${ENCLAVE_KEYRING}" \
    --key="${ENCLAVE_HASH_KEY}" \
    --plaintext-file="${HASH_KEY_PLAIN}" \
    --ciphertext-file="${HASH_KEY_WRAPPED}"
  gcloud storage cp "${HASH_KEY_WRAPPED}" \
    "gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin" --project="${ENCLAVE_PROJECT}"
  require_state "wrapped HMAC key uploaded" "hash-rekey.bin" "name" \
    storage objects describe "gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin" \
    --project="${ENCLAVE_PROJECT}" \
    || fatal "the wrapped HMAC key was not uploaded"
fi

print_bindings

step "Stage 1 infrastructure ready"
cat <<SUMMARY
    edge-wrap key : ${ENCLAVE_KMS_EDGE_KEY_RESOURCE}
    hash key      : ${ENCLAVE_KMS_HASH_KEY_RESOURCE}
    wrapped HMAC  : gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin
    pool          : ${ENCLAVE_WIP} / provider ${ENCLAVE_WIP_PROVIDER}
    workload SA   : ${ENCLAVE_WORKLOAD_SA_EMAIL}
    inbox         : gs://${ENCLAVE_INBOX_BUCKET}
    outbox        : gs://${ENCLAVE_OUTBOX_BUCKET}
    evidence      : gs://${ENCLAVE_EVIDENCE_BUCKET}  (one object per run, refusals included)
    image family  : ${ENCLAVE_IMAGE_FAMILY} (production; the key is refused to the debug family)

    No decrypt right has been granted yet — that names an image digest.
    Next: ./infra/enclave_build.sh   (build the image, print its digest)
     then: ./infra/enclave_pin_digest.sh <digest>   (grant decrypt to that image)
SUMMARY
