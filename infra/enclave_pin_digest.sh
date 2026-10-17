#!/usr/bin/env bash
#
# MuleNet enclave — grant KMS key release to ONE container image digest.
#
# Usage:
#   ./infra/enclave_pin_digest.sh sha256:<digest>          # grant
#   ./infra/enclave_pin_digest.sh --revoke sha256:<digest> # revoke
#   ./infra/enclave_pin_digest.sh --list                   # show who can decrypt
#
# This is the script that makes the Stage 1 claim literally true. The IAM member
# it binds is not a service account but an ATTESTATION ATTRIBUTE:
#
#   principalSet://iam.googleapis.com/projects/<NUM>/locations/global/
#     workloadIdentityPools/<POOL>/attribute.image_digest/sha256:<DIGEST>
#
# Only a workload whose Confidential Space attestation carries that image digest
# can assume that principal, so only that image can decrypt. No key material is
# copied anywhere; the grant is the release policy.
#
# Consequence, stated because it is the point: rebuild the image with one line
# changed and its digest changes, so it is a different principal, so KMS refuses.
# That refusal is the negative test's evidence.
#
# UNEXECUTED: not run; this session had no Google Cloud credentials.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

PROJECT_NUMBER="$(project_number "${ENCLAVE_PROJECT}")"
POOL_PATH="projects/${PROJECT_NUMBER}/locations/${ENCLAVE_WIP_LOCATION}/workloadIdentityPools/${ENCLAVE_WIP}"

digest_member() {
  echo "principalSet://iam.googleapis.com/${POOL_PATH}/attribute.image_digest/$1"
}

show_decrypters() {
  local key_resource="$1" label="$2"
  step "Who can decrypt ${label}"
  # Read the live policy. The listing is the authority on what is permitted --
  # not this script's own log, and not a memory of what was granted.
  gcloud kms keys get-iam-policy "${key_resource}" \
    --project="${ENCLAVE_PROJECT}" \
    --flatten="bindings[].members[]" \
    --filter="bindings.role:roles/cloudkms.cryptoKeyDecrypter" \
    --format="table[no-heading](bindings.members)" 2>/dev/null \
    || warn "could not read the IAM policy on ${label}"
}

case "${1:-}" in
  --list)
    show_decrypters "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" "edge-wrap"
    show_decrypters "${ENCLAVE_KMS_HASH_KEY_RESOURCE}" "hash-rekey"
    exit 0
    ;;
  --revoke)
    DIGEST="${2:-}"
    ACTION="revoke"
    ;;
  "")
    fatal "usage: $0 sha256:<digest> | --revoke sha256:<digest> | --list"
    ;;
  *)
    DIGEST="$1"
    ACTION="grant"
    ;;
esac

[[ -n "${DIGEST:-}" ]] || fatal "a digest is required"
# A tag here instead of a digest would create a binding that silently never
# matches, and the enclave would look "broken" rather than "misconfigured".
[[ "${DIGEST}" == sha256:* ]] || fatal "expected sha256:<hex>, got '${DIGEST}'"
[[ "${DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || fatal "malformed digest '${DIGEST}'"

MEMBER="$(digest_member "${DIGEST}")"

for key_pair in \
  "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}|edge-wrap" \
  "${ENCLAVE_KMS_HASH_KEY_RESOURCE}|hash-rekey"
do
  KEY_RESOURCE="${key_pair%%|*}"
  KEY_LABEL="${key_pair##*|}"
  if [[ "${ACTION}" == "grant" ]]; then
    step "Granting decrypt on ${KEY_LABEL} to image ${DIGEST}"
    gcloud kms keys add-iam-policy-binding "${KEY_RESOURCE}" \
      --project="${ENCLAVE_PROJECT}" \
      --member="${MEMBER}" \
      --role="roles/cloudkms.cryptoKeyDecrypter" \
      --condition=None --quiet >/dev/null
    record_binding "kms:${KEY_LABEL}" "attribute.image_digest/${DIGEST}" \
      "roles/cloudkms.cryptoKeyDecrypter"
  else
    step "Revoking decrypt on ${KEY_LABEL} from image ${DIGEST}"
    gcloud kms keys remove-iam-policy-binding "${KEY_RESOURCE}" \
      --project="${ENCLAVE_PROJECT}" \
      --member="${MEMBER}" \
      --role="roles/cloudkms.cryptoKeyDecrypter" \
      --condition=None --quiet >/dev/null
  fi

  # Verify against the LIVE POLICY, never against the exit code above. An
  # add-iam-policy-binding that races another writer can return 0 having lost
  # the write, and a stage that trusted the status would then report a key
  # release that does not exist.
  POLICY_MEMBERS="$(gcloud kms keys get-iam-policy "${KEY_RESOURCE}" \
    --project="${ENCLAVE_PROJECT}" \
    --flatten="bindings[].members[]" \
    --filter="bindings.role:roles/cloudkms.cryptoKeyDecrypter" \
    --format="value(bindings.members)" 2>/dev/null || true)"

  if [[ "${ACTION}" == "grant" ]]; then
    if [[ "${POLICY_MEMBERS}" == *"attribute.image_digest/${DIGEST}"* ]]; then
      ok "${KEY_LABEL}: policy now names ${DIGEST}"
    else
      fatal "${KEY_LABEL}: the binding is NOT in the live policy after the add. Policy currently lists: ${POLICY_MEMBERS:-<none>}"
    fi
  else
    if [[ "${POLICY_MEMBERS}" == *"attribute.image_digest/${DIGEST}"* ]]; then
      fatal "${KEY_LABEL}: ${DIGEST} is STILL in the live policy after the remove"
    fi
    ok "${KEY_LABEL}: ${DIGEST} is no longer in the policy"
  fi
done

print_bindings
show_decrypters "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" "edge-wrap"

cat <<NOTE

    Least privilege check worth doing by eye: the listing above should contain
    ONLY attribute.image_digest principals. If a plain serviceAccount: appears
    with cryptoKeyDecrypter, that account can read every bank's data without
    attestation, and the Stage 1 claim is void.
NOTE
