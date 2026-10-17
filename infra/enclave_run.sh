#!/usr/bin/env bash
#
# MuleNet enclave — start a Confidential VM, run the workload, collect evidence,
# and shut it down.
#
# Usage:
#   ./infra/enclave_run.sh v1 run-1                        # real run
#   ./infra/enclave_run.sh v1 run-1 production             # named attempt
#   ENCLAVE_IMAGE_FAMILY=confidential-space-debug \
#     ./infra/enclave_run.sh v1 run-1 debug                # the 2nd negative test
#
# The VM is deleted by an EXIT trap, so it goes away even if this script fails
# or is interrupted. A Confidential VM left running is the one way this project
# can quietly spend real money, and "remember to delete it" is not a control.
#
# EVIDENCE COMES FROM A GCS OBJECT, not from the serial console and not from an
# exit code. The workload writes one JSON result object per run to
# ENCLAVE_EVIDENCE_BUCKET, including runs where KMS refused the key, and this
# script reads its `outcome` field.
#
# Why that works even when key release is refused: the workload writes evidence
# with the VM's ATTACHED service account (via the metadata server), while key
# release uses the ATTESTED federated identity. The two are independent, so a
# refused run can still report itself. Reading the serial console used to be the
# alternative, and it only worked on the debug image family — the very family
# that must now be refused the key.
#
# UNEXECUTED: not run; this session had no Google Cloud credentials.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

TAG="${1:-v1}"
RUN_ID="${2:-run-1}"
ATTEMPT="${3:-run}"
EXTRA_CMD="${4:-}"
# STS maps google.subject to this VM's full self-link
# (https://www.googleapis.com/compute/v1/projects/<P>/zones/<Z>/instances/<NAME>)
# and Google caps a mapped attribute at 127 bytes. Observed live: with
# TAG=tampered ATTEMPT=tampered the old "${ENCLAVE_VM_NAME}-${TAG}-${ATTEMPT}-…"
# name produced a 129-byte URL, and STS refused the exchange with
# "google.subject exceeds the 127 bytes limit" -- a config bug, not the
# intended kms_decrypt_refused. The name is now a fixed width regardless of
# TAG/ATTEMPT/RUN_ID length; those go on the instance as labels instead, where
# nothing reads them into an STS-mapped claim.
VM_NAME="mulenet-enc-$(date -u +%y%m%d%H%M%S)-$((RANDOM % 0xffff))"
EVIDENCE_OBJECT="runs/${RUN_ID}/${ATTEMPT}.json"

info "image family: ${ENCLAVE_IMAGE_FAMILY}"
if [[ "${ENCLAVE_IMAGE_FAMILY}" == *debug* ]]; then
  warn "this is the DEBUG image family. The operator can SSH into this VM, so the"
  warn "attestation condition should REFUSE it the bank-data key. Expect a refusal;"
  warn "a success here would mean the condition is not doing its job."
fi

#: Per-hour on-demand rate for the machine type, used only to turn measured
#: uptime into an estimate. NOT verified against Google's price list in the
#: session that wrote this script (the pricing docs were unreachable), so it is
#: an input to be checked, not an authority. Confirm before quoting any number.
ENCLAVE_VM_HOURLY_USD="${ENCLAVE_VM_HOURLY_USD:-0.0}"

DIGEST_FILE="${REPO_ROOT}/.enclave-digest-${TAG}"
if [[ -f "${DIGEST_FILE}" ]]; then
  DIGEST="$(cat "${DIGEST_FILE}")"
  IMAGE_REF="${ENCLAVE_IMAGE_BASE}@${DIGEST}"
  info "running image by digest: ${DIGEST}"
else
  IMAGE_REF="${ENCLAVE_IMAGE_BASE}:${TAG}"
  warn "no ${DIGEST_FILE}; running by TAG ${TAG}. A tag is mutable — for the negative test always run by digest."
fi

PROJECT_NUMBER="$(project_number "${ENCLAVE_PROJECT}")"
WIP_AUDIENCE="//iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/${ENCLAVE_WIP_LOCATION}/workloadIdentityPools/${ENCLAVE_WIP}/providers/${ENCLAVE_WIP_PROVIDER}"

# --- teardown ---------------------------------------------------------------
STARTED_EPOCH=0
cleanup() {
  local status=$?
  if [[ "${STARTED_EPOCH}" -gt 0 ]]; then
    local ended_epoch elapsed
    ended_epoch="$(date -u +%s)"
    elapsed=$(( ended_epoch - STARTED_EPOCH ))
    step "Shutting down and reporting cost"
    echo "    VM        : ${VM_NAME}"
    echo "    uptime    : ${elapsed}s ($(awk "BEGIN{printf \"%.2f\", ${elapsed}/3600}") h)"
    if [[ "${ENCLAVE_VM_HOURLY_USD}" != "0.0" ]]; then
      echo "    estimate  : USD $(awk "BEGIN{printf \"%.4f\", ${elapsed}/3600*${ENCLAVE_VM_HOURLY_USD}}") at ${ENCLAVE_VM_HOURLY_USD}/h (RATE UNVERIFIED — confirm against the price list)"
    else
      echo "    estimate  : not computed. Set ENCLAVE_VM_HOURLY_USD to the verified"
      echo "                on-demand rate for ${ENCLAVE_MACHINE_TYPE} + Confidential"
      echo "                Computing in ${ENCLAVE_REGION}. No rate is hardcoded here"
      echo "                because an unverified price is worse than none."
    fi
  fi
  info "deleting ${VM_NAME} (runs even on failure or interrupt)"
  gcloud compute instances delete "${VM_NAME}" \
    --zone="${ENCLAVE_ZONE}" --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || true

  # Confirm by state that it is really gone. A delete that returned 0 while
  # leaving the instance running is exactly the case that costs money silently.
  local still
  still="$(api_value name compute instances describe "${VM_NAME}" \
    --zone="${ENCLAVE_ZONE}" --project="${ENCLAVE_PROJECT}")"
  if [[ -n "${still}" ]]; then
    err "INSTANCE STILL EXISTS: ${still}. Delete it by hand NOW:"
    err "  gcloud compute instances delete ${VM_NAME} --zone=${ENCLAVE_ZONE} --project=${ENCLAVE_PROJECT}"
  else
    ok "instance deleted (verified absent from the API)"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

# --- metadata ---------------------------------------------------------------
# Confidential Space metadata keys, confirmed against Google's own published
# scripts (codelabs/mig_cs_codelab/create_launch_mig.sh uses
# tee-image-reference and tee-container-log-redirect).
#
# The ^~^ prefix sets ~ as the delimiter so values may contain commas, which
# MULENET_ENCLAVE_BANK_IDS does.
#
# Nothing secret is passed here: VM metadata is readable by the operator, so
# every secret reaches the enclave only as ciphertext KMS must release a key for.
METADATA="^~^tee-image-reference=${IMAGE_REF}"
# Log redirect only on the debug family: the production image's launch policy
# defaults log_redirect to debugonly, and container stdout is operator-readable.
if [[ "${ENCLAVE_IMAGE_FAMILY}" == *debug* ]]; then
  METADATA+="~tee-container-log-redirect=true"
fi
METADATA+="~tee-restart-policy=Never"
METADATA+="~tee-env-MULENET_ENCLAVE_PROJECT=${ENCLAVE_PROJECT}"
METADATA+="~tee-env-MULENET_ENCLAVE_INBOX_BUCKET=${ENCLAVE_INBOX_BUCKET}"
METADATA+="~tee-env-MULENET_ENCLAVE_OUTBOX_BUCKET=${ENCLAVE_OUTBOX_BUCKET}"
METADATA+="~tee-env-MULENET_ENCLAVE_EVIDENCE_BUCKET=${ENCLAVE_EVIDENCE_BUCKET}"
METADATA+="~tee-env-MULENET_ENCLAVE_ATTEMPT=${ATTEMPT}"
METADATA+="~tee-env-MULENET_ENCLAVE_KMS_KEY=${ENCLAVE_KMS_EDGE_KEY_RESOURCE}"
METADATA+="~tee-env-MULENET_ENCLAVE_HASH_KMS_KEY=${ENCLAVE_KMS_HASH_KEY_RESOURCE}"
METADATA+="~tee-env-MULENET_ENCLAVE_HASH_KEY_URI=gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin"
METADATA+="~tee-env-MULENET_ENCLAVE_WIP_AUDIENCE=${WIP_AUDIENCE}"
METADATA+="~tee-env-MULENET_ENCLAVE_WORKLOAD_SA=${ENCLAVE_WORKLOAD_SA_EMAIL}"
METADATA+="~tee-env-MULENET_ENCLAVE_BANK_IDS=${BANK_IDS}"
METADATA+="~tee-env-MULENET_ENCLAVE_RUN_ID=${RUN_ID}"
# Per-bank outbox public keys (see enclave/outbox_crypto.py): NOT secrets, but
# required -- EnclaveConfig.from_env() fails closed if a bank in BANK_IDS has
# no key on file, rather than write that bank's alerts unencrypted. Generate
# with: python -m infra.enclave_outbox_keygen --bank-id bank-a
for _bank_id_var in ${BANK_IDS//,/ }; do
  _env_suffix="$(echo "${_bank_id_var}" | tr '[:lower:]-' '[:upper:]_')"
  _pubkey_var="MULENET_ENCLAVE_OUTBOX_PUBKEY_${_env_suffix}"
  if [[ -z "${!_pubkey_var:-}" ]]; then
    fatal "${_pubkey_var} is not set. Run infra/enclave_outbox_keygen.py for ${_bank_id_var} and export its public key."
  fi
  METADATA+="~tee-env-${_pubkey_var}=${!_pubkey_var}"
done
if [[ -n "${MULENET_ENCLAVE_FLAGGED_BANK_A:-}" ]]; then
  METADATA+="~tee-env-MULENET_ENCLAVE_FLAGGED_BANK_A=${MULENET_ENCLAVE_FLAGGED_BANK_A}"
fi
if [[ -n "${MULENET_ENCLAVE_FLAGGED_BANK_B:-}" ]]; then
  METADATA+="~tee-env-MULENET_ENCLAVE_FLAGGED_BANK_B=${MULENET_ENCLAVE_FLAGGED_BANK_B}"
fi
if [[ -n "${MULENET_ENCLAVE_FLAGGED_BANK_C:-}" ]]; then
  METADATA+="~tee-env-MULENET_ENCLAVE_FLAGGED_BANK_C=${MULENET_ENCLAVE_FLAGGED_BANK_C}"
fi
if [[ -n "${EXTRA_CMD}" ]]; then
  METADATA+="~tee-cmd=[\"${EXTRA_CMD}\"]"
fi

step "Creating Confidential VM ${VM_NAME}"
info "zone=${ENCLAVE_ZONE} machine=${ENCLAVE_MACHINE_TYPE} confidential=${ENCLAVE_CONFIDENTIAL_TYPE}"
STARTED_EPOCH="$(date -u +%s)"

# --shielded-secure-boot is required: the provider's attribute condition demands
# assertion.secboot, so without it the attestation is rejected and the run fails
# for a reason that looks like a KMS problem but is not.
#
# The evidence path is fixed per (run id, attempt), so a re-run finds the last
# run's object at once. Observed live: a re-run reported the previous image's
# verdict. Record the generation now and accept only a different one below.
STALE_GENERATION="$(api_value generation storage objects describe \
  "gs://${ENCLAVE_EVIDENCE_BUCKET}/${EVIDENCE_OBJECT}" --project="${ENCLAVE_PROJECT}")"
if [[ -n "${STALE_GENERATION}" ]]; then
  info "an older evidence object exists (generation ${STALE_GENERATION}); it will be ignored"
fi

if ! gcloud compute instances create "${VM_NAME}" \
    --project="${ENCLAVE_PROJECT}" \
    --zone="${ENCLAVE_ZONE}" \
    --machine-type="${ENCLAVE_MACHINE_TYPE}" \
    --confidential-compute-type="${ENCLAVE_CONFIDENTIAL_TYPE}" \
    --shielded-secure-boot \
    --maintenance-policy=TERMINATE \
    --image-project="${ENCLAVE_IMAGE_PROJECT}" \
    --image-family="${ENCLAVE_IMAGE_FAMILY}" \
    --service-account="${ENCLAVE_WORKLOAD_SA_EMAIL}" \
    --labels="tag=${TAG},attempt=${ATTEMPT}" \
    --scopes=cloud-platform \
    --no-address \
    --metadata="${METADATA}" 2>&1 | tee "${TMPDIR:-/tmp}/enclave-create.log"; then
  err "instance create failed. The API's own message is above — if it mentions"
  err "the confidential compute type or the machine type, ${ENCLAVE_ZONE} cannot"
  err "run ${ENCLAVE_CONFIDENTIAL_TYPE} on ${ENCLAVE_MACHINE_TYPE}. Pick another zone,"
  err "set ENCLAVE_ZONE/ENCLAVE_REGION, and record the reason in docs/enclave.md."
  exit 1
fi

# --no-address means no external IP: the workload needs no inbound access and
# reaches Google APIs over Private Google Access. One less exposed surface on a
# VM that holds three banks' joined data in memory.

require_state "instance running" "RUNNING" "status" \
  compute instances describe "${VM_NAME}" --zone="${ENCLAVE_ZONE}" \
  --project="${ENCLAVE_PROJECT}" \
  || fatal "the instance is not RUNNING"

step "Waiting for the evidence object gs://${ENCLAVE_EVIDENCE_BUCKET}/${EVIDENCE_OBJECT}"
# tee-restart-policy=Never means the container runs once. Poll for the EVIDENCE
# OBJECT rather than for a log line: the object is written on every terminal
# path, including a refusal, so its appearance is the signal that the run
# finished. A log line is not available at all on the production image family.
DEADLINE=$(( $(date -u +%s) + ${ENCLAVE_RUN_TIMEOUT_SECONDS:-600} ))
EVIDENCE_LOCAL="${TMPDIR:-/tmp}/enclave-evidence-${VM_NAME}.json"
FOUND=0

while [[ "$(date -u +%s)" -lt "${DEADLINE}" ]]; do
  sleep 20
  # Existence is checked by reading the object's name back from the API, not by
  # the exit status of a copy.
  OBJ_GENERATION="$(api_value generation storage objects describe \
    "gs://${ENCLAVE_EVIDENCE_BUCKET}/${EVIDENCE_OBJECT}" --project="${ENCLAVE_PROJECT}")"
  if [[ -n "${OBJ_GENERATION}" && "${OBJ_GENERATION}" != "${STALE_GENERATION}" ]]; then
    # Read that exact generation, so the verdict is the object just checked.
    if gcloud storage cat "gs://${ENCLAVE_EVIDENCE_BUCKET}/${EVIDENCE_OBJECT}#${OBJ_GENERATION}" \
         --project="${ENCLAVE_PROJECT}" > "${EVIDENCE_LOCAL}" 2>/dev/null; then
      FOUND=1
      ok "evidence object appeared"
      break
    fi
  fi
  info "  ... still waiting ($(( DEADLINE - $(date -u +%s) ))s left)"
done

step "Evidence"
if [[ "${FOUND}" -eq 1 && -s "${EVIDENCE_LOCAL}" ]]; then
  sed 's/^/    /' "${EVIDENCE_LOCAL}"
else
  warn "no evidence object was written within the timeout."
  warn "That is INCONCLUSIVE, not a refusal and not a pass: the VM may have failed"
  warn "to boot, or failed before it could write. On the debug family only, the"
  warn "serial console may add detail:"
  warn "  gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${ENCLAVE_ZONE} --project=${ENCLAVE_PROJECT}"
  exit 5
fi

# Read the verdict out of the JSON with python3, not grep: a substring match on
# JSON would happily find "key_release_refused" inside a refusal MESSAGE and
# misreport a successful run as refused.
read_field() {
  python3 -c "
import json,sys
with open(sys.argv[1]) as fh:
    doc = json.load(fh)
value = doc.get(sys.argv[2])
print('' if value is None else value)
" "${EVIDENCE_LOCAL}" "$1" 2>/dev/null || true
}

OUTCOME="$(read_field outcome)"
KEY_OK="$(read_field key_release_ok)"
REFUSAL_KIND="$(read_field refusal_kind)"
REFUSAL_MESSAGE="$(read_field refusal_message)"
DEBUG_IMAGE="$(read_field running_debug_image)"

step "Verdict (from the evidence object's outcome field)"
echo "    outcome            : ${OUTCOME:-<missing>}"
echo "    key_release_ok     : ${KEY_OK:-<missing>}"
echo "    running_debug_image: ${DEBUG_IMAGE:-<unknown>}"

case "${OUTCOME}" in
  key_release_refused)
    err "KMS DID NOT RELEASE THE KEY to this image."
    echo "    refusal_kind: ${REFUSAL_KIND:-<unclassified>}"
    echo "    refusal, verbatim:"
    echo "${REFUSAL_MESSAGE}" | sed 's/^/        /'
    echo
    case "${REFUSAL_KIND}" in
      sts_exchange_refused)
        echo "    The ATTRIBUTE CONDITION rejected this attestation before any key was"
        echo "    involved. For the debug-image negative test this is the expected"
        echo "    result and the message above is the evidence."
        ;;
      kms_decrypt_refused)
        echo "    The attestation was acceptable but the KEY'S IAM BINDING does not name"
        echo "    this image's digest. For the tampered-image negative test this is the"
        echo "    expected result and the message above is the evidence."
        ;;
      *)
        warn "the refusal could not be classified. Read it before describing it: an"
        warn "unclassified refusal supports NEITHER negative test's specific claim."
        ;;
    esac
    exit 4
    ;;
  completed)
    ok "the workload completed and the key was released"
    if [[ "${DEBUG_IMAGE}" == "True" ]]; then
      err "BUT the evidence says this VM was running a DEBUG image."
      err "That means the attestation condition did NOT block a VM the operator can"
      err "SSH into. Treat this as a SECURITY FAILURE, not a passing run:"
      err "  ./infra/enclave_setup.sh   # reinstall the condition"
      err "  ./infra/enclave_pin_digest.sh --list"
      exit 6
    fi
    echo "    Outbox objects written this run:"
    gcloud storage ls "gs://${ENCLAVE_OUTBOX_BUCKET}/outbox/**/${RUN_ID}/*" \
      --project="${ENCLAVE_PROJECT}" 2>/dev/null | sed 's/^/        /' || \
      warn "could not list the outbox"
    exit 0
    ;;
  self_check_passed)
    ok "self-check passed ON THE CONFIDENTIAL VM: the image boots and the join,"
    ok "clustering and output-policy logic all run there."
    echo "    Note: KMS was NOT contacted, so this says nothing about key release."
    if [[ "${DEBUG_IMAGE}" == "True" ]]; then
      warn "this was a debug-family VM. Fine for a boot check; it must still be"
      warn "REFUSED the key on a real run."
    fi
    exit 0
    ;;
  self_check_failed)
    err "the self-check FAILED on the Confidential VM. Fix the workload before"
    err "drawing any conclusion about key release."
    exit 1
    ;;
  no_bundles_found)
    warn "the key was released but no bank bundles were in the inbox for ${RUN_ID}."
    warn "Stage 1 is satisfied; there was nothing to score."
    exit 0
    ;;
  *)
    err "unexpected outcome '${OUTCOME:-<missing>}'. Treat as INCONCLUSIVE."
    exit 5
    ;;
esac
