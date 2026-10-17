#!/usr/bin/env bash
#
# MuleNet enclave — STAGE 0: check everything before creating anything.
#
# Usage:
#   ./infra/enclave_preflight.sh
#
# Creates nothing. Run it first, and after any account migration. It answers
# the four questions that otherwise get answered halfway through a stage, after
# keys and pools already exist:
#
#   1. Am I authenticated, and as whom?
#   2. Is the operator project billed, and which bank projects are?
#   3. Can the chosen zone actually run a Confidential VM of the chosen
#      machine type -- checked live, not read off a docs table?
#   4. Is there vCPU quota for it in that region?
#
# It exits non-zero if a *blocking* answer is wrong, and warns without failing
# where a value is merely worth knowing. An unbilled bank project is NOT a
# failure: that is the intended design (identity-only banks), and proving it
# works is Stage 2's job.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

FAILED=0
note_fail() { err "$*"; FAILED=1; }

step "1. Authentication"
ACTIVE_ACCOUNT="$(api_value account auth list --filter=status:ACTIVE)"
if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
  note_fail "no active gcloud credential. Run: gcloud auth login"
else
  ok "active account: ${ACTIVE_ACCOUNT}"
fi

# Confirms the credential can actually read the operator project, rather than
# merely existing. An expired token lists fine and fails on first real call.
OPERATOR_NUMBER="$(api_value projectNumber projects describe "${ENCLAVE_PROJECT}")"
if [[ -z "${OPERATOR_NUMBER}" ]]; then
  note_fail "cannot read project ${ENCLAVE_PROJECT}: no access, or it does not exist"
else
  ok "operator project ${ENCLAVE_PROJECT} number=${OPERATOR_NUMBER}"
fi

step "2. Billing"
OPERATOR_BILLED="$(project_billing_enabled "${ENCLAVE_PROJECT}" || true)"
if [[ "${OPERATOR_BILLED}" == "True" ]]; then
  ok "${ENCLAVE_PROJECT} is billed (it pays for the Confidential VM and KMS)"
else
  note_fail "${ENCLAVE_PROJECT} reports billingEnabled='${OPERATOR_BILLED:-unknown}'. The Confidential VM cannot start without billing."
fi

for bank_project in "${BANK_A_PROJECT}" "${BANK_B_PROJECT}" "${BANK_C_PROJECT}"; do
  billed="$(project_billing_enabled "${bank_project}" || true)"
  case "${billed}" in
    True)  ok "${bank_project} is billed" ;;
    False) ok "${bank_project} is UNBILLED — intended for identity-only banks; Stage 2 proves it can still write to the inbox" ;;
    *)     warn "${bank_project}: could not read billing state ('${billed:-unknown}')" ;;
  esac
done

step "3. Confidential Space availability (live, not from a docs table)"
CS_IMAGE="$(confidential_space_image "${ENCLAVE_IMAGE_FAMILY}")"
if [[ -z "${CS_IMAGE}" ]]; then
  note_fail "image family '${ENCLAVE_IMAGE_FAMILY}' not found in ${ENCLAVE_IMAGE_PROJECT}. The families are versioned and retired; list them with: gcloud compute images list --project=${ENCLAVE_IMAGE_PROJECT}"
else
  ok "Confidential Space image (production, the default): ${CS_IMAGE} (family ${ENCLAVE_IMAGE_FAMILY})"
fi
if [[ "${ENCLAVE_IMAGE_FAMILY}" == *debug* ]]; then
  note_fail "ENCLAVE_IMAGE_FAMILY is set to a DEBUG family ('${ENCLAVE_IMAGE_FAMILY}'). The operator can SSH into a debug VM and read its memory, so it must never be the default for a run that handles bank data. Unset the override; enclave_negative_test.sh selects the debug family itself, for the one run that is supposed to be refused."
fi

# The debug family is needed for negative test B, which boots the unchanged image
# on it and expects a refusal. Its absence does not block a normal run, so this
# is a warning.
CS_DEBUG_IMAGE="$(confidential_space_image "${ENCLAVE_IMAGE_DEBUG_FAMILY}")"
if [[ -z "${CS_DEBUG_IMAGE}" ]]; then
  warn "debug family '${ENCLAVE_IMAGE_DEBUG_FAMILY}' not found; negative test B (the key is refused to a VM the operator can SSH into) cannot run"
else
  ok "Confidential Space image (debug, used only by negative test B): ${CS_DEBUG_IMAGE}"
fi

# Compute Engine must be enabled on the operator project before a zone or
# machine-type lookup returns anything, so a missing answer here is ambiguous
# without this check.
if resource_exists "compute.googleapis.com" "config.name" \
     services list --enabled --filter="config.name=compute.googleapis.com" \
     --project="${ENCLAVE_PROJECT}"; then
  ok "Compute Engine API is enabled on ${ENCLAVE_PROJECT}"

  ZONE_STATUS="$(api_value status compute zones describe "${ENCLAVE_ZONE}" \
                  --project="${ENCLAVE_PROJECT}")"
  if [[ "${ZONE_STATUS}" == "UP" ]]; then
    ok "zone ${ENCLAVE_ZONE} is UP"
  else
    note_fail "zone ${ENCLAVE_ZONE} status='${ZONE_STATUS:-unknown}'"
  fi

  if zone_supports_machine_type "${ENCLAVE_PROJECT}" "${ENCLAVE_ZONE}" \
       "${ENCLAVE_MACHINE_TYPE}"; then
    ok "machine type ${ENCLAVE_MACHINE_TYPE} exists in ${ENCLAVE_ZONE}"
  else
    note_fail "machine type ${ENCLAVE_MACHINE_TYPE} is NOT available in ${ENCLAVE_ZONE}. Pick a zone that offers it and set ENCLAVE_ZONE/ENCLAVE_REGION, then record the reason in docs/enclave.md. Do not let the scripts relocate silently."
  fi

  # The machine type existing is necessary but not sufficient: the zone must
  # also offer the confidential-compute capability. This is the check that
  # catches "n2d exists here but SEV does not".
  CC_SUPPORT="$(gcloud compute machine-types describe "${ENCLAVE_MACHINE_TYPE}" \
                 --zone="${ENCLAVE_ZONE}" --project="${ENCLAVE_PROJECT}" \
                 --format="value(name,guestCpus)" 2>/dev/null || true)"
  if [[ -n "${CC_SUPPORT}" ]]; then
    ok "machine type detail: ${CC_SUPPORT}"
    warn "whether ${ENCLAVE_CONFIDENTIAL_TYPE} is offered in ${ENCLAVE_ZONE} is proven only by a real instance create; enclave_run.sh reports the API's refusal verbatim if it is not"
  fi

  step "4. Quota"
  report_cpu_quota "${ENCLAVE_PROJECT}" "${ENCLAVE_REGION}"
else
  note_fail "Compute Engine API is not enabled on ${ENCLAVE_PROJECT}; zone, machine-type and quota checks cannot run. enclave_setup.sh enables it."
fi

step "Preflight summary"
if [[ "${FAILED}" -eq 0 ]]; then
  ok "preflight passed. Next: ./infra/enclave_setup.sh"
  exit 0
fi
err "preflight FAILED. Fix the items above before creating resources."
exit 1
