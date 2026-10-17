#!/usr/bin/env bash
#
# MuleNet enclave — tear down the billable parts.
#
# Usage:
#   ./infra/enclave_teardown.sh              # VMs only (the safe default)
#   ./infra/enclave_teardown.sh --all        # also buckets, repo, pool, keys
#
# The default deletes only Confidential VMs, because those are the only thing
# that costs money by the hour and the only thing worth removing reflexively.
# KMS keys, the workload identity pool and the buckets are NOT removed by
# default:
#
#   * a destroyed KMS key version cannot be undestroyed, and the wrapped
#     in-enclave HMAC key becomes permanently unreadable -- every enclave
#     identifier ever computed is then unreproducible,
#   * a deleted workload identity pool name is unavailable for ~30 days, so a
#     careless teardown blocks the rebuild it was meant to enable,
#   * the inbox may hold a bank's bundle that nobody else can re-create.
#
# --all therefore asks for confirmation and says exactly what is unrecoverable.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

MODE="${1:-vms}"

step "Confidential VMs in ${ENCLAVE_ZONE}"
# Filtered by name prefix so this cannot delete an unrelated instance.
VMS="$(gcloud compute instances list \
  --project="${ENCLAVE_PROJECT}" \
  --zones="${ENCLAVE_ZONE}" \
  --filter="name~^${ENCLAVE_VM_NAME}" \
  --format="value(name)" 2>/dev/null || true)"

if [[ -z "${VMS}" ]]; then
  ok "no enclave VMs running (nothing billable left behind)"
else
  while read -r vm; do
    [[ -n "${vm}" ]] || continue
    info "deleting ${vm}"
    gcloud compute instances delete "${vm}" \
      --zone="${ENCLAVE_ZONE}" --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || true
    still="$(api_value name compute instances describe "${vm}" \
      --zone="${ENCLAVE_ZONE}" --project="${ENCLAVE_PROJECT}")"
    if [[ -n "${still}" ]]; then
      err "${vm} STILL EXISTS — delete it by hand, it is costing money"
    else
      ok "${vm} deleted (verified absent)"
    fi
  done <<< "${VMS}"
fi

if [[ "${MODE}" != "--all" ]]; then
  cat <<NOTE

    Left in place (not billable by the hour, and destructive to remove):
      KMS keyring  ${ENCLAVE_KEYRING}   (destroying a version is irreversible)
      WIP pool     ${ENCLAVE_WIP}       (name unavailable ~30 days after delete)
      buckets      ${ENCLAVE_INBOX_BUCKET}, ${ENCLAVE_OUTBOX_BUCKET}, ${ENCLAVE_KEYS_BUCKET}
                   ${ENCLAVE_EVIDENCE_BUCKET}
      repo         ${ENCLAVE_ARTIFACT_REPO}

    Run with --all to remove those too.
NOTE
  exit 0
fi

step "--all: this removes things that cannot be undone"
cat <<WARNING
    About to delete:
      - buckets ${ENCLAVE_INBOX_BUCKET}, ${ENCLAVE_OUTBOX_BUCKET}, ${ENCLAVE_KEYS_BUCKET}
        including gs://${ENCLAVE_KEYS_BUCKET}/hash-rekey.bin
      - gs://${ENCLAVE_EVIDENCE_BUCKET} — EVERY RUN'S EVIDENCE, including the
        negative tests' captured refusals. Export anything you need to keep
        BEFORE running this; the refusals are the Stage 1 proof.
      - artifact repo ${ENCLAVE_ARTIFACT_REPO} and every image in it
      - workload identity pool ${ENCLAVE_WIP} (name blocked for ~30 days)
      - KMS key VERSIONS for ${ENCLAVE_EDGE_KEY} and ${ENCLAVE_HASH_KEY}

    UNRECOVERABLE CONSEQUENCE: destroying the hash-rekey key version makes every
    enclave identifier ever computed impossible to reproduce. Any alert a bank
    already holds becomes permanently unjoinable to a future run.
WARNING
read -r -p "    Type DELETE to proceed: " answer
[[ "${answer}" == "DELETE" ]] || { info "aborted; nothing else was removed"; exit 0; }

for bucket in "${ENCLAVE_INBOX_BUCKET}" "${ENCLAVE_OUTBOX_BUCKET}" "${ENCLAVE_KEYS_BUCKET}" \
             "${ENCLAVE_EVIDENCE_BUCKET}"; do
  info "deleting gs://${bucket}"
  gcloud storage rm --recursive "gs://${bucket}" \
    --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || \
    warn "could not delete gs://${bucket}"
done

info "deleting artifact repo ${ENCLAVE_ARTIFACT_REPO}"
gcloud artifacts repositories delete "${ENCLAVE_ARTIFACT_REPO}" \
  --location="${ENCLAVE_REGION}" --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || \
  warn "could not delete the artifact repo"

info "deleting workload identity pool ${ENCLAVE_WIP}"
gcloud iam workload-identity-pools delete "${ENCLAVE_WIP}" \
  --location="${ENCLAVE_WIP_LOCATION}" --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || \
  warn "could not delete the pool"

for key in "${ENCLAVE_EDGE_KEY}" "${ENCLAVE_HASH_KEY}"; do
  VERSIONS="$(gcloud kms keys versions list --key="${key}" \
    --keyring="${ENCLAVE_KEYRING}" --location="${ENCLAVE_KMS_LOCATION}" \
    --project="${ENCLAVE_PROJECT}" --filter="state:ENABLED" \
    --format="value(name)" 2>/dev/null || true)"
  while read -r version; do
    [[ -n "${version}" ]] || continue
    info "destroying ${version}"
    gcloud kms keys versions destroy "${version##*/}" \
      --key="${key}" --keyring="${ENCLAVE_KEYRING}" \
      --location="${ENCLAVE_KMS_LOCATION}" --project="${ENCLAVE_PROJECT}" \
      --quiet >/dev/null 2>&1 || warn "could not destroy ${version}"
  done <<< "${VERSIONS}"
done

ok "teardown complete. The keyring itself remains: KMS keyrings cannot be deleted."
