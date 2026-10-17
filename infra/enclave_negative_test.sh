#!/usr/bin/env bash
#
# MuleNet enclave — STAGE 1 NEGATIVE TESTS (the showpiece).
#
# Usage:
#   ./infra/enclave_negative_test.sh
#
# Proves TWO SEPARATE properties of the key-release policy. Neither implies the
# other, and a demo that showed only the first would be claiming more than it
# proved:
#
#   A. THE KEY FOLLOWS THE CODE.
#      Change one line of the scorer, rebuild, and KMS refuses the key — because
#      the IAM binding names the image digest and the digest moved.
#
#   B. THE KEY REQUIRES NO OPERATOR ACCESS.
#      Take the UNCHANGED image, whose digest IS pinned in KMS, and boot it on
#      the confidential-space-DEBUG family. It is refused anyway, because the
#      debug image lets the operator SSH in and read memory. Without this, the
#      operator could obtain the banks' plaintext while running the blessed code.
#
# Test B is the one that used to be impossible to state honestly: the debug
# family was previously the DEFAULT, chosen so container logs reached the serial
# console. Evidence now comes from a GCS object instead, so the production family
# is the default and the debug family is something to be refused.
#
# ---------------------------------------------------------------------------
# What counts as evidence
# ---------------------------------------------------------------------------
# The `outcome` and `refusal_kind` fields of each run's evidence object in
# gs://${ENCLAVE_EVIDENCE_BUCKET}. NOT an exit code — this project has three
# times been misled by a status that came from a trailing pipe stage. And not a
# grep of a log, which the production image family does not produce anyway.
#
# The two refusals must classify DIFFERENTLY, and this script checks that:
#   test A -> kms_decrypt_refused   (attestation fine, digest not pinned)
#   test B -> sts_exchange_refused  (attestation rejected before any key)
# If both came back the same, one test would be redundant and the report would be
# claiming two properties on the evidence of one.
#
# Nothing here reports a pass on an absent refusal: no evidence means
# INCONCLUSIVE.
#
# UNEXECUTED: not run; this session had no Google Cloud credentials. The refusal
# LOGIC for both tests is covered offline by enclave/test_negative.py.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=infra/enclave_config.sh
source "${HERE}/enclave_config.sh"
# shellcheck source=infra/enclave_common.sh
source "${HERE}/enclave_common.sh"

EVIDENCE_DIR="${REPO_ROOT}/.enclave-evidence"
mkdir -p "${EVIDENCE_DIR}"
# evidence_field() below caches each object at ${EVIDENCE_DIR}/<attempt>.json,
# keyed only by attempt name, not by this run's RUN_ID. Observed live: a
# failed run's cached tampered.json survived to the next run and was reused
# instead of refetched, so the closing verdict printed a stale refusal
# ("refused_unclassified") while the evidence this run actually wrote
# ("kms_decrypt_refused") sat unread in GCS. Clear the cache every run.
rm -f "${EVIDENCE_DIR}"/*.json
TARGET="${REPO_ROOT}/enclave/workload.py"
BACKUP="${EVIDENCE_DIR}/workload.py.orig"
RUN_ID="negtest-$(date -u +%Y%m%d-%H%M%S)"

restore_source() {
  if [[ -f "${BACKUP}" ]]; then
    cp "${BACKUP}" "${TARGET}"
    rm -f "${BACKUP}"
    info "restored enclave/workload.py"
  fi
}
trap restore_source EXIT INT TERM

# Pulls one field out of an evidence object. python3, not grep: a substring match
# on JSON would find "key_release_refused" inside a refusal MESSAGE and misread a
# successful run as refused.
evidence_field() {
  local attempt="$1" field="$2" local_copy
  local_copy="${EVIDENCE_DIR}/${attempt}.json"
  if [[ ! -s "${local_copy}" ]]; then
    gcloud storage cat "gs://${ENCLAVE_EVIDENCE_BUCKET}/runs/${RUN_ID}/${attempt}.json" \
      --project="${ENCLAVE_PROJECT}" > "${local_copy}" 2>/dev/null || true
  fi
  [[ -s "${local_copy}" ]] || return 0
  python3 -c "
import json,sys
with open(sys.argv[1]) as fh:
    doc = json.load(fh)
value = doc.get(sys.argv[2])
print('' if value is None else value)
" "${local_copy}" "${field}" 2>/dev/null || true
}

# ===========================================================================
step "0. POSITIVE — build v1, pin it, and confirm the key IS released"
# ===========================================================================
# Without this, neither refusal proves anything: both could just mean the
# pipeline is broken.
"${HERE}/enclave_build.sh" v1
V1_DIGEST="$(cat "${REPO_ROOT}/.enclave-digest-v1")"
[[ "${V1_DIGEST}" == sha256:* ]] || fatal "v1 digest looks wrong: ${V1_DIGEST}"
"${HERE}/enclave_pin_digest.sh" "${V1_DIGEST}"

set +e
ENCLAVE_IMAGE_FAMILY="${ENCLAVE_IMAGE_FAMILY}" \
  "${HERE}/enclave_run.sh" v1 "${RUN_ID}" production 2>&1 \
  | tee "${EVIDENCE_DIR}/positive-run.log"
set -e

POSITIVE_OUTCOME="$(evidence_field production outcome)"
POSITIVE_KEY_OK="$(evidence_field production key_release_ok)"
POSITIVE_DEBUG="$(evidence_field production running_debug_image)"

if [[ -z "${POSITIVE_OUTCOME}" ]]; then
  fatal "no evidence object for the positive run. INCONCLUSIVE — stop here; a refusal below would prove nothing."
fi
case "${POSITIVE_OUTCOME}" in
  completed|no_bundles_found)
    ok "the pinned production image was released the key (outcome=${POSITIVE_OUTCOME}, key_release_ok=${POSITIVE_KEY_OK})"
    ;;
  *)
    err "the PINNED PRODUCTION image did not get the key (outcome=${POSITIVE_OUTCOME})."
    err "refusal: $(evidence_field production refusal_message)"
    fatal "fix the condition or the binding first. Until the blessed image can get the key, a refusal proves nothing."
    ;;
esac
if [[ "${POSITIVE_DEBUG}" == "True" ]]; then
  fatal "the positive run reports running_debug_image=True. It was supposed to run the PRODUCTION family (${ENCLAVE_IMAGE_FAMILY}); the key must never be released to a debug VM."
fi

# ===========================================================================
step "A. TAMPERED IMAGE — the key follows the code"
# ===========================================================================
step "A.1 Change exactly one line of the scorer"
cp "${TARGET}" "${BACKUP}"
# The one changed line now lives in enclave/tamper.patch, and enclave_build.sh
# applies it for the tampered build only. The change is therefore in git, not
# inline here -- an inline edit and the patch would both apply and collide.
diff -u "${BACKUP}" "${TARGET}" | tee "${EVIDENCE_DIR}/tamper.diff" || true

step "A.2 Rebuild and confirm the digest MOVED"
"${HERE}/enclave_build.sh" tampered
TAMPERED_DIGEST="$(cat "${REPO_ROOT}/.enclave-digest-tampered")"
echo "    v1       : ${V1_DIGEST}"
echo "    tampered : ${TAMPERED_DIGEST}"
if [[ "${V1_DIGEST}" == "${TAMPERED_DIGEST}" ]]; then
  fatal "the digests are IDENTICAL, so this test would be vacuous. The build is probably caching; rebuild with --no-cache."
fi
ok "the digest changed, so the tampered image is a different IAM principal"

step "A.3 Confirm the KMS policy still names ONLY v1"
# Rules out the alternative explanation for a refusal: that the grant was never
# made. Read from the live policy, never from a local record of what was granted.
POLICY_MEMBERS="$(gcloud kms keys get-iam-policy "${ENCLAVE_KMS_EDGE_KEY_RESOURCE}" \
  --project="${ENCLAVE_PROJECT}" \
  --flatten="bindings[].members[]" \
  --filter="bindings.role:roles/cloudkms.cryptoKeyDecrypter" \
  --format="value(bindings.members)" 2>/dev/null || true)"
echo "${POLICY_MEMBERS}" | sed 's/^/        /' | tee "${EVIDENCE_DIR}/kms-decrypters.txt"
[[ "${POLICY_MEMBERS}" == *"${V1_DIGEST}"* ]] || \
  fatal "v1's digest is NOT in the decrypter policy, so the positive run succeeded for some other reason. Investigate before trusting any of this."
[[ "${POLICY_MEMBERS}" != *"${TAMPERED_DIGEST}"* ]] || \
  fatal "the tampered digest IS in the policy. Revoke it: ./infra/enclave_pin_digest.sh --revoke ${TAMPERED_DIGEST}"
ok "policy grants decrypt to v1 and not to the tampered image"

step "A.4 Run the tampered image"
set +e
"${HERE}/enclave_run.sh" tampered "${RUN_ID}" tampered 2>&1 \
  | tee "${EVIDENCE_DIR}/tampered-run.log"
set -e
TAMPERED_OUTCOME="$(evidence_field tampered outcome)"
TAMPERED_KIND="$(evidence_field tampered refusal_kind)"

# ===========================================================================
step "B. DEBUG IMAGE — the key requires no operator access"
# ===========================================================================
restore_source   # test B uses the UNCHANGED, still-pinned image
step "B.1 Confirm we are re-running the SAME, still-pinned image"
echo "    digest under test : ${V1_DIGEST}  (unchanged, and pinned in KMS)"
echo "    image family      : ${ENCLAVE_IMAGE_DEBUG_FAMILY}  (operator can SSH in)"
echo "    the ONLY difference from the positive run is the image family."

step "B.2 Boot the unchanged image on the DEBUG family"
set +e
ENCLAVE_IMAGE_FAMILY="${ENCLAVE_IMAGE_DEBUG_FAMILY}" \
  "${HERE}/enclave_run.sh" v1 "${RUN_ID}" debug 2>&1 \
  | tee "${EVIDENCE_DIR}/debug-run.log"
set -e
DEBUG_OUTCOME="$(evidence_field debug outcome)"
DEBUG_KIND="$(evidence_field debug refusal_kind)"
DEBUG_DBGSTAT="$(evidence_field debug attestation)"

# ===========================================================================
step "VERDICT"
# ===========================================================================
VERDICT=0

report_case() {
  local label="$1" attempt="$2" outcome="$3" kind="$4" expected_kind="$5" claim="$6"
  echo
  echo "  ${label}"
  echo "    outcome      : ${outcome:-<no evidence>}"
  echo "    refusal_kind : ${kind:-<none>}"
  if [[ -z "${outcome}" ]]; then
    err "  INCONCLUSIVE: no evidence object was written. This is NOT a pass and NOT"
    err "  a refusal — the VM may have failed to boot. Re-run before reporting."
    VERDICT=1
    return
  fi
  if [[ "${outcome}" != "key_release_refused" ]]; then
    err "  FAILED: the key was NOT refused (outcome=${outcome})."
    err "  ${claim} is therefore NOT demonstrated."
    VERDICT=1
    return
  fi
  echo "    refusal, verbatim:"
  evidence_field "${attempt}" refusal_message | sed 's/^/        /'
  if [[ "${kind}" == "${expected_kind}" ]]; then
    ok "  PASSED — ${claim}"
  else
    warn "  refused, but classified '${kind}' where '${expected_kind}' was expected."
    warn "  The refusal is real; describe what actually happened rather than the"
    warn "  claim this test was designed to make."
  fi
}

report_case "TEST A — tampered image (the key follows the code)" \
  tampered "${TAMPERED_OUTCOME}" "${TAMPERED_KIND}" "kms_decrypt_refused" \
  "a one-line code change costs the image its key"

report_case "TEST B — unchanged image on the DEBUG family (no operator access)" \
  debug "${DEBUG_OUTCOME}" "${DEBUG_KIND}" "sts_exchange_refused" \
  "a VM the operator can SSH into is refused the key even running blessed code"

echo
step "Are the two tests actually independent?"
echo "    test A refusal_kind : ${TAMPERED_KIND:-<none>}"
echo "    test B refusal_kind : ${DEBUG_KIND:-<none>}"
if [[ -n "${TAMPERED_KIND}" && -n "${DEBUG_KIND}" && "${TAMPERED_KIND}" != "${DEBUG_KIND}" ]]; then
  ok "the two refusals came from DIFFERENT gates, so they prove two properties"
else
  warn "the two refusals classified the same (or one is missing). Then only ONE"
  warn "property is demonstrated — do not report two."
fi
echo
echo "    Attestation claims the debug VM presented (this is also how the"
echo "    support_attributes shape question gets settled):"
echo "        ${DEBUG_DBGSTAT:-<no evidence>}"

step "Evidence written"
cat <<FILES
    ${EVIDENCE_DIR}/
      positive-run.log    the pinned production image DOES get the key
      production.json     its evidence object
      tamper.diff         the one-line change for test A
      kms-decrypters.txt  the live KMS policy at the time of the tests
      tampered.json       test A's evidence, incl. the verbatim refusal
      debug.json          test B's evidence, incl. the verbatim refusal
    Source of truth for both verdicts:
      gs://${ENCLAVE_EVIDENCE_BUCKET}/runs/${RUN_ID}/
FILES

step "Cleanup"
# The tampered TAG goes so a later run of THIS SCRIPT cannot start it by
# accident. Its digest was never pinned, so there is nothing to revoke.
#
# The underlying IMAGE (addressed by digest, not by this tag) is NOT
# deleted, and as of the judge-facing live-proof "tamper test" button
# (enclave/live_proof.py's TAMPERED_IMAGE_DIGEST), that is deliberate: a
# past run of this script already produced
# sha256:8515694b88b23b78cefd85cd1b591e541f67dc99bbf3e7533e2175d8658c6482,
# and that exact digest is now kept forever as the demo's known-refused
# image, reused by reference rather than rebuilt per click. Re-running
# this script builds and discards a DIFFERENT tampered image under the
# same mutable tag -- it does not change, and has no effect on, the one
# digest the live-proof feature hardcodes. Never re-tag or rebuild that
# specific digest; enclave/test_live_proof.py's live KMS-policy check
# is what would catch it if this were ever violated.
gcloud artifacts docker tags delete "${ENCLAVE_IMAGE_BASE}:tampered" \
  --project="${ENCLAVE_PROJECT}" --quiet >/dev/null 2>&1 || \
  warn "could not delete the :tampered tag; remove it by hand"
rm -f "${REPO_ROOT}/.enclave-digest-tampered"
restore_source
ok "source restored and tampered tag removed"

exit "${VERDICT}"
