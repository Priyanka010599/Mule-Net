#!/usr/bin/env bash
#
# MuleNet enclave — shared helpers for the Confidential Space scripts.
#
# Source this, do not execute it.
#
# ---------------------------------------------------------------------------
# Two rules are enforced here rather than left to each script's discipline.
# ---------------------------------------------------------------------------
#
# 1. VERIFY AGAINST STATE, NEVER AN EXIT CODE.
#
#    This project has three times been misled by a chained command whose exit
#    status came from a trailing `grep` or `tail` rather than from the gcloud
#    call. Google's own Confidential Space codelab scripts have the same bug --
#    they run `gcloud kms keys describe ... | grep NAME` and branch on `$?`,
#    which reports whether grep matched, not whether the key exists.
#
#    So every check here reads a FIELD VALUE out of the API with
#    --format="value(...)" and compares it to what was asked for. A resource is
#    "there" only when the API returns its name. Nothing branches on $?.
#
# 2. EVERY gcloud CALL CARRIES --project EXPLICITLY.
#
#    `gcloud config set project` is never used: a leaked default can point a
#    destructive command at an unrelated project. Helpers take the project as
#    an argument, and `require_project_arg` refuses an empty one rather than
#    letting gcloud fall back to a default.
#
# Billing and quota calls additionally need --billing-project, or gcloud routes
# them through its own client project and fails with a misleading
# "API has not been used in project 681255809395".

set -euo pipefail

# --- output -----------------------------------------------------------------

_c_reset=$'\033[0m'; _c_red=$'\033[31m'; _c_green=$'\033[32m'; _c_yellow=$'\033[33m'

info()  { echo "==> $*"; }
step()  { echo; echo "=== $* ==="; }
ok()    { echo "    ${_c_green}OK${_c_reset}   $*"; }
warn()  { echo "    ${_c_yellow}WARN${_c_reset} $*" >&2; }
err()   { echo "    ${_c_red}FAIL${_c_reset} $*" >&2; }

#: Every IAM binding a script creates is appended here and printed at the end,
#: because "report every IAM binding created" is a requirement and a grep of
#: the script text is not the same as a record of what actually ran.
IAM_LOG_FILE="${IAM_LOG_FILE:-${TMPDIR:-/tmp}/mulenet-enclave-iam.log}"

record_binding() {
  # usage: record_binding <scope> <member> <role>
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" \
    >> "${IAM_LOG_FILE}"
}

print_bindings() {
  step "IAM bindings recorded this run"
  if [[ -s "${IAM_LOG_FILE}" ]]; then
    printf '%-21s %-58s %-64s %s\n' "WHEN" "SCOPE" "MEMBER" "ROLE"
    while IFS=$'\t' read -r when scope member role; do
      printf '%-21s %-58s %-64s %s\n' "${when}" "${scope}" "${member}" "${role}"
    done < "${IAM_LOG_FILE}"
  else
    echo "    (none)"
  fi
}

fatal() { err "$*"; exit 1; }

require_project_arg() {
  # A helper that silently accepted an empty project would let gcloud fall back
  # to a default, which is exactly the failure mode --project exists to avoid.
  [[ -n "${1:-}" ]] || fatal "internal: a project id was required but empty"
}

# --- state verification -----------------------------------------------------

api_value() {
  # usage: api_value <format-expr> <gcloud args...>
  #
  # Prints the requested field, or nothing. Deliberately swallows the error:
  # "absent" and "errored" are both "no value", and the caller decides what an
  # empty answer means. Nothing downstream may read $? from this.
  local fmt="$1"; shift
  gcloud "$@" --format="value(${fmt})" 2>/dev/null || true
}

resource_exists() {
  # usage: resource_exists <expected-substring> <format-expr> <gcloud args...>
  #
  # True only when the API returned a value containing the expected string.
  local expected="$1" fmt="$2"; shift 2
  local actual
  actual="$(api_value "${fmt}" "$@")"
  [[ -n "${actual}" && "${actual}" == *"${expected}"* ]]
}

require_state() {
  # usage: require_state <description> <expected-substring> <format-expr> <gcloud args...>
  #
  # The assertion used after a create: re-read the resource from the API and
  # confirm it is really there. A create that "succeeded" but left nothing
  # behind fails here.
  local what="$1" expected="$2" fmt="$3"; shift 3
  local actual pause
  # IAM and several other Google Cloud APIs are eventually consistent: a
  # resource just created can be invisible to a describe for a few seconds.
  # Observed live: `gcloud iam service-accounts create` printed "Created
  # service account" and the describe straight after returned nothing. So
  # re-read with backoff before concluding the create failed. This still
  # FAILS if the resource never appears; it only stops a propagation delay
  # being misreported as a failed create.
  for pause in 0 2 4 8 15 30; do
    (( pause > 0 )) && sleep "${pause}"
    actual="$(api_value "${fmt}" "$@")"
    if [[ -n "${actual}" && "${actual}" == *"${expected}"* ]]; then
      ok "${what}: ${actual}"
      return 0
    fi
  done
  err "${what}: expected a value containing '${expected}', API returned '${actual:-<empty>}' after ~59s of retries"
  return 1
}

# --- idempotent creates -----------------------------------------------------
#
# Each of these checks state, creates only if absent, then re-verifies state.
# Re-running any script is safe.

ensure_api_enabled() {
  # usage: ensure_api_enabled <project> <api>...
  local project="$1"; shift
  require_project_arg "${project}"
  local api
  for api in "$@"; do
    if resource_exists "${api}" "config.name" \
         services list --enabled --filter="config.name=${api}" --project="${project}"; then
      ok "API already enabled: ${api}"
    else
      info "enabling ${api} on ${project}"
      gcloud services enable "${api}" --project="${project}"
      require_state "API enabled: ${api}" "${api}" "config.name" \
        services list --enabled --filter="config.name=${api}" --project="${project}" \
        || fatal "could not enable ${api} on ${project}"
    fi
  done
}

ensure_service_account() {
  # usage: ensure_service_account <project> <account-id> <display-name>
  local project="$1" account_id="$2" display="$3"
  require_project_arg "${project}"
  local email="${account_id}@${project}.iam.gserviceaccount.com"
  if resource_exists "${email}" "email" \
       iam service-accounts describe "${email}" --project="${project}"; then
    ok "service account exists: ${email}"
  else
    info "creating service account ${email}"
    gcloud iam service-accounts create "${account_id}" \
      --display-name="${display}" --project="${project}"
    require_state "service account" "${email}" "email" \
      iam service-accounts describe "${email}" --project="${project}" \
      || fatal "service account ${email} was not created"
  fi
  echo "${email}"
}

ensure_kms_keyring() {
  # usage: ensure_kms_keyring <project> <location> <keyring>
  local project="$1" location="$2" keyring="$3"
  require_project_arg "${project}"
  if resource_exists "/keyRings/${keyring}" "name" \
       kms keyrings describe "${keyring}" --location="${location}" --project="${project}"; then
    ok "keyring exists: ${keyring} (${location})"
  else
    info "creating keyring ${keyring} in ${location}"
    gcloud kms keyrings create "${keyring}" --location="${location}" --project="${project}"
    require_state "keyring" "/keyRings/${keyring}" "name" \
      kms keyrings describe "${keyring}" --location="${location}" --project="${project}" \
      || fatal "keyring ${keyring} was not created"
  fi
}

ensure_kms_key() {
  # usage: ensure_kms_key <project> <location> <keyring> <key> [purpose]
  local project="$1" location="$2" keyring="$3" key="$4" purpose="${5:-encryption}"
  require_project_arg "${project}"
  if resource_exists "/cryptoKeys/${key}" "name" \
       kms keys describe "${key}" --keyring="${keyring}" --location="${location}" \
       --project="${project}"; then
    ok "key exists: ${key}"
  else
    info "creating key ${key} (purpose=${purpose})"
    gcloud kms keys create "${key}" --keyring="${keyring}" --location="${location}" \
      --purpose="${purpose}" --project="${project}"
    require_state "key" "/cryptoKeys/${key}" "name" \
      kms keys describe "${key}" --keyring="${keyring}" --location="${location}" \
      --project="${project}" \
      || fatal "key ${key} was not created"
  fi
  # KMS keys are never deleted by these scripts, only their versions destroyed,
  # because a destroyed keyring name cannot be reused and the rebuild-on-a-new-
  # account path depends on predictable names.
}

ensure_bucket() {
  # usage: ensure_bucket <project> <location> <bucket>
  #
  # Uniform bucket-level access is set on purpose: with ACLs enabled a bank
  # granted object-level ACLs could read another bank's objects, which is the
  # one thing the inbox must not allow.
  local project="$1" location="$2" bucket="$3"
  require_project_arg "${project}"
  if resource_exists "${bucket}" "name" \
       storage buckets describe "gs://${bucket}" --project="${project}"; then
    ok "bucket exists: gs://${bucket}"
  else
    info "creating bucket gs://${bucket} in ${location}"
    gcloud storage buckets create "gs://${bucket}" \
      --location="${location}" \
      --uniform-bucket-level-access \
      --public-access-prevention \
      --project="${project}"
    require_state "bucket" "${bucket}" "name" \
      storage buckets describe "gs://${bucket}" --project="${project}" \
      || fatal "bucket ${bucket} was not created"
  fi
}

ensure_artifact_repo() {
  # usage: ensure_artifact_repo <project> <location> <repo>
  local project="$1" location="$2" repo="$3"
  require_project_arg "${project}"
  if resource_exists "/repositories/${repo}" "name" \
       artifacts repositories describe "${repo}" --location="${location}" \
       --project="${project}"; then
    ok "artifact repo exists: ${repo}"
  else
    info "creating artifact repo ${repo} in ${location}"
    gcloud artifacts repositories create "${repo}" \
      --repository-format=docker --location="${location}" \
      --description="MuleNet Confidential Space workload images" \
      --project="${project}"
    require_state "artifact repo" "/repositories/${repo}" "name" \
      artifacts repositories describe "${repo}" --location="${location}" \
      --project="${project}" \
      || fatal "artifact repo ${repo} was not created"
  fi
}

ensure_workload_identity_pool() {
  # usage: ensure_workload_identity_pool <project> <location> <pool>
  local project="$1" location="$2" pool="$3"
  require_project_arg "${project}"
  if [[ ${#pool} -lt 4 || ${#pool} -gt 32 ]]; then
    fatal "workload identity pool name '${pool}' must be 4-32 characters"
  fi
  if resource_exists "/workloadIdentityPools/${pool}" "name" \
       iam workload-identity-pools describe "${pool}" --location="${location}" \
       --project="${project}"; then
    ok "workload identity pool exists: ${pool}"
  else
    info "creating workload identity pool ${pool}"
    gcloud iam workload-identity-pools create "${pool}" \
      --location="${location}" \
      --display-name="MuleNet enclave attestation" \
      --project="${project}"
    require_state "workload identity pool" "/workloadIdentityPools/${pool}" "name" \
      iam workload-identity-pools describe "${pool}" --location="${location}" \
      --project="${project}" \
      || fatal "workload identity pool ${pool} was not created"
  fi
}

# --- project metadata -------------------------------------------------------

project_number() {
  # usage: project_number <project>
  local project="$1"
  require_project_arg "${project}"
  local number
  number="$(api_value projectNumber projects describe "${project}")"
  [[ -n "${number}" ]] || fatal "could not read the project number for ${project}"
  echo "${number}"
}

project_billing_enabled() {
  # usage: project_billing_enabled <project>
  #
  # --billing-project is required or gcloud routes this through its own client
  # project and returns a misleading "API has not been used" error.
  local project="$1"
  require_project_arg "${project}"
  api_value billingEnabled beta billing projects describe "${project}" \
    --billing-project="${BILLING_QUOTA_PROJECT:-mulenet}"
}

# --- quota ------------------------------------------------------------------

report_cpu_quota() {
  # usage: report_cpu_quota <project> <region>
  #
  # Checked BEFORE anything is created: trial accounts have low per-region vCPU
  # caps, and a Confidential VM that fails to start after the keys and pools
  # are in place wastes a stage. Printed rather than enforced, because the
  # limit that matters depends on the machine type the caller picked.
  local project="$1" region="$2"
  require_project_arg "${project}"
  step "vCPU quota in ${region} (project ${project})"
  local out
  out="$(gcloud compute regions describe "${region}" --project="${project}" \
          --format="table[no-heading](quotas.metric,quotas.limit,quotas.usage)" 2>/dev/null \
          | grep -Ei 'CPUS|N2D' || true)"
  if [[ -n "${out}" ]]; then
    echo "${out}"
  else
    warn "could not read region quota for ${region}; Compute Engine may not be enabled"
  fi
}

# --- Confidential Space availability ---------------------------------------

confidential_space_image() {
  # usage: confidential_space_image <family>
  #
  # Reads the latest image from Google's public confidential-space-images
  # project. Prints the image name, or nothing. The family is checked live
  # rather than assumed, because the image families are versioned and retired.
  local family="$1"
  api_value name compute images describe-from-family "${family}" \
    --project=confidential-space-images
}

zone_supports_machine_type() {
  # usage: zone_supports_machine_type <project> <zone> <machine-type>
  #
  # A live check. Confidential VM support is a property of the machine type in
  # a specific zone, and the docs lag reality in both directions, so this asks
  # the API instead of trusting a table.
  local project="$1" zone="$2" machine_type="$3"
  require_project_arg "${project}"
  resource_exists "${machine_type}" "name" \
    compute machine-types describe "${machine_type}" --zone="${zone}" --project="${project}"
}
