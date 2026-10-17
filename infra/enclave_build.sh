#!/usr/bin/env bash
#
# MuleNet enclave — build the workload image and print its digest.
#
# Usage:
#   ./infra/enclave_build.sh [tag]
#
# The tag defaults to "v1". The negative test builds "tampered" with one line
# changed; everything else about the build is identical.
#
# The DIGEST is the output that matters, not the tag. A tag is mutable and
# proves nothing: `:v1` can be repointed at any image. The KMS binding names a
# digest, so this script resolves the digest from Artifact Registry AFTER the
# push and prints it for infra/enclave_pin_digest.sh.
#
# Cloud Build is invoked with an explicit --service-account because without one
# it fails on its own source upload in this project.
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
IMAGE="${ENCLAVE_IMAGE_BASE}:${TAG}"

step "Building ${IMAGE}"
info "build context: ${REPO_ROOT} (needs shared/ and ml/, so the context is the repo root)"

# A throwaway cloudbuild config rather than `gcloud builds submit --tag`, because
# the Dockerfile lives at enclave/Dockerfile while the context is the repo root,
# and --tag assumes ./Dockerfile.
BUILD_CONFIG="$(mktemp)"
#: The tampered build is the approved source plus enclave/tamper.patch -- one
#: committed line, so the exact difference is in git. Applied only for this
#: build, and always reverted, so the working tree is left as it was found.
PATCH_APPLIED=0
revert_tamper() {
  if (( PATCH_APPLIED )); then
    git -C "${REPO_ROOT}" apply -R "${REPO_ROOT}/enclave/tamper.patch"
  fi
  rm -f "${BUILD_CONFIG}"
}
trap revert_tamper EXIT
if [[ "${TAG}" == "tampered" ]]; then
  git -C "${REPO_ROOT}" apply "${REPO_ROOT}/enclave/tamper.patch"
  PATCH_APPLIED=1
  info "applied enclave/tamper.patch to enclave/workload.py for this build"
fi
cat > "${BUILD_CONFIG}" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - --file=enclave/Dockerfile
      - --tag=${IMAGE}
      - .
images:
  - ${IMAGE}
options:
  logging: CLOUD_LOGGING_ONLY
YAML

gcloud builds submit "${REPO_ROOT}" \
  --config="${BUILD_CONFIG}" \
  --project="${ENCLAVE_PROJECT}" \
  --service-account="${ENCLAVE_BUILD_SA}" \
  --ignore-file="${REPO_ROOT}/infra/enclave_cloudbuild.ignore"

# Resolve the digest from the registry. Verified against registry STATE, not
# against the build's exit code: a build can report success while the push of
# the final image did not land, and every later stage would then pin a digest
# that is not what runs.
step "Resolving the pushed digest"
# Artifact Registry is eventually consistent: observed live, a push that had
# landed (listable seconds later) was not yet describable straight after the
# build. Retry with backoff before concluding the push failed -- this still
# fails if the image never appears.
DIGEST=""
for pause in 0 2 4 8 15 30; do
  (( pause > 0 )) && sleep "${pause}"
  # `images describe` returns the digest under image_summary.digest; `version`
  # is a column of `images list` only, so asking describe for it is always empty.
  DIGEST="$(api_value "image_summary.digest" \
    artifacts docker images describe "${IMAGE}" --project="${ENCLAVE_PROJECT}")"
  [[ -n "${DIGEST}" ]] && break
done

if [[ -z "${DIGEST}" ]]; then
  fatal "no digest found for ${IMAGE} in Artifact Registry. The build may have reported success without pushing; re-run and check the build log."
fi
if [[ "${DIGEST}" != sha256:* ]]; then
  fatal "expected a sha256: digest for ${IMAGE}, got '${DIGEST}'"
fi

ok "image  : ${IMAGE}"
ok "digest : ${DIGEST}"

cat <<NEXT

    Next, grant key release to exactly this image:
        ./infra/enclave_pin_digest.sh ${DIGEST}

    Record the digest — the negative test compares against it.
NEXT

# Written to a file as well so the other scripts do not depend on a human
# copying a 71-character string correctly.
echo "${DIGEST}" > "${REPO_ROOT}/.enclave-digest-${TAG}"
info "digest also written to .enclave-digest-${TAG} (gitignored)"
