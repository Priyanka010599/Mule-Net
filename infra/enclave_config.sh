#!/usr/bin/env bash
#
# MuleNet enclave — every name and region in one place.
#
# Source this, do not execute it. Every value is overridable from the
# environment so the whole stack can be rebuilt under different project ids on
# a fresh Google Cloud account (see docs/enclave.md, "Rebuilding on a new
# account"). Nothing here is a secret: secrets live in Cloud KMS and Secret
# Manager, and the enclave receives only ciphertext.

# --- projects ---------------------------------------------------------------
#
# The operator project owns the inbox, the KMS key, the workload identity pool
# and the Confidential VM. The bank projects are IDENTITY-ONLY: a service
# account and nothing billable. mulenet-bank-c is deliberately UNBILLED --
# the trial billing account is capped at five linked projects and is full --
# which is exactly why enclave_bank_identity.sh verifies that an unbilled
# project's service account can still authenticate and write.
export ENCLAVE_PROJECT="${ENCLAVE_PROJECT:-mulenet}"
export BANK_A_PROJECT="${BANK_A_PROJECT:-mulenet-bank-a}"
export BANK_B_PROJECT="${BANK_B_PROJECT:-mulenet-bank-b}"
export BANK_C_PROJECT="${BANK_C_PROJECT:-mulenet-bank-c}"

#: gcloud routes billing and quota calls through its own client project unless
#: told otherwise, then fails with a misleading "API has not been used in
#: project 681255809395". Every billing call passes this.
export BILLING_QUOTA_PROJECT="${BILLING_QUOTA_PROJECT:-${ENCLAVE_PROJECT}}"

#: Logical bank ids as they appear in the data and in the enclave, distinct
#: from the GCP project ids above.
export BANK_IDS="${BANK_IDS:-bank-a,bank-b,bank-c}"

# --- region ----------------------------------------------------------------
#
# asia-south1 (Mumbai) is the preference: Gemini and Document AI are there, and
# for an India BFSI entry Mumbai is the defensible answer. But Confidential
# Space availability is a property of the machine type in a specific zone, and
# it is NOT assumed here -- infra/enclave_preflight.sh checks it live and
# refuses to continue if the zone cannot run the machine type. If asia-south1
# cannot, set ENCLAVE_ZONE/ENCLAVE_REGION to a zone that can and record the
# reason in docs/enclave.md; the scripts do not silently relocate.
export ENCLAVE_REGION="${ENCLAVE_REGION:-asia-south1}"
export ENCLAVE_ZONE="${ENCLAVE_ZONE:-asia-south1-a}"

#: KMS location. Kept equal to the compute region so the key does not sit in a
#: different jurisdiction from the workload that uses it.
export ENCLAVE_KMS_LOCATION="${ENCLAVE_KMS_LOCATION:-${ENCLAVE_REGION}}"

#: Workload identity pools for Confidential Space are global in every Google
#: sample; a regional pool is not used.
export ENCLAVE_WIP_LOCATION="global"

# --- machine ---------------------------------------------------------------
#
# c2d-standard-2 with AMD SEV. Google's MIG codelab uses n2d, but this project's
# N2D_CPUS quota in asia-south1 is 0 (observed live); C2D also supports SEV and
# has quota. Two vCPUs is plenty; the workload is IO- and memory-light.
export ENCLAVE_MACHINE_TYPE="${ENCLAVE_MACHINE_TYPE:-c2d-standard-2}"
export ENCLAVE_CONFIDENTIAL_TYPE="${ENCLAVE_CONFIDENTIAL_TYPE:-SEV}"

#: PRODUCTION family, and this default is load-bearing rather than tidy.
#:
#: confidential-space-debug allows the OPERATOR TO SSH INTO THE VM and read its
#: memory. A VM running that family holding three banks' joined transaction graph
#: would hand the operator exactly the plaintext this whole design exists to
#: withhold. So the debug family is never the trust model, and the attestation
#: condition in enclave_setup.sh refuses to release the key to it.
#:
#: The debug family was previously the default because serial-console log
#: redirect was how the demo showed the KMS refusal. That is no longer how
#: evidence is collected: the workload writes a JSON result object to
#: ENCLAVE_EVIDENCE_BUCKET on every run, including refused ones. See
#: docs/enclave.md, "Evidence".
#:
#: enclave_negative_test.sh still LAUNCHES a debug VM on purpose, as its second
#: negative test, to demonstrate that the key is refused to it.
export ENCLAVE_IMAGE_FAMILY="${ENCLAVE_IMAGE_FAMILY:-confidential-space}"
export ENCLAVE_IMAGE_DEBUG_FAMILY="${ENCLAVE_IMAGE_DEBUG_FAMILY:-confidential-space-debug}"
export ENCLAVE_IMAGE_PROJECT="confidential-space-images"

# --- resource names --------------------------------------------------------
export ENCLAVE_KEYRING="${ENCLAVE_KEYRING:-mulenet-enclave}"
#: Wraps the banks' per-bundle data-encryption keys.
export ENCLAVE_EDGE_KEY="${ENCLAVE_EDGE_KEY:-edge-wrap}"
#: Wraps the in-enclave HMAC key used for Stage 4 re-keying.
export ENCLAVE_HASH_KEY="${ENCLAVE_HASH_KEY:-hash-rekey}"

export ENCLAVE_WIP="${ENCLAVE_WIP:-mulenet-enclave-wip}"
export ENCLAVE_WIP_PROVIDER="${ENCLAVE_WIP_PROVIDER:-mulenet-attestation}"

export ENCLAVE_WORKLOAD_SA="${ENCLAVE_WORKLOAD_SA:-mulenet-enclave-workload}"
export ENCLAVE_ARTIFACT_REPO="${ENCLAVE_ARTIFACT_REPO:-mulenet-enclave}"
export ENCLAVE_IMAGE_NAME="${ENCLAVE_IMAGE_NAME:-scorer}"

export ENCLAVE_INBOX_BUCKET="${ENCLAVE_INBOX_BUCKET:-${ENCLAVE_PROJECT}-enclave-inbox}"
export ENCLAVE_OUTBOX_BUCKET="${ENCLAVE_OUTBOX_BUCKET:-${ENCLAVE_PROJECT}-enclave-outbox}"
export ENCLAVE_KEYS_BUCKET="${ENCLAVE_KEYS_BUCKET:-${ENCLAVE_PROJECT}-enclave-keys}"
#: Where the workload writes one JSON result object per run, INCLUDING runs where
#: KMS refuses the key. Written with the VM's attached service account, which
#: does not depend on the attestation exchange — that separation is what makes a
#: refusal recordable instead of merely observable. Replaces reading the serial
#: console, which only worked on the debug image family.
export ENCLAVE_EVIDENCE_BUCKET="${ENCLAVE_EVIDENCE_BUCKET:-${ENCLAVE_PROJECT}-enclave-evidence}"

export ENCLAVE_VM_NAME="${ENCLAVE_VM_NAME:-mulenet-enclave-cvm}"

#: Cloud Build fails on its own source upload without an explicit service
#: account in this project.
export ENCLAVE_BUILD_SA="${ENCLAVE_BUILD_SA:-projects/${ENCLAVE_PROJECT}/serviceAccounts/mulenet-build@${ENCLAVE_PROJECT}.iam.gserviceaccount.com}"

# --- attestation condition -------------------------------------------------
#
# The key is released only to a NON-DEBUG, stable Confidential Space image.
#
# dbgstat == 'disabled-since-boot' is the primary gate and the better verified
# one: the documented values are 'disabled-since-boot' for the production image
# and 'enable' for the debug image.
#
# The STABLE support-attribute check is defence in depth. Its CEL shape could NOT
# be verified offline — the documentation describes support_attributes both as a
# list ("'STABLE' in ...") and as a concatenated string ("LATEST=STABLE=USABLE"),
# and one example instead shows `support_attributes.stable == true`. So it is
# isolated in this one variable: if the provider rejects the expression, try the
# alternatives listed below rather than editing the whole condition. The
# workload writes the claim's raw value AND its type into every evidence object,
# so the first real run settles which form is correct.
#
# Alternatives, in the order to try them:
#   'STABLE' in assertion.submods.confidential_space.support_attributes
#   assertion.submods.confidential_space.support_attributes.stable == true
#   assertion.submods.confidential_space.support_attributes.contains('STABLE')
export ENCLAVE_SUPPORT_ATTR_CHECK="${ENCLAVE_SUPPORT_ATTR_CHECK:-'STABLE' in assertion.submods.confidential_space.support_attributes}"

#: Required substring that enclave_setup.sh looks for when it reads the
#: condition back from the API. dbgstat is the check whose absence would mean a
#: debug VM could get the key, so its presence is asserted, not assumed.
export ENCLAVE_REQUIRED_CONDITION_MARKER="disabled-since-boot"

# --- derived ---------------------------------------------------------------
export ENCLAVE_KMS_EDGE_KEY_RESOURCE="projects/${ENCLAVE_PROJECT}/locations/${ENCLAVE_KMS_LOCATION}/keyRings/${ENCLAVE_KEYRING}/cryptoKeys/${ENCLAVE_EDGE_KEY}"
export ENCLAVE_KMS_HASH_KEY_RESOURCE="projects/${ENCLAVE_PROJECT}/locations/${ENCLAVE_KMS_LOCATION}/keyRings/${ENCLAVE_KEYRING}/cryptoKeys/${ENCLAVE_HASH_KEY}"
export ENCLAVE_IMAGE_BASE="${ENCLAVE_REGION}-docker.pkg.dev/${ENCLAVE_PROJECT}/${ENCLAVE_ARTIFACT_REPO}/${ENCLAVE_IMAGE_NAME}"
export ENCLAVE_WORKLOAD_SA_EMAIL="${ENCLAVE_WORKLOAD_SA}@${ENCLAVE_PROJECT}.iam.gserviceaccount.com"

#: APIs the operator project needs. confidentialcomputing is what mints the
#: attestation token; iamcredentials is what the federated credential
#: impersonates the workload service account through.
export ENCLAVE_APIS=(
  cloudkms.googleapis.com
  compute.googleapis.com
  confidentialcomputing.googleapis.com
  iamcredentials.googleapis.com
  sts.googleapis.com
  storage.googleapis.com
  artifactregistry.googleapis.com
  cloudbuild.googleapis.com
  logging.googleapis.com
)
