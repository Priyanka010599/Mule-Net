# MuleNet Confidential Space workload image.
#
# Build from the REPO ROOT, not from enclave/:
#     docker build -f enclave/Dockerfile -t <image> .
# because the workload imports shared/ contracts and ml/features.py, and a
# build context rooted at enclave/ cannot see them.
#
# ---------------------------------------------------------------------------
# The digest of this image is the security boundary
# ---------------------------------------------------------------------------
# Cloud KMS releases the decryption key to one image digest
# (infra/enclave_pin_digest.sh). So every layer is chosen to be reproducible:
# a pinned base image by digest, pinned pip versions, no `apt-get upgrade`, no
# network fetch at runtime. Two builds of the same source should produce the
# same digest; when they do not, the negative test cannot distinguish "someone
# changed the code" from "the build moved", which is the whole point of it.
#
# The base is pinned by DIGEST rather than by tag on purpose: `python:3.11-slim`
# is remapped by upstream regularly, and a remap would silently revoke the
# enclave's key.
#
# The digest below is the real multi-arch index digest for python:3.11-slim,
# resolved from registry-1.docker.io on 2026-09-27. Re-resolve it deliberately
# with:
#     docker buildx imagetools inspect python:3.11-slim
# and re-pin the KMS binding afterwards, because changing the base changes this
# image's digest and therefore revokes its key release.
ARG BASE_IMAGE=python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e
FROM ${BASE_IMAGE}

# Unbuffered so the container launcher's log redirect captures output as it
# happens. Without this, a workload that dies mid-run can lose the very lines
# that explain why -- including the KMS refusal the negative test needs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, so a source-only change does not re-resolve pip.
COPY enclave/requirements.txt /app/enclave/requirements.txt
RUN pip install --no-cache-dir \
      -r /app/enclave/requirements.txt

# Only what the workload imports. Notably absent: simulator/, ingest/, agents/,
# ui/, docai/ and the rest of ml/. A smaller image is a smaller attestation
# surface, and code that cannot be reached cannot be a way out of the enclave.
COPY shared/       /app/shared/
COPY ml/__init__.py ml/config.py ml/features.py /app/ml/
COPY enclave/      /app/enclave/

# Drop privileges. Confidential Space isolates the VM from the operator, not the
# workload from itself, so a bug in the scorer still runs as whoever this is.
RUN useradd --system --uid 10001 --no-create-home enclave \
    && chown -R enclave:enclave /app
USER 10001

# The Confidential Space launcher rejects any tee-env-* or tee-cmd the image does
# not allow (observed live: "env var ... is not allowed to be overridden on this
# image", and the workload never started). These labels are part of the digest
# KMS is pinned to, so the operator can set exactly these and nothing else.
LABEL "tee.launch_policy.allow_env_override"="MULENET_ENCLAVE_ATTEMPT,MULENET_ENCLAVE_BANK_IDS,MULENET_ENCLAVE_EVIDENCE_BUCKET,MULENET_ENCLAVE_FLAGGED_BANK_A,MULENET_ENCLAVE_FLAGGED_BANK_B,MULENET_ENCLAVE_FLAGGED_BANK_C,MULENET_ENCLAVE_HASH_KEY_URI,MULENET_ENCLAVE_HASH_KMS_KEY,MULENET_ENCLAVE_INBOX_BUCKET,MULENET_ENCLAVE_KMS_KEY,MULENET_ENCLAVE_OUTBOX_BUCKET,MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A,MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_B,MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_C,MULENET_ENCLAVE_PROJECT,MULENET_ENCLAVE_RUN_ID,MULENET_ENCLAVE_WIP_AUDIENCE,MULENET_ENCLAVE_WORKLOAD_SA"
LABEL "tee.launch_policy.allow_cmd_override"="true"

# The container launcher runs the entrypoint with no arguments unless the VM's
# tee-cmd metadata overrides them. `--self-check` is available as an override,
# which is how enclave_run.sh separates "the VM booted and the code works" from
# "KMS refused the key".
ENTRYPOINT ["python", "-m", "enclave.workload"]
