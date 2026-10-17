"""Environment-driven configuration for the MuleNet Confidential Space workload.

Every knob the enclave needs at runtime is read from an environment variable
here (see CLAUDE.md: "Config via env vars in one config.py per module").
Nothing in this module talks to Google Cloud; it only resolves values so the
rest of ``enclave/`` stays free of ``os.environ`` lookups.

On a Confidential Space VM these variables arrive as ``tee-env-*`` instance
metadata keys, which the container launcher turns into plain environment
variables inside the workload container. That is the only supported way to
pass configuration in: the VM's own metadata is visible to the operator, so
**nothing secret may travel this way**. Secrets reach the enclave only as
ciphertext the enclave decrypts with a KMS key released to its attestation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Where the container launcher writes the attestation token the workload
#: exchanges for Google credentials. Confirmed against Google's own published
#: sample: ``codelabs/digital_asset_transaction_codelab/src/credential-config.js``
#: in github.com/GoogleCloudPlatform/confidential-space.
ATTESTATION_TOKEN_PATH = Path("/run/container_launcher/attestation_verifier_claims_token")

#: STS endpoint that trades the attestation token for a federated access token.
STS_TOKEN_URL = "https://sts.googleapis.com/v1/token"

#: Subject token type for the Confidential Space attestation JWT.
SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"

#: Workload identity pools for Confidential Space live in the ``global``
#: location in every Google sample; a regional pool is not used here.
WIP_LOCATION = "global"

#: Default AES-GCM nonce length in bytes. 96 bits is the only length AES-GCM
#: is specified for without extra derivation, so it is not configurable.
GCM_NONCE_BYTES = 12

#: Default data-encryption-key length in bytes (AES-256).
DEK_BYTES = 32


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"{name} is not set. On a Confidential Space VM pass it as instance "
            f"metadata key tee-env-{name}."
        )
    return value


@dataclass(frozen=True)
class EnclaveConfig:
    """Resolved settings for one enclave run.

    Attributes:
        operator_project: Project that owns the inbox bucket, the KMS key and
            the Confidential VM. The banks do **not** own this.
        inbox_bucket: Bucket the banks write their encrypted bundles to. The
            operator can list and read objects here but holds no key that
            decrypts them.
        outbox_bucket: Bucket the enclave writes per-bank alert bundles to,
            each one encrypted to that bank's own key.
        evidence_bucket: Bucket the enclave writes a result object to on
            **every** run, including runs where KMS refuses the key. Written
            with the VM's attached service account, which does not depend on
            attestation -- see :mod:`enclave.kms` for why that separation is
            what makes refusals recordable rather than merely observable.
        kms_key: Fully qualified KMS key resource name that wraps the banks'
            data-encryption keys. Released only to the attested image.
        hash_kms_key: Fully qualified KMS key resource name that wrapped the
            in-enclave HMAC key. A different key from ``kms_key``: KMS refuses
            to decrypt ciphertext under any key but the one that made it.
        hash_key_ciphertext_uri: ``gs://`` URI of the KMS-wrapped in-enclave
            HMAC key (Stage 4). Never a plaintext key, and never an env var
            holding one.
        wip_provider_audience: Full STS audience for the workload identity
            pool provider that validates this image's attestation.
        workload_service_account: The service account the VM runs as. It is
            **not** what the attestation is federated into -- the federated
            principal is used directly, so that the KMS key's
            ``attribute.image_digest`` binding is what authorises decrypt (see
            :mod:`enclave.kms`). This account is what writes the evidence object
            and reads the inbox, and it is recorded in every run's evidence so an
            auditor can see which identity wrote it.
        bank_ids: Banks expected to contribute this run, in a stable order.
        outbox_public_keys: Each bank's X25519 public key (32 raw bytes),
            generated offline by the bank itself. Not secret -- see
            :mod:`enclave.outbox_crypto` for why -- but required for every bank
            in ``bank_ids``: the enclave fails closed rather than write an
            unencrypted outbox bundle for a bank with no key on file.
        out_dir: Local scratch directory inside the container.
    """

    operator_project: str
    inbox_bucket: str
    outbox_bucket: str
    evidence_bucket: str
    kms_key: str
    hash_kms_key: str
    hash_key_ciphertext_uri: str
    wip_provider_audience: str
    workload_service_account: str
    bank_ids: tuple[str, ...]
    outbox_public_keys: dict[str, bytes]
    out_dir: Path

    @classmethod
    def from_env(cls) -> EnclaveConfig:
        """Build a config from the process environment.

        Raises:
            ValueError: if a variable with no safe default is missing. The
                enclave fails closed rather than guessing a project or key.
        """
        raw_banks = os.environ.get("MULENET_ENCLAVE_BANK_IDS", "")
        bank_ids = tuple(b.strip() for b in raw_banks.split(",") if b.strip())
        if not bank_ids:
            raise ValueError(
                "MULENET_ENCLAVE_BANK_IDS is not set. Pass a comma-separated "
                "list of bank ids, e.g. bank-a,bank-b,bank-c."
            )
        operator_project = _require("MULENET_ENCLAVE_PROJECT")
        inbox_bucket = _require("MULENET_ENCLAVE_INBOX_BUCKET")
        outbox_bucket = _require("MULENET_ENCLAVE_OUTBOX_BUCKET")
        evidence_bucket = _require("MULENET_ENCLAVE_EVIDENCE_BUCKET")
        kms_key = _require("MULENET_ENCLAVE_KMS_KEY")
        hash_kms_key = _require("MULENET_ENCLAVE_HASH_KMS_KEY")
        hash_key_ciphertext_uri = _require("MULENET_ENCLAVE_HASH_KEY_URI")
        wip_provider_audience = _require("MULENET_ENCLAVE_WIP_AUDIENCE")
        workload_service_account = _require("MULENET_ENCLAVE_WORKLOAD_SA")

        import base64

        outbox_public_keys: dict[str, bytes] = {}
        for bank_id in bank_ids:
            key_var = f"MULENET_ENCLAVE_OUTBOX_PUBKEY_{bank_id.upper().replace('-', '_')}"
            raw_key = _require(key_var)
            outbox_public_keys[bank_id] = base64.b64decode(raw_key)

        return cls(
            operator_project=operator_project,
            inbox_bucket=inbox_bucket,
            outbox_bucket=outbox_bucket,
            evidence_bucket=evidence_bucket,
            kms_key=kms_key,
            hash_kms_key=hash_kms_key,
            hash_key_ciphertext_uri=hash_key_ciphertext_uri,
            wip_provider_audience=wip_provider_audience,
            workload_service_account=workload_service_account,
            bank_ids=bank_ids,
            outbox_public_keys=outbox_public_keys,
            out_dir=_env_path("MULENET_ENCLAVE_OUT_DIR", Path("/tmp/mulenet-enclave")),
        )
