"""Cloud KMS key release, gated on Confidential Space attestation.

This is the only module in ``enclave/`` that talks to Cloud KMS, and it is the
module whose refusal is the Stage 1 showpiece.

## Two identities, on purpose

The enclave uses **two different Google identities**, and keeping them separate
is what lets a run that was refused the key still write down *why*:

* **Cloud KMS decrypt** uses the *attested* federated principal, authorised by
  an IAM binding on the key naming ``attribute.image_digest/<digest>``.
* **GCS evidence and inbox reads** use the VM's *attached service account*,
  authorised by ordinary IAM roles on the buckets.

The KMS path builds an ``external_account`` credential from the attestation
token and uses it **directly** -- no service-account impersonation. That matters
and is not a detail: if the federated token were exchanged for the workload
service account, the caller at KMS would *be* that service account, the
digest-scoped ``principalSet`` binding would never be consulted, and the whole
"the key follows the code" claim would quietly reduce to "whoever holds this
service account can decrypt". ``service_account_email`` on the resulting
credential is therefore ``None`` by design.

The GCS path uses Application Default Credentials, which on a Confidential VM
resolve to the service account attached with ``--service-account``, served by the
instance metadata server. That credential does not depend on the attestation
exchange at all.

**Consequence:** when attestation fails, the KMS call is refused and the GCS
write still succeeds. That is precisely why the evidence model can be "write the
refusal to a bucket" rather than "hope the refusal reached the serial console".
The separation is a property of the design, not luck -- but note the flip side,
stated plainly: the workload service account *can* write to the evidence bucket
without any attestation, so an evidence object alone proves a VM ran, not that it
was attested. What proves attestation is the *content*: a released key, or a
refusal naming the check that refused it.

## Where a refusal lands

Two different failures both mean "this image may not have the key", and the
negative tests must tell them apart because they prove different things:

* the **STS exchange** fails -- the provider's attribute condition rejected the
  token outright, so no credential was ever minted. This is what a *debug* image
  hits: ``dbgstat`` disqualifies it before any key is involved.
* the **KMS decrypt** fails with 403 -- the token was acceptable but the key's
  IAM binding does not name this image's digest. This is what a *tampered* image
  hits.

:func:`classify_refusal` labels which one happened, and the label is written into
the evidence object so the distinction survives into the report.

**Unexecuted.** Nothing here has run against real Google Cloud. Request shapes
were checked against the installed ``google-cloud-kms`` protobuf descriptors and
``google-auth``; the IAM and attestation behaviour is unverified. See
``docs/enclave.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enclave.config import (
    ATTESTATION_TOKEN_PATH,
    STS_TOKEN_URL,
    SUBJECT_TOKEN_TYPE,
)
from enclave.envelope import KeyWrapperError

#: ``dbgstat`` value that means debugging has been off since boot -- i.e. the
#: production Confidential Space image. The debug image reports ``"enable"``.
DBGSTAT_PRODUCTION = "disabled-since-boot"

#: Support attribute present on a hardened (production) image and absent from the
#: debug image. Google's own image database marks a hardened entry
#: ``[LATEST, STABLE, USABLE]`` and a debug entry with no labels at all.
SUPPORT_ATTRIBUTE_STABLE = "STABLE"

#: Refusal classifications written into the evidence object.
REFUSAL_STS = "sts_exchange_refused"
REFUSAL_KMS = "kms_decrypt_refused"
REFUSAL_UNKNOWN = "refused_unclassified"


def credential_config(
    wip_provider_audience: str,
    token_path: Path = ATTESTATION_TOKEN_PATH,
) -> dict[str, Any]:
    """Build the external-account credential config for this attestation.

    Deliberately omits ``service_account_impersonation_url``: the federated
    principal is used directly so that the KMS key's
    ``attribute.image_digest/<digest>`` binding is the thing that authorises the
    call. See the module docstring -- impersonating a service account here would
    silently move the authorisation decision off the attestation.

    Args:
        wip_provider_audience: ``//iam.googleapis.com/projects/{NUMBER}/
            locations/global/workloadIdentityPools/{POOL}/providers/{PROVIDER}``.
        token_path: Where the container launcher wrote the attestation token.

    Returns:
        A dict accepted by ``google.auth.identity_pool.Credentials.from_info``.
    """
    return {
        "type": "external_account",
        "audience": wip_provider_audience,
        "subject_token_type": SUBJECT_TOKEN_TYPE,
        "token_url": STS_TOKEN_URL,
        "credential_source": {"file": str(token_path)},
    }


def attestation_claims(token_path: Path = ATTESTATION_TOKEN_PATH) -> dict[str, Any]:
    """Decode the attestation token's claims **without verifying the signature**.

    For the evidence object only. The security decision is made by STS and IAM
    on the server side, never here: a workload that decided for itself whether
    its own attestation was acceptable would be deciding whether to trust
    itself. This deliberately does not verify, and says so, so nobody later
    mistakes it for a check.

    Returns:
        The token's payload claims, or ``{}`` if the token is absent or
        unparseable (the normal case off a Confidential VM).
    """
    import base64

    try:
        raw = Path(token_path).read_text().strip()
    except OSError:
        return {}
    parts = raw.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return {}


def attestation_posture(claims: dict[str, Any]) -> dict[str, Any]:
    """Extract the claims that decide whether this VM may hold bank data.

    Recorded in every evidence object, for two reasons. The obvious one is
    auditing. The load-bearing one is that the exact *shape* of
    ``support_attributes`` could not be verified offline -- the documentation
    describes it both as a list and as a concatenated string
    (``LATEST=STABLE=USABLE``) -- so the first real run has to settle it. This
    function records the raw value and its Python type alongside the
    interpretation, so the evidence object answers the question rather than
    hiding it.

    This is **observation, not enforcement**. Nothing in the workload refuses
    itself on the strength of these values; the workload identity pool provider
    does that server-side.

    Returns:
        A dict safe to serialise into the evidence object.
    """
    confidential_space = claims.get("submods", {}).get("confidential_space", {})
    support_raw = confidential_space.get("support_attributes")

    if isinstance(support_raw, str):
        # The concatenated form, e.g. "LATEST=STABLE=USABLE".
        stable_present: bool | None = SUPPORT_ATTRIBUTE_STABLE in support_raw.split("=")
    elif isinstance(support_raw, (list, tuple)):
        stable_present = SUPPORT_ATTRIBUTE_STABLE in support_raw
    elif isinstance(support_raw, dict):
        # The `support_attributes.stable == true` form some examples show.
        stable_present = bool(support_raw.get(SUPPORT_ATTRIBUTE_STABLE.lower()))
    else:
        stable_present = None

    dbgstat = claims.get("dbgstat")
    return {
        "swname": claims.get("swname"),
        "hwmodel": claims.get("hwmodel"),
        "secboot": claims.get("secboot"),
        "dbgstat": dbgstat,
        "debugging_disabled_since_boot": (
            None if dbgstat is None else dbgstat == DBGSTAT_PRODUCTION
        ),
        "support_attributes_raw": support_raw,
        "support_attributes_type": type(support_raw).__name__,
        "stable_support_attribute_present": stable_present,
        "image_digest": claims.get("submods", {}).get("container", {}).get("image_digest"),
        "image_reference": (
            claims.get("submods", {}).get("container", {}).get("image_reference")
        ),
    }


def looks_like_debug_image(posture: dict[str, Any]) -> bool | None:
    """Best-effort read of whether this VM is running a debug image.

    For the evidence object's human-readable summary only. Returns ``None`` when
    the claims do not say, because "we could not tell" and "it is production"
    must not collapse into the same answer in a security report.
    """
    if posture.get("dbgstat") is not None:
        return posture["dbgstat"] != DBGSTAT_PRODUCTION
    if posture.get("stable_support_attribute_present") is not None:
        return not posture["stable_support_attribute_present"]
    return None


def classify_refusal(message: str) -> str:
    """Label a refusal as an STS-exchange or a KMS-decrypt failure.

    The two prove different properties -- see the module docstring -- so the
    evidence records which happened rather than reporting a generic "refused".
    Falls back to :data:`REFUSAL_UNKNOWN` rather than guessing: a mislabelled
    refusal would let a debug-image test claim it proved a digest binding.
    """
    lowered = message.lower()
    kms_markers = (
        "cloudkms.cryptokeyversions.usetodecrypt",
        "cryptokeydecrypter",
        "cloudkms",
        "cryptokeys/",
    )
    sts_markers = (
        "sts.googleapis.com",
        "attribute condition",
        "attributecondition",
        "invalid_grant",
        "unable to exchange",
        "could not build a federated credential",
        "permission denied on the workload identity pool",
    )
    if any(marker in lowered for marker in sts_markers):
        return REFUSAL_STS
    if any(marker in lowered for marker in kms_markers):
        return REFUSAL_KMS
    return REFUSAL_UNKNOWN


class KmsKeyWrapper:
    """A :class:`~enclave.envelope.KeyWrapper` backed by Cloud KMS.

    Implements ``wrap``/``unwrap`` via ``cryptoKeys.encrypt``/``decrypt``. The
    AAD from :mod:`enclave.envelope` is passed through as KMS's
    ``additional_authenticated_data``, so the bank binding is enforced by KMS
    too, not only by the local AES-GCM layer.
    """

    def __init__(self, key_name: str, credentials: Any | None = None) -> None:
        """
        Args:
            key_name: ``projects/{p}/locations/{l}/keyRings/{kr}/cryptoKeys/{k}``.
            credentials: Credentials to use. ``None`` means Application Default
                Credentials, which is what a non-enclave caller (a bank sealing
                a bundle) uses.
        """
        from google.cloud import kms

        self._key_name = key_name
        # No explicit quota_project_id: tried live, pinning it to the key's
        # own project ("mulenet") made KMS refuse with USER_PROJECT_DENIED,
        # because that requires the CALLER (an impersonated bank-sender
        # identity) to hold roles/serviceusage.serviceUsageConsumer on
        # mulenet -- a broader grant this design deliberately withholds from
        # bank identities. The unpinned default (no quota project header)
        # is what a working send actually used; a prior send's failure
        # attributing usage to the bank's own (correctly KMS-less) project
        # was transient, not caused by omitting this.
        self._client = kms.KeyManagementServiceClient(credentials=credentials)

    @classmethod
    def from_attestation(
        cls,
        key_name: str,
        wip_provider_audience: str,
        token_path: Path = ATTESTATION_TOKEN_PATH,
    ) -> KmsKeyWrapper:
        """Build a wrapper whose credential comes from this image's attestation.

        Raises:
            KeyWrapperError: if the federated credential cannot be built or the
                STS exchange is refused. That refusal is itself a valid Stage 1
                result -- it means the provider's attribute condition rejected
                this image before any key was involved, which is what a debug
                image is expected to hit.
        """
        from google.auth import identity_pool

        info = credential_config(wip_provider_audience, token_path)
        try:
            credentials = identity_pool.Credentials.from_info(info)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim on purpose
            raise KeyWrapperError(
                f"could not build a federated credential from the attestation token: {exc}"
            ) from exc
        return cls(key_name, credentials=credentials)

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        """Encrypt ``dek`` under the KMS key. Run by a bank, not the enclave."""
        try:
            response = self._client.encrypt(
                request={
                    "name": self._key_name,
                    "plaintext": dek,
                    "additional_authenticated_data": aad,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise KeyWrapperError(f"KMS encrypt refused: {exc}") from exc
        return response.ciphertext

    def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
        """Decrypt the DEK. Succeeds only for an image the release policy allows.

        Raises:
            KeyWrapperError: carrying the backend's message verbatim. For the
                negative tests this text is the evidence, so it is neither
                reworded nor truncated.
        """
        try:
            response = self._client.decrypt(
                request={
                    "name": self._key_name,
                    "ciphertext": wrapped_dek,
                    "additional_authenticated_data": aad,
                }
            )
        except Exception as exc:  # noqa: BLE001
            raise KeyWrapperError(str(exc)) from exc
        return response.plaintext

    def decrypt_blob(self, ciphertext: bytes, aad: bytes = b"") -> bytes:
        """Decrypt a small KMS-encrypted blob, e.g. the Stage 4 HMAC key.

        Used for values that fit inside KMS's 64 KiB limit and therefore need no
        envelope of their own.
        """
        return self.unwrap(ciphertext, aad)
