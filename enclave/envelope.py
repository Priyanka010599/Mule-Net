"""Envelope encryption for the cross-bank bundles the banks send the enclave.

Why envelope encryption rather than handing the payload to Cloud KMS directly:
KMS ``encrypt``/``decrypt`` caps the plaintext at 64 KiB, and a bank's
cross-bank edge list is much larger than that. So each bundle carries its own
random AES-256-GCM data-encryption key (DEK), and only the 32-byte DEK is
wrapped by the KMS key. The KMS key is the thing whose release is gated on
attestation; the DEK is a per-bundle detail.

The split of KMS rights is what makes the operator blind:

* a bank holds ``roles/cloudkms.cryptoKeyEncrypter`` on the enclave key, so it
  can wrap a DEK and cannot unwrap anyone's,
* the attested workload holds ``roles/cloudkms.cryptoKeyDecrypter``, so it can
  unwrap,
* the operator holds neither, so the ciphertext sitting in its own inbox
  bucket is not readable by it.

``KeyWrapper`` is a protocol rather than a concrete KMS client so that the
crypto in this module is testable without Google Cloud, and so that the
negative test (Stage 1) can substitute a wrapper that refuses.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from enclave.config import DEK_BYTES, GCM_NONCE_BYTES

#: Version tag written into every bundle. A reader that does not recognise the
#: version refuses the bundle rather than guessing at its layout.
BUNDLE_FORMAT = "mulenet-enclave-bundle/1"


class KeyWrapperError(RuntimeError):
    """A wrap or unwrap call was refused.

    Raised in place of a provider-specific error so that callers -- and the
    Stage 1 negative test -- see one type whatever the backend. The message
    carries the backend's own text verbatim, because for the negative test the
    exact refusal *is* the evidence.
    """


class KeyWrapper(Protocol):
    """Wraps and unwraps a data-encryption key with a key held elsewhere."""

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        """Return ``dek`` encrypted under the remote key."""
        ...

    def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
        """Return the plaintext DEK.

        Raises:
            KeyWrapperError: if the remote key holder refuses. On Confidential
                Space this is where a failed attestation surfaces.
        """
        ...


@dataclass(frozen=True)
class SealedBundle:
    """A bank's payload, encrypted so only the attested enclave can read it.

    Attributes:
        bank_id: Which bank produced this. Authenticated, not merely labelled:
            it is bound into the AES-GCM additional authenticated data, so an
            operator that relabels a bundle from one bank as another's makes it
            fail to decrypt rather than silently swapping attribution.
        wrapped_dek: The per-bundle DEK, encrypted under the enclave's KMS key.
        nonce: AES-GCM nonce for ``ciphertext``.
        ciphertext: The AES-256-GCM sealed payload.
    """

    bank_id: str
    wrapped_dek: bytes
    nonce: bytes
    ciphertext: bytes

    def to_json(self) -> str:
        """Serialise to the on-the-wire JSON a bank uploads to the inbox."""
        import base64

        return json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "bank_id": self.bank_id,
                "wrapped_dek": base64.b64encode(self.wrapped_dek).decode("ascii"),
                "nonce": base64.b64encode(self.nonce).decode("ascii"),
                "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> SealedBundle:
        """Parse a bundle, refusing an unknown format version.

        Raises:
            ValueError: if the format tag is missing or unrecognised, or a
                required field is absent.
        """
        import base64

        obj = json.loads(raw)
        if obj.get("format") != BUNDLE_FORMAT:
            raise ValueError(
                f"unsupported bundle format {obj.get('format')!r}, expected {BUNDLE_FORMAT!r}"
            )
        missing = {"bank_id", "wrapped_dek", "nonce", "ciphertext"} - obj.keys()
        if missing:
            raise ValueError(f"bundle is missing required fields: {sorted(missing)}")
        return cls(
            bank_id=obj["bank_id"],
            wrapped_dek=base64.b64decode(obj["wrapped_dek"]),
            nonce=base64.b64decode(obj["nonce"]),
            ciphertext=base64.b64decode(obj["ciphertext"]),
        )


def aad_for(bank_id: str) -> bytes:
    """Additional authenticated data binding a payload to its sending bank.

    Included in both the AES-GCM seal and the KMS wrap so that neither the
    payload nor its DEK can be moved between banks' bundles.
    """
    return f"{BUNDLE_FORMAT}:{bank_id}".encode()


def seal(bank_id: str, plaintext: bytes, wrapper: KeyWrapper) -> SealedBundle:
    """Encrypt ``plaintext`` for the enclave. Run by a bank, not the enclave.

    Args:
        bank_id: The sending bank's id.
        plaintext: The serialised payload (see :mod:`enclave.bundle`).
        wrapper: Something holding encrypt rights on the enclave's KMS key.

    Returns:
        A :class:`SealedBundle` safe to hand to the operator's inbox bucket.
    """
    dek = os.urandom(DEK_BYTES)
    nonce = os.urandom(GCM_NONCE_BYTES)
    aad = aad_for(bank_id)
    ciphertext = AESGCM(dek).encrypt(nonce, plaintext, aad)
    return SealedBundle(
        bank_id=bank_id,
        wrapped_dek=wrapper.wrap(dek, aad),
        nonce=nonce,
        ciphertext=ciphertext,
    )


def unseal(bundle: SealedBundle, wrapper: KeyWrapper) -> bytes:
    """Decrypt a bundle. Runs only inside the enclave.

    Args:
        bundle: The sealed bundle read from the inbox.
        wrapper: Something holding decrypt rights on the enclave's KMS key --
            on Confidential Space, rights conditioned on this image's
            attestation.

    Returns:
        The plaintext payload.

    Raises:
        KeyWrapperError: if KMS refuses to release the key. This is the
            failure the Stage 1 negative test is looking for.
        InvalidTag: if the ciphertext or the bank attribution was tampered
            with. Propagated unchanged, because a GCM tag failure is a
            different event from a key-release refusal and collapsing the two
            would hide which one happened.
    """
    aad = aad_for(bundle.bank_id)
    dek = wrapper.unwrap(bundle.wrapped_dek, aad)
    if len(dek) != DEK_BYTES:
        raise ValueError(
            f"unwrapped DEK is {len(dek)} bytes, expected {DEK_BYTES}; refusing to decrypt"
        )
    return AESGCM(dek).decrypt(bundle.nonce, bundle.ciphertext, aad)


def unseal_is_refused(bundle: SealedBundle, wrapper: KeyWrapper) -> tuple[bool, str]:
    """Attempt an unseal and report whether the key was refused.

    Used by the negative test. It returns the backend's refusal text rather
    than a boolean alone, and deliberately does not swallow :class:`InvalidTag`
    into the same answer: "KMS would not give me the key" and "the key worked
    but the payload was corrupt" are different findings, and only the first one
    proves attestation is doing its job.

    Returns:
        ``(True, message)`` if the key was refused, else ``(False, "")``.
    """
    try:
        unseal(bundle, wrapper)
    except KeyWrapperError as exc:
        return True, str(exc)
    except InvalidTag:
        return False, ""
    return False, ""
