"""Per-bank encryption of the enclave's outbox alerts.

``enclave/policy.py`` already redacts each bank's alert bundle so it names no
other bank's account ids. What it does not do is stop the operator -- who owns
the outbox bucket -- from reading a bundle before the bank downloads it. That
gap is named plainly in ``docs/data_sharing.md``: "the operator can currently
read the alerts."

This module closes it without giving any bank project a Cloud KMS key ring of
its own, which would need billing enabled on that project -- exactly what the
identity-only bank model (see ``infra/enclave_bank_identity.sh``) exists to
avoid, and ``mulenet-bank-c`` is deliberately unbilled.

Instead each bank generates an X25519 keypair **offline**, outside any Google
Cloud project, and gives the enclave only its public key (not a secret; safe to
pass as VM metadata like everything else in ``enclave/config.py``). The enclave
encrypts each bank's bundle to that public key before upload. Only the bank's
own, never-uploaded private key can open it -- not the operator, not KMS, not
the enclave itself once the bundle is written.

The construction is anonymous ECIES: a fresh ephemeral X25519 keypair per
message, an ECDH shared secret with the recipient's public key, HKDF-SHA256 to
derive an AES-256 key, then AES-256-GCM -- the same AEAD ``enclave/envelope.py``
already uses for the inbox side, so this file introduces one new primitive
(X25519 + HKDF) rather than a second encryption scheme end to end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from enclave.config import GCM_NONCE_BYTES

#: Version tag written into every sealed outbox object. A reader that does not
#: recognise it refuses rather than guessing at the layout -- same rule as
#: enclave/envelope.py's BUNDLE_FORMAT.
OUTBOX_FORMAT = "mulenet-outbox-bundle/1"

#: Raw X25519 key length in bytes (both public and private).
X25519_KEY_BYTES = 32

#: AES-256 key length in bytes, derived by HKDF from the ECDH shared secret.
_AES_KEY_BYTES = 32


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate one bank's outbox keypair. Run offline, once, by the bank.

    Returns:
        ``(private_key_bytes, public_key_bytes)``, each 32 raw bytes. The
        private half never leaves the bank; only the public half is given to
        the enclave.
    """
    private_key = X25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return private_bytes, public_bytes


def _derive_aes_key(shared_secret: bytes, aad: bytes) -> bytes:
    """HKDF-SHA256 over the ECDH secret, bound to ``aad`` for domain separation."""
    return HKDF(
        algorithm=hashes.SHA256(), length=_AES_KEY_BYTES, salt=None, info=aad
    ).derive(shared_secret)


@dataclass(frozen=True)
class SealedOutbox:
    """One bank's alert bundle, encrypted so only that bank can read it.

    Attributes:
        bank_id: The receiving bank. Bound into the AEAD, so a bundle cannot be
            silently relabelled to another bank's outbox prefix and decrypted
            there -- it is a different HKDF input and the GCM tag fails.
        ephemeral_public_key: The one-time X25519 public key the enclave
            generated for this message. Safe to publish; without the matching
            ephemeral private key (which the enclave discards immediately) it
            gives an attacker nothing.
        nonce: AES-GCM nonce for ``ciphertext``.
        ciphertext: The AES-256-GCM sealed bundle.
    """

    bank_id: str
    ephemeral_public_key: bytes
    nonce: bytes
    ciphertext: bytes

    def to_json(self) -> str:
        import base64

        return json.dumps(
            {
                "format": OUTBOX_FORMAT,
                "bank_id": self.bank_id,
                "ephemeral_public_key": base64.b64encode(self.ephemeral_public_key).decode("ascii"),
                "nonce": base64.b64encode(self.nonce).decode("ascii"),
                "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> SealedOutbox:
        import base64

        obj = json.loads(raw)
        if obj.get("format") != OUTBOX_FORMAT:
            raise ValueError(
                f"unsupported outbox format {obj.get('format')!r}, expected {OUTBOX_FORMAT!r}"
            )
        missing = {"bank_id", "ephemeral_public_key", "nonce", "ciphertext"} - obj.keys()
        if missing:
            raise ValueError(f"sealed outbox is missing required fields: {sorted(missing)}")
        return cls(
            bank_id=obj["bank_id"],
            ephemeral_public_key=base64.b64decode(obj["ephemeral_public_key"]),
            nonce=base64.b64decode(obj["nonce"]),
            ciphertext=base64.b64decode(obj["ciphertext"]),
        )


def _aad_for(bank_id: str) -> bytes:
    return f"{OUTBOX_FORMAT}:{bank_id}".encode()


def seal_for_bank(bank_id: str, plaintext: bytes, recipient_public_key: bytes) -> SealedOutbox:
    """Encrypt ``plaintext`` so only the holder of the matching private key can read it.

    Run inside the enclave, once per bank per ring. Never run by the operator or
    by any bank: the enclave is the only party that holds the plaintext alert
    bundle to begin with.

    Args:
        bank_id: The receiving bank. Bound into the AEAD.
        plaintext: The redacted alert bundle's serialised JSON.
        recipient_public_key: That bank's X25519 public key, 32 raw bytes, from
            ``EnclaveConfig.outbox_public_keys``.

    Returns:
        A :class:`SealedOutbox` safe to upload to the outbox bucket the
        operator owns.
    """
    ephemeral_private = X25519PrivateKey.generate()
    ephemeral_public_bytes = ephemeral_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    recipient_key = X25519PublicKey.from_public_bytes(recipient_public_key)
    shared_secret = ephemeral_private.exchange(recipient_key)
    aad = _aad_for(bank_id)
    aes_key = _derive_aes_key(shared_secret, aad)
    nonce = os.urandom(GCM_NONCE_BYTES)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, aad)
    return SealedOutbox(
        bank_id=bank_id,
        ephemeral_public_key=ephemeral_public_bytes,
        nonce=nonce,
        ciphertext=ciphertext,
    )


def unseal_with_private_key(sealed: SealedOutbox, recipient_private_key: bytes) -> bytes:
    """Decrypt a sealed outbox object. Run by the receiving bank, never by the enclave.

    Args:
        sealed: The object downloaded from the bank's outbox prefix.
        recipient_private_key: The bank's own X25519 private key, 32 raw bytes,
            generated by :func:`generate_keypair` and never uploaded anywhere.

    Returns:
        The plaintext alert bundle JSON.

    Raises:
        cryptography.exceptions.InvalidTag: if the ciphertext, the ephemeral
            key, or the bank attribution was tampered with, or if this is not
            the private key the bundle was sealed to.
    """
    private_key = X25519PrivateKey.from_private_bytes(recipient_private_key)
    ephemeral_key = X25519PublicKey.from_public_bytes(sealed.ephemeral_public_key)
    shared_secret = private_key.exchange(ephemeral_key)
    aad = _aad_for(sealed.bank_id)
    aes_key = _derive_aes_key(shared_secret, aad)
    return AESGCM(aes_key).decrypt(sealed.nonce, sealed.ciphertext, aad)
