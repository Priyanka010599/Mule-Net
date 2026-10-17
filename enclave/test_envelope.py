"""Tests for envelope encryption and the shape of a KMS refusal.

These run without Google Cloud: :class:`FakeWrapper` stands in for Cloud KMS.
That is the point of the :class:`~enclave.envelope.KeyWrapper` protocol -- the
crypto and the refusal handling are testable on a laptop, so the only thing the
credentialed run has to establish is that *real* KMS refuses, not that the code
notices a refusal.
"""

from __future__ import annotations

import os

import pytest
from cryptography.exceptions import InvalidTag

from enclave.envelope import (
    BUNDLE_FORMAT,
    KeyWrapperError,
    SealedBundle,
    aad_for,
    seal,
    unseal,
    unseal_is_refused,
)


class FakeWrapper:
    """An in-memory stand-in for a KMS key, with AEAD-style AAD binding."""

    def __init__(self, key_id: str = "good-key") -> None:
        self.key_id = key_id
        self._store: dict[bytes, tuple[bytes, bytes]] = {}

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        handle = os.urandom(16)
        self._store[handle] = (dek, aad)
        return handle

    def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
        if wrapped_dek not in self._store:
            raise KeyWrapperError("no such wrapped key")
        dek, bound_aad = self._store[wrapped_dek]
        if bound_aad != aad:
            raise KeyWrapperError("AAD mismatch")
        return dek


class RefusingWrapper:
    """A wrapper that refuses to unwrap, the way KMS does on bad attestation.

    The message is the real shape of the error the credentialed negative test
    must produce: a 403 naming the permission that was denied.
    """

    MESSAGE = (
        "Permission 'cloudkms.cryptoKeyVersions.useToDecrypt' denied on resource "
        "'projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/"
        "edge-wrap' (or it may not exist)."
    )

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        return b"unusable"

    def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
        raise KeyWrapperError(self.MESSAGE)


def test_seal_unseal_round_trips():
    wrapper = FakeWrapper()
    payload = b'{"hello":"enclave"}'
    bundle = seal("bank-a", payload, wrapper)
    assert unseal(bundle, wrapper) == payload


def test_ciphertext_does_not_contain_plaintext():
    wrapper = FakeWrapper()
    payload = b"acct-000123456789 paid acct-000987654321"
    bundle = seal("bank-a", payload, wrapper)
    assert payload not in bundle.ciphertext
    assert b"acct-000123456789" not in bundle.ciphertext


def test_each_seal_uses_a_fresh_dek_and_nonce():
    """Two seals of identical plaintext must not produce identical ciphertext.

    A reused nonce under a reused key is catastrophic for AES-GCM, and an
    operator watching the inbox could otherwise tell that a bank re-sent the
    same edge list.
    """
    wrapper = FakeWrapper()
    first = seal("bank-a", b"same", wrapper)
    second = seal("bank-a", b"same", wrapper)
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert first.wrapped_dek != second.wrapped_dek


def test_relabelling_a_bundle_to_another_bank_fails_to_decrypt():
    """The operator must not be able to re-attribute a bundle silently."""
    wrapper = FakeWrapper()
    bundle = seal("bank-a", b"bank a edges", wrapper)
    relabelled = SealedBundle(
        bank_id="bank-b",
        wrapped_dek=bundle.wrapped_dek,
        nonce=bundle.nonce,
        ciphertext=bundle.ciphertext,
    )
    with pytest.raises(KeyWrapperError):
        unseal(relabelled, wrapper)


def test_tampered_ciphertext_raises_invalid_tag_not_a_refusal():
    """A corrupt payload must not be mistaken for a key-release refusal."""
    wrapper = FakeWrapper()
    bundle = seal("bank-a", b"bank a edges", wrapper)
    flipped = bytearray(bundle.ciphertext)
    flipped[0] ^= 0x01
    tampered = SealedBundle(
        bank_id=bundle.bank_id,
        wrapped_dek=bundle.wrapped_dek,
        nonce=bundle.nonce,
        ciphertext=bytes(flipped),
    )
    with pytest.raises(InvalidTag):
        unseal(tampered, wrapper)

    refused, message = unseal_is_refused(tampered, wrapper)
    assert refused is False
    assert message == ""


def test_refusal_is_reported_with_the_backend_message_verbatim():
    """The negative test's evidence is the refusal text, so it must survive."""
    bundle = seal("bank-a", b"payload", FakeWrapper())
    refused, message = unseal_is_refused(bundle, RefusingWrapper())
    assert refused is True
    assert message == RefusingWrapper.MESSAGE
    assert "cloudkms.cryptoKeyVersions.useToDecrypt" in message


def test_short_dek_is_refused():
    """A wrapper returning the wrong-length key must not be used anyway."""

    class ShortKeyWrapper(FakeWrapper):
        def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
            return b"too-short"

    bundle = seal("bank-a", b"payload", FakeWrapper())
    with pytest.raises(ValueError, match="refusing to decrypt"):
        unseal(bundle, ShortKeyWrapper())


def test_json_round_trip():
    wrapper = FakeWrapper()
    bundle = seal("bank-c", b"payload", wrapper)
    restored = SealedBundle.from_json(bundle.to_json())
    assert restored == bundle
    assert unseal(restored, wrapper) == b"payload"


def test_unknown_format_version_is_refused():
    with pytest.raises(ValueError, match="unsupported bundle format"):
        SealedBundle.from_json('{"format": "something-else"}')


def test_missing_field_is_refused():
    with pytest.raises(ValueError, match="missing required fields"):
        SealedBundle.from_json(f'{{"format": "{BUNDLE_FORMAT}", "bank_id": "bank-a"}}')


def test_aad_differs_per_bank():
    assert aad_for("bank-a") != aad_for("bank-b")
