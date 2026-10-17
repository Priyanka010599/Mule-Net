"""Tests for per-bank outbox encryption (see enclave/outbox_crypto.py).

Mirrors enclave/test_envelope.py's shape: round trip, tamper detection, and
the property that matters most here -- a party without the bank's private key,
including the operator holding only the sealed object, cannot recover the
plaintext.
"""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from enclave.outbox_crypto import (
    SealedOutbox,
    generate_keypair,
    seal_for_bank,
    unseal_with_private_key,
)


def test_round_trip():
    private_key, public_key = generate_keypair()
    plaintext = b'{"own_members": ["abc123"], "bank_id": "bank-a"}'
    sealed = seal_for_bank("bank-a", plaintext, public_key)
    assert unseal_with_private_key(sealed, private_key) == plaintext


def test_json_round_trip():
    """The wire format survives serialisation, as it must to sit in GCS."""
    private_key, public_key = generate_keypair()
    plaintext = b"alert bundle payload"
    sealed = seal_for_bank("bank-b", plaintext, public_key)
    restored = SealedOutbox.from_json(sealed.to_json())
    assert unseal_with_private_key(restored, private_key) == plaintext


def test_operator_without_private_key_cannot_read():
    """The whole point: holding the sealed object is not enough."""
    _, public_key = generate_keypair()
    other_private_key, _ = generate_keypair()
    sealed = seal_for_bank("bank-c", b"secret alert", public_key)
    with pytest.raises(InvalidTag):
        unseal_with_private_key(sealed, other_private_key)


def test_wrong_bank_id_fails_even_with_the_right_key():
    """A relabelled bundle must fail to decrypt, not silently re-attribute."""
    private_key, public_key = generate_keypair()
    sealed = seal_for_bank("bank-a", b"payload", public_key)
    relabelled = SealedOutbox(
        bank_id="bank-b",
        ephemeral_public_key=sealed.ephemeral_public_key,
        nonce=sealed.nonce,
        ciphertext=sealed.ciphertext,
    )
    with pytest.raises(InvalidTag):
        unseal_with_private_key(relabelled, private_key)


def test_tampered_ciphertext_fails():
    private_key, public_key = generate_keypair()
    sealed = seal_for_bank("bank-a", b"payload", public_key)
    tampered = SealedOutbox(
        bank_id=sealed.bank_id,
        ephemeral_public_key=sealed.ephemeral_public_key,
        nonce=sealed.nonce,
        ciphertext=sealed.ciphertext[:-1] + bytes([sealed.ciphertext[-1] ^ 0xFF]),
    )
    with pytest.raises(InvalidTag):
        unseal_with_private_key(tampered, private_key)


def test_unknown_format_is_refused():
    with pytest.raises(ValueError, match="unsupported outbox format"):
        SealedOutbox.from_json('{"format": "something-else"}')


def test_missing_fields_are_refused():
    import json

    payload = json.dumps({"format": "mulenet-outbox-bundle/1", "bank_id": "bank-a"})
    with pytest.raises(ValueError, match="missing required fields"):
        SealedOutbox.from_json(payload)
