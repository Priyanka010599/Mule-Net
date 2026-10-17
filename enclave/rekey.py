"""Re-keying account identifiers inside the enclave (Stage 4).

## The problem this solves

Today every account id in MuleNet is ``HMAC-SHA256(key, raw_id)`` under **one
key shared by every participant**, defaulting to ``"demo-key"``
(``simulator/config.py``). Two facts about that:

* The banks must agree on the key or cross-bank edges never join, so the key
  cannot simply be made per-bank.
* Any holder of the key can hash a guessed account number and test whether that
  account appears in another bank's data. Account numbers are low-entropy, so
  a shared HMAC key is a shared *capability to enumerate*, not just a shared
  format.

## What this module does

Identifiers are re-keyed **inside the enclave**, under a key Cloud KMS releases
only to the attested image:

    enclave_id = HMAC-SHA256(enclave_key, owning_bank_id || 0x1f || hashed_id)

The owning bank is bound into the input, not left out, because the same opaque
hash could otherwise be claimed by two banks. Both banks on a cross-bank edge
already agree on *which bank owns* each endpoint, so both derive the same
``enclave_id`` for the same account -- joinability survives -- while no bank
holds the key that produced it.

## What this demonstrates, and what it does not

It demonstrates:

* **No participant holds the joining key.** Joinability comes from the shared
  computation rather than a shared secret. A bank cannot test for the presence
  of a guessed account in another bank's data, because it cannot compute
  ``enclave_id`` at all.
* **Key release is attestation-gated.** Changing the image changes the digest,
  and the key is not released, so a modified enclave cannot even reproduce the
  identifier space of the real one.

It does **not** demonstrate:

* **Preimage protection for the existing data.** The input here is already
  ``HMAC-SHA256("demo-key", raw_id)``, and ``"demo-key"`` is a public default
  in this repo. Anyone who wants to enumerate raw account numbers against the
  *demo-era* hashes could always do so and still can; re-keying does not reach
  back and fix data already hashed under a published key. It protects the
  identifier space the enclave works in, going forward.
* **A production ingest path.** In a deployment the bank would seal the **raw**
  identifier and the enclave would hash it once, here, making the demo key
  disappear entirely. That is a change to ``simulator/`` and ``ingest/``, which
  this change does not own. What is built here is the enclave half: it accepts
  whatever identifier a bank sends as opaque input and re-keys it, so it works
  unchanged the day the input becomes a raw id.

So Stage 4 retires the shared key *as the thing that makes the join possible*.
It does not retroactively unpublish the demo hashes, and it should not be
described as if it did.
"""

from __future__ import annotations

import hmac
from hashlib import sha256

#: Separator between the bank id and the identifier. 0x1f (ASCII unit
#: separator) cannot occur in a bank id or a hex hash, so the concatenation is
#: unambiguous: without a separator, ("bank-ab", "cd") and ("bank-a", "bcd")
#: would hash identically and two different accounts would collide into one.
_SEP = b"\x1f"

#: Length of the hex identifier the enclave emits. Full SHA-256, not truncated:
#: truncation buys nothing here and raises collision probability across the
#: three banks' combined account population.
ENCLAVE_ID_HEX_LEN = 64


class RekeyError(ValueError):
    """An identifier could not be re-keyed and was refused, not silently passed."""


def enclave_account_id(enclave_key: bytes, owning_bank_id: str, hashed_account_id: str) -> str:
    """Re-key one account identifier into the enclave's identifier space.

    Args:
        enclave_key: The HMAC key KMS released to this attested image. Never
            logged, never written out, never leaves the enclave.
        owning_bank_id: The bank that owns this account. Bound into the input
            so two banks cannot claim the same opaque identifier.
        hashed_account_id: The identifier as the bank sent it. Treated as an
            opaque byte string -- this function does not care whether it is a
            demo-era HMAC or a raw account number, which is what lets the
            production ingest path drop in without changing the enclave.

    Returns:
        A 64-character lowercase hex identifier.

    Raises:
        RekeyError: if the key is too short to be a real released key, or
            either identifier component is empty. Failing closed matters: a
            silent empty-string input would map every such account to one
            identifier and merge unrelated accounts into a single "ring".
    """
    if len(enclave_key) < 32:
        raise RekeyError(
            f"enclave HMAC key is {len(enclave_key)} bytes; refusing to re-key with "
            "fewer than 32. A short key here usually means the KMS release "
            "returned something unexpected rather than the real key."
        )
    if not owning_bank_id:
        raise RekeyError("owning_bank_id is empty; refusing to re-key")
    if not hashed_account_id:
        raise RekeyError("hashed_account_id is empty; refusing to re-key")
    if _SEP in owning_bank_id.encode("utf-8"):
        raise RekeyError(f"owning_bank_id {owning_bank_id!r} contains the reserved separator")
    message = owning_bank_id.encode("utf-8") + _SEP + hashed_account_id.encode("utf-8")
    return hmac.new(enclave_key, message, sha256).hexdigest()


class AccountRekeyer:
    """Re-keys identifiers and remembers which bank owns each result.

    The ownership map is the whole basis of the Stage 3 output policy: an alert
    may only be released to the bank that owns the account it names, and after
    re-keying an identifier no longer carries any hint of its bank. So the map
    is built at the moment ownership is still known -- while reading each bank's
    authenticated bundle -- and never reconstructed by guessing afterwards.

    A second bank claiming an account already claimed by another is a conflict,
    not a last-write-wins update: silently reassigning ownership would route
    that account's alerts to the wrong bank.
    """

    def __init__(self, enclave_key: bytes) -> None:
        self._key = enclave_key
        self._owner: dict[str, str] = {}

    def add(self, owning_bank_id: str, hashed_account_id: str) -> str:
        """Re-key an account and record its owner.

        Returns:
            The enclave identifier.

        Raises:
            RekeyError: if a different bank already claimed this account.
        """
        enclave_id = enclave_account_id(self._key, owning_bank_id, hashed_account_id)
        existing = self._owner.get(enclave_id)
        if existing is not None and existing != owning_bank_id:
            raise RekeyError(
                f"account maps to enclave id {enclave_id[:12]}... already owned by "
                f"{existing!r}; {owning_bank_id!r} cannot also claim it"
            )
        self._owner[enclave_id] = owning_bank_id
        return enclave_id

    def rekey_counterparty(self, owning_bank_id: str, hashed_account_id: str) -> str:
        """Re-key an endpoint without asserting ownership of it.

        Used for the far side of a cross-bank edge: the sending bank knows which
        bank the counterparty sits at, but saying so must not make it the owner.
        """
        return enclave_account_id(self._key, owning_bank_id, hashed_account_id)

    def owner_of(self, enclave_id: str) -> str | None:
        """Return the bank that owns ``enclave_id``, or ``None`` if unclaimed.

        ``None`` is a real and expected answer: an account can appear only as
        the far end of another bank's edge if its own bank did not contribute.
        Those stay unowned and, by the Stage 3 policy, are released to nobody.
        """
        return self._owner.get(enclave_id)

    @property
    def owned_ids(self) -> dict[str, str]:
        """A copy of the enclave-id to owning-bank map."""
        return dict(self._owner)
