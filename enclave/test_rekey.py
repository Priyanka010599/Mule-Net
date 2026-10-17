"""Tests for in-enclave re-keying (Stage 4).

The two properties that matter are opposites of each other, so both are pinned
here: the same account at two banks must re-key to the **same** identifier (or
cross-bank detection silently collapses to single-bank), and the identifier must
be **unreachable** without the enclave key (or the shared-key problem survives
the change).
"""

from __future__ import annotations

import os

import pytest

from enclave.rekey import (
    ENCLAVE_ID_HEX_LEN,
    AccountRekeyer,
    RekeyError,
    enclave_account_id,
)

KEY_A = b"k" * 32
KEY_B = b"j" * 32


def test_identifier_is_deterministic():
    first = enclave_account_id(KEY_A, "bank-a", "hash-001")
    second = enclave_account_id(KEY_A, "bank-a", "hash-001")
    assert first == second
    assert len(first) == ENCLAVE_ID_HEX_LEN


def test_both_banks_derive_the_same_id_for_the_same_account():
    """The joinability property. Without it, cross-bank detection dies quietly.

    Bank A reports an edge to account ``hash-b-7`` at bank-b; bank B reports the
    same account as one it owns. Both must land on one identifier or the edge
    and the account never join.
    """
    as_counterparty = enclave_account_id(KEY_A, "bank-b", "hash-b-7")
    as_owner = enclave_account_id(KEY_A, "bank-b", "hash-b-7")
    assert as_counterparty == as_owner


def test_a_different_key_gives_a_different_identifier_space():
    """A bank without the enclave key cannot compute the identifiers at all."""
    assert enclave_account_id(KEY_A, "bank-a", "hash-001") != enclave_account_id(
        KEY_B, "bank-a", "hash-001"
    )


def test_same_hash_at_different_banks_does_not_collide():
    """Two banks' accounts that happen to share an input hash stay distinct."""
    assert enclave_account_id(KEY_A, "bank-a", "same") != enclave_account_id(
        KEY_A, "bank-b", "same"
    )


def test_separator_prevents_boundary_collisions():
    """('bank-ab', 'cd') and ('bank-a', 'bcd') must not hash to one identifier.

    Without a separator the concatenation is ambiguous and two unrelated
    accounts merge into one node -- which would fabricate a ring.
    """
    assert enclave_account_id(KEY_A, "bank-ab", "cd") != enclave_account_id(
        KEY_A, "bank-a", "bcd"
    )


@pytest.mark.parametrize(
    "bank_id, hashed_id",
    [("", "hash-001"), ("bank-a", "")],
)
def test_empty_components_are_refused(bank_id, hashed_id):
    """Failing closed: an empty input would map many accounts to one id."""
    with pytest.raises(RekeyError):
        enclave_account_id(KEY_A, bank_id, hashed_id)


def test_short_key_is_refused():
    """A short key usually means the KMS release returned something unexpected."""
    with pytest.raises(RekeyError, match="refusing to re-key"):
        enclave_account_id(b"short", "bank-a", "hash-001")


def test_reserved_separator_in_bank_id_is_refused():
    with pytest.raises(RekeyError, match="reserved separator"):
        enclave_account_id(KEY_A, "bank\x1fa", "hash-001")


class TestAccountRekeyer:
    def test_records_ownership(self):
        rekeyer = AccountRekeyer(KEY_A)
        enclave_id = rekeyer.add("bank-a", "hash-001")
        assert rekeyer.owner_of(enclave_id) == "bank-a"

    def test_unclaimed_account_has_no_owner(self):
        """An account seen only as a far endpoint is released to nobody."""
        rekeyer = AccountRekeyer(KEY_A)
        far = rekeyer.rekey_counterparty("bank-b", "hash-b-1")
        assert rekeyer.owner_of(far) is None

    def test_counterparty_rekey_does_not_confer_ownership(self):
        """Naming a counterparty must not let a bank claim it."""
        rekeyer = AccountRekeyer(KEY_A)
        rekeyer.rekey_counterparty("bank-b", "hash-b-1")
        assert rekeyer.owned_ids == {}

    def test_re_adding_the_same_owner_is_idempotent(self):
        rekeyer = AccountRekeyer(KEY_A)
        first = rekeyer.add("bank-a", "hash-001")
        second = rekeyer.add("bank-a", "hash-001")
        assert first == second

    def test_two_banks_claiming_one_account_is_a_conflict(self):
        """Last-write-wins here would route a bank's alerts to the wrong bank."""
        rekeyer = AccountRekeyer(KEY_A)
        # Both banks claim an account with the same identifier *and* the same
        # declared owning bank -- the only way two claims can collide, since the
        # owning bank is bound into the hash.
        enclave_id = rekeyer.add("bank-a", "hash-001")
        rekeyer._owner[enclave_id] = "bank-b"  # simulate a prior conflicting claim
        with pytest.raises(RekeyError, match="cannot also claim it"):
            rekeyer.add("bank-a", "hash-001")

    def test_owned_ids_is_a_copy(self):
        """A caller must not be able to rewrite ownership through the accessor."""
        rekeyer = AccountRekeyer(KEY_A)
        enclave_id = rekeyer.add("bank-a", "hash-001")
        rekeyer.owned_ids["injected"] = "bank-z"
        assert rekeyer.owner_of("injected") is None
        assert rekeyer.owner_of(enclave_id) == "bank-a"


def test_real_random_key_works():
    """Sanity check against a key the length KMS actually returns."""
    key = os.urandom(32)
    assert len(enclave_account_id(key, "bank-c", "hash-c-1")) == ENCLAVE_ID_HEX_LEN


class TestJoinabilityAcrossBanks:
    """The property the whole cross-bank result depends on.

    Spelled out separately from the unit tests above because it is the one that,
    if broken, fails *silently*: detection would collapse to single-bank and
    nothing would error. A test that only checked "hashing works" would not
    catch it.
    """

    def test_reporting_bank_and_owning_bank_agree_on_the_same_account(self):
        """Bank A names B's account as a counterparty; Bank B owns it.

        Both must arrive at one identifier. They do because the input binds the
        *owning* bank, which both parties already agree on -- not the bank doing
        the reporting.
        """
        shared_account = "b-hash-7"

        # Bank A's view: a counterparty that sits at bank-b.
        as_reported_by_a = AccountRekeyer(KEY_A).rekey_counterparty("bank-b", shared_account)

        # Bank B's view: an account it owns.
        rekeyer_b = AccountRekeyer(KEY_A)
        as_owned_by_b = rekeyer_b.add("bank-b", shared_account)

        assert as_reported_by_a == as_owned_by_b
        assert rekeyer_b.owner_of(as_reported_by_a) == "bank-b"

    def test_three_banks_all_derive_one_identifier_for_one_account(self):
        """Extends to the 3-bank case: A and C both point at B's account."""
        shared_account = "b-hash-7"
        derived = {
            AccountRekeyer(KEY_A).rekey_counterparty("bank-b", shared_account)
            for _ in range(3)
        }
        assert len(derived) == 1

    def test_a_bank_holding_a_different_key_cannot_reproduce_the_identifier(self):
        """The Stage 4 point: joinability no longer comes from a shared secret.

        A participant with its own key computes a different identifier, so it
        cannot test whether a guessed account appears in the enclave's graph.
        """
        with_enclave_key = enclave_account_id(KEY_A, "bank-b", "b-hash-7")
        with_a_banks_own_key = enclave_account_id(KEY_B, "bank-b", "b-hash-7")
        assert with_enclave_key != with_a_banks_own_key

    def test_the_retired_demo_key_cannot_reproduce_the_identifier(self):
        """Specifically: the published "demo-key" buys an attacker nothing here.

        It is the key this stage exists to retire, and it is public in this repo,
        so it is worth pinning that holding it does not let anyone into the
        enclave's identifier space.
        """
        demo_key = b"demo-key".ljust(32, b"\0")  # padded; the real one is shorter
        assert enclave_account_id(KEY_A, "bank-b", "b-hash-7") != enclave_account_id(
            demo_key, "bank-b", "b-hash-7"
        )

    def test_the_short_demo_key_is_refused_outright(self):
        """b"demo-key" is 8 bytes, so the enclave will not even use it."""
        with pytest.raises(RekeyError, match="refusing to re-key"):
            enclave_account_id(b"demo-key", "bank-b", "b-hash-7")
