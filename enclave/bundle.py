"""The payload a bank seals and the enclave opens.

This module deliberately defines **no new schema**. CLAUDE.md requires code to
be written against ``shared/`` contracts and never to redefine them locally,
and ``shared/`` is frozen for this change, so a bank's contribution is carried
as the contracts that already exist:

* :class:`shared.contracts.Transfer` -- one per cross-bank edge,
* :class:`shared.contracts.Account` -- one per account *this bank owns* that
  appears in those transfers, which is what tells the enclave who owns which
  side of an edge.

A dedicated ``CrossBankEdge`` wire contract would be a better fit and belongs
in ``shared/``; it is not added here because this change does not own that
folder. See ``docs/data_sharing.md`` for what would change if it were added.

What a bank must **not** put in a bundle, and what this module therefore has no
field for: balances, names, KYC documents, devices, or any transfer with both
ends inside the same bank. The cross-bank features in ``ml/features.py`` need
only boundary-crossing edges, so sending more would widen exposure for no
detection gain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from shared.contracts import Account, Transfer


class BundleError(ValueError):
    """A bundle was structurally invalid and was refused rather than repaired."""


@dataclass(frozen=True)
class BankContribution:
    """One bank's cross-bank edges plus the accounts it owns among them.

    Attributes:
        bank_id: The contributing bank.
        accounts: Accounts *owned by this bank* that appear as an endpoint of
            some transfer below. Their ``bank_id`` must all equal ``bank_id``:
            a bank asserting ownership of another bank's account would let it
            claim an account into its own alert bundle, so this is validated
            rather than trusted.
        transfers: Transfers with at least one endpoint at this bank. Both
            endpoints are hashed account ids; the enclave re-keys them before
            they are joined (see :mod:`enclave.rekey`).
    """

    bank_id: str
    accounts: tuple[Account, ...]
    transfers: tuple[Transfer, ...]

    def validate(self) -> None:
        """Check the invariants the enclave relies on.

        Raises:
            BundleError: if an account claims a different bank, or a transfer
                has neither endpoint among the declared accounts.
        """
        foreign = sorted({a.bank_id for a in self.accounts} - {self.bank_id})
        if foreign:
            raise BundleError(
                f"bank {self.bank_id!r} sent accounts it does not own "
                f"(bank_ids {foreign}); refusing the bundle"
            )
        owned = {a.hashed_account_id for a in self.accounts}
        for transfer in self.transfers:
            endpoints = {transfer.from_hashed_account_id, transfer.to_hashed_account_id}
            if not endpoints & owned:
                raise BundleError(
                    f"transfer {transfer.transfer_id} has no endpoint owned by "
                    f"{self.bank_id!r}; refusing the bundle"
                )

    def to_bytes(self) -> bytes:
        """Serialise for sealing. Sorted, so the same contribution is byte-equal."""
        payload = {
            "bank_id": self.bank_id,
            "accounts": [a.model_dump(mode="json") for a in self.accounts],
            "transfers": [t.model_dump(mode="json") for t in self.transfers],
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes, expected_bank_id: str) -> BankContribution:
        """Parse a decrypted payload and validate it.

        Args:
            raw: Plaintext produced by :meth:`to_bytes`.
            expected_bank_id: The bank id authenticated by the bundle's AEAD.
                A payload claiming a different bank is refused: the AEAD proves
                who sealed it, so a mismatch is an attempt to have the enclave
                attribute data to someone else.

        Raises:
            BundleError: on a bank-id mismatch or an invariant violation.
        """
        obj = json.loads(raw)
        claimed = obj.get("bank_id")
        if claimed != expected_bank_id:
            raise BundleError(
                f"bundle payload claims bank {claimed!r} but was sealed by "
                f"{expected_bank_id!r}; refusing the bundle"
            )
        contribution = cls(
            bank_id=expected_bank_id,
            accounts=tuple(Account.model_validate(a) for a in obj.get("accounts", [])),
            transfers=tuple(Transfer.model_validate(t) for t in obj.get("transfers", [])),
        )
        contribution.validate()
        return contribution
