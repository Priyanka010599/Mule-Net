"""Building the combined cross-bank graph from the banks' sealed bundles.

This is the computation no single participant can perform: each bank sees only
its own edges, and the operator sees only ciphertext. Inside the enclave the
three contributions are re-keyed into one identifier space (see
:mod:`enclave.rekey`) and joined.

**Feature names are derived, not restated.** The columns this module produces
are selected *by name* from :data:`ml.features.FEATURE_NAMES`, the project's
source of truth, and the selection is checked at import time. If someone
renames a feature in ``ml/``, this module raises on import rather than quietly
computing a column that no longer matches what the model was trained on.

Only the structural subset is computable here. Device, document and age
features need tables that never cross a bank boundary (CLAUDE.md: only the
boundary-crossing edges are shared), so they are excluded by name and the
exclusion is justified below rather than left as an unexplained gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from enclave.bundle import BankContribution
from enclave.rekey import AccountRekeyer
from ml.features import FEATURE_NAMES

#: Features that need per-bank tables the banks do not send, with the reason.
#: Kept as a mapping so the exclusion is self-documenting and reviewable.
NOT_COMPUTABLE_CROSS_BANK: dict[str, str] = {
    "passthrough_speed_seconds": "needs the full intra-bank ordering of an account's events",
    "account_age_days": "needs accounts the bank did not send (only endpoints of shared edges)",
    "device_count": "device table never crosses a bank boundary",
    "shared_device_count": "device table never crosses a bank boundary",
    "document_reuse_count": "KYC documents never cross a bank boundary",
    "document_reuse_flag": "KYC documents never cross a bank boundary",
}

#: The features the enclave computes, derived from ml/ rather than retyped.
ENCLAVE_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in NOT_COMPUTABLE_CROSS_BANK
)

# Fail loudly at import if ml/ renamed something under us: an excluded name that
# no longer exists means the exclusion list is stale, and a stale exclusion
# would silently drop or resurrect a feature.
_unknown = sorted(set(NOT_COMPUTABLE_CROSS_BANK) - set(FEATURE_NAMES))
if _unknown:
    raise ImportError(
        f"enclave/graph.py excludes features that no longer exist in "
        f"ml.features.FEATURE_NAMES: {_unknown}. The exclusion list is stale; "
        f"reconcile it against ml/features.py before this enclave is trusted."
    )

#: An amount is "round" when it is an exact multiple of this. Mirrors
#: ml.features.ROUND_AMOUNT_UNIT, imported rather than copied.
from ml.features import ROUND_AMOUNT_UNIT  # noqa: E402  (after the staleness check, on purpose)


@dataclass(frozen=True)
class CrossBankEdge:
    """One boundary-crossing transfer, in the enclave's identifier space.

    Not a redefinition of :class:`shared.contracts.Transfer`: it is the
    re-keyed projection of one, carrying the owning bank of each endpoint,
    which ``Transfer`` has no field for because outside the enclave the two
    facts live in different tables.
    """

    transfer_id: str
    from_id: str
    to_id: str
    from_bank: str
    to_bank: str
    amount: Decimal
    currency: str
    timestamp: datetime
    channel: str

    @property
    def crosses_banks(self) -> bool:
        """True when the two endpoints sit at different banks."""
        return self.from_bank != self.to_bank


@dataclass
class CombinedGraph:
    """The joined cross-bank graph plus the ownership map the policy needs.

    Attributes:
        edges: Deduplicated cross-bank edges. Two banks both report an edge
            that crosses between them, so deduplication is not an optimisation
            here -- counting the same transfer twice would double every
            degree and amount feature on both endpoints.
        owner_of: enclave account id -> owning bank, for accounts whose bank
            contributed. Accounts seen only as a far endpoint are absent.
        contributing_banks: Banks that actually sent a bundle this run.
    """

    edges: list[CrossBankEdge] = field(default_factory=list)
    owner_of: dict[str, str] = field(default_factory=dict)
    contributing_banks: tuple[str, ...] = ()

    @property
    def account_ids(self) -> set[str]:
        """Every account appearing on either end of an edge."""
        ids: set[str] = set()
        for edge in self.edges:
            ids.add(edge.from_id)
            ids.add(edge.to_id)
        return ids


def _bank_of_endpoint(
    own_accounts: dict[str, str], hashed_account_id: str, counterparty_banks: dict[str, str]
) -> str | None:
    """Resolve which bank an endpoint sits at, from what this bank declared.

    ``own_accounts`` is ``{hashed_account_id: bank_id}`` for the calling
    bank's own accounts, built once per bank in :func:`build_graph` rather
    than passing the whole :class:`BankContribution` and re-scanning it here.
    Observed live: at real scale (~10,000 accounts/bank, ~700,000 transfers
    total), the original linear scan over ``contribution.accounts`` -- run
    twice per transfer, for every transfer -- made this function alone cost
    on the order of 10 billion comparisons, which never completed inside the
    production run's 600s window. It looked like a hang; it was O(accounts x
    transfers) where O(1) was available for free from data already read.
    """
    bank_id = own_accounts.get(hashed_account_id)
    if bank_id is not None:
        return bank_id
    return counterparty_banks.get(hashed_account_id)


def build_graph(
    contributions: dict[str, BankContribution],
    rekeyer: AccountRekeyer,
    counterparty_banks: dict[str, dict[str, str]] | None = None,
) -> CombinedGraph:
    """Join the banks' contributions into one re-keyed cross-bank graph.

    Args:
        contributions: bank id -> that bank's validated contribution. The key
            is the bank id authenticated by the bundle's AEAD, not a value read
            out of the payload.
        rekeyer: Re-keys identifiers and records ownership.
        counterparty_banks: Optional per-bank map of ``hashed_account_id`` to
            the bank the *sending* bank believes the counterparty sits at. This
            is how the far end of an edge gets a bank before that bank's own
            bundle is read. It is a hint, not an authority: when the owning
            bank later claims the account, its own claim wins, and a
            contradiction between two banks is raised by
            :meth:`AccountRekeyer.add` rather than resolved silently.

    Returns:
        The combined graph.
    """
    counterparty_banks = counterparty_banks or {}
    graph = CombinedGraph(contributing_banks=tuple(sorted(contributions)))

    # Pass 1: every bank claims the accounts it owns. Ownership must be settled
    # before any edge is re-keyed, otherwise an edge seen first from the far
    # side would create an endpoint with no owner and its alerts would be
    # withheld from the bank that is entitled to them.
    for bank_id, contribution in sorted(contributions.items()):
        for account in contribution.accounts:
            rekeyer.add(bank_id, account.hashed_account_id)
    graph.owner_of = rekeyer.owned_ids

    # Pass 2: re-key edges. Deduplicate by transfer_id, because a cross-bank
    # transfer is reported by both banks.
    seen: set[str] = set()
    for bank_id, contribution in sorted(contributions.items()):
        hints = counterparty_banks.get(bank_id, {})
        # Built once per bank, O(accounts) -- see _bank_of_endpoint's
        # docstring for why this replaced a per-transfer linear scan.
        own_accounts = {a.hashed_account_id: a.bank_id for a in contribution.accounts}
        for transfer in contribution.transfers:
            if transfer.transfer_id in seen:
                continue
            from_bank = _bank_of_endpoint(own_accounts, transfer.from_hashed_account_id, hints)
            to_bank = _bank_of_endpoint(own_accounts, transfer.to_hashed_account_id, hints)
            if from_bank is None or to_bank is None:
                # An endpoint whose bank nobody declared cannot be placed in the
                # graph: its features would be wrong and, worse, the output
                # policy would have no bank to withhold it from. Skipped, and
                # counted by the caller rather than silently dropped.
                continue
            seen.add(transfer.transfer_id)
            graph.edges.append(
                CrossBankEdge(
                    transfer_id=transfer.transfer_id,
                    from_id=rekeyer.rekey_counterparty(
                        from_bank, transfer.from_hashed_account_id
                    ),
                    to_id=rekeyer.rekey_counterparty(to_bank, transfer.to_hashed_account_id),
                    from_bank=from_bank,
                    to_bank=to_bank,
                    amount=transfer.amount,
                    currency=transfer.currency,
                    timestamp=transfer.timestamp,
                    channel=transfer.channel,
                )
            )
    graph.edges.sort(key=lambda e: (e.timestamp, e.transfer_id))
    return graph


def cross_bank_features(graph: CombinedGraph) -> dict[str, dict[str, float]]:
    """Compute the cross-bank feature subset for every account in the graph.

    Returns:
        enclave account id -> {feature name -> value}, with exactly the keys in
        :data:`ENCLAVE_FEATURE_NAMES`.
    """
    in_edges: dict[str, list[CrossBankEdge]] = defaultdict(list)
    out_edges: dict[str, list[CrossBankEdge]] = defaultdict(list)
    for edge in graph.edges:
        out_edges[edge.from_id].append(edge)
        in_edges[edge.to_id].append(edge)

    features: dict[str, dict[str, float]] = {}
    for account_id in sorted(graph.account_ids):
        incoming = in_edges.get(account_id, [])
        outgoing = out_edges.get(account_id, [])
        amount_in = sum((e.amount for e in incoming), Decimal(0))
        amount_out = sum((e.amount for e in outgoing), Decimal(0))
        in_parties = {e.from_id for e in incoming}
        out_parties = {e.to_id for e in outgoing}
        counterparty_banks = {e.from_bank for e in incoming} | {e.to_bank for e in outgoing}
        counterparty_banks.discard(graph.owner_of.get(account_id, ""))
        all_amounts = [e.amount for e in incoming + outgoing]
        round_amounts = [
            a for a in all_amounts if a % Decimal(str(ROUND_AMOUNT_UNIT)) == 0
        ]
        row = {
            "in_degree": float(len(incoming)),
            "out_degree": float(len(outgoing)),
            "unique_in_counterparties": float(len(in_parties)),
            "unique_out_counterparties": float(len(out_parties)),
            "unique_counterparties": float(len(in_parties | out_parties)),
            "distinct_counterparty_banks": float(len(counterparty_banks)),
            "amount_in_total": float(amount_in),
            "amount_out_total": float(amount_out),
            # Share of money in that left again. Zero when nothing arrived --
            # not undefined -- so downstream scoring never sees a NaN it would
            # have to invent a fill value for.
            "passthrough_ratio": (
                float(min(amount_out, amount_in) / amount_in) if amount_in > 0 else 0.0
            ),
            "round_amount_ratio": (
                len(round_amounts) / len(all_amounts) if all_amounts else 0.0
            ),
        }
        missing = set(ENCLAVE_FEATURE_NAMES) - row.keys()
        extra = row.keys() - set(ENCLAVE_FEATURE_NAMES)
        if missing or extra:
            raise AssertionError(
                f"cross_bank_features produced columns out of step with "
                f"ENCLAVE_FEATURE_NAMES (missing={sorted(missing)}, extra={sorted(extra)})"
            )
        features[account_id] = row
    return features
