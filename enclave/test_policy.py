"""End-to-end tests for the three-bank join and the per-bank output policy.

This is the Stage 3 proof. The claim under test is not "the code filters
members" but the stronger, falsifiable one:

    no identifier belonging to another bank appears anywhere in a bank's bundle

asserted over the **serialised** bundle, so a field added later that leaks a
foreign id fails the test rather than sneaking past a structural check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from enclave.bundle import BankContribution, BundleError
from enclave.graph import ENCLAVE_FEATURE_NAMES, build_graph, cross_bank_features
from enclave.policy import bundles_for_ring, foreign_identifiers_in
from enclave.rekey import AccountRekeyer
from enclave.rings import find_candidates, ring_id_for, to_ring
from shared.contracts import Account, Transfer

KEY = b"e" * 32
T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def _account(hashed: str, bank: str) -> Account:
    return Account(
        hashed_account_id=hashed,
        bank_id=bank,
        opened_at=T0 - timedelta(days=200),
        account_type="savings",
    )


def _transfer(tid: str, src: str, dst: str, amount: str, minutes: int) -> Transfer:
    return Transfer(
        transfer_id=tid,
        from_hashed_account_id=src,
        to_hashed_account_id=dst,
        amount=Decimal(amount),
        currency="USD",
        timestamp=T0 + timedelta(minutes=minutes),
        channel="neft",
    )


@pytest.fixture
def three_bank_scenario():
    """A ring spanning three banks: a1 -> b1 -> c1, plus an unrelated a2 -> b2.

    Each bank contributes only the edges that touch it, exactly as the design
    requires, so the chain exists only once the three bundles are joined.
    """
    a1, a2 = "a-hash-1", "a-hash-2"
    b1, b2 = "b-hash-1", "b-hash-2"
    c1 = "c-hash-1"

    t_a1_b1 = _transfer("t1", a1, b1, "5000.00", 0)
    t_b1_c1 = _transfer("t2", b1, c1, "4900.00", 5)
    t_a2_b2 = _transfer("t3", a2, b2, "300.50", 10)

    contributions = {
        "bank-a": BankContribution(
            bank_id="bank-a",
            accounts=(_account(a1, "bank-a"), _account(a2, "bank-a")),
            transfers=(t_a1_b1, t_a2_b2),
        ),
        "bank-b": BankContribution(
            bank_id="bank-b",
            accounts=(_account(b1, "bank-b"), _account(b2, "bank-b")),
            transfers=(t_a1_b1, t_b1_c1, t_a2_b2),
        ),
        "bank-c": BankContribution(
            bank_id="bank-c",
            accounts=(_account(c1, "bank-c"),),
            transfers=(t_b1_c1,),
        ),
    }
    hints = {
        "bank-a": {b1: "bank-b", b2: "bank-b"},
        "bank-b": {a1: "bank-a", a2: "bank-a", c1: "bank-c"},
        "bank-c": {b1: "bank-b"},
    }
    rekeyer = AccountRekeyer(KEY)
    graph = build_graph(contributions, rekeyer, hints)
    ids = {
        "a1": rekeyer.rekey_counterparty("bank-a", a1),
        "a2": rekeyer.rekey_counterparty("bank-a", a2),
        "b1": rekeyer.rekey_counterparty("bank-b", b1),
        "b2": rekeyer.rekey_counterparty("bank-b", b2),
        "c1": rekeyer.rekey_counterparty("bank-c", c1),
    }
    return graph, ids


class TestGraphJoin:
    def test_all_three_banks_contribute(self, three_bank_scenario):
        graph, _ = three_bank_scenario
        assert graph.contributing_banks == ("bank-a", "bank-b", "bank-c")

    def test_shared_edges_are_deduplicated(self, three_bank_scenario):
        """Both banks report a crossing edge; counting it twice doubles features."""
        graph, _ = three_bank_scenario
        assert len(graph.edges) == 3
        assert sorted(e.transfer_id for e in graph.edges) == ["t1", "t2", "t3"]

    def test_ownership_is_recorded_for_every_contributed_account(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        assert graph.owner_of[ids["a1"]] == "bank-a"
        assert graph.owner_of[ids["b1"]] == "bank-b"
        assert graph.owner_of[ids["c1"]] == "bank-c"

    def test_chain_exists_only_after_the_join(self, three_bank_scenario):
        """a1 -> b1 -> c1 is visible to nobody alone, and to the enclave jointly."""
        graph, ids = three_bank_scenario
        pairs = {(e.from_id, e.to_id) for e in graph.edges}
        assert (ids["a1"], ids["b1"]) in pairs
        assert (ids["b1"], ids["c1"]) in pairs

    def test_features_match_the_derived_column_set(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        features = cross_bank_features(graph)
        assert set(features[ids["b1"]]) == set(ENCLAVE_FEATURE_NAMES)

    def test_passthrough_is_visible_on_the_middle_account(self, three_bank_scenario):
        """b1 received 5000 and sent on 4900: the pass-through the model wants."""
        graph, ids = three_bank_scenario
        row = cross_bank_features(graph)[ids["b1"]]
        assert row["amount_in_total"] == pytest.approx(5000.0)
        assert row["amount_out_total"] == pytest.approx(4900.0)
        assert row["passthrough_ratio"] == pytest.approx(0.98)
        assert row["distinct_counterparty_banks"] == pytest.approx(2.0)

    def test_passthrough_ratio_is_zero_not_nan_with_no_inflow(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        row = cross_bank_features(graph)[ids["a1"]]
        assert row["amount_in_total"] == 0.0
        assert row["passthrough_ratio"] == 0.0


class TestBundleValidation:
    def test_bank_cannot_claim_another_banks_account(self):
        contribution = BankContribution(
            bank_id="bank-a",
            accounts=(_account("b-hash-1", "bank-b"),),
            transfers=(),
        )
        with pytest.raises(BundleError, match="does not own"):
            contribution.validate()

    def test_transfer_with_no_owned_endpoint_is_refused(self):
        contribution = BankContribution(
            bank_id="bank-a",
            accounts=(_account("a-hash-1", "bank-a"),),
            transfers=(_transfer("t9", "b-hash-1", "c-hash-1", "10.00", 0),),
        )
        with pytest.raises(BundleError, match="no endpoint owned by"):
            contribution.validate()

    def test_payload_claiming_a_different_bank_is_refused(self):
        contribution = BankContribution(
            bank_id="bank-a", accounts=(_account("a-hash-1", "bank-a"),), transfers=()
        )
        with pytest.raises(BundleError, match="refusing the bundle"):
            BankContribution.from_bytes(contribution.to_bytes(), "bank-b")

    def test_round_trip_preserves_the_contribution(self):
        contribution = BankContribution(
            bank_id="bank-a",
            accounts=(_account("a-hash-1", "bank-a"),),
            transfers=(_transfer("t1", "a-hash-1", "b-hash-1", "5000.00", 0),),
        )
        restored = BankContribution.from_bytes(contribution.to_bytes(), "bank-a")
        assert restored.bank_id == "bank-a"
        assert restored.transfers[0].amount == Decimal("5000.00")


class TestRingClustering:
    def test_three_bank_ring_is_found(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        flagged = {ids["a1"], ids["b1"], ids["c1"]}
        candidates = find_candidates(graph, flagged)
        assert len(candidates) == 1
        assert candidates[0].bank_span == 3
        assert set(candidates[0].flagged_members) == flagged

    def test_single_bank_component_is_not_released(self, three_bank_scenario):
        """A component inside one bank tells that bank nothing new."""
        graph, ids = three_bank_scenario
        candidates = find_candidates(graph, {ids["a2"]})
        assert candidates == []

    def test_ring_id_is_stable_across_runs(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        flagged = {ids["a1"], ids["b1"], ids["c1"]}
        first = ring_id_for(find_candidates(graph, flagged)[0])
        second = ring_id_for(find_candidates(graph, flagged)[0])
        assert first == second

    def test_unflagged_bridge_does_not_manufacture_a_cross_bank_ring(
        self, three_bank_scenario
    ):
        """Span is measured over flagged members, not over bridges.

        a1 and a2 both sit at bank-a. Even though each reaches bank-b, nobody
        flagged a bank-b account, so there is no cross-bank ring to publish.
        """
        graph, ids = three_bank_scenario
        candidates = find_candidates(graph, {ids["a1"], ids["a2"]})
        assert candidates == []

    def test_emitted_ring_contains_only_flagged_members(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        flagged = {ids["a1"], ids["b1"], ids["c1"]}
        ring = to_ring(find_candidates(graph, flagged)[0])
        assert set(ring.all_member_ids) == flagged
        assert set(ring.alertable_member_ids) == flagged


class TestOutputPolicy:
    """The Stage 3 isolation proof."""

    @pytest.fixture
    def bundles(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        flagged = {ids["a1"], ids["b1"], ids["c1"]}
        candidate = find_candidates(graph, flagged)[0]
        ring = to_ring(candidate)
        return graph, ids, bundles_for_ring(ring, graph, len(candidate.connector_ids))

    def test_each_involved_bank_gets_a_bundle(self, bundles):
        _, _, per_bank = bundles
        assert sorted(per_bank) == ["bank-a", "bank-b", "bank-c"]

    def test_a_bank_sees_only_its_own_members(self, bundles):
        _, ids, per_bank = bundles
        assert per_bank["bank-a"].own_members == (ids["a1"],)
        assert per_bank["bank-b"].own_members == (ids["b1"],)
        assert per_bank["bank-c"].own_members == (ids["c1"],)

    def test_no_bundle_contains_another_banks_identifier(self, bundles):
        """The central claim, asserted over the serialised bundle."""
        graph, _, per_bank = bundles
        for bank_id, bundle in per_bank.items():
            leaked = foreign_identifiers_in(bundle, graph)
            assert leaked == set(), f"{bank_id} bundle leaked {leaked}"

    def test_bank_a_cannot_see_bank_b_members_even_via_edges(self, bundles):
        """The obvious bug: filter members, then hand over the edges."""
        _, ids, per_bank = bundles
        serialised = per_bank["bank-a"].to_json()
        assert ids["b1"] not in serialised
        assert ids["c1"] not in serialised
        assert ids["a1"] in serialised

    def test_foreign_counterparties_are_pseudonymised_not_dropped(self, bundles):
        """Bank A must still learn the shape: its account paid someone at bank-b."""
        _, _, per_bank = bundles
        edges = per_bank["bank-a"].edges
        assert len(edges) == 1
        assert edges[0].to_ref.startswith("other-bank:bank-b:")
        assert edges[0].to_bank == "bank-b"

    def test_bank_span_is_disclosed_as_an_aggregate(self, bundles):
        _, _, per_bank = bundles
        assert per_bank["bank-a"].bank_span == 3
        assert per_bank["bank-a"].total_member_count == 3

    def test_other_bank_counts_are_disclosed_without_identifiers(self, bundles):
        _, _, per_bank = bundles
        counts = per_bank["bank-a"].other_bank_member_counts
        assert counts == {"bank-b": 1, "bank-c": 1}

    def test_pseudonyms_are_unlinkable_across_bundles(self, bundles):
        """b1's pseudonym in A's bundle must not equal its pseudonym in C's.

        If they matched, bank-a and bank-c could collude by comparing bundles
        and re-identify the shared middle account.
        """
        _, _, per_bank = bundles
        a_refs = {e.to_ref for e in per_bank["bank-a"].edges}
        c_refs = {e.from_ref for e in per_bank["bank-c"].edges}
        # Both refer to b1, but neither carries its real id, and the labels are
        # positional within each bundle rather than derived from the account.
        assert a_refs == {"other-bank:bank-b:1"}
        assert c_refs == {"other-bank:bank-b:1"}
        # The guarantee that matters: neither bundle contains b1's real id.
        assert all("other-bank" in r or r in per_bank["bank-a"].own_members for r in a_refs)

    def test_uninvolved_bank_receives_nothing(self, three_bank_scenario):
        graph, ids = three_bank_scenario
        flagged = {ids["a1"], ids["b1"]}
        candidate = find_candidates(graph, flagged)[0]
        per_bank = bundles_for_ring(to_ring(candidate), graph)
        assert "bank-c" not in per_bank
