"""Tests for the RingMember and AccountExplanation migration.

Two rules landed with the ``RingMember`` contract, and both exist
because a sentence that is *true* can still be read as an accusation:

* An ``identity_link`` member is a lead, not a finding, and must never
  be called a mule -- not even inside a negation, because a truncated
  sentence keeps the noun and drops the "not".
* An unqualified member count is read as a count of alerts, so it must
  equal the alertable population. The all-members figure is publishable
  only when the sentence says what the extra members are.

These tests are adversarial: each asserts the verifier *rejects* the
tempting wording and accepts the careful one.
"""

from __future__ import annotations

import pytest

from agents import mock_agents, tools
from agents.data import DataStore, default_store
from agents.eval.questions import ground_truth_summary
from agents.verify import Claim, InvestigationResult, check_prose, verify
from shared.contracts import (
    AccountExplanation,
    EvidenceScope,
    ExplanationFactor,
    MembershipBasis,
    Ring,
    RingMember,
)


@pytest.fixture(scope="module")
def store() -> DataStore:
    return default_store()


@pytest.fixture(scope="module")
def ring_id(store: DataStore) -> str:
    return str(store.table("rings")["ring_id"].iloc[0])


@pytest.fixture(scope="module")
def mixed_ring_id(store: DataStore) -> str:
    """A real ring carrying both an identity link and a context member."""
    for candidate in store.table("rings")["ring_id"]:
        ring, _, _ = tools.get_ring_subgraph(str(candidate), max_hops=1)
        if ring.context_member_ids and ring.identity_link_member_ids:
            return str(candidate)
    pytest.skip("no published ring carries both link and context members")


def _ring_with_context() -> Ring:
    """A synthetic ring: one velocity, one cross-bank link, one context."""
    return Ring(
        ring_id="ring-test-0001",
        members=[
            RingMember(
                hashed_account_id="a" * 64,
                membership_basis=MembershipBasis.VELOCITY,
                alertable=True,
            ),
            RingMember(
                hashed_account_id="b" * 64,
                membership_basis=MembershipBasis.IDENTITY_LINK,
                alertable=True,
                link_scope=EvidenceScope.CROSS_BANK,
                linked_from_hashed_account_id="a" * 64,
                link_evidence_type="cross_bank_shared_device",
                reason="Linked, needs review: shares a device with a member at BANK_B.",
            ),
            RingMember(
                hashed_account_id="c" * 64,
                membership_basis=MembershipBasis.IDENTITY_LINK,
                alertable=False,
                link_scope=EvidenceScope.SINGLE_BANK,
                linked_from_hashed_account_id="a" * 64,
                link_evidence_type="same_bank_shared_device",
                reason="Context only, not an alert: shares a device at the same bank.",
            ),
        ],
        banks_involved=["BANK_A", "BANK_B"],
    )


def test_ring_populations_are_distinct() -> None:
    ring = _ring_with_context()
    assert len(ring.all_member_ids) == 3
    assert len(ring.alertable_member_ids) == 2
    assert len(ring.context_member_ids) == 1
    assert len(ring.velocity_member_ids) == 1
    assert len(ring.identity_link_member_ids) == 2


# ---------------------------------------------------- wording: identity link


@pytest.mark.parametrize(
    "sentence",
    [
        "This account is linked to the ring and is a mule.",
        "Linked, needs review: the account is a suspect in the network.",
        "The account shares a device with a launderer and needs review.",
        "Linked to the ring, this account is one of the fraudsters.",
        # The hardened form. This sentence is *correct* read in full, and
        # becomes the accusation it denies the moment a UI truncates it.
        "Linked, needs review: this is not evidence that this account is a mule.",
    ],
)
def test_identity_link_sentence_may_not_use_accusation_words(sentence: str) -> None:
    assert check_prose(sentence, []), f"expected rejection of {sentence!r}"


def test_correct_identity_link_wording_passes() -> None:
    assert not check_prose(
        "Linked, needs review: shares a device fingerprint with a ring member "
        "at another bank. This is a lead from identity reuse, not a detection.",
        [],
    )


def test_product_name_is_not_the_word_mule() -> None:
    """Every ring id contains 'mulenet'; that must not trip the rule.

    A substring match would reject every narrative that names the ring
    it is about -- which is all of them.
    """
    assert not check_prose(
        "Ring ring-mulenet-velocity-rule-v2-00002 has linked members needing review.",
        [],
    )


def test_rule_is_scoped_to_identity_link_sentences() -> None:
    """A velocity finding may use the word; it is a detection."""
    assert not check_prose(
        "The velocity rule fired on this account's own pass-through behaviour, "
        "the classic mule signature.",
        [],
    )


def test_published_reasons_contain_no_accusation(store: DataStore) -> None:
    """The bar the narratives are held to is the bar ml/ already meets."""
    for row in store.table("rings").itertuples():
        for member in list(row.members):
            reason = dict(member).get("reason")
            if reason:
                assert "mule" not in reason.lower()


# --------------------------------------------------------- counting: alertable


def test_unqualified_all_members_count_is_rejected() -> None:
    """3 members but 2 alertable: a bare "3 accounts" over-accuses by one."""
    problems = check_prose("The ring has 3 accounts.", [], _ring_with_context())
    assert problems
    assert "only 2 are alertable" in problems[0]


def test_unqualified_alertable_count_passes() -> None:
    assert not check_prose("The ring has 2 accounts.", [], _ring_with_context())


def test_qualified_all_members_count_passes() -> None:
    """The total is publishable when the sentence names the extras."""
    assert not check_prose(
        "The ring has 2 alertable accounts, plus 1 same-bank context member "
        "linked for review.",
        [],
        _ring_with_context(),
    )


def test_counting_rule_is_inert_when_all_members_are_alertable() -> None:
    clean = Ring(
        ring_id="ring-test-0002",
        members=[
            RingMember(
                hashed_account_id=char * 64,
                membership_basis=MembershipBasis.VELOCITY,
                alertable=True,
            )
            for char in "de"
        ],
        banks_involved=["BANK_A"],
    )
    assert not check_prose("The ring has 2 accounts.", [], clean)


# ------------------------------------------- counting: fails closed, never skipped


def test_member_count_with_no_ring_subject_fails_closed() -> None:
    """An uncheckable count is rejected, not waved through.

    Skipping the rule when no ring resolves makes a pass unreadable:
    you cannot tell a count that was checked from one that never was.
    """
    problems = check_prose("The ring has 34 accounts.", [], None)
    assert problems
    assert "unchecked, not wrong" in problems[0]


def test_narrative_without_a_member_count_still_passes_with_no_ring() -> None:
    """The inverse: the rule must not fire on prose that counts nothing."""
    assert not check_prose(
        "Flagged on pass-through velocity and the shape of the transfer graph.",
        [],
        None,
    )


def test_ring_id_digits_are_not_read_as_a_member_count() -> None:
    """Every ring id ends in digits; they are not a count of anything."""
    assert not check_prose(
        "Ring ring-mulenet-velocity-rule-v2-00002 has linked members needing review.",
        [],
        None,
    )


def test_verify_rejects_a_counted_narrative_carrying_only_account_claims(
    ring_id: str,
) -> None:
    """End to end: a valid account claim does not license an unchecked count."""
    ring, accounts, _ = tools.get_ring_subgraph(ring_id, max_hops=1)
    member = next(
        account
        for account in accounts
        if account.hashed_account_id in set(ring.all_member_ids)
    )
    result = InvestigationResult(
        narrative=f"The ring has {len(ring.all_member_ids)} accounts.",
        claims=[
            Claim("c1", "account_bank", member.hashed_account_id, member.bank_id, "")
        ],
    )
    verdict = verify(result)
    assert not verdict.passed
    assert any("no ring subject could be resolved" in item for item in verdict.unsupported)


def test_alertable_claim_kind_verifies(ring_id: str) -> None:
    ring, _, _ = tools.get_ring_subgraph(ring_id, max_hops=1)
    good = InvestigationResult(
        narrative="Flagged on velocity and structure.",
        claims=[
            Claim(
                "c1",
                "ring_alertable_member_count",
                ring_id,
                len(ring.alertable_member_ids),
                "",
            )
        ],
    )
    assert verify(good).passed


def test_alertable_claim_kind_catches_a_wrong_count(ring_id: str) -> None:
    bad = InvestigationResult(
        narrative="Flagged on velocity and structure.",
        claims=[Claim("c1", "ring_alertable_member_count", ring_id, 9999, "")],
    )
    verdict = verify(bad)
    assert not verdict.passed
    assert "alertable" in verdict.unsupported[0]


# ------------------------------------------------- membership basis rendering


def test_investigation_describes_every_membership_basis(mixed_ring_id: str) -> None:
    """Each member is described by how it came to be in the ring."""
    result = mock_agents.investigate(mixed_ring_id)
    joined = "\n".join(result.evidence)
    assert "[detected]" in joined
    assert "[linked, needs review]" in joined
    assert "[context only]" in joined
    assert verify(result).passed


def test_narrative_count_is_the_alertable_count(mixed_ring_id: str) -> None:
    ring, _, _ = tools.get_ring_subgraph(mixed_ring_id, max_hops=1)
    result = mock_agents.investigate(mixed_ring_id)
    assert f"{len(ring.alertable_member_ids)} alertable accounts" in result.narrative
    assert str(len(ring.context_member_ids)) in result.narrative


def test_qa_unqualified_ring_count_is_alertable(mixed_ring_id: str) -> None:
    ring, _, _ = tools.get_ring_subgraph(mixed_ring_id, max_hops=1)
    unqualified = mock_agents.answer_question(
        f"How many accounts are in {mixed_ring_id}?"
    )
    assert unqualified.answer == str(len(ring.alertable_member_ids))

    total = mock_agents.answer_question(
        f"How many accounts are in {mixed_ring_id} in total, including context?"
    )
    assert total.answer == str(len(ring.all_member_ids))


def test_subgraph_expansion_includes_context_members(mixed_ring_id: str) -> None:
    """Context members' counterparties are still worth showing an analyst."""
    ring, accounts, _ = tools.get_ring_subgraph(mixed_ring_id, max_hops=1)
    returned = {account.hashed_account_id for account in accounts}
    assert set(ring.all_member_ids) <= returned


# ------------------------------------------------------- account explanations


def test_account_explanation_is_returned(store: DataStore) -> None:
    """AccountExplanation replaced the deleted GnnExplanation."""
    account = str(store.table("account_explanations")["hashed_account_id"].iloc[0])
    explanation = tools.get_explanation(account, "account")
    assert isinstance(explanation, AccountExplanation)
    assert explanation.narrative
    assert explanation.model_version
    assert all(isinstance(f, ExplanationFactor) for f in explanation.top_factors)


def test_explanation_factor_contribution_is_signed(store: DataStore) -> None:
    """The sign is load-bearing: drop it and a mitigator reads as damning."""
    signs = set()
    for row in store.table("account_explanations").head(200).itertuples():
        for factor in list(row.top_factors):
            signs.add(dict(factor)["contribution"] < 0)
    assert signs == {True, False}, "expected both positive and negative factors"


def test_unknown_account_explanation_raises() -> None:
    with pytest.raises(KeyError):
        tools.get_explanation("0" * 64, "account")


# ---------------------------------------------------------------- eval labels


def test_eval_uses_the_two_named_labels() -> None:
    summary = ground_truth_summary()
    assert summary["passthrough_mules"] == 712
    assert summary["complicit"] == 899
    assert "positives" not in summary, "a bare 'positives' conflates the two"


def test_source_accounts_are_complicit_not_passthrough() -> None:
    """The 187 sources are the whole difference between the two labels."""
    summary = ground_truth_summary()
    difference = summary["complicit"] - summary["passthrough_mules"]
    assert difference == summary["roles"]["source"]
