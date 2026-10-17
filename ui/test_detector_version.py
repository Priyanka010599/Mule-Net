"""``detector_version`` must not depend on parquet row order.

``build_snapshot`` used to set a ring's ``detector_version`` to
``findings[0]["detector_version"]`` — the first ``account_evidence``
row for the ring, in whatever order the parquet file happened to store
it. ``account_evidence.detector_version`` is stamped per ACCOUNT
(``ml.publish.build_evidence``: the velocity rule for an account it
flagged, stage-2 identity propagation for one only pulled in by a
link), so a ring holding both kinds of member had two candidate values
and picked whichever row came first. 11 of the 24 snapshot rings are
such rings.

The tests below rebuild the same ring from shuffled evidence and
assert the payload does not move. The shuffles are seeded rather than
random so a failure is reproducible.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from ui.build_snapshot import _build_ring, _detector_version_of, _ring_evidence
from ui.config import UiConfig

CFG = UiConfig.from_env()

VELOCITY = "mulenet-velocity-rule-v2"
STAGE2 = "mulenet-identity-propagation-v1"

#: A ring id in ``ml.publish``'s format: ``f"ring-{DETECTOR_VERSION}-{n:05d}"``.
RING_ID = f"ring-{VELOCITY}-00007"

#: Two velocity members and one stage-2 member, so the ring's evidence
#: carries both detector versions — the case that used to be a coin flip.
MEMBERS: tuple[dict[str, Any], ...] = (
    {
        "hashed_account_id": "acct_a",
        "membership_basis": "velocity",
        "alertable": True,
        "link_scope": None,
        "linked_from_hashed_account_id": None,
    },
    {
        "hashed_account_id": "acct_b",
        "membership_basis": "velocity",
        "alertable": True,
        "link_scope": None,
        "linked_from_hashed_account_id": None,
    },
    {
        "hashed_account_id": "acct_c",
        "membership_basis": "identity_link",
        "alertable": True,
        "link_scope": "cross_bank",
        "linked_from_hashed_account_id": "acct_a",
        "link_evidence_type": "cross_bank_reused_id_number",
        "reason": (
            "Linked, needs review: shares an ID number with a ring "
            "member at another bank."
        ),
    },
)

BANK_OF = {"acct_a": "BANK_A", "acct_b": "BANK_B", "acct_c": "BANK_C"}


def _evidence() -> pd.DataFrame:
    """Two cross-bank identity rows, one per detector, in one finding.

    ``acct_a`` was flagged by the velocity rule; ``acct_c`` only exists
    in the ring because stage-2 propagation followed an identity link.
    Both rows collapse into the single ``(identity_reuse, cross_bank)``
    finding, which is what gives that finding two detector versions.
    """
    rows = [
        {
            "hashed_account_id": "acct_a",
            "evidence_type": "cross_bank_reused_id_number",
            "scope": "cross_bank",
            "detail": "Shares an ID number with 2 accounts at BANK_B, BANK_C.",
            "counterparty_hashed_account_ids": ["acct_b", "acct_c"],
            "counterparty_bank_ids": ["BANK_B", "BANK_C"],
            "observed_at": pd.Timestamp("2025-01-05T00:00:00Z"),
            "source_table": "kyc_documents",
            "detector_version": VELOCITY,
        },
        {
            "hashed_account_id": "acct_c",
            "evidence_type": "cross_bank_reused_photo",
            "scope": "cross_bank",
            "detail": "Shares a selfie with 1 account at BANK_A.",
            "counterparty_hashed_account_ids": ["acct_a"],
            "counterparty_bank_ids": ["BANK_A"],
            "observed_at": pd.Timestamp("2025-01-06T00:00:00Z"),
            "source_table": "kyc_documents",
            "detector_version": STAGE2,
        },
        {
            "hashed_account_id": "acct_b",
            "evidence_type": "same_bank_reused_id_number",
            "scope": "same_bank",
            "detail": "Shares an ID number with 1 account at BANK_B.",
            "counterparty_hashed_account_ids": ["acct_x"],
            "counterparty_bank_ids": ["BANK_B"],
            "observed_at": pd.Timestamp("2025-01-07T00:00:00Z"),
            "source_table": "kyc_documents",
            "detector_version": VELOCITY,
        },
    ]
    return pd.DataFrame(rows)


def _accounts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hashed_account_id": account_id,
                "bank_id": bank,
                "opened_at": pd.Timestamp("2024-06-01T00:00:00Z"),
                "account_type": "personal",
                "status": "active",
            }
            for account_id, bank in BANK_OF.items()
        ]
    )


def _transfers() -> pd.DataFrame:
    hops = (("acct_a", "acct_b"), ("acct_b", "acct_c"))
    return pd.DataFrame(
        [
            {
                "transfer_id": f"t{idx}",
                "from_hashed_account_id": src,
                "to_hashed_account_id": dst,
                "amount": Decimal("1000.00"),
                "currency": "INR",
                "timestamp": pd.Timestamp(f"2025-01-1{idx}T00:00:00Z"),
                "channel": "imps",
            }
            for idx, (src, dst) in enumerate(hops)
        ]
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hashed_account_id": account_id,
                "score": 0.9,
                "model_version": "mulenet-xgb-v1",
                "scored_at": pd.Timestamp("2025-01-30T00:00:00Z"),
                "top_factors": ["cb_fan_in"],
            }
            for account_id in BANK_OF
        ]
    )


def _ring_row() -> pd.Series:
    return pd.Series(
        {
            "ring_id": RING_ID,
            "members": [dict(m) for m in MEMBERS],
            "banks_involved": ["BANK_A", "BANK_B", "BANK_C"],
            "detected_at": pd.Timestamp("2025-01-30T23:59:58Z"),
            "status": "candidate",
        }
    )


def _explanation_row() -> pd.Series:
    return pd.Series(
        {
            "ring_id": RING_ID,
            "top_factors": ["cb_fan_in"],
            "narrative": "Three accounts across three banks.",
            "generated_at": pd.Timestamp("2025-01-30T23:59:59Z"),
        }
    )


def _build(evidence: pd.DataFrame) -> dict[str, Any]:
    built = _build_ring(
        ring_row=_ring_row(),
        explanation_row=_explanation_row(),
        accounts=_accounts(),
        transfers=_transfers(),
        evidence=evidence,
        scores=_scores(),
        truth=dict.fromkeys(BANK_OF, True),
        roles=dict.fromkeys(BANK_OF, "mule"),
        bank_of=BANK_OF,
        cases={},
        str_reports={},
        cfg=CFG,
    )
    assert built is not None
    return built


#: Every permutation of the three evidence rows. Exhaustive rather than
#: sampled: with three rows there is no reason to leave any untried.
_ORDERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def test_the_fixture_really_mixes_two_detectors() -> None:
    """Guard the guard: a single-detector fixture would prove nothing."""
    assert set(_evidence()["detector_version"]) == {VELOCITY, STAGE2}


@pytest.mark.parametrize("order", _ORDERS, ids=["".join(map(str, o)) for o in _ORDERS])
def test_ring_detector_version_survives_shuffled_evidence(order: tuple[int, ...]) -> None:
    """The ring's ``detector_version`` is the same under any row order."""
    shuffled = _evidence().iloc[list(order)].reset_index(drop=True)
    assert _build(shuffled)["detector_version"] == VELOCITY


@pytest.mark.parametrize("order", _ORDERS, ids=["".join(map(str, o)) for o in _ORDERS])
def test_whole_ring_payload_survives_shuffled_evidence(order: tuple[int, ...]) -> None:
    """Nothing else in the payload moves with row order either.

    Written against the whole payload, not just ``detector_version``:
    ``detail`` and ``source_table`` were picked from a group's first row
    the same way, and the findings list broke ties on insertion order.
    """
    shuffled = _evidence().iloc[list(order)].reset_index(drop=True)
    assert _build(shuffled) == _build(_evidence())


def test_a_finding_reports_every_detector_that_contributed() -> None:
    """Both detectors are named, not one of them chosen."""
    findings = _ring_evidence(_evidence(), set(BANK_OF), BANK_OF)
    cross_bank = next(f for f in findings if f["cross_bank"])
    assert cross_bank["detector_versions"] == sorted([STAGE2, VELOCITY])
    same_bank = next(f for f in findings if not f["cross_bank"])
    assert same_bank["detector_versions"] == [VELOCITY]


def test_ring_detector_version_is_the_detector_that_created_the_ring() -> None:
    """Read off the ``ring_id``, so it needs no evidence at all."""
    assert _detector_version_of(RING_ID) == VELOCITY
    assert _detector_version_of(f"ring-{STAGE2}-00001") == STAGE2


def test_ring_detector_version_holds_with_no_evidence_rows() -> None:
    """The old code fell back to a hardcoded string here; this derives it."""
    empty = _evidence().iloc[0:0]
    built = _build(empty)
    assert built["evidence"] == []
    assert built["detector_version"] == VELOCITY
