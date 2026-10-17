"""The dataset block says why 30,000 accounts becomes 29,973 in the joined graph.

``accounts`` counts every generated account; the joined graph (and so the live
proof's evidence) holds only accounts on a cross-bank transfer. The two
numbers are different on purpose and the snapshot now carries both, so the page
can explain the gap instead of leaving two figures that look like a mistake.
"""

from __future__ import annotations

import pandas as pd

from ui.build_snapshot import _activity_counts
from ui.models import DatasetInfo


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = pd.DataFrame(
        {
            "hashed_account_id": ["a1", "a2", "b1", "b2", "c1"],
            "bank_id": ["A", "A", "B", "B", "C"],
        }
    )
    transfers = pd.DataFrame(
        {
            "from_hashed_account_id": ["a1", "a1", "b1", "a2"],
            "to_hashed_account_id": ["b1", "a2", "b2", "a1"],
        }
    )
    return accounts, transfers


def test_counts_separate_generated_from_cross_bank() -> None:
    counts = _activity_counts(*_frames())
    # a1 -> b1 is the only boundary-crossing transfer.
    assert counts == {
        "accounts_generated": 5,
        "accounts_with_cross_bank_activity": 2,
        "cross_bank_transfers": 1,
    }


def test_dataset_info_accepts_old_and_new_snapshots() -> None:
    base = {
        "accounts": 5,
        "transfers": 4,
        "detected_rings_total": 1,
        "rings_in_snapshot": 1,
        "banks": ["A"],
    }
    old = DatasetInfo(**base)
    assert old.accounts_with_cross_bank_activity is None
    new = DatasetInfo(**base, **_activity_counts(*_frames()))
    assert new.accounts_generated == 5
