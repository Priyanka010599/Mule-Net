"""Item 2: 30,000 accounts vs the enclave's 29,973 are both right, and the API says why."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from ui.build_snapshot import dataset_scope_counts
from ui.main import app


def test_scope_counts_separate_generated_active_and_enclave_scope() -> None:
    accounts = pd.DataFrame(
        {
            "hashed_account_id": ["a1", "a2", "b1", "b2", "idle"],
            "bank_id": ["A", "A", "B", "B", "A"],
        }
    )
    transfers = pd.DataFrame(
        {
            "from_hashed_account_id": ["a1", "a1", "b1"],
            "to_hashed_account_id": ["a2", "b1", "b2"],
        }
    )
    counts = dataset_scope_counts(accounts, transfers)
    assert counts["accounts_generated"] == 5
    assert counts["accounts_with_activity"] == 4  # "idle" never transacts
    assert counts["edges_total"] == 3
    assert counts["edges_in_enclave_scope"] == 1  # only a1 -> b1 crosses banks
    assert counts["accounts_in_enclave_scope"] == 2  # a1, b1; a2/b2 are same-bank only
    assert counts["scope_reason"] == "enclave_sees_cross_bank_edges_only"


def test_enclave_endpoint_names_what_its_counts_count() -> None:
    with TestClient(app) as client:
        body = client.get("/api/enclave").json()
    assert body["edges_in_enclave_scope"] == body["edge_count"]
    assert body["accounts_in_enclave_scope"] == body["account_count"]
    assert body["scope_reason"] == "enclave_sees_cross_bank_edges_only"
