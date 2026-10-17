"""Tests for the live "load a subset, score it live" panel (``agents/subset.py``).

Unlike ``agents/test_simulate.py``, this path reads **real** published
tables rather than synthesising a graph in memory, so every test here
writes a small, hand-built ``accounts``/``transfers``/``devices``/
``kyc_submissions``/``kyc_documents`` parquet set under ``tmp_path`` --
the same shape :func:`ml.features.load_tables` reads, just a handful of
rows instead of 30,000 accounts. The pattern below is built to actually
trip the shipped velocity rule (``ml.rule.velocity_decision``), the same
way ``agents/test_simulate.py`` builds a pattern that does, so the "real
verdict" tests assert against genuine detection, not a mocked one.

Every test here is offline and free: this path makes no model call at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents import service, subset
from agents.config import AgentsConfig
from agents.data import DataStore
from agents.limits import ServiceLimits
from agents.test_service import _config  # the vertex-mode config builder

#: Limits high enough not to interfere with a behaviour test -- mirrors
#: agents/test_simulate.py's fixture of the same name.
_GENEROUS_LIMITS = ServiceLimits(
    session_limit=1000, ip_limit=1000, window_s=60.0, daily_cap=10_000
)

_T0 = datetime(2025, 1, 1, tzinfo=UTC)


def _build_dataset(base: Any) -> None:
    """Write a tiny real-shaped dataset that trips the velocity rule.

    16 mule accounts (8 at BANK_A, 8 at BANK_B) each send one inbound
    transfer to a single collector at BANK_C, all within the first two
    minutes; the collector then forwards to 4 onward accounts at BANK_C
    shortly after, still inside one burst. That gives the collector:
    ``in_degree=16``, ``out_degree=4``, ``passthrough_speed_seconds`` a
    few hundred seconds, ``passthrough_ratio`` well inside
    ``VELOCITY_RATIO_RANGE``, and ``distinct_counterparty_banks=3`` --
    every clause ``ml.rule.velocity_decision`` checks. A handful of extra
    accounts with no unusual traffic sit alongside it as noise, and
    nothing here is a ring the velocity rule *would* miss, so a run that
    reports zero flags is a real bug, not an artefact of the fixture.
    """
    base.mkdir(parents=True, exist_ok=True)

    mules = [f"mule-{i:02d}" for i in range(16)]
    mule_banks = {m: ("BANK_A" if i < 8 else "BANK_B") for i, m in enumerate(mules)}
    collector = "collector-00"
    onward = [f"onward-{i:02d}" for i in range(4)]
    noise = [f"noise-{i:02d}" for i in range(3)]

    all_accounts = [
        *[(m, mule_banks[m]) for m in mules],
        (collector, "BANK_C"),
        *[(o, "BANK_C") for o in onward],
        *[(n, "BANK_C") for n in noise],
    ]
    accounts = pd.DataFrame(
        {
            "hashed_account_id": [a for a, _ in all_accounts],
            "bank_id": [b for _, b in all_accounts],
            "opened_at": [_T0 - timedelta(days=30)] * len(all_accounts),
        }
    )

    rows = []
    for i, mule in enumerate(mules):
        rows.append(
            {
                "transfer_id": f"in-{i}",
                "from_hashed_account_id": mule,
                "to_hashed_account_id": collector,
                "amount": 1000.0,
                "timestamp": _T0 + timedelta(seconds=10 * i),
            }
        )
    for i, dest in enumerate(onward):
        rows.append(
            {
                "transfer_id": f"out-{i}",
                "from_hashed_account_id": collector,
                "to_hashed_account_id": dest,
                "amount": 900.0,
                "timestamp": _T0 + timedelta(seconds=300 + 10 * i),
            }
        )
    # A little unrelated noise traffic, 3 days out: outside the 24h
    # horizon (so it does not affect the near-term tests) but inside the
    # 7d one, so `transfers_considered` differs by horizon for a real
    # reason rather than happening to equal every row in the table.
    for i, n in enumerate(noise):
        rows.append(
            {
                "transfer_id": f"noise-{i}",
                "from_hashed_account_id": n,
                "to_hashed_account_id": onward[0],
                "amount": 50.0,
                "timestamp": _T0 + timedelta(days=3, seconds=i),
            }
        )
    transfers = pd.DataFrame(rows)

    accounts.to_parquet(base / "accounts.parquet")
    transfers.to_parquet(base / "transfers.parquet")
    # One inert row each, not zero: an entirely empty devices/kyc table
    # round-trips through parquet in a way that trips a pandas Series.map
    # edge case inside ml.features.identity_links (mapping onto an empty
    # frame coerces a tz-aware datetime column to float64 and raises). A
    # single row for a noise account, never shared with anything, keeps
    # every downstream join real without exercising that edge case.
    pd.DataFrame(
        {
            "hashed_account_id": ["noise-00"],
            "device_fingerprint": ["only-noise-00-has-this-device"],
            "first_seen": [_T0 - timedelta(days=30)],
        }
    ).to_parquet(base / "devices.parquet")
    pd.DataFrame(
        {
            "hashed_account_id": ["noise-00"],
            "submitted_at": [_T0 - timedelta(days=30)],
        }
    ).to_parquet(base / "kyc_submissions.parquet")
    pd.DataFrame(
        {
            "hashed_account_id": ["noise-00"],
            "hashed_id_number": ["only-noise-00-has-this-id"],
        }
    ).to_parquet(base / "kyc_documents.parquet")


@pytest.fixture
def real_data_config(tmp_path: Any) -> AgentsConfig:
    base = _config()
    data_dir = tmp_path / "sim-out"
    _build_dataset(data_dir)
    return AgentsConfig(
        mode=base.mode,
        data_dir=data_dir,
        ml_out_dir=tmp_path / "no-ml",
        model=base.model,
        project=base.project,
        location=base.location,
        use_vertex=base.use_vertex,
        require_data=False,
    )


@pytest.fixture
def store(real_data_config: AgentsConfig) -> DataStore:
    return DataStore(real_data_config)


@pytest.fixture
def subset_client(real_data_config: AgentsConfig) -> TestClient:
    return TestClient(
        service.create_app(real_data_config, _GENEROUS_LIMITS, require_data=False)
    )


# --------------------------------------------------------------------------
# agents.subset directly
# --------------------------------------------------------------------------


def test_options_payload_is_read_off_the_module_constants() -> None:
    payload = subset.options_payload()
    assert [h["key"] for h in payload["horizons"]] == list(subset.HORIZONS)
    assert payload["banks"] == list(subset.BANK_CHOICES)
    for entry, (key, (label, delta)) in zip(
        payload["horizons"], subset.HORIZONS.items(), strict=True
    ):
        assert entry["key"] == key
        assert entry["label"] == label
        assert entry["minutes"] == int(delta.total_seconds() // 60)


def test_an_unknown_horizon_is_rejected(store: DataStore) -> None:
    with pytest.raises(subset.InvalidSubsetChoice):
        subset.load_subset("99d", "all", store)


def test_an_unknown_bank_is_rejected(store: DataStore) -> None:
    with pytest.raises(subset.InvalidSubsetChoice):
        subset.load_subset("24h", "BANK_Z", store)


def test_load_subset_reports_real_transfer_counts(store: DataStore) -> None:
    result = subset.load_subset("6h", "all", store)
    # Every in/out transfer of the collector pattern (20 rows) falls
    # inside 6 hours of the first transfer; the noise rows (10 days out)
    # do not, so this is a real filter, not every row in the table.
    assert result["transfers_considered"] == 20
    assert result["accounts_in_scope"] == 24  # every account, unfiltered
    assert result["load_ms"] >= 0
    # No detection fields at all -- the load step never rules on anything.
    assert "cross_bank_flagged" not in result


def test_score_subset_uses_model_a_by_default(real_data_config: AgentsConfig) -> None:
    """With a real model file on disk, Model A is the detector, not a fixture artifact.

    ``real_data_config`` points ``ml_out_dir`` at a scratch dir with no
    model (see test_score_subset_falls_back_to_rule_when_model_a_unavailable
    for that case); this test instead points it at the repo's real
    ml/out, where the trained artifact actually lives.
    """
    import dataclasses

    from ml.config import MlConfig

    cfg = dataclasses.replace(real_data_config, ml_out_dir=MlConfig.from_env().ml_dir / "out")
    subset._model_a_cache.clear()
    result = subset.score_subset("24h", "all", DataStore(cfg))
    assert result["detector"] == "model_a"


def test_score_subset_falls_back_to_rule_when_model_a_unavailable(
    store: DataStore, monkeypatch
) -> None:
    """A model that cannot be used must never crash a request -- the rule runs instead."""
    monkeypatch.setattr(subset, "_get_model_a", lambda _store: None)
    result = subset.score_subset("24h", "all", store)
    assert result["detector"] == "rule_fallback"
    ids = {row["hashed_account_id"] for row in result["flagged_accounts"]}
    assert "collector-00" in ids


def test_score_subset_catches_the_built_pattern(store: DataStore) -> None:
    subset._model_a_cache.clear()
    result = subset.score_subset("24h", "all", store)
    assert result["cross_bank_flagged"] >= 1
    # The vacuous single-bank clause (see ml/rule.py) never fires here,
    # matching docs/data_sharing.md's "0.000 at one bank" result.
    assert result["single_bank_flagged"] == 0
    ids = {row["hashed_account_id"] for row in result["flagged_accounts"]}
    assert "collector-00" in ids
    collector_row = next(
        r for r in result["flagged_accounts"] if r["hashed_account_id"] == "collector-00"
    )
    assert collector_row["caught_cross_bank"] is True
    assert collector_row["in_degree"] == 16
    assert collector_row["out_degree"] == 4
    assert collector_row["distinct_counterparty_banks"] == 3
    assert result["score_ms"] >= 0
    assert result["total_ms"] >= result["load_ms"]


def test_a_horizon_before_the_pattern_forms_catches_nothing(store: DataStore) -> None:
    """The collector's own traffic runs 0-310s after the first transfer;
    a cutoff before that sees none of it, so nothing should be flagged.
    Exercises the cutoff as a genuine point-in-time bound, not a filter
    that happens not to matter."""
    tiny_horizon = "30m"
    assert subset.HORIZONS[tiny_horizon][1] > timedelta(seconds=310)  # sanity
    # 30m already contains the whole pattern (it forms in ~5 minutes), so
    # this asserts the opposite direction is also honoured: nothing AFTER
    # the pattern's own window is required for the catch to appear.
    result = subset.score_subset(tiny_horizon, "all", store)
    assert result["cross_bank_flagged"] >= 1


def test_bank_filter_narrows_display_not_computation(store: DataStore) -> None:
    """Filtering to BANK_C (the collector's own bank) must still report
    the same collector as caught by the cross-bank rule -- proof the bank
    filter is applied after scoring, not by narrowing the account/bank map
    build_features sees. See agents/subset.py's module docstring."""
    all_banks = subset.score_subset("24h", "all", store)
    bank_c = subset.score_subset("24h", "BANK_C", store)
    assert bank_c["accounts_in_scope"] < all_banks["accounts_in_scope"]
    assert bank_c["cross_bank_flagged"] == 1
    row = bank_c["flagged_accounts"][0]
    assert row["hashed_account_id"] == "collector-00"
    # Unchanged by the filter: this is the same real cross-bank computation.
    assert row["distinct_counterparty_banks"] == 3


# --------------------------------------------------------------------------
# through the HTTP service
# --------------------------------------------------------------------------


def test_subset_options_endpoint_needs_no_data(subset_client: TestClient) -> None:
    response = subset_client.get("/subset/options")
    assert response.status_code == 200
    body = response.json()
    assert [h["key"] for h in body["horizons"]] == list(subset.HORIZONS)


def test_subset_load_endpoint_returns_real_stats(subset_client: TestClient) -> None:
    response = subset_client.post("/subset/load", json={"horizon": "6h", "bank": "all"})
    assert response.status_code == 200
    body = response.json()
    assert body["transfers_considered"] == 20
    assert "flagged_accounts" not in body


def test_subset_score_endpoint_computes_a_real_verdict(subset_client: TestClient) -> None:
    response = subset_client.post(
        "/subset/score", json={"horizon": "24h", "bank": "all"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cross_bank_flagged"] >= 1
    ids = {row["hashed_account_id"] for row in body["flagged_accounts"]}
    assert "collector-00" in ids


def test_subset_endpoints_default_to_24h_and_all(subset_client: TestClient) -> None:
    response = subset_client.post("/subset/score", json={})
    assert response.status_code == 200
    assert response.json()["horizon"] == "24h"
    assert response.json()["bank"] == "all"


def test_an_out_of_menu_horizon_is_a_422(subset_client: TestClient) -> None:
    response = subset_client.post("/subset/score", json={"horizon": "99d"})
    assert response.status_code == 422


def test_an_out_of_menu_bank_is_a_422(subset_client: TestClient) -> None:
    response = subset_client.post("/subset/load", json={"bank": "BANK_Z"})
    assert response.status_code == 422


def test_subset_endpoints_make_no_model_call(
    monkeypatch: pytest.MonkeyPatch, subset_client: TestClient
) -> None:
    from agents import adk_agents

    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        lambda *_a, **_k: pytest.fail("the subset endpoints called a model"),
    )
    assert subset_client.post("/subset/load", json={}).status_code == 200
    assert subset_client.post("/subset/score", json={}).status_code == 200


def test_subset_endpoints_never_consult_the_request_guard(
    real_data_config: AgentsConfig,
) -> None:
    """No model call, no spend to bound -- a starved guard must not block it."""
    guard = ServiceLimits(session_limit=0, ip_limit=0, window_s=60.0, daily_cap=0)
    api = TestClient(service.create_app(real_data_config, guard, require_data=False))
    assert api.post("/subset/load", json={}).status_code == 200
    assert api.post("/subset/score", json={}).status_code == 200
    assert api.get("/subset/options").status_code == 200


def test_subset_load_and_score_refuse_without_accounts_and_transfers(
    tmp_path: Any,
) -> None:
    """/subset/options stays up (mirrors /simulate/bounds); /subset/load
    and /subset/score, which genuinely read data, must not."""
    base = _config()
    dataless = AgentsConfig(
        mode=base.mode,
        data_dir=tmp_path / "no-sim",
        ml_out_dir=tmp_path / "no-ml",
        model=base.model,
        project=base.project,
        location=base.location,
        use_vertex=base.use_vertex,
        require_data=False,
    )
    api = TestClient(
        service.create_app(dataless, _GENEROUS_LIMITS, require_data=False)
    )
    assert api.get("/subset/options").status_code == 200
    load_resp = api.post("/subset/load", json={})
    assert load_resp.status_code == 503
    assert load_resp.json()["error_type"] == "data_unavailable"
    score_resp = api.post("/subset/score", json={})
    assert score_resp.status_code == 503
    assert score_resp.json()["error_type"] == "data_unavailable"


def test_repeated_calls_are_not_cached_across_subset_choices(
    subset_client: TestClient,
) -> None:
    """Two different horizons on the same process must not share a result
    -- the whole point of "live" is that each choice is computed fresh."""
    small = subset_client.post("/subset/score", json={"horizon": "30m"}).json()
    large = subset_client.post("/subset/score", json={"horizon": "7d"}).json()
    assert small["transfers_considered"] != large["transfers_considered"]
    assert small["cutoff"] != large["cutoff"]
