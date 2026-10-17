"""Item 4: the subset panel builds each horizon's features once, keyed by the
data's content hash, and a slow build becomes a structured 504 instead of a
frozen service. Semantics (the bank filter narrows only what is shown) hold."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import subset
from agents.data import DataStore
from agents.test_subset import real_data_config, store, subset_client  # noqa: F401


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    subset._prepared_cache.clear()
    yield
    subset._prepared_cache.clear()


def _count_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = subset.build_features

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(subset, "build_features", counting)
    return calls


def test_load_then_score_and_every_bank_share_one_build(
    store: DataStore,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = _count_builds(monkeypatch)
    first = subset.load_subset("24h", "all", store)
    second = subset.score_subset("24h", "all", store)
    third = subset.score_subset("24h", "BANK_C", store)
    assert len(builds) == 1
    assert (first["cache"], second["cache"], third["cache"]) == ("miss", "hit", "hit")
    assert second["feature_build_ms"] == first["feature_build_ms"]
    assert first["data_version"] == second["data_version"] == store.data_version()


def test_a_hit_returns_exactly_what_a_fresh_build_would(store: DataStore) -> None:  # noqa: F811
    cold = subset.score_subset("24h", "all", store)
    warm = subset.score_subset("24h", "all", store)
    assert (cold["cache"], warm["cache"]) == ("miss", "hit")
    for key in ("cross_bank_flagged", "single_bank_flagged", "flagged_accounts", "detector"):
        assert warm[key] == cold[key]
    # The shared frame was not mutated by scoring: a different bank still works.
    assert subset.score_subset("24h", "BANK_C", store)["cross_bank_flagged"] == 1


def test_a_different_horizon_or_data_version_is_a_different_build(
    store: DataStore,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = _count_builds(monkeypatch)
    subset.load_subset("6h", "all", store)
    subset.load_subset("24h", "all", store)
    assert len(builds) == 2
    monkeypatch.setattr(store, "data_version", lambda *a, **k: "republished")
    assert subset.load_subset("6h", "all", store)["cache"] == "miss"
    assert len(builds) == 3


def test_data_version_is_a_content_hash_of_the_tables(store: DataStore) -> None:  # noqa: F811
    version = store.data_version()
    assert len(version) == 16 and version == store.data_version()
    path = store.source_path("accounts")
    path.write_bytes(path.read_bytes() + b"x")  # any change to the file...
    assert DataStore(store.config).data_version() != version  # ...is a new version


def test_concurrent_first_requests_build_once(
    store: DataStore,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = _count_builds(monkeypatch)
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(target=lambda: results.append(subset.load_subset("24h", "all", store)))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(builds) == 1
    assert sorted(r["cache"] for r in results) == ["hit", "hit", "hit", "miss"]


def test_warm_builds_every_horizon_and_a_failure_does_not_raise(
    store: DataStore,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset.warm(store)
    assert {h for (_v, h) in subset._prepared_cache} == set(subset.HORIZONS)
    subset._prepared_cache.clear()
    monkeypatch.setattr(subset, "_build_prepared", lambda *a, **k: 1 / 0)
    subset.warm(store)  # logged, not raised
    assert not subset._prepared_cache


def test_a_slow_build_is_a_structured_504_and_the_retry_is_a_hit(
    subset_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = subset._build_prepared

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(subset, "_build_prepared", slow)
    monkeypatch.setattr(subset, "REQUEST_TIMEOUT_S", 0.1)
    response = subset_client.post("/subset/score", json={"horizon": "24h", "bank": "all"})
    assert response.status_code == 504
    body = response.json()
    assert body["error_type"] == "timeout"
    assert body["retry_after_s"] == 15
    time.sleep(0.8)  # the build kept going on its thread and landed in the cache
    monkeypatch.setattr(subset, "REQUEST_TIMEOUT_S", 25.0)
    again = subset_client.post("/subset/score", json={"horizon": "24h", "bank": "all"})
    assert again.status_code == 200
    assert again.json()["cache"] == "hit"


def test_the_service_stays_responsive_while_a_build_runs(
    subset_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = subset._build_prepared
    started = threading.Event()

    def slow(*args: Any, **kwargs: Any) -> Any:
        started.set()
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(subset, "_build_prepared", slow)
    worker = threading.Thread(
        target=lambda: subset_client.post("/subset/load", json={"horizon": "24h"})
    )
    worker.start()
    assert started.wait(5)
    began = time.perf_counter()
    assert subset_client.get("/subset/options").status_code == 200
    assert time.perf_counter() - began < 0.4  # not queued behind the build
    worker.join()
