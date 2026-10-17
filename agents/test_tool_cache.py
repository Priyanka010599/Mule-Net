"""The deterministic-tool cache: keying, invalidation, bounds, thread safety."""

from __future__ import annotations

import threading

import pytest

from agents import adk_agents, tool_cache


class FakeStore:
    """Just enough of DataStore for the cache key."""

    def __init__(self, version: str = "v1") -> None:
        self.version = version

    def data_version(self, names: tuple[str, ...] = ()) -> str:
        return self.version


@pytest.fixture(autouse=True)
def _fresh_cache():
    tool_cache.CACHE.clear()
    yield
    tool_cache.CACHE.clear()


def _counting_tool(store: FakeStore, calls: list[str]):
    @tool_cache.cached(lambda: store)
    def run_sql(query: str, limit: int = 10) -> dict:
        calls.append(query)
        return {"rows": [query, limit]}

    return run_sql


def test_identical_call_is_served_from_cache_and_is_a_copy() -> None:
    store, calls = FakeStore(), []
    tool = _counting_tool(store, calls)
    first = tool("SELECT 1")
    first["rows"].append("mutated")
    second = tool("SELECT 1")
    assert calls == ["SELECT 1"]
    assert second == {"rows": ["SELECT 1", 10]}  # the mutation did not leak
    assert tool_cache.CACHE.stats()["hits"] == 1


def test_positional_keyword_and_default_spellings_share_a_key() -> None:
    store, calls = FakeStore(), []
    tool = _counting_tool(store, calls)
    tool("SELECT 1")
    tool(query="SELECT 1", limit=10)
    tool("  SELECT 1  ")
    assert len(calls) == 1


def test_different_args_miss() -> None:
    store, calls = FakeStore(), []
    tool = _counting_tool(store, calls)
    tool("SELECT 1")
    tool("SELECT 2")
    tool("SELECT 1", limit=5)
    assert len(calls) == 3


def test_new_data_version_invalidates() -> None:
    store, calls = FakeStore("v1"), []
    tool = _counting_tool(store, calls)
    tool("SELECT 1")
    store.version = "v2"  # the parquet was re-published
    tool("SELECT 1")
    assert len(calls) == 2


def test_two_stores_with_the_same_version_never_share() -> None:
    calls: list[str] = []
    holder = {"store": FakeStore("mock")}

    @tool_cache.cached(lambda: holder["store"])
    def tool(query: str) -> str:
        calls.append(query)
        return f"{len(calls)}"

    assert tool("q") == "1"
    holder["store"] = FakeStore("mock")
    assert tool("q") == "2"


def test_errors_are_not_cached() -> None:
    store, calls = FakeStore(), []

    @tool_cache.cached(lambda: store)
    def flaky(query: str) -> str:
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "ok"

    with pytest.raises(RuntimeError):
        flaky("q")
    assert flaky("q") == "ok"
    assert flaky("q") == "ok"
    assert len(calls) == 2


def test_lru_is_bounded_and_evicts_oldest() -> None:
    cache = tool_cache.ToolCache(maxsize=2)
    store = FakeStore()
    cache.put(store, ("a",), 1)
    cache.put(store, ("b",), 2)
    cache.get(store, ("a",))  # a is now the most recent
    cache.put(store, ("c",), 3)
    assert len(cache) == 2
    assert cache.get(store, ("b",))[0] is False
    assert cache.get(store, ("a",))[0] is True


def test_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULENET_AGENTS_TOOL_CACHE", "0")
    store, calls = FakeStore(), []
    tool = _counting_tool(store, calls)
    tool("SELECT 1")
    tool("SELECT 1")
    assert len(calls) == 2
    assert len(tool_cache.CACHE) == 0


def test_concurrent_callers_are_safe() -> None:
    store, calls = FakeStore(), []
    tool = _counting_tool(store, calls)
    errors: list[BaseException] = []

    def hammer(n: int) -> None:
        try:
            for i in range(200):
                assert tool(f"SELECT {i % 7}")["rows"][0] == f"SELECT {i % 7}"
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(tool_cache.CACHE) <= tool_cache.CACHE.maxsize


def test_a_cache_hit_is_still_a_ledger_execution() -> None:
    """The grounding test counts executions, so a hit must still be one."""
    store, calls = FakeStore(), []
    wrapped = adk_agents._model_tool(_counting_tool(store, calls), sink := set())
    adk_agents.reset_tool_log()
    wrapped("SELECT 1")
    wrapped("SELECT 1")
    assert len(calls) == 1
    assert [e["tool"] for e in adk_agents.TOOL_CALLS] == ["run_sql", "run_sql"]
    assert sink == {"run_sql"}


def test_real_tools_are_decorated() -> None:
    from agents import tools

    for name in ("describe_tables", "get_ring_summary", "get_ring_transfer_summary", "run_sql"):
        assert hasattr(getattr(tools, name), "__wrapped__"), name
    assert not hasattr(tools.get_account_profile, "__wrapped__")
