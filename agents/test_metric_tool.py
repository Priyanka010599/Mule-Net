"""``get_metric`` / ``describe_metrics``: chart numbers, from the published artifacts.

The tool reads ``agents/out/metric_catalog.json``, built by ``ml/metric_catalog.py``
from the same sources as ``docs/claims.md``. What these tests pin: values are the
artifact's values exactly (and the ones ``docs/claims.md`` prints), the stated
qualifier travels with every number, an unknown id lists the menu, a withdrawn
claim is refused with its reason, the IBM base-rate discrepancy is returned as
two statements and not resolved, and the answer grounding accepts the numbers that
came from the tool and refuses others. Gemini is not involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents import adk_agents, cases, grounding, hidden_labels, tools
from ml.claims import OUTPUT, SOURCES, load_sources, parse_payload, resolve
from ml.config import MlConfig
from ml.metric_catalog import COALITION, REDTEAM, build_catalog
from shared import metric_catalog

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """The catalog built from the real metrics files, or a skip naming the command."""
    cfg = MlConfig.from_env()
    needed = [ROOT / rel for rel in (*SOURCES.values(), COALITION, REDTEAM)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"metrics files missing ({missing[0]}...); run `python -m ml.early` etc.")
    docs = load_sources(cfg)
    return build_catalog(
        docs,
        json.loads((ROOT / COALITION).read_text("utf-8")),
        json.loads((ROOT / REDTEAM).read_text("utf-8")),
    )


@pytest.fixture
def published(built: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """``agents/out`` holding the real catalog, and a fresh artifact cache."""
    folder = tmp_path / "agents_out"
    folder.mkdir()
    (folder / metric_catalog.ARTIFACT_NAME).write_text(json.dumps(built), encoding="utf-8")
    monkeypatch.setattr(cases, "DEFAULT_OUT_DIR", folder)
    tools._PUBLISHED_CACHE.clear()
    return built


def _by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {v["id"]: v for v in result["values"]}


def test_false_alerts_per_catch_are_the_chart_numbers(published: dict) -> None:
    result = tools.get_metric("false_alerts_per_catch_t5")
    values = _by_id(result)
    assert values["alert_burden_t5_single_bank"]["value"] == 6.23
    assert values["alert_burden_t5_cross_bank"]["value"] == 1.97
    for value in values.values():
        assert value["qualifier"], "a number travels with its qualifier"
        assert value["source_file"] == "ml/experiments/early_detection/metrics.json"


def test_values_equal_docs_claims_md_and_the_source_files(published: dict) -> None:
    printed = parse_payload((ROOT / OUTPUT).read_text("utf-8"))["claims"]
    for record in printed:
        got = tools.get_metric(record["id"])["values"][0]
        assert got["value"] == record["value"], record["id"]
        assert got["qualifier"] == record.get("qualifier", "")
        assert got["path"] == record["path"]
    docs = load_sources(MlConfig.from_env())
    sample = next(r for r in printed if r["id"] == "alert_burden_t5_cross_bank")
    source = resolve(docs["early_detection"], sample["path"])
    assert round(source, sample["decimals"]) == tools.get_metric(sample["id"])["values"][0]["value"]


def test_ring_recall_single_vs_joined(published: dict) -> None:
    values = _by_id(tools.get_metric("ring_recall_single_vs_joined"))
    assert values["ring_recall_t180_single_bank"]["value"] == 0.925
    assert values["ring_recall_t180_cross_bank"]["value"] == 1.0
    assert values["ring_recall_t180_single_bank"]["percent"] == 92.5


def test_coalition_recall_by_banks_joined_is_copied_from_the_file(published: dict) -> None:
    doc = json.loads((ROOT / COALITION).read_text("utf-8"))
    extra = tools.get_metric("coalition_recall_by_banks_joined")["extra"]
    real = extra["worlds"]["three_bank"]
    assert real["is_real_partition"] is True
    assert extra["worlds"]["four_bank"]["is_real_partition"] is False
    for got, want in zip(
        real["by_banks_joined"], doc["worlds"]["three_bank"]["by_size"], strict=True
    ):
        assert got["banks_joined"] == want["banks_joined"]
        assert got["ring_recall"] == want["ring_recall"]
        assert got["rings_caught"] == want["rings_caught"]
    assert [r["banks_joined"] for r in real["by_banks_joined"]] == [1, 2, 3]
    assert real["by_banks_joined"][0]["ring_recall"] == 0.0
    assert real["by_banks_joined"][1]["ring_recall_percent"] == 60.3
    assert "SHIPPED rule" in extra["two_variants"]


def test_redteam_counts_are_the_files(published: dict) -> None:
    doc = json.loads((ROOT / REDTEAM).read_text("utf-8"))
    extra = tools.get_metric("redteam_results")["extra"]
    assert (extra["strategies_run"], extra["caught"], extra["missed"]) == (
        doc["strategies_run"],
        doc["caught"],
        doc["missed"],
    )
    assert extra["isolated_graph_caveat"] == doc["isolated_graph_caveat"]
    assert len(extra["strategies"]) == doc["strategies_run"]


def test_ibm_benchmark_returns_both_base_rates_and_picks_neither(published: dict) -> None:
    result = tools.get_metric("ibm_aml_benchmark")
    values = _by_id(result)
    assert values["ibm_aml_xgboost_pr_auc"]["value"] == 0.0813
    assert values["ibm_aml_giant_component_realized_fpr"]["value"] == 0.7189
    assert "1.23% of accounts positive" in values["ibm_aml_xgboost_pr_auc"]["qualifier"]
    said = [s["says"] for s in result["documentation_discrepancy"]["statements"]]
    assert any("1.23%" in text for text in said)
    assert any("0.0102" in text for text in said)
    assert "does not pick one" in result["documentation_discrepancy"]["status"]
    assert result["extra"]["accounts"] == 515088 and result["extra"]["positives"] == 6357


def test_an_unknown_name_lists_the_menu(published: dict) -> None:
    out = adk_agents._model_tool(tools.get_metric)("made_up_chart")
    assert out["error"]["error_type"] == "ValueError"
    message = out["error"]["message"]
    assert "false_alerts_per_catch_t5" in message and "alert_burden_t5_cross_bank" in message


def test_a_withdrawn_claim_is_refused_with_its_reason(published: dict) -> None:
    retired = published["retired"][0]
    out = adk_agents._model_tool(tools.get_metric)(retired["id"])
    assert "withdrawn" in out["error"]["message"] and retired["id"] in out["error"]["message"]


def test_the_race_chart_points_at_get_ring_race(published: dict) -> None:
    out = adk_agents._model_tool(tools.get_metric)(metric_catalog.RACE_CHART)
    assert "get_ring_race" in out["error"]["message"]


def test_describe_metrics_lists_every_chart_claim_and_the_race_pointer(published: dict) -> None:
    listing = tools.describe_metrics()
    assert {c["name"] for c in listing["charts"]} == set(metric_catalog.METRIC_CHARTS)
    assert {c["id"] for c in listing["claims"]} == set(published["claims"])
    assert metric_catalog.RACE_CHART in listing["race"]
    assert listing["retired"], "withdrawn numbers are listed so they are not quoted"


def test_every_chart_in_the_menu_has_something_to_return(published: dict) -> None:
    for name in metric_catalog.METRIC_CHARTS:
        result = tools.get_metric(name)
        assert result["values"] or result.get("extra"), name
    assert set(metric_catalog.CHART_MENU) == {*metric_catalog.METRIC_CHARTS, "race_to_exit"}


def test_no_ground_truth_key_is_returned(published: dict) -> None:
    for name in metric_catalog.METRIC_CHARTS:
        result = adk_agents._model_tool(tools.get_metric)(name)
        assert result == hidden_labels.scrub(result), name


def test_grounding_accepts_chart_numbers_and_refuses_invented_ones(published: dict) -> None:
    result = adk_agents._model_tool(tools.get_metric)("false_alerts_per_catch_t5")
    grounded = "Single-bank needs 6.23 false alerts per mule caught, cross-bank 1.97."
    assert grounding.ungrounded_numbers(grounded, "q", [result]) == []
    assert grounding.ungrounded_numbers("Cross-bank needs 2.5.", "q", [result]) == ["2.5"]
    coalition = adk_agents._model_tool(tools.get_metric)("coalition_recall_by_banks_joined")
    assert grounding.ungrounded_numbers("Two banks catch 60.3% of rings.", "q", [coalition]) == []


def test_a_missing_catalog_is_a_clear_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cases, "DEFAULT_OUT_DIR", tmp_path)
    tools._PUBLISHED_CACHE.clear()
    out = adk_agents._model_tool(tools.get_metric)("pr_auc_t5")
    assert out["error"]["error_type"] == "MetricsUnavailable"
    assert "metrics unavailable" in out["error"]["message"]
    assert "python -m ml.claims" in out["error"]["message"]


def test_the_tools_are_registered_for_ask_and_listed() -> None:
    for tool in (tools.get_metric, tools.describe_metrics):
        assert tool in tools.ALL_TOOLS
        assert tool in adk_agents.AGENT_TOOLS["analyst_qa"]
