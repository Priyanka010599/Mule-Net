"""Build ``agents/out/metric_catalog.json``: the numbers behind each Metrics chart.

Ask's ``get_metric`` tool reads this file. It is built from exactly the sources
``docs/claims.md`` is built from (``ml.claims.CLAIMS`` resolved over the metrics
files, so every value carries the key path and qualifier it was quoted with) plus
two metrics files that have no claim: the coalition sweep and the red-team run.
Nothing is recomputed or re-rounded: a claim's value is the one ``docs/claims.md``
prints, and coalition and red-team figures are copied from their files as written.

Two additions, both display conveniences and both labelled as such: a ``percent``
for values that are shares (the Metrics tab prints 60.3%, and an answer that says
60.3 must find that number in a tool result), rounded to one decimal exactly as the
tab does, always next to the source value.

A documentation discrepancy is carried rather than resolved: the IBM README calls
0.0102 the "base rate" while the claims say 1.23% of accounts are positive. Both are
returned with the basis each states; the results file's own counts are included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.claims import RETIRED, SOURCES, build_payload, load_sources
from ml.config import MlConfig
from shared.metric_catalog import ARTIFACT_NAME, METRIC_CHARTS, SCHEMA_VERSION

COALITION = "ml/experiments/coalition/metrics.json"
REDTEAM = "ml/experiments/redteam/metrics.json"

#: Claim-id prefixes whose values are shares of a population (0 to 1), so a
#: ``percent`` is offered next to them.
SHARE_PREFIXES = ("ring_recall_", "coverage_span", "ibm_aml_giant_component")

IBM_DISCREPANCY = {
    "topic": "IBM AML benchmark base rate",
    "statements": [
        {
            "source": "docs/claims.md (ibm_aml_xgboost_pr_auc qualifier) and "
            "ml/experiments/ibm_aml/README.md, dataset description",
            "says": "base rate 1.23% of accounts positive (0.102% of transactions)",
            "basis": "share of accounts that touch a laundering transaction",
        },
        {
            "source": "ml/experiments/ibm_aml/README.md, closing paragraph",
            "says": "PR-AUC 5-9x the 0.0102 base rate",
            "basis": "not stated; 0.0102 is also the degree-threshold baseline's realized "
            "false-positive rate in that README's results table",
        },
    ],
    "status": "The two documents disagree and this catalogue does not pick one. Quote the "
    "claim's own qualifier (1.23% of accounts) and say the README's 0.0102 figure is "
    "labelled differently; the owner has to reconcile the documentation.",
}


def _percent(value: float) -> float:
    return round(value * 100, 1)


def _claim_entry(record: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "id": record["id"],
        "statement": record["statement"],
        "value": record["value"],
        "qualifier": record.get("qualifier", ""),
        "source_file": record["source_file"],
        "path": record["path"],
    }
    if "minus_path" in record:
        entry["minus_path"] = record["minus_path"]
    if record["id"].startswith(SHARE_PREFIXES) and 0 <= record["value"] <= 1:
        entry["percent"] = _percent(record["value"])
    return entry


def _coalition(doc: dict[str, Any]) -> dict[str, Any]:
    worlds: dict[str, Any] = {}
    for key, world in doc["worlds"].items():
        worlds[key] = {
            "label": world["label"],
            "is_real_partition": world["is_real_partition"],
            "note": world["note"],
            "by_banks_joined": [
                {
                    **{
                        name: row[name]
                        for name in (
                            "banks_joined",
                            "coalitions",
                            "ring_recall",
                            "ring_recall_min",
                            "ring_recall_max",
                            "ring_recall_no_spread",
                            "ring_recall_reachable",
                            "rings_caught",
                            "rings_reachable",
                        )
                        if name in row
                    },
                    "ring_recall_percent": _percent(row["ring_recall"]),
                }
                for row in world["by_size"]
            ],
        }
    return {
        "source_file": COALITION,
        "generated_at": doc["generated_at"],
        "metric": doc["metric"],
        "metric_definition": doc["metric_definition"],
        "two_variants": doc["two_variants"],
        "not_comparable_to_claims": doc["not_comparable_to_claims"],
        "true_rings_total": doc["true_rings_total"],
        "worlds": worlds,
    }


def _redteam(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": REDTEAM,
        "generated_at": doc["generated_at"],
        "strategies_run": doc["strategies_run"],
        "caught": doc["caught"],
        "missed": doc["missed"],
        "isolated_graph_caveat": doc["isolated_graph_caveat"],
        "scorer_not_attacked": doc["scorer_not_attacked"],
        "strategies": [
            {"name": v["name"], "aimed_at_clause": v.get("aimed_at_clause"), "caught": v["caught"]}
            for v in doc["verdicts"]
        ],
    }


def build_catalog(docs: dict[str, Any], coalition: dict[str, Any], redteam: dict[str, Any]) -> dict:
    """The catalog: every claim by id, and each chart's claims and extra figures."""
    payload = build_payload(docs)
    claims = {record["id"]: _claim_entry(record) for record in payload["claims"]}
    charts: dict[str, dict[str, Any]] = {}
    for key, spec in METRIC_CHARTS.items():
        prefixes = tuple(spec["claim_prefixes"])  # type: ignore[arg-type]
        charts[key] = {
            "title": spec["title"],
            "claim_ids": [cid for cid in claims if prefixes and cid.startswith(prefixes)],
        }
    charts["coalition_recall_by_banks_joined"]["extra"] = _coalition(coalition)
    charts["redteam_results"]["extra"] = _redteam(redteam)
    ibm = docs["ibm_aml"]
    charts["ibm_aml_benchmark"]["extra"] = {
        "source_file": SOURCES["ibm_aml"],
        "accounts": ibm["accounts"],
        "positives": ibm["positives"],
        "label": ibm["label"],
    }
    charts["ibm_aml_benchmark"]["documentation_discrepancy"] = IBM_DISCREPANCY
    return {
        "schema_version": SCHEMA_VERSION,
        "claims": claims,
        "charts": charts,
        "retired": [
            {"id": item.id, "statement": item.statement, "reason": item.reason}
            for item in RETIRED
        ],
    }


def artifact_path(cfg: MlConfig) -> Path:
    return cfg.ml_dir.parent / "agents" / "out" / ARTIFACT_NAME


def write(cfg: MlConfig, docs: dict[str, Any] | None = None) -> Path:
    """Build from the metrics files and write ``agents/out/metric_catalog.json``."""
    root = cfg.ml_dir.parent
    docs = docs or load_sources(cfg)
    catalog = build_catalog(
        docs,
        json.loads((root / COALITION).read_text("utf-8")),
        json.loads((root / REDTEAM).read_text("utf-8")),
    )
    path = artifact_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, sort_keys=True), "utf-8")
    return path


