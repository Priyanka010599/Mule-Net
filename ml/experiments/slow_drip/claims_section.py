"""The slow-drip section appended to ``docs/claims.md``.

Same discipline as ``ml/claims.py``: no number is typed by hand. Each claim
is a key path into ``ml/experiments/slow_drip/metrics.json``, resolved with
``ml.claims.resolve``, rendered into a table and a machine-readable block
that ``ml/experiments/slow_drip/test_slow_drip.py`` re-resolves.
``ml.claims.render_markdown`` appends this section after the original
generated block, so ``ml/test_claims.py``'s exact-file check covers it too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRICS_RELATIVE = "ml/experiments/slow_drip/metrics.json"
BLOCK_START = "<!-- BEGIN SLOW-DRIP CLAIMS -->"
BLOCK_END = "<!-- END SLOW-DRIP CLAIMS -->"

_WORLDS = {"slow_drip_overlay": "slow-drip overlay", "canonical": "canonical"}
_DETECTORS = {
    "velocity_rule": "velocity rule",
    "isolation_forest": "Isolation Forest (unsupervised)",
    "xgboost_oof": "XGBoost (out-of-fold, same world)",
    "xgboost_canonical_transfer": "XGBoost trained on canonical, scored on overlay",
}
_SCOPES = ("single_bank", "cross_bank")
_METRICS = {
    "alerts": ("alerts raised", 0),
    "ring_recall": ("ring recall (>= 1 pass-through member alerted)", 4),
    "false_alerts_per_ring_caught": ("false alerts per ring caught", 3),
}


@dataclass(frozen=True)
class SlowDripClaim:
    id: str
    statement: str
    path: str
    decimals: int


def _alert_budget_text(metrics: dict[str, Any]) -> str:
    totals = {
        sum(w["budget_1pct_per_bank"].values()) for w in metrics["worlds"].values()
    }
    return " / ".join(str(t) for t in sorted(totals))


def _cell(detector: str, metric: str) -> str:
    """Rule is binary (flat keys); score-based detectors use the 1%/bank budget."""
    return metric if detector == "velocity_rule" else f"budget_1pct_per_bank.{metric}"


def claims_for(metrics: dict[str, Any]) -> list[SlowDripClaim]:
    """Derived from the metrics structure, so a new detector appears automatically."""
    out: list[SlowDripClaim] = []
    for world, world_label in _WORLDS.items():
        for scope in _SCOPES:
            detectors = metrics["worlds"][world]["detectors"][scope]
            for detector, det_label in _DETECTORS.items():
                if detector not in detectors:
                    continue
                for metric, (metric_label, decimals) in _METRICS.items():
                    out.append(
                        SlowDripClaim(
                            id=f"slow_drip_{world}_{scope}_{detector}_{metric}",
                            statement=f"{world_label}, {scope}, {det_label}: {metric_label}",
                            path=(
                                f"worlds.{world}.detectors.{scope}.{detector}."
                                f"{_cell(detector, metric)}"
                            ),
                            decimals=decimals,
                        )
                    )
    return out


def build_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    from ml.claims import resolve  # noqa: PLC0415 - ml.claims imports this module lazily

    return [
        {
            "id": c.id,
            "statement": c.statement,
            "value": round(float(resolve(metrics, c.path)), c.decimals),
            "decimals": c.decimals,
            "source_file": METRICS_RELATIVE,
            "path": c.path,
        }
        for c in claims_for(metrics)
    ]


def render_section(repo_root: Path) -> list[str]:
    """Markdown lines for the section, or nothing if the metrics are not there."""
    path = repo_root / METRICS_RELATIVE
    if not path.exists():
        return []
    metrics = json.loads(path.read_text("utf-8"))
    records = build_records(metrics)
    design = metrics["design"]
    lines = [
        "",
        "## Slow-drip stress test (appended section)",
        "",
        "Sourced from `" + METRICS_RELATIVE + "`; generated, do not edit by hand.",
        "Question: do the rule, an unsupervised Isolation Forest and XGBoost still catch",
        "rings whose hops are slower than the velocity rule's ceiling and whose amounts",
        "are small? The overlay is a separate synthetic world; `simulator/out` is untouched.",
        "",
        "**Ring caught** = " + design["ring_caught_definition"].split("= ", 1)[1] + ".",
        "**False alert** = " + design["false_alert_definition"].replace(" whose", ", whose") + ".",
        "**Thresholds**: " + design["thresholds"] + ".",
        "The velocity rule is not given a budget; it alerts wherever its fixed conditions hold,",
        f"so its alert count differs from the {_alert_budget_text(metrics)} alerts (1% per bank)",
        "the score-based detectors get; the `alerts raised` rows show both.",
        "Read the negative results in the table as plainly as the positive ones.",
        "",
        "| value | claim | path |",
        "| --- | --- | --- |",
    ]
    for r in records:
        lines.append(f"| `{r['value']:.{r['decimals']}f}` | {r['statement']} | `{r['path']}` |")
    lines += [
        "",
        BLOCK_START,
        "```json",
        json.dumps({"source": METRICS_RELATIVE, "claims": records}, indent=2, sort_keys=True),
        "```",
        BLOCK_END,
    ]
    return lines


def parse_section(text: str) -> dict[str, Any]:
    body = text.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0]
    return json.loads(body.strip().removeprefix("```json").removesuffix("```"))
