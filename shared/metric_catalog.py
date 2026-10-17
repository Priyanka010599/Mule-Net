"""The chart and claim menu behind Ask's ``get_metric`` tool.

Ask's detection tables cannot answer "explain this chart": the numbers on the
Metrics tab live in metrics artifacts and ``docs/claims.md``, not in a table.
``ml/metric_catalog.py`` builds ``agents/out/metric_catalog.json`` from those
artifacts, the agents service's ``get_metric`` reads it, and the UI and agents
both validate the optional ``chart`` in ``ui_context`` against :data:`CHART_MENU`.
This module is the part both images need: names and the menu.

Nothing here states a number. Which claims a chart draws on is chosen by claim-id
prefix over ``ml.claims.CLAIMS`` when the catalog is built, so a claim added or
renamed there reaches the menu without anyone editing a list here too.
"""

from __future__ import annotations

#: File name under ``agents/out/`` that the agents service reads.
ARTIFACT_NAME = "metric_catalog.json"

#: Bump when the artifact's shape changes, so a reader can refuse an old one.
SCHEMA_VERSION = 1

#: The one chart whose answer comes from ``get_ring_race`` rather than the catalog.
RACE_CHART = "race_to_exit"

#: Chart key -> title shown to the model, and the claim-id prefixes it draws on
#: (``None`` where it comes from a metrics file that has no claim). Keys are the
#: fixed menu ``ui_context.chart`` is validated against.
METRIC_CHARTS: dict[str, dict[str, object]] = {
    "false_alerts_per_catch_t5": {
        "title": "False alerts per mule caught, T=5m (single-bank vs cross-bank)",
        "claim_prefixes": ("alert_burden_t5",),
    },
    "pr_auc_t5": {
        "title": "Early-warning PR-AUC at T=5m (single-bank vs cross-bank)",
        "claim_prefixes": ("pr_auc_t5",),
    },
    "ring_recall_single_vs_joined": {
        "title": "Share of true rings flagged by T=180m, single-bank vs cross-bank",
        "claim_prefixes": ("ring_recall_t180",),
    },
    "warning_coverage_by_span": {
        "title": "Rings warned before cash-out, by ring span",
        "claim_prefixes": ("coverage_span",),
    },
    "lead_time": {
        "title": "How much earlier cross-bank flags than single-bank",
        "claim_prefixes": ("lead_", "exit_minutes_", "exits_inside_grid"),
    },
    "coalition_recall_by_banks_joined": {
        "title": "Ring recall by number of banks joined (coalition sweep)",
        "claim_prefixes": (),
    },
    "redteam_results": {
        "title": "Red-team: Gemini-designed evasion strategies against the rule",
        "claim_prefixes": (),
    },
    "ibm_aml_benchmark": {
        "title": "IBM AML public benchmark (independent dataset sanity check)",
        "claim_prefixes": ("ibm_aml_",),
    },
}

#: Title of the race chart, which has no catalog entry (it is per ring).
RACE_TITLE = "Race to the exit: when each scope flagged a ring against when its money left"

#: Every value ``ui_context.chart`` may take, in menu order.
CHART_MENU: tuple[str, ...] = (*METRIC_CHARTS, RACE_CHART)

#: Chart key -> the title the model is shown for it.
CHART_TITLES: dict[str, str] = {
    **{key: str(spec["title"]) for key, spec in METRIC_CHARTS.items()},
    RACE_CHART: RACE_TITLE,
}
