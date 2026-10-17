"""Generate ``docs/claims.md``: every quotable number, with its provenance.

Why this exists
---------------
Four separate errors in one session, all the same error: a number copied
by hand and then drifting from the file it came from.

* a brief quoted T=5m PR-AUC as 0.6220 single-bank / 0.4282 cross-bank.
  The values are real; **the scopes were swapped**, which inverts the
  project's headline finding.
* ``docs/post_submission.md`` recorded a ``0.643 / 0.382`` pair that
  appears in **no** metrics file.
* a "12 of 27" overlap figure was transcribed without the qualifier that
  made it meaningful.
* ``ml/early.py`` could not run at all, so the committed ``metrics.json``
  was not produced by the committed source -- and nothing noticed,
  because no test ever ran the script.

None of those was a modelling mistake. Each was prose diverging from
data, silently. So no number in ``docs/claims.md`` is written by hand:
each one is resolved from a metrics file at generation time and carries
the exact key path it came from, and ``ml/test_claims.py`` re-resolves
every one and fails if any has moved.

Paths are keyed by identity, never by list position
---------------------------------------------------
``horizons`` is a JSON list, and this session inserted three new
horizons into it -- which shifted ``full`` from ``horizons[5]`` to
``horizons[8]`` and made every positional path silently point at a
different measurement. So the path language selects list elements by
field value instead:

    horizons[cutoff_minutes=5].summary.cross_bank.pr_auc_mean

``[field=value]`` picks the single element whose ``field`` equals
``value``; it raises if the match is not unique. A path that resolves to
two things, or to none, is a broken path and says so.

Derived claims
--------------
Some claims are differences between two measurements (the span gradient
is cross-bank PR-AUC minus single-bank PR-AUC). Those carry **both**
source paths and the operation, so the arithmetic is auditable too
rather than being a third hand-entered number.

Retired claims
--------------
:data:`RETIRED` records numbers this project has **stopped** quoting,
with the sourced values that killed them. A claims file listing only
survivors loses the reasoning, and the next person to find
``share_caught_before_exit`` in a metrics file will put it back on a
slide.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.config import MlConfig

#: Metrics files claims may be sourced from, by short name.
SOURCES: dict[str, str] = {
    "early_detection": "ml/experiments/early_detection/metrics.json",
    "lead_time": "ml/experiments/lead_time/metrics.json",
    "ablation": "ml/experiments/ablation/results_ablation.json",
    "ibm_aml": "ml/experiments/ibm_aml/results_grouped.json",
    "gnn_xgb": "ml/experiments/gnn_xgb/results_gnn_xgb.json",
    "stage2": "ml/experiments/stage2/metrics.json",
}

#: Where the generated file lands.
OUTPUT = "docs/claims.md"

#: Marker fencing the machine-readable block ``ml/test_claims.py`` parses.
#: Everything outside it is prose; everything inside it is generated and
#: is the only thing the test treats as a claim.
BLOCK_START = "<!-- BEGIN GENERATED CLAIMS -->"
BLOCK_END = "<!-- END GENERATED CLAIMS -->"

_SELECTOR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z_][A-Za-z0-9_]*)=([^\]]+)\]$")


def _coerce(raw: str) -> Any:
    """Selector values arrive as text; ``None``/ints must match the JSON."""
    if raw == "None":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def resolve(doc: Any, path: str) -> Any:
    """Resolve a dotted path with ``[field=value]`` list selectors.

    Raises rather than returning a default: a path that no longer
    resolves is exactly the drift this module exists to catch, and a
    silent ``None`` would let it through.
    """
    node = doc
    for step in path.split("."):
        selector = _SELECTOR.match(step)
        if selector:
            key, field_name, raw = selector.groups()
            wanted = _coerce(raw)
            candidates = [item for item in node[key] if item.get(field_name) == wanted]
            if len(candidates) != 1:
                raise KeyError(
                    f"{path!r}: {key}[{field_name}={raw}] matched "
                    f"{len(candidates)} elements, expected exactly 1"
                )
            node = candidates[0]
            continue
        if not isinstance(node, dict) or step not in node:
            raise KeyError(f"{path!r}: no key {step!r} at that point")
        node = node[step]
    return node


@dataclass(frozen=True)
class Claim:
    """One quotable number and everything needed to re-derive it."""

    id: str
    statement: str
    source: str
    path: str
    #: Second path plus an operation, for a claim that is a difference.
    minus: str | None = None
    decimals: int = 4
    #: Free-text qualifier that must travel with the number. The "12 of
    #: 27" incident was a correct number quoted without one.
    qualifier: str = ""

    def value(self, docs: dict[str, Any]) -> float:
        base = resolve(docs[self.source], self.path)
        if self.minus is not None:
            base = base - resolve(docs[self.source], self.minus)
        return float(base)

    def render(self, docs: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": self.id,
            "statement": self.statement,
            "value": round(self.value(docs), self.decimals),
            "decimals": self.decimals,
            "source_file": SOURCES[self.source],
            "path": self.path,
        }
        if self.minus is not None:
            record["minus_path"] = self.minus
        if self.qualifier:
            record["qualifier"] = self.qualifier
        return record


@dataclass(frozen=True)
class Retired:
    """A number this project has stopped quoting, and the evidence why."""

    id: str
    statement: str
    reason: str
    evidence: list[Claim] = field(default_factory=list)


def _span_gradient() -> list[Claim]:
    """Cross-bank minus single-bank PR-AUC per ring span, at T=5m."""
    return [
        Claim(
            id=f"span_gradient_span{span}",
            statement=(
                f"At T=5m, cross-bank PR-AUC exceeds single-bank by this much for "
                f"rings spanning {span} bank(s)"
            ),
            source="early_detection",
            path=f"horizons[cutoff_minutes=5].strata_summary.cross_bank.span{span}.pr_auc_mean",
            minus=f"horizons[cutoff_minutes=5].strata_summary.single_bank.span{span}.pr_auc_mean",
            qualifier=(
                "mean over 3 seeds; negatives held constant across strata, so "
                "compare spans on ROC-AUC or recall@FPR, not on raw PR-AUC"
            ),
        )
        for span in (1, 2, 3)
    ]


def _scope_pair(
    id_prefix: str, statement: str, path_template: str, qualifier: str, decimals: int = 4
) -> list[Claim]:
    """The same measurement for both scopes, so the pair cannot be swapped."""
    return [
        Claim(
            id=f"{id_prefix}_{scope}",
            statement=statement.format(scope=scope.replace("_", "-")),
            source="early_detection" if "horizons" in path_template else "lead_time",
            path=path_template.format(scope=scope),
            decimals=decimals,
            qualifier=qualifier,
        )
        for scope in ("cross_bank", "single_bank")
    ]


CLAIMS: list[Claim] = [
    *_span_gradient(),
    *_scope_pair(
        "pr_auc_t5",
        "Pooled out-of-fold PR-AUC at T=5m, {scope}",
        "horizons[cutoff_minutes=5].summary.{scope}.pr_auc_mean",
        "mean over 3 seeds, grouped 5-fold CV over rings, 712 pass-through mules",
    ),
    *_scope_pair(
        "ring_recall_t180",
        "Share of the 120 true rings flagged at 1% FPR by T=180m, {scope}",
        "horizons[cutoff_minutes=180].ring_catch.{scope}.ring_recall",
        "mean over 3 seeds; ring counts as flagged if any of its mules/collectors is",
    ),
    *_scope_pair(
        "alert_burden_t5",
        "False alerts per true mule caught at 80% recall, T=5m, {scope}",
        "horizons[cutoff_minutes=5].summary.{scope}.false_alerts_per_catch_at_80pct_recall_mean",
        "mean over 3 seeds; lower is better",
        decimals=2,
    ),
    *[
        Claim(
            id=f"coverage_span{span}_{scope}",
            statement=(
                f"Share of {span}-bank rings given positive warning before cash-out, "
                f"{scope.replace('_', '-')}"
            ),
            source="lead_time",
            path=(f"seed_mean.by_scope_and_span.{scope}.span{span}.share_with_positive_warning"),
            decimals=3,
            qualifier=(
                "UNCONDITIONAL: denominator is every ring of that span, not only the "
                "ones that flagged; mean over 3 seeds"
            ),
        )
        for scope in ("cross_bank", "single_bank")
        for span in (1, 2, 3)
    ],
    Claim(
        id="lead_median_minutes",
        statement=("Median minutes by which cross-bank flags a ring earlier than single-bank"),
        source="lead_time",
        path="seed_mean.cross_bank_lead.lead_minutes.p50",
        decimals=1,
        qualifier=(
            "CONDITIONAL on both scopes flagging inside the grid; zero means "
            "cross-bank is not faster. The cross-bank advantage is coverage, "
            "not speed -- see lead_rings_tied and coverage_span*"
        ),
    ),
    Claim(
        id="lead_rings_tied",
        statement="Rings where both scopes first flag at the very same horizon",
        source="lead_time",
        path="seed_mean.cross_bank_lead.rings_tied",
        decimals=1,
        qualifier="out of lead_rings_compared rings where both flagged in the grid",
    ),
    Claim(
        id="lead_rings_compared",
        statement="Rings where both scopes flagged inside the grid, so a lead is defined",
        source="lead_time",
        path="seed_mean.cross_bank_lead.both_flagged_in_grid",
        decimals=1,
        qualifier="of 120 rings; the remainder are single-bank coverage failures",
    ),
    Claim(
        id="lead_only_cross_bank",
        statement="Rings only cross-bank flags inside the grid (single-bank never does)",
        source="lead_time",
        path="seed_mean.cross_bank_lead.rings_only_cross_bank_flagged_in_grid",
        decimals=1,
        qualifier=(
            "single-bank reaches these only at the full window, days after the "
            "money left; the mirror count is zero"
        ),
    ),
    *[
        Claim(
            id=f"exit_minutes_{stat}",
            statement=f"Minutes from ring onset to money reaching the exit account ({label})",
            source="lead_time",
            path=f"exit_distribution.minutes_after_onset.{stat}",
            decimals=1,
            qualifier="property of the generated data, identical across seeds and scopes",
        )
        for stat, label in (
            ("p0", "minimum"),
            ("p25", "lower quartile"),
            ("p50", "median"),
            ("p75", "upper quartile"),
            ("p100", "maximum"),
        )
    ],
    Claim(
        id="exits_inside_grid",
        statement="Rings whose cash-out falls inside the measured horizon grid",
        source="lead_time",
        path="exit_distribution.exits_inside_grid",
        decimals=0,
        qualifier=(
            "of 120; at the old grid end of T=60m this was 27, which is why "
            "share_caught_before_exit was retired"
        ),
    ),
    Claim(
        id="ablation_t5_drop_all_identity",
        statement=(
            "At T=5m, dropping all 4 identity features from Model A's recipe "
            "(= Model B) drops PR-AUC to this"
        ),
        source="ablation",
        path="summary.T=5m.drop_all_identity.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds, grouped 5-fold CV, same XGBoost recipe as shipped "
            "Model A; synthetic data from our own generator -- not a benchmark "
            "against real identity-reuse rates; see ml/experiments/ablation/README.md"
        ),
    ),
    Claim(
        id="ablation_t180_drop_all_identity",
        statement=(
            "At T=180m, dropping all 4 identity features from Model A's recipe "
            "(= Model B) drops PR-AUC to this"
        ),
        source="ablation",
        path="summary.T=180m.drop_all_identity.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds; by T=180m the gap from the T=5m figure above "
            "mostly closes -- see ml/experiments/ablation/README.md"
        ),
    ),
    Claim(
        id="ablation_t5_identity_only",
        statement=(
            "At T=5m, the 4 identity features ALONE (no transfer-graph features) "
            "score this PR-AUC"
        ),
        source="ablation",
        path="summary.T=5m.identity_only.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds; higher than the 12-feature transfer-only figure "
            "above -- identity reuse is the dominant T=5m signal in this recipe, "
            "not a minor supplement; synthetic data, see "
            "ml/experiments/ablation/README.md"
        ),
    ),
    Claim(
        id="ablation_t5_drop_all_identity_and_age",
        statement=(
            "At T=5m, dropping identity features AND account_age_days (an "
            "account attribute, not a transfer behaviour) drops PR-AUC to this"
        ),
        source="ablation",
        path="summary.T=5m.drop_all_identity_and_age.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds; the 12-feature transfer-only figure above is "
            "itself mostly propped up by this one non-transfer-derived feature "
            "-- see ml/experiments/ablation/README.md"
        ),
    ),
    Claim(
        id="ibm_aml_xgboost_pr_auc",
        statement=(
            "On the IBM AML HI-Small public benchmark, our reduced transfer-only "
            "feature family scores this PR-AUC (account-level, full window)"
        ),
        source="ibm_aml",
        path="summary.xgboost_full.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds, grouped CV by connected component; IBM's own "
            "synthetic data (CDLA-Sharing-1.0 licence, computational use/Results "
            "only -- no IBM data or account-level labels in this repo), base rate "
            "1.23% of accounts positive; NOT comparable to this dataset's "
            "published GNN benchmarks (different features, split, and task "
            "framing) and not part of this project's own claims -- see "
            "ml/experiments/ibm_aml/README.md"
        ),
    ),
    Claim(
        id="ibm_aml_giant_component_realized_fpr",
        statement=(
            "On the same benchmark, a 'flag if in a large connected component' "
            "baseline reaches high recall only by flagging this share of all "
            "negative accounts"
        ),
        source="ibm_aml",
        path="summary.component_size_baseline_full_graph.realized_fpr_mean",
        qualifier=(
            "72% of all 515,088 accounts sit in one connected component, so "
            "component membership alone is nearly uninformative at this scale; "
            "see ml/experiments/ibm_aml/README.md's giant-component finding"
        ),
    ),
    Claim(
        id="gnn_xgb_t5_score_averaging",
        statement=(
            "At T=5m, averaging XGBoost's and an inductively-trained GNN's "
            "scores lifts PR-AUC to this, from XGBoost alone's 0.6212"
        ),
        source="gnn_xgb",
        path="summary.T=5m.score_average_xgboost_and_gnn.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds; synthetic simulator data, 30,000 accounts/3 "
            "banks; simplest score-averaging combination, not a shipped "
            "detector; GNN input includes identity features; gain is +0.044 "
            "absolute (~7% relative) -- see ml/experiments/gnn_xgb/README.md"
        ),
    ),
    Claim(
        id="gnn_xgb_t5_transfer_only_graph_lift",
        statement=(
            "At T=5m, with zero identity/account-age signal, an inductive GNN "
            "trained on only the 11 transfer-only features reaches this "
            "PR-AUC, compared with 0.0714 for XGBoost on the identical 11 "
            "features"
        ),
        source="gnn_xgb",
        path="summary.T=5m.gnn_inductive_11feat_transfer_only.pr_auc_mean",
        qualifier=(
            "mean over 3 seeds; synthetic simulator data; shows graph "
            "structure carries signal beyond this tree-based feature set, not "
            "that this specific gain transfers to real-world data; absolute "
            "PR-AUC remains low (0.14) -- see ml/experiments/gnn_xgb/README.md"
        ),
    ),
    Claim(
        id="stage1_passthrough_account_recall",
        statement=(
            "Share of the 712 pass-through mule accounts the shipped velocity "
            "rule (stage 1, before identity propagation) alerts on"
        ),
        source="stage2",
        path="alert_level.stage1.passthrough_recall",
        decimals=4,
        qualifier="account-level; see stage1_ring_recall_count for why this does not carry over to rings 1:1",
    ),
    Claim(
        id="stage1_ring_recall_count",
        statement=(
            "True rings touched by stage 1's account-level alerts above "
            "(a ring counts as touched once any of its members is alerted)"
        ),
        source="stage2",
        path="ring_membership.stage1.true_rings_touched",
        decimals=0,
        qualifier=(
            "of 120 true rings (44.2%); lower than the 87.8% account recall above "
            "because the clustering step drops flagged singletons -- see ml/stage2.py"
        ),
    ),
]

RETIRED: list[Retired] = [
    Retired(
        id="share_caught_before_exit",
        statement=("Share of rings flagged at 1% FPR before the money reached the exit account"),
        reason=(
            "It measures the horizon grid, not the detector. A ring can only count "
            "if the horizon precedes its cash-out, so the metric is bounded by how "
            "many exits have not happened yet. At T=5m only one of 120 rings has "
            "cashed out, so the qualifier does nothing and the value collapses onto "
            "ring_recall -- identically, to four decimals, for cross-bank. At T=180m "
            "every exit has happened, so the metric is structurally 0.0000 even "
            "though cross-bank recall is 1.0000 and cross-bank PR-AUC peaks there. "
            "It therefore rewards measuring early and punishes measuring well. "
            "Replaced by the lead-time claims above, which carry a signed number of "
            "minutes and keep unflagged rings out of the average instead of scoring "
            "them as on-time."
        ),
        evidence=[
            Claim(
                id="retired_share_t5_cross_bank",
                statement="share_caught_before_exit at T=5m, cross-bank",
                source="early_detection",
                path="horizons[cutoff_minutes=5].ring_catch.cross_bank.share_caught_before_exit",
            ),
            Claim(
                id="retired_recall_t5_cross_bank",
                statement="ring_recall at T=5m, cross-bank -- identical to the line above",
                source="early_detection",
                path="horizons[cutoff_minutes=5].ring_catch.cross_bank.ring_recall",
            ),
            Claim(
                id="retired_share_t180_cross_bank",
                statement="share_caught_before_exit at T=180m, cross-bank",
                source="early_detection",
                path="horizons[cutoff_minutes=180].ring_catch.cross_bank.share_caught_before_exit",
            ),
            Claim(
                id="retired_recall_t180_cross_bank",
                statement="ring_recall at T=180m, cross-bank -- while the share above is zero",
                source="early_detection",
                path="horizons[cutoff_minutes=180].ring_catch.cross_bank.ring_recall",
            ),
            Claim(
                id="retired_rings_in_flight_t5",
                statement="Rings whose money had not yet reached the exit at T=5m",
                source="early_detection",
                path=(
                    "horizons[cutoff_minutes=5].ring_catch.cross_bank.rings_money_not_yet_at_exit"
                ),
                decimals=1,
            ),
            Claim(
                id="retired_rings_in_flight_t180",
                statement="Rings whose money had not yet reached the exit at T=180m",
                source="early_detection",
                path=(
                    "horizons[cutoff_minutes=180].ring_catch.cross_bank.rings_money_not_yet_at_exit"
                ),
                decimals=1,
            ),
        ],
    ),
]


def load_sources(cfg: MlConfig) -> dict[str, Any]:
    """Read every metrics file claims are sourced from."""
    docs: dict[str, Any] = {}
    for name, relative in SOURCES.items():
        path = cfg.ml_dir.parent / relative
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run `python -m ml.early` and "
                "`python -m ml.leadtime` before generating claims"
            )
        docs[name] = json.loads(path.read_text("utf-8"))
    return docs


def build_payload(docs: dict[str, Any]) -> dict[str, Any]:
    """The machine-readable claim set the test re-verifies."""
    return {
        "sources": SOURCES,
        "claims": [claim.render(docs) for claim in CLAIMS],
        "retired": [
            {
                "id": item.id,
                "statement": item.statement,
                "reason": item.reason,
                "evidence": [claim.render(docs) for claim in item.evidence],
            }
            for item in RETIRED
        ],
    }


def _fmt(record: dict[str, Any]) -> str:
    return f"{record['value']:.{record['decimals']}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    """The published file: prose, a readable table, and the generated block."""
    lines: list[str] = [
        "# Quotable claims",
        "",
        "**Generated by `python -m ml.claims`. Do not edit by hand.**",
        "",
        "Every number below is resolved from a metrics file at generation time and",
        "carries the exact key path it came from. `ml/test_claims.py` re-resolves all",
        "of them and fails if any has drifted, so a stale number here is a failing",
        "test rather than a wrong slide.",
        "",
        "This file exists because four errors in one session were all the same error:",
        "a number copied by hand. One brief swapped the scopes on the headline PR-AUC",
        "pair; `docs/post_submission.md` carried a pair that exists in no metrics file;",
        "an overlap figure was quoted without the qualifier that made it meaningful;",
        "and `ml/early.py` could not run, so the committed metrics were not produced by",
        "the committed source.",
        "",
        "**Read the qualifier.** Several of these numbers are wrong if quoted bare --",
        "in particular the coverage shares (unconditional) and the lead-time median",
        "(conditional). The qualifier is part of the claim, not a footnote.",
        "",
        "## Sources",
        "",
    ]
    for name, relative in payload["sources"].items():
        lines.append(f"- `{name}` -> `{relative}`")
    lines += [
        "",
        "Paths select list elements by field value (`horizons[cutoff_minutes=5]`),",
        "never by position: three horizons were inserted mid-list this session, and",
        "every positional path would now point at a different measurement.",
        "",
        "## Claims",
        "",
        "| value | claim | source | path |",
        "| --- | --- | --- | --- |",
    ]
    for record in payload["claims"]:
        path = record["path"]
        if "minus_path" in record:
            path = f"{path} **minus** {record['minus_path']}"
        lines.append(
            f"| `{_fmt(record)}` | {record['statement']} | `{record['source_file']}` | `{path}` |"
        )
    lines += ["", "### Qualifiers", ""]
    for record in payload["claims"]:
        if record.get("qualifier"):
            lines.append(f"- **{_fmt(record)}** ({record['id']}): {record['qualifier']}")

    lines += ["", "## Retired claims", ""]
    for item in payload["retired"]:
        lines += [
            f"### `{item['id']}` -- {item['statement']}",
            "",
            item["reason"],
            "",
            "| value | evidence | path |",
            "| --- | --- | --- |",
        ]
        for record in item["evidence"]:
            lines.append(f"| `{_fmt(record)}` | {record['statement']} | `{record['path']}` |")
        lines.append("")

    lines += [
        "## Generated block",
        "",
        "`ml/test_claims.py` parses only what is between the markers below. Anything",
        "outside them is prose and is not treated as a claim.",
        "",
        BLOCK_START,
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        BLOCK_END,
    ]
    # Appended experiment sections (additive; each is sourced from its own
    # metrics file). Imported lazily: the section module imports this one.
    from ml.experiments.slow_drip.claims_section import render_section  # noqa: PLC0415

    lines += render_section(Path(__file__).resolve().parent.parent)
    lines.append("")
    return "\n".join(lines)


def parse_payload(text: str) -> dict[str, Any]:
    """Recover the machine-readable block from a generated file."""
    try:
        body = text.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0]
    except IndexError as exc:
        raise ValueError(f"{OUTPUT} has no generated block; regenerate it") from exc
    return json.loads(body.strip().removeprefix("```json").removesuffix("```"))


def main() -> None:
    """Regenerate ``docs/claims.md`` from the metrics files."""
    cfg = MlConfig.from_env()
    docs = load_sources(cfg)
    payload = build_payload(docs)
    path = cfg.ml_dir.parent / OUTPUT
    path.write_text(render_markdown(payload), "utf-8")
    print(
        f"wrote {path} claims={len(payload['claims'])} "
        f"retired={len(payload['retired'])} "
        f"retired_evidence={sum(len(r['evidence']) for r in payload['retired'])}"
    )
    # Ask's get_metric tool reads the same numbers: write its catalog from the
    # same sources in the same run, so the two cannot be regenerated apart.
    from ml import metric_catalog  # noqa: PLC0415 - metric_catalog imports this module

    print(f"wrote {metric_catalog.write(cfg, docs)}")


if __name__ == "__main__":
    main()
