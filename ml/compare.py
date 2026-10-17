"""Unified single-bank vs cross-bank table across every experiment.

Globs ``ml/experiments/**/metrics.json`` — nothing here knows about
XGBoost specifically. Any future run (GraphSAGE, a retuned baseline, an
ablation) that writes the same JSON shape into its own subdirectory
appears in the table without a code change.

Expected shape, which ``baseline.py`` writes::

    {"model": "<name>",
     "summary": {"<scope>": {"<metric>_mean": ..., "<metric>_std": ...}},
     "runs": [{"scope": ..., "seed": ...}, ...]}

where ``<metric>`` ranges over ``ml.metrics.METRIC_KEYS``.

Files missing ``summary`` are reported and skipped rather than crashing
the table, so one half-written experiment cannot hide the others.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.config import MlConfig

#: Short headers for the metric keys, in report order.
METRIC_HEADERS: dict[str, str] = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "precision_at_k": "P@k",
    "recall_at_fpr_0.1pct": "R@0.1%FPR",
    "recall_at_fpr_1pct": "R@1%FPR",
    "false_alerts_per_catch_at_80pct_recall": "FP/TP@80%R",
    # Hard-decision metrics, reported by the compound rule only.
    "precision": "precision",
    "recall": "recall",
    "fpr": "FPR",
}

#: Metrics rendered per scope, as (summary mean key, summary std key, header).
METRIC_COLUMNS: tuple[tuple[str, str, str], ...] = tuple(
    (f"{key}_mean", f"{key}_std", header) for key, header in METRIC_HEADERS.items()
)

#: Scope order in the table; anything else found is appended after these.
PREFERRED_SCOPES: tuple[str, ...] = ("single_bank", "cross_bank")


def find_metrics_files(experiments_dir: Path) -> list[Path]:
    """Every ``metrics.json`` under the experiments root, sorted by path."""
    return sorted(experiments_dir.glob("**/metrics.json"))


def load_metrics(experiments_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    """Load usable metrics documents, plus a list of problems to report."""
    documents: list[dict[str, object]] = []
    problems: list[str] = []
    for path in find_metrics_files(experiments_dir):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        if not isinstance(document, dict) or not isinstance(document.get("summary"), dict):
            problems.append(f"{path}: no 'summary' object, skipped")
            continue
        document.setdefault("model", path.parent.name)
        document["_path"] = str(path)
        documents.append(document)
    return documents, problems


def _ordered_scopes(documents: list[dict[str, object]]) -> list[str]:
    found: list[str] = []
    for document in documents:
        for scope in document["summary"]:  # type: ignore[index]
            if scope not in found:
                found.append(scope)
    ranked = [s for s in PREFERRED_SCOPES if s in found]
    return ranked + [s for s in found if s not in ranked]


def _cell(summary: dict[str, object], mean_key: str, std_key: str) -> str:
    if mean_key not in summary:
        return "-"
    mean = float(summary[mean_key])  # type: ignore[arg-type]
    std = float(summary.get(std_key, 0.0))  # type: ignore[arg-type]
    return f"{mean:.4f}+/-{std:.4f}"


def render(documents: list[dict[str, object]]) -> str:
    """Render one row per (experiment, scope)."""
    if not documents:
        return "no metrics found"

    scopes = _ordered_scopes(documents)
    header = ["experiment", "scope", "seeds"] + [h for _, _, h in METRIC_COLUMNS]
    rows: list[list[str]] = []
    for document in sorted(documents, key=lambda d: str(d["model"])):
        summary: dict[str, object] = document["summary"]  # type: ignore[assignment]
        runs = document.get("runs", [])
        for scope in scopes:
            if scope not in summary:
                continue
            n_seeds = sum(1 for r in runs if isinstance(r, dict) and r.get("scope") == scope)
            row = [str(document["model"]), scope, str(n_seeds) if n_seeds else "-"]
            row += [_cell(summary[scope], m, s) for m, s, _ in METRIC_COLUMNS]  # type: ignore[arg-type]
            rows.append(row)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(lines)


def main() -> None:
    """Print the unified comparison table."""
    cfg = MlConfig.from_env()
    documents, problems = load_metrics(cfg.experiments_dir)
    print(f"experiments root: {cfg.experiments_dir}")
    print(render(documents))
    for problem in problems:
        print(f"WARNING {problem}")


if __name__ == "__main__":
    main()
