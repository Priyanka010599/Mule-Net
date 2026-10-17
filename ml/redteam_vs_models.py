"""Score ml/redteam.py's cached evasion strategies against Model A and Model B
too, not only the velocity rule.

``ml/redteam.py`` evaluates each Gemini-designed strategy against
``ml.rule.velocity_decision`` alone. This module reuses the same
synthesize()/as_tables()/build_features() pipeline -- so the feature table
scored here is byte-identical to the one the velocity rule saw -- and adds
Model A and Model B (``ml.model_runtime``) as two more detectors on the same
graphs.

Strategies are replayed from the cached ``ui/data/redteam_fallback.json``
(each verdict there carries its full ``pattern`` dict) rather than calling
Gemini again: a fresh Gemini call is not reproducible against the one
velocity-rule row it would be compared to, and this module's whole point is
scoring the three detectors on the *same* synthetic graphs.

Usage::

    python -m ml.redteam_vs_models
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.config import MlConfig
from ml.features import build_features
from ml.model_runtime import MODEL_A, MODEL_B, LoadedModel, load_model, score
from ml.redteam import EvasionStrategy, as_tables, parse_strategies, synthesize

FALLBACK_PATH = Path("ui/data/redteam_fallback.json")
MODELS_DIR = Path("ml/out/models")
OUT_PATH = Path("ml/experiments/redteam/model_comparison.json")


def _load_cached_strategies() -> tuple[list[EvasionStrategy], list[dict[str, Any]]]:
    """Replay the 3 strategies ml/redteam.py's own run already scored.

    The fallback's key is ``aimed_at_clause`` (what ``as_json`` renames the
    field to for the report); ``parse_strategies`` reads it back under its
    original name, ``evades_clause``.
    """
    verdicts = json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))["verdicts"]
    raw = []
    for item in verdicts:
        item = dict(item)
        item["evades_clause"] = item.get("aimed_at_clause", "unstated")
        raw.append(item)
    return parse_strategies(raw), verdicts


def _caught_by(model: LoadedModel, features: Any, collectors: list[str]) -> bool:
    """A model catches a strategy when it scores any collector above its
    own published 1%-FPR threshold -- the same bar ``agents/subset.py`` and
    ``agents/subset_upload.py`` use to decide ``detector == "model_a"/"model_b"``.
    """
    scores = score(model, features)
    if scores is None:
        return False
    flagged = set(features.loc[scores >= model.metadata["threshold_at_1pct_fpr"], "hashed_account_id"])
    return bool(flagged & set(collectors))


def run(cfg: MlConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MlConfig.from_env()
    strategies, velocity_verdicts = _load_cached_strategies()

    model_a = load_model(MODELS_DIR, MODEL_A)
    model_b = load_model(MODELS_DIR, MODEL_B)
    if model_a is None or model_b is None:
        raise RuntimeError(f"could not load model_a/model_b from {MODELS_DIR}")

    rows = []
    for index, strategy in enumerate(strategies, start=1):
        graph, collectors = synthesize(strategy, index)
        features = build_features(cfg=cfg, tables=as_tables(graph))
        rows.append(
            {
                "name": strategy.name,
                "evades_clause": strategy.evades_clause,
                "velocity_caught": bool(velocity_verdicts[index - 1]["caught"]),
                "model_a_caught": _caught_by(model_a, features, collectors),
                "model_b_caught": _caught_by(model_b, features, collectors),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "cached strategies replayed from ui/data/redteam_fallback.json, not a fresh Gemini call",
        "caveats": [
            "Isolated graph: each strategy's feature table holds only that "
            "strategy's own synthetic accounts and transfers, same as "
            "ml/redteam.py. A detector trained or tuned expecting a busy "
            "background graph may behave differently in production.",
            "The attacker knows the detector's exact thresholds: Gemini was "
            "given the velocity rule's five clauses verbatim and told to "
            "beat them. Model A/B were not targeted directly -- but they "
            "are trained on the same cb_ features those clauses read, so a "
            "pattern built to clear one tends to clear the other.",
            "Strategies are replayed, not regenerated: this is the same 3 "
            "patterns ml/redteam.py already scored, not a fresh adversarial "
            "search against Model A/B specifically.",
        ],
        "strategies": rows,
    }


def main() -> int:
    payload = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{'strategy':<32} {'clause':<24} {'velocity':<10} {'model_a':<10} {'model_b':<10}")
    for row in payload["strategies"]:
        print(
            f"{row['name']:<32} {row['evades_clause']:<24} "
            f"{'CAUGHT' if row['velocity_caught'] else 'missed':<10} "
            f"{'CAUGHT' if row['model_a_caught'] else 'missed':<10} "
            f"{'CAUGHT' if row['model_b_caught'] else 'missed':<10}"
        )
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
