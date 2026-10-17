"""Identity-feature ablation on Model A's recipe (Lane D1).

Standalone, offline-only. Never touches ml/out/models (the shipped
artifacts), the live app, Vertex Model Registry, or Gemini. Reuses
ml.evaluate.run_scope, ml.evaluate.XGB_PARAMS and ml.early.cutoff_series
UNMODIFIED -- the same grouped 5-fold x 3-seed harness, same
hyperparameters, same seed base, that produced the shipped
validated_performance. Only the feature set passed to run_scope changes.

Runs (names match the committed recipe's own vocabulary):

  all16                  every cb_ feature (reproduces shipped Model A)
  drop_device_count      all16 minus device_count
  drop_shared_device_count
  drop_document_reuse_count
  drop_document_reuse_flag
  drop_device_pair       all16 minus BOTH device features together
  drop_document_pair     all16 minus BOTH document features together
  drop_all_identity      all16 minus all 4 identity features (= Model B's 12)
  identity_only          ONLY the 4 identity features
  drop_all_identity_and_age
                         Model B's 12 minus account_age_days (11 features) --
                         account_age_days is an account attribute, not
                         transfer-derived, so this asks how much of
                         "transfer-only" leans on it specifically.

Gate: all16 must reproduce ~0.622 PR-AUC and ~1.97 false alerts per
catch at T=5m (ml/out/models/model_a_metadata.json's
validated_performance). If it doesn't, this script stops and says so --
nothing past the gate is trusted without it.

Per-seed checkpointing: each seed's full set of runs (both horizons, all
10 feature configs) is written to results_seed{N}.json as soon as that
seed finishes, so a crash loses at most one seed.

Usage::

    python ml/experiments/ablation/run_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import MlConfig
from ml.early import cutoff_series, ring_onsets
from ml.evaluate import SEEDS, attach_labels, run_scope
from ml.features import build_features, load_tables

HERE = Path(__file__).parent
OUT_PATH = HERE / "results_ablation.json"
MODEL_A_METADATA = Path("ml/out/models/model_a_metadata.json")

HORIZONS: tuple[int, ...] = (5, 180)

IDENTITY = ("device_count", "shared_device_count", "document_reuse_count", "document_reuse_flag")
TRANSFER_ONLY = (
    "in_degree", "out_degree", "unique_in_counterparties", "unique_out_counterparties",
    "unique_counterparties", "distinct_counterparty_banks", "amount_in_total",
    "amount_out_total", "passthrough_ratio", "passthrough_speed_seconds",
    "round_amount_ratio", "account_age_days",
)

#: name -> suffixes to drop (ml.evaluate.run_scope's own drop semantics:
#: suffixes without the "cb_" prefix).
RUN_DROPS: dict[str, tuple[str, ...]] = {
    "all16": (),
    "drop_device_count": ("device_count",),
    "drop_shared_device_count": ("shared_device_count",),
    "drop_document_reuse_count": ("document_reuse_count",),
    "drop_document_reuse_flag": ("document_reuse_flag",),
    "drop_device_pair": ("device_count", "shared_device_count"),
    "drop_document_pair": ("document_reuse_count", "document_reuse_flag"),
    "drop_all_identity": IDENTITY,
    "identity_only": TRANSFER_ONLY,
    "drop_all_identity_and_age": IDENTITY + ("account_age_days",),
}


def check_gate(runs_t5m: list[dict[str, Any]]) -> None:
    shipped = json.loads(MODEL_A_METADATA.read_text())["validated_performance"]["T=5m"]
    all16 = [r for r in runs_t5m if r["_ablation_name"] == "all16"]
    pr_auc = float(np.mean([r["pr_auc"] for r in all16]))
    fa = float(np.mean([r["false_alerts_per_catch_at_80pct_recall"] for r in all16]))
    shipped_pr = shipped["pr_auc_mean"]
    shipped_fa = shipped["false_alerts_per_catch_at_80pct_recall_mean"]
    print(f"GATE: all16 PR-AUC={pr_auc:.4f} (shipped {shipped_pr:.4f}), "
          f"false_alerts/catch={fa:.3f} (shipped {shipped_fa:.3f})")
    if abs(pr_auc - shipped_pr) > 0.02 or abs(fa - shipped_fa) > 0.3:
        raise SystemExit(
            f"GATE FAILED: all16 does not reproduce shipped Model A closely enough "
            f"(PR-AUC diff {pr_auc - shipped_pr:+.4f}, false_alerts diff {fa - shipped_fa:+.3f}). "
            "Stopping -- nothing past this point is trusted."
        )
    print("GATE PASSED.")


def device_doc_base_rates(cfg: MlConfig) -> dict[str, Any]:
    """Full 30,000 accounts, full window: device/document reuse rates by ring status and span."""
    tables = load_tables(cfg)
    frame = attach_labels(build_features(cfg, tables=tables), cfg)
    frame["has_shared_device"] = frame["cb_shared_device_count"] > 0
    frame["has_multi_device"] = frame["cb_device_count"] > 1
    frame["has_document_reuse"] = frame["cb_document_reuse_flag"] == 1

    def rates(sub) -> dict[str, float]:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        return {
            "n": int(n),
            "share_device_count_gt1": float(sub["has_multi_device"].mean()),
            "share_shared_device_count_gt0": float(sub["has_shared_device"].mean()),
            "share_any_document_reuse": float(sub["has_document_reuse"].mean()),
        }

    out: dict[str, Any] = {
        "ring_members": rates(frame.loc[frame["ring_id"] != ""]),
        "non_ring": rates(frame.loc[frame["ring_id"] == ""]),
    }
    for span in (0, 1, 2, 3):
        out[f"bank_span_{span}"] = rates(frame.loc[frame["ring_bank_count"] == span])
    return out


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    keys = ["pr_auc", "recall_at_fpr_1pct", "false_alerts_per_catch_at_80pct_recall"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for horizon_label in sorted({r["_horizon"] for r in runs}):
        out[horizon_label] = {}
        for name in RUN_DROPS:
            rows = [r for r in runs if r["_horizon"] == horizon_label and r["_ablation_name"] == name]
            if not rows:
                continue
            entry: dict[str, float] = {"n_features": rows[0]["n_features"]}
            for k in keys:
                values = np.array([r[k] for r in rows], dtype="float64")
                entry[f"{k}_mean"] = float(values.mean())
                entry[f"{k}_std"] = float(values.std(ddof=0))
            out[horizon_label][name] = entry
    return out


def main() -> None:
    cfg = MlConfig.from_env()
    tables = load_tables(cfg)
    onsets = ring_onsets(tables["transfers"], tables["ground_truth"])

    base_rates = device_doc_base_rates(cfg)
    print("base rates:", json.dumps(base_rates, indent=2))

    all_runs: list[dict[str, Any]] = []
    gate_checked = False
    for seed in SEEDS:
        seed_path = HERE / f"results_seed{seed}.json"
        if seed_path.exists():
            print(f"seed={seed}: reusing checkpoint {seed_path.name}", flush=True)
            seed_runs = json.loads(seed_path.read_text())
            all_runs.extend(seed_runs)
            if not gate_checked:
                check_gate([r for r in seed_runs if r["_horizon"] == "T=5m"])
                gate_checked = True
            continue

        print(f"seed={seed}", flush=True)
        seed_runs: list[dict[str, Any]] = []
        for minutes in HORIZONS:
            horizon_label = f"T={minutes}m"
            cutoff = cutoff_series(tables, onsets, minutes, seed)
            frame = attach_labels(build_features(cfg, tables=tables, cutoff=cutoff), cfg)
            for name, drop in RUN_DROPS.items():
                run = run_scope(frame, "cb_", seed, drop=drop)
                run["_ablation_name"] = name
                run["_horizon"] = horizon_label
                run.pop("_scores", None)
                run.pop("per_pattern_recall_at_fpr_1pct", None)
                run.pop("strata", None)
                seed_runs.append(run)
                print(
                    f"  {horizon_label} {name}: n_features={run['n_features']} "
                    f"pr_auc={run['pr_auc']:.4f} r@1%={run['recall_at_fpr_1pct']:.4f} "
                    f"fa/catch={run['false_alerts_per_catch_at_80pct_recall']:.3f}",
                    flush=True,
                )

        if not gate_checked:
            check_gate([r for r in seed_runs if r["_horizon"] == "T=5m"])
            gate_checked = True

        seed_path.write_text(json.dumps(seed_runs, indent=2, default=float))
        print(f"  wrote {seed_path}", flush=True)
        all_runs.extend(seed_runs)

    results = {
        "recipe": "ml.evaluate.XGB_PARAMS, ml.evaluate.run_scope, unmodified",
        "seeds": list(SEEDS),
        "horizons": {"T=5m": 5, "T=180m": 180},
        "identity_features": IDENTITY,
        "transfer_only_features": TRANSFER_ONLY,
        "run_drops": {k: list(v) for k, v in RUN_DROPS.items()},
        "base_rates_full_30000_accounts": base_rates,
        "summary": summarise(all_runs),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
