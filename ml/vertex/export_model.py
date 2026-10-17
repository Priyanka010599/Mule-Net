"""Export the fitted boosters behind the published scores, for Model Registry.

Why this module exists
----------------------
``ml.early`` fits models and throws them away: it keeps their *pooled
out-of-fold predictions* (``ml/experiments/early_detection/oof/``) and
their metrics, and never writes a booster. That is the right design for
an evaluation -- but Vertex AI Model Registry registers an artefact, so
without something on disk there is nothing to register and the only
options left are registering a placeholder or not registering at all.

So this module re-fits **exactly** the folds that produced the published
cross-bank scores -- scope ``cb_``, full window, seed 42, which is what
``ml/out/manifest.json`` calls ``mulenet-xgboost-cb-oof-v1`` -- using
``ml.evaluate``'s own :data:`~ml.evaluate.XGB_PARAMS`,
:func:`~ml.evaluate.fold_assignment` and per-fold ``random_state``, and
saves the five boosters.

It is re-fitting, so the claim "these are the published model" has to be
checked rather than asserted: :func:`export` recomputes the out-of-fold
scores from the boosters it just saved and requires them **bit-identical**
to the stored OOF file that the published metrics were computed from. If
they differ the export raises, because a registry entry that does not
reproduce the published scores would be worse than no registry entry.

Five boosters are not one servable predictor. The registry entry is a
lineage and version record, not a deployable endpoint -- see
``ml/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.config import MlConfig
from ml.early import OOF_DIRNAME, horizon_tag
from ml.evaluate import (
    N_FOLDS,
    SCOPE_NAMES,
    SEEDS,
    XGB_PARAMS,
    attach_labels,
    fold_assignment,
)
from ml.features import build_features, feature_columns, full_window_cutoff, load_tables

#: The published scope, horizon and seed, each derived from its own source
#: of truth rather than written out again: ``cb_`` is the cross-bank
#: prefix, ``None`` is the full window in ``ml.early.CUTOFF_MINUTES``, and
#: the canonical seed is the first of ``ml.evaluate.SEEDS``.
SCOPE_PREFIX = "cb_"
HORIZON_MINUTES: int | None = None
SEED = SEEDS[0]


def _oof_path(cfg: MlConfig) -> Path:
    scope = SCOPE_NAMES[SCOPE_PREFIX]
    return (
        cfg.experiments_dir
        / "early_detection"
        / OOF_DIRNAME
        / f"{horizon_tag(HORIZON_MINUTES)}_{scope}_seed{SEED}.parquet"
    )


def export(cfg: MlConfig, out_dir: Path) -> dict[str, object]:
    """Fit and save the five canonical boosters; verify they reproduce the scores.

    Raises ``AssertionError`` if the re-fitted out-of-fold scores are not
    bit-identical to the stored ones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(cfg)
    frame = attach_labels(
        build_features(cfg, tables=tables, cutoff=full_window_cutoff(tables)), cfg
    )
    columns = feature_columns(SCOPE_PREFIX)
    x = frame[columns].to_numpy(dtype="float32")
    y = frame["is_passthrough_mule"].to_numpy(dtype="int64")
    fold = fold_assignment(frame, SEED)

    scores = np.zeros(len(frame), dtype="float64")
    files: list[str] = []
    for k in range(N_FOLDS):
        test = fold == k
        train = ~test
        n_pos = int(y[train].sum())
        n_neg = int(train.sum()) - n_pos
        model = XGBClassifier(
            **XGB_PARAMS,
            random_state=SEED * 100 + k,
            scale_pos_weight=n_neg / max(n_pos, 1),
        )
        model.fit(x[train], y[train], verbose=False)
        scores[test] = model.predict_proba(x[test])[:, 1]
        name = f"fold{k}.ubj"
        model.get_booster().save_model(str(out_dir / name))
        files.append(name)

    stored = pd.read_parquet(_oof_path(cfg))
    assert list(stored["hashed_account_id"]) == list(frame["hashed_account_id"]), (
        "row order differs from the stored out-of-fold file; the re-fit is not "
        "aligned with the published scores"
    )
    assert np.array_equal(stored["score"].to_numpy(), scores), (
        "re-fitted out-of-fold scores are not bit-identical to "
        f"{_oof_path(cfg).name}; refusing to register a model that does not "
        "reproduce the published scores"
    )

    metadata = {
        "model_version": "mulenet-xgboost-cb-oof-v1",
        "scope": SCOPE_NAMES[SCOPE_PREFIX],
        "horizon": horizon_tag(HORIZON_MINUTES),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "xgb_params": {k: str(v) for k, v in XGB_PARAMS.items()},
        "feature_columns": columns,
        "booster_files": files,
        "oof_scores_verified_against": _oof_path(cfg).name,
        "servable": False,
        "servable_note": (
            "five per-fold boosters whose pooled out-of-fold predictions are the "
            "published scores; there is no single fitted model to serve, so this "
            "is a lineage record and not a deployable predictor"
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    return metadata


def main() -> None:
    """CLI: export the canonical boosters next to the experiment they came from."""
    cfg = MlConfig.from_env()
    out_dir = cfg.experiments_dir / "early_detection" / "model"
    metadata = export(cfg, out_dir)
    print(f"wrote {len(metadata['booster_files'])} boosters + metadata.json to {out_dir}")


if __name__ == "__main__":
    main()
