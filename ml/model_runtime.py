"""Load and score the three final, servable production models.

Distinct from ``ml/evaluate.py`` (grouped-CV, train-and-discard, the
evaluation harness) and ``ml/vertex/export_model.py`` (a *lineage* record
of five per-fold boosters, explicitly ``servable: false``). Every
artifact this module loads was fit **once**, on 100% of the labeled
data, specifically to be served:

* ``model_a`` -- all 16 ``cb_`` features, including device/KYC identity
  reuse. Serves ``agents/subset.py`` and ``ml/coalition.py``, both of
  which read the real published ``devices``/``kyc_submissions``/
  ``kyc_documents`` tables.
* ``model_b`` -- the 12 transfer-graph-derived features only, **never
  trained on identity features at all** (not a zeroed-out Model A).
  Still serves ``ml/redteam_vs_models.py``'s own comparison; no longer
  serves live uploads as of ``model_c`` below.
* ``model_c`` -- ``model_b``'s 12 minus ``account_age_days`` (11
  features). Serves ``agents/subset_upload.py`` instead of ``model_b``
  as of this lane: a judge's upload has no account-open date, so
  ``account_age_days`` was a constant placeholder there, not a real
  feature -- see ``ml/train_model_c.py`` and
  ``ml/experiments/ablation/README.md``.

Neither artifact can be honestly re-scored on its own training data
without leakage -- their expected performance is the grouped 5-fold CV
x 3-seed estimate from a sibling model built the identical way, recorded
in each model's ``*_metadata.json`` rather than re-derived at load time.

Every function here fails closed to ``None``, never raises past its own
boundary: a corrupt file, a missing path, or a scoring error must let the
caller fall back to ``ml.rule.velocity_decision``, not crash a judge's
request. See ``agents/subset.py``, ``agents/subset_upload.py`` and
``ml/coalition.py`` for the fallback wiring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("mulenet.model_runtime")

#: Names this module knows how to load. Matches the file stems written by
#: the training script that produced ml/out/models/{name}.ubj and
#: {name}_metadata.json.
MODEL_A = "model_a"
MODEL_B = "model_b"
MODEL_C = "model_c"

#: Every name this module knows how to load, in one place. The single
#: source of truth infra/agents_fetch.py reads so that adding a model
#: here without updating the deploy script's fetch list fails a test
#: (infra/test_agents_fetch.py) instead of silently never reaching the
#: deployed service -- see that module's docstring.
ALL_MODELS: tuple[str, ...] = (MODEL_A, MODEL_B, MODEL_C)


@dataclass(frozen=True)
class LoadedModel:
    """A loaded booster plus the metadata that makes its score meaningful.

    ``feature_columns`` is load-bearing, not descriptive: :func:`score`
    selects exactly these columns, in this order, from whatever frame it
    is given, so a model can never silently score on a column it was not
    trained on.
    """

    name: str
    booster: Any  # xgboost.Booster, typed Any so this module does not
    # require xgboost at import time for callers that only check availability
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]


def load_model(models_dir: Path, name: str) -> LoadedModel | None:
    """Load one named model. Returns ``None`` on any failure, never raises.

    A missing file, a corrupt booster, or a missing/malformed metadata
    sidecar are all the same case from a caller's point of view: there is
    no usable model, fall back to the rule. The specific reason is logged,
    not swallowed silently.
    """
    booster_path = models_dir / f"{name}.ubj"
    metadata_path = models_dir / f"{name}_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        columns = tuple(metadata["feature_columns"])
        if not columns:
            raise ValueError("metadata has no feature_columns")

        import xgboost

        booster = xgboost.Booster()
        booster.load_model(str(booster_path))
    except Exception as exc:  # noqa: BLE001 - any failure here means "no model"
        LOG.warning("could not load model %r from %s: %s", name, models_dir, exc)
        return None
    return LoadedModel(name=name, booster=booster, feature_columns=columns, metadata=metadata)


def score(model: LoadedModel, frame: pd.DataFrame) -> np.ndarray | None:
    """Score every row of ``frame`` with ``model``. Returns ``None`` on any failure.

    Missing columns, an empty frame, or a dtype XGBoost rejects all result
    in ``None`` rather than a raised exception -- the caller's job is to
    fall back to the rule, not to interpret a traceback.
    """
    try:
        import xgboost

        missing = [c for c in model.feature_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing columns the model needs: {missing}")
        x = frame[list(model.feature_columns)].to_numpy(dtype="float32")
        dmatrix = xgboost.DMatrix(x, feature_names=list(model.feature_columns))
        return np.asarray(model.booster.predict(dmatrix), dtype="float64")
    except Exception as exc:  # noqa: BLE001 - any failure here means "could not score"
        LOG.warning("model %r failed to score %d row(s): %s", model.name, len(frame), exc)
        return None
