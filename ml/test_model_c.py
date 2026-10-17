"""Tests for Model C: Model B's 12 features minus account_age_days.

See ml/train_model_c.py for the training recipe and
agents/test_subset_upload.py for the upload-path integration tests
(never reads account age; identical scores for two different opened_at
placeholders).
"""

from __future__ import annotations

import pytest

from ml.config import MlConfig
from ml.model_runtime import MODEL_B, MODEL_C, load_model

REAL_MODELS_DIR = MlConfig.from_env().ml_dir / "out" / "models"


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_C}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_model_c_features_are_model_b_minus_account_age():
    model_b = load_model(REAL_MODELS_DIR, MODEL_B)
    model_c = load_model(REAL_MODELS_DIR, MODEL_C)
    assert model_b is not None
    assert model_c is not None
    assert set(model_c.feature_columns) == set(model_b.feature_columns) - {"cb_account_age_days"}
    assert len(model_c.feature_columns) == 11
    assert "cb_account_age_days" not in model_c.feature_columns
