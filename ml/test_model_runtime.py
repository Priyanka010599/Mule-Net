"""Tests for ml/model_runtime.py -- loading and scoring Model A/B.

Every failure mode must return None, never raise: this module's whole
job is to let agents/subset.py, agents/subset_upload.py and
ml/coalition.py fall back to the rule cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.config import MlConfig
from ml.model_runtime import MODEL_A, MODEL_B, load_model, score

REAL_MODELS_DIR = MlConfig.from_env().ml_dir / "out" / "models"


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_A}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_load_model_a_succeeds():
    model = load_model(REAL_MODELS_DIR, MODEL_A)
    assert model is not None
    assert len(model.feature_columns) == 16
    assert "threshold_at_1pct_fpr" in model.metadata


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_B}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_load_model_b_succeeds_and_has_fewer_features_than_a():
    model_a = load_model(REAL_MODELS_DIR, MODEL_A)
    model_b = load_model(REAL_MODELS_DIR, MODEL_B)
    assert model_b is not None
    assert len(model_b.feature_columns) == 12
    identity_cols = {
        "cb_device_count",
        "cb_shared_device_count",
        "cb_document_reuse_count",
        "cb_document_reuse_flag",
    }
    assert not identity_cols & set(model_b.feature_columns)
    assert identity_cols & set(model_a.feature_columns)


def test_load_model_missing_file_returns_none(tmp_path: Path):
    assert load_model(tmp_path, "no_such_model") is None


def test_load_model_corrupt_metadata_returns_none(tmp_path: Path):
    (tmp_path / "broken_metadata.json").write_text("not json", encoding="utf-8")
    (tmp_path / "broken.ubj").write_bytes(b"")
    assert load_model(tmp_path, "broken") is None


def test_load_model_metadata_missing_feature_columns_returns_none(tmp_path: Path):
    (tmp_path / "bad_metadata.json").write_text("{}", encoding="utf-8")
    assert load_model(tmp_path, "bad") is None


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_A}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_score_succeeds_on_a_real_frame():
    model = load_model(REAL_MODELS_DIR, MODEL_A)
    frame = pd.read_parquet(MlConfig.from_env().features_path).head(3)
    scores = score(model, frame)
    assert scores is not None
    assert len(scores) == 3
    assert ((scores >= 0) & (scores <= 1)).all()


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_A}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_score_missing_columns_returns_none():
    model = load_model(REAL_MODELS_DIR, MODEL_A)
    frame = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    assert score(model, frame) is None


@pytest.mark.skipif(
    not (REAL_MODELS_DIR / f"{MODEL_A}.ubj").exists(),
    reason="trained model artifacts not present in this environment",
)
def test_score_empty_frame_does_not_raise():
    model = load_model(REAL_MODELS_DIR, MODEL_A)
    empty = pd.DataFrame({c: [] for c in model.feature_columns})
    result = score(model, empty)
    assert result is None or len(result) == 0
