"""Regression test for the Model C fetch-list gap.

Model C shipped (Lane D1) without its two files being added to
infra/deploy.sh's AGENTS_FETCH list -- the deploy would have silently
never served it to a live upload, falling back to the rule instead,
with no error anywhere. Caught only by inspection, after the fact.

This test exists so that never happens silently again: it derives the
expected model file list directly from ml.model_runtime.ALL_MODELS,
the same single source of truth infra/agents_fetch.py reads, so a
model added there without reaching the fetch list fails here --
whether it is model_c or whatever comes after it.
"""

from __future__ import annotations

import pytest

from infra.agents_fetch import build_agents_fetch, metric_objects, model_objects, race_objects
from ml.model_runtime import ALL_MODELS


def test_every_named_model_is_in_the_agents_fetch_list():
    fetch = build_agents_fetch()
    for name in ALL_MODELS:
        assert f"ml/out/models/{name}.ubj" in fetch, f"{name}.ubj missing from AGENTS_FETCH"
        assert (
            f"ml/out/models/{name}_metadata.json" in fetch
        ), f"{name}_metadata.json missing from AGENTS_FETCH"


def test_every_named_model_file_exists_on_disk():
    """model_objects() itself checks existence; this pins that it currently passes."""
    objects = model_objects()
    assert len(objects) == len(ALL_MODELS) * 2


def test_missing_model_file_fails_loudly_naming_the_file(monkeypatch):
    """A model listed in ALL_MODELS but missing from disk must stop the
    build with an error naming its exact path -- never silently proceed
    without it, and never substitute a different model's files."""
    monkeypatch.setattr("ml.model_runtime.ALL_MODELS", ("model_a", "model_does_not_exist"))
    with pytest.raises(SystemExit) as exc_info:
        model_objects()
    message = str(exc_info.value)
    assert "ml/out/models/model_does_not_exist.ubj" in message
    assert "ml/out/models/model_does_not_exist_metadata.json" in message
    assert "model_a.ubj" not in message


def test_the_ring_race_artifact_is_in_the_agents_fetch_list():
    """Ask's get_ring_race reads agents/out/ring_race.json; it must be fetched."""
    assert "agents/out/ring_race.json" in build_agents_fetch()


def test_a_missing_ring_race_artifact_fails_loudly_naming_the_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        race_objects()
    message = str(exc_info.value)
    assert "agents/out/ring_race.json" in message
    assert "python -m ui.build_snapshot" in message


def test_the_metric_catalog_is_in_the_agents_fetch_list():
    """Ask's get_metric reads agents/out/metric_catalog.json; it must be fetched."""
    assert "agents/out/metric_catalog.json" in build_agents_fetch()


def test_a_missing_metric_catalog_fails_loudly_naming_the_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        metric_objects()
    message = str(exc_info.value)
    assert "agents/out/metric_catalog.json" in message
    assert "python -m ml.claims" in message
