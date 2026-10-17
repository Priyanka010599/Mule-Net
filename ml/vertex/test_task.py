"""Tests for the parts of the Vertex packaging that run without Google Cloud.

The cloud calls are not mocked. Mocking ``storage.Client`` would test the
mock, and the real evidence that the job works is the job's own output in
the bucket -- which the gate checks. What is worth testing here is the
plumbing that would silently ship the wrong thing: the input set, the URI
parsing, and what does or does not end up inside the image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.config import TABLE_FILES
from ml.vertex import task
from ml.vertex.submit import assemble_context


def test_split_uri_handles_prefix_and_bare_bucket() -> None:
    assert task._split_uri("gs://bucket/a/b") == ("bucket", "a/b")
    assert task._split_uri("gs://bucket") == ("bucket", "")
    assert task._split_uri("gs://bucket/a/b/") == ("bucket", "a/b")


def test_split_uri_rejects_a_local_path() -> None:
    """A local path reaching the uploader would silently write nothing."""
    with pytest.raises(ValueError, match="gs://"):
        task._split_uri("/tmp/not-a-bucket")


def test_build_context_carries_every_module_early_and_leadtime_import(tmp_path: Path) -> None:
    """The image must hold the modules the canonical run actually imports."""
    context = assemble_context(tmp_path / "ctx")
    shipped = {p.name for p in (context / "ml").rglob("*.py")}
    required = {
        "config.py",
        "early.py",
        "evaluate.py",
        "features.py",
        "leadtime.py",
        "metrics.py",
        "timeline.py",
        "__init__.py",
    }
    assert required <= shipped
    assert (context / "Dockerfile").exists()
    assert (context / "requirements-train.txt").exists()


def test_build_context_excludes_the_generated_artefacts(tmp_path: Path) -> None:
    """``experiments/`` alone is ~109 MB and the job regenerates all of it."""
    context = assemble_context(tmp_path / "ctx")
    paths = {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}
    assert not any(p.startswith("ml/experiments/") for p in paths)
    assert not any(p.startswith("ml/out/") for p in paths)
    assert not any(p.endswith(".parquet") for p in paths)
    assert not any("__pycache__" in p for p in paths)


def test_staged_input_set_is_the_six_contract_tables() -> None:
    """Derived from TABLE_FILES, so it cannot drift from what training reads."""
    assert len(TABLE_FILES) == 6
    assert sorted(TABLE_FILES.values()) == [
        "accounts.parquet",
        "devices.parquet",
        "ground_truth.parquet",
        "kyc_documents.parquet",
        "kyc_submissions.parquet",
        "transfers.parquet",
    ]
