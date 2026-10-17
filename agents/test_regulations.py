"""Cloud mode must never silently answer from the mock corpus.

Regression test for a real production bug: ``docai/`` was missing from
the agents Docker image, so ``_docai_search()`` always ``ImportError``ed
and ``search()`` fell through to the built-in CORPUS -- which still
returned a normal-looking result list, with no error anywhere, even
though the deployment had set ``MULENET_DOCAI_MODE=cloud``. The model
quoted US/FATF passages as if they were the real RBI/PMLA index for
hours before this was caught by reading logs by hand.
"""

from __future__ import annotations

import pytest

from agents import regulations


@pytest.fixture
def cloud_mode_requested(monkeypatch):
    """Set the env var that means "the operator expects the real index"."""
    monkeypatch.setenv("MULENET_DOCAI_MODE", "cloud")


def test_cloud_mode_raises_when_docai_is_not_importable(
    monkeypatch, cloud_mode_requested
) -> None:
    """The exact bug: docai missing from the image. No mock fallback."""
    monkeypatch.setattr(regulations, "_docai_search", lambda: None)

    with pytest.raises(regulations.RegulationSearchUnavailable):
        regulations.search("suspicious transaction reporting", 3)


def test_cloud_mode_raises_when_the_real_backend_raises(
    monkeypatch, cloud_mode_requested
) -> None:
    """docai is importable but the call itself fails -- still no mock."""

    def _broken_cloud_index(query: str, top_k: int) -> list[dict[str, str]]:
        raise RuntimeError("Vertex AI Search query failed: HTTP 403 PERMISSION_DENIED")

    monkeypatch.setattr(regulations, "_docai_search", lambda: _broken_cloud_index)

    with pytest.raises(regulations.RegulationSearchUnavailable):
        regulations.search("suspicious transaction reporting", 3)


def test_without_cloud_mode_the_mock_corpus_still_works(monkeypatch) -> None:
    """The offline/test path this module exists for is unaffected."""
    monkeypatch.delenv("MULENET_DOCAI_MODE", raising=False)
    monkeypatch.setattr(regulations, "_docai_search", lambda: None)

    results = regulations.search("rapid pass-through of funds", 1)
    assert results
    assert regulations.is_synthetic(results[0])
