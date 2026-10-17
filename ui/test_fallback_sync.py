"""The bundled fallbacks must be byte-equal copies of ``ml/``'s metrics.

``ui/data/metrics_fallback.json`` and ``ui/data/lead_time_fallback.json``
are committed, and on Cloud Run they are what the app actually serves:
``infra/cloudbuild.ignore`` excludes ``ml/experiments/`` from the image,
so the configured live paths do not exist in the container and
``ui/datasource.py`` falls back to these copies. A stale fallback
therefore does not degrade the deployed page -- it silently publishes
different numbers from the ones ``ml/`` measured.

``ui/build_snapshot.py`` refreshes both by ``shutil.copyfile`` from the
same two sources. This test asserts that copy is current. The pairs are
read from ``ui/config.py`` rather than retyped, so a path rename there
cannot leave this test checking a file nobody serves.

Missing sources are handled with ``ml/conftest.py``'s existing
convention: skip on a fresh clone where ``ml/`` has never run, and fail
when ``MULENET_REQUIRE_OOF=1`` says a missing artefact means the
pipeline did not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.conftest import require_artefact
from ui.config import (
    DEFAULT_COALITION_FALLBACK,
    DEFAULT_COALITION_PATH,
    DEFAULT_LEAD_METRICS_FALLBACK,
    DEFAULT_LEAD_METRICS_PATH,
    DEFAULT_METRICS_FALLBACK,
    DEFAULT_METRICS_PATH,
    DEFAULT_REDTEAM_FALLBACK,
    DEFAULT_REDTEAM_PATH,
)

#: (source, fallback) exactly as ``ui/build_snapshot.py`` copies them.
FALLBACK_PAIRS: tuple[tuple[Path, Path], ...] = (
    (DEFAULT_METRICS_PATH, DEFAULT_METRICS_FALLBACK),
    (DEFAULT_LEAD_METRICS_PATH, DEFAULT_LEAD_METRICS_FALLBACK),
    (DEFAULT_COALITION_PATH, DEFAULT_COALITION_FALLBACK),
    (DEFAULT_REDTEAM_PATH, DEFAULT_REDTEAM_FALLBACK),
)


@pytest.mark.parametrize(
    ("source", "fallback"),
    FALLBACK_PAIRS,
    ids=[fallback.name for _, fallback in FALLBACK_PAIRS],
)
def test_fallback_matches_ml_source(source: Path, fallback: Path) -> None:
    """The shipped copy must equal the ``ml/experiments/`` file it mirrors."""
    require_artefact(source, "python -m ui.build_snapshot")
    assert fallback.exists(), (
        f"{fallback} is committed and must exist; "
        "regenerate with `python -m ui.build_snapshot`."
    )
    assert json.loads(fallback.read_text(encoding="utf-8")) == json.loads(
        source.read_text(encoding="utf-8")
    ), (
        f"{fallback.name} is stale against {source}. Cloud Run serves the "
        "fallback, so this would publish numbers ml/ never measured. "
        "Refresh with `python -m ui.build_snapshot`."
    )
