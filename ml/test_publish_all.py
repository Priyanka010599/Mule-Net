"""``ml/out/`` must be regenerated in an order that leaves a whole manifest.

``ml.publish`` writes ``manifest.json`` wholesale; ``ml.timeline``
appends its table to it. Run the wrong way round, or run ``ml.publish``
alone, and the manifest describes five tables while six sit on disk --
with no error anywhere. ``ml.publish_all`` fixes the sequence and these
tests hold it in place.

Why the orchestrator is not executed here
-----------------------------------------
Running ``ml.publish_all.main()`` for real would rewrite the five
published tables. Two of them carry ``generated_at`` from
``datetime.now(UTC)``, so a re-run **cannot** reproduce them: the column
changes by construction. Those files are already loaded into GCS and
BigQuery and the deployed UI reads them, which makes an avoidable
rewrite strictly worse than the ordering bug being fixed.

So the ordering property is tested directly -- with both steps stubbed,
recording the sequence -- and the *consistency* property it exists to
protect is tested against the artefacts actually on disk. Neither writes
anything. What is deliberately **not** covered is whether the two
``main()`` functions succeed when run back to back for real; that is
stated in the handover rather than papered over with a test that does
not do it.
"""

from __future__ import annotations

import json

import pytest

import ml.publish
import ml.publish_all
import ml.timeline
from ml.config import MlConfig
from ml.conftest import require_artefact


def test_steps_run_in_declared_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the module: publish first, then the appenders.

    **Every** step is stubbed, by iterating :data:`ml.publish_all.STEPS`
    rather than naming modules. Naming them is how this test briefly
    wrote to the real ``ml/out/``: a third step was added to ``STEPS``,
    the two hardcoded ``setattr`` calls did not cover it, and running
    ``main()`` executed it for real against the published directory.
    Deriving the stubs from ``STEPS`` makes that impossible to
    reintroduce by adding a step.
    """
    calls: list[str] = []
    for step in ml.publish_all.STEPS:
        monkeypatch.setattr(step, "main", lambda name=step.__name__: calls.append(name))

    ml.publish_all.main()

    assert calls == [step.__name__ for step in ml.publish_all.STEPS]
    assert calls[0] == "ml.publish", "publish truncates the manifest; it must be first"
    assert len(calls) == len(ml.publish_all.STEPS)


def test_every_published_table_is_in_the_manifest() -> None:
    """The failure the ordering bug produces, asserted against real files.

    If someone runs ``ml.publish`` alone after this, the timeline parquet
    stays on disk and its manifest entry vanishes -- and this fails,
    which is the loud error the two-writer arrangement otherwise lacks.
    """
    cfg = MlConfig.from_env()
    out = cfg.ml_dir / "out"
    manifest_path = out / "manifest.json"
    require_artefact(manifest_path, "python -m ml.publish_all")

    manifest = json.loads(manifest_path.read_text("utf-8"))
    # scores_bq is a BigQuery-shaped variant of scores written outside
    # ml.publish, so it is not one of the manifest's tables.
    on_disk = {p.stem for p in out.glob("*.parquet")} - {"scores_bq"}
    assert on_disk == set(manifest["tables"]), (
        "ml/out/ and manifest.json disagree; regenerate with `python -m ml.publish_all`"
    )


def test_manifest_names_the_supported_regeneration_command() -> None:
    """A consumer reading the manifest is told how to rebuild it."""
    cfg = MlConfig.from_env()
    manifest_path = cfg.ml_dir / "out" / "manifest.json"
    require_artefact(manifest_path, "python -m ml.publish_all")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["regenerated_by"] == "python -m ml.publish_all"
