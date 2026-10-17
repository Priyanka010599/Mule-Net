"""Single source of truth for what the agents service fetches at deploy time.

Used by infra/deploy.sh in two places that must never drift apart --
the MULENET_FETCH env var the deployed container reads at startup, and
the list of objects mirrored from the primary bucket into the
agents-region bucket (the agents service only ever reads its own
mirror, never the primary bucket directly). Before this module
existed, that list lived only as a bash heredoc with four hand-written
lines naming Model A and Model B's files; Model C shipped (D1) without
anyone remembering to add its two files there, and nothing caught it --
the deploy would have silently never served Model C to a live upload,
falling back to the rule instead, with no error at all. That gap is
exactly what infra/test_agents_fetch.py exists to catch from now on:
it imports ml.model_runtime.ALL_MODELS directly, so a model added there
without its files reaching this list fails a test before anyone
deploys, rather than failing silently in production.

Models are fetched here as a courtesy, not a requirement: a missing
model is not a readiness failure (ml/model_runtime.py's own fail-closed
design -- agents.service falls back to ml.rule.velocity_decision), so
required_tables() never lists them and the "uncovered" check below
never requires them. But if ml/out/models/{name}.ubj or its metadata
is missing from disk when this runs, that is a real deploy-time problem
-- the deploy would otherwise upload an incomplete model set with no
error at all -- so model_objects() checks local existence itself and
fails loudly, naming the exact missing path, before the deploy gets
anywhere near gcloud.
"""

from __future__ import annotations

from pathlib import Path


def model_objects() -> list[str]:
    """ml/out/models/{name}.ubj and {name}_metadata.json for every named model.

    Raises SystemExit naming the specific missing path if any listed
    model's files are not present locally -- never silently omits one
    or falls back to a different model's files instead.
    """
    from ml.model_runtime import ALL_MODELS

    objects: list[str] = []
    missing: list[str] = []
    for name in ALL_MODELS:
        for suffix in (f"{name}.ubj", f"{name}_metadata.json"):
            path = f"ml/out/models/{suffix}"
            objects.append(path)
            if not Path(path).exists():
                missing.append(path)
    if missing:
        raise SystemExit(
            f"model file(s) missing from disk, cannot build the agents fetch "
            f"list: {missing}. Train and save the missing model before "
            f"deploying -- this does NOT fall back to a different model's "
            f"files; it stops here."
        )
    return objects


def race_objects() -> list[str]:
    """``agents/out/ring_race.json``, which ``get_ring_race`` reads.

    Written by ``python -m ui.build_snapshot`` (``shared/ring_race.py`` defines
    it). Checked on disk here, like the model files: without it Ask cannot answer
    a race-to-the-exit question, and the service would say so only when asked.
    That is the same silent gap the Model C fetch list once had, so a missing
    artifact stops the deploy here, naming the file and the command that writes it.
    """
    from shared.ring_race import ARTIFACT_NAME

    path = f"agents/out/{ARTIFACT_NAME}"
    if not Path(path).exists():
        raise SystemExit(
            f"{path} is missing from disk, cannot build the agents fetch list. "
            "Run `python -m ui.build_snapshot` (it writes this file) before deploying."
        )
    return [path]


def metric_objects() -> list[str]:
    """``agents/out/metric_catalog.json``, which ``get_metric`` reads.

    Written by ``python -m ml.claims`` in the same run that writes
    ``docs/claims.md`` (``ml/metric_catalog.py`` defines it). Checked on disk like
    the race artifact: without it Ask cannot answer a chart question and would
    say so only when asked.
    """
    from shared.metric_catalog import ARTIFACT_NAME

    path = f"agents/out/{ARTIFACT_NAME}"
    if not Path(path).exists():
        raise SystemExit(
            f"{path} is missing from disk, cannot build the agents fetch list. "
            "Run `python -m ml.claims` (it writes this file) before deploying."
        )
    return [path]


def build_agents_fetch() -> list[str]:
    """Every object the agents service must have locally to be fully ready.

    Fails loudly (SystemExit) if the fetch list does not cover every
    table agents/service.py's readiness check requires, or if a named
    model's files are missing from disk (see model_objects()) -- in
    both cases before any gcloud call, not at the container's startup
    probe or a silent runtime fallback.
    """
    from agents.data import ML_TABLES, SIMULATOR_TABLES, DataStore, required_tables

    objects = [f"simulator/out/{name}" for name in SIMULATOR_TABLES.values()]
    objects += [f"ml/out/{name}" for name in ML_TABLES.values()]
    objects += ["agents/out/cases.parquet"]
    # Model A/B/C: trained once, servable artifacts (ml/model_runtime.py), not
    # required_tables -- a missing model is not a readiness failure, it is an
    # automatic fallback to ml.rule.velocity_decision. Fetched the same way as
    # everything else here: from Cloud Storage at container start, never
    # baked into the image.
    objects += model_objects()
    # The label-free race artifact for Ask's get_ring_race tool. Not a
    # required table: the tool degrades to "race data unavailable" without it.
    objects += race_objects()
    # The chart and claim numbers for Ask's get_metric tool; same degradation.
    objects += metric_objects()

    store = DataStore()
    fetched = {obj.rsplit("/", 1)[-1] for obj in objects}
    uncovered = sorted(
        name for name in required_tables() if store.source_path(name).name not in fetched
    )
    if uncovered:
        raise SystemExit(
            "AGENTS_FETCH does not cover the tables agents/service.py readiness "
            f"requires: {uncovered}. The service would fail its startup probe. "
            "Add the missing objects rather than loosening the readiness check."
        )
    return objects


if __name__ == "__main__":
    print(",".join(build_agents_fetch()))
