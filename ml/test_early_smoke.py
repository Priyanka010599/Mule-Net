"""The canonical early-detection script must actually run.

This exists because of a specific bug, and the shape of that bug is the
whole design rationale. ``ml/early.py``'s ``_print_horizon`` referenced
``entry['is_passthrough_mule']`` on a dict whose key is ``label`` --
collateral from a rename. Every metric function was correct, every test
passed, and ``python -m ml.early`` died with a ``KeyError`` on the first
horizon before writing anything. The committed ``metrics.json`` could
not have come from the committed script.

Nothing caught that, because everything under test was an inner
function. The reporting code at the end of ``main()`` had no caller in
the suite. So this test calls **the real** :func:`ml.early.main`, and it
deliberately does not factor out the printing: if
``_print_horizon`` raises, this test fails. That is its only job.

It is not a numerical test. It asserts nothing about PR-AUC or catch
rates -- a 12-ring sample cannot reproduce a published number and
pretending otherwise would give a second, weaker definition of the
canonical result. It proves the script is executable, end to end,
including the parts that only run at the very end.

Isolation
---------
``MlConfig`` reads every path from the environment, so the run is
redirected with ``MULENET_DATA_DIR`` and ``MULENET_ML_DIR`` into
``tmp_path``. Nothing is monkeypatched to make the writes safe -- the
real ``main()`` does its real writes, to a different root. The canonical
``metrics.json`` is hashed before and after and asserted unchanged, so
the isolation is verified rather than assumed.

The sample keeps all three ring spans on purpose:
``ml.evaluate.stratified_metrics`` calls ``roc_auc_score`` per span
stratum, which raises if a stratum has no positives. A sample that
dropped a span would fail for a reason that has nothing to do with the
script being runnable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import ml.early
from ml.config import TABLE_FILES, MlConfig

#: Rings per span stratum. 4 x 3 spans = 12 rings, ~90 positives: enough
#: for every fold to train on positives, small enough to stay quick.
RINGS_PER_SPAN = 4

#: Ring-free accounts kept. At 1% FPR the threshold is a quantile over
#: these, so a few hundred keeps that arithmetic meaningful.
NEGATIVES = 400

#: Horizons and seeds the smoke run uses. One truncated horizon plus the
#: full window covers both branches of ``cutoff_series`` and both sides
#: of the ``minutes is None`` special case in
#: ``catch_share_before_exit``; one seed keeps it fast.
SMOKE_CUTOFFS: tuple[int | None, ...] = (5, None)
SMOKE_SEEDS: tuple[int, ...] = (42,)


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _write_sample(src: MlConfig, data_dir: Path) -> None:
    """Write a small but structurally complete copy of the six tables."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_parquet(src.table_path(name)) for name in TABLE_FILES}
    truth = tables["ground_truth"]

    in_ring = truth["ring_id"] != ""
    per_ring = truth.loc[in_ring].groupby("ring_id")["ring_bank_count"].first()
    rings: list[str] = []
    for span in (1, 2, 3):
        rings.extend(sorted(per_ring.index[per_ring == span])[:RINGS_PER_SPAN])
    assert rings, "sample must contain rings"

    members = truth.loc[truth["ring_id"].isin(rings), "hashed_account_id"]
    free = truth.loc[~in_ring, "hashed_account_id"].sort_values().head(NEGATIVES)
    keep = set(members) | set(free)

    for name, frame in tables.items():
        if name == "transfers":
            # Both endpoints, so intra-ring edges survive intact: ring
            # onsets and exit instants are derived from them.
            frame = frame.loc[
                frame["from_hashed_account_id"].isin(keep)
                & frame["to_hashed_account_id"].isin(keep)
            ]
        else:
            frame = frame.loc[frame["hashed_account_id"].isin(keep)]
        frame.to_parquet(data_dir / TABLE_FILES[name], index=False)

    written = pd.read_parquet(data_dir / TABLE_FILES["transfers"])
    assert not written.empty, "sample kept no transfers"


def test_ml_early_main_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m ml.early`` completes, printing and writing, on a small sample."""
    real = MlConfig.from_env()
    canonical = real.experiments_dir / "early_detection" / "metrics.json"
    before = _digest(canonical)
    out_before = sorted(p.name for p in (real.ml_dir / "out").glob("*"))

    data_dir = tmp_path / "data"
    _write_sample(real, data_dir)

    monkeypatch.setenv("MULENET_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MULENET_ML_DIR", str(tmp_path / "ml"))
    monkeypatch.setattr(ml.early, "CUTOFF_MINUTES", SMOKE_CUTOFFS)
    monkeypatch.setattr(ml.early, "SEEDS", SMOKE_SEEDS)

    ml.early.main()

    # The reporting code is the part that regressed, so assert it spoke.
    printed = capsys.readouterr().out
    assert "--- T=5m ---" in printed
    assert "--- full ---" in printed
    assert "span1" in printed and "span3" in printed

    written = tmp_path / "ml" / "experiments" / "early_detection" / "metrics.json"
    metrics = json.loads(written.read_text("utf-8"))
    labels = [h["label"] for h in metrics["horizons"]]
    assert labels == ["T=5m", "full"]
    for horizon in metrics["horizons"]:
        for scope in ("single_bank", "cross_bank"):
            assert horizon["ring_catch"][scope]["rings_caught_before_exit"] >= 0
            assert str(SMOKE_SEEDS[0]) in horizon["ring_catch_per_seed"][scope]

    # Every (horizon, scope, seed) persisted its scores and cutoffs.
    oof = sorted(p.name for p in (written.parent / ml.early.OOF_DIRNAME).glob("*.parquet"))
    assert len(oof) == len(SMOKE_CUTOFFS) * 2 * len(SMOKE_SEEDS)
    sample = pd.read_parquet(written.parent / ml.early.OOF_DIRNAME / oof[0])
    assert list(sample.columns) == ["hashed_account_id", "score", "as_of", "threshold"]

    # The canonical outputs must be untouched: isolation verified, not assumed.
    assert _digest(canonical) == before, "smoke run overwrote the canonical metrics.json"
    assert sorted(p.name for p in (real.ml_dir / "out").glob("*")) == out_before
