"""Tests for the slow-drip experiment. Deterministic; nothing external is mocked."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.claims import OUTPUT, resolve
from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.evaluate import attach_labels
from ml.experiments.slow_drip import claims_section
from ml.experiments.slow_drip.overlay import (
    DRIP_DELAY_RANGE,
    DRIP_SPACING_RANGE,
    SMALL_AMOUNT_CAP,
    generate_overlay,
    overlay_params,
    slow_drip_ring_activity,
)
from ml.experiments.slow_drip.run import (
    OUT_PATH,
    budget_per_bank,
    evaluate_world,
    fit_isolation_forest,
    ring_report,
    top_k_per_bank,
)
from ml.features import build_features, feature_columns, load_tables
from ml.rule import VELOCITY_MAX_SECONDS
from simulator import generator as sim
from simulator.generator import SimParams

SMALL = SimParams(
    accounts_per_bank=1500,
    rings_per_pattern=5,
    n_businesses=6,
    n_families=12,
    n_dual_bank=12,
    kyc_image_sample=0,
)


@pytest.fixture(scope="module")
def small_overlay(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("overlay")
    generate_overlay(out, overlay_params(SMALL))
    return out


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_is_deterministic_and_separate(small_overlay: Path, tmp_path: Path) -> None:
    generate_overlay(tmp_path, overlay_params(SMALL))
    for name in ("transfers.parquet", "ground_truth.parquet", "devices.parquet"):
        assert _digest(small_overlay / name) == _digest(tmp_path / name)
    canonical = Path(sim.SimulatorConfig.from_env().out_dir).resolve()
    assert small_overlay.resolve() != canonical
    with pytest.raises(ValueError):
        generate_overlay(canonical, overlay_params(SMALL))


def test_overlay_rings_span_two_or_three_banks(small_overlay: Path) -> None:
    truth = pd.read_parquet(small_overlay / "ground_truth.parquet")
    spans = set(truth.loc[truth["ring_id"] != "", "ring_bank_count"])
    assert spans == {2, 3}
    assert truth["ring_id"].nunique() - 1 == 4 * SMALL.rings_per_pattern


def test_drip_constants_derive_from_the_rule_ceiling() -> None:
    assert DRIP_DELAY_RANGE[0] > VELOCITY_MAX_SECONDS
    assert DRIP_SPACING_RANGE[0] > VELOCITY_MAX_SECONDS
    assert DRIP_DELAY_RANGE[0] == int(2.0 * VELOCITY_MAX_SECONDS)


def _run_ring_activity(seed: int = 1) -> tuple[sim._Ring, tuple[np.ndarray, ...], int]:
    ledger = sim._Ledger()
    ring = sim._Ring(
        "ring_x", "mixed", ["BANK_A", "BANK_B"], collector=0, mules=[1, 2], sources=[3]
    )
    horizon = 30 * sim.DAY
    opened = np.full(100, -100 * sim.DAY, dtype=np.int64)
    slow_drip_ring_activity(
        np.random.default_rng(seed), ledger, [ring], opened, horizon, np.arange(10, 100)
    )
    return ring, ledger.arrays(), horizon


def test_hops_are_slower_than_the_rule_ceiling_and_amounts_small() -> None:
    ring, (src, dst, amt, ts, _chan), _ = _run_ring_activity()
    assert amt.max() <= SMALL_AMOUNT_CAP + 1e-9
    for mule in ring.mules:
        credits = np.sort(ts[dst == mule])
        hops = np.sort(ts[(src == mule) & (dst == ring.collector)])
        assert len(hops) > 0
        for hop in hops:
            before = credits[credits <= hop]
            # the nearest ring credit before any forward is further than the ceiling
            assert hop - before.max() > VELOCITY_MAX_SECONDS
        assert np.diff(hops).min() > VELOCITY_MAX_SECONDS


def test_ring_activity_is_seed_deterministic() -> None:
    a = _run_ring_activity(7)[1]
    b = _run_ring_activity(7)[1]
    for x, y in zip(a, b, strict=True):
        assert (x == y).all()


def test_isolation_forest_never_sees_labels(small_overlay: Path) -> None:
    cfg = replace(MlConfig.from_env(), data_dir=small_overlay)
    tables = load_tables(cfg)
    features = build_features(cfg, tables=tables)
    columns = feature_columns("cb_")
    labelled = attach_labels(features, cfg)
    unlabelled_scores = fit_isolation_forest(features, columns)
    # identical scores whether or not labels are even present on the frame, and
    # whatever the labels say
    assert np.array_equal(unlabelled_scores, fit_isolation_forest(labelled, columns))
    shuffled = labelled.copy()
    shuffled["is_passthrough_mule"] = np.random.default_rng(0).permutation(
        shuffled["is_passthrough_mule"].to_numpy()
    )
    assert np.array_equal(unlabelled_scores, fit_isolation_forest(shuffled, columns))
    with pytest.raises(ValueError):
        fit_isolation_forest(labelled, [*columns, "is_passthrough_mule"])


def test_budget_is_equal_per_bank_and_label_free() -> None:
    bank = np.array(["A"] * 200 + ["B"] * 100)
    scores = np.arange(300, dtype=float)
    budget = budget_per_bank(bank)
    assert budget == {"A": 2, "B": 1}
    alert = top_k_per_bank(scores, bank, budget)
    assert alert.sum() == 3
    assert alert[198] and alert[199] and alert[299]


def test_ring_report_definitions() -> None:
    frame = pd.DataFrame(
        {
            "ring_id": ["r1", "r1", "r1", "r2", "r2", "", ""],
            "role": ["mule", "mule", "source", "collector", "mule", "", ""],
            "pattern_type": ["mixed"] * 3 + ["shared_device"] * 2 + ["", ""],
            "ring_bank_count": [2, 2, 2, 3, 3, 0, 0],
        }
    )
    frame["is_passthrough_mule"] = frame["role"].isin(["mule", "collector"]).astype("int8")
    frame["is_complicit"] = frame["role"].isin(["mule", "collector", "source"]).astype("int8")
    #            r1 mule   r1 mule  r1 source r2 coll  r2 mule  normal  normal
    alert = np.array([True, False, True, False, False, True, True])
    report = ring_report(frame, alert)
    assert report["rings"] == 2 and report["rings_caught"] == 1
    assert report["rings_caught_majority"] == 1
    assert report["source_alerts"] == 1
    assert report["false_alerts"] == 2
    assert report["false_alerts_per_ring_caught"] == 2.0
    assert report["by_pattern"]["shared_device"]["caught"] == 0


def test_evaluate_world_end_to_end_on_small_overlay(small_overlay: Path) -> None:
    cfg = replace(MlConfig.from_env(), data_dir=small_overlay)
    tables = load_tables(cfg)
    frame = attach_labels(build_features(cfg, tables=tables), cfg)
    frame["bank_id"] = frame["hashed_account_id"].map(
        tables["accounts"].set_index("hashed_account_id")["bank_id"]
    )
    result = evaluate_world(frame, transfer_from=frame)
    for scope in ("single_bank", "cross_bank"):
        detectors = result["detectors"][scope]
        assert {"velocity_rule", "isolation_forest", "xgboost_oof"} <= set(detectors)
        for name in ("isolation_forest", "xgboost_oof", "xgboost_canonical_transfer"):
            alerts = detectors[name]["budget_1pct_per_bank"]["alerts"]
            assert alerts == sum(result["budget_1pct_per_bank"].values())
    # single-bank scope cannot satisfy the cross-bank clause, and the rule drops it there
    assert result["detectors"]["single_bank"]["velocity_rule"]["rings"] == 4 * 5


# ---------------------------------------------------------------- docs/claims.md


@pytest.fixture(scope="module")
def metrics() -> dict:
    require_artefact(OUT_PATH, "python -m ml.experiments.slow_drip.run")
    return json.loads(OUT_PATH.read_text("utf-8"))


def test_metrics_record_the_rule_ceiling_it_was_derived_from(metrics: dict) -> None:
    assert metrics["design"]["velocity_max_seconds_from_rule"] == VELOCITY_MAX_SECONDS
    assert metrics["design"]["drip_delay_range_seconds"] == list(DRIP_DELAY_RANGE)


def test_claims_section_matches_metrics_literally(metrics: dict) -> None:
    path = Path(MlConfig.from_env().ml_dir.parent) / OUTPUT
    require_artefact(path, "see ml/experiments/slow_drip/claims_section.py")
    published = claims_section.parse_section(path.read_text("utf-8"))
    assert published["source"] == claims_section.METRICS_RELATIVE
    expected = {c.id: c for c in claims_section.claims_for(metrics)}
    records = {r["id"]: r for r in published["claims"]}
    assert set(records) == set(expected)
    for claim_id, record in records.items():
        claim = expected[claim_id]
        assert record["path"] == claim.path
        raw = resolve(metrics, claim.path)
        assert record["value"] == pytest.approx(round(float(raw), claim.decimals), abs=1e-9)
