"""Tests for the claims-level cloud-vs-local gate.

A gate that cannot fail is worth nothing, and this one is weaker than
:mod:`ml.vertex.gate` by construction, so its failure modes matter more,
not less. Every check here has a paired test: one that the gate passes
and one that it must fail.

The sibling-enumeration test is a regression test for a real bug. The
first version of :func:`ml.vertex.claims_gate._siblings` varied the two
halves of a derived difference independently, which paired
``cross_bank`` with ``cross_bank``, made every pooled spread ``0.0`` and
turned the value check into a bit-exact comparison that failed three
claims for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.claims import CLAIMS, SOURCES, Claim
from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.vertex.claims_gate import (
    Pair,
    check_directions,
    check_span_ordering,
    check_values,
    derive_pairs,
    load,
    main,
    per_seed_values,
    run,
    seed_values,
    sign_at,
    spread,
    tolerance,
)

SEEDS = (42, 43, 44)
CUTOFFS = (5, 10, None)

#: Per-(scope, span) PR-AUC of the synthetic run. Cross-bank beats
#: single-bank on spans 2 and 3 and ties on span 1, so the fixture has
#: the same shape as the real result the gate is written against.
BASE = {
    ("cross_bank", "span1"): 0.30,
    ("cross_bank", "span2"): 0.50,
    ("cross_bank", "span3"): 0.70,
    ("single_bank", "span1"): 0.30,
    ("single_bank", "span2"): 0.30,
    ("single_bank", "span3"): 0.30,
}


def _early(jitter: float = 0.0, span2_cross: float | None = None) -> dict:
    """A minimal ``early_detection`` document with a real seed dimension.

    ``jitter`` shifts seed 44 only, so the seed-to-seed spread is a known
    number rather than an accident. ``span2_cross`` overrides cross-bank
    span-2 PR-AUC, which is how a test flips a claim's direction.
    """

    def strata(scope: str, seed: int) -> dict:
        out = {}
        for span in ("span1", "span2", "span3"):
            value = BASE[(scope, span)]
            if span2_cross is not None and scope == "cross_bank" and span == "span2":
                value = span2_cross
            out[span] = {"pr_auc": value + (jitter if seed == 44 else 0.0)}
        return out

    horizons = []
    for cutoff in CUTOFFS:
        seeds = SEEDS if cutoff is not None else (42,)
        runs = [
            {
                "scope": scope,
                "seed": seed,
                "pr_auc": sum(strata(scope, seed)[s]["pr_auc"] for s in ("span1", "span2", "span3"))
                / 3,
                "strata": strata(scope, seed),
            }
            for scope in ("single_bank", "cross_bank")
            for seed in seeds
        ]
        horizons.append(
            {
                "cutoff_minutes": cutoff,
                "summary": {
                    scope: {
                        "pr_auc_mean": sum(
                            r["pr_auc"] for r in runs if r["scope"] == scope
                        )
                        / len(seeds)
                    }
                    for scope in ("single_bank", "cross_bank")
                },
                "strata_summary": {
                    scope: {
                        span: {
                            "pr_auc_mean": sum(
                                r["strata"][span]["pr_auc"] for r in runs if r["scope"] == scope
                            )
                            / len(seeds)
                        }
                        for span in ("span1", "span2", "span3")
                    }
                    for scope in ("single_bank", "cross_bank")
                },
                "runs": runs,
                "ring_catch": {
                    "cross_bank": {"ring_recall": 0.9},
                    "single_bank": {"ring_recall": 0.8},
                },
                "ring_catch_per_seed": {
                    scope: {
                        str(seed): {"ring_recall": 0.9 if scope == "cross_bank" else 0.8}
                        for seed in seeds
                    }
                    for scope in ("single_bank", "cross_bank")
                },
            }
        )
    return {"horizons": horizons}


def _lead() -> dict:
    """A minimal ``lead_time`` document with a per-seed mirror."""
    per_seed = {
        str(seed): {"cross_bank_lead": {"rings_tied": 90.0 + (seed - 42)}} for seed in SEEDS
    }
    return {
        "per_seed": per_seed,
        "seed_mean": {"cross_bank_lead": {"rings_tied": 91.0}},
        "exit_distribution": {"minutes_after_onset": {"p50": 89.4}},
    }


def _docs(**kwargs) -> dict:
    return {"early_detection": _early(**kwargs), "lead_time": _lead()}


PR_AUC_5_CROSS = "horizons[cutoff_minutes=5].summary.cross_bank.pr_auc_mean"
PR_AUC_5_SINGLE = "horizons[cutoff_minutes=5].summary.single_bank.pr_auc_mean"
SPAN2_CROSS = "horizons[cutoff_minutes=5].strata_summary.cross_bank.span2.pr_auc_mean"
SPAN2_SINGLE = "horizons[cutoff_minutes=5].strata_summary.single_bank.span2.pr_auc_mean"

PAIR = Pair(
    id="pr_auc",
    source="early_detection",
    cross_path=PR_AUC_5_CROSS,
    single_path=PR_AUC_5_SINGLE,
)


# --- per-seed resolution -------------------------------------------------


def test_summary_mean_maps_back_to_its_runs() -> None:
    doc = _early(jitter=0.03)
    assert per_seed_values(doc, PR_AUC_5_CROSS) == {
        42: pytest.approx(0.5),
        43: pytest.approx(0.5),
        44: pytest.approx(0.53),
    }


def test_strata_mean_maps_back_to_its_runs() -> None:
    values = per_seed_values(_early(jitter=0.03), SPAN2_CROSS)
    assert spread(values) == pytest.approx(0.03)


def test_ring_catch_maps_to_ring_catch_per_seed() -> None:
    path = "horizons[cutoff_minutes=5].ring_catch.cross_bank.ring_recall"
    assert per_seed_values(_early(), path) == {42: 0.9, 43: 0.9, 44: 0.9}


def test_lead_time_seed_mean_maps_to_per_seed() -> None:
    path = "seed_mean.cross_bank_lead.rings_tied"
    assert per_seed_values(_lead(), path) == {42: 90.0, 43: 91.0, 44: 92.0}


def test_a_path_with_no_seed_dimension_has_no_per_seed_values() -> None:
    """``exit_distribution`` is a property of the data, not of a fit."""
    assert per_seed_values(_lead(), "exit_distribution.minutes_after_onset.p50") == {}


def test_spread_of_a_single_seed_is_zero() -> None:
    assert spread({42: 0.5}) == 0.0


def test_derived_claim_differences_seed_for_seed() -> None:
    doc = _early(jitter=0.03)
    # Both scopes are jittered identically, so the difference is flat.
    assert spread(seed_values(doc, SPAN2_CROSS, SPAN2_SINGLE)) == pytest.approx(0.0)


# --- tolerance -----------------------------------------------------------


def test_tolerance_pools_over_siblings_not_just_this_claim() -> None:
    """The full-window horizon has one seed, so pooling is what saves it."""
    doc = _early(jitter=0.04)
    full = "horizons[cutoff_minutes=None].summary.cross_bank.pr_auc_mean"
    assert spread(per_seed_values(doc, full)) == 0.0
    assert tolerance(doc, full, None).value == pytest.approx(0.04)


def test_tolerance_of_a_difference_is_not_zero() -> None:
    """Regression: the two halves of a difference must vary in lockstep.

    Varying them independently pairs ``cross_bank`` with ``cross_bank``,
    which makes every spread the spread of zero.
    """
    doc = _early(jitter=0.04)
    # span3 cross-bank is jittered, single-bank is too, but span-level
    # siblings include the horizon axis, so some sibling has real spread.
    tol = tolerance(doc, SPAN2_CROSS, SPAN2_SINGLE)
    assert tol.pooled_over == 9  # 3 spans x 3 horizons
    for path in (SPAN2_CROSS, SPAN2_SINGLE):
        assert "cross_bank" in path or "single_bank" in path


def test_tolerance_is_zero_without_any_seed_dimension() -> None:
    """A data property must match exactly; the gate says so with tol 0."""
    tol = tolerance(_lead(), "exit_distribution.minutes_after_onset.p50", None)
    assert tol.value == 0.0


# --- sign resolution -----------------------------------------------------


def test_a_gap_inside_the_tolerance_is_a_tie_not_a_direction() -> None:
    assert sign_at(0.004, 0.01) == 0
    assert sign_at(0.02, 0.01) == 1
    assert sign_at(-0.02, 0.01) == -1


# --- pair derivation -----------------------------------------------------


def test_pairs_are_derived_from_the_real_claim_set() -> None:
    ids = {pair.id for pair in derive_pairs()}
    assert ids == {"pr_auc_t5", "ring_recall_t180", "alert_burden_t5"} | {
        f"coverage_span{span}" for span in (1, 2, 3)
    }


def test_derived_difference_claims_are_not_treated_as_scope_pairs() -> None:
    """``span_gradient_*`` already *is* a difference; check 2 owns it."""
    assert not any(pair.id.startswith("span_gradient") for pair in derive_pairs())


def test_an_unpaired_cross_bank_claim_is_not_gated_as_a_pair() -> None:
    lonely = Claim(
        id="lonely_cross_bank",
        statement="x",
        source="early_detection",
        path=PR_AUC_5_CROSS,
    )
    assert derive_pairs([lonely]) == []


# --- check 1: directions -------------------------------------------------


def test_directions_pass_when_the_cloud_matches() -> None:
    report = check_directions(_docs(), _docs(), [PAIR])
    assert report["passed"]
    assert [row["at"] for row in report["rows"]] == ["T=5m", "T=10m", "full"]


def test_directions_are_checked_at_every_horizon_not_just_the_headline() -> None:
    """A sign that flips at T=10m only must still fail the gate."""
    local = _docs()
    cloud = _docs()
    horizon = cloud["early_detection"]["horizons"][1]
    horizon["summary"]["cross_bank"]["pr_auc_mean"] = 0.1
    report = check_directions(local, cloud, [PAIR])
    assert not report["passed"]
    (failure,) = [row for row in report["rows"] if not row["passed"]]
    assert failure["at"] == "T=10m"
    assert failure["local_sign"] == 1
    assert failure["cloud_sign"] == -1


def test_a_direction_collapsing_to_a_tie_fails() -> None:
    """Losing the effect is a failure, not merely a smaller number."""
    cloud = _docs()
    for horizon in cloud["early_detection"]["horizons"]:
        horizon["summary"]["cross_bank"]["pr_auc_mean"] = horizon["summary"]["single_bank"][
            "pr_auc_mean"
        ]
    report = check_directions(_docs(), cloud, [PAIR])
    assert not report["passed"]


# --- check 2: span ordering ---------------------------------------------


def test_span_ordering_passes_on_the_fixture_shape() -> None:
    report = check_span_ordering(_docs(), _docs())
    assert report["passed"]


def test_span_ordering_fails_when_span2_stops_being_positive() -> None:
    cloud = _docs(span2_cross=0.30)
    report = check_span_ordering(_docs(), cloud)
    assert not report["passed"]
    failed = [row["check"] for row in report["rows"] if not row["passed"]]
    assert "span2 gradient @ T=5m (cloud)" in failed


def test_span_ordering_fails_when_span1_leaves_the_near_zero_band() -> None:
    cloud = _docs()
    strata = cloud["early_detection"]["horizons"][0]["strata_summary"]
    strata["cross_bank"]["span1"]["pr_auc_mean"] = 0.9
    report = check_span_ordering(_docs(), cloud)
    assert not report["passed"]


def test_span_ordering_requires_every_seed_positive_not_just_the_mean() -> None:
    """A mean carried by one seed is not a reproduced finding."""
    cloud = _docs()
    horizon = cloud["early_detection"]["horizons"][0]
    for record in horizon["runs"]:
        if record["scope"] != "cross_bank":
            continue
        record["strata"]["span2"]["pr_auc"] = 2.0 if record["seed"] == 44 else 0.1
    report = check_span_ordering(_docs(), cloud)
    assert not report["passed"]


# --- check 3: values ----------------------------------------------------


def test_values_pass_within_the_derived_tolerance() -> None:
    claim = Claim(id="pr_auc", statement="x", source="early_detection", path=PR_AUC_5_CROSS)
    local = {"early_detection": _early(jitter=0.04)}
    cloud = {"early_detection": _early(jitter=0.04)}
    cloud["early_detection"]["horizons"][0]["summary"]["cross_bank"]["pr_auc_mean"] += 0.01
    report = check_values(local, cloud, [claim])
    assert report["passed"]
    assert report["rows"][0]["tolerance"] == pytest.approx(0.04)


def test_values_fail_outside_the_derived_tolerance() -> None:
    claim = Claim(id="pr_auc", statement="x", source="early_detection", path=PR_AUC_5_CROSS)
    local = {"early_detection": _early(jitter=0.04)}
    cloud = {"early_detection": _early(jitter=0.04)}
    cloud["early_detection"]["horizons"][0]["summary"]["cross_bank"]["pr_auc_mean"] += 0.2
    report = check_values(local, cloud, [claim])
    assert not report["passed"]
    assert report["rows"][0]["delta"] == pytest.approx(0.2)


def test_a_data_property_must_match_exactly() -> None:
    claim = Claim(
        id="exit_p50",
        statement="x",
        source="lead_time",
        path="exit_distribution.minutes_after_onset.p50",
    )
    local = _docs()
    cloud = _docs()
    cloud["lead_time"]["exit_distribution"]["minutes_after_onset"]["p50"] = 89.5
    assert not check_values(local, cloud, [claim])["passed"]
    assert check_values(local, local, [claim])["passed"]


# --- CLI and the real files ---------------------------------------------


def test_load_fails_loudly_on_a_missing_metrics_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


#: How to regenerate each ml.claims.SOURCES entry, for require_artefact's
#: error message below. A source added without an entry here still gets a
#: usable (if generic) hint rather than a KeyError.
_REGENERATE_WITH = {
    "early_detection": "python -m ml.early",
    "lead_time": "python -m ml.leadtime",
    "ablation": "python ml/experiments/ablation/run_ablation.py",
    "ibm_aml": "python ml/experiments/ibm_aml/run_grouped_cv.py",
    "gnn_xgb": "python ml/experiments/gnn_xgb/run_gnn_xgb.py",
}


@pytest.fixture
def experiments() -> object:
    """The real local canonical metrics, or a skip / CI failure.

    Every ml.claims.SOURCES entry, not a hardcoded pair -- SOURCES grew to
    four after this fixture was written and a hardcoded two silently
    stopped covering it, which is exactly the kind of drift this file's
    own gate exists to catch elsewhere.
    """
    cfg = MlConfig.from_env()
    directory = cfg.ml_dir / "experiments"
    for name, relative in SOURCES.items():
        hint = _REGENERATE_WITH.get(name, f"the script that produces {relative}")
        require_artefact(directory / Path(relative).relative_to("ml/experiments"), hint)
    return directory


def test_cli_passes_on_the_real_run_against_itself(experiments, capsys) -> None:
    argv = [
        "--local-experiments",
        str(experiments),
        "--cloud-experiments",
        str(experiments),
    ]
    assert main(argv) == 0
    assert "CLAIMS GATE: PASSED" in capsys.readouterr().out


def test_cli_fails_on_the_real_run_with_one_headline_moved(
    experiments, tmp_path, capsys
) -> None:
    """The gate must fail on real files, not only on fixtures."""
    for name, relative in SOURCES.items():
        rel_path = Path(relative).relative_to("ml/experiments")
        target_file = tmp_path / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        source = (experiments / rel_path).read_text("utf-8")
        target_file.write_text(source, "utf-8")
    path = tmp_path / "early_detection" / "metrics.json"
    doc = json.loads(path.read_text("utf-8"))
    doc["horizons"][0]["summary"]["cross_bank"]["pr_auc_mean"] = 0.10
    path.write_text(json.dumps(doc), "utf-8")
    argv = ["--local-experiments", str(experiments), "--cloud-experiments", str(tmp_path)]
    assert main(argv) == 1
    assert "CLAIMS GATE: FAILED" in capsys.readouterr().out


def test_every_real_claim_resolves_through_the_per_seed_mapping(experiments) -> None:
    """This is what catches a claim ``ml/claims.py`` adds that the
    per-seed mapping cannot follow -- and every claim must appear in the
    report, so a silently skipped claim is a failure too."""
    passed, report = run(experiments, experiments)
    assert passed, report
    for claim in CLAIMS:
        assert claim.id in report
