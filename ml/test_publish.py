"""Guarantees on the published tables, the evidence split and the tuner.

Three things are worth a test here, and they are not the same thing:

1. the tables in ``ml/out/`` really do satisfy the ``shared/`` contracts,
   checked by re-validating **every** written row rather than trusting
   that ``publish.py`` validated them on the way out;
2. identity evidence never reaches the firing condition — the whole
   point of the restructure, and exactly the kind of property that
   quietly regresses when someone "improves" the rule later;
3. the fold tuner cannot see held-out rows, checked by corrupting them
   and asserting the choice does not move.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml.config import MlConfig
from ml.evidence import CROSS_BANK_COLUMNS, EVIDENCE_RECORD_COLUMNS, SAME_BANK_COLUMNS
from ml.publish import validate_rows
from ml.rule import velocity_decision
from ml.tune import Thresholds, grid, scope_arrays, tune_on, velocity_mask
from shared.contracts import (
    AccountEvidence,
    AccountExplanation,
    EvidenceScope,
    ExplanationFactor,
    MembershipBasis,
    Ring,
    RingExplanation,
    RingMember,
    RiskScore,
)

#: Written table stem -> the model its rows must satisfy.
TABLE_MODELS = {
    "scores": RiskScore,
    "account_explanations": AccountExplanation,
    "rings": Ring,
    "ring_explanations": RingExplanation,
    "account_evidence": AccountEvidence,
}

#: Timestamp columns that must come back tz-aware UTC.
TIMESTAMP_COLUMNS = {
    "scores": "scored_at",
    "account_explanations": "generated_at",
    "rings": "detected_at",
    "ring_explanations": "generated_at",
    "account_evidence": "observed_at",
}


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def out_dir(cfg: MlConfig):
    path = cfg.ml_dir / "out"
    if not (path / "scores.parquet").exists():
        pytest.skip("ml/out/ not built; run `python -m ml.publish`")
    return path


@pytest.mark.parametrize("name", sorted(TABLE_MODELS))
def test_every_written_row_satisfies_its_contract(out_dir, name: str) -> None:
    """Re-validate all rows, not a sample: publish-time checks can drift."""
    model = TABLE_MODELS[name]
    table = pd.read_parquet(out_dir / f"{name}.parquet")
    assert set(table.columns) == set(model.model_fields), name
    records = table.to_dict("records")
    for record in records:
        # parquet hands list columns back as numpy arrays.
        clean = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in record.items()}
        model.model_validate(clean)
    assert len(records) > 0, name


@pytest.mark.parametrize("name", sorted(TIMESTAMP_COLUMNS))
def test_timestamps_are_tz_aware_utc(out_dir, name: str) -> None:
    """A naive datetime would silently mismatch TIMESTAMPTZ in shared/ddl.sql."""
    column = TIMESTAMP_COLUMNS[name]
    series = pd.read_parquet(out_dir / f"{name}.parquet")[column]
    assert series.dt.tz is not None, f"{name}.{column} is tz-naive"
    assert str(series.dt.tz) == "UTC"
    assert isinstance(series.iloc[0].to_pydatetime(), datetime)


def test_evidence_table_matches_the_documented_schema(out_dir) -> None:
    """Both scopes are published now, and each type must match its scope.

    Same-bank rows exist so a ring's CONTEXT members have something to
    join to. A ``same_bank_*`` type carrying ``cross_bank`` scope (or the
    reverse) would be worse than a missing row: it would present ordinary
    same-bank document reuse as the cross-bank fact an STR cites.
    """
    table = pd.read_parquet(out_dir / "account_evidence.parquet")
    assert set(EVIDENCE_RECORD_COLUMNS) <= set(table.columns)
    assert set(table["scope"]) == {
        EvidenceScope.CROSS_BANK.value,
        EvidenceScope.SINGLE_BANK.value,
    }
    cross = table["scope"] == EvidenceScope.CROSS_BANK.value
    assert table.loc[cross, "evidence_type"].isin(CROSS_BANK_COLUMNS).all()
    assert table.loc[~cross, "evidence_type"].isin(SAME_BANK_COLUMNS).all()
    assert not table.duplicated(["hashed_account_id", "evidence_type", "scope"]).any()


def test_same_bank_evidence_only_cites_the_accounts_own_bank(cfg: MlConfig, out_dir) -> None:
    """The mirror of the cross-bank test: "same bank" has to mean it too."""
    table = pd.read_parquet(out_dir / "account_evidence.parquet")
    same = table.loc[table["scope"] == EvidenceScope.SINGLE_BANK.value]
    assert len(same) > 0
    bank = pd.read_parquet(cfg.table_path("accounts")).set_index("hashed_account_id")["bank_id"]
    own = same["hashed_account_id"].map(bank)
    for own_bank, banks in zip(own, same["counterparty_bank_ids"], strict=True):
        assert set(banks) == {own_bank}


def test_scope_lands_in_parquet_as_a_string_not_an_enum_object(out_dir) -> None:
    """``EvidenceScope`` is an Enum; ``model_dump()`` yields the member.

    Writing the member itself would put an opaque Python object in the
    column, which pandas stores and every non-Python reader chokes on.
    ``==`` still passes for a ``str`` Enum, so the comparison above is
    not enough to catch it: the type has to be asserted directly.
    """
    table = pd.read_parquet(out_dir / "account_evidence.parquet")
    assert table["scope"].map(type).eq(str).all()
    assert not table["scope"].map(lambda v: isinstance(v, EvidenceScope)).any()
    assert set(table["scope"]) <= {scope.value for scope in EvidenceScope}


def test_every_account_has_an_explanation(cfg: MlConfig, out_dir) -> None:
    """``get_explanation(subject_type="account")`` must not raise for a real account."""
    table = pd.read_parquet(out_dir / "account_explanations.parquet")
    accounts = pd.read_parquet(cfg.table_path("accounts"))["hashed_account_id"]
    assert set(table["hashed_account_id"]) == set(accounts)
    assert table["hashed_account_id"].is_unique
    assert (table["top_factors"].map(len) > 0).all()


def test_explanation_factors_round_trip_as_nested_records(out_dir) -> None:
    """``top_factors`` is a list of structs in parquet, not a list of strings.

    The nested write is the likeliest thing to break silently: pyarrow
    will happily store a list of dicts and hand back a numpy array of
    dicts, and a consumer that assumed ``list[str]`` gets an object it
    cannot render rather than an error. So every factor of every row is
    reconstructed into an :class:`ExplanationFactor` here.
    """
    rows = pd.read_parquet(out_dir / "account_explanations.parquet")["top_factors"]
    assert len(rows) > 0
    seen = 0
    for row in rows:
        factors = list(row) if isinstance(row, np.ndarray) else row
        assert isinstance(factors, list) and factors
        for factor in factors:
            assert isinstance(factor, dict), type(factor)
            assert set(factor) == set(ExplanationFactor.model_fields)
            rebuilt = ExplanationFactor.model_validate(factor)
            assert ExplanationFactor.model_validate_json(rebuilt.model_dump_json()) == rebuilt
            seen += 1
    assert seen == 4 * len(rows)


def test_explanation_descriptions_are_sentences_not_column_names(out_dir) -> None:
    """A raw feature name is not an explanation an analyst can act on.

    ``feature_name`` carries the column for reproducibility and the
    contract forbids rendering it in place of ``description``, so the
    two are asserted apart: the column must be in ``feature_name`` and
    must *not* leak into the sentence.
    """
    from ml.features import FEATURE_NAMES, feature_columns

    columns = set(feature_columns("cb_"))
    factors = [
        f
        for row in pd.read_parquet(out_dir / "account_explanations.parquet")["top_factors"]
        for f in row
    ]
    assert factors
    for factor in factors[:2000]:
        description = factor["description"]
        assert factor["feature_name"] in columns, factor
        assert not any(description.startswith(name) for name in FEATURE_NAMES), factor
        assert "cb_" not in description and "sb_" not in description, factor
        # Direction now lives in the signed float, so it must not also
        # be spelled in the prose: two copies are two chances to disagree.
        assert "raises risk" not in description and "lowers risk" not in description, factor


def test_explanation_contribution_sign_is_carried_and_used(out_dir) -> None:
    """The sign is load-bearing: a mitigating factor must be negative."""
    table = pd.read_parquet(out_dir / "account_explanations.parquet")
    contributions = [float(f["contribution"]) for row in table["top_factors"] for f in row]
    assert any(c < 0 for c in contributions), "no mitigating factor anywhere is implausible"
    assert any(c > 0 for c in contributions)
    # Largest |SHAP| first, which is the order a UI renders.
    for row in table["top_factors"].iloc[:500]:
        magnitudes = [abs(float(f["contribution"])) for f in row]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_model_version_matches_the_score_it_explains(out_dir) -> None:
    """An explanation shown next to a score it was not computed from is wrong."""
    explanations = pd.read_parquet(out_dir / "account_explanations.parquet")
    scores = pd.read_parquet(out_dir / "scores.parquet")
    merged = explanations.merge(scores, on="hashed_account_id", suffixes=("_e", "_s"))
    assert len(merged) == len(explanations)
    assert (merged["model_version_e"] == merged["model_version_s"]).all()
    assert (merged["score_e"] - merged["score_s"]).abs().max() == 0.0


def test_stage2_cannot_originate_a_ring(cfg: MlConfig) -> None:
    """Identity must never be a firing condition of its own.

    Stage 2's whole safety argument is that it only grows rings stage 1
    already found. With no stage-1 rings there is nothing to grow, no
    matter how dense the identity graph is.
    """
    from ml.features import full_window_cutoff, load_tables
    from ml.stage2 import identity_edges, propagate

    tables = load_tables(cfg)
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    edges = identity_edges(tables, full_window_cutoff(tables))
    assert len(edges) > 0, "the identity graph must be non-empty for this test to mean anything"
    assert propagate({}, edges, bank) == {}


def test_stage2_only_adds_identity_neighbours_of_ring_members(cfg: MlConfig) -> None:
    """Every account stage 2 adds must have an identity edge to a seed."""
    from ml.features import full_window_cutoff, load_tables
    from ml.stage2 import identity_edges, propagate

    tables = load_tables(cfg)
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    edges = identity_edges(tables, full_window_cutoff(tables))
    seeds = sorted(set(edges["hashed_account_id"].iloc[:50]))
    links = propagate({"ring-test": seeds}, edges, bank)["ring-test"]
    neighbours = set(edges.loc[edges["hashed_account_id"].isin(seeds), "hashed_account_id_other"])
    added = {link.hashed_account_id for link in links}
    assert added <= neighbours - set(seeds)
    # The provenance has to be real, not decorative: the account it says
    # it was linked from must be a seed, and the scope must be the two
    # accounts' banks compared — not a constant.
    for link in links:
        assert link.linked_from_hashed_account_id in set(seeds)
        crosses = bank[link.hashed_account_id] != bank[link.linked_from_hashed_account_id]
        assert (link.scope is EvidenceScope.CROSS_BANK) is bool(crosses)
        assert link.alertable is bool(crosses)


def test_cross_bank_evidence_never_cites_the_accounts_own_bank(cfg: MlConfig, out_dir) -> None:
    """"Cross-bank" has to mean it: a same-bank partner is not this evidence."""
    table = pd.read_parquet(out_dir / "account_evidence.parquet")
    table = table.loc[table["scope"] == EvidenceScope.CROSS_BANK.value]
    bank = pd.read_parquet(cfg.table_path("accounts")).set_index("hashed_account_id")["bank_id"]
    own = table["hashed_account_id"].map(bank)
    for own_bank, banks in zip(own, table["counterparty_bank_ids"], strict=True):
        assert own_bank not in set(banks)


def _members(row: object) -> list[dict]:
    """One ring's ``members``, as plain dicts. Parquet hands back ndarrays."""
    return [dict(member) for member in (list(row) if isinstance(row, np.ndarray) else row)]


def _all_members(out_dir) -> list[tuple[str, dict]]:
    """Every ``(ring_id, member)`` pair in the written ring table."""
    rings = pd.read_parquet(out_dir / "rings.parquet")
    return [
        (ring_id, member)
        for ring_id, row in zip(rings["ring_id"], rings["members"], strict=True)
        for member in _members(row)
    ]


def test_rings_have_at_least_two_velocity_members(out_dir) -> None:
    """A ring is still a stage-1 finding: the detected core must be ≥2.

    Counting ``members`` would not say this any more. Stage-2 members are
    leads, and a "ring" of one flagged account plus its identity links
    would be an alert dressed up as a ring — exactly what dropping
    singleton components exists to prevent.
    """
    rings = pd.read_parquet(out_dir / "rings.parquet")
    assert rings["ring_id"].is_unique
    for row in rings["members"]:
        members = _members(row)
        velocity = [m for m in members if m["membership_basis"] == "velocity"]
        assert len(velocity) >= 2
        ids = [m["hashed_account_id"] for m in members]
        assert len(ids) == len(set(ids)), "an account may appear at most once in a ring"


def test_every_ring_member_round_trips_as_a_ring_member_record(out_dir) -> None:
    """``members`` is a list of structs in parquet, not a list of strings.

    Same failure mode as ``account_explanations.top_factors``: pyarrow
    stores a list of dicts happily and hands back a numpy array of dicts,
    so a consumer that assumed ``list[str]`` gets an object it cannot
    render rather than an error. Every member of every ring is rebuilt
    into a :class:`RingMember` here, which also re-runs the contract's
    basis/link/alertable validator against what was actually written.
    """
    pairs = _all_members(out_dir)
    assert pairs
    for _ring_id, member in pairs:
        assert set(member) == set(RingMember.model_fields), member
        rebuilt = RingMember.model_validate(member)
        assert RingMember.model_validate_json(rebuilt.model_dump_json()) == rebuilt
        # str-backed enums compare equal to their value, so assert the type.
        assert isinstance(member["membership_basis"], str)
        assert not isinstance(member["membership_basis"], MembershipBasis)
        assert member["link_scope"] is None or isinstance(member["link_scope"], str)


def test_no_same_bank_identity_link_is_ever_alertable(out_dir) -> None:
    """Stage-2 alerting is cross-bank only; same-bank links are context.

    Same-bank identity reuse is dominated by legitimate cover, so a
    same-bank link that reached an alert queue would be an accusation
    against a household member or a refiled document.
    """
    pairs = _all_members(out_dir)
    same_bank = [m for _, m in pairs if m["link_scope"] == EvidenceScope.SINGLE_BANK.value]
    assert same_bank, "the fixture must contain same-bank links for this to mean anything"
    for member in same_bank:
        assert member["alertable"] is False or member["alertable"] == 0, member
        assert member["membership_basis"] == MembershipBasis.IDENTITY_LINK.value
    for _ring_id, member in pairs:
        if member["membership_basis"] == MembershipBasis.VELOCITY.value:
            assert member["alertable"], "a velocity member is a detection and is alertable"
            assert member["link_scope"] is None


def test_every_identity_link_member_carries_a_lead_shaped_reason(out_dir) -> None:
    """A lead has to read as one, in the row itself.

    ``ui/`` renders ``reason`` directly, so the wording is the qualifier
    that stops an identity link being read as a detection. It must say
    what was shared and that it needs review, and it must never assert
    the account is a mule.
    """
    links = [
        m
        for _, m in _all_members(out_dir)
        if m["membership_basis"] == MembershipBasis.IDENTITY_LINK.value
    ]
    assert links
    for member in links:
        reason = member["reason"]
        assert reason, member
        assert member["linked_from_hashed_account_id"], member
        assert member["link_evidence_type"], member
        lowered = reason.lower()
        assert "shares a" in lowered, reason
        assert ("needs review" in lowered) or ("context only" in lowered), reason
        for verdict in ("is a mule", "laundering", "confirmed", "proven", "guilty"):
            assert verdict not in lowered, reason


def test_no_dangling_evidence_joins(out_dir) -> None:
    """Every ``link_evidence_type`` must join to a real evidence row.

    ``RingMember.link_evidence_type`` is a join key into
    ``account_evidence``; a member whose key matches nothing renders as a
    blank panel in a UI that follows the join — the member's accusation
    shown with its qualifier missing. The counterparty is checked too:
    the row has to be about *this* link, not merely about the account.
    """
    evidence = pd.read_parquet(out_dir / "account_evidence.parquet")
    counterparties = {
        (account, evidence_type): set(partners)
        for account, evidence_type, partners in zip(
            evidence["hashed_account_id"],
            evidence["evidence_type"],
            evidence["counterparty_hashed_account_ids"],
            strict=True,
        )
    }
    checked = 0
    for _ring_id, member in _all_members(out_dir):
        evidence_type = member["link_evidence_type"]
        if evidence_type is None:
            continue
        checked += 1
        key = (member["hashed_account_id"], evidence_type)
        assert key in counterparties, f"no account_evidence row for {key}"
        assert member["linked_from_hashed_account_id"] in counterparties[key], key
    assert checked > 0, "no identity_link members to check"


def test_stage2_seeded_with_nothing_publishes_no_identity_link_members(
    cfg: MlConfig,
) -> None:
    """The safety property, asserted at the table rather than the function.

    Identity must never be a firing condition of its own. With no stage-1
    rings there is nothing to grow, so the published ring table is empty
    — not a table of rings whose every member arrived on a shared device.
    """
    from datetime import UTC

    from ml.features import full_window_cutoff, load_tables
    from ml.publish import build_rings
    from ml.stage2 import identity_edges, propagate

    tables = load_tables(cfg)
    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    edges = identity_edges(tables, full_window_cutoff(tables))
    assert len(edges) > 0, "the identity graph must be non-empty for this test to mean anything"
    added = propagate({}, edges, bank)
    assert added == {}
    assert build_rings({}, added, bank, datetime.now(UTC)) == []


def test_validate_rows_rejects_a_bad_row() -> None:
    """The publish-time guard must actually fail, not merely be present."""
    with pytest.raises(ValidationError):
        validate_rows([{"hashed_account_id": "a", "score": "not-a-number"}], RiskScore)


def test_evidence_columns_cannot_reach_the_firing_condition(cfg: MlConfig) -> None:
    """Corrupt every evidence column; the decision must be bit-identical.

    This is the restructure's central claim — velocity decides, identity
    is only evidence — expressed as something that can fail.
    """
    frame = pd.read_parquet(cfg.features_path)
    before = velocity_decision(frame, "cb_", require_bank_spread=True)
    poisoned = frame.copy()
    for column in ("cb_shared_device_count", "cb_document_reuse_count", "cb_document_reuse_flag"):
        poisoned[column] = 999
    after = velocity_decision(poisoned, "cb_", require_bank_spread=True)
    assert (before == after).all()


def test_tuner_ignores_held_out_rows(cfg: MlConfig) -> None:
    """Scramble the held-out fold's labels; the chosen thresholds must not move."""
    from ml.evaluate import attach_labels, fold_assignment

    frame = attach_labels(pd.read_parquet(cfg.features_path), cfg)
    arrays = scope_arrays(frame, "cb_")
    y = frame["is_passthrough_mule"].to_numpy().astype(bool)
    train = fold_assignment(frame, seed=42) != 0
    candidates = grid()[:60]

    chosen, _ = tune_on(arrays, y, train, True, candidates)
    poisoned = y.copy()
    poisoned[~train] = ~poisoned[~train]
    chosen_again, _ = tune_on(arrays, poisoned, train, True, candidates)
    assert chosen == chosen_again


def test_velocity_mask_agrees_with_the_shipped_rule(cfg: MlConfig) -> None:
    """``tune.velocity_mask`` and ``rule.velocity_decision`` must be one rule."""
    from ml.rule import (
        VELOCITY_MAX_SECONDS,
        VELOCITY_MIN_IN_DEGREE,
        VELOCITY_MIN_OUT_DEGREE,
        VELOCITY_RATIO_RANGE,
    )

    frame = pd.read_parquet(cfg.features_path)
    shipped = Thresholds(
        speed_max_seconds=VELOCITY_MAX_SECONDS,
        ratio_min=VELOCITY_RATIO_RANGE[0],
        ratio_max=VELOCITY_RATIO_RANGE[1],
        min_in_degree=float(VELOCITY_MIN_IN_DEGREE),
        min_out_degree=float(VELOCITY_MIN_OUT_DEGREE),
    )
    from_tune = velocity_mask(scope_arrays(frame, "cb_"), shipped, True)
    from_rule = velocity_decision(frame, "cb_", require_bank_spread=True)
    assert (from_tune == from_rule).all()
