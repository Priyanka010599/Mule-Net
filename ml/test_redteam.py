"""Tests for the Gemini red-team harness.

Every test here is offline and free. The Vertex path is exercised the way
``agents/test_service.py`` does it -- by replacing exactly one function,
:func:`agents.adk_agents._run_agent`, the only place a model is called --
so the prompt registration, the JSON parse, the retry ledger and the
strategy typing all run through their real code with no network and no
bill. The detection half needs no fake at all: it runs the real
``ml.features`` -> ``ml.rule`` -> ``ml.publish`` -> ``ml.stage2`` chain
over an in-memory graph, so no simulator or ``ml/out/`` artefact is
required either.

Two properties are worth more than the rest and are the reason this file
exists.

**The harness must deliver the pattern it was given.** A synthesiser that
quietly hands a collector a fan-in of 16 when the strategy said 15 is
reporting a verdict on a pattern nobody designed, and that bug was
written twice while building this -- once by chaining collectors without
accounting for the chain credit, once by running them concurrently so a
2-hour hold produced a 0-second pass-through. Both are pinned below.

**A verdict must not be able to drift from the rule.** The thresholds
live in ``ml/rule.py`` and are imported, never retyped, so the test that
matters is the one asserting this harness grades against the shipped
constants rather than a copy of them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents import adk_agents
from agents.config import AgentsConfig
from ml import redteam
from ml.config import MlConfig
from ml.rule import (
    MIN_BANKS,
    VELOCITY_MAX_SECONDS,
    VELOCITY_MIN_IN_DEGREE,
    VELOCITY_MIN_OUT_DEGREE,
    VELOCITY_RATIO_RANGE,
)

#: Positions in ``redteam.PATTERN_BOUNDS``' ``(low, high, default)``
#: tuples. The bounds are read from the module under test rather than
#: restated, so a widened bound cannot leave these tests asserting the
#: old one.
_HIGH, _DEFAULT = 1, 2

#: A pattern that clears every clause with room to spare. The control:
#: if the detector cannot catch this, nothing below means anything.
CAUGHT_PATTERN: dict[str, Any] = {
    "collectors": 2,
    "mules_per_collector": 25,
    "onward_accounts": 8,
    "hop_delay_seconds": 120,
    "amount_per_inbound_hop": 5000.0,
    "retained_fraction": 0.05,
    "counterparty_banks": 3,
    "reuse_device": False,
    "reuse_kyc_id": False,
}

#: One clause -> the pattern override that should defeat exactly it, and
#: nothing else. Derived from the shipped constants rather than written
#: as literals, so retuning the rule moves these with it: a fan-in floor
#: raised to 20 makes this test ask for 19, not keep asking for 15.
EVASION_OVERRIDES: dict[str, dict[str, Any]] = {
    "in_degree": {"mules_per_collector": VELOCITY_MIN_IN_DEGREE - 1},
    "out_degree": {"onward_accounts": VELOCITY_MIN_OUT_DEGREE - 1},
    "distinct_counterparty_banks": {"counterparty_banks": MIN_BANKS - 1},
    "passthrough_speed_seconds": {"hop_delay_seconds": VELOCITY_MAX_SECONDS * 2},
    "passthrough_ratio": {"retained_fraction": 0.95},
}


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


def _strategy(
    name: str = "control",
    clause: str = "ring_clustering",
    **overrides: Any,
) -> redteam.EvasionStrategy:
    """One typed strategy, built through the real parser.

    Goes through :func:`redteam.parse_strategies` rather than constructing
    the dataclass directly, so the clamping and defaulting a live reply
    would meet are in the path of every test here.
    """
    pattern = dict(CAUGHT_PATTERN) | overrides
    return redteam.parse_strategies(
        [
            {
                "name": name,
                "description": f"test strategy {name}",
                "targets": "velocity",
                "evades_clause": clause,
                "why_it_should_evade": "constructed by the test",
                "pattern": pattern,
            }
        ]
    )[0]


# --------------------------------------------------------------------------
# the harness must run the pattern it was handed
# --------------------------------------------------------------------------


def test_every_collector_gets_exactly_the_fan_in_the_strategy_asked_for() -> None:
    """The chain credit counts toward in_degree, so it replaces a mule.

    This is the bug the chaining introduced: the downstream collector was
    credited by its predecessor *and* by a full set of mules, so a
    strategy specifying 15 produced a collector at 16 -- over the floor
    it was built to duck. The verdict then described a pattern the model
    never designed.
    """
    strategy = _strategy(mules_per_collector=15)
    graph, collectors = redteam.synthesize(strategy, 1)
    tables = redteam.as_tables(graph)
    inbound = tables["transfers"]["to_hashed_account_id"].value_counts()
    for collector in collectors:
        assert inbound[collector] == 15, collector


def test_every_collector_gets_exactly_the_fan_out_the_strategy_asked_for() -> None:
    strategy = _strategy(onward_accounts=6)
    graph, collectors = redteam.synthesize(strategy, 1)
    outbound = redteam.as_tables(graph)["transfers"][
        "from_hashed_account_id"
    ].value_counts()
    for collector in collectors:
        assert outbound[collector] == 6, collector


def test_the_collector_chain_is_sequenced_in_time_not_concurrent(cfg: MlConfig) -> None:
    """A slow hop must stay slow on every collector in the chain.

    Running the collectors concurrently made the predecessor's payment
    land seconds before the successor's own outbound debit, which is a
    0-second pass-through. A strategy holding funds for twice the speed
    ceiling then *passed* the speed clause on the downstream collector,
    and the harness reported it caught on the strength of its own timing
    artefact.
    """
    strategy = _strategy(hop_delay_seconds=VELOCITY_MAX_SECONDS * 2)
    verdict = redteam.evaluate(strategy, 1, cfg)
    assert len(verdict.clause_values) == strategy.pattern.collectors
    for collector, clauses in verdict.clause_values.items():
        assert not clauses["passthrough_speed_seconds"]["passes"], collector


def test_a_single_collector_pattern_forms_no_ring(cfg: MlConfig) -> None:
    """Clustering needs two flagged endpoints, and a mule is never one.

    Reported as an account-level alert rather than as a catch-with-ring,
    which is what MuleNet itself does: "a component of one is an alert,
    not a ring".
    """
    verdict = redteam.evaluate(_strategy(collectors=1), 1, cfg)
    assert verdict.rings_formed == 0
    assert verdict.flagged_collectors == 1
    assert verdict.caught
    assert verdict.caught_by == "velocity_alert_no_ring"


# --------------------------------------------------------------------------
# the verdicts themselves
# --------------------------------------------------------------------------


def test_the_control_pattern_is_caught_and_forms_a_ring(cfg: MlConfig) -> None:
    """If this fails, every evasion result below is meaningless."""
    verdict = redteam.evaluate(_strategy(), 1, cfg)
    assert verdict.caught
    assert verdict.caught_by == "velocity_ring"
    assert verdict.rings_formed == 1
    for clauses in verdict.clause_values.values():
        assert all(clause["passes"] for clause in clauses.values())


@pytest.mark.parametrize("clause", sorted(EVASION_OVERRIDES))
def test_each_clause_can_be_evaded_and_only_that_clause_fails(
    clause: str, cfg: MlConfig
) -> None:
    """Defeating one clause defeats the rule, and is reported as itself.

    Both halves matter. That the pattern evades is the red-team result;
    that the *named* clause is the one that failed is what makes the
    reasoning line trustworthy rather than a guess -- a harness that
    reported "evaded" while a different threshold did the work would
    send somebody to retune the wrong number.
    """
    verdict = redteam.evaluate(_strategy(clause=clause, **EVASION_OVERRIDES[clause]), 1, cfg)
    assert not verdict.caught
    assert verdict.caught_by == "missed"
    for collector, clauses in verdict.clause_values.items():
        failed = {name for name, c in clauses.items() if not c["passes"]}
        assert failed == {clause}, (collector, failed)
    assert clause in verdict.reasoning
    assert "not the designed one" not in verdict.reasoning


def test_a_miss_on_an_unaimed_clause_says_so(cfg: MlConfig) -> None:
    """Evading by accident is a weaker result and is labelled as one."""
    strategy = _strategy(clause="in_degree", counterparty_banks=1)
    verdict = redteam.evaluate(strategy, 1, cfg)
    assert not verdict.caught
    assert "not the designed one" in verdict.reasoning


# --------------------------------------------------------------------------
# stage 2 cannot rescue a stage-1 miss, and that is by design
# --------------------------------------------------------------------------


def test_stage2_widens_a_ring_stage1_already_found(cfg: MlConfig) -> None:
    """A shared device pulls the pattern's other accounts in."""
    verdict = redteam.evaluate(_strategy(reuse_device=True), 1, cfg)
    assert verdict.caught_by == "velocity_ring"
    assert verdict.stage2_alertable > 0


def test_identity_reuse_alone_never_produces_a_catch(cfg: MlConfig) -> None:
    """The safety property in ``ml/stage2.py``, measured from the outside.

    Same shared device as the test above, but the pattern also ducks the
    bank-spread clause, so stage 1 flags nothing and there is no ring to
    propagate out of. Stage 2 must then add nothing at all -- if a shared
    credential could originate a ring, this is where it would show, and
    it would mean the detector can accuse an account on identity reuse
    alone.
    """
    verdict = redteam.evaluate(
        _strategy(clause="distinct_counterparty_banks", counterparty_banks=1, reuse_device=True),
        1,
        cfg,
    )
    assert verdict.rings_formed == 0
    assert verdict.stage2_alertable == 0
    assert verdict.stage2_context == 0
    assert not verdict.caught


def test_same_bank_identity_links_are_never_counted_as_caught(cfg: MlConfig) -> None:
    """Context is not a catch, and the counts are kept apart.

    A same-bank link is downgraded to context by the ruling, so counting
    it toward ``caught`` would claim credit for an accusation MuleNet
    refuses to make.
    """
    verdict = redteam.evaluate(_strategy(reuse_kyc_id=True), 1, cfg)
    assert verdict.stage2_context > 0
    assert verdict.caught_by == "velocity_ring"


# --------------------------------------------------------------------------
# the thresholds are the shipped ones, not a copy
# --------------------------------------------------------------------------


def test_the_clause_report_grades_against_ml_rule_not_a_local_copy(
    cfg: MlConfig,
) -> None:
    """The requirement strings must quote the imported constants.

    The whole harness is worthless if it grades against last week's
    thresholds, and a retyped number is how that happens quietly.
    """
    clauses = next(iter(redteam.evaluate(_strategy(), 1, cfg).clause_values.values()))
    assert clauses["in_degree"]["requirement"] == f"x >= {VELOCITY_MIN_IN_DEGREE}"
    assert clauses["out_degree"]["requirement"] == f"x >= {VELOCITY_MIN_OUT_DEGREE}"
    assert (
        clauses["distinct_counterparty_banks"]["requirement"] == f"x >= {MIN_BANKS}"
    )
    assert clauses["passthrough_speed_seconds"]["requirement"] == (
        f"0 <= x <= {VELOCITY_MAX_SECONDS:g}"
    )
    assert clauses["passthrough_ratio"]["requirement"] == (
        f"{VELOCITY_RATIO_RANGE[0]:g} <= x <= {VELOCITY_RATIO_RANGE[1]:g}"
    )


def test_the_five_clauses_reported_are_the_five_the_rule_reads() -> None:
    """No sixth clause, no missing one: the report is the rule's shape."""
    assert set(redteam.EVADABLE_CLAUSES) == {
        "passthrough_speed_seconds",
        "passthrough_ratio",
        "in_degree",
        "out_degree",
        "distinct_counterparty_banks",
        # Not a threshold: the requirement that a flagged account's
        # counterparty be flagged too.
        "ring_clustering",
    }


# --------------------------------------------------------------------------
# clamping: the model's numbers, or an honest note that they were changed
# --------------------------------------------------------------------------


def test_an_out_of_range_value_is_clamped_and_named() -> None:
    strategies = redteam.parse_strategies(
        [{"name": "greedy", "pattern": dict(CAUGHT_PATTERN) | {"mules_per_collector": 9999}}]
    )
    pattern = strategies[0].pattern
    assert pattern.mules_per_collector == redteam.PATTERN_BOUNDS["mules_per_collector"][_HIGH]
    assert any("mules_per_collector" in note for note in pattern.clamped)


def test_a_missing_field_takes_its_default_and_is_not_called_clamped() -> None:
    """Declining to choose is not a decision this harness overruled."""
    pattern = redteam.parse_strategies([{"name": "sparse", "pattern": {}}])[0].pattern
    assert pattern.clamped == ()
    assert pattern.mules_per_collector == redteam.PATTERN_BOUNDS["mules_per_collector"][_DEFAULT]


def test_an_unreadable_value_falls_back_and_says_so() -> None:
    pattern = redteam.parse_strategies(
        [{"name": "prose", "pattern": {"mules_per_collector": "as many as possible"}}]
    )[0].pattern
    assert pattern.mules_per_collector == redteam.PATTERN_BOUNDS["mules_per_collector"][_DEFAULT]
    assert any("unreadable" in note for note in pattern.clamped)


def test_a_nonsense_evades_clause_is_kept_verbatim() -> None:
    """What the model claimed to attack is half the result.

    Correcting it here would make the report agree with itself by
    construction, which is the one thing a red-team report must not do.
    """
    strategy = redteam.parse_strategies(
        [{"name": "x", "evades_clause": "the vibes", "pattern": {}}]
    )[0]
    assert strategy.evades_clause == "the vibes"
    assert strategy.evades_clause not in redteam.EVADABLE_CLAUSES


# --------------------------------------------------------------------------
# synthetic ids
# --------------------------------------------------------------------------


def test_every_synthetic_id_is_shaped_like_a_real_hashed_account_id() -> None:
    """64 hex characters, so nothing downstream can special-case them."""
    graph, _ = redteam.synthesize(_strategy(), 1)
    tables = redteam.as_tables(graph)
    for account_id in tables["accounts"]["hashed_account_id"]:
        assert len(account_id) == 64
        assert set(account_id) <= set("0123456789abcdef")


def test_two_runs_of_one_strategy_produce_the_same_ids() -> None:
    """A fixed key and a fixed epoch, so a verdict is reproducible."""
    first, _ = redteam.synthesize(_strategy(), 7)
    second, _ = redteam.synthesize(_strategy(), 7)
    assert redteam.as_tables(first)["accounts"].equals(
        redteam.as_tables(second)["accounts"]
    )


def test_different_strategies_never_share_an_account() -> None:
    """The strategy index is in the HMAC input, so graphs stay disjoint."""
    first, _ = redteam.synthesize(_strategy(), 1)
    second, _ = redteam.synthesize(_strategy(), 2)
    left = set(redteam.as_tables(first)["accounts"]["hashed_account_id"])
    right = set(redteam.as_tables(second)["accounts"]["hashed_account_id"])
    assert not left & right


def test_the_synthetic_key_is_not_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody can point this harness at a real hashing key.

    If the key were configurable, a misconfigured run would emit ids that
    collide with production ones -- synthetic rows wearing real account
    identities, in a repo whose whole discipline is that only hashed ids
    travel.
    """
    monkeypatch.setenv("MULENET_HASH_KEY", "a-real-looking-key")
    before, _ = redteam.synthesize(_strategy(), 1)
    assert redteam.SYNTHETIC_HASH_KEY == b"mulenet-redteam-synthetic-not-a-secret"
    after, _ = redteam.synthesize(_strategy(), 1)
    assert redteam.as_tables(before)["accounts"].equals(
        redteam.as_tables(after)["accounts"]
    )


def test_no_ground_truth_table_is_written() -> None:
    """This harness does not get to label its own accounts as mules."""
    tables = redteam.as_tables(redteam.synthesize(_strategy(), 1)[0])
    assert "ground_truth" not in tables


# --------------------------------------------------------------------------
# the Gemini path, with the model replaced
# --------------------------------------------------------------------------


def _reply(strategies: list[dict[str, Any]]) -> str:
    return json.dumps({"strategies": strategies})


def _fake_gemini(text: str | Exception) -> Any:
    """Stand-in for ``adk_agents._run_agent``, mirroring its signature."""

    def fake_run_agent(
        name: str,
        message: str,
        config: Any = None,
        sink: set[str] | None = None,
        citations: list[dict[str, str]] | None = None,
    ) -> str:
        assert name == "redteam", name
        if isinstance(text, Exception):
            raise text
        return text

    return fake_run_agent


@pytest.fixture
def config() -> AgentsConfig:
    return AgentsConfig.from_env()


def test_the_redteam_agent_is_registered_with_no_tools() -> None:
    """It must not be able to read the detection tables.

    A strategy fitted to the 24 rings already in ``ml/out/`` evades this
    dataset rather than this rule, and the only way to be sure it was not
    is for the agent to have no way to look.
    """
    assert adk_agents.AGENT_TOOLS["redteam"] == ()
    assert "redteam" in adk_agents._DESCRIPTIONS
    assert adk_agents.MAX_OUTPUT_TOKENS["redteam"] > 0


def test_the_redteam_prompt_exists_and_states_every_threshold() -> None:
    """The model cannot aim at a threshold it was not told.

    Read from the prompt file rather than asserted in prose, so a retuned
    rule with a stale prompt fails here instead of producing strategies
    aimed at numbers that moved.
    """
    from agents.config import load_prompt  # noqa: PLC0415 - one call site

    prompt = load_prompt("redteam")
    for value in (
        f"{VELOCITY_MAX_SECONDS:g}",
        str(VELOCITY_MIN_IN_DEGREE),
        str(VELOCITY_MIN_OUT_DEGREE),
        f"{VELOCITY_RATIO_RANGE[0]:g}",
        f"{VELOCITY_RATIO_RANGE[1]:g}",
    ):
        assert value in prompt, value


def test_a_model_reply_is_parsed_typed_and_run(
    monkeypatch: pytest.MonkeyPatch, cfg: MlConfig, config: AgentsConfig
) -> None:
    """End to end with only the model faked: parse, type, synthesise, detect."""
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _fake_gemini(
            _reply(
                [
                    {
                        "name": "control",
                        "description": "clears every clause",
                        "targets": "velocity",
                        "evades_clause": "ring_clustering",
                        "why_it_should_evade": "it should not",
                        "pattern": CAUGHT_PATTERN,
                    },
                    {
                        "name": "thin-fan-in",
                        "description": "holds fan-in under the floor",
                        "targets": "velocity",
                        "evades_clause": "in_degree",
                        "why_it_should_evade": "one under",
                        "pattern": dict(CAUGHT_PATTERN)
                        | EVASION_OVERRIDES["in_degree"],
                    },
                ]
            )
        ),
    )
    verdicts = redteam.run(cfg=cfg)
    assert [v.strategy.name for v in verdicts] == ["control", "thin-fan-in"]
    assert [v.caught for v in verdicts] == [True, False]


def test_a_fenced_reply_is_still_parsed(
    monkeypatch: pytest.MonkeyPatch, cfg: MlConfig
) -> None:
    """Models fence JSON more often than not; ``_parse_json`` strips it."""
    fenced = "```json\n" + _reply([{"name": "fenced", "pattern": CAUGHT_PATTERN}]) + "\n```"
    monkeypatch.setattr(adk_agents, "_run_agent", _fake_gemini(fenced))
    assert [v.strategy.name for v in redteam.run(cfg=cfg)] == ["fenced"]


def test_a_reply_with_no_strategies_array_is_a_contract_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a bad answer -- nothing to run at all, so it raises."""
    monkeypatch.setattr(
        adk_agents, "_run_agent", _fake_gemini(json.dumps({"sorry": "no"}))
    )
    with pytest.raises(RuntimeError, match="no strategies array"):
        adk_agents.generate_evasion_strategies()


def test_unparseable_replies_exhaust_the_retry_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared ledger applies here too, not just to the four older agents."""
    monkeypatch.setattr(adk_agents, "_run_agent", _fake_gemini("no json at all"))
    with pytest.raises(adk_agents.ModelUnusableError):
        adk_agents.generate_evasion_strategies()


def test_a_strategy_the_model_got_wrong_is_run_anyway(
    monkeypatch: pytest.MonkeyPatch, cfg: MlConfig
) -> None:
    """The detector's verdict is the output; the model is not pre-graded.

    Here the model claims to duck the fan-in floor and then picks a
    fan-in well above it. Filtering that out would be grading the
    model's homework with the answer key; running it reports the truth,
    which is that the strategy was caught.
    """
    monkeypatch.setattr(
        adk_agents,
        "_run_agent",
        _fake_gemini(
            _reply(
                [
                    {
                        "name": "wrong-way-round",
                        "evades_clause": "in_degree",
                        "why_it_should_evade": "claims to be under the floor",
                        "pattern": dict(CAUGHT_PATTERN)
                        | {"mules_per_collector": VELOCITY_MIN_IN_DEGREE + 9},
                    }
                ]
            )
        ),
    )
    verdict = redteam.run(cfg=cfg)[0]
    assert verdict.caught
    assert verdict.strategy.evades_clause == "in_degree"


# --------------------------------------------------------------------------
# the report carries its own caveat
# --------------------------------------------------------------------------


def test_the_json_report_carries_the_isolated_graph_caveat(cfg: MlConfig) -> None:
    """A ``missed`` row lifted into a slide is the misreading to prevent.

    So the caveat travels with the numbers rather than living only in a
    doc nobody opens beside them.
    """
    report = redteam.as_json([redteam.evaluate(_strategy(), 1, cfg)])
    assert "MISSED" in report["isolated_graph_caveat"]
    assert "does NOT transfer" in report["isolated_graph_caveat"]
    assert "XGBoost" in report["scorer_not_attacked"]


def test_the_json_report_states_the_thresholds_it_graded_against(
    cfg: MlConfig,
) -> None:
    """A verdict without the thresholds behind it is not reproducible."""
    report = redteam.as_json([redteam.evaluate(_strategy(), 1, cfg)])
    assert report["detector"]["velocity_min_in_degree"] == VELOCITY_MIN_IN_DEGREE
    assert report["detector"]["min_distinct_counterparty_banks"] == MIN_BANKS


def test_the_json_report_counts_caught_and_missed_consistently(cfg: MlConfig) -> None:
    verdicts = [
        redteam.evaluate(_strategy(), 1, cfg),
        redteam.evaluate(
            _strategy(clause="in_degree", **EVASION_OVERRIDES["in_degree"]), 2, cfg
        ),
    ]
    report = redteam.as_json(verdicts)
    assert report["strategies_run"] == 2
    assert report["caught"] == 1
    assert report["missed"] == 1
    assert report["caught"] + report["missed"] == report["strategies_run"]
