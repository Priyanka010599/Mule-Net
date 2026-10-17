"""Live evasion-pattern simulator: a judge's own pattern, ruled on for real.

Backs ``POST /simulate`` and ``GET /simulate/bounds`` in
``agents/service.py``. Every field a judge submits is fed straight into
:func:`ml.redteam.evaluate` -- the same function ``ml/redteam.py`` runs
Gemini-designed strategies through -- so the verdict on screen is a real,
per-request computation over the real velocity rule, ring clustering and
stage-2 propagation, not a canned response.

**No model call anywhere in this path.** :func:`ml.redteam._coerce_pattern`
only clamps numbers and :func:`ml.redteam.evaluate` is pure deterministic
computation over an in-memory synthetic graph that ``PATTERN_BOUNDS``
keeps tiny -- at most 4 collectors, each with up to 40 mules and 20
onward accounts, so a few hundred rows at the extreme. That is what makes
this safe to expose on a public request path with nothing beyond the
bounds already enforced in ``ml/redteam.py``: no new rate limiter earns
its keep for a computation this small.

Why this needs its own generator stand-in
------------------------------------------
``ml.redteam.synthesize()`` mints every synthetic id and bank assignment
through ``ml.redteam._generator()``, which lazily imports
``simulator.generator`` for two things only: the three bank ids
(``simulator/generator.py:47``) and its HMAC hashing convention
(``hmac_hex``, ``simulator/generator.py:198``). That module also imports
``simulator.kyc``, which imports Pillow to render KYC document images --
a dependency this service has no other reason to carry and one
``infra/Dockerfile.agents`` does not install. Pulling it in for the sake
of three strings and a one-line hash would drag an imaging library onto
a live request path for no reason connected to what the path does.

So this module re-derives the two things actually needed, byte for byte,
and monkeypatches ``ml.redteam._generator`` to hand them back instead of
importing ``simulator.generator``. ``ml.redteam._generator`` is looked up
as a plain module-global at call time (never bound at import), so the
patch below reaches every ``synthesize()`` / ``evaluate()`` call made
from this process for as long as it runs -- exactly what is wanted here,
since this service never needs the real ``simulator.generator`` for
anything else. ``agents/test_simulate.py`` asserts the copies below still
match ``simulator.generator.BANKS`` and ``hmac_hex`` exactly, so a change
to either source is caught rather than silently drifting.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

from ml import redteam
from ml.config import MlConfig
from ml.rule import (
    MIN_BANKS,
    VELOCITY_MAX_SECONDS,
    VELOCITY_MIN_IN_DEGREE,
    VELOCITY_MIN_OUT_DEGREE,
    VELOCITY_RATIO_RANGE,
)
from shared.evasion_verdict import structured_verdict

#: Mirrors ``simulator.generator.BANKS`` exactly -- see the module
#: docstring for why this is a copy rather than an import.
BANKS: tuple[str, ...] = ("BANK_A", "BANK_B", "BANK_C")


def hmac_hex(key: bytes, raw: str) -> str:
    """Mirrors ``simulator.generator.hmac_hex`` exactly (same reasoning)."""
    return hmac.new(key, raw.encode("utf-8"), hashlib.sha256).hexdigest()


#: Stands in for the ``simulator.generator`` module wherever
#: ``ml.redteam`` reaches for one, exposing only the two attributes it
#: actually uses: ``BANKS`` and ``hmac_hex``.
_LITE_GENERATOR = SimpleNamespace(BANKS=BANKS, hmac_hex=hmac_hex)

# Patched once, at import time, for the life of the process. See the
# module docstring for why this is safe and why it is scoped to this
# process rather than to one call.
redteam._generator = lambda: _LITE_GENERATOR  # noqa: SLF001 - sanctioned, see docstring

#: Fixed rather than incrementing across requests. Every call synthesises
#: and rules on its own self-contained in-memory graph -- nothing here is
#: persisted or compared across requests -- so reusing one index just
#: means two requests with identical inputs mint identical ids, which is
#: the same determinism ``ml.redteam.EPOCH`` is fixed for.
_STRATEGY_INDEX = 1


def bounds_payload() -> dict[str, Any]:
    """The pattern's field bounds and clause names, for the UI form.

    Read straight off ``ml.redteam.PATTERN_BOUNDS`` and
    ``ml.redteam.EVADABLE_CLAUSES`` rather than retyped, so a bound
    tightened there reaches the form the moment this service restarts,
    with no second place to remember to edit.
    """
    return {
        "fields": {
            name: {
                "low": low,
                "high": high,
                "default": default,
                "integer": name in redteam.INTEGER_FIELDS,
            }
            for name, (low, high, default) in redteam.PATTERN_BOUNDS.items()
        },
        "evadable_clauses": sorted(redteam.EVADABLE_CLAUSES),
    }


class PatternValidationError(ValueError):
    """One or more submitted fields are outside the documented ranges.

    ``errors`` is a list of ``{field, message, low, high}`` so a caller can put
    each message next to its slider. This is the HTTP edge's check: the
    clamping in :func:`ml.redteam._coerce_pattern` stays, because the Gemini
    path relies on it, but a judge who asked for 99 collectors should be told
    the range, not shown a verdict about 4.
    """

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


def validate_pattern(raw: dict[str, Any]) -> None:
    """Reject any present field that is not a finite number inside its bounds
    (and whole, for the integer fields). Omitted fields are fine: they take
    their defaults."""
    errors: list[dict[str, Any]] = []
    for name, (low, high, _default) in redteam.PATTERN_BOUNDS.items():
        if name not in raw or raw[name] is None:
            continue
        value = raw[name]
        problem: str | None = None
        if isinstance(value, bool) or not isinstance(value, int | float) or value != value:
            problem = "must be a number"
        elif not low <= value <= high:
            problem = f"must be between {low:g} and {high:g}"
        elif name in redteam.INTEGER_FIELDS and float(value) != int(value):
            problem = "must be a whole number"
        if problem:
            errors.append({"field": name, "message": problem, "low": low, "high": high})
    if errors:
        raise PatternValidationError(errors)


#: Pattern field each clause is driven by, and what that field alone implies
#: the clause will measure for the first collector. ``expected`` exists
#: because the two differ for speed: the feature counts from the FIRST
#: inbound credit, while ``hop_delay_seconds`` is held from the LAST one, so
#: the measured speed is the hold plus the inbound leg's own duration.
def _clause_inputs(p: redteam.EvasionPattern) -> dict[str, dict[str, Any]]:
    inbound_leg = max(p.mules_per_collector - 1, 0) * redteam.INBOUND_SPACING_SECONDS
    return {
        "passthrough_speed_seconds": {
            "input_field": "hop_delay_seconds",
            "input_value": p.hop_delay_seconds,
            "expected_measured": p.hop_delay_seconds + inbound_leg,
            "note": "speed_counts_from_first_inbound_credit",
            "decomposition": {
                "hold_seconds": p.hop_delay_seconds,
                "inbound_leg_seconds": inbound_leg,
                "inbound_spacing_seconds": redteam.INBOUND_SPACING_SECONDS,
            },
        },
        "passthrough_ratio": {
            "input_field": "retained_fraction",
            "input_value": p.retained_fraction,
            "expected_measured": 1.0 - p.retained_fraction,
            "note": "ratio_is_one_minus_retained_for_the_first_collector",
        },
        "in_degree": {
            "input_field": "mules_per_collector",
            "input_value": p.mules_per_collector,
            "expected_measured": p.mules_per_collector,
            "note": None,
        },
        "out_degree": {
            "input_field": "onward_accounts",
            "input_value": p.onward_accounts,
            "expected_measured": p.onward_accounts,
            "note": None,
        },
        "distinct_counterparty_banks": {
            "input_field": "counterparty_banks",
            "input_value": p.counterparty_banks,
            "expected_measured": p.counterparty_banks,
            "note": None,
        },
    }


def _thresholds() -> dict[str, dict[str, float | None]]:
    """The rule's bounds, imported from ``ml.rule`` and never retyped."""
    return {
        "passthrough_speed_seconds": {"min": 0.0, "max": VELOCITY_MAX_SECONDS},
        "passthrough_ratio": {"min": VELOCITY_RATIO_RANGE[0], "max": VELOCITY_RATIO_RANGE[1]},
        "in_degree": {"min": float(VELOCITY_MIN_IN_DEGREE), "max": None},
        "out_degree": {"min": float(VELOCITY_MIN_OUT_DEGREE), "max": None},
        "distinct_counterparty_banks": {"min": float(MIN_BANKS), "max": None},
    }


def clause_table(
    pattern: redteam.EvasionPattern, clause_values: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per clause: the slider input, what it implies, what was measured, the
    bounds, and pass/fail -- per collector and summarised.

    ``measured`` is the first collector's value (the one the inputs describe
    exactly; later collectors in the chain are also credited by their
    predecessor and so measure slightly differently). ``passes`` is true when
    the clause passes on at least one collector, the same reading the verdict
    uses to decide which clause actually stopped the pattern.
    """
    inputs, bounds = _clause_inputs(pattern), _thresholds()
    collectors = list(clause_values.values())
    rows: list[dict[str, Any]] = []
    for clause, meta in inputs.items():
        per = [c[clause] for c in collectors if clause in c]
        rows.append(
            {
                "clause": clause,
                **meta,
                "measured": per[0]["value"] if per else None,
                "measured_by_collector": [c["value"] for c in per],
                "min": bounds[clause]["min"],
                "max": bounds[clause]["max"],
                "passes": any(c["passes"] for c in per),
                "passes_on_collectors": sum(1 for c in per if c["passes"]),
                "collectors": len(per),
            }
        )
    return rows


def _describe(
    pattern: redteam.EvasionPattern, evades_clause: str
) -> tuple[str, str, str]:
    """Name, description and rationale for a judge-built pattern.

    A Gemini strategy carries these three as model output
    (``agents/prompts/redteam.md``); a judge's own pattern has no model
    behind it, so they are plain sentences derived from the numbers
    themselves -- exactly what the task brief asks for rather than
    inventing reasoning nobody supplied.
    """
    name = "judge-built pattern"
    description = (
        f"{pattern.collectors} collector(s), {pattern.mules_per_collector} "
        f"mule(s) feeding each, {pattern.onward_accounts} onward account(s), "
        f"a {pattern.hop_delay_seconds:g}s hold before forwarding, "
        f"{pattern.retained_fraction:.0%} retained, spread across "
        f"{pattern.counterparty_banks} bank(s)."
        + (" Device reused across every account." if pattern.reuse_device else "")
        + (" KYC id reused across every account." if pattern.reuse_kyc_id else "")
    )
    why = f"Hand-picked live, to test the {evades_clause!r} clause specifically."
    return name, description, why


def run_simulation(raw: dict[str, Any], evades_clause: str) -> dict[str, Any]:
    """Clamp, synthesise and rule on one judge-submitted pattern, right now.

    Reuses exactly what ``ml/redteam.py`` runs a Gemini strategy through:
    :func:`ml.redteam._coerce_pattern` for the same bounds enforcement
    (never reimplemented) and :func:`ml.redteam.evaluate` for the actual
    detection run -- features, the velocity decision, ring clustering and
    stage-2 propagation, in that order. Returns a dict shaped like one
    entry of ``ml.redteam.as_json()['verdicts']``, so a UI already reading
    that shape needs no second parser.

    ``evades_clause`` is never validated against
    ``ml.redteam.EVADABLE_CLAUSES``: same treatment
    :func:`ml.redteam.parse_strategies` gives a model's claim, kept
    verbatim rather than corrected, because what was *aimed at* is part
    of the record even when it names the wrong clause.
    """
    pattern = redteam._coerce_pattern(raw)  # noqa: SLF001 - sanctioned, see module docstring
    name, description, why = _describe(pattern, evades_clause)
    strategy = redteam.EvasionStrategy(
        name=name,
        description=description,
        targets="judge-specified pattern",
        evades_clause=evades_clause,
        why_it_should_evade=why,
        pattern=pattern,
    )
    verdict = redteam.evaluate(strategy, _STRATEGY_INDEX, MlConfig.from_env())
    structured = structured_verdict(
        caught_by=verdict.caught_by,
        clause_values=verdict.clause_values,
        aimed_clause=verdict.strategy.evades_clause,
        flagged_accounts=verdict.flagged_accounts,
        flagged_collectors=verdict.flagged_collectors,
        rings_formed=verdict.rings_formed,
        stage2_alertable=verdict.stage2_alertable,
    )
    return {
        **structured,
        "clauses": clause_table(pattern, verdict.clause_values),
        "name": verdict.strategy.name,
        "description": verdict.strategy.description,
        "targets": verdict.strategy.targets,
        "aimed_at_clause": verdict.strategy.evades_clause,
        "why_it_should_evade": verdict.strategy.why_it_should_evade,
        "pattern": asdict(verdict.strategy.pattern),
        "caught": verdict.caught,
        "caught_by": verdict.caught_by,
        "reasoning": verdict.reasoning,
        "flagged_accounts": verdict.flagged_accounts,
        "flagged_collectors": verdict.flagged_collectors,
        "rings_formed": verdict.rings_formed,
        "stage2_alertable": verdict.stage2_alertable,
        "stage2_context": verdict.stage2_context,
        "clause_values": verdict.clause_values,
    }
