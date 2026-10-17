"""Gemini designs evasion patterns; the real detector rules on them.

What this is
------------
Gemini is told MuleNet's five velocity thresholds and its two-stage ring
logic, and asked for transfer patterns that move the same money while
failing one of those clauses. Each strategy comes back as numbers --
fan-in, fan-out, hop delay, retained fraction, bank spread, credential
reuse -- and this module **executes** them: it synthesises the accounts
and transfers the numbers imply, runs them through
:func:`ml.features.build_features`, :func:`ml.rule.velocity_decision`,
:func:`ml.publish.high_risk_components` and :func:`ml.stage2.propagate`,
and reports caught or missed per strategy.

Nothing about the detector is reimplemented here. The five thresholds are
read off ``ml.rule`` at report time rather than restated (see
:func:`clause_report`), so a tuning change in ``ml/rule.py`` moves this
harness with it instead of leaving it grading against last week's rule.

Why this is worth having
------------------------
Every other number in this repo measures the detector against the attack
patterns ``simulator/generator.py`` was written to produce. Those
patterns were designed by the same people who then built the rule, which
is the oldest problem in detection work: the test set and the detector
share an author. Gemini is not a better adversary than a real launderer,
but it is an *independent* one, and it produces patterns nobody on this
project chose.

**A miss is the valuable outcome.** A harness that reports a clean sweep
has either found a genuinely robust rule or a generator that only
proposes attacks the rule already stops, and the two look identical from
the outside. The verdicts below are printed as they come out.

What a verdict here does and does not mean
------------------------------------------
Each strategy runs as an **isolated graph**: the only transfers in the
feature table are the ones the strategy itself creates. That cuts both
ways and the two directions are not symmetric.

``caught`` *is* solid evidence.
    ``in_degree``, ``out_degree`` and ``distinct_counterparty_banks`` are
    lower-bound clauses over monotone counts. Embedding the same pattern
    in a bank's real traffic can only raise those counts, so a pattern
    the rule catches in isolation is caught at least as easily in
    company.

``missed`` is **not** evidence that the pattern survives in production.
    It means the pattern clears the thresholds *on its own transactions*.
    A real collector account also carries ordinary traffic, and that
    traffic contributes to the same degree counts -- a mule holding
    fan-in at 15 to stay under a floor of 16 is one legitimate salary
    credit away from firing. It also shifts ``passthrough_speed_seconds``
    (a median over bursts) and ``passthrough_ratio`` in directions this
    harness cannot predict.

So the honest reading of a miss is "this is a hole in the rule as
written, worth a look", never "this evades MuleNet". Closing the gap
means re-running the surviving strategies with the pattern injected into
``simulator/out/`` rather than standing alone, which needs the simulator
tables and is not what this module does.

Two further limits, stated rather than discovered later:

* The **XGBoost scorer is not in this loop.** Rings come from the
  velocity rule, so that is what is attacked; an account the rule misses
  may still score high in the alert queue, and this harness does not
  check.
* The synthetic accounts have **no KYC or device history** beyond what a
  strategy's ``reuse_device`` / ``reuse_kyc_id`` flags create, so the
  stage-2 verdict is about propagation mechanics, not about how dense
  legitimate credential sharing is in the real population. In the real
  data 4,102 of the 4,773 accounts carrying an identity edge are not
  complicit (``ml/stage2.py``); nothing here reproduces that.

Running it
----------
::

    MULENET_AGENT_MODE=vertex python -m ml.redteam

Needs Vertex credentials, and no simulator or ``ml/out/`` artefact: the
graph is built in memory. Every id is an HMAC digest of a loop counter
under :data:`SYNTHETIC_HASH_KEY`, which is a literal in this file because
nothing it protects is real -- there is no customer data here to leak,
and a fixed key is what makes a run reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from ml.config import MlConfig
from ml.features import build_features, full_window_cutoff
from ml.publish import high_risk_components
from ml.rule import (
    MIN_BANKS,
    VELOCITY_MAX_SECONDS,
    VELOCITY_MIN_IN_DEGREE,
    VELOCITY_MIN_OUT_DEGREE,
    VELOCITY_RATIO_RANGE,
    velocity_decision,
)
from ml.stage2 import identity_edges, propagate

#: HMAC key for the synthetic ids. A literal on purpose: these accounts
#: do not exist, so there is no secret here, and a fixed key makes a run
#: reproducible. It is deliberately *not* read from the environment, so
#: nobody can point this harness at the real hashing key and have it emit
#: ids that collide with production ones.
SYNTHETIC_HASH_KEY = b"mulenet-redteam-synthetic-not-a-secret"

#: Prefix on every synthetic id's HMAC input, so a redteam account can
#: never be confused with a simulator one even if the keys were swapped.
ID_NAMESPACE = "redteam"

#: Wall-clock anchor for a run. Fixed rather than ``now()``: the feature
#: table is point-in-time and a moving anchor would make two runs of the
#: same strategy incomparable.
EPOCH = datetime(2025, 2, 1, 9, 0, 0, tzinfo=UTC)

#: Seconds between the mule credits landing on one collector. Small
#: enough that the inbound leg is one burst at the default 1800s gap, so
#: the hop delay a strategy chooses is what decides the burst split
#: rather than the spacing of the credits.
INBOUND_SPACING_SECONDS = 60

#: Seconds between a collector's outbound debits, for the same reason.
OUTBOUND_SPACING_SECONDS = 30

#: Field -> (low, high, default), inclusive. The prompt states these
#: ranges and this dict enforces them; a model that ignores one gets its
#: value clamped and the clamp reported, because silently running a
#: different pattern from the one the model designed would make the
#: verdict about the wrong thing.
PATTERN_BOUNDS: dict[str, tuple[float, float, float]] = {
    "collectors": (1, 4, 1),
    "mules_per_collector": (1, 40, 8),
    "onward_accounts": (1, 20, 4),
    "hop_delay_seconds": (0, 604800, 600),
    "amount_per_inbound_hop": (100.0, 1000000.0, 5000.0),
    "retained_fraction": (0.0, 0.99, 0.05),
    "counterparty_banks": (1, 3, 2),
}

#: Pattern fields that must come back as ints.
INTEGER_FIELDS: frozenset[str] = frozenset(
    {"collectors", "mules_per_collector", "onward_accounts", "counterparty_banks"}
)

#: The clause names a strategy may claim to evade. ``ring_clustering`` is
#: not a threshold -- it is the requirement that a flagged account's
#: counterparty also be flagged -- and is listed because a strategy
#: aimed at it is a different and legitimate attack.
EVADABLE_CLAUSES: frozenset[str] = frozenset(
    {
        "passthrough_speed_seconds",
        "passthrough_ratio",
        "in_degree",
        "out_degree",
        "distinct_counterparty_banks",
        "ring_clustering",
    }
)


@dataclass(frozen=True)
class EvasionPattern:
    """The executable half of a strategy: numbers, already clamped.

    Separate from :class:`EvasionStrategy` because this is the part that
    has to be valid to run at all, while the prose around it is only
    ever reported. ``clamped`` names every field the model put outside
    its bound, so a verdict can be read next to the fact that it was not
    quite the pattern the model asked for.
    """

    collectors: int
    mules_per_collector: int
    onward_accounts: int
    hop_delay_seconds: float
    amount_per_inbound_hop: float
    retained_fraction: float
    counterparty_banks: int
    reuse_device: bool
    reuse_kyc_id: bool
    clamped: tuple[str, ...] = ()

    @property
    def accounts_per_collector(self) -> int:
        """Collector plus its mules plus its onward accounts."""
        return 1 + self.mules_per_collector + self.onward_accounts


@dataclass(frozen=True)
class EvasionStrategy:
    """One strategy as the model described it, plus its clamped pattern."""

    name: str
    description: str
    targets: str
    evades_clause: str
    why_it_should_evade: str
    pattern: EvasionPattern


@dataclass
class StrategyVerdict:
    """What the real detector did with one strategy's synthetic graph.

    ``caught`` is true when the velocity rule fired on a collector, with
    ``caught_by`` saying whether that produced a ring or only an
    account-level alert. Stage-2 propagation is reported but never
    decides the verdict: it cannot start a ring, so it has nothing to
    work with on a pattern stage 1 missed entirely.

    ``stage2_context`` is counted and deliberately excluded from
    ``caught``. A same-bank identity link is downgraded to context by the
    ruling, and counting context as a catch would claim credit for an
    accusation MuleNet refuses to make.
    """

    strategy: EvasionStrategy
    flagged_accounts: int
    flagged_collectors: int
    rings_formed: int
    stage2_alertable: int
    stage2_context: int
    #: Collector id -> that collector's five clause values. Keyed by
    #: account rather than flattened, because the chain makes the
    #: collectors genuinely different rows.
    clause_values: dict[str, dict[str, Any]]
    caught_by: str
    reasoning: str

    @property
    def caught(self) -> bool:
        return self.caught_by != "missed"


def _coerce_pattern(raw: dict[str, Any]) -> EvasionPattern:
    """Read a model's ``pattern`` block into bounded numbers.

    A missing field takes its default and is **not** reported as clamped:
    the model declined to choose, so there was no choice to override. A
    field present and out of range is clamped and named, because that one
    was a decision this harness overruled.
    """
    values: dict[str, Any] = {}
    clamped: list[str] = []
    for name, (low, high, default) in PATTERN_BOUNDS.items():
        if name not in raw:
            values[name] = int(default) if name in INTEGER_FIELDS else float(default)
            continue
        try:
            given = float(raw[name])
        except (TypeError, ValueError):
            values[name] = int(default) if name in INTEGER_FIELDS else float(default)
            clamped.append(f"{name} (unreadable, used {default})")
            continue
        bounded = min(max(given, low), high)
        if bounded != given:
            clamped.append(f"{name} {given:g} -> {bounded:g}")
        values[name] = int(bounded) if name in INTEGER_FIELDS else float(bounded)

    return EvasionPattern(
        **values,
        reuse_device=bool(raw.get("reuse_device", False)),
        reuse_kyc_id=bool(raw.get("reuse_kyc_id", False)),
        clamped=tuple(clamped),
    )


def parse_strategies(payload: list[dict[str, Any]]) -> list[EvasionStrategy]:
    """Type the model's ``strategies`` array, dropping nothing.

    A strategy with a nonsense ``evades_clause`` keeps it verbatim rather
    than being corrected or discarded: what the model *claimed* it was
    attacking is half the result, and rewriting the claim would make the
    report agree with itself by construction.
    """
    out: list[EvasionStrategy] = []
    for index, item in enumerate(payload, start=1):
        out.append(
            EvasionStrategy(
                name=str(item.get("name") or f"unnamed-{index}"),
                description=str(item.get("description", "")).strip(),
                targets=str(item.get("targets", "unstated")),
                evades_clause=str(item.get("evades_clause", "unstated")),
                why_it_should_evade=str(item.get("why_it_should_evade", "")).strip(),
                pattern=_coerce_pattern(item.get("pattern") or {}),
            )
        )
    return out


def _generator() -> Any:
    """Import ``simulator.generator``, for :data:`BANKS` and ``hmac_hex``.

    Imported LAZILY, and from there rather than restated here, for two
    separate reasons.

    From there, because ``simulator/generator.py:47`` is where the three
    bank ids are defined and where the id-hashing convention lives. A
    second copy in ``ml/`` would be a second definition, and the one that
    drifts is always the copy (see CLAUDE.md: derive lists from their
    source of truth).

    Lazily, because that module imports ``simulator.kyc``, which imports
    Pillow to render document images. ``ml/`` has no other reason to pull
    an imaging library in, and a module-level import would make
    ``python -m ml.redteam`` die on a missing image library for the sake
    of a three-element tuple.
    """
    from simulator import generator  # noqa: PLC0415 - heavy transitive import

    return generator


def _synthetic_id(*parts: object) -> str:
    """One 64-hex id, minted the way the simulator mints them.

    Every id in this module comes through here: accounts, transfers,
    devices, documents and submissions. Same HMAC-SHA256 shape as a real
    hashed account id, so the whole pipeline treats these like any other
    row and nothing downstream can special-case them -- which is the
    point, since a detector that recognised its own test data would
    prove nothing.
    """
    raw = ":".join([ID_NAMESPACE, *(str(part) for part in parts)])
    return _generator().hmac_hex(SYNTHETIC_HASH_KEY, raw)


def _account_id(strategy_index: int, local: int) -> str:
    """A synthetic hashed account id for one account in one strategy."""
    return _synthetic_id("acct", strategy_index, f"{local:05d}")


@dataclass
class _Counter:
    """Two independent sequences: local account numbers and transfer ids.

    They were plain locals first, and chaining the collectors made that a
    bug waiting to happen -- the loop that mints accounts and the loop
    that mints transfers stopped advancing together the moment one
    onward slot reused an existing account.
    """

    account: int = 0
    transfer: int = 0

    def take(self) -> int:
        """Next local account number."""
        value = self.account
        self.account += 1
        return value

    def txn(self) -> int:
        """Next transfer number."""
        value = self.transfer
        self.transfer += 1
        return value


@dataclass
class _Graph:
    """Accumulator for one strategy's synthetic tables."""

    accounts: list[dict[str, Any]] = field(default_factory=list)
    transfers: list[dict[str, Any]] = field(default_factory=list)
    devices: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    submissions: list[dict[str, Any]] = field(default_factory=list)
    collectors: list[str] = field(default_factory=list)

    def add_account(self, account_id: str, bank: str) -> None:
        self.accounts.append(
            {
                "hashed_account_id": account_id,
                "bank_id": bank,
                # Opened well before EPOCH: account_age_days is a feature,
                # and a zero-age account would be an artefact of this
                # harness rather than part of any strategy.
                "opened_at": EPOCH - timedelta(days=400),
                "account_type": "personal",
                "status": "active",
            }
        )

    def add_transfer(
        self, index: int, src: str, dst: str, amount: float, when: datetime
    ) -> None:
        self.transfers.append(
            {
                "transfer_id": _synthetic_id("txn", index),
                "from_hashed_account_id": src,
                "to_hashed_account_id": dst,
                # NUMERIC(15,2) everywhere else in this repo, so Decimal
                # here too: ml.features casts to float itself, and money
                # arriving as a float would be the one place in the
                # pipeline it does.
                "amount": Decimal(f"{amount:.2f}"),
                "currency": "INR",
                "timestamp": when,
                "channel": "imps",
            }
        )


def synthesize(strategy: EvasionStrategy, strategy_index: int) -> tuple[_Graph, list[str]]:
    """Build the accounts and transfers one strategy's pattern implies.

    Shape, per collector: ``mules_per_collector`` accounts each credit
    the collector once, spaced :data:`INBOUND_SPACING_SECONDS` apart;
    ``hop_delay_seconds`` later the collector pays
    ``1 - retained_fraction`` of the total onward, split evenly over
    ``onward_accounts`` accounts.

    Bank assignment is what makes the spread clause reachable: the
    collector sits at ``BANK_A`` and its counterparties are dealt round
    robin over the first ``counterparty_banks`` banks. With
    ``counterparty_banks == 1`` every counterparty is also at ``BANK_A``,
    so ``cb_distinct_counterparty_banks`` is 1 and the rule cannot fire
    however extreme the rest of the pattern is.

    Returns the graph and the collector ids, which are the accounts the
    verdict is about -- a mule that only ever sends one transfer is not
    what the velocity rule is looking for and never was.
    """
    pattern = strategy.pattern
    graph = _Graph()
    counter = _Counter()
    all_banks = _generator().BANKS
    banks = list(all_banks[: pattern.counterparty_banks])

    # Collectors are minted up front because they are CHAINED: each one
    # pays the next as the first of its onward accounts. Without that
    # chain no ring can ever form here -- clustering needs a transfer
    # whose two endpoints are both flagged, and a mule sending one
    # transfer has out_degree 1 and in_degree 0, so it never fires. A
    # strategy asking for one collector therefore gets an account-level
    # alert at best, which is a property of the pattern and is reported
    # as such, not a gap in this harness.
    collectors = [
        _account_id(strategy_index, counter.take()) for _ in range(pattern.collectors)
    ]
    for collector in collectors:
        graph.add_account(collector, all_banks[0])
    graph.collectors.extend(collectors)

    inbound_from_chain = 0.0
    # Where the next collector's inbound leg begins. The chain is
    # sequenced in TIME, not just in topology: the predecessor's payment
    # is the first credit of the successor's inbound leg, so the money
    # walks the chain instead of every collector firing at once.
    #
    # Running them concurrently was tried and was actively misleading.
    # With a shared start, a downstream collector's own outbound debit
    # landed seconds after the chain credit arrived, which is a
    # 0-second pass-through -- so a strategy built to defeat the speed
    # clause by holding funds for two hours produced a collector that
    # passed that clause, and the harness reported the strategy caught on
    # the strength of its own timing artefact.
    next_inbound_start = EPOCH
    for collector_index, collector in enumerate(collectors):
        inbound_start = next_inbound_start
        # Money already credited by the predecessor in the chain. It is
        # part of what this collector received, so it belongs in the
        # ratio's denominator; leaving it out made a downstream collector
        # pay out more than it took in and pushed passthrough_ratio above
        # the band the strategy chose.
        total_in = inbound_from_chain
        # Every collector after the first is also credited by its
        # predecessor in the chain, and that credit counts toward
        # in_degree like any other. So it gets one fewer mule, and every
        # collector ends up with exactly the fan-in the strategy asked
        # for. Without this, chaining silently handed a strategy that
        # specified 15 a downstream collector with 16 -- over the floor
        # it was built to duck, and the verdict was then about a pattern
        # nobody designed.
        mule_count = pattern.mules_per_collector - (1 if collector_index else 0)
        # Slot 0 of the inbound leg is the chain credit for every
        # collector but the first, so its mules start one slot later and
        # the whole leg still spans mules_per_collector slots.
        first_mule_slot = 1 if collector_index else 0
        for mule_index in range(mule_count):
            mule = _account_id(strategy_index, counter.take())
            graph.add_account(mule, banks[mule_index % len(banks)])
            when = inbound_start + timedelta(
                seconds=(first_mule_slot + mule_index) * INBOUND_SPACING_SECONDS
            )
            graph.add_transfer(
                counter.txn(), mule, collector, pattern.amount_per_inbound_hop, when
            )
            total_in += pattern.amount_per_inbound_hop

        # The hop delay is measured from the LAST inbound credit, not the
        # first. A strategy setting it to 0 means "forward the moment the
        # money is all in", which is the behaviour the rule is named for;
        # measuring from the first credit would silently add the inbound
        # leg's own duration to every delay.
        last_inbound = inbound_start + timedelta(
            seconds=max(pattern.mules_per_collector - 1, 0) * INBOUND_SPACING_SECONDS
        )
        outbound_start = last_inbound + timedelta(seconds=pattern.hop_delay_seconds)
        paid_out = total_in * (1.0 - pattern.retained_fraction)
        per_onward = paid_out / pattern.onward_accounts

        # The next collector takes the first onward slot, so the money
        # actually walks the chain rather than the chain being drawn
        # alongside it. The last collector has no successor and pays all
        # its onward slots to fresh exit accounts.
        successor = (
            collectors[collector_index + 1]
            if collector_index + 1 < len(collectors)
            else None
        )
        for onward_index in range(pattern.onward_accounts):
            if onward_index == 0 and successor is not None:
                onward = successor
            else:
                onward = _account_id(strategy_index, counter.take())
                graph.add_account(onward, banks[onward_index % len(banks)])
            when = outbound_start + timedelta(
                seconds=onward_index * OUTBOUND_SPACING_SECONDS
            )
            graph.add_transfer(counter.txn(), collector, onward, per_onward, when)

        # The successor takes onward slot 0, so that payment is both the
        # amount it has been credited and the instant its own inbound leg
        # begins. Advancing the clock here is what sequences the chain.
        inbound_from_chain = per_onward if successor is not None else 0.0
        next_inbound_start = outbound_start

    _add_identity(graph, strategy, strategy_index)
    return graph, graph.collectors


def _add_identity(graph: _Graph, strategy: EvasionStrategy, strategy_index: int) -> None:
    """Attach device and KYC rows, sharing credentials only if asked.

    Every account gets its own device and its own KYC document, because
    an account with neither is invisible to stage 2 for a reason that has
    nothing to do with the strategy. ``reuse_device`` and
    ``reuse_kyc_id`` then overwrite the shared field with one value
    across the whole strategy, which is the condition stage-2
    propagation follows.
    """
    pattern = strategy.pattern
    shared_device = _synthetic_id("device", "shared", strategy_index)
    shared_id = _synthetic_id("id", "shared", strategy_index)
    # Credentials are seen at onboarding, long before the transfers. A
    # sighting after an account's cutoff is invisible to the
    # point-in-time join in ml.features and ml.stage2, so a link dated
    # mid-run would be dropped for the wrong reason.
    seen_at = EPOCH - timedelta(days=200)

    for position, row in enumerate(graph.accounts):
        account_id = row["hashed_account_id"]
        fingerprint = (
            shared_device
            if pattern.reuse_device
            else _synthetic_id("device", strategy_index, position)
        )
        graph.devices.append(
            {
                "device_id": _synthetic_id("deviceid", strategy_index, position),
                "hashed_account_id": account_id,
                "device_fingerprint": fingerprint,
                "first_seen": seen_at,
                "last_seen": seen_at,
            }
        )
        id_number = (
            shared_id
            if pattern.reuse_kyc_id
            else _synthetic_id("id", strategy_index, position)
        )
        graph.documents.append(
            {
                "doc_id": _synthetic_id("doc", strategy_index, position),
                "hashed_account_id": account_id,
                "doc_type": "national_id",
                "hashed_id_number": id_number,
                "normalized_address": f"synthetic address {position}",
                "photo_embedding_ref": None,
            }
        )
        graph.submissions.append(
            {
                "submission_id": _synthetic_id("sub", strategy_index, position),
                "hashed_account_id": account_id,
                "bank_id": row["bank_id"],
                "submitted_at": seen_at,
            }
        )


def as_tables(graph: _Graph) -> dict[str, pd.DataFrame]:
    """The five frames ``ml.features.build_features`` reads.

    ``ground_truth`` is absent on purpose. ``build_features`` does not
    read it, and writing one here would mean this harness declaring which
    of its own accounts are mules -- a label the detector is not being
    asked about and that nothing downstream should be able to join to.
    """
    return {
        "accounts": pd.DataFrame(graph.accounts),
        "transfers": pd.DataFrame(graph.transfers),
        "devices": pd.DataFrame(graph.devices),
        "kyc_submissions": pd.DataFrame(graph.submissions),
        "kyc_documents": pd.DataFrame(graph.documents),
    }


def clause_report(row: pd.Series) -> dict[str, Any]:
    """One account's five clause values next to the thresholds they face.

    The thresholds are the imported constants, never retyped, so this
    cannot drift from :func:`ml.rule.velocity_decision`. This is the
    "reasoning" a verdict carries: "missed" on its own is an assertion,
    while "in_degree 12 against a floor of 16" is a finding somebody can
    act on.
    """
    speed = float(row["cb_passthrough_speed_seconds"])
    ratio = float(row["cb_passthrough_ratio"])
    return {
        "passthrough_speed_seconds": {
            "value": speed,
            "passes": 0.0 <= speed <= VELOCITY_MAX_SECONDS,
            "requirement": f"0 <= x <= {VELOCITY_MAX_SECONDS:g}",
        },
        "passthrough_ratio": {
            "value": ratio,
            "passes": VELOCITY_RATIO_RANGE[0] <= ratio <= VELOCITY_RATIO_RANGE[1],
            "requirement": (
                f"{VELOCITY_RATIO_RANGE[0]:g} <= x <= {VELOCITY_RATIO_RANGE[1]:g}"
            ),
        },
        "in_degree": {
            "value": int(row["cb_in_degree"]),
            "passes": int(row["cb_in_degree"]) >= VELOCITY_MIN_IN_DEGREE,
            "requirement": f"x >= {VELOCITY_MIN_IN_DEGREE}",
        },
        "out_degree": {
            "value": int(row["cb_out_degree"]),
            "passes": int(row["cb_out_degree"]) >= VELOCITY_MIN_OUT_DEGREE,
            "requirement": f"x >= {VELOCITY_MIN_OUT_DEGREE}",
        },
        "distinct_counterparty_banks": {
            "value": int(row["cb_distinct_counterparty_banks"]),
            "passes": int(row["cb_distinct_counterparty_banks"]) >= MIN_BANKS,
            "requirement": f"x >= {MIN_BANKS}",
        },
    }


def _reasoning(
    verdict_source: str,
    clauses: dict[str, dict[str, Any]],
    strategy: EvasionStrategy,
) -> str:
    """One sentence saying what the detector did and which clause decided.

    Names the clause that *actually* failed, and says so explicitly when
    that is not the clause the model aimed at. A strategy that evades by
    accident is a weaker result than one that evades as designed, and the
    difference is invisible unless it is written down.

    On a miss, a clause counts as failing only when it fails on **every**
    collector. A clause that holds one collector back while another one
    fires is not what stopped the pattern, and naming it would point a
    reader at the wrong threshold.
    """
    per_collector = list(clauses.values())
    failed = [
        name
        for name in (per_collector[0] if per_collector else {})
        if all(not c[name]["passes"] for c in per_collector)
    ]
    if verdict_source == "missed":
        if not failed:  # pragma: no cover - every clause passing means flagged
            return (
                "Reported as missed with every clause passing on the "
                "collector. That combination should be impossible -- the "
                "velocity rule is exactly the conjunction of these five -- so "
                "read it as a bug in this harness, not as a detector result."
            )
        first = per_collector[0]
        detail = ", ".join(
            f"{name}={first[name]['value']} (needs {first[name]['requirement']})"
            for name in failed
        )
        aimed = strategy.evades_clause
        note = (
            ""
            if aimed in failed
            else (
                f" The strategy aimed at {aimed!r}, which passed; "
                "this miss is not the designed one."
            )
        )
        return f"Evaded. Failing clause(s) on the collector: {detail}.{note}"
    if verdict_source == "velocity_ring":
        return (
            "Caught by the stage-1 velocity rule, and a ring formed: every "
            "clause passed on the collector, so the pattern never left the "
            "thresholds it was designed to clear (it aimed at "
            f"{strategy.evades_clause!r})."
        )
    if verdict_source == "velocity_alert_no_ring":
        return (
            "Caught by the stage-1 velocity rule as an account-level ALERT, "
            "but no ring formed: nothing the collector transacts with was "
            "flagged too, so clustering had nothing to join it to. The "
            "account reaches an analyst's queue; the ring view does not show "
            "it. A pattern with one collector cannot form a ring here by "
            "construction -- see synthesize()."
        )
    raise AssertionError(f"unknown verdict source {verdict_source!r}")


def evaluate(
    strategy: EvasionStrategy,
    strategy_index: int,
    cfg: MlConfig | None = None,
) -> StrategyVerdict:
    """Run one strategy's synthetic graph through the real detectors.

    The four real stages, in the order ``ml/publish.py`` runs them:
    features, the velocity decision, connected components over
    both-endpoints-flagged transfers, and one hop of identity
    propagation out of whatever components that produced.
    """
    cfg = cfg or MlConfig.from_env()
    graph, collectors = synthesize(strategy, strategy_index)
    tables = as_tables(graph)

    features = build_features(cfg=cfg, tables=tables)
    flagged = velocity_decision(features, "cb_", require_bank_spread=True)
    flagged_ids = pd.Index(features.loc[flagged, "hashed_account_id"])
    flagged_collectors = sorted(set(collectors) & set(flagged_ids))

    cutoff = pd.Timestamp(tables["transfers"]["timestamp"].max())
    rings = high_risk_components(tables["transfers"], flagged_ids, cutoff)

    bank = tables["accounts"].set_index("hashed_account_id")["bank_id"]
    edges = identity_edges(tables, full_window_cutoff(tables))
    added = propagate(rings, edges, bank) if rings else {}
    # Counted over every account in the strategy, not just the
    # collectors. Stage 2 grows a ring outwards, so the accounts it adds
    # are by definition the ones NOT already in it -- restricting this to
    # collectors counted the one population propagation cannot reach and
    # reported 0 every time.
    links = [link for ring_links in added.values() for link in ring_links]
    alertable = [link for link in links if link.alertable]
    context = [link for link in links if not link.alertable]

    # One clause table per collector, not one for the first. They are
    # built from the same numbers but they are NOT identical: a collector
    # in the middle of the chain is credited by its predecessor and pays
    # its successor, which moves passthrough_speed (a median over bursts)
    # in a way the strategy did not specify. Reporting only the first
    # made a strategy look evaded while the collector that actually
    # fired sat one row below, unprinted.
    indexed = features.set_index("hashed_account_id")
    clauses = {
        collector: clause_report(indexed.loc[collector]) for collector in collectors
    }

    # Two ways to be caught, and they are not the same strength of
    # result. MuleNet's own distinction is kept: "A component of one is
    # an alert, not a ring" (ml.publish.high_risk_components), so a
    # flagged collector with nothing flagged to join it to is a real
    # account-level detection that produced no ring. Collapsing the two
    # would either overstate the detector (calling an alert a ring) or
    # understate it (calling a fired rule a miss); both were written and
    # both read wrong.
    #
    # There is deliberately no third "caught by stage 2" branch. Stage 2
    # cannot create a ring -- that is the safety property ml/stage2.py is
    # built around and ml/test_publish.py asserts -- so in a graph holding
    # one strategy and nothing else, a pattern that clears the velocity
    # rule leaves propagation with nothing to propagate out of. Stage 2
    # can only ever widen a catch stage 1 already made, and the
    # stage2_alertable / stage2_context counts report that widening. A
    # branch here would be unreachable code implying a rescue that the
    # design forbids.
    caught_by = (
        "velocity_ring"
        if flagged_collectors and rings
        else "velocity_alert_no_ring"
        if flagged_collectors
        else "missed"
    )

    return StrategyVerdict(
        strategy=strategy,
        flagged_accounts=len(flagged_ids),
        flagged_collectors=len(flagged_collectors),
        rings_formed=len(rings),
        stage2_alertable=len(alertable),
        stage2_context=len(context),
        clause_values=clauses,
        caught_by=caught_by,
        reasoning=_reasoning(caught_by, clauses, strategy),
    )


def run(
    count: int = 3,
    cfg: MlConfig | None = None,
    strategies: list[EvasionStrategy] | None = None,
) -> list[StrategyVerdict]:
    """Generate strategies with Gemini and rule on each one.

    ``strategies`` is the seam the tests use: passing them skips the
    model call entirely, so the synthesis and detection halves are
    testable offline. Left ``None``, this calls Gemini.

    The ADK import is lazy, matching every other entry point in
    ``agents/``: ``ml/`` must stay importable and testable with ADK
    absent, and it is absent in most environments this runs in.
    """
    if strategies is None:
        from agents.adk_agents import (  # noqa: PLC0415 - optional dependency
            generate_evasion_strategies,
        )

        strategies = parse_strategies(generate_evasion_strategies(count))
    return [
        evaluate(strategy, index, cfg)
        for index, strategy in enumerate(strategies, start=1)
    ]


def _print_verdicts(verdicts: list[StrategyVerdict]) -> None:
    """Print each strategy, its verdict and the clause that decided it."""
    caught = sum(1 for v in verdicts if v.caught)
    print("")
    print("RED TEAM: Gemini-designed evasion strategies vs the real detector")
    print(f"  strategies: {len(verdicts)}   caught: {caught}   missed: {len(verdicts) - caught}")

    for index, verdict in enumerate(verdicts, start=1):
        strategy = verdict.strategy
        mark = "CAUGHT" if verdict.caught else "MISSED"
        print("")
        print(f"[{index}] {strategy.name}  ->  {mark} ({verdict.caught_by})")
        print(f"    targets={strategy.targets} aimed_at={strategy.evades_clause}")
        print(f"    {strategy.description}")
        if strategy.pattern.clamped:
            print(f"    CLAMPED: {'; '.join(strategy.pattern.clamped)}")
        print(
            f"    flagged={verdict.flagged_accounts} "
            f"flagged_collectors={verdict.flagged_collectors} "
            f"rings={verdict.rings_formed} "
            f"stage2_alertable={verdict.stage2_alertable} "
            f"stage2_context={verdict.stage2_context}"
        )
        for position, (collector, clauses) in enumerate(verdict.clause_values.items()):
            # First 12 hex of the id: enough to tell two collectors apart
            # in a terminal, and the full 64-hex digest is in the JSON.
            print(f"      collector {position} ({collector[:12]}...)")
            for name, clause in clauses.items():
                state = "pass" if clause["passes"] else "FAIL"
                print(
                    f"        {name:<28}{str(clause['value']):>12}  {state}  "
                    f"(needs {clause['requirement']})"
                )
        print(f"    {verdict.reasoning}")

    print("")
    print(
        "A MISS is not proof the pattern survives in production: each graph is "
        "isolated, and a real account's own traffic raises the same degree "
        "counts. See the module docstring."
    )


def as_json(verdicts: list[StrategyVerdict]) -> dict[str, Any]:
    """The run as a JSON-safe dict, carrying its own caveat.

    The caveat travels with the numbers rather than living only in a doc
    nobody opens next to them: a ``missed`` row lifted out of this file
    and pasted into a slide is exactly the misreading the docstring warns
    about.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "detector": {
            "velocity_max_seconds": VELOCITY_MAX_SECONDS,
            "velocity_ratio_range": list(VELOCITY_RATIO_RANGE),
            "velocity_min_in_degree": VELOCITY_MIN_IN_DEGREE,
            "velocity_min_out_degree": VELOCITY_MIN_OUT_DEGREE,
            "min_distinct_counterparty_banks": MIN_BANKS,
        },
        "strategies_run": len(verdicts),
        "caught": sum(1 for v in verdicts if v.caught),
        "missed": sum(1 for v in verdicts if not v.caught),
        "isolated_graph_caveat": (
            "Each strategy runs as its own graph, so the only transfers in the "
            "feature table are the ones the strategy creates. A CAUGHT verdict "
            "holds at least as strongly with background traffic, because the "
            "degree clauses are lower bounds over monotone counts. A MISSED "
            "verdict does NOT transfer: a real collector's own legitimate "
            "traffic raises the same counts and shifts the speed median."
        ),
        "scorer_not_attacked": (
            "Rings come from the velocity rule, so the rule is what is "
            "attacked here. The XGBoost alert queue is not in this loop and an "
            "account this harness reports as missed may still score high."
        ),
        "verdicts": [
            {
                "name": v.strategy.name,
                "description": v.strategy.description,
                "targets": v.strategy.targets,
                "aimed_at_clause": v.strategy.evades_clause,
                "why_it_should_evade": v.strategy.why_it_should_evade,
                "pattern": asdict(v.strategy.pattern),
                "caught": v.caught,
                "caught_by": v.caught_by,
                "reasoning": v.reasoning,
                "flagged_accounts": v.flagged_accounts,
                "flagged_collectors": v.flagged_collectors,
                "rings_formed": v.rings_formed,
                "stage2_alertable": v.stage2_alertable,
                "stage2_context": v.stage2_context,
                "clause_values": v.clause_values,
            }
            for v in verdicts
        ],
    }


def main() -> None:
    """Generate, run and report. Writes the run beside the ml experiments."""
    cfg = MlConfig.from_env()
    verdicts = run(cfg=cfg)
    _print_verdicts(verdicts)

    out_dir = cfg.experiments_dir / "redteam"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(as_json(verdicts), indent=2, default=str), "utf-8")
    print("")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
