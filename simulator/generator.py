"""Seeded synthetic data generator for MuleNet's three-bank world.

Produces six parquet files plus placeholder KYC images. The whole world is
a pure function of ``(seed, params, hash_key)`` so reruns are byte-stable.

Design goal: cross-bank detection must beat single-bank detection. Most
mule rings therefore span 2-3 banks and hide part of their structure from
any single bank, while the hard-negative cohorts reproduce each individual
cross-bank signal (shared devices, reused documents, cross-bank flows) in
a legitimate setting so no single signal is sufficient on its own.

A minority of rings live entirely inside ONE bank. Those are the control
group: their pattern (shared device, reused document, fast pass-through)
is fully visible to that bank acting alone, so the experiment can compare
"ring the bank can see whole" against "ring the bank sees a third of"
instead of only measuring the cross-bank best case. The mix is set by
``SimParams.span_mix`` and recorded per account in ``ring_bank_count``.

Column names come from ``shared/contracts.py`` — schemas are never
redefined here.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from simulator.config import SimulatorConfig
from simulator.kyc import (
    accent_for,
    display_id_number,
    holder_label,
    render_address_proof,
    render_id_card,
)

BANKS: tuple[str, ...] = ("BANK_A", "BANK_B", "BANK_C")

PATTERN_TYPES: tuple[str, ...] = (
    "shared_device",
    "reused_document",
    "fast_passthrough_only",
    "mixed",
)

#: Fixed epoch for the simulated window; keeps timestamps seed-stable.
SIM_START = datetime(2025, 1, 1, tzinfo=UTC)

#: Parquet storage type for money, matching ``NUMERIC(15, 2)`` in
#: ``shared/ddl.sql`` and ``MoneyAmount`` in ``shared/contracts.py``. Writing
#: decimal128 rather than float64 keeps binary floats off the ingest path.
MONEY_TYPE = pa.decimal128(15, 2)

_CENTS = Decimal("0.01")


def to_money(value: object) -> Decimal:
    """Quantize a value to a 2dp ``Decimal``, exactly as ``_to_money`` does.

    Goes through ``str`` so a float's binary representation error is dropped
    at the decimal boundary instead of being carried into the Decimal.
    """
    return Decimal(str(value)).quantize(_CENTS, rounding=ROUND_HALF_UP)

DAY = 86_400
MIN = 60

COHORT_MULE = "mule"
COHORT_BUSINESS = "hard_negative_business"
COHORT_FAMILY = "hard_negative_family"
COHORT_DUAL_BANK = "hard_negative_dual_bank"
COHORT_NORMAL = "normal"

HARD_NEGATIVE_COHORTS = (COHORT_BUSINESS, COHORT_FAMILY, COHORT_DUAL_BANK)

#: Ordinary peer-transfer amounts. Ring edges draw from this same
#: distribution, because an amount that only mules produce is a giveaway any
#: single bank can act on.
PEER_LOG_MU = np.log(118.0)
PEER_LOG_SIGMA = 0.95

#: The band ring traffic is smeared into: a large sum is split across many
#: transfers that each sit inside ordinary peer volumes.
PEER_BAND_LO = 20.0
PEER_BAND_HI = 200.0
PEER_MIN_CHUNK = 1.0

#: Share of ring edges drawn from the *unbounded* peer distribution. Without
#: this tail, a mule's largest same-bank transfer would be conspicuously
#: small — the amount leak inverted, and still a single-bank giveaway.
RING_TAIL_PROB = 0.08

#: Share of peer-style transfers that land on an exact hundred. Real people
#: send "$200"; applying one rate to legitimate *and* ring traffic stops
#: roundness from separating them. 0% would be an inverse tell.
ROUND_SNAP_PROB = 0.005

#: Ring member counts are drawn uniformly over this inclusive range.
RING_SIZE_MIN = 3
RING_SIZE_MAX = 12

#: How many banks a ring spans, as weights over spans 1, 2, 3. The 1-bank
#: share is the control group (see the module docstring): without it the
#: dataset can only answer "how much does a join help on a ring that is
#: already split", which is the cross-bank best case and has no control.
#: Applied per pattern type so span and pattern are not confounded.
DEFAULT_SPAN_MIX: tuple[float, ...] = (0.30, 0.40, 0.30)

#: Share of ring members that are *not* freshly opened — herders buy aged
#: real accounts as often as they open new ones. Without this, account age
#: alone separates mules inside one bank.
MULE_AGED_SHARE = 0.45

#: Share of ordinary accounts opened inside the simulated window, so "new
#: account" is a crowded signal rather than a mule fingerprint.
NEW_ACCOUNT_LEGIT_RATE = 0.10

#: Share of ordinary accounts that receive money and send it on again within
#: minutes (rent paid the moment the salary clears, a sweep to a second
#: account). This is the legitimate cover for pass-through *speed*.
FAST_PASSTHROUGH_RATE = 0.38

#: Share of ordinary accounts that are barely used — the long tail every
#: retail book carries. They give a mule's thin same-bank outbound footprint
#: somewhere to hide: without them, "almost no same-bank counterparties" is
#: itself a mule signal a single bank could act on.
LIGHT_ACCOUNT_RATE = 0.22

#: Share of ordinary accounts that share a KYC identity with another account
#: at the same bank (joint accounts, a second account, parent and child).
SAME_BANK_ID_SHARE_RATE = 0.05

#: Share of ordinary accounts that share a device fingerprint with another
#: account at the same bank (one household, one phone).
SAME_BANK_DEVICE_SHARE_RATE = 0.06

_STREETS = ("MAPLE", "CEDAR", "BIRCH", "ALDER", "JUNIPER", "SORREL", "QUARTZ", "LARKIN")
_SUFFIX = ("ST", "AVE", "RD", "LN")

OUTPUT_FILES = (
    "accounts.parquet",
    "transfers.parquet",
    "devices.parquet",
    "kyc_submissions.parquet",
    "kyc_documents.parquet",
    "ground_truth.parquet",
)


@dataclass(frozen=True)
class SimParams:
    """Scale knobs. Defaults are the full competition dataset."""

    accounts_per_bank: int = 10_000
    n_days: int = 30
    rings_per_pattern: int = 30
    n_businesses: int = 50
    #: Families and dual-bank people are the legitimate cover for cross-bank
    #: device and document sharing. Sized so that reuse across banks, on its
    #: own, flags far more legitimate accounts than mules — the join has to
    #: combine it with timing and structure to be worth acting on.
    n_families: int = 200
    n_dual_bank: int = 200
    #: Number of ordinary accounts (beyond every cohort account) that get real
    #: rendered KYC images. Rendering 30k x 2 PNGs is slow and pointless.
    kyc_image_sample: int = 300
    #: Weights over ring spans (1 bank, 2 banks, 3 banks), applied to each
    #: pattern type independently. Override to change the control-group size,
    #: e.g. ``SimParams(span_mix=(0.5, 0.25, 0.25))``.
    span_mix: tuple[float, ...] = DEFAULT_SPAN_MIX
    seed: int = 20250101


@dataclass
class _Ring:
    ring_id: str
    pattern_type: str
    banks: list[str]
    collector: int
    mules: list[int] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)

    @property
    def members(self) -> list[int]:
        return [self.collector, *self.mules, *self.sources]


def hmac_hex(key: bytes, raw: str) -> str:
    """HMAC-SHA256 of ``raw`` under ``key``, as lowercase hex."""
    return hmac.new(key, raw.encode("utf-8"), hashlib.sha256).hexdigest()


class _Ledger:
    """Accumulates transfer events as numpy arrays, concatenated once."""

    def __init__(self) -> None:
        self._src: list[np.ndarray] = []
        self._dst: list[np.ndarray] = []
        self._amt: list[np.ndarray] = []
        self._ts: list[np.ndarray] = []
        self._ch: list[np.ndarray] = []

    def add(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        amount: np.ndarray,
        ts: np.ndarray,
        channel: str,
    ) -> None:
        if len(src) == 0:
            return
        self._src.append(np.asarray(src, dtype=np.int64))
        self._dst.append(np.asarray(dst, dtype=np.int64))
        self._amt.append(np.asarray(amount, dtype=np.float64))
        self._ts.append(np.asarray(ts, dtype=np.int64))
        self._ch.append(np.full(len(src), channel, dtype="<U8"))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._src:
            empty_i = np.empty(0, dtype=np.int64)
            return (empty_i, empty_i, np.empty(0), empty_i, np.empty(0, dtype="<U8"))
        return (
            np.concatenate(self._src),
            np.concatenate(self._dst),
            np.concatenate(self._amt),
            np.concatenate(self._ts),
            np.concatenate(self._ch),
        )


def _occurrence_index(counts: np.ndarray) -> np.ndarray:
    """For counts [2,0,3] -> [0,1,0,1,2]: the within-group running index."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) - np.repeat(starts, counts)


class _Pool:
    """Hands out distinct personal account indices, per bank, without replacement."""

    def __init__(self, rng: np.random.Generator, by_bank: dict[str, np.ndarray]) -> None:
        self._q = {b: list(rng.permutation(v).tolist()) for b, v in by_bank.items()}

    def take(self, bank: str, k: int = 1) -> list[int]:
        q = self._q[bank]
        if len(q) < k:
            raise ValueError(f"account pool for {bank} exhausted")
        out = [int(x) for x in q[:k]]
        del q[:k]
        return out


def _interleave_by_bank(members: list[int], bank_arr: np.ndarray) -> list[int]:
    """Round-robin the members across their banks so neighbours differ by bank."""
    buckets: dict[str, list[int]] = {}
    for m in members:
        buckets.setdefault(str(bank_arr[m]), []).append(m)
    keys = sorted(buckets)
    out: list[int] = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop(0))
    return out


def _ring_chunks(
    members: list[int], bank_arr: np.ndarray, n_groups: int, *, cross_bank: bool
) -> list[list[int]]:
    """Split a ring's members into the groups that share a device or identity.

    Every group must have at least two members, or the "shared" artefact is
    shared with nobody. For a multi-bank ring (``cross_bank``) each group must
    additionally straddle a bank boundary, so the signal really is one no
    single bank can see; the group count is reduced until that holds. For a
    single-bank ring the sharing is deliberately within the one bank — that
    is the control group, and it is meant to be visible to that bank.
    """
    ordered = _interleave_by_bank(members, bank_arr) if cross_bank else sorted(members)
    while n_groups > 1:
        size = -(-len(ordered) // n_groups)
        chunks = [c for i in range(n_groups) if (c := ordered[i * size : (i + 1) * size])]
        ok = all(len(c) >= 2 for c in chunks)
        if cross_bank:
            ok = ok and all(len({str(bank_arr[m]) for m in c}) >= 2 for c in chunks)
        if ok:
            return chunks
        n_groups -= 1
    return [ordered]


def _span_counts(n_rings: int, span_mix: tuple[float, ...]) -> list[int]:
    """Split ``n_rings`` into per-span counts by ``span_mix``, exactly.

    Entry ``i`` is the weight for rings spanning ``i + 1`` banks. Largest
    remainder, so the counts always sum back to ``n_rings`` rather than
    drifting by a ring or two per pattern.
    """
    if not span_mix or len(span_mix) > len(BANKS):
        raise ValueError(f"span_mix needs 1..{len(BANKS)} weights, got {span_mix!r}")
    if len(span_mix) > RING_SIZE_MIN:
        raise ValueError(f"span_mix cannot exceed the smallest ring ({RING_SIZE_MIN})")
    if any(w < 0 for w in span_mix) or sum(span_mix) <= 0:
        raise ValueError(f"span_mix weights must be >= 0 and sum > 0, got {span_mix!r}")
    total = float(sum(span_mix))
    exact = [n_rings * w / total for w in span_mix]
    counts = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: (counts[i] - exact[i], i))
    for i in order[: n_rings - sum(counts)]:
        counts[i] += 1
    return counts


def _synthetic_address(identity_key: str) -> str:
    h = identity_key
    number = 100 + int(h[0:4], 16) % 8900
    street = _STREETS[int(h[4:6], 16) % len(_STREETS)]
    suffix = _SUFFIX[int(h[6:8], 16) % len(_SUFFIX)]
    unit = 1 + int(h[8:10], 16) % 40
    zip_code = int(h[10:14], 16) % 10_000
    return f"{number} {street} {suffix}, UNIT {unit}, TESTVILLE ZZ-{zip_code:04d}"


# --------------------------------------------------------------------------
# amounts
# --------------------------------------------------------------------------


def _snap_some_round(rng: np.random.Generator, amount: np.ndarray) -> np.ndarray:
    """Round a small share of amounts to an exact hundred.

    Applied to legitimate and ring traffic alike at the same rate, so
    ``round_amount_ratio`` carries no information about either.
    """
    snap = rng.random(len(amount)) < ROUND_SNAP_PROB
    if not snap.any():
        return amount
    amount = amount.copy()
    amount[snap] = np.maximum(np.round(amount[snap] / 100.0), 1.0) * 100.0
    return amount


def _peer_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Ordinary peer-transfer amounts, including the usual round ones."""
    return _snap_some_round(rng, rng.lognormal(PEER_LOG_MU, PEER_LOG_SIGMA, n))


def _banded_peer_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Peer amounts restricted to the ``$20-$200`` band ring edges hide in.

    Redrawn rather than clipped, so the shape inside the band is the real
    peer shape and no mass piles up on the band edges.
    """
    out = rng.lognormal(PEER_LOG_MU, PEER_LOG_SIGMA, n)
    for _ in range(16):
        bad = (out < PEER_BAND_LO) | (out > PEER_BAND_HI)
        if not bad.any():
            break
        out[bad] = rng.lognormal(PEER_LOG_MU, PEER_LOG_SIGMA, int(bad.sum()))
    return _snap_some_round(rng, np.clip(out, PEER_BAND_LO, PEER_BAND_HI))


def _ring_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Ring edge amounts: mostly the peer band, with the same occasional
    larger transfer ordinary accounts make."""
    out = _banded_peer_amounts(rng, n)
    tail = rng.random(n) < RING_TAIL_PROB
    if tail.any():
        out[tail] = _peer_amounts(rng, int(tail.sum()))
    return out


def _split_into_peer_chunks(rng: np.random.Generator, total: float) -> list[float]:
    """Smear ``total`` across transfers that each sit in the peer band.

    This is what real rings do — many small hops precisely so no single
    amount trips a threshold — and it is why ring edges no longer sit an
    order of magnitude above legitimate ones.
    """
    chunks: list[float] = []
    remaining = float(total)
    while remaining > PEER_BAND_HI:
        cut = min(float(_ring_amounts(rng, 1)[0]), remaining)
        chunks.append(cut)
        remaining -= cut
    if remaining >= PEER_MIN_CHUNK:
        chunks.append(remaining)
    return chunks


# --------------------------------------------------------------------------
# behaviour blocks
# --------------------------------------------------------------------------


def _normal_activity(
    rng: np.random.Generator,
    ledger: _Ledger,
    actors: np.ndarray,
    start_s: np.ndarray,
    horizon_s: int,
    merchants: np.ndarray,
    employers: np.ndarray,
    billers: np.ndarray,
    peers: np.ndarray,
    *,
    salary_rate: float = 0.62,
    merchant_lambda: float = 14.0,
    peer_lambda: float = 3.0,
) -> None:
    """Salary, bills, peer transfers and merchant payments for ordinary accounts."""
    live = (horizon_s - start_s) > DAY
    actors = actors[live]
    start_s = start_s[live]
    n = len(actors)
    if n == 0:
        return

    # --- salary: regular cadence, near-identical amounts, one employer ---
    has_salary = rng.random(n) < salary_rate
    biweekly = rng.random(n) < 0.45
    n_pay = np.where(biweekly, 2, 1) * has_salary
    rep = np.repeat(np.arange(n), n_pay)
    if len(rep):
        occ = _occurrence_index(n_pay)
        first_day = np.where(biweekly, rng.integers(1, 15, n), rng.integers(1, 29, n))
        base = np.round(rng.lognormal(np.log(3400.0), 0.45, n), 2)
        employer = employers[rng.integers(0, len(employers), n)]
        ts = (first_day[rep] + occ * 14) * DAY + rng.integers(0, DAY, len(rep))
        amt = base[rep] * (1.0 + rng.normal(0.0, 0.008, len(rep)))
        keep = (ts >= start_s[rep]) & (ts < horizon_s)
        ledger.add(employer[rep][keep], actors[rep][keep], amt[keep], ts[keep], "ach")

    # --- bills: monthly, recurring payees, stable amounts ---
    n_bills = rng.integers(1, 4, n)
    rep = np.repeat(np.arange(n), n_bills)
    if len(rep):
        total = len(rep)
        payee = billers[rng.integers(0, len(billers), total)]
        ts = rng.integers(1, 29, total) * DAY + rng.integers(0, DAY, total)
        amt = rng.lognormal(np.log(74.0), 0.7, total) * (1.0 + rng.normal(0.0, 0.02, total))
        keep = (ts >= start_s[rep]) & (ts < horizon_s)
        ledger.add(actors[rep][keep], payee[keep], amt[keep], ts[keep], "ach")

    # --- peer transfers: sporadic, lognormal ---
    n_peer = rng.poisson(peer_lambda, n)
    rep = np.repeat(np.arange(n), n_peer)
    if len(rep):
        total = len(rep)
        counter = peers[rng.integers(0, len(peers), total)]
        span = horizon_s - start_s[rep]
        ts = start_s[rep] + (rng.random(total) * span).astype(np.int64)
        amt = _peer_amounts(rng, total)
        ledger.add(actors[rep], counter, amt, ts, "p2p")

    # --- merchant payments: frequent, small, varied ---
    n_merch = rng.poisson(merchant_lambda, n)
    rep = np.repeat(np.arange(n), n_merch)
    if len(rep):
        total = len(rep)
        merchant = merchants[rng.integers(0, len(merchants), total)]
        span = horizon_s - start_s[rep]
        ts = start_s[rep] + (rng.random(total) * span).astype(np.int64)
        amt = rng.lognormal(np.log(27.0), 0.85, total)
        ledger.add(actors[rep], merchant, amt, ts, "card")


def _fast_legit_passthrough(
    rng: np.random.Generator,
    ledger: _Ledger,
    actors: np.ndarray,
    bank_arr: np.ndarray,
    by_bank: dict[str, np.ndarray],
    horizon_s: int,
    rate: float,
) -> None:
    """Legitimate money that arrives and leaves again within 2-30 minutes.

    Rent paid the moment the salary clears, a sweep into a second account,
    a supplier settled out of the morning's takings. Both legs are booked
    at the *same* bank on purpose: a single bank must be able to see plenty
    of fast in-then-out bursts that are not mules, otherwise pass-through
    speed alone separates rings without any cross-bank join.
    """
    if len(actors) == 0 or rate <= 0.0:
        return
    chosen = actors[rng.random(len(actors)) < rate]
    if len(chosen) == 0:
        return
    n_events = rng.integers(1, 4, len(chosen))
    rep = np.repeat(np.arange(len(chosen)), n_events)
    total = len(rep)
    if total == 0:
        return

    banks = bank_arr[chosen[rep]]
    counterparts = np.empty((2, total), dtype=np.int64)
    for bank, pool_idx in by_bank.items():
        mask = banks == bank
        k = int(mask.sum())
        if k == 0 or len(pool_idx) == 0:
            continue
        counterparts[0, mask] = pool_idx[rng.integers(0, len(pool_idx), k)]
        counterparts[1, mask] = pool_idx[rng.integers(0, len(pool_idx), k)]

    in_ts = rng.integers(0, max(horizon_s - 2 * 3600, 1), total)
    # a mix of salary-sized arrivals and ordinary peer credits
    salary_like = rng.random(total) < 0.45
    in_amt = np.where(
        salary_like, rng.lognormal(np.log(2600.0), 0.5, total), _peer_amounts(rng, total)
    )
    ledger.add(counterparts[0], chosen[rep], in_amt, in_ts, "ach")

    out_ts = in_ts + rng.integers(2 * MIN, 30 * MIN, total)
    out_amt = in_amt * rng.uniform(0.55, 0.98, total)
    keep = out_ts < horizon_s
    ledger.add(
        chosen[rep][keep], counterparts[1][keep], _snap_some_round(rng, out_amt)[keep],
        out_ts[keep], "p2p",
    )


def _business_activity(
    rng: np.random.Generator,
    ledger: _Ledger,
    businesses: list[int],
    horizon_s: int,
    customers: np.ndarray,
    suppliers: np.ndarray,
) -> None:
    """Hard negative #1: busy small businesses — high fan-in, many counterparties.

    Most of the outbound is a slow weekly supplier/payroll schedule, but a
    minority is settled same-day, minutes after the takings land — real
    businesses do that, and it is what stops "fast in-then-out" from being
    a mule-only shape inside one bank.
    """
    for biz in businesses:
        n_cust = int(rng.integers(40, 91))
        payers = customers[rng.choice(len(customers), size=n_cust, replace=False)]
        payers = payers[payers != biz]
        n_in = len(payers) + int(rng.integers(10, 40))
        src = payers[rng.integers(0, len(payers), n_in)]
        ts = rng.integers(0, horizon_s, n_in)
        amt = rng.lognormal(np.log(88.0), 0.85, n_in)
        ledger.add(src, np.full(n_in, biz), amt, ts, "card")

        n_out = int(rng.integers(8, 13))
        payee = suppliers[rng.integers(0, len(suppliers), n_out)]
        out_ts = (np.arange(n_out) * (horizon_s // max(n_out, 1))) + int(
            rng.integers(6 * 3600, 20 * 3600)
        )
        out_ts = np.clip(out_ts, 0, horizon_s - 1)
        out_amt = rng.lognormal(np.log(1450.0), 0.6, n_out)
        ledger.add(np.full(n_out, biz), payee, out_amt, out_ts, "ach")

        # same-day settlement: leaves 2-30 minutes after a receipt
        n_fast = int(rng.integers(3, 9))
        pick = rng.integers(0, n_in, n_fast)
        fast_ts = np.minimum(ts[pick] + rng.integers(2 * MIN, 30 * MIN, n_fast), horizon_s - 1)
        fast_payee = suppliers[rng.integers(0, len(suppliers), n_fast)]
        ledger.add(
            np.full(n_fast, biz), fast_payee, amt[pick] * rng.uniform(0.6, 0.95, n_fast),
            fast_ts, "instant",
        )


def _family_activity(
    rng: np.random.Generator,
    ledger: _Ledger,
    families: list[list[int]],
    horizon_s: int,
) -> None:
    """Hard negative #2: families sharing a device, transferring among themselves."""
    for members in families:
        n_tx = int(rng.integers(3, 10))
        a = rng.integers(0, len(members), n_tx)
        b = (a + 1 + rng.integers(0, len(members) - 1, n_tx)) % len(members)
        src = np.array([members[i] for i in a], dtype=np.int64)
        dst = np.array([members[i] for i in b], dtype=np.int64)
        ts = rng.integers(0, horizon_s, n_tx)
        amt = _snap_some_round(rng, rng.lognormal(np.log(155.0), 0.8, n_tx))
        ledger.add(src, dst, amt, ts, "p2p")


def _dual_bank_activity(
    rng: np.random.Generator,
    ledger: _Ledger,
    pairs: list[tuple[int, int]],
    horizon_s: int,
) -> None:
    """Hard negative #3: one person, two banks, moving their own money.

    Some of those moves happen minutes after their salary lands, which looks
    exactly like a pass-through hop to whichever bank sees only one leg.
    """
    for left, right in pairs:
        n_tx = int(rng.integers(4, 9))
        forward = rng.random(n_tx) < 0.6
        src = np.where(forward, left, right)
        dst = np.where(forward, right, left)
        ts = np.sort(rng.integers(0, horizon_s, n_tx))
        amt = _snap_some_round(rng, rng.lognormal(np.log(880.0), 0.55, n_tx))
        ledger.add(src, dst, amt, ts, "instant")


def _ring_activity(
    rng: np.random.Generator,
    ledger: _Ledger,
    rings: list[_Ring],
    opened_s: np.ndarray,
    horizon_s: int,
    externals: np.ndarray,
) -> None:
    """Mule ring behaviour: fan-in, then pass-through in 2-30 minutes.

    Amounts deliberately carry no signal of their own. Every ring edge is
    drawn from the same peer-transfer band as legitimate traffic and each
    forwarded sum is smeared across many small hops, so neither magnitude
    nor roundness distinguishes a ring edge. What remains is the structure:
    fan-in, a fast hop, and — crucially — the fact that the sender, the
    hop and the cash-out sit at different banks.
    """
    for ring in rings:
        members = ring.members
        active_from = max(0, int(max(opened_s[m] for m in members)) + DAY)
        latest_start = horizon_s - (10 * 3600)
        if active_from >= latest_start:
            active_from = max(0, latest_start - DAY)

        n_waves = int(rng.integers(4, 9))
        for _ in range(n_waves):
            wave_t0 = int(rng.integers(active_from, latest_start))
            last_out = wave_t0
            wave_total = 0.0
            for mule in ring.mules:
                k_in = int(rng.integers(3, 8))
                n_src = min(len(ring.sources), int(rng.integers(1, 3)))
                senders = [
                    int(x) for x in rng.choice(ring.sources, size=n_src, replace=False)
                ]
                senders += [
                    int(x)
                    for x in externals[rng.integers(0, len(externals), k_in - n_src)]
                ]
                in_ts = wave_t0 + rng.integers(0, 3 * 3600, len(senders))
                in_amt = _ring_amounts(rng, len(senders))
                ledger.add(
                    np.array(senders, dtype=np.int64),
                    np.full(len(senders), mule, dtype=np.int64),
                    in_amt,
                    in_ts,
                    "p2p",
                )

                # pass-through: the mule's 1-5% cut stays, the rest leaves
                # 2-30 minutes later, split so no hop stands out by size
                forwarded = float(in_amt.sum()) * (1.0 - float(rng.uniform(0.01, 0.05)))
                chunks = _split_into_peer_chunks(rng, forwarded)
                if not chunks:
                    continue
                out_ts = int(in_ts.max()) + rng.integers(2 * MIN, 30 * MIN, len(chunks))
                ledger.add(
                    np.full(len(chunks), mule, dtype=np.int64),
                    np.full(len(chunks), ring.collector, dtype=np.int64),
                    np.asarray(chunks, dtype=np.float64),
                    out_ts,
                    "instant",
                )
                last_out = max(last_out, int(out_ts.max()))
                wave_total += float(sum(chunks))

            # collector cashes out well after the hops, to off-ring accounts
            cash_chunks = _split_into_peer_chunks(
                rng, wave_total * (1.0 - float(rng.uniform(0.01, 0.05)))
            )
            if cash_chunks:
                cash_ts = np.minimum(
                    last_out + rng.integers(3600, 6 * 3600, len(cash_chunks)), horizon_s - 1
                )
                ledger.add(
                    np.full(len(cash_chunks), ring.collector, dtype=np.int64),
                    externals[rng.integers(0, len(externals), len(cash_chunks))],
                    np.asarray(cash_chunks, dtype=np.float64),
                    cash_ts,
                    "instant",
                )


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame, pinning any money column to ``decimal128(15, 2)``."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    index = table.schema.get_field_index("amount")
    if index >= 0:
        table = table.set_column(
            index, pa.field("amount", MONEY_TYPE), table.column(index).cast(MONEY_TYPE)
        )
    pq.write_table(table, path)


def generate(
    params: SimParams | None = None,
    *,
    hash_key: bytes = b"demo-key",
    out_dir: Path | None = None,
    render_images: bool = True,
) -> dict[str, Path]:
    """Build the whole synthetic world and write parquet + images.

    Returns a mapping of file name -> path for the six parquet outputs.
    """
    params = params or SimParams()
    out_dir = Path(out_dir) if out_dir is not None else SimulatorConfig.from_env().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "kyc_images"

    seeds = np.random.SeedSequence(params.seed).spawn(7)
    rng_struct = np.random.default_rng(seeds[0])
    rng_ring = np.random.default_rng(seeds[1])
    rng_normal = np.random.default_rng(seeds[2])
    rng_hard = np.random.default_rng(seeds[3])
    rng_dev = np.random.default_rng(seeds[4])
    rng_kyc = np.random.default_rng(seeds[5])
    rng_misc = np.random.default_rng(seeds[6])

    per_bank = params.accounts_per_bank
    n_banks = len(BANKS)
    n_accounts = per_bank * n_banks
    horizon_s = params.n_days * DAY

    bank_arr = np.repeat(np.array(BANKS, dtype=object), per_bank)
    local_seq = np.tile(np.arange(per_bank), n_banks)
    acct_hash = np.array(
        [hmac_hex(hash_key, f"acct:{b}:{s:07d}") for b, s in zip(bank_arr, local_seq, strict=True)],
        dtype=object,
    )

    # --- account types: a small infrastructure layer plus personal accounts ---
    n_merch = max(6, int(per_bank * 0.02))
    n_emp = max(3, int(per_bank * 0.005))
    n_bill = max(3, int(per_bank * 0.002))
    acct_type = np.full(n_accounts, "personal", dtype=object)
    for b in range(n_banks):
        base = b * per_bank
        acct_type[base : base + n_merch] = "merchant"
        acct_type[base + n_merch : base + n_merch + n_emp] = "employer"
        acct_type[base + n_merch + n_emp : base + n_merch + n_emp + n_bill] = "biller"

    merchants = np.flatnonzero(acct_type == "merchant")
    employers = np.flatnonzero(acct_type == "employer")
    billers = np.flatnonzero(acct_type == "biller")
    personal = np.flatnonzero(acct_type == "personal")

    # opened_at, in seconds relative to SIM_START (negative = before the window)
    opened_s = -rng_struct.integers(30 * DAY, 8 * 365 * DAY, n_accounts).astype(np.int64)
    # a bank is always onboarding: a tenth of the book is brand new, so a
    # young account is a crowded signal rather than a mule fingerprint
    fresh = rng_struct.random(n_accounts) < NEW_ACCOUNT_LEGIT_RATE
    opened_s[fresh] = rng_struct.integers(-21 * DAY, 20 * DAY, int(fresh.sum()))

    pool = _Pool(rng_struct, {b: personal[bank_arr[personal] == b] for b in BANKS})

    # --- mule rings -------------------------------------------------------
    rings: list[_Ring] = []
    ring_no = 0
    span_counts = _span_counts(params.rings_per_pattern, params.span_mix)
    for pattern in PATTERN_TYPES:
        # every pattern gets the same span mix, so span and pattern type are
        # not confounded and the ml side can stratify on either independently
        spans = [s + 1 for s, count in enumerate(span_counts) for _ in range(count)]
        spans = [spans[i] for i in rng_ring.permutation(len(spans)).tolist()]
        for span in spans:
            size = int(rng_ring.integers(RING_SIZE_MIN, RING_SIZE_MAX + 1))
            chosen = [
                BANKS[i]
                for i in rng_ring.choice(n_banks, size=span, replace=False).tolist()
            ]
            # one bank slot per member, covering every chosen bank, then
            # shuffled so a member's role does not predict its bank
            slots = [chosen[i % span] for i in range(size)]
            slots = [slots[i] for i in rng_ring.permutation(size).tolist()]
            n_sources = int(rng_ring.integers(1, max(2, size // 3 + 1)))
            n_mules = size - 1 - n_sources
            ring = _Ring(
                ring_id=f"ring_{ring_no:03d}",
                pattern_type=pattern,
                banks=chosen,
                collector=pool.take(slots[0])[0],
            )
            for slot in slots[1 : 1 + n_mules]:
                ring.mules.append(pool.take(slot)[0])
            for slot in slots[1 + n_mules :]:
                ring.sources.append(pool.take(slot)[0])
            rings.append(ring)
            ring_no += 1

    ring_member_idx: list[int] = []
    for ring in rings:
        ring_member_idx.extend(ring.members)
        # most ring accounts are newly opened, but a herder buys aged real
        # accounts too; leaving every mule brand new would let one bank
        # separate rings on account age alone
        for m in ring.members:
            if rng_ring.random() < MULE_AGED_SHARE:
                opened_s[m] = -int(rng_ring.integers(60 * DAY, 8 * 365 * DAY))
            else:
                opened_s[m] = int(rng_ring.integers(-7 * DAY, 10 * DAY))
    ring_members = np.array(sorted(ring_member_idx), dtype=np.int64)

    # --- hard negatives ---------------------------------------------------
    businesses: list[int] = []
    for i in range(params.n_businesses):
        idx = pool.take(BANKS[i % n_banks])[0]
        acct_type[idx] = "business"
        opened_s[idx] = -int(rng_hard.integers(365 * DAY, 10 * 365 * DAY))
        businesses.append(idx)

    families: list[list[int]] = []
    for i in range(params.n_families):
        size = int(rng_hard.integers(3, 6))
        # half the families straddle two banks, so "shared device across banks"
        # on its own is not evidence of a ring
        cross = i % 2 == 0
        home = BANKS[i % n_banks]
        other = BANKS[(i + 1) % n_banks]
        members = pool.take(home, size - 1)
        members.append(pool.take(other if cross else home)[0])
        families.append(members)

    dual_bank: list[tuple[int, int]] = []
    for i in range(params.n_dual_bank):
        left = pool.take(BANKS[i % n_banks])[0]
        right = pool.take(BANKS[(i + 1) % n_banks])[0]
        dual_bank.append((left, right))

    # --- cohort labels ----------------------------------------------------
    cohort = np.full(n_accounts, COHORT_NORMAL, dtype=object)
    role = np.full(n_accounts, "", dtype=object)
    ring_id_col = np.full(n_accounts, "", dtype=object)
    pattern_col = np.full(n_accounts, "", dtype=object)
    ring_banks_col = np.zeros(n_accounts, dtype=np.int64)
    for ring in rings:
        for m in ring.members:
            cohort[m] = COHORT_MULE
            ring_id_col[m] = ring.ring_id
            pattern_col[m] = ring.pattern_type
            ring_banks_col[m] = len(ring.banks)
        role[ring.collector] = "collector"
        for m in ring.mules:
            role[m] = "mule"
        for s in ring.sources:
            role[s] = "source"
    for idx in businesses:
        cohort[idx] = COHORT_BUSINESS
    for members in families:
        for idx in members:
            cohort[idx] = COHORT_FAMILY
    for left, right in dual_bank:
        cohort[left] = COHORT_DUAL_BANK
        cohort[right] = COHORT_DUAL_BANK

    # --- transfers --------------------------------------------------------
    ledger = _Ledger()
    start_s = np.maximum(opened_s, 0)

    ordinary = np.setdiff1d(personal, ring_members, assume_unique=False)
    peers = ordinary
    light = ordinary[rng_normal.random(len(ordinary)) < LIGHT_ACCOUNT_RATE]
    engaged = np.setdiff1d(ordinary, light, assume_unique=False)
    _normal_activity(
        rng_normal,
        ledger,
        engaged,
        start_s[engaged],
        horizon_s,
        merchants,
        employers,
        billers,
        peers,
    )
    _normal_activity(
        rng_normal,
        ledger,
        light,
        start_s[light],
        horizon_s,
        merchants,
        employers,
        billers,
        peers,
        salary_rate=0.22,
        merchant_lambda=1.4,
        peer_lambda=0.3,
    )
    # ring accounts still have a thin retail footprint so they are not pure signal
    _normal_activity(
        rng_normal,
        ledger,
        ring_members,
        start_s[ring_members],
        horizon_s,
        merchants,
        employers,
        billers,
        peers,
        # a mule account is usually a real person's account: it keeps a
        # believable retail footprint, so its ordinary traffic does not look
        # conspicuously absent to its own bank
        salary_rate=0.35,
        merchant_lambda=5.0,
        peer_lambda=1.2,
    )
    biz_idx = np.array(businesses, dtype=np.int64)
    _normal_activity(
        rng_normal,
        ledger,
        biz_idx,
        start_s[biz_idx],
        horizon_s,
        merchants,
        employers,
        billers,
        peers,
        salary_rate=0.0,
        merchant_lambda=6.0,
        peer_lambda=1.0,
    )
    ordinary_by_bank = {b: ordinary[bank_arr[ordinary] == b] for b in BANKS}
    # Same-bank device/identity cover is drawn only from plain accounts: the
    # hard negatives already carry deliberate *cross*-bank links and must not
    # have them overwritten.
    plain = ordinary[cohort[ordinary] == COHORT_NORMAL]
    plain_by_bank = {b: plain[bank_arr[plain] == b] for b in BANKS}
    _fast_legit_passthrough(
        rng_normal,
        ledger,
        ordinary,
        bank_arr,
        ordinary_by_bank,
        horizon_s,
        FAST_PASSTHROUGH_RATE,
    )
    _business_activity(rng_hard, ledger, businesses, horizon_s, ordinary, merchants)
    _family_activity(rng_hard, ledger, families, horizon_s)
    _dual_bank_activity(rng_hard, ledger, dual_bank, horizon_s)
    _ring_activity(rng_ring, ledger, rings, opened_s, horizon_s, ordinary)

    src, dst, amt, ts, chan = ledger.arrays()
    order = np.lexsort((dst, src, ts))
    src, dst, amt, ts, chan = src[order], dst[order], amt[order], ts[order], chan[order]
    transfer_ids = [hmac_hex(hash_key, f"txn:{i}") for i in range(len(src))]

    transfers = pd.DataFrame(
        {
            "transfer_id": transfer_ids,
            "from_hashed_account_id": acct_hash[src],
            "to_hashed_account_id": acct_hash[dst],
            "amount": [to_money(v) for v in amt.tolist()],
            "currency": "USD",
            "timestamp": pd.to_datetime(SIM_START) + pd.to_timedelta(ts, unit="s"),
            "channel": chan,
        }
    )

    # --- devices ----------------------------------------------------------
    device_rows: list[tuple[int, str]] = []
    fp_of = np.array(
        [hmac_hex(hash_key, f"device:{i}") for i in range(n_accounts)], dtype=object
    )
    for i in range(n_accounts):
        device_rows.append((i, fp_of[i]))
    second = np.flatnonzero(rng_dev.random(n_accounts) < 0.12)
    for i in second.tolist():
        device_rows.append((i, hmac_hex(hash_key, f"device:secondary:{i}")))

    shared_device_rings = {"shared_device", "mixed"}
    for ring in rings:
        if ring.pattern_type not in shared_device_rings:
            continue
        members = ring.members
        cross = len(ring.banks) > 1
        if ring.pattern_type == "mixed":
            # only part of a mixed ring shares hardware
            members = _ring_chunks(members, bank_arr, 2, cross_bank=cross)[0]
        n_shared = 1 if len(members) <= 6 else 2
        for k, chunk in enumerate(
            _ring_chunks(members, bank_arr, n_shared, cross_bank=cross)
        ):
            shared = hmac_hex(hash_key, f"device:ring:{ring.ring_id}:{k}")
            for m in chunk:
                device_rows.append((m, shared))
    for i, members in enumerate(families):
        shared = hmac_hex(hash_key, f"device:family:{i}")
        for m in members:
            device_rows.append((m, shared))
    for i, (left, right) in enumerate(dual_bank):
        shared = hmac_hex(hash_key, f"device:dual:{i}")
        device_rows.append((left, shared))
        device_rows.append((right, shared))

    # ordinary households share one phone or laptop inside a single bank, so
    # "this account shares a device" is not by itself something one bank can
    # act on
    for bank, pool_idx in plain_by_bank.items():
        n_pairs = int(len(pool_idx) * SAME_BANK_DEVICE_SHARE_RATE / 2)
        if n_pairs < 1:
            continue
        picks = rng_dev.choice(pool_idx, size=n_pairs * 2, replace=False)
        for k in range(n_pairs):
            shared = hmac_hex(hash_key, f"device:household:{bank}:{k}")
            device_rows.append((int(picks[2 * k]), shared))
            device_rows.append((int(picks[2 * k + 1]), shared))

    unseen = np.iinfo(np.int64).max
    first_seen_s = np.full(n_accounts, unseen, dtype=np.int64)
    last_seen_s = np.full(n_accounts, -1, dtype=np.int64)
    for actor in (src, dst):
        np.minimum.at(first_seen_s, actor, ts)
        np.maximum.at(last_seen_s, actor, ts)
    fallback = np.maximum(start_s, 0)
    first_seen_s = np.where(first_seen_s == unseen, fallback, first_seen_s)
    last_seen_s = np.where(last_seen_s < 0, np.minimum(fallback + DAY, horizon_s), last_seen_s)

    dev_acct = np.array([a for a, _ in device_rows], dtype=np.int64)
    dev_fp = np.array([f for _, f in device_rows], dtype=object)
    devices = pd.DataFrame(
        {
            "device_id": [
                hmac_hex(hash_key, f"devrow:{a}:{f}") for a, f in zip(dev_acct, dev_fp, strict=True)
            ],
            "hashed_account_id": acct_hash[dev_acct],
            "device_fingerprint": dev_fp,
            "first_seen": pd.to_datetime(SIM_START)
            + pd.to_timedelta(first_seen_s[dev_acct], unit="s"),
            "last_seen": pd.to_datetime(SIM_START)
            + pd.to_timedelta(last_seen_s[dev_acct], unit="s"),
        }
    )

    # --- KYC identities ---------------------------------------------------
    identity = np.array(
        [hmac_hex(hash_key, f"identity:{i}") for i in range(n_accounts)], dtype=object
    )
    reuse_rings = {"reused_document", "mixed"}
    for ring in rings:
        if ring.pattern_type not in reuse_rings:
            continue
        members = ring.members
        cross = len(ring.banks) > 1
        if ring.pattern_type == "mixed":
            members = _ring_chunks(members, bank_arr, 2, cross_bank=cross)[-1]
        # 1-2 stolen identities recycled across the ring's banks (or, for a
        # single-bank control ring, inside that one bank)
        n_ident = 1 if len(members) <= 6 else 2
        for k, chunk in enumerate(
            _ring_chunks(members, bank_arr, n_ident, cross_bank=cross)
        ):
            ident = hmac_hex(hash_key, f"identity:ring:{ring.ring_id}:{k}")
            for m in chunk:
                identity[m] = ident
    # a genuine person really does present the same identity at two banks
    for i, (left, right) in enumerate(dual_bank):
        shared = hmac_hex(hash_key, f"identity:dual:{i}")
        identity[left] = shared
        identity[right] = shared

    # --- legitimate *same-bank* document reuse -----------------------------
    # Joint accounts, a second account at the same bank, a parent opening an
    # account for a child: one identity, two accounts, one bank. Without this
    # cover, within-bank document reuse would be a mule-only signal that a
    # single bank could act on, which is exactly what must not be true.
    for i, members in enumerate(families):
        same_bank = [m for m in members if bank_arr[m] == bank_arr[members[0]]]
        if len(same_bank) >= 2 and rng_kyc.random() < 0.15:
            joint = hmac_hex(hash_key, f"identity:joint:{i}")
            identity[same_bank[0]] = joint
            identity[same_bank[1]] = joint
    for biz in businesses:
        candidates = plain_by_bank[str(bank_arr[biz])]
        if len(candidates) and rng_kyc.random() < 0.25:
            owner = int(candidates[rng_kyc.integers(0, len(candidates))])
            identity[owner] = identity[biz]
    for pool_idx in plain_by_bank.values():
        n_pairs = int(len(pool_idx) * SAME_BANK_ID_SHARE_RATE / 2)
        if n_pairs < 1:
            continue
        picks = rng_kyc.choice(pool_idx, size=n_pairs * 2, replace=False)
        for k in range(n_pairs):
            identity[int(picks[2 * k + 1])] = identity[int(picks[2 * k])]

    render_set: set[str] = set()
    for ring in rings:
        render_set.update(identity[m] for m in ring.members)
    for idx in businesses:
        render_set.add(identity[idx])
    for members in families:
        render_set.update(identity[m] for m in members)
    for left, _right in dual_bank:
        render_set.add(identity[left])
    if params.kyc_image_sample > 0:
        sample = rng_kyc.choice(
            ordinary, size=min(params.kyc_image_sample, len(ordinary)), replace=False
        )
        render_set.update(identity[i] for i in sample.tolist())

    hashed_id = np.array(
        [hmac_hex(hash_key, f"idnum:{k}") for k in identity], dtype=object
    )
    addresses = np.array([_synthetic_address(k) for k in identity], dtype=object)

    doc_types = ("id_card", "address_proof")
    image_ref = {}
    for key in sorted(render_set):
        for doc_type in doc_types:
            image_ref[(key, doc_type)] = f"kyc_images/{key[:16]}_{doc_type}.png"

    if render_images:
        rendered: set[str] = set()
        for key in sorted(render_set):
            if key in rendered:
                continue
            rendered.add(key)
            hid = hmac_hex(hash_key, f"idnum:{key}")
            render_id_card(
                img_dir / f"{key[:16]}_id_card.png",
                holder_name=holder_label(key),
                id_number_display=display_id_number(hid),
                address=_synthetic_address(key),
                issuer="SPECIMEN REGISTRY",
                accent=accent_for(key),
            )
            render_address_proof(
                img_dir / f"{key[:16]}_address_proof.png",
                holder_name=holder_label(key),
                address=_synthetic_address(key),
                issuer="SPECIMEN REGISTRY",
                statement_period="2024-12 (SAMPLE)",
                accent=accent_for(key),
            )

    submitted_s = opened_s
    doc_rows: dict[str, list] = {
        "doc_id": [],
        "hashed_account_id": [],
        "doc_type": [],
        "hashed_id_number": [],
        "normalized_address": [],
        "photo_embedding_ref": [],
    }
    sub_doc_ids: list[list[str]] = []
    for i in range(n_accounts):
        ids: list[str] = []
        for doc_type in doc_types:
            doc_id = hmac_hex(hash_key, f"doc:{i}:{doc_type}")
            ids.append(doc_id)
            doc_rows["doc_id"].append(doc_id)
            doc_rows["hashed_account_id"].append(acct_hash[i])
            doc_rows["doc_type"].append(doc_type)
            doc_rows["hashed_id_number"].append(hashed_id[i])
            doc_rows["normalized_address"].append(addresses[i])
            doc_rows["photo_embedding_ref"].append(image_ref.get((identity[i], doc_type)))
        sub_doc_ids.append(ids)

    kyc_documents = pd.DataFrame(doc_rows)
    kyc_submissions = pd.DataFrame(
        {
            "submission_id": [hmac_hex(hash_key, f"sub:{i}") for i in range(n_accounts)],
            "hashed_account_id": acct_hash,
            "bank_id": bank_arr,
            "submitted_at": pd.to_datetime(SIM_START)
            + pd.to_timedelta(submitted_s, unit="s"),
            "doc_ids": sub_doc_ids,
        }
    )

    # --- accounts + ground truth -----------------------------------------
    status = np.full(n_accounts, "active", dtype=object)
    dormant = rng_misc.random(n_accounts) < 0.02
    dormant[ring_members] = False
    status[dormant] = "dormant"

    accounts = pd.DataFrame(
        {
            "hashed_account_id": acct_hash,
            "bank_id": bank_arr,
            "opened_at": pd.to_datetime(SIM_START) + pd.to_timedelta(opened_s, unit="s"),
            "account_type": acct_type,
            "status": status,
        }
    )

    is_hard_negative = np.isin(cohort.astype(str), HARD_NEGATIVE_COHORTS)
    ground_truth = pd.DataFrame(
        {
            "hashed_account_id": acct_hash,
            "bank_id": bank_arr,
            "is_mule": cohort == COHORT_MULE,
            "ring_id": ring_id_col,
            "role": role,
            "pattern_type": pattern_col,
            "cohort": cohort,
            "is_hard_negative": is_hard_negative,
            "ring_bank_count": ring_banks_col,
            "opened_at": accounts["opened_at"],
        }
    )

    frames = {
        "accounts.parquet": accounts,
        "transfers.parquet": transfers,
        "devices.parquet": devices,
        "kyc_submissions.parquet": kyc_submissions,
        "kyc_documents.parquet": kyc_documents,
        "ground_truth.parquet": ground_truth,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out_dir / name
        _write_parquet(frame, path)
        paths[name] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m simulator.generator``."""
    cfg = SimulatorConfig.from_env()
    parser = argparse.ArgumentParser(description="Generate the MuleNet synthetic dataset.")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--accounts-per-bank", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--kyc-image-sample", type=int, default=300)
    parser.add_argument(
        "--span-mix",
        default=",".join(str(w) for w in DEFAULT_SPAN_MIX),
        help="comma-separated weights for rings spanning 1,2,3 banks",
    )
    parser.add_argument("--out", type=Path, default=cfg.out_dir)
    parser.add_argument("--no-images", action="store_true", help="skip PNG rendering")
    args = parser.parse_args(argv)

    params = SimParams(
        accounts_per_bank=args.accounts_per_bank,
        n_days=args.days,
        kyc_image_sample=args.kyc_image_sample,
        span_mix=tuple(float(w) for w in args.span_mix.split(",")),
        seed=args.seed,
    )
    paths = generate(
        params,
        hash_key=cfg.hash_key,
        out_dir=args.out,
        render_images=not args.no_images,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
