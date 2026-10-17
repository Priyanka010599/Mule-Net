"""Slow-drip overlay: a ring pattern built to sit under the velocity rule's ceiling.

What this is
------------
A *separate* synthetic world, written to ``ml/experiments/slow_drip/out/``
(gitignored), in which every mule ring forwards its money slowly and in
small amounts. Nothing under ``simulator/out`` is read or written.

How it reuses the canonical simulator
-------------------------------------
``simulator.generator.generate`` is called unchanged, so cover traffic
(salary, bills, peers, merchants, fast legitimate pass-through, the three
hard-negative cohorts), ring membership, bank spread, device sharing,
KYC reuse and account ages follow exactly the same rules as the canonical
world. The only thing swapped is the ring *behaviour* block
(``_ring_activity``), by rebinding that module attribute for the duration
of the call. ``simulator/`` is not edited.

What differs from canonical ring behaviour (and only this)
----------------------------------------------------------
* every transfer amount is capped at the **median** peer amount
  (``exp(PEER_LOG_MU)``), instead of the full $20-$200 band plus a tail;
* the mule's hop to the collector starts ``DRIP_DELAY_RANGE`` after the
  last inbound credit, and successive hop chunks are ``DRIP_SPACING_RANGE``
  apart. Both ranges are **multiples of the velocity rule's ceiling**
  (``ml.rule.VELOCITY_MAX_SECONDS``), so every hop is slower than the rule's
  limit by construction and the figures follow if the rule's ceiling moves;
* fan-in (3-7 credits per mule per wave, 4-8 waves) is untouched, so the
  rule's degree and bank-spread clauses are *not* what lets the pattern
  through: the one variable changed is hop speed (plus small amounts).

Ring spans are 2 or 3 banks only (``OVERLAY_SPAN_MIX``). The four
``pattern_type`` values (identity-sharing artefacts) are kept as in the
canonical world; ``fast_passthrough_only`` rings carry no identity reuse
and are the purest test of behaviour-only detection.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.rule import VELOCITY_MAX_SECONDS
from simulator import generator as sim
from simulator.generator import SimParams

#: Where the overlay world lands. Gitignored (``out/``).
OVERLAY_DIR = Path(__file__).resolve().parent / "out"

#: Deterministic seed for the overlay world (distinct from the canonical one).
OVERLAY_SEED = 20250202

#: Weights over spans (1, 2, 3 banks): none single-bank, half 2 banks, half 3.
OVERLAY_SPAN_MIX: tuple[float, ...] = (0.0, 0.5, 0.5)

#: Delay from a mule's last inbound credit to its first forward, as
#: multiples of the velocity rule's ceiling.
DRIP_DELAY_MULTIPLE: tuple[float, float] = (2.0, 6.0)

#: Gap between successive forwarded chunks, as multiples of the ceiling.
DRIP_SPACING_MULTIPLE: tuple[float, float] = (1.5, 3.0)

DRIP_DELAY_RANGE: tuple[int, int] = (
    int(DRIP_DELAY_MULTIPLE[0] * VELOCITY_MAX_SECONDS),
    int(DRIP_DELAY_MULTIPLE[1] * VELOCITY_MAX_SECONDS),
)
DRIP_SPACING_RANGE: tuple[int, int] = (
    int(DRIP_SPACING_MULTIPLE[0] * VELOCITY_MAX_SECONDS),
    int(DRIP_SPACING_MULTIPLE[1] * VELOCITY_MAX_SECONDS),
)

#: Every drip transfer is at most this: the median ordinary peer amount.
SMALL_AMOUNT_CAP = float(np.exp(sim.PEER_LOG_MU))

#: Upper bound on forwarded chunks per mule per wave used to size a wave.
_MAX_CHUNKS = 12

#: Longest a wave can last: 3h of credits, the longest first delay,
#: ``_MAX_CHUNKS`` widest spacings, and the collector's cash-out delay.
WAVE_SPAN_SECONDS = 3 * 3600 + DRIP_DELAY_RANGE[1] + _MAX_CHUNKS * DRIP_SPACING_RANGE[1] + 6 * 3600

#: Waves must start early enough for the slowest drip to finish in-window.
_TAIL_SECONDS = 2 * sim.DAY


def _small_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Banded peer amounts folded under the cap (deterministic, no redraw loop)."""
    amount = sim._banded_peer_amounts(rng, n)
    over = amount > SMALL_AMOUNT_CAP
    span = SMALL_AMOUNT_CAP - sim.PEER_BAND_LO
    amount[over] = sim.PEER_BAND_LO + (amount[over] - sim.PEER_BAND_LO) % span
    return np.minimum(amount, SMALL_AMOUNT_CAP)


def _small_chunks(rng: np.random.Generator, total: float) -> list[float]:
    """Smear ``total`` into transfers that are each at most the cap."""
    chunks: list[float] = []
    remaining = float(total)
    while remaining > SMALL_AMOUNT_CAP:
        cut = min(float(_small_amounts(rng, 1)[0]), remaining)
        chunks.append(cut)
        remaining -= cut
    if remaining >= sim.PEER_MIN_CHUNK:
        chunks.append(remaining)
    return chunks


def slow_drip_ring_activity(
    rng: np.random.Generator,
    ledger: sim._Ledger,
    rings: list[sim._Ring],
    opened_s: np.ndarray,
    horizon_s: int,
    externals: np.ndarray,
) -> None:
    """Drop-in replacement for ``simulator.generator._ring_activity``."""
    for ring in rings:
        members = ring.members
        active_from = max(0, int(max(opened_s[m] for m in members)) + sim.DAY)
        latest_start = horizon_s - _TAIL_SECONDS
        if active_from >= latest_start:
            active_from = max(0, latest_start - sim.DAY)

        # Waves get disjoint time slots. Overlapping waves would let a NEW
        # wave's credits land inside an OLD wave's slow drip, re-creating the
        # fast in-then-out bursts the pattern is defined to avoid (seen in a
        # first draft: the drip was then not slower than the rule's ceiling).
        n_waves = min(
            int(rng.integers(4, 9)), max(1, (latest_start - active_from) // WAVE_SPAN_SECONDS)
        )
        slot = (latest_start - active_from) // n_waves
        for w in range(n_waves):
            slack = max(slot - WAVE_SPAN_SECONDS, 1)
            wave_t0 = active_from + w * slot + int(rng.integers(0, slack))
            last_out = wave_t0
            wave_total = 0.0
            for mule in ring.mules:
                k_in = int(rng.integers(3, 8))
                n_src = min(len(ring.sources), int(rng.integers(1, 3)))
                senders = [int(x) for x in rng.choice(ring.sources, size=n_src, replace=False)]
                senders += [
                    int(x) for x in externals[rng.integers(0, len(externals), k_in - n_src)]
                ]
                in_ts = wave_t0 + rng.integers(0, 3 * 3600, len(senders))
                in_amt = _small_amounts(rng, len(senders))
                ledger.add(
                    np.array(senders, dtype=np.int64),
                    np.full(len(senders), mule, dtype=np.int64),
                    in_amt,
                    in_ts,
                    "p2p",
                )

                forwarded = float(in_amt.sum()) * (1.0 - float(rng.uniform(0.01, 0.05)))
                chunks = _small_chunks(rng, forwarded)
                if not chunks:
                    continue
                gaps = rng.integers(DRIP_SPACING_RANGE[0], DRIP_SPACING_RANGE[1] + 1, len(chunks))
                gaps[0] = rng.integers(DRIP_DELAY_RANGE[0], DRIP_DELAY_RANGE[1] + 1)
                out_ts = np.minimum(int(in_ts.max()) + np.cumsum(gaps), horizon_s - 1)
                ledger.add(
                    np.full(len(chunks), mule, dtype=np.int64),
                    np.full(len(chunks), ring.collector, dtype=np.int64),
                    np.asarray(chunks, dtype=np.float64),
                    out_ts,
                    "instant",
                )
                last_out = max(last_out, int(out_ts.max()))
                wave_total += float(sum(chunks))

            cash_chunks = _small_chunks(rng, wave_total * (1.0 - float(rng.uniform(0.01, 0.05))))
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


@contextmanager
def _slow_drip_patched() -> Iterator[None]:
    original = sim._ring_activity
    sim._ring_activity = slow_drip_ring_activity  # type: ignore[assignment]
    try:
        yield
    finally:
        sim._ring_activity = original  # type: ignore[assignment]


def overlay_params(base: SimParams | None = None) -> SimParams:
    """Canonical ``SimParams`` (or ``base``) with the overlay seed and 2-3 bank spans."""
    return replace(base or SimParams(), span_mix=OVERLAY_SPAN_MIX, seed=OVERLAY_SEED)


def generate_overlay(
    out_dir: Path | None = None, params: SimParams | None = None
) -> dict[str, Path]:
    """Write the six overlay tables (ground truth included) into ``out_dir``.

    Refuses to write into the canonical simulator output directory.
    """
    target = Path(out_dir) if out_dir is not None else OVERLAY_DIR
    canonical = Path(sim.SimulatorConfig.from_env().out_dir).resolve()
    if target.resolve() == canonical:
        raise ValueError(f"refusing to write the overlay into canonical output {canonical}")
    with _slow_drip_patched():
        return sim.generate(
            params or overlay_params(), hash_key=b"demo-key", out_dir=target, render_images=False
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the slow-drip overlay world.")
    parser.add_argument("--out", type=Path, default=OVERLAY_DIR)
    args = parser.parse_args(argv)
    for name, path in generate_overlay(args.out).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
