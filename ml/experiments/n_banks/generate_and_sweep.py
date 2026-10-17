"""Does the cross-bank advantage hold with more than 3 banks?

Standalone experiment -- never imports into or writes to simulator/out,
ml/out, docs/claims.md, the agents service, or the UI. Generates synthetic
N-bank worlds (N=5, N=6) by monkeypatching simulator.generator.BANKS for
the duration of one generate() call, never editing simulator/ itself, then
runs the same ring-recall-by-coalition-size sweep ml/coalition.py already
runs for the shipped 3-bank world.

Comparability choices (see README.md in this directory for the reasoning):
  - accounts_per_bank held at 10,000 (same per-bank population as the
    shipped world).
  - rings_per_pattern scaled with bank count to hold the MULE SHARE of
    the population constant: x5/3 for N=5, x2 for N=6.
  - span_mix left at the shipped (0.30, 0.40, 0.30) over spans 1-3 banks.
    rng_ring.choice(n_banks, size=span, replace=False) in
    simulator/generator.py already draws WHICH banks a ring touches from
    every bank in BANKS, not just the first 3 -- so leaving span_mix
    unchanged already gives "span distribution fixed, bank pool larger"
    with no code change at all. This is the PRIMARY variant.
  - SCALE_HARD_NEGATIVES (below): when True, n_businesses/n_families/
    n_dual_bank are scaled by the same ratio as rings_per_pattern,
    instead of left at the shipped defaults -- so the hard-negative
    cohorts stay the same SHARE of a bigger world, same reasoning as
    the ring-count scaling above. Writes to n{N}_seed{seed}_hn/ and
    results_hn.json, a second pass alongside the unscaled run rather
    than overwriting it, so both stay on disk to compare.

Usage::

    python -m ml.experiments.n_banks.generate_and_sweep
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import simulator.generator as gen
from ml.coalition import by_size, sweep
from ml.config import MlConfig
from ml.features import load_tables

SEEDS: tuple[int, ...] = (42, 43, 44)

#: The shipped 3-bank world's value (simulator/generator.py's SimParams
#: default), scaled to hold mule share of the population constant as N
#: grows: rings_per_pattern x n_banks is (very roughly) the mule count,
#: since ring size does not depend on n_banks.
BASE_RINGS_PER_PATTERN = 30

#: shipped 3-bank world's SimParams defaults (simulator/generator.py).
BASE_N_BUSINESSES = 50
BASE_N_FAMILIES = 200
BASE_N_DUAL_BANK = 200

#: True once the hard-negative cohorts are scaled by the same ratio as
#: rings_per_pattern below, rather than left at the shipped defaults. A
#: second pass over the same N/seeds, in new directories (suffixed "_hn")
#: so the un-scaled run stays on disk for comparison rather than being
#: silently overwritten.
SCALE_HARD_NEGATIVES = True

WORLDS: dict[int, dict[str, Any]] = {
    5: {
        "banks": ("BANK_A", "BANK_B", "BANK_C", "BANK_D", "BANK_E"),
        "scale": 5 / 3,
        "rings_per_pattern": round(BASE_RINGS_PER_PATTERN * 5 / 3),  # 50
    },
    6: {
        "banks": ("BANK_A", "BANK_B", "BANK_C", "BANK_D", "BANK_E", "BANK_F"),
        "scale": 2.0,
        "rings_per_pattern": BASE_RINGS_PER_PATTERN * 2,  # 60
    },
}

OUT_ROOT = Path(__file__).parent / "data"
RESULTS_PATH = (
    Path(__file__).parent / ("results_hn.json" if SCALE_HARD_NEGATIVES else "results.json")
)


def generate_world(n_banks: int, seed: int) -> Path:
    """One seed of one N-bank world, written under this directory's data/.

    Idempotent: skips regeneration if ground_truth.parquet already exists,
    so a re-run after adding a seed does not re-pay the cost of the ones
    already done.
    """
    spec = WORLDS[n_banks]
    suffix = "_hn" if SCALE_HARD_NEGATIVES else ""
    out_dir = OUT_ROOT / f"n{n_banks}_seed{seed}{suffix}"
    if (out_dir / "ground_truth.parquet").exists():
        print(f"  n={n_banks} seed={seed}: already generated, skipping")
        return out_dir

    original_banks = gen.BANKS
    try:
        gen.BANKS = spec["banks"]
        scale = spec["scale"]
        params = gen.SimParams(
            rings_per_pattern=spec["rings_per_pattern"],
            seed=seed,
            **(
                {
                    "n_businesses": round(BASE_N_BUSINESSES * scale),
                    "n_families": round(BASE_N_FAMILIES * scale),
                    "n_dual_bank": round(BASE_N_DUAL_BANK * scale),
                }
                if SCALE_HARD_NEGATIVES
                else {}
            ),
        )
        gen.generate(params, out_dir=out_dir, render_images=False)
    finally:
        gen.BANKS = original_banks
    return out_dir


def sweep_world(out_dir: Path, n_banks: int, seed: int) -> list[dict[str, Any]]:
    """Ring recall by coalition size, same sweep ml/coalition.py runs."""
    base = MlConfig.from_env()
    cfg = dataclasses.replace(base, data_dir=out_dir, ml_dir=out_dir, seed=seed)
    tables = load_tables(cfg)
    results = sweep(tables, cfg)
    rows = by_size(results)
    for row in rows:
        row["n_banks"] = n_banks
        row["seed"] = seed
    return rows


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for n_banks, spec in WORLDS.items():
        for seed in SEEDS:
            print(f"==> n={n_banks} seed={seed}: generating "
                  f"({spec['rings_per_pattern']} rings/pattern)")
            out_dir = generate_world(n_banks, seed)
            print(f"==> n={n_banks} seed={seed}: sweeping coalitions")
            all_rows.extend(sweep_world(out_dir, n_banks, seed))

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "worlds": {str(n): spec for n, spec in WORLDS.items()},
        "rows": all_rows,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
