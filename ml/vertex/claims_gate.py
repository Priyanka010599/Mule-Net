"""Claims-level gate: did the cloud run reproduce the *conclusions*?

Why this exists beside :mod:`ml.vertex.gate`
--------------------------------------------
``ml/vertex/gate.py`` asks "is the cloud output bit-identical to the
local canonical run?" and the answer measured on ``run-20260927t124500``
is **no**. The diagnosis is in ``ml/README.md``: the data path is
bit-identical -- staged inputs, per-account cutoffs, fold-independent
features, ring/exit structure, OOF account ids and ``as_of`` instants all
match -- and only the XGBoost fit moves, because xgboost 3.4.1 is a
``win_amd64`` wheel locally and a ``manylinux_2_28`` wheel in the
container. Two different compiled binaries of ``tree_method="hist"``
cannot be made to agree by pinning a version, so bit-identical XGBoost
across platforms is not a thing this project can demand.

What it *can* demand is that the cloud run supports the same statements
``docs/claims.md`` publishes. That is what this module asserts, in three
checks:

1. **Direction at every horizon.** Every scope comparison in
   ``docs/claims.md`` -- every claim that exists as a cross-bank /
   single-bank pair -- must keep the sign it has locally, at **all nine**
   horizons, not only the one the claim quotes.
2. **Span ordering.** At T=5m the span-1 gradient must stay *near zero*
   and spans 2 and 3 must stay positive, and span3 > span2 > span1 must
   hold at every horizon.
3. **Headline values within tolerance** of the local canonical run, for
   every claim in ``docs/claims.md``.

The comparison set is **derived** from :data:`ml.claims.CLAIMS`, not
re-listed here. A claim added or retired there changes what this gate
checks, with no second list to keep in step -- the same reason
``docs/claims.md`` is generated rather than written.

Where the tolerance comes from
------------------------------
**The tolerance is the local run's own seed-to-seed spread.** Every
published number is a mean over the three seeds in
``ml.evaluate.SEEDS``; the seed is a nuisance parameter this project
already treats as not changing any conclusion, because it quotes the
mean and reports the spread beside it. So the spread across seeds is the
measured size of a difference this project has already decided it does
not read anything into, and it is the honest yardstick for "the cloud
landed in the same place".

Concretely, for one claim:

* take the **per-seed** quantity the claim averages (a ``*_mean`` leaf
  maps back to the per-seed ``runs[]`` / ``ring_catch_per_seed`` /
  ``per_seed`` record it was averaged from),
* compute ``max - min`` over the three seeds,
* pool that over every sibling of the claim at the same level of
  aggregation -- the other scope, the other spans, the other horizons --
  and take the **maximum**.

Pooling is deliberate: a spread estimated from three draws is a poor
estimate of anything, and for the full-window horizon (one seed) or a
quantity that happens to land identically on all three seeds it is
``0.0``, which would silently turn this into a bit-exact gate again. The
report prints both the pooled tolerance and the claim's own spread, so a
reader can see how much of the allowance came from pooling.

Pooling is over siblings at the **same level of aggregation**, not over
the whole file: a summary-level claim is not given the noise of a thin
span-3 stratum. That distinction is worth 7.6 vs 1.04 on
``false_alerts_per_catch``, so it is the difference between a tolerance
larger than the claim and a tolerance smaller than it.

Two families have **no** seed dimension anywhere -- ``exit_distribution``
is a property of the generated data, not of any fit. Their tolerance is
therefore ``0.0`` and they are required to match exactly. That is not an
oversight: the bit-exact diagnosis established the data path *is*
reproducible across platforms, so demanding exactness there is a live
check that would catch a data-path regression rather than a formality.

What this gate does NOT promise
-------------------------------
It does not say the cloud run reproduced the local run. It says the
cloud run does not contradict any published claim, at a resolution set
by the local run's own seed noise. A real effect smaller than that
resolution is invisible to it, and the span-1 gradient at T=5m
(+0.0085 local, spread 0.0349) is exactly such an effect -- which is why
the span-1 assertion is "near zero", not "positive". A cloud run could
pass this gate and still be a materially different model. Only
:mod:`ml.vertex.gate` rules that out, and it only ever will on the same
platform.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.claims import CLAIMS, SOURCES, Claim, resolve

#: The two scope tokens every paired claim differs by.
SCOPES = ("cross_bank", "single_bank")

#: The three ring-span strata tokens.
SPANS = ("span1", "span2", "span3")

#: Horizon the span-ordering claim is stated at, in ``docs/claims.md``.
SPAN_CLAIM_CUTOFF = 5

_CUTOFF = re.compile(r"\[cutoff_minutes=[^\]]+\]")

_SUMMARY = re.compile(
    r"^horizons\[cutoff_minutes=(?P<cut>[^\]]+)\]\.summary\."
    r"(?P<scope>cross_bank|single_bank)\.(?P<metric>.+)_mean$"
)
_STRATA = re.compile(
    r"^horizons\[cutoff_minutes=(?P<cut>[^\]]+)\]\.strata_summary\."
    r"(?P<scope>cross_bank|single_bank)\.(?P<span>span\d)\.(?P<metric>.+)_mean$"
)
_RING = re.compile(
    r"^horizons\[cutoff_minutes=(?P<cut>[^\]]+)\]\.ring_catch\."
    r"(?P<scope>cross_bank|single_bank)\.(?P<metric>\w+)$"
)
_LEAD = re.compile(r"^seed_mean\.(?P<rest>.+)$")


def _horizon(doc: Any, cutoff: str) -> Any:
    """The one horizon whose ``cutoff_minutes`` matches, by field not index."""
    return resolve(doc, f"horizons[cutoff_minutes={cutoff}]")


def _walk(node: Any, dotted: str) -> Any:
    for step in dotted.split("."):
        node = node[step]
    return node


def per_seed_values(doc: Any, path: str) -> dict[int, float]:
    """The per-seed quantities the ``*_mean`` at ``path`` was averaged from.

    Returns ``{}`` for a path with no seed dimension at all, which is how
    ``exit_distribution`` -- a property of the data rather than of a fit
    -- reports that its tolerance must be zero. Raises ``KeyError`` for a
    path that does not exist, which is how sibling enumeration discovers
    that a substitution it tried is not a real measurement.
    """
    match = _SUMMARY.match(path)
    if match:
        horizon = _horizon(doc, match["cut"])
        return {
            run["seed"]: run[match["metric"]]
            for run in horizon["runs"]
            if run["scope"] == match["scope"]
        }
    match = _STRATA.match(path)
    if match:
        horizon = _horizon(doc, match["cut"])
        return {
            run["seed"]: run["strata"][match["span"]][match["metric"]]
            for run in horizon["runs"]
            if run["scope"] == match["scope"]
        }
    match = _RING.match(path)
    if match:
        horizon = _horizon(doc, match["cut"])
        per_seed = horizon["ring_catch_per_seed"][match["scope"]]
        return {int(seed): record[match["metric"]] for seed, record in per_seed.items()}
    match = _LEAD.match(path)
    if match:
        return {
            int(seed): _walk(record, match["rest"]) for seed, record in doc["per_seed"].items()
        }
    return {}


def seed_values(doc: Any, path: str, minus: str | None) -> dict[int, float]:
    """Per-seed values of a claim, differencing seed-for-seed when derived."""
    base = per_seed_values(doc, path)
    if minus is None:
        return base
    other = per_seed_values(doc, minus)
    return {seed: base[seed] - other[seed] for seed in base if seed in other}


def spread(values: dict[int, float]) -> float:
    """``max - min`` over the seeds; ``0.0`` when fewer than two exist."""
    if len(values) < 2:
        return 0.0
    return max(values.values()) - min(values.values())


def _siblings(paths: tuple[str, ...], doc: Any, vary_scope: bool) -> list[tuple[str, ...]]:
    """Sibling measurements at the *same level of aggregation* as ``paths``.

    ``paths`` is one path, or the two paths of a derived difference. Every
    substitution is applied to **all** of them at once: varying the two
    halves of a difference independently would pair ``cross_bank`` with
    ``cross_bank`` and estimate the spread of zero.

    ``vary_scope`` is false for a difference, because there the scope is
    the contrast being measured and substituting it destroys the
    quantity. A substitution that is not a real measurement raises in
    :func:`seed_values` and is dropped by the caller.
    """
    groups = ((SCOPES, SPANS) if vary_scope else (SPANS,))
    out = {paths}
    for group in groups:
        present = [token for token in group if token in paths[0]]
        if len(present) == 1:
            out = {
                tuple(p.replace(present[0], token) for p in sibling)
                for sibling in out
                for token in group
            }
    if _CUTOFF.search(paths[0]):
        cutoffs = [h["cutoff_minutes"] for h in doc["horizons"]]
        out = {
            tuple(_CUTOFF.sub(f"[cutoff_minutes={cutoff}]", p) for p in sibling)
            for sibling in out
            for cutoff in cutoffs
        }
    return sorted(out)


@dataclass(frozen=True)
class Tolerance:
    """One claim's allowance, and enough to audit where it came from."""

    value: float
    own_spread: float
    pooled_over: int


def tolerance(doc: Any, path: str, minus: str | None) -> Tolerance:
    """Pooled seed-to-seed spread for the quantity ``path`` measures."""
    paths = (path,) if minus is None else (path, minus)
    spreads: list[float] = []
    for sibling in _siblings(paths, doc, vary_scope=minus is None):
        try:
            sibling_minus = sibling[1] if len(sibling) > 1 else None
            spreads.append(spread(seed_values(doc, sibling[0], sibling_minus)))
        except (KeyError, IndexError):
            continue
    return Tolerance(
        value=max(spreads, default=0.0),
        own_spread=spread(seed_values(doc, path, minus)),
        pooled_over=len(spreads),
    )


def sign_at(value: float, tol: float) -> int:
    """Sign of ``value`` at a resolution of ``tol``; ``0`` means a tie.

    A comparison whose gap is inside the seed noise has no direction to
    reproduce, and calling that a pass would be dishonest. It is reported
    as ``TIE``, and the cloud is required to tie in the same place.
    """
    if abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1


@dataclass(frozen=True)
class Pair:
    """A cross-bank / single-bank comparison derived from the claim set."""

    id: str
    source: str
    cross_path: str
    single_path: str


def derive_pairs(claims: list[Claim] | None = None) -> list[Pair]:
    """Find every scope comparison in the claim set, by path identity.

    Two claims are the same comparison when one path is the other with
    ``cross_bank`` swapped for ``single_bank``. Deriving the set means a
    claim added to ``ml/claims.py`` is gated automatically and a claim
    retired there stops being gated, with no second list to maintain.
    """
    claims = CLAIMS if claims is None else claims
    by_path = {(claim.source, claim.path): claim for claim in claims if claim.minus is None}
    pairs: list[Pair] = []
    for (source, path), claim in sorted(by_path.items()):
        if "cross_bank" not in path:
            continue
        single = path.replace("cross_bank", "single_bank")
        if (source, single) not in by_path:
            continue
        pairs.append(
            Pair(
                id=claim.id.removesuffix("_cross_bank"),
                source=source,
                cross_path=path,
                single_path=single,
            )
        )
    return pairs


def _at_every_horizon(path: str, doc: Any) -> list[str]:
    """``path`` at every horizon, in grid order; ``[path]`` if not horizoned."""
    if not _CUTOFF.search(path):
        return [path]
    return [_CUTOFF.sub(f"[cutoff_minutes={h['cutoff_minutes']}]", path) for h in doc["horizons"]]


def _cutoff_label(path: str) -> str:
    found = _CUTOFF.search(path)
    if not found:
        return "-"
    cutoff = found.group(0).split("=", 1)[1].rstrip("]")
    return "full" if cutoff == "None" else f"T={cutoff}m"


def check_directions(
    local: dict[str, Any], cloud: dict[str, Any], pairs: list[Pair] | None = None
) -> dict[str, Any]:
    """Every derived scope comparison keeps its sign, at every horizon."""
    rows: list[dict[str, Any]] = []
    for pair in derive_pairs() if pairs is None else pairs:
        local_doc, cloud_doc = local[pair.source], cloud[pair.source]
        for cross in _at_every_horizon(pair.cross_path, local_doc):
            single = cross.replace("cross_bank", "single_bank")
            tol = tolerance(local_doc, cross, single).value
            local_delta = resolve(local_doc, cross) - resolve(local_doc, single)
            cloud_delta = resolve(cloud_doc, cross) - resolve(cloud_doc, single)
            local_sign = sign_at(local_delta, tol)
            cloud_sign = sign_at(cloud_delta, tol)
            rows.append(
                {
                    "id": pair.id,
                    "at": _cutoff_label(cross),
                    "tolerance": tol,
                    "local_delta": local_delta,
                    "cloud_delta": cloud_delta,
                    "local_sign": local_sign,
                    "cloud_sign": cloud_sign,
                    "passed": local_sign == cloud_sign,
                }
            )
    return {"name": "directions", "rows": rows, "passed": all(r["passed"] for r in rows)}


def _gradient_paths(span: int, cutoff: Any) -> tuple[str, str]:
    stem = f"horizons[cutoff_minutes={cutoff}].strata_summary"
    return (
        f"{stem}.cross_bank.span{span}.pr_auc_mean",
        f"{stem}.single_bank.span{span}.pr_auc_mean",
    )


def check_span_ordering(local: dict[str, Any], cloud: dict[str, Any]) -> dict[str, Any]:
    """Span-1 near zero, spans 2 and 3 positive, and the ordering intact.

    "Near zero" is **the seed-to-seed spread of that same gradient at that
    same horizon** -- not the pooled tolerance the value check uses. The
    span-1 gradient is a statement about one quantity at one horizon, and
    its own spread is the resolution at which that quantity was measured:
    locally +0.0085 with a spread of 0.0349 across seeds, and the per-seed
    signs are ``-, +, -``. A quantity whose sign is not even consistent
    across the three seeds it was averaged over is not a positive
    finding, and the honest claim is that it is indistinguishable from
    zero. Spans 2 and 3 must clear that same bar *and* be positive on
    every individual seed, which is the strictly stronger statement.
    """
    rows: list[dict[str, Any]] = []
    local_doc, cloud_doc = local["early_detection"], cloud["early_detection"]
    cross, single = _gradient_paths(1, SPAN_CLAIM_CUTOFF)
    near_zero = spread(seed_values(local_doc, cross, single))

    for span in (1, 2, 3):
        cross, single = _gradient_paths(span, SPAN_CLAIM_CUTOFF)
        for label, doc in (("local", local_doc), ("cloud", cloud_doc)):
            delta = resolve(doc, cross) - resolve(doc, single)
            per_seed = seed_values(doc, cross, single)
            all_positive = bool(per_seed) and all(v > 0 for v in per_seed.values())
            if span == 1:
                passed, want = abs(delta) <= near_zero, f"|d| <= {near_zero:.4f}"
            else:
                passed = delta > near_zero and all_positive
                want = f"d > {near_zero:.4f}, every seed > 0"
            rows.append(
                {
                    "check": f"span{span} gradient @ T={SPAN_CLAIM_CUTOFF}m ({label})",
                    "value": delta,
                    "requirement": want,
                    "seeds": sorted(round(v, 4) for v in per_seed.values()),
                    "passed": passed,
                }
            )

    for horizon in local_doc["horizons"]:
        cutoff = horizon["cutoff_minutes"]
        label_at = _cutoff_label(f"[cutoff_minutes={cutoff}]")
        for label, doc in (("local", local_doc), ("cloud", cloud_doc)):
            deltas = [
                resolve(doc, c) - resolve(doc, s)
                for c, s in (_gradient_paths(span, cutoff) for span in (1, 2, 3))
            ]
            rows.append(
                {
                    "check": f"span3 > span2 > span1 @ {label_at} ({label})",
                    "value": deltas[2] - deltas[1],
                    "requirement": "strictly ordered by span",
                    "seeds": [round(d, 4) for d in deltas],
                    "passed": deltas[2] > deltas[1] > deltas[0],
                }
            )
    return {
        "name": "span_ordering",
        "near_zero": near_zero,
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }


def check_values(
    local: dict[str, Any], cloud: dict[str, Any], claims: list[Claim] | None = None
) -> dict[str, Any]:
    """Every claim in ``docs/claims.md``, cloud value within tolerance."""
    rows: list[dict[str, Any]] = []
    for claim in CLAIMS if claims is None else claims:
        tol = tolerance(local[claim.source], claim.path, claim.minus)
        local_value = claim.value(local)
        cloud_value = claim.value(cloud)
        delta = cloud_value - local_value
        rows.append(
            {
                "id": claim.id,
                "local": local_value,
                "cloud": cloud_value,
                "delta": delta,
                "tolerance": tol.value,
                "own_spread": tol.own_spread,
                "pooled_over": tol.pooled_over,
                "passed": abs(delta) <= tol.value,
            }
        )
    return {"name": "values", "rows": rows, "passed": all(r["passed"] for r in rows)}


_SIGN = {1: "cb>sb", -1: "cb<sb", 0: "TIE"}


def format_report(directions: dict, spans: dict, values: dict) -> str:
    """The gate's verdict, per claim, with every tolerance shown."""
    lines = [
        "=== 1. claim directions, at every horizon ===",
        f"{'claim':<26}{'at':<8}{'tol':>9}{'local d':>11}{'cloud d':>11}"
        f"{'local':>8}{'cloud':>8}  verdict",
    ]
    for row in directions["rows"]:
        lines.append(
            f"{row['id']:<26}{row['at']:<8}{row['tolerance']:>9.4f}"
            f"{row['local_delta']:>+11.4f}{row['cloud_delta']:>+11.4f}"
            f"{_SIGN[row['local_sign']]:>8}{_SIGN[row['cloud_sign']]:>8}"
            f"  {'ok' if row['passed'] else 'FAIL'}"
        )
    lines += ["", f"=== 2. span ordering (near-zero band = {spans['near_zero']:.4f}) ==="]
    for row in spans["rows"]:
        lines.append(
            f"{row['check']:<48}{row['value']:>+10.4f}  {row['requirement']:<34}"
            f"{str(row['seeds']):<32}{'ok' if row['passed'] else 'FAIL'}"
        )
    lines += [
        "",
        "=== 3. headline values vs the local canonical run ===",
        f"{'claim':<34}{'local':>12}{'cloud':>12}{'delta':>11}{'tol':>11}"
        f"{'own':>11}{'pool':>6}  verdict",
    ]
    for row in values["rows"]:
        lines.append(
            f"{row['id']:<34}{row['local']:>12.4f}{row['cloud']:>12.4f}"
            f"{row['delta']:>+11.4f}{row['tolerance']:>11.4f}{row['own_spread']:>11.4f}"
            f"{row['pooled_over']:>6}  {'ok' if row['passed'] else 'FAIL'}"
        )
    lines.append("")
    for name, report in (("directions", directions), ("span_ordering", spans), ("values", values)):
        failures = [r for r in report["rows"] if not r["passed"]]
        lines.append(
            f"{name:<16}: {'PASS' if report['passed'] else 'FAIL'} "
            f"({len(report['rows']) - len(failures)}/{len(report['rows'])} ok)"
        )
    passed = directions["passed"] and spans["passed"] and values["passed"]
    lines.append("CLAIMS GATE: " + ("PASSED" if passed else "FAILED"))
    return "\n".join(lines)


def load(experiments_dir: Path) -> dict[str, Any]:
    """Load both metrics files from an ``experiments/`` directory."""
    docs: dict[str, Any] = {}
    for name, relative in SOURCES.items():
        path = experiments_dir / Path(relative).relative_to("ml/experiments")
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        docs[name] = json.loads(path.read_text("utf-8"))
    return docs


def run(local_dir: Path, cloud_dir: Path) -> tuple[bool, str]:
    """Run all three checks and render the report."""
    local, cloud = load(local_dir), load(cloud_dir)
    directions = check_directions(local, cloud)
    spans = check_span_ordering(local, cloud)
    values = check_values(local, cloud)
    passed = directions["passed"] and spans["passed"] and values["passed"]
    return passed, format_report(directions, spans, values)


def main(argv: list[str] | None = None) -> int:
    """CLI: non-zero when any of the three checks fails."""
    parser = argparse.ArgumentParser(description="claims-level cloud-vs-local gate")
    parser.add_argument(
        "--local-experiments",
        type=Path,
        default=Path("ml/experiments"),
        help="local canonical experiments/ directory",
    )
    parser.add_argument(
        "--cloud-experiments",
        type=Path,
        required=True,
        help="downloaded cloud experiments/ directory",
    )
    args = parser.parse_args(argv)
    passed, report = run(args.local_experiments, args.cloud_experiments)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
