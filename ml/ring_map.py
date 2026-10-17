"""Map detector ring ids to ground-truth ring ids, with match quality attached.

``ml/out/rings.parquet`` names rings after the detector that found them
(``ring-mulenet-velocity-rule-v2-00007``, 45 of them);
``ml/out/ring_detection_timeline.parquet`` and
``simulator/out/ground_truth.parquet`` name them after the generator
(``ring_003``, 120 of them). The two id spaces **do not intersect at
all**, so a consumer holding a detected ring has no way to reach that
ring's ground truth, and the UI's replay view cannot join the timeline to
anything it displays. This table is that join, published here because the
provenance lives here.

The pairing, and why it is weaker than it looks
-----------------------------------------------
Each detected ring is paired with the true ring it shares the most
members with. That is the obvious definition and it is **not a reliable
statement of identity**, for a reason visible in the data rather than in
principle: the detector's clusters are connected components of the
transfer graph restricted to flagged accounts
(``ml.publish.high_risk_components``), and nothing stops one component
from swallowing several true rings that happen to transact with each
other.

Measured on this dataset: **7 of 45 detected rings overlap more than one
true ring**, and the worst spans four
(``...-00002``: 12 members of ``ring_092``, 6 of ``ring_017``, 6 of
``ring_003``, 3 of ``ring_009``). For that ring, "the ring being
replayed is ring_092" describes a minority of what the analyst is
looking at. One pair is an exact tie on overlap.

So ``true_ring_id`` is the *best available* pairing, not a verdict, and
every row carries the numbers needed to see how good it is:
:data:`AMBIGUITY_COLUMNS` mark the cases a UI must caveat rather than
present as fact. A consumer that renders ``true_ring_id`` without
consulting ``match_is_ambiguous`` will, on this data, tell a judge that
a four-ring merge is one ring.

Tie-breaking
------------
Deterministic and independent of dict, file and row order: the pair is
chosen by ``(-overlap_members, true_ring_id)``, i.e. **most overlap
wins; an exact tie is broken by the lexicographically smallest true ring
id**. The alternative -- leaving ties to iteration order -- would let the
published pairing change between runs on identical inputs, which for a
table the UI attributes to a judge is not acceptable. The tie itself is
not hidden: ``overlap_tied`` marks it.

Membership basis
----------------
Overlap is counted over **every** member of the detected ring, velocity
and identity-link alike, because that is the cluster the UI actually
displays. It does change one pairing: ``...-00013`` pairs with
``ring_016`` on all members and with ``ring_001`` on velocity members
alone. That ring is flagged ambiguous on independent grounds (it is the
exact tie), so a consumer honouring the flag is already warned.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ml.config import MlConfig
from ml.features import load_tables

#: Name of the published table, without extension.
TABLE_NAME = "ring_id_map"

#: The columns a consumer must consult before presenting a pairing as
#: fact. ``match_is_ambiguous`` is their disjunction, for a UI that wants
#: one gate rather than three.
AMBIGUITY_COLUMNS: tuple[str, ...] = (
    "overlap_tied",
    "true_rings_touched",
    "detected_rings_touching_true_ring",
)


def true_ring_of(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """hashed_account_id -> ground-truth ring_id, ring members only."""
    truth = tables["ground_truth"]
    in_ring = truth.loc[truth["ring_id"] != ""]
    return in_ring.set_index("hashed_account_id")["ring_id"]


def overlap_matrix(rings: pd.DataFrame, true_of: pd.Series) -> dict[str, Counter[str]]:
    """detected_ring_id -> Counter of true ring ids over its members."""
    lookup = true_of.to_dict()
    matrix: dict[str, Counter[str]] = {}
    for row in rings.itertuples():
        members = [m["hashed_account_id"] for m in row.members]
        matrix[row.ring_id] = Counter(lookup[account] for account in members if account in lookup)
    return matrix


def best_match(counts: Counter[str]) -> tuple[str | None, int, bool]:
    """``(true_ring_id, overlap, tied)`` under the documented tie-break.

    Most overlap wins; an exact tie goes to the lexicographically
    smallest true ring id, so the result never depends on iteration
    order. Returns ``(None, 0, False)`` for a detected ring that overlaps
    no true ring at all -- a false-positive cluster, which is a real
    finding and keeps its row.
    """
    if not counts:
        return None, 0, False
    ring_id, overlap = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    tied = sum(1 for n in counts.values() if n == overlap) > 1
    return ring_id, overlap, tied


def build_map(cfg: MlConfig) -> pd.DataFrame:
    """One row per detected ring, with its best true-ring pairing."""
    rings = pd.read_parquet(cfg.ml_dir / "out" / "rings.parquet")
    tables = load_tables(cfg)
    true_of = true_ring_of(tables)
    true_size = true_of.groupby(true_of).size()

    matrix = overlap_matrix(rings, true_of)
    sizes = {row.ring_id: len(row.members) for row in rings.itertuples()}

    matched: dict[str, tuple[str | None, int, bool]] = {
        detected: best_match(counts) for detected, counts in matrix.items()
    }
    # How many detected rings claim each true ring as their best match,
    # and how many merely overlap it. The first is the "same replay under
    # two rings" risk; the second catches a true ring split across
    # clusters where neither claims it.
    claims = Counter(t for t, _, _ in matched.values() if t is not None)
    touches = Counter(true_id for counts in matrix.values() for true_id in counts)

    rows: list[dict[str, object]] = []
    for detected in sorted(matched):
        true_id, overlap, tied = matched[detected]
        touched = len(matrix[detected])
        claimed = claims[true_id] if true_id else 0
        touching = touches[true_id] if true_id else 0
        rows.append(
            {
                "detected_ring_id": detected,
                "true_ring_id": true_id,
                "overlap_members": overlap,
                "detected_ring_size": sizes[detected],
                "true_ring_size": int(true_size[true_id]) if true_id else None,
                # Merge detector: >1 means this cluster spans several
                # true rings and the pairing names only the largest.
                "true_rings_touched": touched,
                # >1 means two detected rings would replay as the same
                # true ring.
                "detected_rings_claiming_true_ring": claimed,
                # >1 means the true ring is split across clusters.
                "detected_rings_touching_true_ring": touching,
                "overlap_tied": tied,
                "match_is_ambiguous": bool(tied or touched > 1 or touching > 1),
            }
        )

    frame = pd.DataFrame(rows)
    frame["true_ring_size"] = frame["true_ring_size"].astype("Int64")
    return frame


def update_manifest(cfg: MlConfig, frame: pd.DataFrame) -> Path:
    """Record the table in ``ml/out/manifest.json`` beside the others."""
    path = cfg.ml_dir / "out" / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["regenerated_by"] = "python -m ml.publish_all"
    unmatched = int(frame["true_ring_id"].isna().sum())
    manifest["tables"][TABLE_NAME] = {
        "rows": len(frame),
        # No shared/ contract covers this table; shared/ is frozen until
        # after submission. Logged in docs/post_submission.md.
        "validated_against": None,
        "columns": list(frame.columns),
        "source": (
            "rings.parquet members[] vs simulator ground_truth.ring_id; "
            "NO model is fitted by ml.ring_map"
        ),
        "written_by": (
            "ml.ring_map, which must run AFTER ml.publish (it reads rings.parquet "
            "and appends to this manifest). Use `python -m ml.publish_all`."
        ),
        "purpose": (
            "joins the detector ring id space (rings.parquet, 45 rings) to the "
            "ground-truth space (ring_detection_timeline / ground_truth, 120 "
            "rings); the two id spaces do not intersect"
        ),
        "pairing_rule": (
            "the true ring sharing the most members with the detected ring, over "
            "ALL members (velocity and identity_link); ties broken by the "
            "lexicographically smallest true_ring_id, so the pairing never "
            "depends on row or dict order"
        ),
        "true_ring_id": (
            f"NULL for the {unmatched} detected ring(s) overlapping no true ring "
            "at all (false-positive clusters); those rows are kept deliberately"
        ),
        "match_is_ambiguous": (
            "TRUE when the pairing must not be presented as identity: overlap tied, "
            "or the detected ring spans >1 true ring (a merge), or the true ring is "
            "overlapped by >1 detected ring (a split). A consumer rendering "
            "true_ring_id without checking this will describe a merge as one ring"
        ),
        "caveat": (
            "best-member-overlap is a heuristic, not a verdict: detector clusters "
            "are connected components over flagged accounts, so one cluster can "
            "swallow several true rings that transact with each other"
        ),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    return path


def main() -> None:
    """Build the id map and record it in the manifest."""
    cfg = MlConfig.from_env()
    frame = build_map(cfg)
    out_dir = cfg.ml_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TABLE_NAME}.parquet"
    frame.to_parquet(path, index=False)
    print(f"wrote {path} rows={len(frame)}")
    print(
        f"  matched={int(frame['true_ring_id'].notna().sum())} "
        f"unmatched={int(frame['true_ring_id'].isna().sum())} "
        f"ambiguous={int(frame['match_is_ambiguous'].sum())} "
        f"merges={int((frame['true_rings_touched'] > 1).sum())} "
        f"tied={int(frame['overlap_tied'].sum())}"
    )
    print(f"wrote {update_manifest(cfg, frame)}")


if __name__ == "__main__":
    main()
