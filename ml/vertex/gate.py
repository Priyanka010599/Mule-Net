"""Compare a cloud run's ``metrics.json`` against the local canonical file.

What this gate actually asserts
------------------------------
**Every JSON leaf of both metrics files is equal, exactly, except the
single top-level ``generated_at`` string in each.**

That exception is not a weakening chosen for convenience, it is forced:
``ml.early.main`` and ``ml.leadtime.main`` both write
``"generated_at": datetime.now(UTC).isoformat()`` into the file they
produce. A byte-for-byte gate on ``metrics.json`` therefore cannot pass
*by construction*, in exactly the same way one on ``ml/out/`` cannot --
so the honest statement of the gate is "all leaves but the wall clock",
and :data:`IGNORED_PATHS` is the whole list of what is skipped. It holds
exactly one path -- the top-level ``generated_at`` -- and the match is on
the full dotted path, so a ``generated_at`` nested anywhere else would
still be compared.

Leaves are compared by :func:`repr`, not by tolerance. A float that
differs in its last bit is a **failure** and is reported with old value,
new value and delta. Nothing here rounds, and nothing here has a
tolerance knob to turn: the point of the gate is to find out whether the
cloud reproduced the local run, and a tolerance would answer a different
and easier question.

The second half of the gate is genuinely bit-exact
--------------------------------------------------
:func:`compare_bytes` md5s the artefacts that carry **no** wall clock --
the 50 files of the out-of-fold score store and
``lead_time/per_ring.parquet`` -- and requires them byte-for-byte
identical. Those files were measured reproducible across two local runs
(50/50 and 1/1 identical), so the word "bit-exact" applies to them
without an exception list. Between the two halves, the only thing in the
whole output that is not compared bit-for-bit is two timestamp strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

#: Leaf paths excluded from the comparison, and the only one.
#: ``generated_at`` is a ``datetime.now(UTC)`` wall clock written by the
#: module that produced the file, so it can never match across two runs.
IGNORED_PATHS: frozenset[str] = frozenset({"generated_at"})


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to ``path -> leaf`` with dotted paths.

    List elements get their index in the path, so a reordered list is a
    difference rather than a silent pass.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _delta(old: Any, new: Any) -> str:
    """Numeric delta when both leaves are numbers, else a dash."""
    if isinstance(old, bool) or isinstance(new, bool):
        return "-"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return f"{new - old:+.17g}"
    return "-"


def compare(local: Any, cloud: Any) -> dict[str, Any]:
    """Leaf-by-leaf comparison of two decoded metrics documents.

    Returns a report carrying the compared-leaf count and, for every
    mismatch, the leaf path with its local value, cloud value and delta.
    ``missing``/``extra`` cover leaves present on only one side, which is
    a structural difference and just as much a failure as a wrong number.
    """
    flat_local = flatten(local)
    flat_cloud = flatten(cloud)
    ignored = sorted(p for p in flat_local | flat_cloud if p in IGNORED_PATHS)

    keys_local = set(flat_local) - IGNORED_PATHS
    keys_cloud = set(flat_cloud) - IGNORED_PATHS
    shared = keys_local & keys_cloud

    differences = [
        {
            "path": path,
            "local": flat_local[path],
            "cloud": flat_cloud[path],
            "delta": _delta(flat_local[path], flat_cloud[path]),
        }
        for path in sorted(shared)
        if repr(flat_local[path]) != repr(flat_cloud[path])
    ]
    return {
        "leaves_compared": len(shared),
        "leaves_ignored": ignored,
        "missing_in_cloud": sorted(keys_local - keys_cloud),
        "extra_in_cloud": sorted(keys_cloud - keys_local),
        "differences": differences,
        "passed": not differences and keys_local == keys_cloud and len(shared) > 0,
    }


def compare_files(local_path: Path, cloud_path: Path) -> dict[str, Any]:
    """Compare two ``metrics.json`` files on disk."""
    return compare(
        json.loads(local_path.read_text("utf-8")),
        json.loads(cloud_path.read_text("utf-8")),
    )


def format_report(name: str, report: dict[str, Any]) -> str:
    """Render one comparison as the gate's human-readable verdict."""
    lines = [
        f"=== {name} ===",
        f"leaves compared : {report['leaves_compared']}",
        f"leaves ignored  : {', '.join(report['leaves_ignored']) or '(none)'}",
        f"missing in cloud: {len(report['missing_in_cloud'])}",
        f"extra in cloud  : {len(report['extra_in_cloud'])}",
        f"differing leaves: {len(report['differences'])}",
        f"VERDICT         : {'PASS' if report['passed'] else 'FAIL'}",
    ]
    for path in report["missing_in_cloud"]:
        lines.append(f"  MISSING IN CLOUD  {path}")
    for path in report["extra_in_cloud"]:
        lines.append(f"  EXTRA IN CLOUD    {path}")
    for diff in report["differences"]:
        lines.append(
            f"  DIFF {diff['path']}\n"
            f"       local = {diff['local']!r}\n"
            f"       cloud = {diff['cloud']!r}\n"
            f"       delta = {diff['delta']}"
        )
    return "\n".join(lines)


def md5(path: Path) -> str:
    """Hex md5 of one file, read in chunks."""
    digest = hashlib.md5()  # noqa: S324 - integrity check, not a security hash
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_bytes(local_dir: Path, cloud_dir: Path, pattern: str) -> dict[str, Any]:
    """Require every file matching ``pattern`` byte-identical on both sides.

    Unlike :func:`compare` this has no exception list, because the files
    it covers contain no timestamp. A file present on one side only is a
    failure, so a cloud run that simply wrote fewer artefacts cannot pass.
    """
    local = {p.name: p for p in sorted(local_dir.glob(pattern))}
    cloud = {p.name: p for p in sorted(cloud_dir.glob(pattern))}
    shared = sorted(set(local) & set(cloud))
    differences = [
        {
            "path": name,
            "local": md5(local[name]),
            "cloud": md5(cloud[name]),
            "delta": "-",
        }
        for name in shared
        if md5(local[name]) != md5(cloud[name])
    ]
    return {
        "leaves_compared": len(shared),
        "leaves_ignored": [],
        "missing_in_cloud": sorted(set(local) - set(cloud)),
        "extra_in_cloud": sorted(set(cloud) - set(local)),
        "differences": differences,
        "passed": not differences and set(local) == set(cloud) and bool(shared),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: compare paired local/cloud metrics files; non-zero on any fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        nargs=3,
        metavar=("NAME", "LOCAL", "CLOUD"),
        help="named pair of metrics.json paths to compare; repeatable",
    )
    parser.add_argument(
        "--bytes",
        action="append",
        default=[],
        nargs=4,
        metavar=("NAME", "LOCAL_DIR", "CLOUD_DIR", "PATTERN"),
        help="named pair of directories compared byte-for-byte; repeatable",
    )
    args = parser.parse_args(argv)

    failed = False
    for name, local_raw, cloud_raw in args.pair:
        report = compare_files(Path(local_raw), Path(cloud_raw))
        print(format_report(name, report))
        print()
        failed = failed or not report["passed"]
    for name, local_dir, cloud_dir, pattern in args.bytes:
        report = compare_bytes(Path(local_dir), Path(cloud_dir), pattern)
        print(format_report(f"{name} (byte-for-byte)", report))
        print()
        failed = failed or not report["passed"]
    print("GATE: " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
