"""Run the real, Gemini-backed draft_report() over every ring.

One-off batch job, not a service module. Replaces the Report tab's
deterministic ui/build_snapshot.py placeholder (four hardcoded
US/FATF citations, no Gemini call at all) with real drafts, each
citing the real RBI/PMLA/FIU-IND index wired up this session.

Resumable: writes one JSON line per completed ring to OUT_PATH as it
finishes, and a re-run skips any ring already recorded there with
success=True. Concurrency is a ProcessPoolExecutor, not threads --
agents.adk_agents.LAST_USAGE is a process-global dict with no locking,
so the only way to get a correct per-ring cost reading under real
concurrency is to give each in-flight ring its own process.

Usage::

    python -m agents.batch_draft_reports [--budget-usd 30] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents.config import DEFAULT_LOCATION, DEFAULT_MODEL

OUT_PATH = Path("agents/out/batch_str_reports.jsonl")

#: Env vars the worker process needs -- it imports agents.adk_agents
#: fresh, which re-reads AgentsConfig.from_env() per call.
_REQUIRED_ENV = {
    "MULENET_AGENT_MODE": "vertex",
    "GOOGLE_GENAI_USE_VERTEXAI": "1",
    "GOOGLE_CLOUD_PROJECT": "mulenet",
    "GOOGLE_CLOUD_LOCATION": DEFAULT_LOCATION,
    "MULENET_DOCAI_MODE": "cloud",
    "MULENET_REGULATIONS_ENGINE": "mulenet-aml-regulations-app",
    "MULENET_SEARCH_LOCATION": "global",
}


def _ring_sizes() -> list[tuple[str, int]]:
    """(ring_id, member_count), largest first."""
    df = pd.read_parquet("ml/out/rings.parquet")
    sizes = [(row.ring_id, len(row.members)) for row in df.itertuples()]
    sizes.sort(key=lambda pair: -pair[1])
    return sizes


def _case_ids() -> dict[str, str]:
    """ring_id -> case_id, from the same cases.parquet the UI reads."""
    df = pd.read_parquet("agents/out/cases.parquet")
    return dict(zip(df["ring_id"], df["case_id"], strict=True))


def _already_done() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("success"):
            done.add(record["ring_id"])
    return done


def _run_one_ring(ring_id: str, case_id: str) -> dict[str, Any]:
    """Runs in a worker process. Fresh LAST_USAGE, so no cross-ring leakage."""
    import os  # noqa: PLC0415

    for key, value in _REQUIRED_ENV.items():
        os.environ.setdefault(key, value)

    from agents import adk_agents, regulations  # noqa: PLC0415
    from shared.contracts import StrReport  # noqa: PLC0415

    adk_agents.reset_usage()
    started = time.monotonic()
    try:
        draft = adk_agents.draft_report(ring_id)
    except Exception as exc:  # noqa: BLE001 - reported, not raised, to the parent
        return {
            "ring_id": ring_id,
            "case_id": case_id,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 1),
        }
    elapsed = time.monotonic() - started
    usage = dict(adk_agents.LAST_USAGE)

    # Zero mock-corpus citations allowed -- the whole point of this run
    # is replacing the hardcoded US/FATF placeholders with the real
    # index. is_synthetic() is the same check the live chat's verifier
    # already relies on (SYNTHETIC_MARKERS: PLACEHOLDER, SYNTHETIC,
    # mock://).
    mock_citations = [c for c in draft.citations if regulations.is_synthetic({"citation": c})]

    report = StrReport(
        case_id=case_id,
        narrative=draft.narrative,
        regulation_citations=draft.citations,
        citations_verified=draft.citations_verified,
    )

    from agents.eval.record import estimate_cost  # noqa: PLC0415

    cost_usd = estimate_cost(DEFAULT_MODEL, usage)

    return {
        "ring_id": ring_id,
        "case_id": case_id,
        "success": True,
        "report_id": report.report_id,
        "narrative": report.narrative,
        "regulation_citations": report.regulation_citations,
        "citations_verified": report.citations_verified,
        "generated_at": report.generated_at.isoformat(),
        "status": report.status.value,
        "grounds_count": len(draft.grounds),
        "narrative_withheld": draft.narrative_withheld,
        "mock_citations_found": mock_citations,
        "usage": usage,
        "cost_usd": cost_usd,
        "elapsed_s": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    # Must be set here, in the parent, before the pool spawns -- not
    # inside _run_one_ring. Windows multiprocessing uses "spawn": each
    # child re-imports this module fresh (running `import pandas as pd`
    # at the top) before _run_one_ring's body ever executes, and BLAS
    # reads its thread-pool-size env vars once, at numpy's own import
    # time. Setting them inside the worker function is too late; it has
    # to be in os.environ before the OS creates the child process.
    for key, value in {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done()
    sizes = _ring_sizes()
    case_ids = _case_ids()

    todo = [(ring_id, case_ids[ring_id]) for ring_id, _ in sizes if ring_id not in done]
    print(f"{len(sizes)} rings total, {len(done)} already done, {len(todo)} to run")
    if not todo:
        print("Nothing to do.")
        return 0

    cumulative_cost = 0.0
    budget_crossed = False
    succeeded = 0
    failed = 0
    started = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool, OUT_PATH.open(
        "a", encoding="utf-8"
    ) as out:
        futures = {}
        pending = list(todo)

        def _submit_next() -> None:
            if pending and not budget_crossed:
                ring_id, case_id = pending.pop(0)
                futures[pool.submit(_run_one_ring, ring_id, case_id)] = ring_id

        for _ in range(args.workers):
            _submit_next()

        while futures:
            for future in list(as_completed(futures, timeout=None)):
                ring_id = futures.pop(future)
                result = future.result()
                result["at"] = datetime.now(UTC).isoformat()
                out.write(json.dumps(result) + "\n")
                out.flush()

                if result["success"]:
                    succeeded += 1
                    cumulative_cost += result["cost_usd"]
                    mock_flag = "MOCK CITATIONS FOUND" if result["mock_citations_found"] else "ok"
                    print(
                        f"[{succeeded + failed}/{len(todo)}] {ring_id}: "
                        f"${result['cost_usd']:.3f} ({result['elapsed_s']}s, "
                        f"{result['grounds_count']} grounds, citations: {mock_flag}) "
                        f"-- cumulative ${cumulative_cost:.2f}"
                    )
                    if result["mock_citations_found"]:
                        print(f"    !! {result['mock_citations_found']}")
                else:
                    failed += 1
                    print(
                        f"[{succeeded + failed}/{len(todo)}] {ring_id}: FAILED "
                        f"-- {result['error']}"
                    )

                if cumulative_cost >= args.budget_usd and not budget_crossed:
                    budget_crossed = True
                    print(
                        f"\n!! Hard stop: cumulative cost ${cumulative_cost:.2f} "
                        f"crossed the ${args.budget_usd:.2f} budget. "
                        f"Not submitting further rings. "
                        f"{len(pending)} rings left unrun -- re-run this script "
                        "later to resume.\n"
                    )
                _submit_next()
                break  # re-evaluate futures dict fresh each completion

    elapsed_total = time.monotonic() - started
    print(
        f"\nDone. {succeeded} succeeded, {failed} failed, "
        f"${cumulative_cost:.2f} spent, {elapsed_total / 60:.1f} min wall time."
    )
    write_str_reports_parquet()
    return 0


#: Where infra/load_bigquery.py's SOURCES points the "str_reports" BigQuery
#: table at. restate() there casts whatever is written here to
#: shared.bq_schemas.STR_REPORTS_SCHEMA's exact Arrow types at load time, so
#: this only needs matching column names and castable dtypes, not the
#: contract's own types.
STR_REPORTS_PATH = Path("agents/out/str_reports.parquet")


def write_str_reports_parquet() -> None:
    """Project OUT_PATH's ledger onto shared.contracts.StrReport's shape.

    One row per ring: the latest success=True record, so a ring that
    failed once and succeeded on a later retry contributes only its
    successful draft, never the stale failure recorded alongside it.
    """
    latest: dict[str, dict[str, Any]] = {}
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("success"):
            latest[record["ring_id"]] = record

    if not latest:
        print(f"no successful reports in {OUT_PATH}; not writing {STR_REPORTS_PATH}")
        return

    df = pd.DataFrame(
        {
            "report_id": [r["report_id"] for r in latest.values()],
            "case_id": [r["case_id"] for r in latest.values()],
            "narrative": [r["narrative"] for r in latest.values()],
            "regulation_citations": [r["regulation_citations"] for r in latest.values()],
            "generated_at": pd.to_datetime([r["generated_at"] for r in latest.values()], utc=True),
            "status": [r["status"] for r in latest.values()],
            "citations_verified": [r["citations_verified"] for r in latest.values()],
        }
    )
    STR_REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STR_REPORTS_PATH, index=False)
    print(f"wrote {STR_REPORTS_PATH} ({len(df)} reports)")


if __name__ == "__main__":
    raise SystemExit(main())
