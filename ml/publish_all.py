"""The only supported way to regenerate ``ml/out/``.

``ml/out/manifest.json`` has two writers. ``ml.publish`` builds the five
contract-validated tables and writes the manifest **wholesale**;
``ml.timeline`` adds ``ring_detection_timeline`` to it by
read-modify-write. Run in that order the manifest ends up describing all
six tables. Run in the other order — or run ``ml.publish`` alone, which
is the easy mistake — and the timeline entry is silently dropped while
the parquet file it describes stays on disk, so ``ml/out/`` claims five
tables and contains six.

That is a data-integrity bug with no loud failure, which is exactly the
kind this repo tries not to ship. So the two entry points are not to be
invoked individually any more:

    python -m ml.publish_all

is the supported command, and :data:`STEPS` is the order. The
individual ``main()`` functions remain importable and are still what
does the work; this module only fixes the sequence.

The real fix is one module owning the manifest, with both writers
feeding it their table entries instead of taking turns editing a file.
That is a larger change than the ordering bug warrants before
submission and is logged in ``docs/post_submission.md``.
"""

from __future__ import annotations

import argparse
from types import ModuleType

from ml import bank_queues, publish, ring_map, timeline

#: Publishing steps in the only order that leaves a consistent manifest.
#: ``ml.publish`` first: it truncates the manifest, so anything that
#: appends to it has to follow. ``ml.ring_map`` additionally *reads*
#: ``rings.parquet``, so it cannot precede the step that writes it.
#:
#: Modules rather than bound ``main`` functions, so the call is looked up
#: at call time. A tuple of function objects would capture whatever
#: ``main`` was at import, which makes the ordering untestable without
#: reaching into this tuple -- and the ordering is the only thing this
#: module exists to guarantee.
STEPS: tuple[ModuleType, ...] = (publish, timeline, ring_map, bank_queues)

#: Steps that only append their own table to the manifest and may be run
#: alone (``--only``) without rewriting any other table. ``ml.publish`` is
#: not one of them: it rewrites all five contract tables (new
#: ``generated_at``) and truncates the manifest.
ADDITIVE_STEPS: dict[str, ModuleType] = {"bank_alert_queues": bank_queues}


def main(only: str | None = None) -> None:
    """Run every publishing step in order, or one additive step (``only``)."""
    if only:
        ADDITIVE_STEPS[only].main()
        return
    for i, step in enumerate(STEPS, start=1):
        print(f"\n=== [{i}/{len(STEPS)}] {step.__name__} ===")
        step.main()
    print(f"\nregenerated ml/out/ via: {' -> '.join(s.__name__ for s in STEPS)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate ml/out/.")
    parser.add_argument("--only", choices=sorted(ADDITIVE_STEPS), help="run just this table")
    main(parser.parse_args().only)
