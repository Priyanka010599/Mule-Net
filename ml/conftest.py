"""Shared test helper for artefacts that are gitignored, not absent.

Several tests assert against files that ``ml/`` generates and
``.gitignore`` excludes: the out-of-fold score store under
``ml/experiments/early_detection/oof/``, the tables in ``ml/out/``, and
``metrics.json``. A fresh clone has the code but none of the evidence,
so those tests skip rather than fail — twenty-odd red tests on a clone
would say "this repo is broken" when the truth is "nothing has been
generated yet".

That default is wrong for CI, where a missing artefact means the
pipeline did not run and a skip is a silent pass. Setting
:data:`REQUIRE_ARTEFACTS_ENV` to ``1`` turns every one of those skips
into a failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Set to ``1`` in CI. Turns "artefact missing, skipping" into a failure.
REQUIRE_ARTEFACTS_ENV = "MULENET_REQUIRE_OOF"


def require_artefact(path: Path, regenerate_with: str) -> None:
    """Skip, or fail when :data:`REQUIRE_ARTEFACTS_ENV` is ``1``.

    Call before reading a generated artefact. Does nothing if ``path``
    exists, so the caller reads it immediately afterwards.
    """
    if path.exists():
        return
    message = f"{path} is missing; regenerate with `{regenerate_with}`."
    if os.environ.get(REQUIRE_ARTEFACTS_ENV) == "1":
        pytest.fail(message)
    pytest.skip(f"{message} (Set {REQUIRE_ARTEFACTS_ENV}=1 to make this a failure.)")
