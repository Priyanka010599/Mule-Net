"""``mulenet-agents`` must deploy with exactly ``--max-instances=1``.

``agents/limits.py`` keeps the per-session limit, the per-IP limit and the
daily spend cap in process memory, so every extra Cloud Run instance multiplies
the advertised cap (see the comment above the agents deploy in
``infra/deploy.sh``). This reads the deploy scripts as text, so a change to the
flag fails CI instead of surfacing as a bill.
"""

from __future__ import annotations

import re
from pathlib import Path

INFRA = Path(__file__).resolve().parent
_AGENTS_SERVICE = re.compile(r'^\s*gcloud run deploy\s+"?(\$\{SVC_AGENTS\}|mulenet-agents)"?')


def _agents_deploy_blocks() -> list[tuple[str, str]]:
    """Every ``gcloud run deploy`` of the agents service: (script name, command text)."""
    blocks: list[tuple[str, str]] = []
    for script in sorted(INFRA.glob("*.sh")):
        lines = script.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if _AGENTS_SERVICE.match(line):
                command = [line]
                while command[-1].rstrip().endswith("\\") and i + len(command) < len(lines):
                    command.append(lines[i + len(command)])
                blocks.append((script.name, "\n".join(command)))
    return blocks


def test_the_agents_deploy_block_is_found() -> None:
    """If the block is renamed or moved, this guard must fail rather than go quiet."""
    assert any(name == "deploy.sh" for name, _ in _agents_deploy_blocks())


def test_every_agents_deploy_passes_exactly_max_instances_1() -> None:
    for name, command in _agents_deploy_blocks():
        flags = re.findall(r"--max-instances[= ]\s*(\S+?)\s*(?:\\|$)", command, flags=re.M)
        assert flags == ["1"], (
            f"{name}: the agents service must deploy with exactly --max-instances=1 "
            f"(found {flags}); its rate limits are per-instance, see agents/limits.py"
        )
