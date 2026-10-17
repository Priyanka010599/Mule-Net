"""Environment-driven configuration for the MuleNet simulator.

Every knob the simulator needs at runtime is read from an environment
variable here (see CLAUDE.md: "Config via env vars in one config.py per
module"). Nothing in this module talks to Google Cloud; it only resolves
values so ``generator.py`` and ``replay.py`` stay free of ``os.environ``
lookups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Fallback HMAC key for local runs. Real runs must set ``HASH_KEY`` from
#: Secret Manager; this default exists so the repo is runnable offline and
#: never ships a plausible-looking production key.
DEFAULT_HASH_KEY = "demo-key"

#: Parquet/JPEG output lands next to the simulator code and is gitignored.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

#: Seed for the whole synthetic world. Same seed => byte-identical output.
DEFAULT_SEED = 42

#: Used when ``shared/pubsub_schema.py`` is not importable (see replay.py).
DEFAULT_TRANSFERS_TOPIC = "mulenet-transfers"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class SimulatorConfig:
    """Resolved simulator settings."""

    hash_key: bytes
    out_dir: Path
    seed: int
    gcp_project: str | None
    transfers_topic: str

    @classmethod
    def from_env(cls) -> SimulatorConfig:
        """Build a config from the process environment, applying defaults."""
        return cls(
            hash_key=os.environ.get("HASH_KEY", DEFAULT_HASH_KEY).encode("utf-8"),
            out_dir=_env_path("MULENET_OUT_DIR", DEFAULT_OUT_DIR),
            seed=_env_int("MULENET_SEED", DEFAULT_SEED),
            gcp_project=os.environ.get("MULENET_GCP_PROJECT") or None,
            transfers_topic=os.environ.get(
                "MULENET_TRANSFERS_TOPIC", DEFAULT_TRANSFERS_TOPIC
            ),
        )
