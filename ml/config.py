"""Environment-driven configuration for the MuleNet ``ml/`` module.

Every path and knob ``features.py``, ``baseline.py`` and ``compare.py``
need is resolved here (see CLAUDE.md: "Config via env vars in one
config.py per module"). Nothing in this module reads data or talks to
Google Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent

#: Parquet input written by ``simulator/generator.py``.
DEFAULT_DATA_DIR = _ML_DIR.parent / "simulator" / "out"

#: Where ``features.parquet`` and ``experiments/`` land. Gitignored.
DEFAULT_ML_DIR = _ML_DIR

#: Maximum idle gap inside one pass-through burst. Two consecutive
#: transfers on the same account more than this far apart start a new
#: burst. 30 minutes is the top of the simulator's 2-30 minute
#: mule pass-through window, so a genuine hop never gets split in two.
DEFAULT_BURST_GAP_SECONDS = 1800

#: Seed for the ring-aware train/val/test split and the first model run.
DEFAULT_SEED = 42

#: BigQuery project and dataset holding the loaded `transfers` table.
#: Same env var names ``infra/load_bigquery.py`` uses, so one export
#: configures both. gcloud's default project is *not* mulenet, so the
#: project is always passed explicitly rather than inherited.
DEFAULT_BQ_PROJECT = "mulenet"
DEFAULT_BQ_DATASET = "mulenet"

#: Dataset table name -> parquet file stem.
TABLE_FILES: dict[str, str] = {
    "accounts": "accounts.parquet",
    "transfers": "transfers.parquet",
    "devices": "devices.parquet",
    "kyc_submissions": "kyc_submissions.parquet",
    "kyc_documents": "kyc_documents.parquet",
    "ground_truth": "ground_truth.parquet",
}


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
class MlConfig:
    """Resolved settings for feature building and model training."""

    data_dir: Path
    ml_dir: Path
    burst_gap_seconds: int
    seed: int
    bq_project: str
    bq_dataset: str

    @classmethod
    def from_env(cls) -> MlConfig:
        """Build a config from the process environment, applying defaults."""
        return cls(
            data_dir=_env_path("MULENET_DATA_DIR", DEFAULT_DATA_DIR),
            ml_dir=_env_path("MULENET_ML_DIR", DEFAULT_ML_DIR),
            burst_gap_seconds=_env_int(
                "MULENET_BURST_GAP_SECONDS", DEFAULT_BURST_GAP_SECONDS
            ),
            seed=_env_int("MULENET_SEED", DEFAULT_SEED),
            bq_project=os.environ.get("PROJECT_ID") or DEFAULT_BQ_PROJECT,
            bq_dataset=os.environ.get("BQ_DATASET") or DEFAULT_BQ_DATASET,
        )

    def table_path(self, table: str) -> Path:
        """Return the parquet path for one of the six simulator tables."""
        try:
            filename = TABLE_FILES[table]
        except KeyError as exc:
            raise KeyError(f"unknown table {table!r}; known: {sorted(TABLE_FILES)}") from exc
        return self.data_dir / filename

    @property
    def features_path(self) -> Path:
        """Output of ``features.py``, input to ``baseline.py``."""
        return self.ml_dir / "features.parquet"

    @property
    def experiments_dir(self) -> Path:
        """Root that ``compare.py`` globs for ``metrics.json`` files."""
        return self.ml_dir / "experiments"
