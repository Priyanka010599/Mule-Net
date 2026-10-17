"""Environment-driven configuration for the Vertex AI training job.

Separate from :mod:`ml.config` on purpose: nothing in ``ml/`` outside
this subpackage talks to Google Cloud, and :mod:`ml.config` says so in
its own docstring. Keeping the cloud knobs here preserves that.

``project`` and ``location`` are always resolved here and always passed
explicitly to every client and every ``gcloud`` invocation. gcloud's
default project on a developer machine is not necessarily ``mulenet``,
and a call that omits the project acts on whatever that default is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The project every call is pinned to. Never inherited from gcloud config.
DEFAULT_PROJECT = "mulenet"

#: Vertex AI region. ``asia-south1`` because (a) Vertex AI custom jobs and
#: Model Registry are regional and the rest of MuleNet's Vertex usage --
#: Gemini -- is already there (``docs/residency.md``), (b) the
#: ``mulenet`` Artifact Registry repo exists there, and (c) the training
#: inputs are mirrored there, so the job reads **same-region** data and
#: there is no cross-region read at all.
DEFAULT_LOCATION = "asia-south1"

#: The asia-south1 mirror of the data bucket. Its six simulator parquet
#: files were verified md5-identical to ``simulator/out/`` before use;
#: reading the ``us-central1`` original instead would make the job a
#: cross-region reader for no benefit.
DEFAULT_BUCKET = "mulenet-mulenet-data-asia"

#: 4 vCPU because ``ml.evaluate.XGB_PARAMS`` pins ``n_jobs=4``. XGBoost's
#: ``hist`` tree method sums gradients per thread, so the thread count is
#: part of the numerical result: a machine with fewer usable cores than
#: ``n_jobs`` would not reproduce the local file.
DEFAULT_MACHINE_TYPE = "n1-standard-4"

#: GCS prefix holding the six simulator parquet tables the job reads.
INPUT_PREFIX = "simulator/out"

#: GCS prefix under which each run writes its own timestamped folder.
RUNS_PREFIX = "ml/vertex_runs"

#: Vertex AI Experiments experiment that every run logs into.
EXPERIMENT_NAME = "mulenet-early-detection"

#: Model Registry display name. Matches the ``score_model_version``
#: already recorded in ``ml/out/manifest.json``, so the registry entry
#: and the published scores name the same thing.
MODEL_DISPLAY_NAME = "mulenet-xgboost-cb-oof-v1"

#: Artifact Registry repo holding the training image.
AR_REPO = "mulenet"

#: Cloud Build runs as this service account. Without it the build fails
#: on its own source upload.
BUILD_SERVICE_ACCOUNT = (
    "projects/mulenet/serviceAccounts/mulenet-build@mulenet.iam.gserviceaccount.com"
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class VertexConfig:
    """Resolved Google Cloud settings for one training job."""

    project: str
    location: str
    bucket: str
    machine_type: str

    @classmethod
    def from_env(cls) -> VertexConfig:
        """Build from the environment, applying the documented defaults."""
        return cls(
            project=_env("MULENET_GCP_PROJECT", DEFAULT_PROJECT),
            location=_env("MULENET_VERTEX_LOCATION", DEFAULT_LOCATION),
            bucket=_env("MULENET_VERTEX_BUCKET", DEFAULT_BUCKET),
            machine_type=_env("MULENET_VERTEX_MACHINE_TYPE", DEFAULT_MACHINE_TYPE),
        )

    @property
    def input_uri(self) -> str:
        """GCS folder the job reads the six simulator tables from."""
        return f"gs://{self.bucket}/{INPUT_PREFIX}"

    def run_uri(self, run_id: str) -> str:
        """GCS folder one run writes every artefact into."""
        return f"gs://{self.bucket}/{RUNS_PREFIX}/{run_id}"

    @property
    def image_uri_base(self) -> str:
        """Artifact Registry path for the training image, without a tag."""
        return f"{self.location}-docker.pkg.dev/{self.project}/{AR_REPO}/mulenet-ml-train"
