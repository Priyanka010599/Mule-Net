"""Environment-driven configuration for the MuleNet ``ui/`` module.

Every path, port and feature switch the FastAPI app needs is resolved
here (see CLAUDE.md: "Config via env vars in one config.py per
module"). Nothing in this module reads data or talks to Google Cloud.

Data sources
------------
The UI reads three things, and each has a *configured* location so a
real backend can be swapped in without touching the views:

``snapshot_path``
    A pre-built JSON bundle of accounts, transfers, rings and evidence
    (see ``ui/build_snapshot.py``). Today it is built offline from
    ``simulator/out/``; later it can be regenerated from ``ml/out/``
    or served from AlloyDB behind the same shape.

``metrics_path``
    ``ml/``'s early-detection metrics JSON. Read live when present so
    the metrics page tracks the real experiment; falls back to the
    bundled copy in ``ui/data/`` when ``ml/`` has not run.

``chat_backend``
    ``"mock"`` for the offline keyword router; ``"agents"`` calls the
    ``agents/`` Cloud Run service over HTTP (docs/agents_api.md).

``read_mode``
    ``"snapshot"`` (the default) or ``"bigquery"``. In ``bigquery`` mode
    the RING INDEX is derived by one query against the BigQuery dataset
    (see ``ui/bqrings.py``) and the snapshot remains the fallback. It
    changes nothing else: the metrics page has no BigQuery source,
    because the dataset has no metrics table.

The agents service
------------------
``agents_url`` and ``agents_audience`` are server-side only. The agents
service is deployed **without public invoker access**, so the browser
cannot call it and must never be told where it is: the browser talks to
``ui``, and ``ui`` talks to ``agents`` with a Cloud Run service-to-service
ID token. Nothing here is ever rendered into a template or an API
response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shared import ring_race

_UI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _UI_DIR.parent

#: Bundled data that ships with the app (snapshot + metrics fallback).
DEFAULT_UI_DATA_DIR = _UI_DIR / "data"

#: Parquet input written by ``simulator/generator.py``. Read-only, and
#: only by ``build_snapshot.py`` — never by a request handler.
DEFAULT_RAW_DATA_DIR = _REPO_ROOT / "simulator" / "out"

#: ``ml/``'s published output tables. Read-only: the ml-engineer owns
#: this folder. Holds rings.parquet, scores.parquet,
#: ring_explanations.parquet and account_evidence.parquet.
DEFAULT_ML_OUT_DIR = _REPO_ROOT / "ml" / "out"

#: ``agents/``'s published output. Read-only: the agents-engineer owns
#: this folder. Holds ``cases.parquet``, one row per ring whose
#: investigation passed verification. A ring whose narrative was
#: withheld has **no row here at all**, and the UI must render that
#: absence as "no case yet" rather than inventing a placeholder.
DEFAULT_AGENTS_OUT_DIR = _REPO_ROOT / "agents" / "out"

#: ``ml/``'s early-detection experiment output. Read-only.
DEFAULT_METRICS_PATH = _REPO_ROOT / "ml" / "experiments" / "early_detection" / "metrics.json"

#: Shipped copy of the above, used when ``ml/`` has not been run.
DEFAULT_METRICS_FALLBACK = DEFAULT_UI_DATA_DIR / "metrics_fallback.json"

#: ``ml/``'s lead-time experiment output. Read-only, and the ONLY source
#: for ring coverage: it publishes ``share_with_positive_warning``, the
#: share of rings flagged at a measured horizon *before the money reached
#: the exit account*. The looser "flagged at all" rate is derivable from
#: the timeline parquet and must not be, because it hides the span-1
#: control result that makes the whole comparison credible.
#: ``docs/claims.md`` cites these same paths, so the page and the claims
#: file resolve one value from one place.
DEFAULT_LEAD_METRICS_PATH = _REPO_ROOT / "ml" / "experiments" / "lead_time" / "metrics.json"

#: Shipped copy of the above, used when ``ml/`` has not been run.
DEFAULT_LEAD_METRICS_FALLBACK = DEFAULT_UI_DATA_DIR / "lead_time_fallback.json"

#: ``ml/``'s coalition sweep: ring recall as a function of how many banks
#: joined, for every subset of banks. Read-only. This is what the
#: "one more bank joins" control on the metrics tab reads; every point in
#: it is measured by ``ml/coalition.py`` before the page is ever served,
#: so the control switches between precomputed numbers and recomputes
#: nothing. See ``docs/data_sharing.md``.
DEFAULT_COALITION_PATH = _REPO_ROOT / "ml" / "experiments" / "coalition" / "metrics.json"

#: Shipped copy of the above, used when ``ml/`` has not been run.
DEFAULT_COALITION_FALLBACK = DEFAULT_UI_DATA_DIR / "coalition_fallback.json"

#: ``ml/redteam.py``'s live-Gemini evasion run: strategies the model
#: designed against the detector's own published thresholds, each executed
#: through the real detection chain. Read-only, and expensive to
#: regenerate -- it is the one script in this repo with no rate limit on
#: the model call, so unlike the other experiments here it is never
#: recomputed by anything this app runs, not even offline tooling.
DEFAULT_REDTEAM_PATH = _REPO_ROOT / "ml" / "experiments" / "redteam" / "metrics.json"

#: Shipped copy of the above, used when ``ml/`` has not been run.
DEFAULT_REDTEAM_FALLBACK = DEFAULT_UI_DATA_DIR / "redteam_fallback.json"

#: One real, credentialed Confidential Space run's evidence (docs/enclave.md,
#: Stage 3), captured live against Google Cloud -- proof that the cross-bank
#: join happened under attestation and that each bank's outbox alert is
#: encrypted to a key only that bank holds (enclave/outbox_crypto.py). Unlike
#: the pairs above, there is no ``ml/`` (or any) script in this repo that
#: regenerates this file: doing so needs a real GCP project, KMS and a
#: Confidential VM, none of which this container can reach. So there is no
#: "live" counterpart to fall back from -- this committed file is the only
#: copy, reviewed once and served as-is.
DEFAULT_ENCLAVE_EVIDENCE_PATH = DEFAULT_UI_DATA_DIR / "enclave_evidence.json"

#: Pre-built demo bundle served to the browser. Lands under ``out/``,
#: which ``.gitignore`` excludes: it is derived transaction data and
#: this is a public repo. Build it with ``python -m ui.build_snapshot``.
DEFAULT_SNAPSHOT_PATH = _UI_DIR / "out" / "demo_snapshot.json"

#: Chat backends the app knows how to talk to.
CHAT_BACKENDS: frozenset[str] = frozenset({"mock", "agents"})

#: Where the RING INDEX is read from. ``snapshot`` is the default and
#: behaves exactly as it always has; ``bigquery`` derives the ring index
#: with one query over ``rings``/``transfers``/``account_evidence``/
#: ``cases`` and falls back to the snapshot, visibly, on any failure.
#: It affects the ring index ONLY — ring detail, bank tiles, replay and
#: the metrics page are unchanged, and the metrics page in particular has
#: no BigQuery source to read: there is no metrics table in the dataset.
READ_MODES: frozenset[str] = frozenset({"snapshot", "bigquery"})

#: BigQuery project and dataset ``infra/load_bigquery.py`` loads into.
#: ``gcloud``'s default project is deliberately not consulted: it is not
#: this project, and inheriting it would query somebody else's dataset.
DEFAULT_BQ_PROJECT = "mulenet"
DEFAULT_BQ_DATASET = "mulenet"

#: Wall-clock budget for the ring-index query. Past it the snapshot
#: answers, labelled. BigQuery has a seconds-scale latency floor against
#: an effectively instant file, so this is a ceiling on a known
#: regression, not a guard against a rare one.
DEFAULT_BQ_TIMEOUT_S = 20.0

#: Wall-clock budget for one ``POST /chat`` against ``agents/``. Past it
#: the UI degrades to the mock rather than holding the browser open: a
#: judge waiting on a spinner learns nothing, a labelled fallback does.
DEFAULT_AGENTS_TIMEOUT_S = 30.0

#: Horizons the metrics page headlines. The full-window numbers are
#: ceiling-bound (PR-AUC ~0.999 cross-bank) and are shown only as a
#: footnote, never as the claim.
DEFAULT_HEADLINE_HORIZONS: tuple[int, ...] = (5, 10, 30)

#: Hard cap on nodes handed to Cytoscape in one graph payload. 30,000
#: accounts cannot be laid out interactively; the joint view renders a
#: ring subgraph plus bounded context instead.
DEFAULT_MAX_GRAPH_NODES = 90

#: Counterparty hops out from ring members included in that subgraph.
DEFAULT_GRAPH_HOPS = 1


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class UiConfig:
    """Resolved settings for the FastAPI app and the snapshot builder."""

    snapshot_path: Path
    raw_data_dir: Path
    ml_out_dir: Path
    agents_out_dir: Path
    metrics_path: Path
    metrics_fallback_path: Path
    lead_metrics_path: Path
    lead_metrics_fallback_path: Path
    coalition_path: Path
    coalition_fallback_path: Path
    redteam_path: Path
    redteam_fallback_path: Path
    enclave_evidence_path: Path
    chat_backend: str
    #: ``snapshot`` (default) or ``bigquery``. Ring index only.
    read_mode: str
    bq_project: str
    bq_dataset: str
    bq_timeout_s: float
    #: Base URL of the ``agents/`` Cloud Run service. Server-side only:
    #: never rendered into a page or returned by an endpoint.
    agents_url: str
    #: Audience the ID token is minted for. Defaults to ``agents_url``,
    #: which is what Cloud Run expects; separately configurable only
    #: because a load balancer in front changes the audience.
    agents_audience: str
    agents_timeout_s: float
    #: Mint a real Google ID token. Off only where there is no metadata
    #: server and no ADC -- a local fake agents service, or a test.
    agents_auth: bool
    #: Shows and serves the "Try tampered code" demo. OFF until the tampered
    #: image's refusal has been verified live -- see enclave/live_proof.py.
    live_proof_tamper_enabled: bool
    max_graph_nodes: int
    graph_hops: int
    headline_horizons: tuple[int, ...]
    #: Per-client ceilings on the public edge (ui/ratelimit.py), per instance.
    #: ``/api/simulate`` is cheap, so its limit is loose; the live proof
    #: launches a paid VM, so its limits are tight (item 1).
    simulate_rate_per_minute: int = 30
    subset_rate_per_minute: int = 20
    #: ``/api/chat`` per client per minute (each may cost a Gemini call).
    chat_rate_per_minute: int = 10
    live_proof_rate_per_ip_per_hour: int = 3
    live_proof_rate_global_per_hour: int = 6
    live_proof_cooldown_s: int = 60
    #: What ``/api/banks`` does when the caller does not say. True keeps the
    #: simulator labels in the default payload (the page hides them client-side
    #: today); False makes ``?reveal_labels=1`` the only way to get them.
    labels_default_revealed: bool = True
    #: Origins allowed to call the API cross-origin. Empty (the default) means
    #: same-origin only: the page is served by this app, so nothing else needs in.
    cors_allow_origins: tuple[str, ...] = ()
    #: Largest body accepted on a POST other than ``/api/subset/upload`` (which
    #: has its own, larger cap).
    max_body_bytes: int = 65536

    @classmethod
    def from_env(cls) -> UiConfig:
        """Build a config from the process environment, applying defaults."""
        chat_backend = _env_str("MULENET_UI_CHAT_BACKEND", "mock")
        if chat_backend not in CHAT_BACKENDS:
            raise ValueError(
                f"MULENET_UI_CHAT_BACKEND must be one of {sorted(CHAT_BACKENDS)}, "
                f"got {chat_backend!r}"
            )
        read_mode = _env_str("MULENET_READ_MODE", "snapshot")
        if read_mode not in READ_MODES:
            # A typo must not silently mean "snapshot": an operator who
            # set MULENET_READ_MODE=bq would otherwise demo the file
            # while believing BigQuery was on the read path.
            raise ValueError(
                f"MULENET_READ_MODE must be one of {sorted(READ_MODES)}, "
                f"got {read_mode!r}"
            )
        agents_url = _env_str("MULENET_UI_AGENTS_URL", "").rstrip("/")
        if chat_backend == "agents" and not agents_url:
            # A misconfiguration, not a degradation: there is nothing to
            # fall back *from*. Fail at startup rather than answer every
            # question from the mock under an "agents" label.
            raise ValueError(
                "MULENET_UI_CHAT_BACKEND=agents requires MULENET_UI_AGENTS_URL"
            )
        return cls(
            snapshot_path=_env_path("MULENET_UI_SNAPSHOT", DEFAULT_SNAPSHOT_PATH),
            raw_data_dir=_env_path("MULENET_DATA_DIR", DEFAULT_RAW_DATA_DIR),
            ml_out_dir=_env_path("MULENET_UI_ML_OUT", DEFAULT_ML_OUT_DIR),
            agents_out_dir=_env_path("MULENET_UI_AGENTS_OUT", DEFAULT_AGENTS_OUT_DIR),
            metrics_path=_env_path("MULENET_UI_METRICS", DEFAULT_METRICS_PATH),
            metrics_fallback_path=_env_path(
                "MULENET_UI_METRICS_FALLBACK", DEFAULT_METRICS_FALLBACK
            ),
            lead_metrics_path=_env_path(
                "MULENET_UI_LEAD_METRICS", DEFAULT_LEAD_METRICS_PATH
            ),
            coalition_path=_env_path("MULENET_UI_COALITION", DEFAULT_COALITION_PATH),
            coalition_fallback_path=_env_path(
                "MULENET_UI_COALITION_FALLBACK", DEFAULT_COALITION_FALLBACK
            ),
            redteam_path=_env_path("MULENET_UI_REDTEAM", DEFAULT_REDTEAM_PATH),
            redteam_fallback_path=_env_path(
                "MULENET_UI_REDTEAM_FALLBACK", DEFAULT_REDTEAM_FALLBACK
            ),
            enclave_evidence_path=_env_path(
                "MULENET_UI_ENCLAVE_EVIDENCE", DEFAULT_ENCLAVE_EVIDENCE_PATH
            ),
            lead_metrics_fallback_path=_env_path(
                "MULENET_UI_LEAD_METRICS_FALLBACK", DEFAULT_LEAD_METRICS_FALLBACK
            ),
            chat_backend=chat_backend,
            read_mode=read_mode,
            bq_project=_env_str("MULENET_BQ_PROJECT", DEFAULT_BQ_PROJECT),
            bq_dataset=_env_str("MULENET_BQ_DATASET", DEFAULT_BQ_DATASET),
            bq_timeout_s=_env_float("MULENET_BQ_TIMEOUT_S", DEFAULT_BQ_TIMEOUT_S),
            agents_url=agents_url,
            agents_audience=_env_str("MULENET_UI_AGENTS_AUDIENCE", agents_url),
            agents_timeout_s=_env_float(
                "MULENET_UI_AGENTS_TIMEOUT_S", DEFAULT_AGENTS_TIMEOUT_S
            ),
            agents_auth=_env_bool("MULENET_UI_AGENTS_AUTH", True),
            live_proof_tamper_enabled=_env_bool("MULENET_UI_LIVE_PROOF_TAMPER", False),
            max_graph_nodes=_env_int("MULENET_UI_MAX_GRAPH_NODES", DEFAULT_MAX_GRAPH_NODES),
            graph_hops=_env_int("MULENET_UI_GRAPH_HOPS", DEFAULT_GRAPH_HOPS),
            headline_horizons=DEFAULT_HEADLINE_HORIZONS,
            simulate_rate_per_minute=_env_int("MULENET_UI_SIMULATE_RATE_PER_MIN", 30),
            subset_rate_per_minute=_env_int("MULENET_UI_SUBSET_RATE_PER_MIN", 20),
            chat_rate_per_minute=_env_int("MULENET_UI_CHAT_RATE_PER_MIN", 10),
            live_proof_rate_per_ip_per_hour=_env_int("MULENET_UI_LIVE_PROOF_IP_PER_HOUR", 3),
            live_proof_rate_global_per_hour=_env_int("MULENET_UI_LIVE_PROOF_GLOBAL_PER_HOUR", 6),
            live_proof_cooldown_s=_env_int("MULENET_UI_LIVE_PROOF_COOLDOWN_S", 60),
            labels_default_revealed=_env_bool("MULENET_UI_LABELS_DEFAULT_REVEALED", True),
            cors_allow_origins=tuple(
                o.strip() for o in _env_str("MULENET_UI_CORS_ORIGINS", "").split(",") if o.strip()
            ),
            max_body_bytes=_env_int("MULENET_UI_MAX_BODY_BYTES", 65536),
        )

    @property
    def static_dir(self) -> Path:
        """Directory served at ``/static``."""
        return _UI_DIR / "static"

    @property
    def templates_dir(self) -> Path:
        """Directory holding the single-page shell."""
        return _UI_DIR / "templates"

    def raw_table_path(self, table: str) -> Path:
        """Parquet path for one simulator table, used only at build time."""
        return self.raw_data_dir / f"{table}.parquet"

    def ml_table_path(self, table: str) -> Path:
        """Parquet path for one published ``ml/out/`` table, build time only."""
        return self.ml_out_dir / f"{table}.parquet"

    def agents_table_path(self, table: str) -> Path:
        """Parquet path for one published ``agents/out/`` table.

        May legitimately not exist: ``agents/`` publishes cases only for
        investigations that passed verification, and on a machine where
        the agents have never run there is no file at all. Callers treat
        absence as "no case yet", never as an error.
        """
        return self.agents_out_dir / f"{table}.parquet"

    @property
    def ring_race_path(self) -> Path:
        """Where the label-free race artifact for the agents service is written.

        ``agents/out/`` because that is the directory the agents service
        already reads and the deploy already mirrors to its bucket. See
        ``shared/ring_race.py``.
        """
        return self.agents_out_dir / ring_race.ARTIFACT_NAME
