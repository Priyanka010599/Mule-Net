"""Environment-driven configuration for the MuleNet ``agents/`` module.

Every knob the tools, agents and eval harness need is resolved here (see
CLAUDE.md: "Config via env vars in one config.py per module"). Nothing in
this module reads data or talks to Google Cloud.

The one switch that matters is :attr:`AgentsConfig.mode`:

``mock``
    Fully offline. Agents return deterministic templated responses and the
    regulation search is served from a small built-in corpus. No
    credentials, no network. This is the default because the dev
    environment has no Google Cloud credentials.
``vertex``
    The real path: ADK ``LlmAgent`` instances on Gemini via Vertex AI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AGENTS_DIR.parent

#: Parquet tables written by ``simulator/generator.py``.
DEFAULT_DATA_DIR = _REPO_ROOT / "simulator" / "out"

#: Parquet tables written by ``ml/``. May not exist yet; see ``agents/data.py``.
DEFAULT_ML_OUT_DIR = _REPO_ROOT / "ml" / "out"

#: Directory holding ``router.md``, ``investigator.md`` and friends.
PROMPTS_DIR = _AGENTS_DIR / "prompts"

#: Gemini model used by every agent on the Vertex path.
#:
#: Flash, not Pro: every agent here either classifies a sentence or reads
#: a number back out of a tool result, and the HTTP service in
#: ``agents/service.py`` is driven by a public page's traffic, so the
#: per-token price is a security property rather than a preference.
#:
#: ``gemini-3.5-flash``, not ``gemini-2.5-flash``: the 2.5 family (Pro,
#: Flash, Flash-Lite) retires on Vertex AI 16-20 Oct 2026. 3.5-flash is
#: the model Google names as the direct successor, and does not retire
#: before 2027-05-19 -- see https://ai.google.dev/gemini-api/docs/deprecations.
#: Picked over the newer 3.6/3.7/3.8 tiers specifically because of where
#: it runs: see DEFAULT_LOCATION below.
DEFAULT_MODEL = "gemini-3.5-flash"

#: Vertex region for every Gemini call.
#:
#: ``asia-south1`` (Mumbai) rather than ``us-central1`` or the ``global``
#: endpoint, verified live with ``generate_content`` probes against this
#: project (not read off a docs page, which would not have caught this):
#:
#: ```
#: model                  asia-south1   global
#: gemini-3.5-flash             200       200
#: gemini-3.6-flash             404       200
#: gemini-3.7-flash             404       200
#: gemini-3.8-flash             404       200
#: gemini-3.5-flash-lite         404       200
#: ```
#:
#: Only 3.5-flash is offered region-pinned in Mumbai; every newer 3.x
#: tier is global-only today (re-probed 2026-10-03, after 2026-09-27's
#: first pass missed plain 3.5-flash and nearly led to an unnecessary
#: move to the global endpoint -- probe every candidate, not just the
#: newest-looking one). Region and model are still two separate
#: decisions, same as before: ``us-central1`` serves no Gemini 3.x at
#: all, so staying there was never an option regardless of which 3.x
#: model is chosen.
#:
#: An India BFSI entry is also better served by Mumbai than by Iowa on
#: its own merits. The data (GCS bucket, BigQuery dataset, both Cloud Run
#: services) has since moved to ``asia-south1`` too, in a parallel stack
#: alongside the original ``us-central1`` one kept as a fallback during
#: the soak period. See ``docs/residency.md`` and
#: ``infra/asia_south1_migration.md``.
DEFAULT_LOCATION = "asia-south1"

#: Valid values for ``MULENET_AGENT_MODE``.
MODES = frozenset({"mock", "vertex"})

#: Valid values for ``MULENET_AGENTS_DATA_SOURCE``. ``parquet`` is the
#: default and the only path the live service runs today -- see
#: ``agents/data.py``'s module docstring and ``docs/reviewer_notes.md``.
#: ``bigquery`` is additive: tested, not yet deployed.
DATA_SOURCES = frozenset({"parquet", "bigquery"})

#: BigQuery project and dataset ``infra/load_bigquery.py`` loads into.
#: Same defaults as ``ui/config.py``'s ``DEFAULT_BQ_PROJECT`` /
#: ``DEFAULT_BQ_DATASET`` -- ``gcloud``'s default project is deliberately
#: not consulted here either, for the same reason ``ui/config.py`` gives.
DEFAULT_BQ_PROJECT = "mulenet"
DEFAULT_BQ_DATASET = "mulenet"

#: Per-HTTP-call timeout for one Gemini request, in seconds.
#:
#: This bounds a single generate call, not a whole agent run: an agent
#: that calls tools makes several. A hung call must not hold an HTTP
#: request open, so this is always set.
DEFAULT_MODEL_TIMEOUT_S = 30.0

#: Wall-clock budget for one ``POST /chat`` request, in seconds.
#:
#: Larger than :data:`DEFAULT_MODEL_TIMEOUT_S` because a routed request
#: is two agent runs and each run is several calls. Whichever fires
#: first ends the request.
DEFAULT_REQUEST_TIMEOUT_S = 90.0

#: Ceiling on model calls inside one agent run.
#:
#: ADK's ``RunConfig`` default is 500. On a public URL that is an
#: unbounded bill per request, so it is always overridden.
DEFAULT_MAX_LLM_CALLS = 12

#: Ceiling on model calls for ``analyst_qa`` specifically.
#:
#: Lower than :data:`DEFAULT_MAX_LLM_CALLS` because a factual question is
#: a short tool loop: the baseline answers in about two calls. An agent
#: still circling after six is not converging, and the remaining calls
#: buy nothing but bill.
#:
#: Measured: ``gemini-3.5-flash`` burned all twelve on two eval questions
#: it should have declined, and those two runs were a large part of that
#: run's cost. The investigator and report agents keep the larger budget
#: -- they legitimately take more turns.
ANALYST_QA_MAX_LLM_CALLS = 6

#: Requests one session may make in :data:`DEFAULT_RATE_WINDOW_S`.
DEFAULT_SESSION_LIMIT = 5

#: Requests one client IP may make in :data:`DEFAULT_RATE_WINDOW_S`.
DEFAULT_IP_LIMIT = 10

#: Sliding window both per-session and per-IP limits are measured over.
DEFAULT_RATE_WINDOW_S = 60.0

#: Model-backed requests the whole service will serve in one UTC day.
#:
#: The session and IP limits are keyed on values the caller supplies or
#: can forge, so this is the only ceiling a hostile caller cannot move.
#: It is the number that bounds the bill.
DEFAULT_DAILY_CAP = 200

#: Longest chat message accepted, in characters. Input is billed too.
DEFAULT_MAX_MESSAGE_CHARS = 1000

#: Gemini tokens (prompt + output + thinking) the service will spend in one
#: UTC day. The request cap above counts questions; this counts what they
#: cost, so a few very expensive questions cannot slip under it.
DEFAULT_DAILY_TOKEN_CAP = 2_000_000


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive int, refusing values that would disable a limit.

    A limit set to ``0`` by a typo is worse than no limit at all,
    because the deployment looks protected. So the floor is enforced
    here rather than trusted.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    """Read a positive float, refusing values that would disable a timeout."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class AgentsConfig:
    """Resolved settings for the agent system."""

    mode: str
    data_dir: Path
    ml_out_dir: Path
    model: str
    project: str | None
    location: str
    use_vertex: bool
    #: Per-Gemini-call HTTP timeout, seconds.
    model_timeout_s: float = DEFAULT_MODEL_TIMEOUT_S
    #: Whole-request wall-clock budget for the HTTP service, seconds.
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    #: Ceiling on model calls in one agent run.
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    #: Whether missing detection tables must stop the service from
    #: serving at all. True by default: a deployment that answers from
    #: mock tables is worse than one that refuses to start, because the
    #: wrong answers carry real provenance.
    require_data: bool = True
    #: Global cap on generated tokens per call. ``None`` leaves each
    #: agent on its own budget in ``agents.adk_agents.MAX_OUTPUT_TOKENS``;
    #: a value caps every agent at no more than this.
    max_output_tokens: int | None = None
    #: Where ``agents.data.DataStore`` loads ``simulator/``- and
    #: ``ml/``-published tables from. ``"parquet"`` (default) is the
    #: unchanged local-file path; ``"bigquery"`` queries
    #: ``bq_project.bq_dataset`` instead, once at warm-up. ``cases`` is
    #: never affected either way -- see ``agents.data.BIGQUERY_BACKED_TABLES``.
    data_source: str = "parquet"
    #: BigQuery project/dataset used only when :attr:`data_source` is
    #: ``"bigquery"``.
    bq_project: str = DEFAULT_BQ_PROJECT
    bq_dataset: str = DEFAULT_BQ_DATASET

    def max_llm_calls_for(self, agent: str) -> int:
        """The call ceiling for one named agent.

        ``analyst_qa`` gets a smaller budget than the rest; see
        :data:`ANALYST_QA_MAX_LLM_CALLS`. ``min`` rather than a plain
        lookup so that lowering ``MULENET_AGENT_MAX_LLM_CALLS`` still
        lowers every agent -- an operator tightening the global ceiling
        should not find one agent quietly exempt from it.
        """
        if agent == "analyst_qa":
            return min(ANALYST_QA_MAX_LLM_CALLS, self.max_llm_calls)
        return self.max_llm_calls

    @classmethod
    def from_env(cls) -> AgentsConfig:
        """Build a config from the process environment, applying defaults."""
        mode = os.environ.get("MULENET_AGENT_MODE", "mock").strip().lower()
        if mode not in MODES:
            raise ValueError(
                f"MULENET_AGENT_MODE must be one of {sorted(MODES)}, got {mode!r}"
            )
        data_source = (
            os.environ.get("MULENET_AGENTS_DATA_SOURCE", "parquet").strip().lower()
        )
        if data_source not in DATA_SOURCES:
            raise ValueError(
                f"MULENET_AGENTS_DATA_SOURCE must be one of {sorted(DATA_SOURCES)}, "
                f"got {data_source!r}"
            )
        override = os.environ.get("MULENET_AGENT_MAX_OUTPUT_TOKENS")
        return cls(
            mode=mode,
            data_dir=_env_path("MULENET_DATA_DIR", DEFAULT_DATA_DIR),
            ml_out_dir=_env_path("MULENET_ML_OUT_DIR", DEFAULT_ML_OUT_DIR),
            model=os.environ.get("MULENET_GEMINI_MODEL", DEFAULT_MODEL),
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
            use_vertex=_env_flag("GOOGLE_GENAI_USE_VERTEXAI", True),
            model_timeout_s=_env_float(
                "MULENET_AGENT_MODEL_TIMEOUT_SECONDS", DEFAULT_MODEL_TIMEOUT_S
            ),
            request_timeout_s=_env_float(
                "MULENET_AGENT_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_S
            ),
            max_llm_calls=_env_int(
                "MULENET_AGENT_MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS
            ),
            require_data=_env_flag("MULENET_AGENT_REQUIRE_DATA", True),
            max_output_tokens=(
                _env_int("MULENET_AGENT_MAX_OUTPUT_TOKENS", 0, minimum=1)
                if override and override.strip()
                else None
            ),
            data_source=data_source,
            bq_project=os.environ.get("MULENET_BQ_PROJECT", DEFAULT_BQ_PROJECT),
            bq_dataset=os.environ.get("MULENET_BQ_DATASET", DEFAULT_BQ_DATASET),
        )

    @property
    def is_mock(self) -> bool:
        """True when agents must run fully offline."""
        return self.mode == "mock"

    @property
    def models_dir(self) -> Path:
        """Where Model A/B's booster + metadata.json files live.

        Under ``ml_out_dir``, same as every other ML-pipeline artifact --
        fetched from Cloud Storage at container start the same way the
        published tables are (``infra/fetch_data.py``, ``MULENET_FETCH``),
        never baked into the image. See ``ml/model_runtime.py``.
        """
        return self.ml_out_dir / "models"

    @property
    def allow_mock_tables(self) -> bool:
        """Whether a missing parquet may be replaced by a built-in mock.

        Only in ``mock`` mode. Tied to :attr:`mode` rather than given its
        own env var so there is one switch and no configuration in which
        a deployed service answers over invented numbers: the Vertex path
        is the deployed one, and there a missing table is a hard error in
        :mod:`agents.data`. See ``docs/agents_api.md`` on why -- the reply
        would still report ``backend: "gemini"`` and real-looking
        ``sources``, so the wrong answer arrives with authentic
        provenance.
        """
        return self.is_mock

    def prompt_path(self, name: str) -> Path:
        """Return the path to ``agents/prompts/<name>.md``."""
        return PROMPTS_DIR / f"{name}.md"


def load_prompt(name: str) -> str:
    """Read one agent prompt from ``agents/prompts/``.

    Prompts live on disk rather than in string literals so they can be
    reviewed and diffed independently of the code (see the agent brief).
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file at {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ServiceLimits:
    """Cost ceilings for the public HTTP service in ``agents/service.py``.

    Cloud Run auth bounds who may call the service; it does not bound
    what they spend, and the one authorised caller is a public page with
    no sign-in. So every default here has to be safe against a caller
    that is not friendly. They are deliberately low: a judge asking a
    handful of questions never notices them, and a script driving the UI
    stops within seconds.

    Two of the three keys are caller-supplied. ``session_limit`` is keyed
    on a session id the client chooses and ``ip_limit`` on
    ``X-Forwarded-For``, which a client can pad. Both are therefore
    friction, not enforcement. :attr:`daily_cap` is counted on the server
    against nothing the caller controls, and it is the only number that
    actually bounds the bill.
    """

    session_limit: int = DEFAULT_SESSION_LIMIT
    ip_limit: int = DEFAULT_IP_LIMIT
    window_s: float = DEFAULT_RATE_WINDOW_S
    daily_cap: int = DEFAULT_DAILY_CAP
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS
    daily_token_cap: int = DEFAULT_DAILY_TOKEN_CAP

    @classmethod
    def from_env(cls) -> ServiceLimits:
        """Build limits from the process environment, applying defaults."""
        return cls(
            session_limit=_env_int(
                "MULENET_AGENT_SESSION_LIMIT", DEFAULT_SESSION_LIMIT
            ),
            ip_limit=_env_int("MULENET_AGENT_IP_LIMIT", DEFAULT_IP_LIMIT),
            window_s=_env_float(
                "MULENET_AGENT_RATE_WINDOW_SECONDS", DEFAULT_RATE_WINDOW_S, minimum=1.0
            ),
            daily_cap=_env_int("MULENET_AGENT_DAILY_CAP", DEFAULT_DAILY_CAP),
            max_message_chars=_env_int(
                "MULENET_AGENT_MAX_MESSAGE_CHARS", DEFAULT_MAX_MESSAGE_CHARS
            ),
            daily_token_cap=_env_int(
                "MULENET_AGENT_DAILY_TOKEN_CAP", DEFAULT_DAILY_TOKEN_CAP
            ),
        )
