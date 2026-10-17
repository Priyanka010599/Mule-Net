"""Live "load a subset, score it live" panel -- a bounded slice of the
**real** published data, run through the real detector, per request.

Backs ``GET /subset/options``, ``POST /subset/load`` and ``POST
/subset/score`` in ``agents/service.py``. This is the second live,
no-model-call feature in ``agents/`` (the first is
``agents/simulate.py``'s ``POST /simulate``), and it is a different kind
of "live" from that one: ``/simulate`` synthesises a tiny **synthetic**
graph from a judge's numbers; this reads a bounded slice of the **actual
572,890-transfer, 30,000-account dataset** the rest of the app is built
from, and scores it with the unmodified :func:`ml.features.build_features`
and :func:`ml.rule.velocity_decision`. The feature build for each horizon is
cached per ``(data version, horizon)`` (:func:`prepared_for`) because it is the
expensive part and depends on nothing a caller chooses beyond the horizon; the
rule and the model are re-run, and the bank filter re-applied, on every
request. Responses say ``cache: hit|miss`` and what the build cost.

What "subset" means here, and why
----------------------------------
``ml/early.py`` is this repo's whole headline result: scoring every
account from only what its bank could see **by a cutoff instant**, at a
grid of horizons (``ml.early.CUTOFF_MINUTES``). That grid is precomputed
and lives behind ``GET /api/metrics``. This panel exposes the *same
concept* -- a time cutoff that bounds what the feature builder is allowed
to see -- as something a caller picks and gets scored on the spot, rather
than reading a number someone else already computed.

Two differences from ``ml.early``, both deliberate simplifications, not
oversights:

* ``ml.early`` cuts each account off relative to *its own ring's onset*
  (or a resampled reference time for a non-ring account), which is the
  right design for a fair recall comparison but needs ``ground_truth`` --
  the very labels a live detector would not have. This panel instead uses
  one **absolute** calendar cutoff (`` :data:`T0` + horizon``, the same
  instant for every account), which reads no label at all: the flags
  below come from exactly the code path a live system would run,
  unlabelled data in, a decision out. The published metrics grid remains
  the citable, ring-fair number; this panel is honest about being a
  different, simpler cut of the same idea -- see ``docs/redteam.md``.
* The horizon choices stop at 7 days, not the full 30-day window
  ``ml.early`` sweeps to. See :data:`HORIZONS` for the measured runtimes
  that bound this.

A caller may also narrow the *display* to one bank. That narrowing never
touches the computation: every account and every transfer up to the
cutoff, across all three banks, is fed to :func:`ml.features.build_features`
regardless of the bank filter, because ``cb_distinct_counterparty_banks``
would silently be wrong for accounts whose cross-bank counterparty had
been filtered out of the bank map. The filter only trims which of the
already-correct results are returned. See :func:`_prepare` for where this
is enforced.

Two real HTTP calls, not one, for the two-step structure
----------------------------------------------------------
The brief is to make "load data" a visible step, separate from "score
it" -- not two labels painted on one spinner. ``ml.features.build_features``
*is* the load step (the feature engineering over the bounded transfer
slice) and :func:`ml.rule.velocity_decision` *is* the score step (a few
vectorised comparisons over an already-built table, effectively free next
to feature-building). So :func:`load_subset` does only the first and
:func:`score_subset` does both -- meaning a full click-through
(load, then score) runs feature-building twice. That duplication is a
deliberate trade for genuine two-step visibility with no server-side
session state to hold a half-finished computation between two requests:
see ``agents/service.py``'s two route docstrings.

Why this needs no new rate limiter
-----------------------------------
:data:`HORIZONS` and :data:`BANK_CHOICES` are the whole input surface --
both closed, small enumerations FastAPI rejects anything outside of with
its own ``422`` (see the ``Literal`` types on ``SubsetIn`` in
``agents/service.py``) -- so a caller cannot ask for a wider slice than the
widest entry already measured below. That mirrors exactly how
``ml.redteam.PATTERN_BOUNDS`` is the only guard ``POST /simulate`` needs.
The measured ceiling (the ``7d`` row) is what makes that bound honest
rather than assumed; see the docstring on :data:`HORIZONS`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from agents.data import DataStore
from ml.config import MlConfig
from ml.features import build_features
from ml.model_runtime import MODEL_A, load_model, score as model_score
from ml.rule import velocity_decision

_log = logging.getLogger("agents.subset")

#: Wall-clock bound on one ``/subset/*`` request (see ``agents/service.py``).
#: The measured worst cold build is a few seconds; this is the allowance for a
#: slow instance, and sits inside the UI's own 30s call timeout so the UI sees
#: this structured 504 rather than its own timeout.
REQUEST_TIMEOUT_S = 25.0

#: horizon key -> (label, timedelta since the dataset's first transfer).
#:
#: **Measured, not guessed.** On the canonical seed-42 dataset (30,000
#: accounts, 572,890 transfers over a 30-day window), filtering
#: ``transfers`` to ``timestamp <= T0 + horizon`` and then running
#: :func:`ml.features.build_features` over the result took, wall clock,
#: single runs on this machine:
#:
#: ========  ===========  ==============
#: horizon   transfers    build_features
#: ========  ===========  ==============
#: 30m               294        ~0.8s
#: 2h               1,193        ~0.9s
#: 6h               3,610        ~0.9s
#: 24h             14,481        ~1.0s
#: 3d              48,744        ~1.2s
#: 7d             118,406        ~1.7s
#: ========  ===========  ==============
#: (contrast: the *full* 30-day window is 572,890 transfers and took
#: ~8.5s -- comfortably outside "a few seconds", which is exactly why
#: the grid below stops at 7 days rather than reaching the full window.
#: The full-window comparison already exists, precomputed, on the metrics
#: tab; this panel's job is the fast end nothing there shows live.)
#:
#: Every entry here is well under the few-seconds target with headroom,
#: which is the actual abuse bound: the worst a caller can ask for costs
#: about 2 seconds of server time, from a fixed menu they cannot widen.
HORIZONS: dict[str, tuple[str, timedelta]] = {
    "30m": ("30 minutes after the first transfer", timedelta(minutes=30)),
    "2h": ("2 hours after the first transfer", timedelta(hours=2)),
    "6h": ("6 hours after the first transfer", timedelta(hours=6)),
    "24h": ("1 day after the first transfer", timedelta(hours=24)),
    "3d": ("3 days after the first transfer", timedelta(days=3)),
    "7d": ("7 days after the first transfer", timedelta(days=7)),
}

#: Bank filter choices. Mirrors ``simulator.generator.BANKS`` (see
#: ``agents/simulate.py`` for why this repo keeps small hard-coded copies
#: of that tuple rather than importing the module that pulls Pillow in).
#: ``"all"`` means no display filter -- the default and the only choice
#: that shows the cross-bank story the rest of this app is about.
BANK_CHOICES: tuple[str, ...] = ("all", "BANK_A", "BANK_B", "BANK_C")

#: Flagged accounts returned per call, sorted by cross-bank in-degree
#: descending. A bound for the same reason ``ui/``'s graph panel caps
#: nodes sent to the browser: the point is a representative sample of
#: what got flagged, not a second full-table dump.
MAX_FLAGGED_ACCOUNTS_RETURNED = 50


class InvalidSubsetChoice(ValueError):
    """Raised for a ``horizon`` or ``bank`` outside the fixed menus."""


def options_payload() -> dict[str, Any]:
    """The horizon and bank choices for the live subset form.

    Read off :data:`HORIZONS` and :data:`BANK_CHOICES` rather than
    retyped, so a horizon added or removed here reaches the form with no
    second place to edit -- same convention as
    ``agents.simulate.bounds_payload``.
    """
    return {
        "horizons": [
            {"key": key, "label": label, "minutes": int(delta.total_seconds() // 60)}
            for key, (label, delta) in HORIZONS.items()
        ],
        "banks": list(BANK_CHOICES),
    }


def _validate(horizon: str, bank: str) -> None:
    if horizon not in HORIZONS:
        raise InvalidSubsetChoice(
            f"unknown horizon {horizon!r}; choose one of {sorted(HORIZONS)}"
        )
    if bank not in BANK_CHOICES:
        raise InvalidSubsetChoice(
            f"unknown bank {bank!r}; choose one of {list(BANK_CHOICES)}"
        )


@dataclass(frozen=True)
class Prepared:
    """One horizon's feature build over every account and every transfer up
    to that horizon's cutoff, and what it cost to build."""

    features: pd.DataFrame
    bank_of: pd.Series
    cutoff: pd.Timestamp
    transfers_considered: int
    build_ms: float
    data_version: str


#: ``(data_version, horizon)`` -> :class:`Prepared`. Keyed by the data's content
#: hash, so a re-published table can never be served from an old build. At
#: most ``len(HORIZONS)`` entries per data version, a few MB each: the key
#: space is the fixed horizon menu, so this cannot grow without bound. The
#: bank filter is deliberately NOT part of the key -- it only trims what is
#: shown, so every bank choice at one horizon shares one build.
_prepared_cache: dict[tuple[str, str], Prepared] = {}
_prepared_locks: dict[tuple[str, str], threading.Lock] = {}
_prepared_guard = threading.Lock()


def _build_prepared(horizon: str, store: DataStore, version: str) -> Prepared:
    accounts = store.table("accounts")
    transfers = store.table("transfers")
    _label, horizon_delta = HORIZONS[horizon]
    t0 = transfers["timestamp"].min()
    cutoff = t0 + horizon_delta

    sub_transfers = transfers.loc[transfers["timestamp"] <= cutoff]
    tables = {
        "accounts": accounts,
        "transfers": sub_transfers,
        "devices": store.table("devices"),
        "kyc_submissions": store.table("kyc_submissions"),
        "kyc_documents": store.table("kyc_documents"),
    }
    cutoff_series = pd.Series(cutoff, index=pd.Index(accounts["hashed_account_id"]))

    started = time.perf_counter()
    features = build_features(MlConfig.from_env(), tables=tables, cutoff=cutoff_series)
    build_ms = (time.perf_counter() - started) * 1000.0
    bank_of = accounts.set_index("hashed_account_id")["bank_id"]
    return Prepared(features, bank_of, cutoff, len(sub_transfers), build_ms, version)


def prepared_for(horizon: str, store: DataStore) -> tuple[Prepared, bool]:
    """The cached build for ``horizon``, building it once if needed.

    Returns ``(prepared, cache_hit)``. Concurrent first requests for the same
    key build once: the others wait on the key's lock and then hit.
    """
    version = store.data_version()
    key = (version, horizon)
    cached = _prepared_cache.get(key)
    if cached is not None:
        return cached, True
    with _prepared_guard:
        lock = _prepared_locks.setdefault(key, threading.Lock())
    with lock:
        cached = _prepared_cache.get(key)
        if cached is not None:
            return cached, True
        built = _build_prepared(horizon, store, version)
        _prepared_cache[key] = built
        return built, False


def warm(store: DataStore) -> None:
    """Build every horizon's features now, cheapest first.

    Run in a background thread at service start so the first judge to open the
    panel does not pay the build. Best effort: a failure is logged and the
    panel simply builds on demand as before.
    """
    for horizon in HORIZONS:
        try:
            prepared_for(horizon, store)
        except Exception:  # noqa: BLE001 - warming must never take the service down
            _log.warning("subset warm-up failed at horizon %s", horizon, exc_info=True)
            return


def _prepare(horizon: str, bank: str, store: DataStore) -> tuple[Prepared, bool, float]:
    """Build (or reuse) the features for ``horizon``. The "load" step.

    Returns ``(prepared, cache_hit, load_ms)``. ``load_ms`` is what THIS
    request spent: the full build on a miss, almost nothing on a hit;
    ``prepared.build_ms`` is what the build itself cost. See
    :func:`prepared_for` for the cache.

    ``store`` supplies ``accounts``/``transfers``/``devices``/
    ``kyc_submissions``/``kyc_documents`` -- the same
    :class:`agents.data.DataStore` ``/screen`` and ``/chat`` already share
    on this process, so a table already read for one endpoint is not
    re-read from parquet for this one. ``ground_truth`` is deliberately
    never touched: a live detector has no ring labels to peek at, and
    reading it here would make this panel a quieter version of the
    precomputed metrics grid rather than what it claims to be.

    The **bank filter is never applied here.** ``accounts`` stays every
    account in the dataset and ``transfers`` stays every bank's traffic up
    to the cutoff, because ``ml.features.build_features`` needs the full
    account/bank map to resolve a cross-bank counterparty's bank -- narrow
    ``accounts`` to one bank and every other bank's accounts silently drop
    out of the bank lookup, undercounting
    ``cb_distinct_counterparty_banks`` for exactly the accounts a
    cross-bank rule cares about. The filter is applied afterwards, in
    :func:`_display_frame`, to *which rows are shown*, never to what was
    computed.
    """
    started = time.perf_counter()
    prepared, hit = prepared_for(horizon, store)
    load_ms = (time.perf_counter() - started) * 1000.0
    return prepared, hit, load_ms


def _display_frame(
    features: pd.DataFrame, bank_of: pd.Series, bank: str
) -> pd.DataFrame:
    """Apply the display-only bank filter. See :func:`_prepare`."""
    if bank == "all":
        return features
    keep = features["hashed_account_id"].map(bank_of) == bank
    return features.loc[keep]


def load_subset(horizon: str, bank: str, store: DataStore) -> dict[str, Any]:
    """The "load" step alone: build features for this subset, report timing.

    No rule is evaluated and nothing is flagged here -- see the module
    docstring on why this is a separate call from :func:`score_subset`
    rather than a first phase of one response.
    """
    _validate(horizon, bank)
    prepared, hit, load_ms = _prepare(horizon, bank, store)
    shown = _display_frame(prepared.features, prepared.bank_of, bank)
    label = HORIZONS[horizon][0]
    return {
        "horizon": horizon,
        "horizon_label": label,
        "bank": bank,
        "cutoff": prepared.cutoff.isoformat(),
        "transfers_considered": prepared.transfers_considered,
        "accounts_in_scope": int(len(shown)),
        "load_ms": int(round(load_ms)),
        "cache": "hit" if hit else "miss",
        "feature_build_ms": int(round(prepared.build_ms)),
        "data_version": prepared.data_version,
    }


#: Feature columns surfaced per flagged account, from the ``cb_`` scope --
#: the five the rule itself decides on, per ``ml.rule.velocity_decision``.
_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "distinct_counterparty_banks",
    "passthrough_ratio",
    "passthrough_speed_seconds",
)


#: Loaded once per process, not per request -- a booster load is a disk
#: read + deserialize, not something to repeat on every call. ``None``
#: means "not loaded yet"; a failed load caches as the sentinel below so a
#: missing model does not retry a failing disk read on every request.
_model_a_cache: list[Any] = []  # [LoadedModel | _NO_MODEL]
_NO_MODEL = object()


def _get_model_a(store: DataStore) -> Any:
    if not _model_a_cache:
        loaded = load_model(store.config.models_dir, MODEL_A)
        _model_a_cache.append(loaded if loaded is not None else _NO_MODEL)
    cached = _model_a_cache[0]
    return None if cached is _NO_MODEL else cached


def score_subset(horizon: str, bank: str, store: DataStore) -> dict[str, Any]:
    """Score this subset, live, on this request.

    Reruns :func:`_prepare` (see the module docstring on why this
    duplicates the ``load`` call's work rather than reusing it), then
    scores the ``cb_`` features with Model A
    (``ml/model_runtime.py``) -- the 16-feature model trained on the real
    published devices/KYC tables this path has access to. If the model is
    unavailable or fails to score (:func:`ml.model_runtime.score` returns
    ``None`` on any failure, never raises), this falls back to
    :func:`ml.rule.velocity_decision` automatically -- a judge never sees
    a crash, only a different, clearly-labelled detector. ``sb_`` (the
    single-bank comparison column) always uses the rule: Model A was
    never scoped as a single-bank detector.
    """
    _validate(horizon, bank)
    prepared, hit, load_ms = _prepare(horizon, bank, store)
    features, bank_of = prepared.features, prepared.bank_of
    cutoff, transfers_considered = prepared.cutoff, prepared.transfers_considered

    started = time.perf_counter()
    model_a = _get_model_a(store)
    cb_scores = model_score(model_a, features) if model_a is not None else None
    if cb_scores is not None:
        cb_flag = cb_scores >= model_a.metadata["threshold_at_1pct_fpr"]
        detector = "model_a"
    else:
        cb_flag = velocity_decision(features, "cb_")
        detector = "rule_fallback"
    sb_flag = velocity_decision(features, "sb_", require_bank_spread=False)
    score_ms = (time.perf_counter() - started) * 1000.0

    shown = _display_frame(features, bank_of, bank).copy()
    # cb_flag/sb_flag are numpy arrays positionally aligned to `features`,
    # not to `shown` (which may be bank-filtered), so they are reindexed
    # onto `shown` by position rather than by label. Leading underscores
    # are avoided in these column names: itertuples() below cannot expose
    # them as namedtuple attributes and silently renames them instead.
    shown_positions = features.index.get_indexer(shown.index)
    shown["flag_cb"] = cb_flag[shown_positions]
    shown["flag_sb"] = sb_flag[shown_positions]
    shown["display_bank_id"] = shown["hashed_account_id"].map(bank_of)

    flagged = shown.loc[shown["flag_cb"] | shown["flag_sb"]].sort_values(
        "cb_in_degree", ascending=False
    )
    truncated = len(flagged) > MAX_FLAGGED_ACCOUNTS_RETURNED
    flagged_accounts = [
        {
            "hashed_account_id": str(row.hashed_account_id),
            "bank_id": str(row.display_bank_id),
            "caught_cross_bank": bool(row.flag_cb),
            "caught_single_bank": bool(row.flag_sb),
            **{name: _jsonable(getattr(row, f"cb_{name}")) for name in _EVIDENCE_COLUMNS},
        }
        for row in flagged.head(MAX_FLAGGED_ACCOUNTS_RETURNED).itertuples()
    ]

    label = HORIZONS[horizon][0]
    return {
        "horizon": horizon,
        "horizon_label": label,
        "bank": bank,
        "cutoff": cutoff.isoformat(),
        "transfers_considered": transfers_considered,
        "accounts_in_scope": int(len(shown)),
        "cross_bank_flagged": int((shown["flag_cb"]).sum()),
        "single_bank_flagged": int((shown["flag_sb"]).sum()),
        "flagged_accounts": flagged_accounts,
        "flagged_accounts_truncated": truncated,
        "detector": detector,
        "load_ms": int(round(load_ms)),
        "score_ms": int(round(score_ms)),
        "total_ms": int(round(load_ms + score_ms)),
        "cache": "hit" if hit else "miss",
        "feature_build_ms": int(round(prepared.build_ms)),
        "data_version": prepared.data_version,
    }


def _jsonable(value: Any) -> Any:
    """int64/float64 -> plain Python, so FastAPI can serialise it."""
    if hasattr(value, "item"):
        return value.item()
    return value
