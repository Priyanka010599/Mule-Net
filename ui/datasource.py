"""Data access for the UI. One seam, so real sources swap in cleanly.

Everything a view needs arrives through :class:`DataSource`. Today it is
backed by the JSON snapshot ``ui/build_snapshot.py`` derives from
``simulator/out/``; tomorrow the same class can read ``ml/out/*.parquet``
or AlloyDB. The views and the API models never learn which, because the
only thing that changes is how :meth:`DataSource._read_snapshot` fills
the same dict.

``MULENET_READ_MODE=bigquery`` moves one thing off the snapshot: the
ring index, which :mod:`ui.bqrings` derives with a single query. The
snapshot stays the default and the fallback, and the payload's
``ring_list_source`` always says which of the two answered. Nothing else
moves -- in particular the metrics page cannot, because the BigQuery
dataset has no metrics table.

Metrics are separate and already live: ``ml/experiments/early_detection/
metrics.json`` is read straight through when it exists, with a bundled
copy in ``ui/data/`` as the fallback so the page works on a machine
where ``ml/`` has never run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ui.bqrings import (
    BIGQUERY_FIELDS,
    SNAPSHOT_ONLY_FIELDS,
    BigQueryRingsError,
    fetch_ring_rows,
)
from shared.evasion_verdict import structured_verdict
from ui.config import UiConfig
from ui.models import (
    BankTile,
    BoundingInfo,
    CoalitionPayload,
    CoalitionPoint,
    CoalitionWorld,
    CoverageSpan,
    DemoPlan,
    DemoState,
    EnclaveDebugRefusal,
    EnclaveOutboxProof,
    EnclavePayload,
    HorizonMetrics,
    LeadStats,
    MetricsPayload,
    Overview,
    RedteamPayload,
    RedteamStrategyRow,
    RedteamVerdict,
    ReplayPayload,
    RetiredMetric,
    RingCoverage,
    RingDetail,
    RingListSource,
    RingSummary,
    ScopeMetrics,
    SeedCoverage,
    SpanComparison,
)

#: Metric key in ``ml``'s JSON -> field on :class:`ScopeMetrics`.
_METRIC_FIELDS: dict[str, str] = {
    "pr_auc_mean": "pr_auc",
    "roc_auc_mean": "roc_auc",
    "precision_at_k_mean": "precision_at_k",
    "recall_at_fpr_1pct_mean": "recall_at_fpr_1pct",
    "false_alerts_per_catch_at_80pct_recall_mean": "false_alerts_per_catch",
}

#: Shown under the metrics table. The full-window numbers are
#: ceiling-bound (cross-bank PR-AUC ~0.999) and say nothing useful, so
#: the page headlines the early horizons instead.
CAVEAT = (
    "Two different questions, two different horizons. For the ACCOUNT-level "
    "scores below, early horizons are the credible regime: at T=5m the model "
    "has seen only five minutes of ring activity, which is the window in which "
    "an intervention can still stop the money. For the RING-level coverage "
    "claim the last finite horizon is the honest one, because that is where no "
    "ring still has money in flight and exit-side censoring no longer shapes "
    "the number. The full-window row is shown for completeness only - by then "
    "both models are ceiling-bound and the comparison is meaningless."
)

#: Stated on the page in one line so nobody re-derives the looser
#: measure. "Flagged at some horizon" is 1.000 for both scopes at span 1
#: and hides the control result; a flag that lands after the money left
#: is not coverage in any sense a bank cares about, which is exactly what
#: the replay panel already says about a post-exit flag.
COVERAGE_DEFINITION = (
    "Coverage = the share of ALL rings in that span flagged at a measured "
    "horizon BEFORE the money reached the exit account. Not 'flagged "
    "eventually': a flag after cash-out does not count here, for the same "
    "reason it is not called a catch on the replay tab."
)

#: Both the coverage table and the lead sentence are quoted on this
#: basis. They must never be mixed: a 3-seed rate beside a single-seed
#: count reads as one measurement and is two.
LEAD_BASIS = "3-seed mean (seeds 42, 43, 44)"

#: Milliseconds the money-trail replay runs for, end to end.
REPLAY_DURATION_MS = 12_000

#: Label when ``MULENET_READ_MODE=bigquery`` was asked for and BigQuery
#: answered. Shown so the opposite case has something to contrast with:
#: a page that says nothing when the query works and nothing when it
#: fails discloses nothing either way.
BIGQUERY_LIVE_LABEL = "ring list live from BigQuery"

#: Label when it was asked for and did NOT answer. Deliberately not the
#: chat dock's "offline demo mode": that phrase means "a keyword matcher
#: answered instead of Gemini", this one means "a file answered instead
#: of a query", and one wording for two different degradations would
#: leave a viewer unable to tell which had happened.
BIGQUERY_FALLBACK_LABEL = "BigQuery unavailable - ring list read from file"

#: Said on the metrics page whenever the read mode is ``bigquery``. The
#: dataset has nine tables and none of them is a metrics table, so this
#: page is on files in both modes and must say so where a viewer who saw
#: the BigQuery label in the ring picker would otherwise assume otherwise.
METRICS_READ_MODE_NOTE = (
    "MULENET_READ_MODE=bigquery moved the RING INDEX onto BigQuery and "
    "nothing on this page. Every number here was read from a file, named "
    "above: the BigQuery dataset has no metrics table to read - its nine "
    "tables are transfers, scores, rings, ring_explanations, "
    "account_explanations, account_evidence, cases, features and str_reports."
)


class SnapshotMissingError(RuntimeError):
    """Raised when the demo snapshot has not been built yet."""


def _strategy_row(verdict: dict[str, Any]) -> RedteamStrategyRow:
    """One red-team verdict as a counter row. No re-run: the clause tables
    already in the file say which clauses stopped the pattern."""
    structured = structured_verdict(
        caught_by=str(verdict["caught_by"]),
        clause_values=verdict.get("clause_values") or {},
        aimed_clause=str(verdict["aimed_at_clause"]),
        flagged_accounts=int(verdict.get("flagged_accounts", 0)),
        flagged_collectors=int(verdict.get("flagged_collectors", 0)),
        rings_formed=int(verdict.get("rings_formed", 0)),
        stage2_alertable=int(verdict.get("stage2_alertable", 0)),
    )
    return RedteamStrategyRow(
        name=str(verdict["name"]),
        targets_clause=str(verdict["aimed_at_clause"]),
        result=str(verdict["caught_by"]),
        caught=bool(verdict["caught"]),
        failing_clauses=structured["failing_clauses"],
        aimed_clause_passed=structured["aimed_clause_passed"],
        is_designed_miss=structured["is_designed_miss"],
    )


class DataSource:
    """Serves overview, bank tiles, ring detail and metrics.

    The snapshot is loaded once and cached; it is a few megabytes of
    already-shaped JSON, so no request ever pays for parsing parquet.
    """

    def __init__(self, cfg: UiConfig) -> None:
        self.cfg = cfg
        self._snapshot: dict[str, Any] | None = None
        self._metrics: tuple[dict[str, Any], str] | None = None
        #: Cached ring-index query, so one page load is one query rather
        #: than one per endpoint. Dropped by :meth:`reload`, which the
        #: demo-reset button calls, so a reset re-queries.
        self._ring_list: tuple[dict[str, dict[str, Any]] | None, RingListSource] | None
        self._ring_list = None

    # -- snapshot -----------------------------------------------------

    def _read_snapshot(self) -> dict[str, Any]:
        """The one place that knows where ring data physically lives.

        Swap point for real data: replace this body with a reader over
        ``ml/out/rings.parquet`` + ``scores.parquet`` + ``evidence.parquet``
        that returns the same keys. Nothing else in ``ui/`` changes.
        """
        path = self.cfg.snapshot_path
        if not path.exists():
            raise SnapshotMissingError(
                f"no demo snapshot at {path}. Build it with: "
                "python -m ui.build_snapshot"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def snapshot(self) -> dict[str, Any]:
        """Cached snapshot payload."""
        if self._snapshot is None:
            self._snapshot = self._read_snapshot()
        return self._snapshot

    def reload(self) -> None:
        """Drop the caches so the next request re-reads from disk."""
        self._snapshot = None
        self._metrics = None
        self._ring_list = None

    @property
    def bounding(self) -> BoundingInfo:
        """How the graph payload is capped."""
        return BoundingInfo(**self.snapshot["bounding"])

    def _ring_raw(self, ring_id: str) -> dict[str, Any]:
        for ring in self.snapshot["rings"]:
            if ring["ring_id"] == ring_id:
                return ring
        raise KeyError(ring_id)

    @staticmethod
    def _summary(
        ring: dict[str, Any], derived: dict[str, Any] | None = None
    ) -> RingSummary:
        """One ring-index row.

        ``derived``, when given, is the same ring's fields as BigQuery
        computed them (``ui.bqrings.BIGQUERY_FIELDS``). It *overrides*
        the snapshot values for those fields only; the fields BigQuery
        cannot answer stay on the snapshot and the payload's
        ``ring_list_source.snapshot_fields`` names them.
        """
        values: dict[str, Any] = {
            "ring_id": ring["ring_id"],
            "span": ring["span"],
            "banks_involved": ring["banks_involved"],
            "alertable_member_count": ring["alertable_member_count"],
            "context_member_count": ring["context_member_count"],
            "total_member_count": ring["total_member_count"],
            "status": ring["status"],
            "detected_at": ring["detected_at"],
            "onset": ring["onset"],
            "detector_version": ring["detector_version"],
            "total_intra_ring_amount": ring["total_intra_ring_amount"],
            "intra_ring_transfer_count": ring["intra_ring_transfer_count"],
            "cross_bank_transfer_count": ring["cross_bank_transfer_count"],
            "cross_bank_evidence_count": sum(
                1 for e in ring["evidence"] if e["cross_bank"]
            ),
            "true_positive_alertable_members": ring["true_positive_alertable_members"],
            "has_case": ring["case"] is not None,
            "true_ring_id": ring["true_ring_id"],
            "match_is_ambiguous": ring["match_is_ambiguous"],
            "has_replay": ring["has_replay"],
        }
        if derived:
            values.update({k: derived[k] for k in BIGQUERY_FIELDS if k in derived})
        return RingSummary(**values)

    # -- ring index provenance ---------------------------------------

    def _bigquery_rows(
        self, ring_ids: list[str]
    ) -> tuple[dict[str, dict[str, Any]] | None, RingListSource]:
        """Try BigQuery for the ring index; describe whatever happened.

        Never raises. Any failure -- no credentials, API error, timeout,
        zero rows, or a row missing for a ring the snapshot has -- returns
        ``None`` and a source block whose ``served_by`` is ``snapshot``
        and whose ``label`` says so. A silent fallback here would let a
        judge believe the ring counts came out of BigQuery when they came
        out of a file.
        """
        try:
            result = fetch_ring_rows(self.cfg, ring_ids)
            missing = [r for r in ring_ids if r not in result.rows]
            if missing:
                raise BigQueryRingsError(
                    f"{len(missing)} of {len(ring_ids)} rings absent from "
                    f"BigQuery, first {missing[0]}"
                )
        except BigQueryRingsError as exc:
            # One line on the page, the whole thing in the payload.
            # Google's errors carry a multi-line "Location / Job ID"
            # block, and a wall of it would bury the sentence that
            # actually matters -- that a FILE answered.
            headline = str(exc).splitlines()[0].strip().rstrip('.')
            return None, RingListSource(
                mode="bigquery",
                served_by="snapshot",
                label=BIGQUERY_FALLBACK_LABEL,
                detail=(
                    "MULENET_READ_MODE=bigquery was configured, but the ring-index "
                    f"query against {self.cfg.bq_project}.{self.cfg.bq_dataset} did "
                    f"not answer: {headline}. The rings below were read from the local "
                    f"snapshot file {self.cfg.snapshot_path.name} instead, so "
                    "NOTHING on this page came from BigQuery."
                ),
                snapshot_fields=sorted({*BIGQUERY_FIELDS, *SNAPSHOT_ONLY_FIELDS}),
                fallback_reason=str(exc),
            )
        return result.rows, RingListSource(
            mode="bigquery",
            served_by="bigquery",
            label=BIGQUERY_LIVE_LABEL,
            detail=(
                f"{len(result.rows)} rings derived by one query over "
                f"{result.table_ref}.rings joined to transfers, account_evidence "
                f"and cases: {result.bytes_processed:,} bytes scanned in "
                f"{result.latency_ms:.0f} ms"
                + (" (cached result, nothing billed)" if result.cache_hit else "")
                + ". "
                + f"{len(SNAPSHOT_ONLY_FIELDS)} fields per ring are NOT derivable "
                "from this dataset and still come from the snapshot: "
                + ", ".join(SNAPSHOT_ONLY_FIELDS)
                + "."
            ),
            bigquery_fields=list(BIGQUERY_FIELDS),
            snapshot_fields=list(SNAPSHOT_ONLY_FIELDS),
            bytes_processed=result.bytes_processed,
            bytes_billed=result.bytes_billed,
            cache_hit=result.cache_hit,
            latency_ms=result.latency_ms,
            table_ref=result.table_ref,
        )

    def overview(self) -> Overview:
        """Header payload plus the ring index, widest span first."""
        snap = self.snapshot
        derived: dict[str, dict[str, Any]] | None = None
        if self.cfg.read_mode == "bigquery":
            if self._ring_list is None:
                self._ring_list = self._bigquery_rows(
                    [r["ring_id"] for r in snap["rings"]]
                )
            derived, source = self._ring_list
        else:
            # The default. Unchanged behaviour, and the label stays empty
            # because there is no degradation to disclose.
            source = RingListSource(
                mode="snapshot",
                served_by="snapshot",
                detail=f"ring index from {self.cfg.snapshot_path.name}",
            )
        rings = [
            self._summary(r, derived.get(r["ring_id"]) if derived else None)
            for r in snap["rings"]
        ]
        rings.sort(key=lambda r: (-r.span, -r.cross_bank_evidence_count, r.ring_id))
        return Overview(
            ring_list_source=source,
            generated_at=snap["generated_at"],
            source=snap["source"],
            mock=snap["mock"],
            dataset=snap["dataset"],
            detector=snap["detector"],
            bounding=self.bounding,
            featured_ring_id=snap["featured_ring_id"],
            rings=rings,
            ring_match=snap["ring_match"],
            demo=self.demo_plan,
        )

    @property
    def demo_plan(self) -> DemoPlan:
        """The two demo rings, derived at snapshot-build time."""
        return DemoPlan(**self.snapshot["demo"])

    def replay(self, ring_id: str) -> ReplayPayload:
        """Replay timeline for one detected cluster.

        Every cluster the detector published has an entry, including the
        ones this snapshot left out and the one that matched no true ring
        at all: those come back with ``available`` false and a named
        reason, which is what the panel renders. Only an id that was never
        published at all raises.
        """
        replay = self.snapshot["replays"].get(ring_id)
        if replay is None:
            raise KeyError(ring_id)
        return ReplayPayload(**replay)

    @staticmethod
    def _withhold_labels(bank: dict[str, Any]) -> dict[str, Any]:
        """A copy of one snapshot bank with every ground-truth-derived value nulled.

        ``queue_depth`` and ``reached_recall`` go too: the depth that reaches
        80% recall is computed from the labels, so it states how many mules
        there are even though it names none.
        """
        queue = {
            **bank["queue"],
            **dict.fromkeys(
                ("queue_depth", "queue_true_positives", "queue_false_positives",
                 "false_alerts_per_catch", "reached_recall")
            ),
        }
        return {
            **bank,
            "mules_in_bank": None,
            "alert_true_positives": None,
            "alert_false_positives": None,
            "alerts": [{**a, "is_mule": None} for a in bank["alerts"]],
            "queue": queue,
        }

    def banks(
        self, ring_id: str | None = None, reveal_labels: bool = True
    ) -> list[BankTile]:
        """The three bank dashboards, each restricted to its own data.

        When ``ring_id`` is given, each tile also reports the fragment of
        that ring the bank can see alone -- which is the whole point of
        the before/after argument. ``reveal_labels=False`` withholds the
        simulator's ground truth (see :meth:`_withhold_labels`).
        """
        fragments: dict[str, dict[str, Any]] = {}
        if ring_id:
            fragments = self._ring_raw(ring_id)["single_bank_view"]
        tiles: list[BankTile] = []
        for bank in self.snapshot["banks"]:
            fragment = fragments.get(bank["bank_id"])
            if not reveal_labels:
                bank = self._withhold_labels(bank)
            tiles.append(
                BankTile(
                    **bank,
                    labels_revealed=reveal_labels,
                    visible_member_hashed_account_ids=(
                        fragment["visible_member_hashed_account_ids"] if fragment else []
                    ),
                    invisible_member_count=(
                        fragment["invisible_member_count"] if fragment else 0
                    ),
                    visible_intra_ring_transfers=(
                        fragment["visible_intra_ring_transfers"] if fragment else 0
                    ),
                    visible_alertable_hashed_account_ids=(
                        fragment["visible_alertable_hashed_account_ids"]
                        if fragment
                        else []
                    ),
                )
            )
        return tiles

    def ring(self, ring_id: str) -> RingDetail:
        """Full ring payload for the joint view and the case panel."""
        raw = self._ring_raw(ring_id)
        summary = self._summary(raw)
        return RingDetail(
            **summary.model_dump(),
            all_member_hashed_account_ids=raw["all_member_hashed_account_ids"],
            alertable_member_hashed_account_ids=raw[
                "alertable_member_hashed_account_ids"
            ],
            context_member_hashed_account_ids=raw[
                "context_member_hashed_account_ids"
            ],
            velocity_member_hashed_account_ids=raw[
                "velocity_member_hashed_account_ids"
            ],
            identity_link_member_hashed_account_ids=raw[
                "identity_link_member_hashed_account_ids"
            ],
            members=raw["members"],
            context_accounts=raw["context_accounts"],
            transfers=raw["transfers"],
            evidence=raw["evidence"],
            explanation=raw["explanation"],
            case=raw["case"],
            report=raw["report"],
            single_bank_view=raw["single_bank_view"],
            bounding=self.bounding,
        )

    def demo_reset(self) -> DemoState:
        """Reset demo state and pick the ring that makes the argument."""
        self.reload()
        snap = self.snapshot
        ring_id = snap["featured_ring_id"]
        ring = self._ring_raw(ring_id)
        plan = self.demo_plan
        reason = (
            f"{len(ring['members'])} accounts across {ring['span']} banks, "
            f"{ring['cross_bank_transfer_count']} of "
            f"{ring['intra_ring_transfer_count']} hops cross a bank boundary, "
            f"{sum(1 for e in ring['evidence'] if e['cross_bank'])} cross-bank "
            "identity reuse matches"
        )
        if plan.primary and plan.primary.ring_id == ring_id:
            # Say what the ring was picked FOR, not just that it is big.
            reason = f"{reason}. {plan.primary.claim}"
        return DemoState(
            reset_at=datetime.now(UTC).isoformat(),
            featured_ring_id=ring_id,
            reason=reason,
            replay_transfer_count=len(ring["transfers"]),
            replay_duration_ms=REPLAY_DURATION_MS,
            plan=plan,
        )

    # -- metrics ------------------------------------------------------

    @staticmethod
    def _provenance(
        live: Path, fallback: Path, source: str, generated_at: str
    ) -> str:
        """Name the file that was actually opened, never the configured one.

        ``infra/cloudbuild.ignore`` excludes ``ml/experiments/`` from the
        container image, so in a deployed container the live paths do not
        exist and ``ui/data/`` is what gets read. A provenance line
        hard-coded to the live path would name a file the process never
        opened -- true on a dev box, false where the demo runs. Both the
        early-detection header and the coverage block resolve their line
        here, from the same ``source`` flag the read returned.
        """
        if source == "ml":
            return f"read live from {live.as_posix()}"
        return (
            f"bundled copy in {fallback.name}, "
            f"generated {generated_at[:10]} (ml/ has not run here)"
        )

    def _read_metrics(self) -> tuple[dict[str, Any], str]:
        if self.cfg.metrics_path.exists():
            return json.loads(self.cfg.metrics_path.read_text(encoding="utf-8")), "ml"
        fallback: Path = self.cfg.metrics_fallback_path
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8")), "fallback"
        raise SnapshotMissingError(
            f"no metrics at {self.cfg.metrics_path} or {fallback}. Run ml/early.py "
            "or build the snapshot, which seeds the fallback."
        )

    def _read_lead_metrics(self) -> tuple[dict[str, Any], str]:
        if self.cfg.lead_metrics_path.exists():
            return (
                json.loads(self.cfg.lead_metrics_path.read_text(encoding="utf-8")),
                "ml",
            )
        fallback: Path = self.cfg.lead_metrics_fallback_path
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8")), "fallback"
        raise SnapshotMissingError(
            f"no lead-time metrics at {self.cfg.lead_metrics_path} or {fallback}. "
            "Run ml/leadtime.py or build the snapshot, which seeds the fallback."
        )

    def _read_coalition(self) -> tuple[dict[str, Any], str]:
        if self.cfg.coalition_path.exists():
            return (
                json.loads(self.cfg.coalition_path.read_text(encoding="utf-8")),
                "ml",
            )
        fallback: Path = self.cfg.coalition_fallback_path
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8")), "fallback"
        raise SnapshotMissingError(
            f"no coalition sweep at {self.cfg.coalition_path} or {fallback}. "
            "Run ml/coalition.py or build the snapshot, which seeds the fallback."
        )

    def coalition(self) -> CoalitionPayload:
        """Ring recall by number of banks joined, from ``ml/coalition.py``.

        Reads a file and reshapes it. Nothing here runs a detector: the
        sweep is 22 feature builds and about 50 seconds, which is an
        offline job, not a request. The control on the page therefore
        moves between numbers that were all measured before the page
        loaded -- which is also what makes it honest, since a number
        computed on demand from a container that has no
        ``simulator/out/`` would have to be invented.

        Worlds are returned in the order ``ml/coalition.py`` wrote them,
        real partition first, so the page shows the measured curve before
        the split-derived one.
        """
        raw, source = self._read_coalition()
        worlds = [
            CoalitionWorld(
                label=str(world["label"]),
                is_real_partition=bool(world["is_real_partition"]),
                note=str(world["note"]),
                points=[CoalitionPoint(**point) for point in world["by_size"]],
            )
            for world in raw["worlds"].values()
        ]
        return CoalitionPayload(
            generated_at=str(raw["generated_at"]),
            source=source,  # type: ignore[arg-type]
            source_detail=self._provenance(
                self.cfg.coalition_path,
                self.cfg.coalition_fallback_path,
                source,
                str(raw["generated_at"]),
            ),
            true_rings_total=int(raw["true_rings_total"]),
            metric_definition=str(raw["metric_definition"]),
            not_comparable_to_claims=str(raw["not_comparable_to_claims"]),
            two_variants=str(raw["two_variants"]),
            worlds=worlds,
        )

    def _read_redteam(self) -> tuple[dict[str, Any], str]:
        if self.cfg.redteam_path.exists():
            return (
                json.loads(self.cfg.redteam_path.read_text(encoding="utf-8")),
                "ml",
            )
        fallback: Path = self.cfg.redteam_fallback_path
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8")), "fallback"
        raise SnapshotMissingError(
            f"no red-team run at {self.cfg.redteam_path} or {fallback}. "
            "Run ml/redteam.py or build the snapshot, which seeds the fallback."
        )

    def redteam(self) -> RedteamPayload:
        """Gemini-designed evasion strategies, run through the real detector.

        Reads a file and reshapes it. This is the one experiment in the
        repo this app never recomputes under any circumstance:
        ``ml/redteam.py`` calls live Gemini with no rate limit, so a public
        endpoint that re-ran it on request would have no cost ceiling. The
        result rendered here was reviewed once and is served as-is.
        """
        raw, source = self._read_redteam()
        verdicts = [
            RedteamVerdict(
                name=str(v["name"]),
                description=str(v["description"]),
                targets=str(v["targets"]),
                aimed_at_clause=str(v["aimed_at_clause"]),
                caught=bool(v["caught"]),
                caught_by=str(v["caught_by"]),
                reasoning=str(v["reasoning"]),
            )
            for v in raw["verdicts"]
        ]
        return RedteamPayload(
            generated_at=str(raw["generated_at"]),
            source=source,  # type: ignore[arg-type]
            source_detail=self._provenance(
                self.cfg.redteam_path,
                self.cfg.redteam_fallback_path,
                source,
                str(raw["generated_at"]),
            ),
            strategies_run=int(raw["strategies_run"]),
            caught=int(raw["caught"]),
            missed=int(raw["missed"]),
            isolated_graph_caveat=str(raw["isolated_graph_caveat"]),
            scorer_not_attacked=str(raw["scorer_not_attacked"]),
            verdicts=verdicts,
            strategies=[_strategy_row(v) for v in raw["verdicts"]],
        )

    def enclave(self) -> EnclavePayload:
        """One real Confidential Space run's evidence (docs/enclave.md, Stage 3).

        Reads a single committed file and reshapes it. There is no live
        counterpart to fall back from here, unlike every other payload in
        this module: reproducing this data needs a real GCP project, KMS
        and a Confidential VM, which nothing this app runs can reach. So
        this is not "the fallback because ml/ hasn't run" -- it is the
        only copy, captured once from a real run and served as-is.
        """
        if not self.cfg.enclave_evidence_path.exists():
            raise SnapshotMissingError(
                f"no enclave evidence at {self.cfg.enclave_evidence_path}. "
                "This file ships committed; see docs/enclave.md for how it "
                "was captured -- it cannot be regenerated by this app."
            )
        raw = json.loads(self.cfg.enclave_evidence_path.read_text(encoding="utf-8"))
        proof_raw = raw["outbox_proof"]
        try:
            dataset = self.snapshot.get("dataset", {})
        except SnapshotMissingError:
            dataset = {}
        return EnclavePayload(
            generated_at=str(raw["generated_at"]),
            run_id=str(raw["run_id"]),
            attempt=str(raw["attempt"]),
            outcome=str(raw["outcome"]),
            key_release_ok=bool(raw["key_release_ok"]),
            running_debug_image=bool(raw["running_debug_image"]),
            image_digest=str(raw["image_digest"]),
            dbgstat=str(raw["dbgstat"]),
            secboot=bool(raw["secboot"]),
            hwmodel=str(raw["hwmodel"]),
            support_attributes=list(raw["support_attributes"]),
            contributing_banks=list(raw["contributing_banks"]),
            edge_count=int(raw["edge_count"]),
            account_count=int(raw["account_count"]),
            flagged_count=int(raw["flagged_count"]),
            ring_count=int(raw["ring_count"]),
            bundles_written=list(raw["bundles_written"]),
            outbox_proof=EnclaveOutboxProof(
                bank_id=str(proof_raw["bank_id"]),
                object_path=str(proof_raw["object_path"]),
                ciphertext_bytes_total=int(proof_raw["ciphertext_bytes_total"]),
                ciphertext_preview=str(proof_raw["ciphertext_preview"]),
                own_members_sample=list(proof_raw["own_members_sample"]),
                own_members_total=int(proof_raw["own_members_total"]),
                other_bank_member_counts=dict(proof_raw["other_bank_member_counts"]),
                edges_sample=list(proof_raw["edges_sample"]),
                decrypt_failure_bank_id=str(proof_raw["decrypt_failure_bank_id"]),
                decrypt_failure_error=str(proof_raw["decrypt_failure_error"]),
            ),
            note=str(raw["note"]),
            debug_refusal=EnclaveDebugRefusal(
                run_id=str(raw["debug_refusal"]["run_id"]),
                image_digest=str(raw["debug_refusal"]["image_digest"]),
                dbgstat=str(raw["debug_refusal"]["dbgstat"]),
                refusal_kind=str(raw["debug_refusal"]["refusal_kind"]),
                refusal_message=str(raw["debug_refusal"]["refusal_message"]),
                note=str(raw["debug_refusal"]["note"]),
            ),
            tamper_enabled=self.cfg.live_proof_tamper_enabled,
            edges_in_enclave_scope=int(raw["edge_count"]),
            accounts_in_enclave_scope=int(raw["account_count"]),
            accounts_generated=dataset.get("accounts_generated"),
            accounts_with_activity=dataset.get("accounts_with_activity"),
        )

    #: Spans the coverage table reports, and the key each one lives under
    #: in ``by_scope_and_span``. Resolution is by IDENTITY -- scope name
    #: and span key -- never by position in a list. A positional lookup
    #: against a reordered JSON object silently reports span 3's numbers
    #: under span 1, which is precisely the failure that makes the
    #: control look like a win.
    _COVERAGE_SPANS: tuple[int, ...] = (1, 2, 3)

    def coverage(self) -> RingCoverage | None:
        """Ring coverage by span, from ``ml/experiments/lead_time``.

        The published measure is ``share_with_positive_warning``: the
        share of ALL rings in the stratum flagged at a measured horizon
        **before the money reached the exit account**. The looser "flagged
        at some horizon" rate is derivable from the timeline parquet and
        is deliberately NOT used -- it reads 1.000 for both scopes at span
        1 and so conceals the control result, and it would also contradict
        the replay panel, which already refuses to call a post-exit flag a
        catch.
        """
        try:
            raw, source = self._read_lead_metrics()
        except SnapshotMissingError:
            # The rest of the metrics page still works without it.
            return None

        by_span = raw["seed_mean"]["by_scope_and_span"]
        overall = raw["seed_mean"]["by_scope"]
        per_seed_raw = raw.get("per_seed", {})
        # The replay tab is seed 42 throughout, so the per-seed column
        # shows seed 42 and the two tabs describe the same run. Chosen by
        # key, not by position.
        seed = "42" if "42" in per_seed_raw else next(iter(sorted(per_seed_raw)), None)

        def rate(block: dict[str, Any], scope: str, key: str) -> float:
            return float(block[scope][key]["share_with_positive_warning"])

        def rings(block: dict[str, Any], scope: str, key: str) -> float:
            return float(block[scope][key]["rings_with_positive_warning"])

        spans: list[CoverageSpan] = []
        for span in self._COVERAGE_SPANS:
            key = f"span{span}"
            if key not in by_span["single_bank"] or key not in by_span["cross_bank"]:
                continue
            single = rate(by_span, "single_bank", key)
            cross = rate(by_span, "cross_bank", key)
            per_seed = None
            if seed is not None:
                seed_span = per_seed_raw[seed]["by_scope_and_span"]
                per_seed = SeedCoverage(
                    seed=seed,
                    single_bank_rings=int(rings(seed_span, "single_bank", key)),
                    cross_bank_rings=int(rings(seed_span, "cross_bank", key)),
                    single_bank_rate=round(rate(seed_span, "single_bank", key), 4),
                    cross_bank_rate=round(rate(seed_span, "cross_bank", key), 4),
                )
            spans.append(
                CoverageSpan(
                    span=span,
                    label=f"{span} bank" if span == 1 else f"{span} banks",
                    rings=float(by_span["single_bank"][key]["rings"]),
                    single_bank_rings=round(rings(by_span, "single_bank", key), 2),
                    cross_bank_rings=round(rings(by_span, "cross_bank", key), 2),
                    single_bank_rate=round(single, 4),
                    cross_bank_rate=round(cross, 4),
                    delta=round(cross - single, 4),
                    per_seed=per_seed,
                )
            )

        lead_raw = raw["seed_mean"]["cross_bank_lead"]
        lead = LeadStats(
            basis=LEAD_BASIS,
            rings_both_scopes_flag=round(float(lead_raw["both_flagged_in_grid"]), 1),
            ties=round(float(lead_raw["rings_tied"]), 1),
            cross_bank_earlier=round(float(lead_raw["rings_cross_bank_flagged_first"]), 1),
            single_bank_earlier=round(
                float(lead_raw["rings_single_bank_flagged_first"]), 1
            ),
            median_lead_minutes=round(float(lead_raw["lead_minutes"]["p50"]), 1),
            mean_lead_minutes=round(float(lead_raw["lead_minutes"]["mean"]), 1),
        )
        return RingCoverage(
            source=self._provenance(
                self.cfg.lead_metrics_path,
                self.cfg.lead_metrics_fallback_path,
                source,
                str(raw["generated_at"]),
            ),
            source_path="seed_mean.by_scope_and_span.<scope>.span<n>"
            ".share_with_positive_warning",
            basis=LEAD_BASIS,
            definition=COVERAGE_DEFINITION,
            generated_at=str(raw["generated_at"]),
            rings=float(overall["single_bank"]["rings"]),
            horizon_grid=[int(t) for t in raw["horizon_grid"]],
            single_bank_rings=round(
                float(overall["single_bank"]["rings_with_positive_warning"]), 2
            ),
            cross_bank_rings=round(
                float(overall["cross_bank"]["rings_with_positive_warning"]), 2
            ),
            single_bank_rate=round(
                float(overall["single_bank"]["share_with_positive_warning"]), 4
            ),
            cross_bank_rate=round(
                float(overall["cross_bank"]["share_with_positive_warning"]), 4
            ),
            spans=spans,
            lead=lead,
            note=str(raw.get("conditional_note", "")),
        )

    @staticmethod
    def _scope(block: dict[str, Any]) -> ScopeMetrics:
        values = {field: float(block[key]) for key, field in _METRIC_FIELDS.items()}
        positives = block.get("positives")
        return ScopeMetrics(**values, positives=float(positives) if positives else None)

    def metrics(self) -> MetricsPayload:
        """Single-bank vs cross-bank, per horizon, stratified by ring span."""
        raw, source = self._metrics or self._read_metrics()
        self._metrics = (raw, source)

        horizons: list[HorizonMetrics] = []
        for entry in raw["horizons"]:
            single = self._scope(entry["summary"]["single_bank"])
            cross = self._scope(entry["summary"]["cross_bank"])
            spans: list[SpanComparison] = []
            strata = entry.get("strata_summary", {})
            for span in (1, 2, 3):
                key = f"span{span}"
                single_span = strata.get("single_bank", {}).get(key)
                cross_span = strata.get("cross_bank", {}).get(key)
                if not single_span or not cross_span:
                    continue
                s_metrics = self._scope(single_span)
                c_metrics = self._scope(cross_span)
                spans.append(
                    SpanComparison(
                        span=span,
                        label=f"{span} bank" if span == 1 else f"{span} banks",
                        single_bank=s_metrics,
                        cross_bank=c_metrics,
                        pr_auc_delta=round(c_metrics.pr_auc - s_metrics.pr_auc, 4),
                        roc_auc_delta=round(c_metrics.roc_auc - s_metrics.roc_auc, 4),
                    )
                )
            catch = entry.get("ring_catch", {})
            cutoff = entry.get("cutoff_minutes")
            horizons.append(
                HorizonMetrics(
                    cutoff_minutes=cutoff,
                    label=entry.get("label") or "full window",
                    headline=cutoff in self.cfg.headline_horizons,
                    single_bank=single,
                    cross_bank=cross,
                    pr_auc_delta=round(cross.pr_auc - single.pr_auc, 4),
                    spans=spans,
                    ring_recall_single_bank=catch.get("single_bank", {}).get("ring_recall"),
                    ring_recall_cross_bank=catch.get("cross_bank", {}).get("ring_recall"),
                    rings_money_not_yet_at_exit=catch.get("cross_bank", {}).get(
                        "rings_money_not_yet_at_exit"
                    ),
                )
            )

        # Headline on the LAST finite horizon. Exit-side censoring is gone
        # there, so ring recall is a statement about the detector rather
        # than about how much of the ring had happened yet.
        finite = [h for h in horizons if h.cutoff_minutes is not None]
        headline = finite[-1] if finite else horizons[-1]

        return MetricsPayload(
            model=raw["model"],
            generated_at=raw["generated_at"],
            evaluation=raw["evaluation"],
            onset=raw["onset"],
            exit_account=raw["exit_account"],
            source=source,  # type: ignore[arg-type]
            source_detail=self._provenance(
                self.cfg.metrics_path,
                self.cfg.metrics_fallback_path,
                source,
                str(raw["generated_at"]),
            ),
            horizons=horizons,
            caveat=CAVEAT,
            headline_ring_recall_horizon=headline.label,
            headline_ring_recall_single_bank=headline.ring_recall_single_bank,
            headline_ring_recall_cross_bank=headline.ring_recall_cross_bank,
            coverage=self.coverage(),
            retired=self._retired_metrics(raw),
            # Only in bigquery mode, and only ever to DENY a BigQuery
            # source. ``source_detail`` above already names the file.
            read_mode_note=(
                METRICS_READ_MODE_NOTE if self.cfg.read_mode == "bigquery" else None
            ),
        )

    @staticmethod
    def _retired_metrics(raw: dict[str, Any]) -> list[RetiredMetric]:
        """Metrics withdrawn from display, with the numbers that did it.

        ``share_caught_before_exit`` was on this page and is not any more.
        The evidence below is recomputed from the same ``metrics.json``
        the table is drawn from, so the justification cannot drift away
        from the data it cites: if a regenerated run stops supporting the
        retirement, these lines say so instead of repeating a slogan.
        """
        finite = [h for h in raw["horizons"] if h.get("cutoff_minutes") is not None]
        if not finite:  # pragma: no cover - ml always publishes finite horizons
            return []
        first, last = finite[0], finite[-1]

        def catch(entry: dict[str, Any], scope: str, key: str) -> float:
            return float(entry.get("ring_catch", {}).get(scope, {}).get(key, 0.0))

        in_flight = [
            (e["label"], catch(e, "cross_bank", "rings_money_not_yet_at_exit"))
            for e in finite
        ]
        # Ring total comes from the per-seed block, which is the only place
        # ml states it. Never a literal: a literal would keep saying "120"
        # after the generator changed.
        per_seed = first.get("ring_catch_per_seed", {}).get("cross_bank", {})
        total = max(
            (int(run.get("rings", 0)) for run in per_seed.values()), default=0
        )
        evidence = [
            f"At {first['label']} cross-bank share_caught_before_exit "
            f"({catch(first, 'cross_bank', 'share_caught_before_exit'):.4f}) equals "
            f"cross-bank ring_recall "
            f"({catch(first, 'cross_bank', 'ring_recall'):.4f}) to four decimals: "
            f"{catch(first, 'cross_bank', 'rings_money_not_yet_at_exit'):.0f} of "
            f"{total} exits have not happened yet, so the 'before exit' qualifier "
            "does nothing.",
            f"At {last['label']} the share is "
            f"{catch(last, 'single_bank', 'share_caught_before_exit'):.4f} for BOTH "
            "scopes while cross-bank ring_recall is "
            f"{catch(last, 'cross_bank', 'ring_recall'):.4f} and cross-bank PR-AUC is "
            f"{float(last['summary']['cross_bank']['pr_auc_mean']):.4f}. No ring "
            "still has money in flight, so the metric is structurally zero exactly "
            "where detection is strongest.",
            "It is bounded by the rings-still-in-flight count and tracks it ("
            + ", ".join(f"{label} {n:.0f}" for label, n in in_flight)
            + "). That measures the horizon grid, not the detector.",
        ]
        return [
            RetiredMetric(
                name="share_caught_before_exit",
                reason=(
                    "Retired from display. It is not a property of the detector: it "
                    "is the share of rings caught before cash-out among rings whose "
                    "money had not yet left, and that denominator collapses to zero "
                    "as the horizon grows. Ring recall and per-span coverage are "
                    "shown instead. The per-ring before/after-exit comparison on the "
                    "replay tab is unaffected - there it is a fact about one ring, "
                    "not a rate."
                ),
                evidence=evidence,
            )
        ]
