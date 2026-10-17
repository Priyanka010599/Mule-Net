"""Endpoint shape tests for the MuleNet UI.

Two things are checked: that every endpoint returns the shape
``ui/models.py`` documents, and that the snapshot the app serves still
validates against ``shared/contracts.py``. The second is the one that
matters over time -- if the architect renames a contract field, these
tests fail here rather than the browser rendering ``undefined``.

Every test is skipped, not failed, when the snapshot has not been built.
A fresh clone has no ``ui/out/demo_snapshot.json`` (it is gitignored
derived data) and a missing artefact is an operator step, not a defect.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from shared.contracts import (
    Account,
    Investigation,
    Ring,
    RingExplanation,
    RingMember,
    RiskScore,
    StrReport,
    Transfer,
    case_id_for_ring,
)
from ui.build_snapshot import _cross_bank_alone_in_time
from ui.chat import AGENTS, MockRouter, route
from ui.config import CHAT_BACKENDS, UiConfig
from ui.datasource import DataSource, SnapshotMissingError
from ui.main import app
from ui.models import (
    BankTile,
    ChatRequest,
    CoverageSpan,
    DemoPlan,
    EnclaveOutboxProof,
    EnclavePayload,
    HorizonMetrics,
    LeadStats,
    MatchAmbiguity,
    MetricsPayload,
    RedteamPayload,
    RedteamVerdict,
    ReplayHop,
    ReplayPayload,
    RetiredMetric,
    RingCoverage,
    RingDetail,
    RingMatchInfo,
    RingMemberView,
    ScopeReplay,
    SeedCoverage,
)

CFG = UiConfig.from_env()

_needs_snapshot = pytest.mark.skipif(
    not CFG.snapshot_path.exists(),
    reason="run `python -m ui.build_snapshot` first",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def data() -> DataSource:
    return DataSource(CFG)


# ── config ───────────────────────────────────────────────────────────


def test_config_defaults_are_inside_the_repo() -> None:
    assert CFG.chat_backend in CHAT_BACKENDS
    assert CFG.max_graph_nodes > 0
    assert CFG.graph_hops >= 1
    assert CFG.headline_horizons == (5, 10, 30)
    assert CFG.ml_out_dir.name == "out"


def test_config_rejects_an_unknown_chat_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "gemini-pro")
    with pytest.raises(ValueError, match="MULENET_UI_CHAT_BACKEND"):
        UiConfig.from_env()


def test_missing_snapshot_raises_a_directive_error(tmp_path) -> None:
    cfg = UiConfig.from_env()
    broken = DataSource(
        UiConfig(
            **{**cfg.__dict__, "snapshot_path": tmp_path / "nope.json"}
        )
    )
    with pytest.raises(SnapshotMissingError, match="ui.build_snapshot"):
        broken.overview()


# ── liveness and shell ───────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_the_probe_is_not_on_a_path_cloud_run_reserves(client: TestClient) -> None:
    """``/healthz`` never reaches a container on ``*.run.app``.

    Cloud Run's front end answers paths ending in ``z`` itself, so a probe
    there is permanently unhealthy however well the app runs. The route
    must not come back under that name.
    """
    assert client.get("/healthz").status_code == 404


def test_index_serves_the_shell_and_loads_cytoscape_from_a_cdn(
    client: TestClient,
) -> None:
    body = client.get("/").text
    assert '<div id="cy"' in body
    assert 'id="bank-grid"' in body
    assert "cdnjs.cloudflare.com/ajax/libs/cytoscape/" in body
    # Cytoscape must not be vendored into the repo.
    assert not (CFG.static_dir / "cytoscape.min.js").exists()


# ── overview ─────────────────────────────────────────────────────────


@_needs_snapshot
def test_overview_shape(client: TestClient) -> None:
    body = client.get("/api/overview").json()
    assert body["dataset"]["accounts"] == 30000
    assert body["dataset"]["banks"] == ["BANK_A", "BANK_B", "BANK_C"]
    assert body["mock"] is False, "the snapshot is built from ml/out, not mock data"
    assert body["detector"]["score_model_version"].startswith("mulenet-xgboost")
    assert body["detector"]["detector_version"].startswith("mulenet-velocity-rule")
    assert body["featured_ring_id"] in {r["ring_id"] for r in body["rings"]}
    assert body["bounding"]["max_graph_nodes"] == CFG.max_graph_nodes


@_needs_snapshot
def test_ring_index_is_ordered_widest_span_first(client: TestClient) -> None:
    rings = client.get("/api/overview").json()["rings"]
    spans = [r["span"] for r in rings]
    assert spans == sorted(spans, reverse=True)
    assert {1, 2, 3} <= set(spans), "all three strata must be reachable in the demo"


@_needs_snapshot
def test_featured_ring_is_the_one_the_argument_rests_on(client: TestClient) -> None:
    """Featured is the demo's primary pick, chosen on detection timing.

    It used to be the widest, largest ring. That is no longer the rule and
    must not be reinstated by accident: a ring is featured because
    cross-bank catches it and single-bank does not, which can perfectly
    well be a 2-bank ring with few members. What still holds is that it
    must be multi-bank (a one-bank ring cannot make the point) and must
    cross a bank boundary.
    """
    body = client.get("/api/overview").json()
    featured = next(r for r in body["rings"] if r["ring_id"] == body["featured_ring_id"])
    assert body["demo"]["primary"]["ring_id"] == body["featured_ring_id"]
    assert featured["span"] >= 2
    assert featured["cross_bank_transfer_count"] > 0


# ── bank tiles ───────────────────────────────────────────────────────


@_needs_snapshot
def test_banks_return_three_tiles_holding_only_their_own_accounts(
    client: TestClient,
) -> None:
    tiles = client.get("/api/banks").json()
    assert [t["bank_id"] for t in tiles] == ["BANK_A", "BANK_B", "BANK_C"]
    for tile in tiles:
        assert tile["account_count"] == 10000
        assert all(a["bank_id"] == tile["bank_id"] for a in tile["alerts"])
        assert tile["transfers_leaving_perimeter"] < tile["transfers_seen"]
        assert tile["queue"]["queue_depth"] >= tile["queue"]["queue_true_positives"]
        # The queue is built on the cross-bank model, and the tile must
        # say so rather than implying a single-bank score exists.
        assert "single-bank" in tile["queue_caveat"]


@_needs_snapshot
def test_bank_alerts_are_ranked_by_descending_score(client: TestClient) -> None:
    for tile in client.get("/api/banks").json():
        scores = [a["score"] for a in tile["alerts"]]
        assert scores == sorted(scores, reverse=True)


@_needs_snapshot
def test_bank_fragments_of_a_ring_partition_its_members(
    client: TestClient, data: DataSource
) -> None:
    ring_id = data.overview().featured_ring_id
    ring = data.ring(ring_id)
    tiles = client.get("/api/banks", params={"ring_id": ring_id}).json()

    visible: list[str] = []
    for tile in tiles:
        visible.extend(tile["visible_member_hashed_account_ids"])
        seen = len(tile["visible_member_hashed_account_ids"])
        if seen:
            # No bank sees the whole ring: that is the whole argument.
            assert tile["invisible_member_count"] == ring.total_member_count - seen
    assert sorted(visible) == sorted(ring.all_member_hashed_account_ids)
    assert len(visible) == len(set(visible)), "an account belongs to exactly one bank"


@_needs_snapshot
def test_banks_reject_an_unknown_ring(client: TestClient) -> None:
    assert client.get("/api/banks", params={"ring_id": "nope"}).status_code == 404


# ── ring detail ──────────────────────────────────────────────────────


@_needs_snapshot
def test_ring_detail_shape(client: TestClient, data: DataSource) -> None:
    ring_id = data.overview().featured_ring_id
    body = client.get(f"/api/rings/{ring_id}").json()

    assert body["ring_id"] == ring_id
    assert body["span"] == len(body["banks_involved"])
    assert len(body["members"]) == body["total_member_count"]
    assert (
        body["alertable_member_count"] + body["context_member_count"]
        == body["total_member_count"]
    )
    assert body["true_positive_alertable_members"] <= body["alertable_member_count"]
    assert body["explanation"]["narrative"]
    assert body["explanation"]["top_factors"]
    # A report exists only where a verified case does. citations_verified
    # is True when agents/batch_draft_reports.py published a real Gemini
    # draft for this case (ui/build_snapshot.py:_load_str_reports), False
    # for the synthesized PLACEHOLDER_CITATIONS fallback -- either way
    # there must be citations, never an empty list.
    if body["case"] is None:
        assert body["report"] is None
    else:
        assert body["report"]["citations_verified"] in (True, False)
        assert body["report"]["regulation_citations"]
    assert set(body["single_bank_view"]) == set(body["banks_involved"])


@_needs_snapshot
def test_ring_subgraph_is_bounded(client: TestClient, data: DataSource) -> None:
    """30,000 accounts cannot go into Cytoscape; every ring must be capped."""
    overview = data.overview()
    for summary in overview.rings:
        ring = data.ring(summary.ring_id)
        nodes = len(ring.members) + len(ring.context_accounts)
        assert nodes <= ring.bounding.max_graph_nodes, ring.ring_id
        assert nodes < overview.dataset.accounts

        node_ids = {m.hashed_account_id for m in ring.members} | {
            c.hashed_account_id for c in ring.context_accounts
        }
        # Every edge must land inside the node set, or Cytoscape drops it.
        for transfer in ring.transfers:
            assert transfer.from_hashed_account_id in node_ids
            assert transfer.to_hashed_account_id in node_ids


@_needs_snapshot
def test_intra_ring_edges_are_kept_in_full(data: DataSource) -> None:
    """The money trail is what the demo animates; it must not be truncated."""
    ring = data.ring(data.overview().featured_ring_id)
    members = set(ring.all_member_hashed_account_ids)
    for transfer in ring.transfers:
        expected = (
            transfer.from_hashed_account_id in members
            and transfer.to_hashed_account_id in members
        )
        assert transfer.intra_ring is expected
    intra = [t for t in ring.transfers if t.intra_ring]
    assert len(intra) == ring.intra_ring_transfer_count
    assert sum(1 for t in intra if t.cross_bank) == ring.cross_bank_transfer_count


@_needs_snapshot
def test_edge_cap_applies_to_context_edges_only(data: DataSource) -> None:
    """The edge cap bounds context, never the money trail.

    Intra-ring hops are kept in full, so the rendered edge total may
    exceed ``max_transfers_per_ring``. The UI must state the bound that
    way rather than claiming a cap it does not enforce.
    """
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        intra = sum(1 for t in ring.transfers if t.intra_ring)
        context = len(ring.transfers) - intra
        assert context <= ring.bounding.max_context_edges, ring.ring_id
        assert len(ring.transfers) <= ring.bounding.max_transfers_per_ring, ring.ring_id


@_needs_snapshot
def test_no_context_account_is_stranded_without_an_edge(data: DataSource) -> None:
    """An unconnected node is a dot that tells the viewer nothing."""
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        touched = set()
        for transfer in ring.transfers:
            touched.add(transfer.from_hashed_account_id)
            touched.add(transfer.to_hashed_account_id)
        for context in ring.context_accounts:
            assert context.hashed_account_id in touched, ring.ring_id


@_needs_snapshot
def test_evidence_never_double_counts_identity_reuse(data: DataSource) -> None:
    """Reused ID number and matching photo are one fact, not two.

    They fire on identical accounts in this dataset (Jaccard 1.000), so
    an STR that cited both would overstate the corroboration.
    """
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        # One finding per (kind, scope). Scope stays in the key: same-bank
        # reuse is household noise and must never merge into the
        # cross-bank finding an STR would cite.
        keys = [(e.kind, e.scope) for e in ring.evidence]
        assert len(keys) == len(set(keys)), ring.ring_id
        for item in ring.evidence:
            assert item.scope in {"cross_bank", "single_bank"}
            assert item.cross_bank == (item.scope == "cross_bank")
            assert item.accounts_with_this == len(item.ring_member_hashed_account_ids)
            if len(item.raw_evidence_types) > 1:
                assert item.kind == "identity_reuse"
                assert item.note, "a collapsed finding must explain why"
            if not item.cross_bank:
                assert "context only" in item.note.lower(), (
                    "same-bank reuse must be labelled context, not corroboration"
                )


@_needs_snapshot
def test_unknown_ring_is_404(client: TestClient) -> None:
    assert client.get("/api/rings/ring-does-not-exist").status_code == 404


# ── shared/ contract conformance ─────────────────────────────────────


@_needs_snapshot
def test_snapshot_rows_validate_against_shared_contracts(data: DataSource) -> None:
    """Re-parse served rows through the real contracts, not lookalikes."""
    ring = data.ring(data.overview().featured_ring_id)

    parsed_ring = Ring(
        ring_id=ring.ring_id,
        members=[
            RingMember(
                hashed_account_id=m.hashed_account_id,
                membership_basis=m.membership_basis,
                alertable=m.alertable,
                link_scope=m.link_scope,
                linked_from_hashed_account_id=m.linked_from_hashed_account_id,
                link_evidence_type=m.link_evidence_type,
                reason=m.reason,
            )
            for m in ring.members
        ],
        banks_involved=ring.banks_involved,
        detected_at=ring.detected_at,
        status=ring.status,
    )
    # The served populations must match what the contract derives from
    # the same rows -- not a list the UI assembled on its own.
    assert parsed_ring.all_member_ids == ring.all_member_hashed_account_ids
    assert parsed_ring.alertable_member_ids == ring.alertable_member_hashed_account_ids
    assert parsed_ring.context_member_ids == ring.context_member_hashed_account_ids
    assert parsed_ring.velocity_member_ids == ring.velocity_member_hashed_account_ids

    for member in ring.members:
        account = Account(
            hashed_account_id=member.hashed_account_id,
            bank_id=member.bank_id,
            opened_at=member.opened_at,
            account_type=member.account_type,
            status=member.status,
        )
        assert account.hashed_account_id == member.hashed_account_id
        score = RiskScore(
            hashed_account_id=member.hashed_account_id,
            score=member.risk_score,
            model_version=member.model_version,
            top_factors=member.top_factors,
        )
        assert 0.0 <= score.score <= 1.0

    for transfer in ring.transfers[:50]:
        parsed = Transfer(
            transfer_id=transfer.transfer_id,
            from_hashed_account_id=transfer.from_hashed_account_id,
            to_hashed_account_id=transfer.to_hashed_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
            timestamp=transfer.timestamp,
            channel=transfer.channel,
        )
        # MoneyAmount is Decimal-backed and serializes to 2dp as a string.
        assert isinstance(parsed.amount, Decimal)
        assert parsed.model_dump(mode="json")["amount"] == transfer.amount

    explanation = RingExplanation(
        ring_id=ring.explanation.ring_id,
        top_factors=ring.explanation.top_factors,
        narrative=ring.explanation.narrative,
        generated_at=ring.explanation.generated_at,
    )
    assert explanation.ring_id == ring.ring_id

    if ring.case is None:
        assert ring.report is None
        return
    case = Investigation(
        case_id=ring.case.case_id,
        ring_id=ring.case.ring_id,
        verified_at=ring.case.verified_at,
        verification_attempts=ring.case.verification_attempts,
        verified_claim_count=ring.case.verified_claim_count,
        analyst_notes=ring.case.analyst_notes,
        status=ring.case.status,
        opened_at=ring.case.opened_at,
        closed_at=ring.case.closed_at,
    )
    # The case id is derived from the ring, not minted: one ring, one case.
    assert case.case_id == case_id_for_ring(ring.ring_id)
    assert ring.report is not None
    report = StrReport(
        report_id=ring.report.report_id,
        case_id=ring.report.case_id,
        narrative=ring.report.narrative,
        regulation_citations=ring.report.regulation_citations,
        status=ring.report.status,
    )
    assert report.case_id == case.case_id


@_needs_snapshot
def test_only_hashed_account_ids_are_served(data: DataSource) -> None:
    """CLAUDE.md: only hashed account IDs anywhere."""
    ring = data.ring(data.overview().featured_ring_id)
    for member in ring.members:
        assert len(member.hashed_account_id) == 64
        assert all(c in "0123456789abcdef" for c in member.hashed_account_id)


# ── metrics ──────────────────────────────────────────────────────────


@_needs_snapshot
def test_metrics_expose_the_span_gradient_not_one_number(client: TestClient) -> None:
    body = client.get("/api/metrics").json()
    assert body["source"] in {"ml", "fallback"}

    horizons = {h["cutoff_minutes"]: h for h in body["horizons"]}
    assert {5, 10, 30} <= set(horizons)
    assert horizons[5]["headline"] is True
    assert horizons[None]["headline"] is False, "full window is ceiling-bound"

    early = horizons[5]
    assert early["cross_bank"]["pr_auc"] > early["single_bank"]["pr_auc"]
    assert early["cross_bank"]["false_alerts_per_catch"] < early[
        "single_bank"
    ]["false_alerts_per_catch"]

    spans = {s["span"]: s for s in early["spans"]}
    assert set(spans) == {1, 2, 3}
    # The dose-response gradient: the more the ring is split across
    # banks, the more joining the data up is worth.
    assert spans[1]["pr_auc_delta"] < spans[2]["pr_auc_delta"] < spans[3]["pr_auc_delta"]


@_needs_snapshot
def test_metrics_reproduce_the_published_experiment_exactly(
    client: TestClient, data: DataSource
) -> None:
    """The API must transcribe ml/'s numbers, not approximate them.

    Asserting hard-coded literals here would turn a legitimate ml/
    re-run into a UI failure, so what is pinned instead is that every
    value served equals the value in the configured metrics file. That
    catches the failure that actually matters -- a wrong key, a lost
    sign, a single/cross swap -- without going stale.
    """
    payload = client.get("/api/metrics").json()
    source_path = (
        CFG.metrics_path if CFG.metrics_path.exists() else CFG.metrics_fallback_path
    )
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    raw_by_cutoff = {h.get("cutoff_minutes"): h for h in raw["horizons"]}

    assert payload["model"] == raw["model"]
    for horizon in payload["horizons"]:
        source = raw_by_cutoff[horizon["cutoff_minutes"]]
        for scope in ("single_bank", "cross_bank"):
            assert horizon[scope]["pr_auc"] == pytest.approx(
                source["summary"][scope]["pr_auc_mean"]
            )
            assert horizon[scope]["false_alerts_per_catch"] == pytest.approx(
                source["summary"][scope][
                    "false_alerts_per_catch_at_80pct_recall_mean"
                ]
            )
        # The delta must be derived, never restated independently.
        assert horizon["pr_auc_delta"] == pytest.approx(
            round(
                source["summary"]["cross_bank"]["pr_auc_mean"]
                - source["summary"]["single_bank"]["pr_auc_mean"],
                4,
            )
        )
        for span in horizon["spans"]:
            key = f"span{span['span']}"
            strata = source["strata_summary"]
            assert span["single_bank"]["pr_auc"] == pytest.approx(
                strata["single_bank"][key]["pr_auc_mean"]
            )
            assert span["cross_bank"]["pr_auc"] == pytest.approx(
                strata["cross_bank"][key]["pr_auc_mean"]
            )


@_needs_snapshot
def test_the_headline_horizon_carries_the_argument(client: TestClient) -> None:
    """Sanity-check the shape of the claim the demo actually makes."""
    early = next(
        h for h in client.get("/api/metrics").json()["horizons"]
        if h["cutoff_minutes"] == 5
    )
    # Cross-bank wins on ranking and on analyst burden at T=5m.
    assert early["cross_bank"]["pr_auc"] > early["single_bank"]["pr_auc"]
    assert (
        early["cross_bank"]["false_alerts_per_catch"]
        < early["single_bank"]["false_alerts_per_catch"]
    )
    spans = {s["span"]: s["pr_auc_delta"] for s in early["spans"]}
    # A one-bank ring is already fully visible to its own bank, so the
    # gain there must be near zero -- if it were large, the comparison
    # would be measuring something other than shared visibility.
    assert abs(spans[1]) < 0.05
    assert spans[3] > spans[2] > spans[1]


# ── chat ─────────────────────────────────────────────────────────────


def test_router_dispatches_on_intent() -> None:
    assert route("trace the money flow") == "graph_analyst"
    assert route("what document reuse is there?") == "kyc_analyst"
    assert route("draft the STR report") == "compliance_writer"
    assert route("how does PR-AUC compare?") == "metrics_analyst"
    assert route("hello") == "router"
    assert set(AGENTS) >= {"router", "graph_analyst", "kyc_analyst", "compliance_writer"}


def test_chat_without_a_ring_asks_for_one() -> None:
    reply = MockRouter().reply(ChatRequest(message="what is this?"), None)
    assert reply.backend == "mock"
    assert "Select a ring" in reply.reply


def test_screen_endpoint_is_reachable_and_mock_backed(client: TestClient) -> None:
    """No agents/ configured in this test app, so it must answer from the mock."""
    body = client.get("/api/screen/acct-anything").json()
    assert body["hashed_account_id"] == "acct-anything"
    assert body["flagged"] is False
    assert body["backend"] == "mock"


def test_simulate_endpoints_are_reachable_and_mock_backed(client: TestClient) -> None:
    """No agents/ configured in this test app: the mock has no offline
    detector to run a live pattern through, and must say so rather than
    inventing a caught/missed verdict (see ui.chat.MockRouter.simulate)."""
    bounds = client.get("/api/simulate/bounds").json()
    assert bounds["backend"] == "mock"
    assert bounds["offline_demo"] is True
    assert bounds["fields"] == {}

    verdict = client.post("/api/simulate", json={"collectors": 2}).json()
    assert verdict["backend"] == "mock"
    assert verdict["offline_demo"] is True
    assert verdict["caught"] is False
    assert verdict["caught_by"] == "unavailable"


def test_subset_endpoints_are_reachable_and_mock_backed(client: TestClient) -> None:
    """No agents/ configured in this test app: the mock has no offline
    feature builder or detector to run a subset through, and must say so
    rather than inventing stats or a flag count (see
    ui.chat.MockRouter.subset_load / subset_score)."""
    options = client.get("/api/subset/options").json()
    assert options["backend"] == "mock"
    assert options["offline_demo"] is True
    assert options["horizons"] == []

    loaded = client.post("/api/subset/load", json={"horizon": "24h", "bank": "all"}).json()
    assert loaded["backend"] == "mock"
    assert loaded["offline_demo"] is True
    assert loaded["transfers_considered"] == 0

    scored = client.post("/api/subset/score", json={"horizon": "24h", "bank": "all"}).json()
    assert scored["backend"] == "mock"
    assert scored["offline_demo"] is True
    assert scored["cross_bank_flagged"] == 0
    assert scored["flagged_accounts"] == []


def test_subset_endpoints_default_horizon_and_bank_when_omitted(
    client: TestClient,
) -> None:
    """``ui/main.py`` fills in ``24h``/``all`` when the body omits them,
    matching agents.subset's own defaults."""
    loaded = client.post("/api/subset/load", json={}).json()
    assert loaded["horizon"] == "24h"
    assert loaded["bank"] == "all"


@_needs_snapshot
def test_chat_answers_from_the_loaded_ring(client: TestClient, data: DataSource) -> None:
    ring_id = data.overview().featured_ring_id
    ring = data.ring(ring_id)
    body = client.post(
        "/api/chat", json={"message": "trace the money flow", "ring_id": ring_id}
    ).json()
    assert body["agent"] == "graph_analyst"
    assert body["backend"] == "mock"
    # Numbers in the answer must be numbers the case panel also shows.
    assert str(ring.intra_ring_transfer_count) in body["reply"]
    assert str(ring.cross_bank_transfer_count) in body["reply"]


@_needs_snapshot
def test_chat_report_answer_matches_the_report_s_own_verification_state(
    client: TestClient, data: DataSource
) -> None:
    """The "do not file" warning tracks citations_verified, not any fixed
    ring. Once agents/batch_draft_reports.py has drafted a real, cited STR
    for a case, ui/chat.py correctly stops warning on it -- so this reads
    the featured ring's own report state rather than assuming it unverified.
    """
    ring_id = data.overview().featured_ring_id
    citations_verified = client.get(f"/api/rings/{ring_id}").json()["report"]["citations_verified"]
    body = client.post(
        "/api/chat", json={"message": "draft the report", "ring_id": ring_id}
    ).json()
    assert body["agent"] == "compliance_writer"
    assert body["citations"]
    assert ("do not file" in body["reply"].lower()) != citations_verified


def test_chat_report_answer_carries_the_unverified_warning(tmp_path) -> None:
    """Independent of the live snapshot's own state -- agents/ may have
    drafted and verified every ring's STR by the time this runs -- force
    one ring's report back to unverified and check ui/chat.py still warns
    on it. Same snapshot-mutation pattern as
    test_a_ring_without_a_verified_case_renders_no_case below.
    """
    cfg = UiConfig.from_env()
    raw = json.loads(cfg.snapshot_path.read_text(encoding="utf-8"))
    featured_id = next(r["ring_id"] for r in raw["rings"] if r.get("report"))
    for ring in raw["rings"]:
        if ring["ring_id"] == featured_id:
            ring["report"]["citations_verified"] = False
            ring["report"]["regulation_citations"] = [
                "FATF Recommendation 20 - Reporting of suspicious transactions"
            ]

    stripped = tmp_path / "unverified.json"
    stripped.write_text(json.dumps(raw), encoding="utf-8")
    source = DataSource(UiConfig(**{**cfg.__dict__, "snapshot_path": stripped}))

    ring = source.ring(featured_id)
    assert ring.report is not None
    assert ring.report.citations_verified is False
    reply = MockRouter().reply(ChatRequest(message="draft the report"), ring)
    assert "do not file" in reply.reply.lower()
    assert reply.citations


@_needs_snapshot
def test_chat_evidence_answer_refuses_to_call_reuse_proof(
    client: TestClient, data: DataSource
) -> None:
    # Ask about a ring that HAS cross-bank reuse. The featured ring is
    # now chosen for its detection timing, not its evidence, and may
    # legitimately carry none -- in which case the answer is "there is
    # none", not a caveat about proof. Deriving the ring keeps this test
    # about the caveat rather than about which ring happens to be
    # featured.
    ring_id = next(
        summary.ring_id
        for summary in data.overview().rings
        if summary.cross_bank_evidence_count
    )
    body = client.post(
        "/api/chat", json={"message": "what device sharing is there?", "ring_id": ring_id}
    ).json()
    assert body["agent"] == "kyc_analyst"
    assert "not proof" in body["reply"].lower()


def test_chat_rejects_an_empty_message(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422


def test_chat_rejects_an_unknown_ring(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "hi", "ring_id": "nope"})
    assert response.status_code == 404


# ── demo mode ────────────────────────────────────────────────────────


@_needs_snapshot
def test_demo_reset_selects_the_featured_ring(client: TestClient) -> None:
    body = client.post("/api/demo/reset").json()
    assert body["featured_ring_id"] == client.get("/api/overview").json()["featured_ring_id"]
    assert body["replay_transfer_count"] > 0
    assert body["replay_duration_ms"] > 0
    assert "bank" in body["reason"]


# ── membership basis: alert against context ──────────────────────────


@_needs_snapshot
def test_every_member_carries_its_own_provenance(data: DataSource) -> None:
    """A linked relative must never be indistinguishable from a mule."""
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        for member in ring.members:
            assert member.membership_basis in {"velocity", "identity_link"}
            if member.membership_basis == "velocity":
                # A detection about the account's own behaviour.
                assert member.alertable is True
                assert member.link_scope is None
                assert member.linked_from_hashed_account_id is None
            else:
                # A lead. It must name the link it arrived by and say why.
                assert member.link_scope in {"cross_bank", "single_bank"}
                assert member.linked_from_hashed_account_id
                assert member.link_evidence_type
                assert member.reason, "a lead must always carry its qualifier"
                # Stage-2 alerting is cross-bank only.
                assert member.alertable == (member.link_scope == "cross_bank")


@_needs_snapshot
def test_populations_partition_the_members(data: DataSource) -> None:
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        alertable = set(ring.alertable_member_hashed_account_ids)
        context = set(ring.context_member_hashed_account_ids)
        assert not alertable & context
        assert alertable | context == set(ring.all_member_hashed_account_ids)
        assert len(alertable) == ring.alertable_member_count
        assert len(context) == ring.context_member_count
        assert ring.total_member_count == len(ring.members)

        velocity = set(ring.velocity_member_hashed_account_ids)
        linked = set(ring.identity_link_member_hashed_account_ids)
        assert not velocity & linked
        assert velocity | linked == set(ring.all_member_hashed_account_ids)
        # Every velocity member is alertable; the reverse is not true.
        assert velocity <= alertable


@_needs_snapshot
def test_context_members_are_not_counted_as_detections(data: DataSource) -> None:
    """Precision is quoted over the accused population only."""
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        assert ring.true_positive_alertable_members <= ring.alertable_member_count
        # A bank tile's visible-alertable list never includes a context member.
        context = set(ring.context_member_hashed_account_ids)
        for view in ring.single_bank_view.values():
            assert not set(view.visible_alertable_hashed_account_ids) & context


@_needs_snapshot
def test_the_demo_shows_at_least_one_context_member(data: DataSource) -> None:
    """The solid/dashed distinction must be demonstrable, not theoretical."""
    total_context = sum(r.context_member_count for r in data.overview().rings)
    assert total_context > 0, "no ring in the snapshot exercises the context styling"


# ── cases: real, or honestly absent ──────────────────────────────────


@_needs_snapshot
def test_cases_come_from_agents_out_and_are_never_fabricated(
    data: DataSource,
) -> None:
    for summary in data.overview().rings:
        ring = data.ring(summary.ring_id)
        assert summary.has_case == (ring.case is not None)
        if ring.case is None:
            assert ring.report is None
            continue
        assert ring.case.case_id == case_id_for_ring(ring.ring_id)
        assert ring.case.verified_at
        # The retry budget that actually exists.
        assert 1 <= ring.case.verification_attempts <= 2
        assert ring.case.verified_claim_count >= 1
        assert ring.report is not None
        assert ring.report.case_id == ring.case.case_id


def test_a_ring_without_a_verified_case_renders_no_case(tmp_path) -> None:
    """Point the builder at an empty agents/out and the case must vanish.

    This is the branch the user ruling is about, and the live data has a
    case for every ring — so without forcing it, the "no case yet" path
    would never be exercised.
    """
    import json

    cfg = UiConfig.from_env()
    raw = json.loads(cfg.snapshot_path.read_text(encoding="utf-8"))
    for ring in raw["rings"]:
        ring["case"] = None
        ring["report"] = None
    stripped = tmp_path / "no_cases.json"
    stripped.write_text(json.dumps(raw), encoding="utf-8")

    source = DataSource(UiConfig(**{**cfg.__dict__, "snapshot_path": stripped}))
    for summary in source.overview().rings:
        assert summary.has_case is False
        ring = source.ring(summary.ring_id)
        assert ring.case is None
        assert ring.report is None
        # The rest of the panel still works: absence is not an error.
        assert ring.explanation.narrative
        assert ring.members

    ring = source.ring(source.overview().featured_ring_id)
    reply = MockRouter().reply(ChatRequest(message="draft the report"), ring)
    assert "no case yet" in reply.reply.lower()
    assert not reply.citations


@_needs_snapshot
def test_endpoints_emit_every_declared_field(client: TestClient, data: DataSource) -> None:
    """Catch a field the data layer passes but the model never declared.

    Pydantic ignores unknown keyword arguments, so a field added to the
    snapshot and forwarded by ``datasource.py`` but missing from the
    response model is dropped in silence and the browser reads
    ``undefined``. That is not hypothetical: it is how
    ``visible_alertable_hashed_account_ids`` went missing.
    """
    ring_id = data.overview().featured_ring_id

    tiles = client.get("/api/banks", params={"ring_id": ring_id}).json()
    for tile in tiles:
        missing = set(BankTile.model_fields) - set(tile)
        assert not missing, f"BankTile fields absent from the response: {missing}"

    ring = client.get(f"/api/rings/{ring_id}").json()
    missing = set(RingDetail.model_fields) - set(ring)
    assert not missing, f"RingDetail fields absent from the response: {missing}"
    for member in ring["members"]:
        member_missing = set(RingMemberView.model_fields) - set(member)
        assert not member_missing, member_missing


# ── replay: the flag markers against the exit ────────────────────────


def _replay_source_tables() -> tuple[object, object]:
    """The two ml/out/ tables the replay is built from."""
    import pandas as pd

    return (
        pd.read_parquet(CFG.ml_table_path("ring_detection_timeline")),
        pd.read_parquet(CFG.ml_table_path("ring_id_map")),
    )


_needs_ml_tables = pytest.mark.skipif(
    not CFG.ml_table_path("ring_detection_timeline").exists()
    or not CFG.ml_table_path("ring_id_map").exists(),
    reason="run `python -m ml.publish_all` first",
)


@_needs_snapshot
@_needs_ml_tables
def test_replay_reproduces_the_published_timeline_exactly() -> None:
    """Every replay value must be the parquet's value, not a derived one.

    The browser renders precomputed offsets and statuses. If the snapshot
    drifts from ``ring_detection_timeline.parquet`` the page is
    confidently wrong about who caught what and when, and nothing else
    would notice.
    """
    import pandas as pd

    timeline, ring_map = _replay_source_tables()
    matched = {
        row["detected_ring_id"]: row["true_ring_id"]
        for _, row in ring_map.iterrows()
        if not pd.isna(row["true_ring_id"])
    }
    source = DataSource(CFG)

    checked = 0
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        if not replay.available:
            continue
        true_ring_id = matched[summary.ring_id]
        assert replay.true_ring_id == true_ring_id
        rows = timeline[timeline["ring_id"] == true_ring_id]

        onset = pd.Timestamp(rows["onset_at"].iloc[0])
        exit_at = pd.Timestamp(rows["exit_reached_at"].iloc[0])
        assert pd.Timestamp(replay.onset_at) == onset
        assert pd.Timestamp(replay.exit_reached_at) == exit_at
        assert replay.exit_offset_minutes == pytest.approx(
            (exit_at - onset).total_seconds() / 60.0, abs=0.01
        )

        for scope in replay.scopes:
            scope_rows = rows[rows["scope"] == scope.scope]
            finite = scope_rows[scope_rows["T"].notna()].sort_values("T")
            hit = finite[finite["flagged"]]
            expected_t = None if hit.empty else int(hit["T"].iloc[0])
            assert scope.first_flag_T == expected_t, scope.scope
            # A NULL-T row flags every ring by construction, so it is
            # reported separately and never as the first flag.
            assert scope.flagged_full_window == bool(
                scope_rows[scope_rows["T"].isna()]["flagged"].iloc[0]
            )
            if expected_t is None:
                assert scope.status == "never_flagged"
                assert scope.before_exit is None
            else:
                assert scope.before_exit is (expected_t <= replay.exit_offset_minutes)
                assert scope.status == (
                    "flags_before_exit" if scope.before_exit else "flags_after_exit"
                )
                assert pd.Timestamp(scope.first_flag_at) == onset + pd.Timedelta(
                    minutes=expected_t
                )
        checked += 1
    assert checked, "no ring in the snapshot had a replay to check"


@_needs_snapshot
def test_replay_carries_both_scopes_and_never_hides_one() -> None:
    """Two markers always, so an absent one cannot mean "never flagged"."""
    source = DataSource(CFG)
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        if not replay.available:
            continue
        assert {s.scope for s in replay.scopes} == {"single_bank", "cross_bank"}
        for scope in replay.scopes:
            if scope.status == "never_flagged":
                # Said in words, not merely left off the axis.
                assert "not flagged" in scope.status_label
                assert scope.detail, "a never-flagged scope must say so in full"
                assert scope.flag_offset_minutes is None
            else:
                assert scope.status_label
                assert scope.flag_offset_minutes is not None


@_needs_snapshot
def test_replay_hops_are_ordered_and_inside_the_window() -> None:
    source = DataSource(CFG)
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        if not replay.available:
            continue
        offsets = [h.offset_minutes for h in replay.hops]
        assert offsets == sorted(offsets), summary.ring_id
        assert all(0.0 <= o <= replay.window_minutes for o in offsets)
        assert [h.index for h in replay.hops] == list(range(1, len(replay.hops) + 1))
        assert replay.hop_count == len(replay.hops)
        # The window must reach the exit, or the contrast the whole
        # feature is about would fall off the end of the axis.
        assert replay.window_minutes >= replay.exit_offset_minutes
        # Excluded hops are counted, never dropped in silence.
        if replay.hops_before_window or replay.hops_after_window:
            assert replay.window_note


@_needs_snapshot
def test_every_replay_names_its_run_and_its_exit(data: DataSource) -> None:
    for summary in data.overview().rings:
        replay = data.replay(summary.ring_id)
        assert replay.label == "Replay: seed 42 evaluation run"
        assert replay.exit_label == "money reaches the exit account"


# ── ambiguity: a cluster is not a ring ───────────────────────────────


@_needs_snapshot
@_needs_ml_tables
def test_an_ambiguous_cluster_carries_the_caveat_naming_every_true_ring() -> None:
    """The caveat must state N, the named ring, and the coverage.

    A detector cluster is a connected component over flagged accounts, so
    one cluster can hold parts of several true rings. A replay captioned
    with a single true ring id and no caveat tells a judge the cluster IS
    that ring.
    """
    _, ring_map = _replay_source_tables()
    source = DataSource(CFG)
    ambiguous = 0
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        row = ring_map[ring_map["detected_ring_id"] == summary.ring_id].iloc[0]
        assert replay.ambiguity is not None
        assert replay.ambiguity.is_ambiguous == bool(row["match_is_ambiguous"])
        # Also exposed on the ring index, so the picker can warn first.
        assert summary.match_is_ambiguous == bool(row["match_is_ambiguous"])
        if not row["match_is_ambiguous"]:
            assert replay.ambiguity.caveat == "", summary.ring_id
            continue
        ambiguous += 1
        caveat = replay.ambiguity.caveat
        assert f"parts of {int(row['true_rings_touched'])} true rings" in caveat
        assert str(row["true_ring_id"]) in caveat
        assert (
            f"{int(row['overlap_members'])} of {int(row['detected_ring_size'])} members"
            in caveat
        )
        # Never asserts the cluster is that one ring.
        assert "the replay shows" in caveat
        if bool(row["overlap_tied"]):
            assert "tied" in caveat
    assert ambiguous, "the snapshot must include an ambiguous cluster to exercise this"


@_needs_snapshot
@_needs_ml_tables
def test_a_cluster_with_no_true_ring_renders_no_replay_available() -> None:
    """The false-positive cluster must say why, not render blank or crash."""
    import pandas as pd

    _, ring_map = _replay_source_tables()
    orphans = [
        row["detected_ring_id"]
        for _, row in ring_map.iterrows()
        if pd.isna(row["true_ring_id"])
    ]
    assert orphans, "ring_id_map is expected to keep its false-positive cluster"
    source = DataSource(CFG)
    for ring_id in orphans:
        replay = source.replay(ring_id)
        assert replay.available is False
        assert replay.true_ring_id is None
        assert "No replay available" in replay.unavailable_reason
        assert "false-positive cluster" in replay.unavailable_reason
        # An empty panel is not an explanation. The scopes are absent
        # BECAUSE there is no ground truth, and the reason says so.
        assert replay.scopes == []
        assert replay.hops == []


@_needs_snapshot
def test_a_cluster_outside_the_snapshot_degrades_with_its_own_reason(
    client: TestClient, data: DataSource
) -> None:
    """Absent-from-snapshot and no-ground-truth must not read the same."""
    in_snapshot = {s.ring_id for s in data.overview().rings}
    replays = data.snapshot["replays"]
    outside = [
        ring_id
        for ring_id, replay in replays.items()
        if ring_id not in in_snapshot
        and "outside this snapshot" in replay["unavailable_reason"]
    ]
    assert outside, "MAX_RINGS is expected to leave some clusters out"
    response = client.get(f"/api/replay/{outside[0]}")
    # 200 with a reason, not a 404: "there is no replay for this" is an
    # answer the panel renders.
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "outside this snapshot" in payload["unavailable_reason"]
    assert "false-positive" not in payload["unavailable_reason"]
    assert client.get("/api/replay/ring-does-not-exist").status_code == 404


# ── demo mode: derived, and never an ambiguous ring ──────────────────


@_needs_snapshot
def test_demo_auto_selection_never_picks_an_ambiguous_ring(
    client: TestClient, data: DataSource
) -> None:
    """The one rule that cannot bend: an auto-pick must be nameable."""
    overview = data.overview()
    by_id = {s.ring_id: s for s in overview.rings}
    plan = overview.demo
    picked = [p for p in (plan.primary, plan.control) if p is not None]
    assert picked, "the demo must select at least one ring"
    for pick in picked:
        assert by_id[pick.ring_id].match_is_ambiguous is False, pick.ring_id
        replay = data.replay(pick.ring_id)
        assert replay.available is True
        assert replay.ambiguity is not None
        assert replay.ambiguity.is_ambiguous is False
        assert replay.ambiguity.caveat == ""
        assert pick.true_ring_id == replay.true_ring_id

    # The featured ring is what every entry point auto-selects, including
    # first page load, so it is held to the same rule.
    assert by_id[overview.featured_ring_id].match_is_ambiguous is False
    assert client.post("/api/demo/reset").json()["featured_ring_id"] == (
        overview.featured_ring_id
    )


@_needs_snapshot
def test_the_demo_picks_are_derived_not_hardcoded(data: DataSource) -> None:
    """Both picks must satisfy the criterion they claim, or say they do not.

    ``criterion_met`` false is a legitimate outcome and the point of the
    field: the failure mode worth guarding is a pick quietly relabelled
    to fit the narrative, so the label is checked against the numbers.
    """
    plan = data.demo_plan
    assert plan.label == "Replay: seed 42 evaluation run"
    assert "hard-coded" in plan.selection_note

    primary = plan.primary
    assert primary is not None
    # Multi-bank, not necessarily 3-bank: the pick is made on detection
    # timing, and a 2-bank ring can make the point better than a 3-bank
    # one. A 1-bank ring never can, which is what is actually asserted.
    assert primary.span >= 2
    assert primary.role == "primary"
    scopes = {s.scope: s for s in data.replay(primary.ring_id).scopes}
    # The one thing the primary must actually do: catch it in time.
    assert scopes["cross_bank"].status == "flags_before_exit"
    assert scopes["cross_bank"].caught_in_time is True
    # And the one thing single-bank must NOT do.
    assert scopes["single_bank"].caught_in_time is False
    assert primary.cross_bank_first_flag_T == scopes["cross_bank"].first_flag_T
    assert primary.single_bank_first_flag_T == scopes["single_bank"].first_flag_T
    assert primary.single_bank_status == scopes["single_bank"].status
    assert primary.cross_bank_status == scopes["cross_bank"].status
    # The criterion, recomputed here from the three published numbers
    # rather than trusted from the status strings the selection filtered
    # on: cross-bank's first flag before the exit, single-bank's after it
    # (or never). If the timeline's numbers ever stop supporting the
    # label, this fails and the page shows "criterion NOT met".
    replay = data.replay(primary.ring_id)
    assert primary.exit_offset_minutes == pytest.approx(replay.exit_offset_minutes, abs=0.01)
    holds = _cross_bank_alone_in_time(
        primary.cross_bank_first_flag_T,
        primary.single_bank_first_flag_T,
        primary.exit_offset_minutes,
    )
    assert primary.criterion_met is holds
    assert primary.honest_note
    if not holds:
        assert "NOT loosened" in primary.honest_note
    else:
        # A pick that passes still has to say what it is not claiming.
        assert "median lead is zero" in primary.honest_note

    secondary = plan.secondary
    if secondary is not None:
        assert secondary.role == "secondary"
        assert secondary.span >= 2
        assert secondary.ring_id != primary.ring_id
        sec_scopes = {s.scope: s for s in data.replay(secondary.ring_id).scopes}
        assert sec_scopes["single_bank"].caught_in_time is True
        assert sec_scopes["cross_bank"].caught_in_time is True
        assert (
            secondary.cross_bank_first_flag_T < secondary.single_bank_first_flag_T
        )
        # A lead-time example must not be sold as the headline: the median
        # lead across the table is zero.
        assert "median lead is zero" in secondary.honest_note.lower()

    control = plan.control
    assert control is not None
    assert control.span == 1
    assert control.role == "control"
    same_time = control.single_bank_first_flag_T == control.cross_bank_first_flag_T
    assert control.criterion_met is same_time
    assert control.honest_note
    if not same_time:
        assert "closest" in control.honest_note.lower()


@pytest.mark.parametrize(
    ("cross_T", "single_T", "exit_offset", "expected"),
    [
        # ring_014's real shape: cross-bank 4.2 min before the exit,
        # single-bank 170.8 min after it.
        (5, 180, 9.2, True),
        # Single-bank never fires in the grid: the second half holds a
        # fortiori.
        (5, None, 9.2, True),
        # The relabelling this criterion must still reject: single-bank
        # also catches it in time, so cross-bank was not alone.
        (5, 8, 9.2, False),
        # Cross-bank itself is late. No amount of single-bank lateness
        # rescues that.
        (30, 180, 9.2, False),
        # Cross-bank never fires at all.
        (None, 180, 9.2, False),
        # Flagging exactly at the exit is not flagging before it.
        (10, 180, 10.0, False),
    ],
)
def test_the_primary_criterion_is_checkable_in_both_directions(
    cross_T: int | None, single_T: int | None, exit_offset: float, expected: bool
) -> None:
    """The criterion must be failable, or it measures nothing.

    The predecessor of this criterion -- "single-bank flags only at the
    full window" -- was satisfied by no openable cluster, so every pick
    wore a failure badge and the label carried no information. The
    restatement describes what the data does show, and these cases pin
    down that it is a real property: a ring where single-bank is also in
    time fails it, and a ring where cross-bank is late fails it too.
    """
    assert _cross_bank_alone_in_time(cross_T, single_T, exit_offset) is expected



@_needs_snapshot
def test_the_metrics_provenance_line_names_the_file_actually_opened(
    tmp_path: pathlib.Path,
) -> None:
    """The early-detection header must not name a file it never opened.

    ``infra/cloudbuild.ignore`` excludes ``ml/experiments/`` from the
    container image, so in the deployed app the configured live path does
    not exist and ``ui/data/metrics_fallback.json`` is what gets read. The
    header used to say "read live from
    ml/experiments/early_detection/metrics.json" whenever ``source ==
    "ml"`` -- which is true on a dev box and false where the demo runs.
    Every number was right; the claim about where they came from was not.

    Both directions are checked, because a line that is honest only about
    the fallback would break the moment ``ml/`` did run.
    """
    deployed = DataSource(
        UiConfig(
            **{
                **CFG.__dict__,
                # Exactly what the container sees: ml/experiments/ absent.
                "metrics_path": tmp_path / "ml" / "experiments" / "metrics.json",
                "lead_metrics_path": tmp_path / "ml" / "experiments" / "lead.json",
            }
        )
    )
    payload = deployed.metrics()
    assert payload.source == "fallback"
    assert CFG.metrics_fallback_path.name in payload.source_detail
    assert "ml/ has not run here" in payload.source_detail
    # The precise failure being guarded: naming the file that is not there.
    assert "ml/experiments" not in payload.source_detail
    # The coverage block's line is built by the same helper, so the two
    # cannot drift into disagreeing about the same container.
    assert payload.coverage is not None
    assert "ml/experiments" not in payload.coverage.source

    live = DataSource(CFG)
    live_payload = live.metrics()
    if live_payload.source == "ml":
        assert live_payload.source_detail == (
            f"read live from {CFG.metrics_path.as_posix()}"
        )
    else:
        assert "ml/ has not run here" in live_payload.source_detail


@_needs_snapshot
def test_demo_reset_hands_back_the_walkthrough(client: TestClient) -> None:
    body = client.post("/api/demo/reset").json()
    plan = body["plan"]
    assert plan["primary"]["ring_id"] == body["featured_ring_id"]
    assert plan["control"]["ring_id"] != plan["primary"]["ring_id"]
    assert plan["primary"]["claim"] in body["reason"]


@_needs_snapshot
def test_replay_endpoint_emits_every_declared_field(
    client: TestClient, data: DataSource
) -> None:
    payload = client.get(f"/api/replay/{data.overview().featured_ring_id}").json()
    missing = set(ReplayPayload.model_fields) - set(payload)
    assert not missing, f"ReplayPayload fields absent from the response: {missing}"
    for scope in payload["scopes"]:
        assert not set(ScopeReplay.model_fields) - set(scope)
    for hop in payload["hops"]:
        assert not set(ReplayHop.model_fields) - set(hop)
    assert not set(MatchAmbiguity.model_fields) - set(payload["ambiguity"])
    overview = client.get("/api/overview").json()
    assert not set(DemoPlan.model_fields) - set(overview["demo"])
    assert not set(RingMatchInfo.model_fields) - set(overview["ring_match"])


# ── the horizon grid, and the three ways a scope can fail to catch ───


@_needs_snapshot
@_needs_ml_tables
def test_the_horizon_grid_is_read_off_the_table_not_written_down() -> None:
    """Extending the horizons in ml/ must widen the UI with no edit here.

    This is the test that would have caught the T=60 regression: the
    snapshot claimed a scope was never flagged at any measured horizon
    while the regenerated table flagged it at T=120.
    """
    timeline, _ = _replay_source_tables()
    published = sorted(int(t) for t in timeline["T"].dropna().unique())
    source = DataSource(CFG)
    seen = 0
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        assert replay.horizon_grid == published, summary.ring_id
        if not replay.available:
            continue
        for scope in replay.scopes:
            finite = [h.T for h in scope.horizons if h.T is not None]
            assert finite == published
            # Exactly one full-window row, and it is never counted as a
            # timed flag.
            assert [h.T for h in scope.horizons].count(None) == 1
        seen += 1
    assert seen
    # The lead-time experiment must be on the same grid as the timeline,
    # or the coverage table and the replay panel are describing different
    # runs while sitting two tabs apart.
    coverage = source.coverage()
    assert coverage is not None
    assert coverage.horizon_grid == published


@_needs_snapshot
@_needs_ml_tables
def test_the_three_failure_cases_never_collapse_into_one_phrase() -> None:
    """"flagged late", "full window only" and "never" are different facts.

    Each has to be reachable from the status alone and has to say
    something different on screen. Collapsing them is how a detection
    that arrived days after cash-out gets read as a catch.
    """
    import pandas as pd

    timeline, ring_map = _replay_source_tables()
    matched = {
        row["detected_ring_id"]: row["true_ring_id"]
        for _, row in ring_map.iterrows()
        if not pd.isna(row["true_ring_id"])
    }
    source = DataSource(CFG)
    seen: set[str] = set()
    for summary in source.overview().rings:
        replay = source.replay(summary.ring_id)
        if not replay.available:
            continue
        rows = timeline[timeline["ring_id"] == matched[summary.ring_id]]
        for scope in replay.scopes:
            scope_rows = rows[rows["scope"] == scope.scope]
            finite = scope_rows[scope_rows["T"].notna()]
            hit = finite[finite["flagged"]]
            full = bool(scope_rows[scope_rows["T"].isna()]["flagged"].iloc[0])

            if hit.empty:
                expected = (
                    "flagged_full_window_only" if full else "never_flagged"
                )
            elif int(hit["T"].iloc[0]) <= replay.exit_offset_minutes:
                expected = "flags_before_exit"
            else:
                expected = "flags_after_exit"
            assert scope.status == expected, (summary.ring_id, scope.scope)
            seen.add(expected)

            # Only an in-time flag counts as a catch.
            assert scope.caught_in_time is (expected == "flags_before_exit")

            if expected == "flags_before_exit":
                assert "BEFORE the exit" in scope.status_label
            elif expected == "flags_after_exit":
                # Flagged -- but the label must not read like a save.
                assert "AFTER the money left" in scope.status_label
                assert "BEFORE" not in scope.status_label
                assert "not a save" in scope.detail
            elif expected == "flagged_full_window_only":
                assert scope.status_label == (
                    "flagged only at the full window (after the money left)"
                )
                assert "not a catch" in scope.detail
                assert scope.flagged_full_window is True
            else:
                assert scope.status_label == "never flagged"
                assert scope.flagged_full_window is False

    assert "flags_before_exit" in seen
    assert "flags_after_exit" in seen, (
        "the snapshot must reach a ring flagged after its exit, or the "
        "late-flag wording is never exercised"
    )
    # The four labels are all distinct strings; no two cases share one.
    labels = {
        s.status: s.status_label
        for summary in source.overview().rings
        for s in source.replay(summary.ring_id).scopes
    }
    assert len(set(labels.values())) == len(labels)


@_needs_snapshot
@_needs_ml_tables
def test_a_flag_after_the_exit_is_never_called_a_catch() -> None:
    """Whatever the horizon grid, a late flag must not claim a save."""
    source = DataSource(CFG)
    late = [
        (summary.ring_id, scope)
        for summary in source.overview().rings
        for scope in source.replay(summary.ring_id).scopes
        if scope.status == "flags_after_exit"
    ]
    assert late, "no late-flagging ring in the snapshot to check"
    for ring_id, scope in late:
        assert scope.caught_in_time is False, ring_id
        assert scope.before_exit is False
        assert scope.first_flag_T is not None
        replay = source.replay(ring_id)
        assert scope.first_flag_T > replay.exit_offset_minutes


# ── metrics: share_caught_before_exit is retired, coverage headlines ──


@_needs_snapshot
def test_share_caught_before_exit_is_gone_from_the_payload(
    client: TestClient,
) -> None:
    """The retired metric must not survive anywhere in the response."""
    body = client.get("/api/metrics").json()
    blob = json.dumps(body)
    for horizon in body["horizons"]:
        assert "share_caught_before_exit_single_bank" not in horizon
        assert "share_caught_before_exit_cross_bank" not in horizon
    # It may be NAMED, in the retirement note. It may not be a value.
    assert blob.count("share_caught_before_exit") == sum(
        r["reason"].count("share_caught_before_exit")
        + r["name"].count("share_caught_before_exit")
        + sum(e.count("share_caught_before_exit") for e in r["evidence"])
        for r in body["retired"]
    )


@_needs_snapshot
def test_the_retirement_note_quotes_the_numbers_that_justify_it(
    data: DataSource,
) -> None:
    """The reason must be recomputed from metrics.json, not asserted.

    A hand-written rationale drifts away from the data the moment ml/
    regenerates. These lines are rebuilt from the same file the table is
    drawn from, so they cannot say something the numbers do not.
    """
    payload = data.metrics()
    retired = payload.retired
    assert len(retired) == 1
    note = retired[0]
    assert note.name == "share_caught_before_exit"
    assert len(note.evidence) == 3

    raw, _ = data._read_metrics()
    finite = [h for h in raw["horizons"] if h.get("cutoff_minutes") is not None]
    first, last = finite[0], finite[-1]

    def catch(entry, scope, key):
        return float(entry["ring_catch"][scope][key])

    # 1. At the first horizon the qualifier does nothing.
    assert f"{catch(first, 'cross_bank', 'share_caught_before_exit'):.4f}" in (
        note.evidence[0]
    )
    assert catch(first, "cross_bank", "share_caught_before_exit") == pytest.approx(
        catch(first, "cross_bank", "ring_recall"), abs=1e-4
    )
    # 2. At the last horizon it is zero where detection is strongest.
    assert catch(last, "single_bank", "share_caught_before_exit") == 0.0
    assert catch(last, "cross_bank", "share_caught_before_exit") == 0.0
    assert catch(last, "cross_bank", "ring_recall") > 0.9
    assert f"{catch(last, 'cross_bank', 'ring_recall'):.4f}" in note.evidence[1]
    # 3. It tracks the rings-still-in-flight count, which ends at zero.
    assert catch(last, "cross_bank", "rings_money_not_yet_at_exit") == 0.0
    assert "horizon grid, not the detector" in note.evidence[2]
    # The replay tab's per-ring comparison is explicitly NOT retired.
    assert "replay tab is unaffected" in note.reason


_LEAD_METRICS_PATH = (
    "seed_mean.by_scope_and_span.{scope}.span{span}.share_with_positive_warning"
)

_needs_lead_metrics = pytest.mark.skipif(
    not CFG.lead_metrics_path.exists() and not CFG.lead_metrics_fallback_path.exists(),
    reason="run `python -m ml.leadtime` first",
)


def _lead_metrics() -> dict:
    path = (
        CFG.lead_metrics_path
        if CFG.lead_metrics_path.exists()
        else CFG.lead_metrics_fallback_path
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(raw: dict, dotted: str):
    """Walk a dotted path the way docs/claims.md cites one.

    Every step is a NAMED key. Nothing here indexes a list by position,
    because a positional lookup against a reordered object reports one
    span's numbers under another span's label, and the span that would
    quietly go wrong is the control.
    """
    node = raw
    for part in dotted.split("."):
        assert isinstance(node, dict), f"{part}: not an object in {dotted}"
        assert part in node, f"missing key {part!r} in {dotted}"
        node = node[part]
    return node


@_needs_snapshot
@_needs_lead_metrics
def test_coverage_resolves_from_the_lead_time_metrics_by_identity(
    data: DataSource,
) -> None:
    """Displayed coverage must BE the published number, keyed by name.

    Two failure modes this is aimed at. First drift: if ml/ regenerates
    and the rates move, the page must move with them, so the expected
    values are read from the file rather than written down here. Second
    mis-keying: an ml agent hit a real bug this session from a
    position-keyed lookup, and the stratum that silently goes wrong is
    the 1-bank control -- the one result whose whole job is to be
    unflattering.
    """
    raw = _lead_metrics()
    coverage = data.coverage()
    assert coverage is not None
    assert coverage.generated_at == raw["generated_at"]
    assert coverage.horizon_grid == [int(t) for t in raw["horizon_grid"]]
    # The page states where its numbers come from, in the same notation
    # docs/claims.md uses.
    assert "by_scope_and_span" in coverage.source_path
    assert "share_with_positive_warning" in coverage.source_path

    seen_spans = set()
    for block in coverage.spans:
        seen_spans.add(block.span)
        for scope, shown_rate, shown_rings in (
            ("single_bank", block.single_bank_rate, block.single_bank_rings),
            ("cross_bank", block.cross_bank_rate, block.cross_bank_rings),
        ):
            published = _resolve(
                raw, _LEAD_METRICS_PATH.format(scope=scope, span=block.span)
            )
            assert shown_rate == pytest.approx(float(published), abs=5e-5), (
                scope,
                block.span,
            )
            group = _resolve(
                raw, f"seed_mean.by_scope_and_span.{scope}.span{block.span}"
            )
            # A rate never travels without the count behind it.
            assert shown_rings == pytest.approx(
                float(group["rings_with_positive_warning"]), abs=5e-3
            )
            assert block.rings == pytest.approx(float(group["rings"]), abs=5e-3)
            assert shown_rate == pytest.approx(shown_rings / block.rings, abs=5e-4)

        assert block.delta == pytest.approx(
            block.cross_bank_rate - block.single_bank_rate, abs=5e-5
        )

        # The per-seed column is a DIFFERENT run and must resolve from the
        # per-seed block, never from the mean.
        if block.per_seed is not None:
            seed_group = _resolve(
                raw,
                f"per_seed.{block.per_seed.seed}.by_scope_and_span"
                f".single_bank.span{block.span}",
            )
            assert block.per_seed.single_bank_rings == int(
                seed_group["rings_with_positive_warning"]
            )
            assert block.per_seed.single_bank_rate == pytest.approx(
                float(seed_group["share_with_positive_warning"]), abs=5e-5
            )

    assert seen_spans == {1, 2, 3}

    # Mis-keying guard: each span's rates must be the ones filed under
    # THAT span. Swapping any two would break at least one of these.
    published_single = [
        float(_resolve(raw, _LEAD_METRICS_PATH.format(scope="single_bank", span=n)))
        for n in (1, 2, 3)
    ]
    assert [round(b.single_bank_rate, 4) for b in coverage.spans] == [
        round(v, 4) for v in published_single
    ]

    # Overall totals come from by_scope, not from re-adding the strata.
    assert coverage.cross_bank_rate == pytest.approx(
        float(_resolve(raw, "seed_mean.by_scope.cross_bank.share_with_positive_warning")),
        abs=5e-5,
    )
    assert coverage.single_bank_rate == pytest.approx(
        float(
            _resolve(raw, "seed_mean.by_scope.single_bank.share_with_positive_warning")
        ),
        abs=5e-5,
    )


@_needs_snapshot
@_needs_lead_metrics
def test_coverage_is_the_before_exit_measure_not_the_loose_one(
    data: DataSource,
) -> None:
    """The page must use "warned in time", not "warned eventually".

    The looser rate -- flagged at any horizon, exit ignored -- is 1.000
    for both scopes at span 1 and so erases the control. It would also
    contradict the replay tab, which already refuses to call a post-exit
    flag a catch. The definition is stated on the page so nobody
    re-derives the other one.
    """
    coverage = data.coverage()
    assert coverage is not None
    assert "BEFORE the money reached the exit account" in coverage.definition
    assert "share of ALL rings" in coverage.definition
    assert "3-seed" in coverage.basis

    raw = _lead_metrics()
    for block in coverage.spans:
        for scope, shown in (
            ("single_bank", block.single_bank_rate),
            ("cross_bank", block.cross_bank_rate),
        ):
            group = _resolve(
                raw, f"seed_mean.by_scope_and_span.{scope}.span{block.span}"
            )
            loose = (
                float(group["category_counts"]["flagged_in_grid"])
                / float(group["rings"])
            )
            # Where the two measures differ, the page must show the
            # stricter one. This is what makes the control visible.
            if abs(loose - shown) > 5e-4:
                assert shown < loose, (scope, block.span)

    control = next(b for b in coverage.spans if b.span == 1)
    loose_cross = float(
        _resolve(raw, "seed_mean.by_scope_and_span.cross_bank.span1")[
            "category_counts"
        ]["flagged_in_grid"]
    ) / float(_resolve(raw, "seed_mean.by_scope_and_span.cross_bank.span1")["rings"])
    loose_single = float(
        _resolve(raw, "seed_mean.by_scope_and_span.single_bank.span1")[
            "category_counts"
        ]["flagged_in_grid"]
    ) / float(_resolve(raw, "seed_mean.by_scope_and_span.single_bank.span1")["rings"])
    if loose_cross == loose_single:
        # Exactly the situation the loose measure hides. Under the
        # published measure the two must still be reported separately,
        # whatever the sign of the difference.
        assert control.single_bank_rate is not None
        assert control.cross_bank_rate is not None


@_needs_snapshot
@_needs_lead_metrics
def test_the_span_1_control_is_published_whatever_it_says(
    data: DataSource,
) -> None:
    """The control must be present and unrounded, favourable or not.

    Joining up should buy nothing on rings wholly inside one bank. If it
    buys something, that is evidence the span-2/3 gaps are an artefact;
    if it buys slightly less than nothing, that is the strongest evidence
    they are not. Either way it is displayed. Hiding or rounding it away
    is the single most tempting dishonesty on this page.
    """
    coverage = data.coverage()
    assert coverage is not None
    control = next((s for s in coverage.spans if s.span == 1), None)
    assert control is not None, "the 1-bank stratum must be shown"
    assert control.rings > 0
    # Counts travel with the rate, so a rounded rate cannot hide a
    # difference in the underlying rings.
    assert control.single_bank_rings <= control.rings
    assert control.cross_bank_rings <= control.rings
    assert control.delta == pytest.approx(
        control.cross_bank_rate - control.single_bank_rate, abs=5e-5
    )
    # Four decimals of delta survive to the payload: a sign-carrying
    # difference of one percentage point must not round to zero.
    raw = _lead_metrics()
    published_delta = float(
        _resolve(raw, _LEAD_METRICS_PATH.format(scope="cross_bank", span=1))
    ) - float(_resolve(raw, _LEAD_METRICS_PATH.format(scope="single_bank", span=1)))
    assert control.delta == pytest.approx(published_delta, abs=5e-5)
    if abs(published_delta) >= 1e-4:
        assert control.delta != 0.0, "a real control difference was rounded away"


@_needs_snapshot
def test_the_page_claims_coverage_and_not_speed(data: DataSource) -> None:
    """The lead statistics must be published next to the coverage gap.

    Cross-bank flags more rings; it does not flag them sooner. Quoting the
    first without the second is the easy misreading, so the numbers that
    refute it ship in the same payload.
    """
    payload = data.metrics()
    coverage = payload.coverage
    assert coverage is not None
    lead = coverage.lead
    assert lead.rings_both_scopes_flag > 0
    assert (
        lead.ties + lead.cross_bank_earlier + lead.single_bank_earlier
        == lead.rings_both_scopes_flag
    )
    # Most rings tie. If that ever stops being true the wording has to
    # change, and this is what would force it.
    assert lead.ties > lead.rings_both_scopes_flag / 2
    assert lead.median_lead_minutes == 0.0

    # ONE basis for both. A 3-seed rate quoted beside a single-seed count
    # reads as one measurement and is two, so the lead sentence and the
    # coverage table must agree on which run they describe, and say so.
    assert lead.basis == coverage.basis
    assert "seed" in lead.basis

    # The headline is ring recall at the last finite horizon.
    finite = [h for h in payload.horizons if h.cutoff_minutes is not None]
    assert payload.headline_ring_recall_horizon == finite[-1].label
    assert payload.headline_ring_recall_cross_bank == finite[-1].ring_recall_cross_bank
    assert payload.headline_ring_recall_single_bank == (
        finite[-1].ring_recall_single_bank
    )


@_needs_snapshot
def test_metrics_endpoint_emits_every_declared_field(client: TestClient) -> None:
    body = client.get("/api/metrics").json()
    assert not set(MetricsPayload.model_fields) - set(body)
    assert not set(RingCoverage.model_fields) - set(body["coverage"])
    assert not set(LeadStats.model_fields) - set(body["coverage"]["lead"])
    for span in body["coverage"]["spans"]:
        assert not set(CoverageSpan.model_fields) - set(span)
        if span["per_seed"] is not None:
            assert not set(SeedCoverage.model_fields) - set(span["per_seed"])
    for retired in body["retired"]:
        assert not set(RetiredMetric.model_fields) - set(retired)
    for horizon in body["horizons"]:
        assert not set(HorizonMetrics.model_fields) - set(horizon)


def test_redteam_endpoint_emits_every_declared_field(client: TestClient) -> None:
    """The red-team card's fields, plus the committed real result's shape.

    Not gated by ``_needs_snapshot``: unlike the demo snapshot,
    ``ui/data/redteam_fallback.json`` is committed and exists on a fresh
    clone, and the live ``ml/experiments/redteam/metrics.json`` result is
    committed too -- so this endpoint answers with no build step at all.
    """
    body = client.get("/api/redteam").json()
    assert not set(RedteamPayload.model_fields) - set(body)
    assert body["strategies_run"] == len(body["verdicts"])
    assert body["caught"] + body["missed"] == body["strategies_run"]
    for verdict in body["verdicts"]:
        assert not set(RedteamVerdict.model_fields) - set(verdict)


def test_enclave_endpoint_emits_every_declared_field(client: TestClient) -> None:
    """The enclave evidence card's fields, plus the committed real run's shape.

    ``ui/data/enclave_evidence.json`` is committed and exists on a fresh
    clone -- it is the only copy of this data, not a fallback for a live
    path, so this endpoint answers with no build step and no GCP
    credentials at all.
    """
    body = client.get("/api/enclave").json()
    assert not set(EnclavePayload.model_fields) - set(body)
    assert not set(EnclaveOutboxProof.model_fields) - set(body["outbox_proof"])
    assert body["outcome"] == "completed"
    assert body["key_release_ok"] is True
    assert body["running_debug_image"] is False
    # One outbox object per bank per ring, not one per ring.
    assert len(body["bundles_written"]) == body["ring_count"] * len(body["contributing_banks"])
    assert body["outbox_proof"]["decrypt_failure_bank_id"] != body["outbox_proof"]["bank_id"]


@_needs_snapshot
def test_the_demo_picks_are_forced_into_the_snapshot(data: DataSource) -> None:
    """A pick the operator cannot open is not a pick.

    Selection ranks on detection timing over all published clusters, while
    the snapshot keeps the largest rings per span. Without pinning, the
    ring that best makes the argument is silently dropped for being small.
    """
    overview = data.overview()
    in_snapshot = {s.ring_id for s in overview.rings}
    plan = overview.demo
    assert plan.pinned_ring_ids
    for ring_id in plan.pinned_ring_ids:
        assert ring_id in in_snapshot, ring_id
        assert data.replay(ring_id).available is True
    picks = [p for p in (plan.primary, plan.secondary, plan.control) if p]
    assert [p.ring_id for p in picks] == plan.pinned_ring_ids
