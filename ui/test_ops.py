"""Item 8: request ids, one error envelope, body cap, CORS, safe logs, readiness."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import urllib.error
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui import ops
from ui.config import UiConfig
from ui.errors import register_handlers
from ui.main import app

ACCOUNT = "835a8624f907bada67dbe2ff128bc69242677053a3367348eac49d713411f1f4"


def _small_app(**overrides: object) -> FastAPI:
    """A tiny app wired exactly like ui.main: handlers, CORS, ops middleware."""
    cfg = dataclasses.replace(UiConfig.from_env(), **overrides)
    small = FastAPI()
    register_handlers(small)
    ops.install_middleware(small, cfg)

    @small.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret question text")

    @small.post("/echo")
    def echo(body: dict) -> dict:
        return {"n": len(body)}

    @small.get("/api/screen/{hashed_account_id}")
    def screen(hashed_account_id: str) -> dict:
        return {"ok": True}

    return small


# -- request id ------------------------------------------------------------


def test_request_id_is_generated_returned_and_in_the_error_envelope() -> None:
    with TestClient(app) as client:
        response = client.get("/api/rings/ring-does-not-exist")
    rid = response.headers["x-request-id"]
    assert len(rid) == 32
    body = response.json()
    assert body["request_id"] == rid
    assert body["code"] == "not_found" and body["detail"] == body["message"]


def test_a_sane_inbound_request_id_is_honoured_and_a_hostile_one_is_not() -> None:
    with TestClient(app) as client:
        kept = client.get("/health", headers={"X-Request-Id": "trace-abc_123.xyz"})
        replaced = client.get("/health", headers={"X-Request-Id": "bad id; <script>"})
        too_long = client.get("/health", headers={"X-Request-Id": "a" * 65})
    assert kept.headers["x-request-id"] == "trace-abc_123.xyz"
    assert replaced.headers["x-request-id"] != "bad id; <script>"
    assert len(too_long.headers["x-request-id"]) == 32


# -- one envelope ------------------------------------------------------------


def test_validation_errors_use_the_envelope_and_never_echo_input() -> None:
    with TestClient(app) as client:
        response = client.post("/api/simulate", json="my-secret-question-text")
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"] and {"field", "message"} <= set(body["errors"][0])
    assert isinstance(body["detail"], list)  # FastAPI's shape, kept for the page
    assert "my-secret-question-text" not in response.text
    assert body["request_id"] == response.headers["x-request-id"]


def test_unknown_route_and_method_use_the_envelope() -> None:
    with TestClient(app) as client:
        missing = client.get("/api/nope")
        wrong = client.delete("/api/metrics")
    assert missing.json()["code"] == "not_found"
    assert wrong.status_code == 405 and wrong.json()["code"] == "method_not_allowed"


def test_unhandled_error_is_a_generic_500_envelope_without_the_message() -> None:
    with TestClient(_small_app(), raise_server_exceptions=False) as client:
        response = client.get("/boom")
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "secret question text" not in response.text


# -- body cap ----------------------------------------------------------------


def test_oversized_post_is_413_by_content_length_and_when_chunked() -> None:
    with TestClient(_small_app(max_body_bytes=1000)) as client:
        big = {"k": "x" * 2000}
        declared = client.post("/echo", json=big)

        def chunks():
            yield json.dumps(big).encode()

        streamed = client.post(
            "/echo", content=chunks(), headers={"Content-Type": "application/json"}
        )
        small = client.post("/echo", json={"k": "v"})
    for response in (declared, streamed):
        assert response.status_code == 413
        assert response.json()["code"] == "payload_too_large"
        assert response.json()["request_id"] == response.headers["x-request-id"]
    assert small.status_code == 200


def test_upload_path_is_exempt_from_the_small_cap() -> None:
    padding = "x" * (ops_default_cap() + 10)
    with TestClient(app) as client:
        chat = client.post("/api/chat", json={"message": padding})
        upload = client.post("/api/subset/upload", json={"banks": [], "pad": padding})
    assert chat.status_code == 413
    assert upload.status_code != 413


def ops_default_cap() -> int:
    return UiConfig.from_env().max_body_bytes


# -- CORS --------------------------------------------------------------------


def test_default_is_same_origin_only() -> None:
    with TestClient(app) as client:
        simple = client.get("/health", headers={"Origin": "https://evil.example"})
        preflight = client.options(
            "/api/chat",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert "access-control-allow-origin" not in simple.headers
    assert "access-control-allow-origin" not in preflight.headers


def test_only_configured_origins_are_allowed() -> None:
    with TestClient(_small_app(cors_allow_origins=("https://ok.example",))) as client:
        ok = client.get("/api/screen/x", headers={"Origin": "https://ok.example"})
        bad = client.get("/api/screen/x", headers={"Origin": "https://evil.example"})
    assert ok.headers["access-control-allow-origin"] == "https://ok.example"
    assert "access-control-allow-origin" not in bad.headers


# -- logs --------------------------------------------------------------------


def test_access_log_uses_the_route_template_never_the_account_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="mulenet.ui.access"):
        with TestClient(_small_app()) as client:
            client.get(f"/api/screen/{ACCOUNT}?q=secret")
            client.get("/api/screen/another")
    records = [r for r in caplog.records if r.name == "mulenet.ui.access"]
    assert [r.route for r in records] == ["/api/screen/{hashed_account_id}"] * 2
    assert records[0].first_request is True and records[0].since_process_start_ms >= 0
    assert not hasattr(records[1], "first_request")
    assert ACCOUNT not in "".join(str(vars(r)) for r in records)


def test_json_formatter_redacts_and_drops_unknown_fields() -> None:
    record = logging.LogRecord(
        "mulenet.x", logging.INFO, __file__, 1,
        "GET /api/screen/%s for alice@example.com id %s", (ACCOUNT, ACCOUNT), None,
    )
    record.question = "what is my balance"  # not a whitelisted field
    record.status = 200
    line = json.loads(ops.JsonFormatter().format(record))
    assert ACCOUNT not in json.dumps(line) and "alice@example.com" not in json.dumps(line)
    assert "question" not in line and line["status"] == 200
    assert line["severity"] == "INFO" and "request_id" in line


def test_redact_filter_scrubs_uvicorn_style_access_lines() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", f"/api/screen/{ACCOUNT}", "1.1", 200), None,
    )
    ops.RedactFilter().filter(record)
    assert ACCOUNT not in record.getMessage()
    assert "/api/screen/[id]" in record.getMessage()


def test_unhandled_exception_logs_only_the_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="mulenet.ui.errors"):
        with TestClient(_small_app(), raise_server_exceptions=False) as client:
            client.get("/boom")
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(ops.JsonFormatter())
    for record in caplog.records:
        handler.emit(record)
    assert "RuntimeError" in out.getvalue()
    assert "secret question text" not in out.getvalue()


# -- readiness ---------------------------------------------------------------


def test_ready_is_distinct_from_health_and_reports_snapshot_and_mock_agents() -> None:
    with TestClient(app) as client:
        response = client.get("/ready")
        health = client.get("/health")
    assert health.json() == {"status": "ok"}
    if not UiConfig.from_env().snapshot_path.exists():
        assert response.status_code == 503 and response.json()["status"] == "not_ready"
        return
    body = response.json()
    assert response.status_code == 200 and body["status"] == "ready"
    assert body["snapshot_loaded"] is True
    assert body["agents"] == {"backend": "mock", "configured": False, "reachable": None}
    assert body["startup_ms"] >= 0


def test_ready_reports_agents_unreachable_as_degraded_without_leaking_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MULENET_UI_CHAT_BACKEND", "agents")
    monkeypatch.setenv("MULENET_UI_AGENTS_URL", "https://agents-internal.example.run.app")
    monkeypatch.setenv("MULENET_UI_AGENTS_AUTH", "0")
    seen: list[str] = []

    def refuse(request, timeout):  # noqa: ANN001
        seen.append(request.full_url)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(ops.urllib.request, "urlopen", refuse)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert seen == ["https://agents-internal.example.run.app/health"]  # health, not /chat
    body = response.json()
    assert body["agents"]["reachable"] is False
    assert "agents-internal" not in response.text
    if UiConfig.from_env().snapshot_path.exists():
        assert response.status_code == 200 and body["status"] == "degraded"


def test_ready_is_503_when_the_snapshot_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MULENET_UI_SNAPSHOT", str(tmp_path / "missing.json"))
    with TestClient(app) as client:
        response = client.get("/ready")
        overview = client.get("/api/overview")
    assert response.status_code == 503
    assert response.json()["snapshot_loaded"] is False
    assert overview.status_code == 503 and overview.json()["code"] == "snapshot_missing"
    assert overview.json()["detail"]  # kept for the page's existing handler


def test_scrub_internal_removes_kms_resources_accounts_buckets_and_service_urls() -> None:
    raw = {
        "detail": (
            "Denied on projects/p1/locations/asia-south1/keyRings/r/cryptoKeys/k "
            "for sa@p1.iam.gserviceaccount.com via gs://bkt/obj and https://x-1.a.run.app/y"
        ),
        "steps": [{"note": "fine"}],
    }
    text = json.dumps(ops.scrub_internal(raw))
    for leaked in ("projects/p1", "cryptoKeys", "gserviceaccount", "gs://bkt", "run.app"):
        assert leaked not in text
    assert ops.scrub_internal(raw)["steps"] == [{"note": "fine"}]
