"""Item 5 at the public edge: size cap, per-client limit, and opt-in HTTP 422."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents import subset_upload
from ui.main import MAX_UPLOAD_BODY_BYTES, app
from ui.models import UploadScoreResult

REJECTED = UploadScoreResult(
    banks=["Bank A"],
    backend="agents",
    error="Bank A row 2: invalid amount",
    errors=[{"bank": "Bank A", "row": 2, "field": "amount", "message": "invalid amount"}],
)


class _StubChat:
    def __init__(self, result: UploadScoreResult) -> None:
        self.result = result
        self.seen: list[Any] = []

    def subset_upload(self, banks: list[tuple[str, str]]) -> UploadScoreResult:
        self.seen.append(banks)
        return self.result


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("MULENET_UI_SUBSET_RATE_PER_MIN", "3")
    with TestClient(app) as c:
        app.state.chat = _StubChat(REJECTED)
        yield c


BODY = {"banks": [{"name": "Bank A", "csv_text": "from_account\n"}]}


def test_the_body_cap_mirrors_the_agents_limits() -> None:
    exact = subset_upload.MAX_BANKS * subset_upload.MAX_FILE_BYTES
    assert exact <= MAX_UPLOAD_BODY_BYTES < exact * 1.2


def test_a_rejected_file_is_still_a_200_with_error_and_errors_by_default(client: Any) -> None:
    body = client.post("/api/subset/upload", json=BODY).json()
    assert body["error"] == "Bank A row 2: invalid amount"  # what the current page reads
    assert body["errors"][0]["field"] == "amount"


def test_http_errors_opts_in_to_a_real_422(client: Any) -> None:
    response = client.post("/api/subset/upload?http_errors=1", json=BODY)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"] == REJECTED.errors


def test_an_oversized_upload_is_413_and_never_reaches_the_backend(client: Any) -> None:
    huge = {"banks": [{"name": "Bank A", "csv_text": "x" * (MAX_UPLOAD_BODY_BYTES + 1)}]}
    response = client.post("/api/subset/upload", json=huge)
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert app.state.chat.seen == []


def test_a_malformed_body_is_a_422_not_a_500(client: Any) -> None:
    for bad in ({"banks": "nope"}, {"banks": ["x"]}):
        assert client.post("/api/subset/upload", json=bad).status_code == 422


def test_uploads_are_rate_limited_per_client(client: Any) -> None:
    statuses = [client.post("/api/subset/upload", json=BODY).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]


def test_a_live_proof_launch_limit_reaches_the_browser_as_a_structured_429() -> None:
    from ui.chat import AgentsUnavailable, _raise_if_launch_denied
    from ui.errors import ApiError

    with pytest.raises(ApiError) as info:
        _raise_if_launch_denied(AgentsUnavailable("cooldown", 30, "Please wait."))
    assert (info.value.status, info.value.code, info.value.retry_after_seconds) == (429, "cooldown", 30)
    _raise_if_launch_denied(AgentsUnavailable("unreachable"))  # an outage is not a limit: no raise
