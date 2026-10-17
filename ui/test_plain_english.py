"""The plain-English walkthrough (tab 1) and number guide (tab 3) render and say real numbers.

Needs Playwright and a Chromium (``MULENET_TEST_CHROMIUM``); skipped without them.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import sync_playwright  # noqa: E402

from ui.config import UiConfig  # noqa: E402
from ui.main import app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not UiConfig.from_env().snapshot_path.exists(), reason="run `python -m ui.build_snapshot`"
)


@pytest.fixture(scope="module")
def base_url():
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(
                executable_path=os.environ.get("MULENET_TEST_CHROMIUM") or None,
                args=["--no-sandbox"],
            )
        except Exception as err:
            pytest.skip(f"no Chromium available: {err}")
        yield b
        b.close()


def _page(browser, base_url, width=1280):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    ctx.add_init_script("try{localStorage.setItem('mulenet-guide-seen','1')}catch(e){}")
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/cytoscape.min.js",
        lambda r: r.fulfill(status=200, content_type="application/javascript", body=""),
    )
    page.goto(base_url)
    page.wait_for_selector("#view-banks.active")
    return ctx, page, errors


def test_walkthrough_opens_first_visit_then_stays_closed(browser, base_url) -> None:
    ctx, page, errors = _page(browser, base_url)
    try:
        assert page.locator("#how-works").evaluate("e => e.open") is True
        assert page.locator("#how-works .how-steps li").count() == 6
        page.reload()
        page.wait_for_selector("#view-banks.active")
        assert page.locator("#how-works").evaluate("e => e.open") is False
        page.click("#how-works summary")
        assert page.locator("#how-works").evaluate("e => e.open") is True
        assert errors == []
    finally:
        ctx.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_no_sideways_scroll_with_the_walkthrough_open(browser, base_url, width) -> None:
    ctx, page, _ = _page(browser, base_url, width)
    try:
        sizes = page.evaluate("[document.documentElement.scrollWidth, innerWidth]")
        assert sizes[0] <= sizes[1]
    finally:
        ctx.close()


def test_metrics_guide_quotes_the_numbers_on_the_cards(browser, base_url) -> None:
    ctx, page, errors = _page(browser, base_url)
    try:
        page.click(".tab[data-view=metrics]")
        page.wait_for_selector("#plain-numbers", timeout=15000)
        guide = page.locator(".plain-grid").inner_text()
        card = page.locator(".metric-cards .mcard").first.inner_text()
        # The PR-AUC the first card shows (three decimals) is the one the guide quotes.
        big = page.locator(".metric-cards .mcard .big").first.inner_text().strip()
        assert big in guide and "PR-AUC" in guide, (big, card)
        assert "—" not in guide
        assert errors == []
    finally:
        ctx.close()


def test_rings_card_states_the_account_vs_ring_recall_gap(browser, base_url) -> None:
    """87.8% of accounts but 53 of 120 rings (44.2%): the gap is stated, with its cause."""
    ctx, page, errors = _page(browser, base_url)
    try:
        page.click(".tab[data-view=metrics]")
        page.wait_for_selector("#plain-numbers", timeout=15000)
        card = page.locator(".plain-card", has_text="Rings flagged").inner_text()
        for needle in ("87.8%", "53 of 120", "44.2%", "singletons"):
            assert needle in card, needle
        assert "—" not in card
        assert errors == []
    finally:
        ctx.close()
