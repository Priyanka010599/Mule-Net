"""Every tab must render something when it is the first tab opened.

The evasion tab (and tabs 3 and 5) fill their body from a fetch. Opened
first on a fresh load, with a slow backend, the body used to be an empty
box under the footer until the answer arrived: the page looked blank.
Each tab now shows a loading state at once, then its content, and a second
visit shares the load in flight rather than starting another.

This drives the real app in a real browser, so it needs Playwright and a
Chromium. Both are optional: without them the test is skipped, and
``MULENET_TEST_CHROMIUM`` can point at a browser binary.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import sync_playwright  # noqa: E402

from ui.main import app  # noqa: E402

VIEWS = ["banks", "joint", "metrics", "simulate", "subset"]


@pytest.fixture(scope="module")
def base_url():
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the UI server did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            exe = os.environ.get("MULENET_TEST_CHROMIUM") or None
            b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        except Exception as err:  # no browser installed
            pytest.skip(f"no Chromium available: {err}")
        yield b
        b.close()


def release(page):
    while page.held:
        page.held.pop().continue_()


def _fresh_page(browser, base_url, scroll):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ctx.add_init_script("try{localStorage.setItem('mulenet-guide-seen','1')}catch(e){}")
    page = ctx.new_page()

    held: list = []

    # The API calls that fill a tab are slow on a cold backend: hold each
    # one until the test lets it go.
    for pattern in ("**/api/simulate/bounds", "**/api/subset/options", "**/api/metrics"):
        page.route(pattern, lambda route: held.append(route))
    page.held = held
    # The graph library comes from a CDN; tab 2 only needs to not hang here.
    page.route(
        "**/cytoscape.min.js",
        lambda r: r.fulfill(status=200, content_type="application/javascript", body=""),
    )
    page.goto(base_url)
    page.wait_for_selector("#view-banks.active")
    if scroll:
        page.evaluate(f"window.scrollTo(0, {scroll})")
        page.wait_for_timeout(300)
    return ctx, page


@pytest.mark.parametrize("scroll", [0, 1200])
@pytest.mark.parametrize("view", VIEWS)
def test_each_tab_renders_as_the_first_tab_opened(browser, base_url, view, scroll):
    ctx, page = _fresh_page(browser, base_url, scroll)
    try:
        page.click(f".tab[data-view={view}]")
        # Almost at once: not an empty box, whatever the backend is doing.
        page.wait_for_timeout(300)
        height = page.evaluate(
            f"document.querySelector('#view-{view}').getBoundingClientRect().height"
        )
        text = page.evaluate(f"document.querySelector('#view-{view}').innerText.trim().length")
        assert height > 30 and text > 0, f"tab {view} is blank while loading"
        # And, once the data is in, its real content.
        release(page)
        page.wait_for_timeout(1500)
        assert page.locator(f"#view-{view} .view-loading").count() == 0, f"tab {view} still loading"
        if view == "metrics":
            # The loading line must not be decorated as if it were the page.
            assert page.locator("#view-metrics .m-glance").count() <= 1, "glance band duplicated"
        if view == "simulate":
            assert (
                page.locator(
                    "#sim-run-btn, #simulate-body .banner, #simulate-body .chat-offline"
                ).count()
                > 0
            )
    finally:
        ctx.close()


def test_returning_to_a_loading_tab_does_not_load_it_twice(browser, base_url):
    ctx, page = _fresh_page(browser, base_url, 0)
    calls: list[str] = []
    page.on("request", lambda r: calls.append(r.url) if "/api/simulate/bounds" in r.url else None)
    try:
        page.click(".tab[data-view=simulate]")
        page.wait_for_timeout(200)
        page.click(".tab[data-view=subset]")
        page.click(".tab[data-view=simulate]")
        page.wait_for_timeout(300)
        release(page)
        page.wait_for_timeout(1500)
        assert len(calls) == 1, calls
    finally:
        ctx.close()


def test_opening_the_ask_panel_sends_no_question(browser, base_url):
    """The panel shows suggested questions and waits; it does not call the model."""
    ctx, page = _fresh_page(browser, base_url, 0)
    asked: list[str] = []
    page.on("request", lambda r: asked.append(r.url) if "/api/chat" in r.url else None)
    try:
        release(page)
        page.click("#chat-btn")
        page.wait_for_timeout(800)
        assert asked == []
        assert page.locator("#chat-examples .chip").count() >= 1
    finally:
        ctx.close()
