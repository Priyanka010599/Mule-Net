"""Browser checks for the accessibility pass: graph wheel, phone header, every tab.

Real Chromium through Playwright, both optional (the module skips without
them; ``MULENET_TEST_CHROMIUM`` points at a browser binary). The graph library
comes from a CDN the sandbox cannot reach, so the zoom tests need a local copy
named by ``MULENET_TEST_CYTOSCAPE`` and skip without one; every other test
stubs the library out.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import sync_playwright  # noqa: E402

from ui.config import UiConfig  # noqa: E402
from ui.main import app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not UiConfig.from_env().snapshot_path.exists(), reason="run `python -m ui.build_snapshot`"
)

VIEWS = ["banks", "joint", "metrics", "simulate", "subset"]
NARROW = [390, 375, 320]
CYTOSCAPE = os.environ.get("MULENET_TEST_CYTOSCAPE", "")


@pytest.fixture(scope="module")
def base_url():
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the UI server did not start"
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
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


def open_page(browser, base_url, width, height=800, with_graph=False):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    ctx.add_init_script("try{localStorage.setItem('mulenet-guide-seen','1')}catch(e){}")
    page = ctx.new_page()
    page.problems = []
    page.on("console", lambda m: page.problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: page.problems.append(str(e)))
    body = Path(CYTOSCAPE).read_text() if with_graph else ""
    page.route(
        "**/cytoscape.min.js",
        lambda r: r.fulfill(status=200, content_type="application/javascript", body=body),
    )
    page.goto(base_url)
    page.wait_for_selector("#view-banks.active")
    return ctx, page


def replay_rows(base_url) -> list[dict]:
    import json
    import urllib.request

    with urllib.request.urlopen(f"{base_url}/api/replay-index") as r:
        return json.load(r)["rings"]


# ── item 1: the graph does not trap the page's wheel ───────────────────────


@pytest.mark.skipif(not CYTOSCAPE or not Path(CYTOSCAPE).exists(), reason="no local cytoscape")
def test_plain_wheel_scrolls_page_and_ctrl_wheel_zooms(browser, base_url):
    ctx, page = open_page(browser, base_url, 1280, with_graph=True)
    try:
        page.click(".tab[data-view=joint]")
        page.wait_for_function("document.querySelector('#cy')._cyreg?.cy?.nodes().length > 0")
        zoom = "document.querySelector('#cy')._cyreg.cy.zoom()"

        def hover_graph():
            page.evaluate(
                "document.querySelector('#cy').scrollIntoView({block:'start',behavior:'instant'})"
            )
            page.wait_for_timeout(300)
            box = page.locator("#cy").bounding_box()
            x, y = box["x"] + box["width"] / 2, box["y"] + min(box["height"], 400) / 2
            page.mouse.move(x, y)
            assert page.evaluate(f"!!document.elementFromPoint({x}, {y})?.closest('#cy')")

        hover_graph()
        before_zoom, before_y = page.evaluate(zoom), page.evaluate("window.scrollY")
        page.mouse.wheel(0, 150)
        page.wait_for_timeout(500)
        assert page.evaluate(zoom) == before_zoom, "a plain wheel must not zoom"
        assert page.evaluate("window.scrollY") > before_y, "a plain wheel must scroll the page"
        hover_graph()
        page.keyboard.down("Control")
        page.mouse.wheel(0, -300)
        page.keyboard.up("Control")
        page.wait_for_timeout(400)
        assert page.evaluate(zoom) != before_zoom, "ctrl + wheel must zoom"
        # The hint is a polite status that never takes clicks.
        hint = page.locator("#cy-hint")
        assert hint.get_attribute("role") == "status"
        assert hint.evaluate("e => getComputedStyle(e).pointerEvents") == "none"
        assert "to zoom" in hint.inner_text()
        # The buttons still zoom.
        z = page.evaluate(zoom)
        page.click("#zoom-in-btn")
        assert page.evaluate(zoom) > z
        assert page.problems == []
    finally:
        ctx.close()


# ── item 2: the phone header ───────────────────────────────────────────────


def rects(page, selector):
    return page.evaluate(
        """(sel) => [...document.querySelectorAll(sel)].map((e) => {
            const r = e.getBoundingClientRect();
            return {id: e.id, x: r.x, y: r.y, w: r.width, h: r.height, r: r.right};
        })""",
        selector,
    )


@pytest.mark.parametrize("width", NARROW)
def test_phone_header_fits_and_targets_are_40px(browser, base_url, width):
    ctx, page = open_page(browser, base_url, width, height=700)
    try:
        page.wait_for_timeout(500)
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        shown = [
            r
            for r in rects(page, ".topbar-actions > button, .topbar-actions > .more-wrap > button")
            if r["w"] > 0
        ]
        assert {r["id"] for r in shown} == {"demo-btn", "chat-btn", "more-btn"}
        assert len({round(r["y"]) for r in shown}) == 1, "header buttons share one row"
        for r in shown:
            assert r["w"] >= 40 and r["h"] >= 40, r
            assert r["x"] >= 0 and r["r"] <= width, r
        ordered = sorted(shown, key=lambda r: r["x"])
        for a, b in zip(ordered, ordered[1:], strict=False):
            assert a["r"] <= b["x"], f"{a['id']} overlaps {b['id']}"

        more = page.locator("#more-btn")
        assert more.get_attribute("aria-haspopup") == "true"
        assert more.get_attribute("aria-expanded") == "false"
        more.click()
        assert more.get_attribute("aria-expanded") == "true"
        items = rects(page, "#more-menu button")
        assert {i["id"] for i in items} == {
            "theme-btn", "density-btn", "print-btn", "guide-btn", "jump-live-proof-btn",
        }
        for i in items:
            assert i["h"] >= 40 and i["w"] >= 40 and i["x"] >= 0 and i["r"] <= width, i
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        page.keyboard.press("Escape")
        assert more.get_attribute("aria-expanded") == "false"
        assert page.evaluate("document.activeElement.id") == "more-btn"
        more.click()
        page.mouse.click(5, 650)
        assert more.get_attribute("aria-expanded") == "false"
        # An item still runs its own handler (theme cycles), and closes the menu.
        more.click()
        label = page.locator("#theme-btn").inner_text()
        page.click("#theme-btn")
        assert page.locator("#theme-btn").inner_text() != label
        assert more.get_attribute("aria-expanded") == "false"
    finally:
        ctx.close()


def test_desktop_header_keeps_all_seven_buttons(browser, base_url):
    ctx, page = open_page(browser, base_url, 1280)
    try:
        ids = [
            r["id"]
            for r in rects(page, ".topbar-actions > button, .topbar-actions > .more-wrap > button")
            if r["w"] > 0
        ]
        assert ids == [
            "demo-btn", "theme-btn", "density-btn", "print-btn", "guide-btn",
            "jump-live-proof-btn", "chat-btn",
        ]
        assert page.locator("#more-btn").count() == 0 or not page.locator("#more-btn").is_visible()
    finally:
        ctx.close()


# ── item 3: type size ──────────────────────────────────────────────────────

SMALL_TEXT = """() => {
  const out = [];
  document.querySelectorAll('body *').forEach((e) => {
    if (e.closest('svg') && !['text', 'tspan'].includes(e.tagName)) return;
    const own = [...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (!own) return;
    const r = e.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return;
    const fs = parseFloat(getComputedStyle(e).fontSize);
    if (fs < 12) out.push(fs + ' ' + e.tagName + '.' + e.className);
  });
  return out;
}"""


@pytest.mark.parametrize("width", [1280, 390])
def test_no_visible_text_under_12px_on_any_tab(browser, base_url, width):
    ctx, page = open_page(browser, base_url, width)
    try:
        small = []
        for view in VIEWS:
            page.click(f".tab[data-view={view}]")
            page.wait_for_timeout(1200)
            page.evaluate("document.querySelectorAll('details').forEach((d) => { d.open = true; })")
            small += [f"{view}: {s}" for s in page.evaluate(SMALL_TEXT)]
        assert small == []
    finally:
        ctx.close()


# ── item 4: the tie caption comes from the replay's numbers ────────────────


def _tied(rows):
    out = []
    for r in rows:
        s = {x["scope"]: x for x in r["scopes"]}
        if len(s) == 2 and r["available"]:
            a, b = s["single_bank"], s["cross_bank"]
            if (
                a["flag_offset_minutes"] is not None
                and a["flag_offset_minutes"] == b["flag_offset_minutes"]
            ):
                out.append(r)
    return out


def test_tie_caption_states_the_replay_numbers_and_only_on_ties(browser, base_url):
    rows = replay_rows(base_url)
    tied = _tied(rows)
    helped = [r for r in rows if r["joining_helps"]]
    assert tied and helped, "the snapshot should hold both kinds of ring"
    ctx, page = open_page(browser, base_url, 1280)
    try:
        page.click(".tab[data-view=joint]")
        page.wait_for_selector("#ring-select option", state="attached")
        for row in tied[:3]:
            page.select_option("#ring-select", row["ring_id"])
            page.wait_for_function(
                "id => document.querySelector('#replay .pill')?.textContent.includes(id)",
                arg=row["ring_id"][-5:],
            )
            text = page.locator("#replay .race-same").inner_text()
            s = {x["scope"]: x for x in row["scopes"]}["cross_bank"]
            assert f"T={s['first_flag_T']}m" in text
            assert f"{s['minutes_before_exit']:.1f} min before" in text
            assert "no lead time" in text and "coverage" not in text
        page.select_option("#ring-select", helped[0]["ring_id"])
        page.wait_for_function(
            "id => document.querySelector('#replay .pill')?.textContent.includes(id)",
            arg=helped[0]["ring_id"][-5:],
        )
        assert page.locator("#replay .race-same").count() == 0
    finally:
        ctx.close()


# ── every tab, desktop and phone ───────────────────────────────────────────


@pytest.mark.parametrize("width", [1280, 390])
def test_every_tab_renders_without_errors_or_sideways_scroll(browser, base_url, width):
    ctx, page = open_page(browser, base_url, width)
    try:
        tabs = page.evaluate("[...document.querySelectorAll('.tab')].map(t => t.dataset.view)")
        assert tabs == VIEWS
        for view in VIEWS:
            page.click(f".tab[data-view={view}]")
            page.wait_for_timeout(1500)
            assert page.locator(f"#view-{view}.active").count() == 1
            assert page.locator(f"#view-{view} .view-loading").count() == 0, view
            assert (
                page.evaluate(f"document.querySelector('#view-{view}').innerText.trim().length")
                > 50
            )
            assert page.evaluate("document.documentElement.scrollWidth") <= width, view
        assert page.problems == []
    finally:
        ctx.close()
