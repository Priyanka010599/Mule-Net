"""WCAG AA contrast of the text tokens, computed from the CSS itself.

The colour tokens are custom properties spread over several stylesheets
(later ones win). This reads them in the order index.html links them, for
the light theme and for ``data-theme="dark"``, and checks the ink tokens
against the surfaces they sit on. No browser needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).parent
HTML = (UI / "templates" / "index.html").read_text()
SHEETS = re.findall(r'href="/static/([\w.-]+\.css)"', HTML)

# Normal text needs 4.5:1, large text (18px, or 14px bold) 3:1.
AA_TEXT = 4.5
AA_LARGE = 3.0

# (foreground token, minimum). Accent, danger, good, warn and bank colours
# are used both as fills and as text; their text-safe "-ink" variants are
# the ones text must use.
TEXT_TOKENS = {
    "--ink": AA_TEXT,
    "--ink-2": AA_TEXT,
    "--ink-3": AA_TEXT,
    "--accent": AA_TEXT,
    "--good-ink": AA_TEXT,
    "--danger-ink": AA_TEXT,
    "--warn-ink": AA_TEXT,
    "--bank-b-ink": AA_TEXT,
}
SURFACES = ["--bg", "--surface", "--surface-2"]


def _blocks(css: str, selector_re: str) -> list[dict[str, str]]:
    """Declarations of every top-level (not @media) rule whose selector matches."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: list[dict[str, str]] = []
    depth = 0
    i = 0
    while i < len(css):
        if css[i] == "{" and depth == 0:
            sel_start = css.rfind("}", 0, i) + 1
            sel = css[sel_start:i].strip()
            end = css.index("}", i)
            if "{" in css[i + 1 : end]:  # an @media wrapper: skip its inside
                depth = 1
                i += 1
                continue
            if re.fullmatch(selector_re, sel):
                out.append(dict(re.findall(r"(--[\w-]+)\s*:\s*([^;}]+)", css[i + 1 : end])))
            i = end
        elif css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return out


def tokens(theme: str) -> dict[str, str]:
    sel = r":root" if theme == "light" else r":root(?:\[data-theme=\"dark\"\])?"
    env: dict[str, str] = {}
    for sheet in SHEETS:
        # print.css-style overrides live in @media print and are skipped above.
        for decls in _blocks((UI / "static" / sheet).read_text(), sel):
            env.update({k: v.strip() for k, v in decls.items()})
    return env


def resolve(env: dict[str, str], name: str) -> str:
    v = env[name]
    while v.startswith("var("):
        v = env[v[4:-1].strip()]
    return v


def lum(hex_: str) -> float:
    h = hex_.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a: str, b: str) -> float:
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_tokens_meet_aa_on_every_surface(theme):
    env = tokens(theme)
    failures = []
    for fg, minimum in TEXT_TOKENS.items():
        for bg in SURFACES:
            r = ratio(resolve(env, fg), resolve(env, bg))
            if r < minimum:
                failures.append(
                    f"{theme}: {fg} {resolve(env, fg)} on {bg} {resolve(env, bg)} = {r:.2f}"
                )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_button_text_on_primary_and_accent_soft(theme):
    env = tokens(theme)
    assert ratio(resolve(env, "--on-accent"), resolve(env, "--accent")) >= AA_TEXT
    assert ratio(resolve(env, "--ink"), resolve(env, "--accent-soft")) >= AA_TEXT
    assert ratio(resolve(env, "--ink-2"), resolve(env, "--accent-soft")) >= AA_TEXT
