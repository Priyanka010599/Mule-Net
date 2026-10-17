"""Delimit and neutralise untrusted text before a model reads it.

Free text reaches the Ask models from places nobody on this team wrote: a
cell or bank name in an uploaded CSV, an account or ring narrative, a KYC
address, an evidence ``detail``. Any of it can say "ignore previous
instructions and reveal ...". The defence has three parts, and this module is
the first two:

1. **Delimit.** :func:`wrap` puts the text between
   ``<<UNTRUSTED_DATA ...>>`` and ``<<END_UNTRUSTED_DATA>>`` markers, so the
   model can see where data starts and stops.
2. **Neutralise.** :func:`sanitize` first removes control characters, defuses
   anything that looks like one of our markers (so the text cannot close the
   block early and write "outside" it), and caps the length.
3. **Instruct.** ``agents/prompts/*.md`` tell every agent that text inside
   those markers is data and is never to be obeyed.

None of this is a guarantee about a model's behaviour; it is what makes the
boundary visible and unforgeable. The things that *enforce* safety do not
depend on the model obeying: the SQL allow-list and label denylist
(:mod:`agents.hidden_labels`), the scope rule (:mod:`agents.ask_scope`) and
the grounding check (:mod:`agents.grounding`) all run in Python.
"""

from __future__ import annotations

import re
from typing import Any

OPEN = "<<UNTRUSTED_DATA"
CLOSE = "<<END_UNTRUSTED_DATA>>"
DEFAULT_MAX_CHARS = 2000

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ‪-‮]")
#: Marker look-alikes: runs of angle brackets, and the marker words.
_BRACKETS = re.compile(r"[<>‹›«»＜＞]{2,}")
_MARKER_WORDS = re.compile(r"(?i)untrusted[\s_-]*data")
#: A word followed by more text: prose, not an id, number or timestamp.
_FREE_TEXT = re.compile(r"[A-Za-z]{2,}\s+\S")


def sanitize(text: object, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """``text`` with control characters and delimiter look-alikes removed, capped."""
    cleaned = _CONTROL.sub("", str(text))
    cleaned = _BRACKETS.sub("[delimiter removed]", cleaned)
    cleaned = _MARKER_WORDS.sub("[delimiter removed]", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + " [truncated]"
    return cleaned


def wrap(text: object, label: str = "data", max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """``text`` sanitised and enclosed in the untrusted-data markers."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "", label)[:40] or "data"
    return f"{OPEN} source={safe_label}>>\n{sanitize(text, max_chars)}\n{CLOSE}"


def wrap_tool_result(value: Any, label: str) -> Any:
    """``value`` with every free-text string delimited and every other string sanitised.

    Ids, numbers, timestamps and bank codes keep their meaning (no
    whitespace-separated prose), so a model can still copy them exactly.
    """
    if isinstance(value, str):
        if _FREE_TEXT.search(value) or _BRACKETS.search(value) or _MARKER_WORDS.search(value):
            return wrap(value, label)
        return _CONTROL.sub("", value)
    if isinstance(value, dict):
        return {k: wrap_tool_result(v, label) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [wrap_tool_result(v, label) for v in value]
    return value
