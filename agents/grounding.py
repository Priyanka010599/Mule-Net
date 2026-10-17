"""Every number an Ask answer states must be one a tool returned.

The existing gates are about *claims*: the model emits structured claims and
:mod:`agents.verify` re-derives them. That leaves a hole this module closes.
An answer whose claims are empty (or are true but about something else) is
published as ``verified: false`` -- honest, but it still carries whatever
number the model wrote. A model that estimates, rounds, adds two results
together or recalls a "typical" figure produces exactly such an answer, and
nothing stopped it shipping.

The rule here is mechanical and deliberately blunt: collect every number
that appears in any tool result this run (plus the row count of a list a tool
returned, which the model may legitimately quote, plus any number the asker
wrote in the question), and refuse an answer that states a
free-standing number outside that set. It does not judge whether the number
is the *right* one -- that is the verifier's job -- only whether it has a
source. Digits inside identifiers (ring ids, hashed account ids, timestamps)
are not free-standing numbers and never reach the comparison; see
:data:`agents.verify._FREE_NUMBER`.

Where results are collected: :class:`RunSink`, the provenance set the tool
wrappers already write into. ADK runs tools on its own thread, so the
collector is bound at wrap time like the sink itself (see
:func:`agents.adk_agents._model_tool`), and it is cleared with the sink
between attempts so only the attempt that answered is judged.
"""

from __future__ import annotations

import math
import re
from typing import Any

from agents.verify import _FREE_NUMBER
from shared.tool_interfaces import SCORE_TOLERANCE

#: ``12,34,567.50`` and ``1,234,567`` -> one number. Grouping commas are
#: formatting, not a second value.
_GROUPING = re.compile(r"(?<=\d),(?=\d{2,3}(?!\d))")


class RunSink(set):  # noqa: SLOT000 - a set subclass: it travels as the `sources` sink
    """The ``sources`` set, which also keeps what each tool returned.

    Everything that accepted a ``set[str]`` still does. The extra surface is
    :meth:`record_result`, which the tool wrapper calls if the sink has it,
    and :meth:`tool_done`, a hook for subclasses (jobs report progress).
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[Any] = []

    def record_result(self, tool: str, result: Any) -> None:
        self.results.append(result)

    def tool_done(self) -> None:
        """Called when any tool finishes. Subclasses may override."""

    def clear(self) -> None:
        super().clear()
        self.results.clear()


def _walk(value: Any, numbers: list[float]) -> None:
    """Every number inside a JSON-safe value, plus the length of each list."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        if math.isfinite(value):
            numbers.append(float(value))
    elif isinstance(value, str):
        numbers.extend(float(m.group(0)) for m in _FREE_NUMBER.finditer(_GROUPING.sub("", value)))
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, numbers)
    elif isinstance(value, list | tuple | set):
        for item in value:
            _walk(item, numbers)


def answer_numbers(text: str) -> list[str]:
    """The free-standing numbers an answer states, as written."""
    return [m.group(0) for m in _FREE_NUMBER.finditer(_GROUPING.sub("", text))]


def ungrounded_numbers(answer: str, question: str, results: list[Any]) -> list[str]:
    """Numbers in ``answer`` that no tool result (or the question) contains.

    Empty means every stated number has a source. Comparison is numeric
    within :data:`~shared.tool_interfaces.SCORE_TOLERANCE`, so ``0.987`` is
    grounded by a returned ``0.98700001`` and ``45`` by ``45`` or ``"45"``.
    """
    known: list[float] = []
    for result in results:
        _walk(result, known)
        if isinstance(result, list):
            known.append(float(len(result)))  # the row count of a query
    _walk(question, known)
    missing = []
    for text in answer_numbers(answer):
        value = float(text)
        if not any(abs(value - have) <= SCORE_TOLERANCE for have in known):
            missing.append(text)
    return missing
