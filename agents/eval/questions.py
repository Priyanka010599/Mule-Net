"""The 25-question eval set, with answers derived from the source data.

Every expected answer is computed here, directly from
``simulator/out/ground_truth.parquet`` and the published parquet tables,
using plain pandas. The agent reaches its answer through a completely
different path -- the router, the tool layer and the SQL guard. The two
paths share no code, which is what makes agreement worth measuring.
Hardcoding the answers would test nothing but my typing.

The set is deliberately split into two categories:

``detection``
    Answerable from the tables the agent is allowed to read. A correct
    answer means the tool path works.
``declined``
    Answerable only from ``ground_truth``, which is **not** in
    ``ALLOWED_SQL_TABLES``. The correct behaviour is to decline. These
    questions are the more interesting half: an agent that invents a
    number here is worse than one that fails outright, because the
    number will look plausible. Nine of the twenty-five are of this kind.

**Two labels, never a bare "positive".** The raw ``is_mule`` column
counts 899 accounts and conflates two different populations, so this
module does not use it:

``is_passthrough_mule``
    ``role in {mule, collector}`` -- **712**. Accounts that moved other
    people's money onward. This is the population a pass-through or
    velocity question means.
``is_complicit``
    ``role in {mule, collector, source}`` -- **899**. Adds the 187
    ``source`` accounts.

``source`` accounts are complicit feeders, **not victims**: the
simulator allocates them from the mule pool, gives them the ring's
shared devices and reused KYC documents, opens them on the herder's
schedule, and sends 74% of their outflow to ring-mates at a median
in-degree of zero. They belong in ``is_complicit`` and not in
``is_passthrough_mule``, and every question below states which it means
-- an agent answering 712 to a pass-through question is right, and the
eval must not mark it wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.config import AgentsConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: How many questions :func:`build_questions` returns.
EXPECTED_COUNT = 25

#: The answer an agent must give for a ``declined`` question. Matched
#: loosely by the scorer, so any clear refusal counts.
DECLINED = "cannot be answered from the allowed tables"

#: ``is_passthrough_mule``: accounts that forwarded other people's money.
PASSTHROUGH_ROLES = frozenset({"mule", "collector"})

#: ``is_complicit``: the pass-through roles plus the ``source`` feeders.
COMPLICIT_ROLES = frozenset({"mule", "collector", "source"})


def _label_masks(gt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the ``(is_passthrough_mule, is_complicit)`` masks.

    Derived from ``role`` rather than read from ``is_mule``, so the two
    populations can never be conflated by a column that answers neither
    question precisely.
    """
    role = gt["role"].astype("string").fillna("")
    return role.isin(PASSTHROUGH_ROLES), role.isin(COMPLICIT_ROLES)


@dataclass(frozen=True)
class EvalQuestion:
    """One eval item.

    Attributes:
        id: Stable handle, ``q01``..``q25``.
        question: The natural-language question put to the router.
        expected: The correct answer, derived from source data.
        category: ``detection`` or ``declined``.
        note: Why this question is in the set.
    """

    id: str
    question: str
    expected: str
    category: str
    note: str = ""


def _load(config: AgentsConfig) -> dict[str, pd.DataFrame]:
    import pandas as pd

    return {
        "ground_truth": pd.read_parquet(config.data_dir / "ground_truth.parquet"),
        "accounts": pd.read_parquet(config.data_dir / "accounts.parquet"),
        "rings": pd.read_parquet(config.ml_out_dir / "rings.parquet"),
        "scores": pd.read_parquet(config.ml_out_dir / "scores.parquet"),
        "evidence": pd.read_parquet(config.ml_out_dir / "account_evidence.parquet"),
    }


def ground_truth_summary(config: AgentsConfig | None = None) -> dict[str, object]:
    """Summarise the label file, for the eval report and for sanity checks."""
    config = config or AgentsConfig.from_env()
    frames = _load(config)
    gt = frames["ground_truth"]
    labelled = gt[gt["ring_id"].astype("string").fillna("") != ""]
    per_ring_banks = labelled.groupby("ring_id")["bank_id"].nunique()

    passthrough, complicit = _label_masks(gt)

    return {
        "accounts": int(len(gt)),
        # No bare "positives": the two labels answer different questions.
        "passthrough_mules": int(passthrough.sum()),
        "complicit": int(complicit.sum()),
        "true_rings": int(labelled["ring_id"].nunique()),
        "rings_by_bank_span": {
            int(span): int(count)
            for span, count in per_ring_banks.value_counts().sort_index().items()
        },
        "roles": {
            str(k): int(v) for k, v in gt[complicit]["role"].value_counts().items()
        },
        "cohorts": {str(k): int(v) for k, v in gt["cohort"].value_counts().items()},
        "pattern_types": {
            str(k): int(v)
            for k, v in gt[complicit]["pattern_type"].value_counts().items()
        },
        "detected_rings": int(len(frames["rings"])),
    }


def build_questions(config: AgentsConfig | None = None) -> list[EvalQuestion]:
    """Build the 25 questions with answers derived from the source data."""
    config = config or AgentsConfig.from_env()
    frames = _load(config)
    gt = frames["ground_truth"]
    accounts = frames["accounts"]
    rings = frames["rings"]
    scores = frames["scores"]
    evidence = frames["evidence"]

    labelled = gt[gt["ring_id"].astype("string").fillna("") != ""]
    per_ring_banks = labelled.groupby("ring_id")["bank_id"].nunique()
    ring_span = rings["banks_involved"].map(len)
    passthrough, complicit = _label_masks(gt)

    # A ring and an account that actually exist, so subject-scoped
    # questions are answerable rather than testing the not-found path.
    sample_ring = rings.sort_values("ring_id").iloc[0]
    ring_id = str(sample_ring["ring_id"])
    # The frame carries member structs; an unqualified count is the
    # alertable population, so both are derived here.
    ring_member_records = [dict(m) for m in list(sample_ring["members"])]
    ring_alertable = [m for m in ring_member_records if m["alertable"]]
    ring_banks = sorted(sample_ring["banks_involved"])

    top_account = str(scores.sort_values("score", ascending=False).iloc[0][
        "hashed_account_id"
    ])
    top_bank = str(
        accounts[accounts["hashed_account_id"] == top_account]["bank_id"].iloc[0]
    )
    top_score = float(scores[scores["hashed_account_id"] == top_account]["score"].iloc[0])

    questions: list[EvalQuestion] = [
        # --- detection: aggregates over the published tables ------------
        EvalQuestion(
            "q01",
            "How many rings has the detector found?",
            str(len(rings)),
            "detection",
            "detected ring count, not the true ring count",
        ),
        EvalQuestion(
            "q02",
            "How many accounts are in the dataset?",
            str(len(accounts)),
            "detection",
        ),
        EvalQuestion(
            "q03",
            "How many banks are in the dataset?",
            str(accounts["bank_id"].nunique()),
            "detection",
        ),
        EvalQuestion(
            "q04",
            "How many accounts are held at BANK_A?",
            str(int((accounts["bank_id"] == "BANK_A").sum())),
            "detection",
        ),
        EvalQuestion(
            "q05",
            "How many detected rings span three banks?",
            str(int((ring_span == 3).sum())),
            "detection",
        ),
        EvalQuestion(
            "q06",
            "How many detected rings span two banks?",
            str(int((ring_span == 2).sum())),
            "detection",
        ),
        EvalQuestion(
            "q07",
            "How many detected rings are cross-bank, spanning more than one bank?",
            str(int((ring_span > 1).sum())),
            "detection",
        ),
        EvalQuestion(
            "q08",
            "How many accounts have cross-bank identity evidence attached?",
            str(evidence["hashed_account_id"].nunique()),
            "detection",
            "evidence is attached to flagged accounts, never used to flag them",
        ),
        EvalQuestion(
            "q09",
            "How many evidence rows have scope cross_bank?",
            str(int((evidence["scope"] == "cross_bank").sum())),
            "detection",
        ),
        EvalQuestion(
            "q10",
            "How many shared device evidence rows are there?",
            str(
                int(
                    (evidence["evidence_type"] == "cross_bank_shared_device").sum()
                )
            ),
            "detection",
        ),
        EvalQuestion(
            "q11",
            "Which model version produced the risk scores?",
            str(scores["model_version"].iloc[0]),
            "detection",
        ),
        EvalQuestion(
            "q12",
            "Which account has the highest risk score?",
            top_account,
            "detection",
        ),
        # --- detection: scoped to one ring or account -------------------
        EvalQuestion(
            "q13",
            f"How many accounts are in {ring_id}?",
            str(len(ring_alertable)),
            "detection",
            "unqualified, so the answer is the alertable population, not "
            "every member: same-bank context members are not alerts",
        ),
        EvalQuestion(
            "q14",
            f"How many banks does {ring_id} span?",
            str(len(ring_banks)),
            "detection",
        ),
        EvalQuestion(
            "q15",
            f"Which banks are involved in {ring_id}?",
            ", ".join(ring_banks),
            "detection",
        ),
        EvalQuestion(
            "q16",
            f"What is the risk score for account {top_account}?",
            f"{top_score:.3f}",
            "detection",
        ),
        EvalQuestion(
            "q17",
            f"Which bank holds account {top_account}?",
            top_bank,
            "detection",
        ),
        # --- declined: only ground_truth can answer these ---------------
        EvalQuestion(
            "q18",
            "How many accounts forwarded money onward as pass-through mules "
            "according to the labels?",
            DECLINED,
            "declined",
            f"is_passthrough_mule is {int(passthrough.sum())} and is_complicit is "
            f"{int(complicit.sum())}; neither is queryable, and the question "
            "names which it means so a refusal is the only correct answer",
        ),
        EvalQuestion(
            "q19",
            "How many true rings are there in the ground truth?",
            DECLINED,
            "declined",
            f"truth is {labelled['ring_id'].nunique()} vs {len(rings)} detected",
        ),
        EvalQuestion(
            "q20",
            "How many true rings span three banks according to the labels?",
            DECLINED,
            "declined",
            f"truth is {int((per_ring_banks == 3).sum())}",
        ),
        EvalQuestion(
            "q21",
            "How many accounts are in the hard_negative_dual_bank cohort?",
            DECLINED,
            "declined",
            "dual-bank customers reuse ids legitimately; the count is a label",
        ),
        EvalQuestion(
            "q22",
            "How many mules have the role of collector?",
            DECLINED,
            "declined",
            "role is a ground-truth column",
        ),
        EvalQuestion(
            "q23",
            "How many mule accounts follow the fast_passthrough_only pattern?",
            DECLINED,
            "declined",
            "pattern_type is a ground-truth column",
        ),
        EvalQuestion(
            "q24",
            "Is account "
            + top_account
            + " actually a mule according to the ground truth?",
            DECLINED,
            "declined",
            "asking the agent to confirm a label it must not read",
        ),
        EvalQuestion(
            "q25",
            "What is the detector's recall against the labelled mules?",
            DECLINED,
            "declined",
            "recall needs labels; the agent must not estimate it",
        ),
    ]

    if len(questions) != EXPECTED_COUNT:  # pragma: no cover - guard
        raise AssertionError(
            f"expected {EXPECTED_COUNT} questions, built {len(questions)}"
        )
    return questions
