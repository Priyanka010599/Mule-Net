"""Tests for the coalition sweep.

The sweep's numbers come from the real detector over the real dataset, so
they are not what gets asserted here -- pinning literals would turn a
legitimate re-run into a failure. What gets asserted is the **machinery
around** the detector, because that is where this could be wrong in a way
nobody notices:

* the coalition view must be a strict subgraph, and must not leak an edge
  with one endpoint outside the coalition;
* the split that makes a fourth bank must move only ``bank_id``, and must
  move the same accounts on every run;
* ground truth must follow the split, or every four-bank coalition's
  reachability is computed against the pre-split banks;
* and the published curve must actually be a mean over *every* subset of
  a size, not over whichever ones were convenient.

The heavy tests skip on a fresh clone, following ``ml/conftest.py``'s
convention: a missing ``simulator/out/`` means the generator has not run,
which is an operator step rather than a defect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml import coalition
from ml.config import MlConfig
from ml.conftest import require_artefact
from ml.features import load_tables


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def tables(cfg: MlConfig) -> dict[str, pd.DataFrame]:
    require_artefact(cfg.table_path("transfers"), "python -m simulator.generator")
    return load_tables(cfg)


# --------------------------------------------------------------------------
# a coalition's view is a strict subgraph
# --------------------------------------------------------------------------


def test_a_coalition_sees_only_its_own_banks_accounts(
    tables: dict[str, pd.DataFrame],
) -> None:
    view = coalition.coalition_tables(tables, ("BANK_A", "BANK_B"))
    assert set(view["accounts"]["bank_id"]) == {"BANK_A", "BANK_B"}


def test_no_transfer_survives_with_an_endpoint_outside_the_coalition(
    tables: dict[str, pd.DataFrame],
) -> None:
    """The leak that would make every coalition look better than it is.

    A bank does see its own side of a payment to an outside account, but
    the coalition cannot build a counterparty feature from an account it
    does not hold. Keeping those edges would let a two-bank coalition read
    structure belonging to the third.
    """
    banks = ("BANK_A", "BANK_B")
    view = coalition.coalition_tables(tables, banks)
    inside = set(view["accounts"]["hashed_account_id"])
    assert view["transfers"]["from_hashed_account_id"].isin(inside).all()
    assert view["transfers"]["to_hashed_account_id"].isin(inside).all()


def test_the_full_coalition_is_the_whole_dataset(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Joining every bank must reconstruct the published graph exactly.

    If it does not, the top of the curve is not the number the rest of the
    repo reports, and the whole series is measuring something else.
    """
    banks = tuple(sorted(set(tables["accounts"]["bank_id"])))
    view = coalition.coalition_tables(tables, banks)
    assert len(view["accounts"]) == len(tables["accounts"])
    assert len(view["transfers"]) == len(tables["transfers"])


def test_a_coalition_view_never_grows(tables: dict[str, pd.DataFrame]) -> None:
    """Monotone by construction: adding a bank can only add rows."""
    two = coalition.coalition_tables(tables, ("BANK_A", "BANK_B"))
    three = coalition.coalition_tables(tables, ("BANK_A", "BANK_B", "BANK_C"))
    assert len(two["accounts"]) <= len(three["accounts"])
    assert len(two["transfers"]) <= len(three["transfers"])


def test_the_coalition_view_carries_no_ground_truth(
    tables: dict[str, pd.DataFrame],
) -> None:
    """A coalition does not get to see the labels it is scored against."""
    view = coalition.coalition_tables(tables, ("BANK_A",))
    assert "ground_truth" not in view


# --------------------------------------------------------------------------
# the fourth bank is a relabelling, and nothing more
# --------------------------------------------------------------------------


def test_the_split_moves_only_the_bank_id(tables: dict[str, pd.DataFrame]) -> None:
    """No account added, removed or re-keyed; no transfer touched.

    This is the whole claim the four-bank curve rests on. If the split
    could change anything else, the curve would be measured on a
    different market rather than a redivided one.
    """
    before = tables["accounts"]
    after = coalition.split_one_bank(before)
    assert len(after) == len(before)
    assert after["hashed_account_id"].tolist() == before["hashed_account_id"].tolist()
    for column in before.columns:
        if column != "bank_id":
            assert after[column].equals(before[column]), column


def test_the_split_only_takes_accounts_from_the_named_bank(
    tables: dict[str, pd.DataFrame],
) -> None:
    before = tables["accounts"]
    after = coalition.split_one_bank(before)
    moved = before["bank_id"] != after["bank_id"]
    assert (before.loc[moved, "bank_id"] == coalition.SPLIT_SOURCE_BANK).all()
    assert (after.loc[moved, "bank_id"] == coalition.SYNTHETIC_BANK).all()


def test_the_split_produces_four_banks_and_leaves_the_others_intact(
    tables: dict[str, pd.DataFrame],
) -> None:
    after = coalition.split_one_bank(tables["accounts"])
    assert len(set(after["bank_id"])) == 4
    assert coalition.SYNTHETIC_BANK in set(after["bank_id"])
    # The split bank still exists: it lost half its customers, not all.
    assert coalition.SPLIT_SOURCE_BANK in set(after["bank_id"])


def test_the_split_is_identical_on_every_run(
    tables: dict[str, pd.DataFrame],
) -> None:
    """A salted hash would make the published curve unreproducible.

    Python's own ``hash`` is salted per process, so using it would have
    given a different fourth bank on every run -- and the committed
    fallback would then disagree with anything recomputed from it.
    """
    first = coalition.split_one_bank(tables["accounts"])
    second = coalition.split_one_bank(tables["accounts"])
    assert first["bank_id"].equals(second["bank_id"])


def test_the_split_is_roughly_even(tables: dict[str, pd.DataFrame]) -> None:
    """Hashing on the id should halve the bank, not shave it.

    A lopsided split would make the fourth bank unrepresentative of the
    one it came from, which is the property that lets the curve be read
    as "four banks" at all.
    """
    after = coalition.split_one_bank(tables["accounts"])
    source = (after["bank_id"] == coalition.SPLIT_SOURCE_BANK).sum()
    synthetic = (after["bank_id"] == coalition.SYNTHETIC_BANK).sum()
    assert abs(source - synthetic) / (source + synthetic) < 0.05


def test_ground_truth_follows_the_split(tables: dict[str, pd.DataFrame]) -> None:
    """Otherwise every four-bank reachability count is against the old banks.

    ``ground_truth`` carries its own ``bank_id``, and ``rings_reachable``
    is computed from it. Left unmapped, a ring member split out to
    ``BANK_D`` would still be recorded at ``BANK_C``, and a coalition
    holding only ``BANK_D`` would be credited with rings it cannot see.
    """
    accounts = coalition.split_one_bank(tables["accounts"])
    truth = coalition._reassign_truth_banks(tables["ground_truth"], accounts)
    bank = accounts.set_index("hashed_account_id")["bank_id"]
    mapped = truth["hashed_account_id"].map(bank)
    assert (truth["bank_id"] == mapped).all()
    assert coalition.SYNTHETIC_BANK in set(truth["bank_id"])


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------


def test_one_bank_alone_flags_nothing_under_the_shipped_rule(
    tables: dict[str, pd.DataFrame], cfg: MlConfig
) -> None:
    """The zero at N=1 is the rule declining to fire, and it is load-bearing.

    ``ml/rule.py``: the bank-spread clause is uncomputable in the ``sb_``
    scope, so the literal single-bank transcription fires on exactly zero
    accounts. The generous variant must be non-zero on the same
    coalition, because that is the baseline the page quotes gains
    against -- if both were zero the control would be comparing against
    nothing.
    """
    truth = coalition._true_rings(tables["ground_truth"])
    result = coalition.evaluate_coalition(tables, ("BANK_A",), truth, cfg)
    assert result.flagged_accounts == 0
    assert result.ring_recall == 0.0
    assert result.flagged_accounts_no_spread > 0
    assert result.ring_recall_no_spread > 0.0


def test_joining_every_bank_reproduces_the_published_detector(
    tables: dict[str, pd.DataFrame], cfg: MlConfig
) -> None:
    """The top of the curve must be the detector this repo ships.

    676 flagged accounts and 45 rings are what ``ml/publish.py`` writes
    and what ``ui/README.md`` quotes. Asserting the relationship rather
    than the literals: the full coalition is the whole dataset, so it must
    agree with a direct run of the rule on it.
    """
    import pandas as pd  # noqa: PLC0415 - local to keep the import list honest

    from ml.features import build_features  # noqa: PLC0415
    from ml.rule import velocity_decision  # noqa: PLC0415

    banks = tuple(sorted(set(tables["accounts"]["bank_id"])))
    truth = coalition._true_rings(tables["ground_truth"])
    result = coalition.evaluate_coalition(tables, banks, truth, cfg)

    features = build_features(cfg=cfg, tables=tables)
    direct = pd.Index(
        features.loc[velocity_decision(features, "cb_"), "hashed_account_id"]
    )
    assert result.flagged_accounts == len(direct)


def test_by_size_averages_over_every_coalition_of_that_size() -> None:
    """A mean over "some of the pairs" is a number nobody can reproduce."""
    made = [
        coalition.CoalitionResult(
            banks=banks,
            accounts_visible=10,
            transfers_visible=10,
            flagged_accounts=1,
            rings_formed=0,
            true_rings_total=10,
            rings_reachable=5,
            rings_caught=caught,
        )
        for banks, caught in (
            (("A", "B"), 2),
            (("A", "C"), 4),
            (("B", "C"), 6),
        )
    ]
    rows = coalition.by_size(made)
    assert len(rows) == 1
    row = rows[0]
    assert row["banks_joined"] == 2
    assert row["coalitions"] == 3
    # (0.2 + 0.4 + 0.6) / 3
    assert row["ring_recall"] == pytest.approx(0.4)
    assert row["ring_recall_min"] == pytest.approx(0.2)
    assert row["ring_recall_max"] == pytest.approx(0.6)


def test_a_coalition_holding_no_account_in_a_ring_cannot_catch_it() -> None:
    """``rings_reachable`` is the numerator's ceiling, by definition."""
    result = coalition.CoalitionResult(
        banks=("A",),
        accounts_visible=1,
        transfers_visible=0,
        flagged_accounts=0,
        rings_formed=0,
        true_rings_total=120,
        rings_reachable=0,
        rings_caught=0,
    )
    assert result.ring_recall == 0.0
    # No division by zero when nothing is reachable.
    assert result.ring_recall_reachable == 0.0


def test_the_unconditional_and_conditional_recalls_differ() -> None:
    """Both are published, because quoting either one bare is misleading."""
    result = coalition.CoalitionResult(
        banks=("A",),
        accounts_visible=1,
        transfers_visible=1,
        flagged_accounts=1,
        rings_formed=0,
        true_rings_total=120,
        rings_reachable=60,
        rings_caught=30,
    )
    assert result.ring_recall == pytest.approx(0.25)
    assert result.ring_recall_reachable == pytest.approx(0.5)


def test_only_passthrough_roles_are_scored(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Sources are complicit but the velocity rule cannot fire on them.

    Median in-degree 0 -- they originate ring funds rather than forwarding
    them -- so scoring the rule on them would measure a mechanism it does
    not have. ``ml/stage2.py`` exists precisely because of that hole.
    """
    truth = coalition._true_rings(tables["ground_truth"])
    roles = set(
        tables["ground_truth"]
        .loc[tables["ground_truth"]["hashed_account_id"].isin(truth["hashed_account_id"])]
        ["role"]
    )
    assert roles == {"mule", "collector"}
    assert "source" not in roles
