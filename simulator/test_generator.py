"""Tests for the MuleNet synthetic data generator.

These run against a scaled-down world (2000 accounts/bank) so the suite stays
fast; ``SimParams()`` defaults remain the full 30k-account dataset and
``test_default_params_are_full_scale`` guards that.

A large block here is *leak regression*. The first version of this generator
let a single bank identify mules from amount roundness and amount magnitude
alone, and made within-bank document reuse a mule-only property. Those tests
are named ``test_leak_*`` and exist so the giveaways cannot quietly return.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from shared.contracts import Account, Device, KycDocument, KycSubmission, Transfer
from simulator.generator import (
    BANKS,
    COHORT_BUSINESS,
    COHORT_DUAL_BANK,
    COHORT_FAMILY,
    COHORT_MULE,
    COHORT_NORMAL,
    DEFAULT_SPAN_MIX,
    OUTPUT_FILES,
    PATTERN_TYPES,
    RING_SIZE_MAX,
    RING_SIZE_MIN,
    SimParams,
    _span_counts,
    generate,
)
from simulator.kyc import display_id_number, render_id_card

TEST_PARAMS = SimParams(accounts_per_bank=2000, kyc_image_sample=0, seed=7)
TEST_KEY = b"test-key"
RINGS_PER_PATTERN = TEST_PARAMS.rings_per_pattern
TOTAL_RINGS = RINGS_PER_PATTERN * len(PATTERN_TYPES)
#: Rings per pattern at each span, derived from the configured mix rather
#: than hardcoded, so changing ``span_mix`` moves the tests with it.
SPAN_COUNTS = _span_counts(RINGS_PER_PATTERN, TEST_PARAMS.span_mix)
SINGLE_BANK_PER_PATTERN = SPAN_COUNTS[0]
CROSS_BANK_PER_PATTERN = sum(SPAN_COUNTS[1:])

HEX64 = re.compile(r"^[0-9a-f]{64}$")
RAW_KEY_PATTERN = re.compile(r"BANK_[ABC]:\d+")

#: Every column that must carry an HMAC digest rather than a raw identifier.
HASHED_COLUMNS: dict[str, tuple[str, ...]] = {
    "accounts.parquet": ("hashed_account_id",),
    "transfers.parquet": ("transfer_id", "from_hashed_account_id", "to_hashed_account_id"),
    "devices.parquet": ("device_id", "hashed_account_id", "device_fingerprint"),
    "kyc_documents.parquet": ("doc_id", "hashed_account_id", "hashed_id_number"),
    "kyc_submissions.parquet": ("submission_id", "hashed_account_id"),
    "ground_truth.parquet": ("hashed_account_id",),
}


def _build(out_dir, params: SimParams = TEST_PARAMS) -> dict[str, pd.DataFrame]:
    generate(params, hash_key=TEST_KEY, out_dir=out_dir, render_images=False)
    return {name: pd.read_parquet(out_dir / name) for name in OUTPUT_FILES}


@pytest.fixture(scope="session")
def data(tmp_path_factory) -> dict[str, pd.DataFrame]:
    return _build(tmp_path_factory.mktemp("mulenet_sim"))


@pytest.fixture(scope="session")
def truth(data) -> pd.DataFrame:
    return data["ground_truth.parquet"]


@pytest.fixture(scope="session")
def mules(truth) -> pd.DataFrame:
    return truth[truth["is_mule"]]


# --------------------------------------------------------------------- rings


def test_default_params_are_full_scale() -> None:
    params = SimParams()
    assert params.accounts_per_bank == 10_000
    assert params.n_days == 30


def test_thirty_rings_per_pattern_type(mules) -> None:
    per_pattern = mules.groupby("pattern_type")["ring_id"].nunique()
    assert set(per_pattern.index) == set(PATTERN_TYPES)
    assert per_pattern.to_dict() == dict.fromkeys(PATTERN_TYPES, RINGS_PER_PATTERN)
    assert mules["ring_id"].nunique() == TOTAL_RINGS


def test_ring_sizes_cover_the_whole_range_without_clustering(mules) -> None:
    sizes = mules.groupby("ring_id").size()
    assert sizes.min() == RING_SIZE_MIN
    assert sizes.max() == RING_SIZE_MAX
    # a distribution piled up on the mean would make ring size uninformative
    assert sizes.nunique() >= RING_SIZE_MAX - RING_SIZE_MIN
    assert sizes.std() > 2.0


def test_mule_share_stays_under_five_percent_at_full_scale(truth) -> None:
    """Ring count is scale-free, so the full-scale ratio is checkable here."""
    full_scale_accounts = SimParams().accounts_per_bank * len(BANKS)
    assert int(truth["is_mule"].sum()) / full_scale_accounts < 0.05


def _ring_span_table(mules: pd.DataFrame) -> pd.DataFrame:
    """One row per ring: observed bank spread, recorded span, pattern type."""
    return mules.groupby("ring_id").agg(
        observed_banks=("bank_id", "nunique"),
        recorded_banks=("ring_bank_count", "first"),
        pattern_type=("pattern_type", "first"),
    )


def test_recorded_ring_bank_count_matches_the_observed_bank_spread(mules) -> None:
    """``ring_bank_count`` is the single source of truth for a ring's span.

    There is deliberately no second ``banks_spanned`` column: two columns
    that could disagree would be worse than one that is checked here.
    """
    rings = _ring_span_table(mules)
    assert (rings["observed_banks"] == rings["recorded_banks"]).all()
    assert set(rings["recorded_banks"]) <= {1, 2, 3}


def test_span_mix_produces_the_configured_control_group(mules) -> None:
    """Single-bank rings are the control group; they must actually exist."""
    rings = _ring_span_table(mules)
    counts = rings["recorded_banks"].value_counts().to_dict()
    expected = {
        span: count * len(PATTERN_TYPES)
        for span, count in enumerate(SPAN_COUNTS, start=1)
        if count
    }
    assert counts == expected, counts
    assert sum(counts.values()) == TOTAL_RINGS
    assert counts[1] > 0, "no same-bank rings: the experiment has no control group"


def test_every_pattern_type_appears_at_every_span(mules) -> None:
    """Span and pattern must not be confounded.

    If, say, all single-bank rings were ``fast_passthrough_only``, a
    difference between spans could just be a difference between patterns.
    """
    rings = _ring_span_table(mules)
    table = pd.crosstab(rings["pattern_type"], rings["recorded_banks"])
    for span, count in enumerate(SPAN_COUNTS, start=1):
        if not count:
            continue
        assert (table[span] == count).all(), table.to_dict()


def test_span_mix_is_configurable(tmp_path) -> None:
    """The mix is a parameter, not a constant baked into the generator."""
    assert SimParams().span_mix == DEFAULT_SPAN_MIX
    params = SimParams(
        accounts_per_bank=TEST_PARAMS.accounts_per_bank,
        kyc_image_sample=0,
        span_mix=(1.0, 0.0, 0.0),
        seed=TEST_PARAMS.seed,
    )
    truth = _build(tmp_path / "all_single", params)["ground_truth.parquet"]
    rings = _ring_span_table(truth[truth["is_mule"]])
    assert set(rings["recorded_banks"]) == {1}
    assert len(rings) == TOTAL_RINGS


def test_span_counts_allocate_every_ring() -> None:
    assert _span_counts(30, (0.30, 0.40, 0.30)) == [9, 12, 9]
    # an awkward split still sums back to the ring count
    assert sum(_span_counts(7, (0.30, 0.40, 0.30))) == 7
    assert _span_counts(10, (0.0, 1.0)) == [0, 10]
    for bad in ((), (0.0, 0.0), (-1.0, 2.0), (1.0, 1.0, 1.0, 1.0)):
        with pytest.raises(ValueError):
            _span_counts(30, bad)


def test_every_ring_has_collector_mules_and_sources(mules) -> None:
    roles = mules.groupby("ring_id")["role"].agg(set)
    for ring_id, present in roles.items():
        assert present == {"collector", "mule", "source"}, ring_id
    assert (mules.groupby("ring_id")["role"].apply(lambda s: (s == "collector").sum()) == 1).all()


def test_most_ring_members_are_newly_opened_but_not_all(mules, data) -> None:
    """New accounts stay a ring signal, without being a mule-only property.

    If *every* mule were brand new, account age alone would separate rings
    inside a single bank, so a minority of members are aged bought accounts.
    """
    accounts = data["accounts.parquet"].set_index("hashed_account_id")
    opened = accounts.loc[mules["hashed_account_id"], "opened_at"]
    window_start = data["transfers.parquet"]["timestamp"].min()
    fresh = opened > window_start - pd.Timedelta(days=30)
    assert 0.4 < fresh.mean() < 0.85

    # and plenty of ordinary accounts are brand new too
    truth = data["ground_truth.parquet"].set_index("hashed_account_id")
    legit = accounts.loc[truth.index[~truth["is_mule"]], "opened_at"]
    assert (legit > window_start - pd.Timedelta(days=30)).mean() > 0.05


# ------------------------------------------------------------ pass-through


def test_mule_passthrough_hops_complete_under_30_minutes(mules, data) -> None:
    transfers = data["transfers.parquet"]
    ring_of = dict(zip(mules["hashed_account_id"], mules["ring_id"], strict=True))
    role_of = dict(zip(mules["hashed_account_id"], mules["role"], strict=True))

    mule_ids = {a for a, r in role_of.items() if r == "mule"}
    collector_ids = {a for a, r in role_of.items() if r == "collector"}

    hops = transfers[
        transfers["from_hashed_account_id"].isin(mule_ids)
        & transfers["to_hashed_account_id"].isin(collector_ids)
    ]
    hops = hops[
        [
            ring_of[f] == ring_of[t]
            for f, t in zip(
                hops["from_hashed_account_id"], hops["to_hashed_account_id"], strict=True
            )
        ]
    ]
    assert len(hops) > 50, "expected a meaningful number of pass-through hops"

    inbound = transfers[transfers["to_hashed_account_id"].isin(mule_ids)]
    gaps: list[float] = []
    for account, group in hops.groupby("from_hashed_account_id"):
        arrivals = inbound.loc[inbound["to_hashed_account_id"] == account, "timestamp"].to_numpy()
        for out_ts in group["timestamp"].to_numpy():
            prior = arrivals[arrivals <= out_ts]
            assert len(prior), "a hop left before anything arrived"
            gaps.append((out_ts - prior.max()) / pd.Timedelta(minutes=1))

    assert max(gaps) < 30.0
    # money always arrives before it leaves; with this many transfers a credit
    # occasionally lands on the same whole second as the hop, hence >= not >
    assert min(gaps) >= 0.0
    assert pd.Series(gaps).median() > 1.0


def test_legitimate_accounts_also_pass_money_through_fast(data, truth) -> None:
    """Pass-through *speed* must not be a mule-only shape inside one bank."""
    transfers = data["transfers.parquet"]
    bank = truth.set_index("hashed_account_id")["bank_id"]
    intra = transfers[
        transfers["from_hashed_account_id"].map(bank).to_numpy()
        == transfers["to_hashed_account_id"].map(bank).to_numpy()
    ]
    legit = set(truth.loc[~truth["is_mule"], "hashed_account_id"])

    arrivals = (
        intra[intra["to_hashed_account_id"].isin(legit)]
        .groupby("to_hashed_account_id")["timestamp"]
        .apply(list)
    )
    fast = 0
    for account, group in intra[intra["from_hashed_account_id"].isin(legit)].groupby(
        "from_hashed_account_id"
    ):
        incoming = arrivals.get(account)
        if not incoming:
            continue
        arrived = pd.Series(incoming).sort_values().to_numpy()
        for out_ts in group["timestamp"].to_numpy():
            prior = arrived[arrived <= out_ts]
            if len(prior) and (out_ts - prior.max()) <= pd.Timedelta(minutes=30):
                fast += 1
                break
    assert fast > 0.10 * len(legit), f"only {fast} legitimate accounts pass through fast"


# ------------------------------------------------------- leak regressions


def _ring_edge_mask(transfers: pd.DataFrame, mules: pd.DataFrame):
    """Transfers with a ring account on either end."""
    ids = set(mules["hashed_account_id"])
    return transfers["from_hashed_account_id"].isin(ids) | transfers[
        "to_hashed_account_id"
    ].isin(ids)


def test_leak_ring_amounts_are_no_rounder_than_legitimate_ones(data, mules) -> None:
    """Regression: ring edges were once 83% exact multiples of 100.

    Both populations must sit near the same low base rate, and neither may be
    exactly zero — an inverse tell separates just as well as a direct one.
    """
    transfers = data["transfers.parquet"]
    ring = _ring_edge_mask(transfers, mules)
    amount = transfers["amount"].astype(float)
    ring_rate = float((amount[ring] % 100 == 0).mean())
    legit_rate = float((amount[~ring] % 100 == 0).mean())

    assert 0.0 < ring_rate < 0.05, ring_rate
    assert 0.0 < legit_rate < 0.05, legit_rate
    assert ring_rate < 10 * legit_rate


def test_leak_per_account_round_ratios_overlap(data, truth) -> None:
    """The mule and legitimate round-ratio ranges must genuinely overlap.

    They previously did not: mule min 0.50 was above normal max 0.136, so one
    threshold on one feature was a near-perfect within-bank detector.
    """
    transfers = data["transfers.parquet"]
    amount = transfers["amount"].astype(float)
    events = pd.DataFrame(
        {
            "account": pd.concat(
                [transfers["from_hashed_account_id"], transfers["to_hashed_account_id"]],
                ignore_index=True,
            ),
            "round_flag": pd.concat([amount % 100 == 0] * 2, ignore_index=True),
        }
    )
    ratio = events.groupby("account")["round_flag"].mean()
    is_mule = truth.set_index("hashed_account_id")["is_mule"].reindex(ratio.index).fillna(False)

    mule_ratio, legit_ratio = ratio[is_mule], ratio[~is_mule]
    assert mule_ratio.min() <= legit_ratio.max()
    assert legit_ratio.min() <= mule_ratio.max()
    assert mule_ratio.mean() < 10 * legit_ratio.mean()


def test_leak_ring_edge_amounts_match_legitimate_magnitude(data, mules) -> None:
    """Regression: ring edges once sat at a $2500 median against $38 elsewhere."""
    transfers = data["transfers.parquet"]
    ring = _ring_edge_mask(transfers, mules)
    amount = transfers["amount"].astype(float)
    ring_median = float(amount[ring].median())
    legit_median = float(amount[~ring].median())
    assert 0.2 < ring_median / legit_median < 5.0, (ring_median, legit_median)
    # and the bulk of ring traffic sits inside ordinary peer volumes
    assert amount[ring].quantile(0.75) < amount[~ring].quantile(0.95)


def test_leak_legitimate_accounts_reuse_documents_within_a_bank(data, truth) -> None:
    """Regression: same-bank document reuse was 0.00% for every legit cohort."""
    merged = data["kyc_documents.parquet"].merge(
        truth[["hashed_account_id", "bank_id", "cohort", "is_mule"]], on="hashed_account_id"
    )
    pairs = merged[["hashed_account_id", "hashed_id_number", "bank_id"]].drop_duplicates()
    per_id_bank = pairs.groupby(["hashed_id_number", "bank_id"])["hashed_account_id"].nunique()
    shared = {k for k, v in per_id_bank.items() if v > 1}
    pairs["same_bank_reuse"] = [
        (i, b) in shared for i, b in zip(pairs["hashed_id_number"], pairs["bank_id"], strict=True)
    ]
    flag = pairs.set_index("hashed_account_id")["same_bank_reuse"]

    cohorts = truth.set_index("hashed_account_id")["cohort"]
    share = flag.groupby(cohorts.reindex(flag.index)).mean()
    assert share[COHORT_NORMAL] > 0.03, share.to_dict()
    assert share[COHORT_FAMILY] > 0.0, share.to_dict()
    assert share[COHORT_BUSINESS] > 0.0, share.to_dict()

    # The rate among plain accounts is what makes the signal ambiguous, and it
    # is scale-free: at the full 30k scale it puts thousands of legitimate
    # accounts on the wrong side of any same-bank-reuse threshold. The raw
    # counts are not comparable here because mules are a far larger share of
    # this deliberately small test world than of the real dataset.
    is_mule = truth.set_index("hashed_account_id")["is_mule"].reindex(flag.index)
    assert flag[~is_mule].sum() > 100
    assert share[COHORT_NORMAL] > 0.1 * share[COHORT_MULE], share.to_dict()


def test_leak_legitimate_accounts_share_devices_within_a_bank(data, truth) -> None:
    """Same-bank device sharing must be a crowded signal, not a ring marker."""
    merged = data["devices.parquet"].merge(
        truth[["hashed_account_id", "bank_id", "is_mule"]], on="hashed_account_id"
    )
    per_fp_bank = merged.groupby(["device_fingerprint", "bank_id"])["hashed_account_id"].nunique()
    shared = {k for k, v in per_fp_bank.items() if v > 1}
    merged["same_bank_share"] = [
        (f, b) in shared
        for f, b in zip(merged["device_fingerprint"], merged["bank_id"], strict=True)
    ]
    by_account = merged.groupby("hashed_account_id")["same_bank_share"].any()
    is_mule = truth.set_index("hashed_account_id")["is_mule"].reindex(by_account.index)
    assert by_account[~is_mule].sum() > by_account[is_mule].sum()
    assert by_account[~is_mule].mean() > 0.02


# ------------------------------------------------------------- id hygiene


def test_no_raw_or_sequential_ids_leak(data) -> None:
    for name, columns in HASHED_COLUMNS.items():
        frame = data[name]
        for column in columns:
            values = frame[column].astype(str)
            bad = values[~values.str.match(HEX64)]
            assert bad.empty, f"{name}:{column} has non-hash values {bad.head().tolist()}"


def test_no_output_column_contains_the_raw_account_key(data) -> None:
    for name, frame in data.items():
        for column in frame.columns:
            if frame[column].dtype.kind not in ("O", "U", "S"):
                continue
            sample = frame[column].head(2000).astype(str)
            assert not sample.str.contains(RAW_KEY_PATTERN).any(), f"{name}:{column}"


def test_account_ids_are_unique_and_unordered(data) -> None:
    ids = data["accounts.parquet"]["hashed_account_id"]
    assert ids.is_unique
    # a sequential/derived id scheme would leave the column sorted or clustered
    assert not ids.is_monotonic_increasing
    assert ids.str[:8].nunique() / len(ids) > 0.95


# ----------------------------------------------------------- device signal


def _cross_bank_fingerprints(devices: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = devices.merge(
        truth[["hashed_account_id", "bank_id", "cohort", "ring_id", "pattern_type"]],
        on="hashed_account_id",
    )
    spans = merged.groupby("device_fingerprint")["bank_id"].nunique()
    merged["cross_bank"] = merged["device_fingerprint"].map(spans) > 1
    return merged


def test_shared_device_rings_share_fingerprints_across_banks(data, truth) -> None:
    """Only the multi-bank rings hide their device link from a single bank."""
    merged = _cross_bank_fingerprints(data["devices.parquet"], truth)
    spans = truth.groupby("ring_id")["ring_bank_count"].first()
    for pattern in ("shared_device", "mixed"):
        rings = set(
            merged.loc[(merged["pattern_type"] == pattern) & merged["cross_bank"], "ring_id"]
        )
        assert len(rings) == CROSS_BANK_PER_PATTERN, (
            f"{pattern}: {len(rings)} rings share a device across banks"
        )
        assert all(spans[r] > 1 for r in rings)


def test_single_bank_shared_device_rings_share_within_their_one_bank(data, truth) -> None:
    """The control group: this ring's device link IS visible to its own bank."""
    merged = data["devices.parquet"].merge(
        truth[["hashed_account_id", "bank_id", "ring_id", "pattern_type", "ring_bank_count"]],
        on="hashed_account_id",
    )
    single = merged[
        (merged["ring_bank_count"] == 1)
        & merged["pattern_type"].isin(("shared_device", "mixed"))
    ]
    per_ring_fp = single.groupby(["ring_id", "device_fingerprint"])[
        "hashed_account_id"
    ].nunique()
    sharing_rings = {r for (r, _fp), n in per_ring_fp.items() if n > 1}
    assert len(sharing_rings) == SINGLE_BANK_PER_PATTERN * 2, sorted(sharing_rings)


def test_passthrough_only_rings_share_no_device(data, truth) -> None:
    merged = _cross_bank_fingerprints(data["devices.parquet"], truth)
    shared = merged.loc[merged["pattern_type"] == "fast_passthrough_only"]
    assert not shared["cross_bank"].any()


def test_hard_negative_families_also_share_devices(data, truth) -> None:
    """The device signal alone must not separate rings from families."""
    merged = data["devices.parquet"].merge(
        truth[["hashed_account_id", "bank_id", "cohort"]], on="hashed_account_id"
    )
    families = merged[merged["cohort"] == COHORT_FAMILY]
    per_fp = families.groupby("device_fingerprint")["hashed_account_id"].nunique()
    shared_fps = per_fp[per_fp > 1]
    # one fingerprint per family, plus any household device a family member
    # happens to also share with a plain account at the same bank
    assert len(shared_fps) >= TEST_PARAMS.n_families

    cross_bank = families[families["device_fingerprint"].isin(shared_fps.index)]
    spans = cross_bank.groupby("device_fingerprint")["bank_id"].nunique()
    assert (spans > 1).sum() >= 1, "some families must share a device across banks"


# --------------------------------------------------------- document signal


def _cross_bank_documents(documents: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = documents.merge(
        truth[["hashed_account_id", "bank_id", "cohort", "ring_id", "pattern_type"]],
        on="hashed_account_id",
    )
    id_span = merged.groupby("hashed_id_number")["bank_id"].nunique()
    merged["reused_id_across_banks"] = merged["hashed_id_number"].map(id_span) > 1
    image_span = merged.dropna(subset=["photo_embedding_ref"]).groupby("photo_embedding_ref")[
        "bank_id"
    ].nunique()
    merged["reused_image_across_banks"] = (
        merged["photo_embedding_ref"].map(image_span).fillna(1) > 1
    )
    return merged


def test_reused_document_rings_reuse_ids_and_images_across_banks(data, truth) -> None:
    merged = _cross_bank_documents(data["kyc_documents.parquet"], truth)
    spans = truth.groupby("ring_id")["ring_bank_count"].first()
    for pattern in ("reused_document", "mixed"):
        subset = merged[merged["pattern_type"] == pattern]
        by_id = set(subset.loc[subset["reused_id_across_banks"], "ring_id"])
        by_image = set(subset.loc[subset["reused_image_across_banks"], "ring_id"])
        assert len(by_id) == CROSS_BANK_PER_PATTERN, (
            f"{pattern}: {len(by_id)} rings reuse an id number across banks"
        )
        assert len(by_image) == CROSS_BANK_PER_PATTERN, (
            f"{pattern}: {len(by_image)} rings reuse an image across banks"
        )
        assert all(spans[r] > 1 for r in by_id | by_image)


def test_single_bank_reuse_rings_reuse_documents_inside_their_one_bank(data, truth) -> None:
    """The control group again: one bank can see this reuse without a join."""
    merged = data["kyc_documents.parquet"].merge(
        truth[["hashed_account_id", "ring_id", "pattern_type", "ring_bank_count"]],
        on="hashed_account_id",
    )
    single = merged[
        (merged["ring_bank_count"] == 1)
        & merged["pattern_type"].isin(("reused_document", "mixed"))
    ]
    per_ring_id = single.groupby(["ring_id", "hashed_id_number"])["hashed_account_id"].nunique()
    sharing_rings = {r for (r, _i), n in per_ring_id.items() if n > 1}
    assert len(sharing_rings) == SINGLE_BANK_PER_PATTERN * 2, sorted(sharing_rings)


def test_single_bank_rings_have_no_cross_bank_edges_devices_or_documents(data, truth) -> None:
    """A same-bank ring must be genuinely same-bank in every artefact.

    Its members, its money, its devices and its identity documents all sit
    at one institution -- otherwise it is not a control for anything.
    """
    single_ids = set(truth.loc[truth["ring_bank_count"] == 1, "hashed_account_id"])
    assert single_ids
    bank = truth.set_index("hashed_account_id")["bank_id"]

    members = truth[truth["hashed_account_id"].isin(single_ids)]
    assert (members.groupby("ring_id")["bank_id"].nunique() == 1).all()

    transfers = data["transfers.parquet"]
    internal = transfers[
        transfers["from_hashed_account_id"].isin(single_ids)
        & transfers["to_hashed_account_id"].isin(single_ids)
    ]
    assert len(internal) > 0, "same-bank rings moved no money between members"
    assert (
        internal["from_hashed_account_id"].map(bank).to_numpy()
        == internal["to_hashed_account_id"].map(bank).to_numpy()
    ).all()

    devices = data["devices.parquet"]
    ring_fps = set(
        devices.loc[devices["hashed_account_id"].isin(single_ids), "device_fingerprint"]
    )
    fp_span = devices.merge(truth[["hashed_account_id", "bank_id"]], on="hashed_account_id")
    fp_span = fp_span[fp_span["device_fingerprint"].isin(ring_fps)]
    assert fp_span.groupby("device_fingerprint")["bank_id"].nunique().max() == 1

    docs = data["kyc_documents.parquet"].merge(
        truth[["hashed_account_id", "bank_id"]], on="hashed_account_id"
    )
    ring_nums = set(docs.loc[docs["hashed_account_id"].isin(single_ids), "hashed_id_number"])
    reach = docs[docs["hashed_id_number"].isin(ring_nums)]
    assert reach.groupby("hashed_id_number")["bank_id"].nunique().max() == 1


def test_other_ring_patterns_do_not_reuse_documents(data, truth) -> None:
    merged = _cross_bank_documents(data["kyc_documents.parquet"], truth)
    subset = merged[merged["pattern_type"].isin(("shared_device", "fast_passthrough_only"))]
    assert not subset["reused_id_across_banks"].any()


def test_dual_bank_people_legitimately_reuse_documents(data, truth) -> None:
    """The document signal alone must not separate rings from real dual-bank customers."""
    merged = _cross_bank_documents(data["kyc_documents.parquet"], truth)
    dual = merged[merged["cohort"] == COHORT_DUAL_BANK]
    assert dual["reused_id_across_banks"].all()


def test_cross_bank_signals_survive_and_stay_ambiguous(data, truth) -> None:
    """The point of the project: cross-bank reuse is present *and* crowded.

    Mules must still share devices and documents across banks (or there is
    nothing for a joint model to find), and legitimate cohorts must still do
    the same in comparable numbers (or a single cross-bank signal would be
    enough on its own and no join of signals would be needed).
    """
    devices = _cross_bank_fingerprints(data["devices.parquet"], truth)
    docs = _cross_bank_documents(data["kyc_documents.parquet"], truth)

    dev_share = devices.groupby("cohort")["cross_bank"].any()
    assert dev_share[COHORT_MULE]
    assert dev_share[COHORT_FAMILY]
    assert dev_share[COHORT_DUAL_BANK]

    doc_share = docs.groupby("cohort")["reused_id_across_banks"].any()
    assert doc_share[COHORT_MULE]
    assert doc_share[COHORT_DUAL_BANK]

    # legitimate cross-bank sharers must not be rarer than the mules
    legit_accounts = set(truth.loc[~truth["is_mule"], "hashed_account_id"])
    for frame, column in ((devices, "cross_bank"), (docs, "reused_id_across_banks")):
        flagged = frame.loc[frame[column], "hashed_account_id"]
        legit = flagged.isin(legit_accounts).sum()
        assert legit >= (~flagged.isin(legit_accounts)).sum(), column


# -------------------------------------------------------------- cohorts


def test_hard_negative_cohort_sizes(truth) -> None:
    counts = truth["cohort"].value_counts()
    assert counts[COHORT_BUSINESS] == TEST_PARAMS.n_businesses
    assert counts[COHORT_DUAL_BANK] == TEST_PARAMS.n_dual_bank * 2
    assert 3 * TEST_PARAMS.n_families <= counts[COHORT_FAMILY] <= 5 * TEST_PARAMS.n_families
    assert counts[COHORT_MULE] == truth["is_mule"].sum()


def test_hard_negatives_are_never_labelled_mules(truth) -> None:
    hard = truth[truth["is_hard_negative"]]
    assert not hard["is_mule"].any()
    assert (hard["ring_id"] == "").all()
    assert set(truth.loc[truth["is_mule"], "cohort"]) == {COHORT_MULE}


def test_banks_and_row_counts(data) -> None:
    accounts = data["accounts.parquet"]
    assert set(accounts["bank_id"]) == set(BANKS)
    assert len(accounts) == TEST_PARAMS.accounts_per_bank * len(BANKS)
    assert len(data["ground_truth.parquet"]) == len(accounts)
    assert len(data["kyc_submissions.parquet"]) == len(accounts)
    assert len(data["kyc_documents.parquet"]) == len(accounts) * 2


# ------------------------------------------------------------- contracts


def _doc_record(row: dict) -> dict:
    """Normalise the nullable photo ref: pandas 3.x reads a null back as NaN."""
    row = dict(row)
    ref = row.get("photo_embedding_ref")
    if ref is None or (isinstance(ref, float) and pd.isna(ref)):
        row["photo_embedding_ref"] = None
    return row


def test_rows_validate_against_shared_contracts(data) -> None:
    for row in data["accounts.parquet"].head(50).to_dict(orient="records"):
        Account(**row)
    for row in data["transfers.parquet"].head(50).to_dict(orient="records"):
        Transfer(**row)
    for row in data["devices.parquet"].head(50).to_dict(orient="records"):
        Device(**row)
    for row in data["kyc_documents.parquet"].head(50).to_dict(orient="records"):
        KycDocument(**_doc_record(row))


def test_submissions_rebuild_into_the_shared_model(data) -> None:
    documents = data["kyc_documents.parquet"].set_index("doc_id")
    for row in data["kyc_submissions.parquet"].head(25).to_dict(orient="records"):
        docs = [
            KycDocument(**_doc_record(documents.loc[d].to_dict()), doc_id=d)
            for d in row["doc_ids"]
        ]
        submission = KycSubmission(
            submission_id=row["submission_id"],
            hashed_account_id=row["hashed_account_id"],
            bank_id=row["bank_id"],
            submitted_at=row["submitted_at"],
            documents=docs,
        )
        assert {d.hashed_account_id for d in submission.documents} == {row["hashed_account_id"]}
        assert {d.doc_type for d in submission.documents} == {"id_card", "address_proof"}


# ------------------------------------------------------------ determinism


def test_same_seed_produces_identical_output(tmp_path) -> None:
    first = _build(tmp_path / "a")
    second = _build(tmp_path / "b")
    for name in OUTPUT_FILES:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_different_seed_produces_different_output(tmp_path) -> None:
    first = _build(tmp_path / "a")
    other = _build(
        tmp_path / "c",
        SimParams(accounts_per_bank=TEST_PARAMS.accounts_per_bank, kyc_image_sample=0, seed=8),
    )
    assert not first["transfers.parquet"].equals(other["transfers.parquet"])


# -------------------------------------------------------------- kyc images


def test_rendered_id_card_is_a_valid_png(tmp_path) -> None:
    from PIL import Image

    path = render_id_card(
        tmp_path / "card.png",
        holder_name="SYNTHETIC HOLDER ABC123",
        id_number_display=display_id_number("a" * 64),
        address="100 MAPLE ST, UNIT 1, TESTVILLE ZZ-0001",
        issuer="SPECIMEN REGISTRY",
        accent=(36, 78, 140),
    )
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size == (640, 400)


def test_display_id_number_is_obviously_fake() -> None:
    rendered = display_id_number("deadbeefcafe" + "0" * 52)
    assert rendered.startswith("SYN-")
    assert rendered == "SYN-DEAD-BEEF-CAFE"
