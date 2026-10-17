"""Per-account graph features for MuleNet, computed twice per account.

Every feature is built under two *visibility scopes*:

``single_bank`` (``sb_`` prefix)
    What one bank can see on its own. Only transfers whose two endpoints
    are both held at the account's own bank exist; devices and KYC
    identities belonging to accounts at other banks are **invisible**,
    not merely zero-weighted. A device shared with an account at another
    bank contributes nothing to ``sb_shared_device_count`` because the
    single bank never observes the other side at all.

``cross_bank`` (``cb_`` prefix)
    The merged three-bank graph: every transfer, every device link,
    every KYC identity link.

The pair is the whole experiment. Feature names follow
``shared.bq_schemas.FEATURES_SCHEMA`` so the same names carry over to
BigQuery; ``scope`` there becomes the column prefix here, because the
downstream model needs both scopes side by side on one row.

Point-in-time correctness
-------------------------
Every account carries its own **cutoff**: the wall-clock instant the
feature row is supposed to be observed at. Nothing after an account's
cutoff may enter that account's row. Three separate clocks are honoured:

* transfers: only ``timestamp <= cutoff``;
* device links: only ``devices.first_seen <= cutoff``, for *both* sides
  of a shared-fingerprint pair (a link the bank has not seen yet cannot
  raise an alert);
* KYC identity links: only ``kyc_submissions.submitted_at <= cutoff``,
  again for both sides;
* ``account_age_days`` is measured against the cutoff, not the end of
  the simulation.

``build_features`` with no ``cutoff`` gives every account the last
transfer timestamp in the dataset, which reproduces the full-window
table written to ``ml/features.parquet``.

Definitions that are choices rather than facts:

*Burst*
    A maximal run of an account's own transfers (in **and** out, in the
    scope's edge set) where consecutive events are at most
    ``MlConfig.burst_gap_seconds`` apart (default 1800s = 30 min, the top
    of the simulator's 2-30 minute mule hop window). A new account or a
    larger gap starts a new burst.

*Pass-through speed*
    Per burst: seconds from the first inbound credit in that burst to the
    first outbound debit at or after it. The account's feature is the
    median over its bursts. Accounts with no such in-then-out burst get
    the sentinel ``NO_PASSTHROUGH`` (-1.0) — distinguishable from a
    genuine 0-second hop, and finite so the model can split on it.

*Pass-through ratio*
    ``amount_out_total / amount_in_total``. When ``amount_in_total`` is
    zero the ratio is undefined, so it is reported as ``0.0``: no money
    came in, therefore none of it was passed through. Accounts that pay
    out without ever being credited are still visible via
    ``amount_out_total`` and ``out_degree``, so nothing is lost.

*Round amount*
    An amount that is an exact multiple of 100.00. ``round_amount_ratio``
    is the share of the account's transfers (both directions) that are
    round; 0.0 when the account has no transfers in scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import MlConfig

#: Sentinel for "this account never showed an in-then-out burst".
NO_PASSTHROUGH = -1.0

#: An amount is "round" when it is an exact multiple of this.
ROUND_AMOUNT_UNIT = 100.0

#: Scope name -> column prefix. Order fixes the column order on disk.
SCOPES: dict[str, str] = {"single_bank": "sb_", "cross_bank": "cb_"}

#: Feature columns produced per scope, in output order.
FEATURE_NAMES: tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "unique_counterparties",
    "distinct_counterparty_banks",
    "amount_in_total",
    "amount_out_total",
    "passthrough_ratio",
    "passthrough_speed_seconds",
    "round_amount_ratio",
    "account_age_days",
    "device_count",
    "shared_device_count",
    "document_reuse_count",
    "document_reuse_flag",
)


def load_tables(cfg: MlConfig) -> dict[str, pd.DataFrame]:
    """Read all six simulator parquet tables."""
    return {
        "accounts": pd.read_parquet(cfg.table_path("accounts")),
        "transfers": pd.read_parquet(cfg.table_path("transfers")),
        "devices": pd.read_parquet(cfg.table_path("devices")),
        "kyc_submissions": pd.read_parquet(cfg.table_path("kyc_submissions")),
        "kyc_documents": pd.read_parquet(cfg.table_path("kyc_documents")),
        "ground_truth": pd.read_parquet(cfg.table_path("ground_truth")),
    }


def _bank_of(accounts: pd.DataFrame) -> pd.Series:
    """hashed_account_id -> bank_id."""
    return accounts.set_index("hashed_account_id")["bank_id"]


def identity_links(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Device and KYC-identity links, each stamped with its observation time.

    Both frames have columns ``hashed_account_id``, the link key, and
    ``observed_at`` — the earliest instant at which that account/key pair
    became visible. ``kyc_documents`` carries no timestamp of its own, so
    the identity link inherits ``kyc_submissions.submitted_at`` for the
    account that filed the document.
    """
    devices = tables["devices"][["hashed_account_id", "device_fingerprint", "first_seen"]]
    devices = devices.rename(columns={"first_seen": "observed_at"})

    submitted = tables["kyc_submissions"].groupby("hashed_account_id")["submitted_at"].min()
    documents = tables["kyc_documents"][["hashed_account_id", "hashed_id_number"]].copy()
    documents["observed_at"] = documents["hashed_account_id"].map(submitted)
    documents = documents.dropna(subset=["observed_at"])
    return {"device_fingerprint": devices, "hashed_id_number": documents}


def _edge_events(transfers: pd.DataFrame, bank: pd.Series) -> pd.DataFrame:
    """Explode transfers into one row per (account, direction) endpoint.

    ``amount`` arrives as ``decimal.Decimal`` (NUMERIC(15,2)); it is cast
    to float64 here once, explicitly, rather than assumed numeric.
    """
    amount = transfers["amount"].astype("float64")
    src = transfers["from_hashed_account_id"]
    dst = transfers["to_hashed_account_id"]
    out_side = pd.DataFrame(
        {
            "account": src,
            "counterparty": dst,
            "counterparty_bank": dst.map(bank),
            "amount": amount,
            "timestamp": transfers["timestamp"],
            "is_inbound": False,
        }
    )
    in_side = pd.DataFrame(
        {
            "account": dst,
            "counterparty": src,
            "counterparty_bank": src.map(bank),
            "amount": amount,
            "timestamp": transfers["timestamp"],
            "is_inbound": True,
        }
    )
    events = pd.concat([out_side, in_side], ignore_index=True)
    return events.sort_values(["account", "timestamp"], kind="stable", ignore_index=True)


def _passthrough_speed(events: pd.DataFrame, index: pd.Index, gap_seconds: int) -> pd.Series:
    """Median in-then-out burst latency per account, in seconds."""
    if events.empty:
        return pd.Series(NO_PASSTHROUGH, index=index, dtype="float64")

    ts = events["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64")
    account = events["account"].to_numpy()
    new_account = np.empty(len(events), dtype=bool)
    new_account[0] = True
    new_account[1:] = account[1:] != account[:-1]
    gap = np.empty(len(events), dtype=bool)
    gap[0] = False
    gap[1:] = (ts[1:] - ts[:-1]) > gap_seconds
    burst = np.cumsum(new_account | gap)

    work = pd.DataFrame(
        {
            "account": account,
            "burst": burst,
            "ts": ts,
            "is_inbound": events["is_inbound"].to_numpy(),
        }
    )
    first_in = work.loc[work["is_inbound"]].groupby("burst")["ts"].min()
    outbound = work.loc[~work["is_inbound"]].copy()
    outbound["first_in"] = outbound["burst"].map(first_in)
    hops = outbound.loc[outbound["ts"] >= outbound["first_in"]]
    if hops.empty:
        return pd.Series(NO_PASSTHROUGH, index=index, dtype="float64")

    first_out = hops.groupby(["account", "burst"], sort=False).agg(
        ts=("ts", "min"), first_in=("first_in", "min")
    )
    delta = (first_out["ts"] - first_out["first_in"]).astype("float64")
    median = delta.groupby(level="account").median()
    return median.reindex(index).fillna(NO_PASSTHROUGH).astype("float64")


def _visible_links(links: pd.DataFrame, key: str, cutoff: pd.Series) -> pd.DataFrame:
    """One row per (account, key) that the account's own bank can see by its cutoff."""
    earliest = links.groupby(["hashed_account_id", key], sort=False)["observed_at"].min()
    frame = earliest.reset_index()
    own_cutoff = frame["hashed_account_id"].map(cutoff)
    return frame.loc[frame["observed_at"] <= own_cutoff]


def _link_counts(
    links: pd.DataFrame,
    key: str,
    bank: pd.Series,
    index: pd.Index,
    cutoff: pd.Series,
    same_bank_only: bool,
) -> pd.Series:
    """Count other accounts sharing ``key`` with each account, as of its cutoff.

    ``links`` has columns ``hashed_account_id``, ``key`` and
    ``observed_at``. A partner counts only when *both* observations land
    at or before the **observing** account's cutoff: a bank cannot act on
    a shared fingerprint whose other sighting has not happened yet.
    With ``same_bank_only`` the partner must be at the same bank — the
    single-bank scope simply cannot observe partners elsewhere, so those
    links do not exist for it rather than counting as zero.
    """
    pairs = links.groupby(["hashed_account_id", key], sort=False)["observed_at"].min().reset_index()
    joined = pairs.merge(pairs, on=key, suffixes=("", "_other"))
    joined = joined.loc[joined["hashed_account_id"] != joined["hashed_account_id_other"]]
    own_cutoff = joined["hashed_account_id"].map(cutoff)
    joined = joined.loc[
        (joined["observed_at"] <= own_cutoff) & (joined["observed_at_other"] <= own_cutoff)
    ]
    if same_bank_only:
        own = joined["hashed_account_id"].map(bank)
        other = joined["hashed_account_id_other"].map(bank)
        joined = joined.loc[own == other]
    counts = joined.groupby("hashed_account_id")["hashed_account_id_other"].nunique()
    return counts.reindex(index).fillna(0).astype("int64")


def _scope_features(
    accounts: pd.DataFrame,
    transfers: pd.DataFrame,
    links: dict[str, pd.DataFrame],
    bank: pd.Series,
    cutoff: pd.Series,
    gap_seconds: int,
    same_bank_only: bool,
) -> pd.DataFrame:
    """Build the full feature block for one visibility scope."""
    index = pd.Index(accounts["hashed_account_id"], name="hashed_account_id")

    if same_bank_only:
        intra = transfers["from_hashed_account_id"].map(bank) == transfers[
            "to_hashed_account_id"
        ].map(bank)
        transfers = transfers.loc[intra]

    events = _edge_events(transfers, bank)
    # Point-in-time: an account never sees a transfer after its own cutoff.
    events = events.loc[events["timestamp"] <= events["account"].map(cutoff)]
    inbound = events.loc[events["is_inbound"]]
    outbound = events.loc[~events["is_inbound"]]

    def _agg(frame: pd.DataFrame, column: str, how: str) -> pd.Series:
        if frame.empty:
            return pd.Series(0.0, index=index, dtype="float64")
        return (
            frame.groupby("account")[column].agg(how).reindex(index).fillna(0.0).astype("float64")
        )

    in_degree = _agg(inbound, "amount", "size")
    out_degree = _agg(outbound, "amount", "size")
    amount_in_total = _agg(inbound, "amount", "sum")
    amount_out_total = _agg(outbound, "amount", "sum")

    # Zero inbound => ratio undefined => 0.0 (see module docstring).
    inflow = amount_in_total.to_numpy()
    has_inflow = inflow > 0.0
    passthrough_ratio = np.where(
        has_inflow, amount_out_total.to_numpy() / np.where(has_inflow, inflow, 1.0), 0.0
    )

    total_degree = in_degree + out_degree
    is_round = events.assign(
        round_flag=(np.mod(events["amount"].to_numpy(), ROUND_AMOUNT_UNIT) == 0.0)
    )
    round_count = _agg(is_round, "round_flag", "sum")
    degree = total_degree.to_numpy()
    has_edges = degree > 0.0
    round_amount_ratio = np.where(
        has_edges, round_count.to_numpy() / np.where(has_edges, degree, 1.0), 0.0
    )

    opened_at = accounts.set_index("hashed_account_id")["opened_at"].reindex(index)
    age_days = ((cutoff.reindex(index) - opened_at).dt.total_seconds() / 86400.0).astype("float64")

    visible_devices = _visible_links(links["device_fingerprint"], "device_fingerprint", cutoff)
    device_count = (
        visible_devices.groupby("hashed_account_id")["device_fingerprint"]
        .nunique()
        .reindex(index)
        .fillna(0)
        .astype("int64")
    )
    shared_device_count = _link_counts(
        links["device_fingerprint"],
        "device_fingerprint",
        bank,
        index,
        cutoff,
        same_bank_only=same_bank_only,
    )
    document_reuse_count = _link_counts(
        links["hashed_id_number"],
        "hashed_id_number",
        bank,
        index,
        cutoff,
        same_bank_only=same_bank_only,
    )

    return pd.DataFrame(
        {
            "in_degree": in_degree.astype("int64"),
            "out_degree": out_degree.astype("int64"),
            "unique_in_counterparties": _agg(inbound, "counterparty", "nunique").astype("int64"),
            "unique_out_counterparties": _agg(outbound, "counterparty", "nunique").astype("int64"),
            "unique_counterparties": _agg(events, "counterparty", "nunique").astype("int64"),
            "distinct_counterparty_banks": _agg(events, "counterparty_bank", "nunique").astype(
                "int64"
            ),
            "amount_in_total": amount_in_total,
            "amount_out_total": amount_out_total,
            "passthrough_ratio": passthrough_ratio,
            "passthrough_speed_seconds": _passthrough_speed(events, index, gap_seconds),
            "round_amount_ratio": round_amount_ratio,
            "account_age_days": age_days,
            "device_count": device_count,
            "shared_device_count": shared_device_count,
            "document_reuse_count": document_reuse_count,
            "document_reuse_flag": (document_reuse_count > 0).astype("int64"),
        },
        index=index,
    )[list(FEATURE_NAMES)]


def full_window_cutoff(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """Every account observed at the last transfer instant in the dataset."""
    as_of = tables["transfers"]["timestamp"].max()
    return pd.Series(as_of, index=pd.Index(tables["accounts"]["hashed_account_id"]))


def build_features(
    cfg: MlConfig | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    cutoff: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the paired single-bank / cross-bank feature table.

    ``cutoff`` is a per-account observation instant indexed by
    ``hashed_account_id``; it defaults to the full window.
    """
    cfg = cfg or MlConfig.from_env()
    tables = tables if tables is not None else load_tables(cfg)
    accounts = tables["accounts"]
    transfers = tables["transfers"]
    bank = _bank_of(accounts)
    cutoff = full_window_cutoff(tables) if cutoff is None else cutoff
    cutoff = cutoff.reindex(accounts["hashed_account_id"])
    links = identity_links(tables)

    blocks = []
    for scope, prefix in SCOPES.items():
        block = _scope_features(
            accounts=accounts,
            transfers=transfers,
            links=links,
            bank=bank,
            cutoff=cutoff,
            gap_seconds=cfg.burst_gap_seconds,
            same_bank_only=(scope == "single_bank"),
        )
        blocks.append(block.add_prefix(prefix))

    features = pd.concat(blocks, axis=1).reset_index()
    features.insert(1, "as_of", cutoff.to_numpy())
    return features


def feature_columns(prefix: str) -> list[str]:
    """Column names belonging to one scope prefix (``sb_`` or ``cb_``)."""
    return [f"{prefix}{name}" for name in FEATURE_NAMES]


def main() -> None:
    """Build features and write ``ml/features.parquet``."""
    cfg = MlConfig.from_env()
    features = build_features(cfg)
    cfg.features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg.features_path, index=False)
    print(f"wrote {cfg.features_path} rows={len(features)} cols={features.shape[1]}")


if __name__ == "__main__":
    main()
