"""Group unit for the IBM AML HI-Small benchmark (Lane D2).

Standalone and offline. Reads the raw files from the gitignored
ml/experiments/public_benchmark/ folder and writes:

- ml/experiments/ibm_aml/split.json (gitignored, NOT committed -- see
  licence note below): the grouping rule, counts, and input
  fingerprints.
- ml/experiments/public_benchmark/data/assignment.parquet (gitignored):
  one row per account with its group.

Group unit: connected components over all POSITIVE (Is Laundering=1)
transactions, nodes keyed by (bank, account). Accounts with no positive
transaction are each their own group. Folds are dealt per run by
run_grouped_cv.py, so every account in a group lands in the same fold.

No time holdout: accounts are active from day 1 (median first activity
Sep 1 00:28; 1 of 515,088 first appears after Sep 15), so a time split
is not meaningful on this data.

Licence note (CDLA-Sharing-1.0, see README.md): split.json contains no
account numbers, entity IDs, or sample rows -- only aggregate counts
and SHA-256 fingerprints of the raw files -- so it was not itself a
sharing-clause risk. It is still kept local-only and regenerated
on demand: this script has no randomness (union-find over the raw
file's own edges, no seed needed) and the same inputs always produce
byte-identical output, so EXPECTED_SPLIT_SHA256 below is what is
committed in its place; main() checks the freshly written file against
it and fails loudly on a mismatch rather than silently drifting.

Usage::

    python ml/experiments/ibm_aml/make_split.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW_DIR = HERE.parent / "public_benchmark"
DATA_DIR = RAW_DIR / "data"
TRANS_PATH = RAW_DIR / "HI-Small_Trans.csv"
ACCOUNTS_PATH = RAW_DIR / "HI-Small_accounts.csv"
PATTERNS_PATH = RAW_DIR / "patterns.txt"
SPLIT_PATH = HERE / "split.json"
ASSIGNMENT_PATH = DATA_DIR / "assignment.parquet"

#: Expected SHA-256 of split.json's content, given the HI-Small raw
#: files' own fingerprints below (which are themselves committed, inside
#: split.json's own inputs_sha256 field -- but split.json itself stays
#: gitignored; this constant is what stands in for it in git history).
#: Recompute with sha256_file(SPLIT_PATH) after any change to this
#: script's split-building logic, and update this constant deliberately
#: -- a silent mismatch here means the regenerated file no longer
#: matches what was reviewed and is a bug, not noise.
EXPECTED_SPLIT_SHA256 = "a28e3db912654fd357d267f2aebf76c1c439c57713520268b9afc77cc2b0ac8b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_transactions() -> pd.DataFrame:
    trans = pd.read_csv(
        TRANS_PATH,
        dtype={"From Bank": str, "To Bank": str, "Account": str, "Account.1": str},
    )
    trans["from_key"] = trans["From Bank"].astype(int).astype(str) + ":" + trans["Account"]
    trans["to_key"] = trans["To Bank"].astype(int).astype(str) + ":" + trans["Account.1"]
    trans["timestamp"] = pd.to_datetime(trans["Timestamp"], format="%Y/%m/%d %H:%M")
    return trans


def union_find(nodes: list[str], edges: pd.DataFrame) -> dict[str, str]:
    index = {n: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges["from_key"], edges["to_key"]):
        ra, rb = find(index[a]), find(index[b])
        if ra != rb:
            parent[ra] = rb
    return {n: nodes[find(index[n])] for n in nodes}


def main() -> None:
    trans = load_transactions()
    positives = trans.loc[trans["Is Laundering"] == 1]

    pos_nodes = sorted(set(positives["from_key"]) | set(positives["to_key"]))
    root_of = union_find(pos_nodes, positives)

    all_accounts = sorted(set(trans["from_key"]) | set(trans["to_key"]))
    accounts = pd.DataFrame({"acct": all_accounts})
    accounts["root"] = accounts["acct"].map(root_of)
    accounts["group"] = np.where(
        accounts["root"].notna(),
        "comp:" + accounts["root"].fillna(""),
        "free:" + accounts["acct"],
    )
    accounts["has_positive"] = accounts["root"].notna().astype("int8")
    accounts = accounts[["acct", "group", "has_positive"]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    accounts.to_parquet(ASSIGNMENT_PATH, index=False)

    split = {
        "method": (
            "Group unit: connected components over all positive (Is Laundering=1) "
            "transactions, nodes keyed by (bank, account). Accounts with no positive "
            "transaction are each their own group. Folds are dealt per run so every "
            "account in a group shares a fold. No time holdout: accounts are active from "
            "day 1 (median first activity Sep 1 00:28; 1 of 515,088 first appears after "
            "Sep 15). Regenerated locally (gitignored, CDLA-Sharing-1.0); this file's own "
            "SHA-256 is checked against EXPECTED_SPLIT_SHA256 in make_split.py, which is "
            "what is committed instead."
        ),
        "data_window": [trans["timestamp"].min().isoformat(), trans["timestamp"].max().isoformat()],
        "accounts": int(len(accounts)),
        "positive_accounts": int(accounts["has_positive"].sum()),
        "positive_components": int(accounts.loc[accounts["has_positive"] == 1, "group"].nunique()),
        "free_accounts": int((accounts["has_positive"] == 0).sum()),
        "inputs_sha256": {
            "HI-Small_Trans.csv": sha256_file(TRANS_PATH),
            "HI-Small_accounts.csv": sha256_file(ACCOUNTS_PATH),
            "patterns.txt": sha256_file(PATTERNS_PATH),
        },
        "assignment_file": "ml/experiments/public_benchmark/data/assignment.parquet (gitignored)",
        "assignment_sha256": sha256_file(ASSIGNMENT_PATH),
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(json.dumps(split, indent=2))

    actual = sha256_file(SPLIT_PATH)
    if actual != EXPECTED_SPLIT_SHA256:
        raise SystemExit(
            f"split.json's SHA-256 is {actual}, expected {EXPECTED_SPLIT_SHA256}. "
            "Either the raw HI-Small files changed, or this script's split-building "
            "logic did -- in the latter case, update EXPECTED_SPLIT_SHA256 "
            "deliberately after reviewing the new split.json, don't just paste the "
            "new hash in to make this pass."
        )
    print(f"split.json SHA-256 matches EXPECTED_SPLIT_SHA256: {actual}")


if __name__ == "__main__":
    main()
