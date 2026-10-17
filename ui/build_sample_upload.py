"""Hand-built synthetic CSVs for the "Score your own data" upload tab.

Three per-bank files in the upload's own format (``agents/subset_upload.py``;
an account belongs to the bank whose file lists it as ``from_account``). They
hold:

* one planted three-bank mule ring -- victims fan in to a collector at each
  bank, which forwards the money on within minutes to cash-out accounts at the
  other banks (the shape the shipped velocity rule describes);
* two decoys that look busy but are not that shape -- a merchant that takes
  many small payments and settles slowly, and a family group paying each other;
* ordinary background traffic.

This is illustrative data written by hand, NOT drawn from the simulator and not
a benchmark: it exists so a judge can see the upload flow end to end. Accuracy
claims stay in ``docs/claims.md``. Account ids are SHA-256 hashes of made-up
labels, so nothing in it is real or traceable.

Usage:  python -m ui.build_sample_upload          (writes ui/static/samples/)
        python -m ui.build_sample_upload --check  (also scores it, prints verdict)
Deterministic: the same command always writes the same bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "static" / "samples"
BANKS = ("BANK_A", "BANK_B", "BANK_C")
HEADER = ("from_account", "to_account", "amount", "timestamp", "channel")
SEED = 20261007

#: Day 0 of the sample week; the ring runs on day 4, in a three-hour window.
EPOCH = datetime(2026, 3, 1, tzinfo=UTC)
RING_START = EPOCH + timedelta(days=4, hours=13)

BACKGROUND_PER_BANK = 70
VICTIMS_PER_BANK = 8
CASHOUTS_PER_BANK = 2


def acct(label: str) -> str:
    return hashlib.sha256(f"mulenet-sample-upload/{label}".encode()).hexdigest()


class Builder:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.rows: dict[str, list[tuple[str, str, str, datetime, str]]] = {b: [] for b in BANKS}

    def add(
        self, bank: str, src: str, dst: str, amount: float, when: datetime, channel: str
    ) -> None:
        self.rows[bank].append((src, dst, f"{amount:.2f}", when, channel))

    def render(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for bank, rows in self.rows.items():
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(HEADER)
            for src, dst, amount, when, channel in sorted(rows, key=lambda r: (r[3], r[0], r[1])):
                writer.writerow([src, dst, amount, when.strftime("%Y-%m-%dT%H:%M:%SZ"), channel])
            out[bank] = buf.getvalue()
        return out


def build() -> dict[str, str]:
    b = Builder()
    rng = b.rng

    # --- the planted ring -------------------------------------------------
    victims = {
        bank: [acct(f"victim/{bank}/{i}") for i in range(VICTIMS_PER_BANK)] for bank in BANKS
    }
    collector = {bank: acct(f"collector/{bank}") for bank in BANKS}
    cashouts = {
        bank: [acct(f"cashout/{bank}/{i}") for i in range(CASHOUTS_PER_BANK)] for bank in BANKS
    }

    for bank in BANKS:
        others = [o for o in BANKS if o != bank]
        targets = [c for o in others for c in cashouts[o]]  # cash-outs at the OTHER banks
        # Each collector is paid by every victim at the OTHER two banks (16 distinct
        # senders, two banks) and forwards ~92% of each deposit to a cash-out
        # account at another bank 2-12 minutes later (fast pass-through).
        n = 0
        for other in others:
            for v in victims[other]:
                paid = RING_START + timedelta(minutes=rng.randint(0, 90))
                amount = rng.uniform(300, 900)
                b.add(other, v, collector[bank], amount, paid, "p2p")
                forwarded = paid + timedelta(minutes=rng.randint(2, 12))
                b.add(
                    bank,
                    collector[bank],
                    targets[n % len(targets)],
                    amount * 0.92,
                    forwarded,
                    "instant",
                )
                n += 1

    # Cash-out accounts empty out to outside accounts shortly after being paid, so
    # each is a sender in its own bank's file (that is what puts it inside the
    # upload's perimeter).
    for bank in BANKS:
        for i, c in enumerate(cashouts[bank]):
            for k in range(4):
                when = RING_START + timedelta(minutes=rng.randint(20, 150))
                b.add(bank, c, acct(f"outside/{bank}/{i}/{k}"), rng.uniform(250, 600), when, "card")

    # --- decoy 1: a busy merchant at BANK_A, many small payers, slow settlement
    merchant = acct("merchant/BANK_A")
    for i in range(45):
        when = EPOCH + timedelta(days=rng.randint(0, 6), hours=rng.randint(8, 21))
        payer_bank = BANKS[i % 3]
        b.add(
            payer_bank,
            acct(f"shopper/{payer_bank}/{i}"),
            merchant,
            rng.uniform(8, 60),
            when,
            "card",
        )
    for day in (2, 5):
        b.add(
            "BANK_A",
            merchant,
            acct("merchant-settlement"),
            1200.0,
            EPOCH + timedelta(days=day, hours=23),
            "ach",
        )

    # --- decoy 2: a family group at BANK_B paying each other small amounts
    family = [acct(f"family/{i}") for i in range(5)]
    for _ in range(25):
        src, dst = rng.sample(family, 2)
        when = EPOCH + timedelta(days=rng.randint(0, 6), hours=rng.randint(7, 22))
        b.add("BANK_B", src, dst, rng.uniform(15, 120), when, "p2p")

    # --- ordinary background: small salary/card/p2p traffic, spread over the week
    for bank in BANKS:
        people = [acct(f"person/{bank}/{i}") for i in range(BACKGROUND_PER_BANK)]
        for p in people:
            for _ in range(rng.randint(2, 4)):
                when = EPOCH + timedelta(
                    days=rng.randint(0, 6), hours=rng.randint(6, 22), minutes=rng.randint(0, 59)
                )
                if rng.random() < 0.5:
                    b.add(
                        bank,
                        p,
                        acct(f"shop/{rng.randint(0, 40)}"),
                        rng.uniform(5, 140),
                        when,
                        "card",
                    )
                else:
                    b.add(bank, p, rng.choice(people), rng.uniform(10, 250), when, "p2p")
    return b.render()


def check(files: dict[str, str]) -> None:
    from agents.subset_upload import score_upload

    ring = {acct(f"collector/{bank}") for bank in BANKS}
    ring |= {acct(f"cashout/{bank}/{i}") for bank in BANKS for i in range(CASHOUTS_PER_BANK)}
    decoys = {acct("merchant/BANK_A"), *(acct(f"family/{i}") for i in range(5))}
    for label, banks in (("all three banks", BANKS), ("BANK_A alone", BANKS[:1])):
        res = score_upload([(bk, files[bk]) for bk in banks])
        flagged = res["flagged_accounts"]
        cross = {a["hashed_account_id"] for a in flagged if a["caught_cross_bank"]}
        single = {a["hashed_account_id"] for a in flagged if a["caught_single_bank"]}
        rule = {a["hashed_account_id"] for a in flagged if a["caught_by_rule"]}
        every = cross | single | rule
        print(
            f"{label}: model_c ring={len(cross & ring)}/{len(ring)} "
            f"velocity rule ring={len(rule & ring)}/{len(ring)} "
            f"single-bank rule ring={len(single & ring)}/{len(ring)} "
            f"decoys flagged={len(every & decoys)} other flagged={len(every - ring - decoys)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the sample upload CSVs.")
    parser.add_argument(
        "--check", action="store_true", help="score the files and print the verdict"
    )
    args = parser.parse_args()
    files = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bank, text in files.items():
        (OUT_DIR / f"sample_{bank.lower()}.csv").write_text(text, encoding="utf-8", newline="\n")
        print(f"{bank}: {text.count(chr(10)) - 1} rows, {len(text.encode()) // 1024} KB")
    if args.check:
        check(files)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
