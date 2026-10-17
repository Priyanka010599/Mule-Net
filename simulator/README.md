# simulator/ — synthetic three-bank world

Seeded generator for MuleNet's demo data: three banks, 30 000 accounts,
30 days of transfers, 120 mule rings (84 cross-bank plus 36 same-bank
controls), hard-negative cohorts, and
placeholder KYC document images. Everything is a pure function of
`(seed, SimParams, HASH_KEY)`, so the same seed reproduces byte-identical
output.

All schemas come from `shared/contracts.py`. Nothing is redefined here.

## Run it

```bash
# full dataset (~70s, writes simulator/out/)
python -m simulator.generator

# scaled down, no images
python -m simulator.generator --accounts-per-bank 400 --no-images --out /tmp/sim

# replay to Pub/Sub at 1 simulated minute per real second, printing only
python -m simulator.replay --dry-run --speed 60 --limit 20
```

Output lands in `simulator/out/`, which the root `.gitignore` excludes
(`out/` and `*.parquet`). The full run is ~169 MB: `transfers.parquet` is
104 MB and `kyc_images/` is 38 MB.

## Configuration (`simulator/config.py`)

| Env var | Default | Meaning |
| --- | --- | --- |
| `HASH_KEY` | `demo-key` | HMAC-SHA256 key for every hashed identifier |
| `MULENET_OUT_DIR` | `simulator/out` | Where parquet and images are written |
| `MULENET_SEED` | `20250101` | Master seed |
| `MULENET_GCP_PROJECT` | *(unset)* | Project for real Pub/Sub publishing |
| `MULENET_TRANSFERS_TOPIC` | `mulenet-transfers` | Overrides `shared.pubsub_schema.TRANSFERS_TOPIC` |

Set `HASH_KEY` from Secret Manager for anything but a local demo. The
default exists only so the repo runs offline.

## Output files

All six are written to the output directory.

| File | Columns |
| --- | --- |
| `accounts.parquet` | `hashed_account_id, bank_id, opened_at, account_type, status` |
| `transfers.parquet` | `transfer_id, from_hashed_account_id, to_hashed_account_id, amount, currency, timestamp, channel` |
| `devices.parquet` | `device_id, hashed_account_id, device_fingerprint, first_seen, last_seen` |
| `kyc_documents.parquet` | `doc_id, hashed_account_id, doc_type, hashed_id_number, normalized_address, photo_embedding_ref` |
| `kyc_submissions.parquet` | `submission_id, hashed_account_id, bank_id, submitted_at, doc_ids` |
| `ground_truth.parquet` | `hashed_account_id, bank_id, is_mule, ring_id, role, pattern_type, cohort, is_hard_negative, ring_bank_count, opened_at` |

Notes:

* `amount` is stored as parquet `decimal128(15, 2)`, not float64, matching
  `MoneyAmount` in `shared/contracts.py` and `NUMERIC(15, 2)` in
  `shared/ddl.sql`. It reads back as `decimal.Decimal`, so no binary float
  reaches the ingest path.
* Every datetime column is timezone-aware UTC, matching the `TIMESTAMPTZ`
  columns in `shared/ddl.sql`.
* `kyc_submissions.parquet` stores `doc_ids` (a list) instead of nested
  documents. Join to `kyc_documents.parquet` on `doc_id` to rebuild a
  `shared.contracts.KycSubmission`.
* `photo_embedding_ref` is nullable and holds a path relative to the output
  directory, e.g. `kyc_images/<identity>_id_card.png`. Under pandas 3.x a
  null reads back as `NaN`, not `None` — normalise it before feeding
  pydantic.
* Every identifier column is a 64-character HMAC-SHA256 digest. No raw,
  sequential or bank-native account number appears anywhere in the output.
  `ring_id` (`ring_000`…`ring_119`) and `bank_id` are labels, not identifiers.

## Ground truth

`cohort` is the label the `ml/` module should evaluate against:

| cohort | Meaning |
| --- | --- |
| `mule` | Ring member. `is_mule` is true, `role` ∈ {collector, mule, source} |
| `hard_negative_business` | Busy small business: high fan-in, many counterparties, some same-day settlement |
| `hard_negative_family` | Family sharing a device fingerprint, half of them across two banks |
| `hard_negative_dual_bank` | One real person holding accounts at two banks, same ID number at both |
| `normal` | Everything else, including merchants, employers and billers |

`is_hard_negative` is true for the three hard-negative cohorts, so false
positives on them can be measured separately from ordinary accounts.

## Ring patterns

Thirty rings each, 120 total. Ring size is drawn uniformly over 3–12
members, so size itself carries no signal:

| `pattern_type` | Shared device | Reused document | Fast pass-through |
| --- | --- | --- | --- |
| `shared_device` | yes, across banks | no | yes |
| `reused_document` | no | yes, across banks | yes |
| `fast_passthrough_only` | no | no | yes |
| `mixed` | one half of the ring | the other half | yes |

Every ring fans money in from many senders to mule accounts, then moves it
on to a collector within 2–30 minutes, and the collector cashes out hours
later. In a cross-bank ring that collector sits at a **different** bank.

## Ring span, and the same-bank control group

`SimParams.span_mix` weights how many banks a ring spans — `(0.30, 0.40,
0.30)` over spans 1, 2 and 3 by default, applied to **each pattern type
independently** so span and pattern are not confounded. At the default that
is 36 / 48 / 36 rings, nine of each span per pattern:

| `pattern_type` | 1 bank | 2 banks | 3 banks |
| --- | --- | --- | --- |
| `shared_device` | 9 | 12 | 9 |
| `reused_document` | 9 | 12 | 9 |
| `fast_passthrough_only` | 9 | 12 | 9 |
| `mixed` | 9 | 12 | 9 |

The 36 single-bank rings are the **control group**. Their pattern carries
over inside the one bank — a same-bank `shared_device` ring really does share
a fingerprint among its members at that bank, a same-bank `reused_document`
ring really does reuse a `hashed_id_number` there — so that bank can see the
whole ring on its own. Without them the dataset could only answer "how much
does a cross-bank join help on a ring that is *already* split", which is the
cross-bank best case with nothing to compare it against.

The span is recorded per account in **`ring_bank_count`** in
`ground_truth.parquet` (1, 2 or 3; `0` for non-mules). There is deliberately
no separate `banks_spanned` column — one column that is asserted against the
observed bank spread beats two that can drift apart. Override the mix with
`SimParams(span_mix=...)` or `--span-mix 0.5,0.25,0.25` on the CLI.

Ring **amounts carry no signal of their own**. Every ring edge is drawn from
the same peer-transfer distribution as legitimate traffic, each forwarded sum
is smeared across many small hops after a 1–5% cut, and exact-hundred amounts
occur at the same ~0.5% rate on both sides. An earlier version drew ring
amounts from a fixed round-number table, which let a single bank identify
mules from `round_amount_ratio` alone; `test_leak_*` in `test_generator.py`
exists to stop that returning.

Most ring members are newly opened, but a minority are aged bought accounts,
so account age is not a mule-only property either.

## Why the hard negatives matter

Each individual cross-bank signal is deliberately contaminated, so no single
feature separates rings from legitimate customers. From the full run:

* Cross-bank device sharing: 256 mule accounts, against 788 legitimate ones
  (families and dual-bank customers).
* Cross-bank ID reuse: 249 mule accounts, against 400 dual-bank customers.
* **Same-bank** document reuse: 292 mule accounts against 1442 legitimate
  ones, so within-bank reuse is not actionable on its own.
* 38% of legitimate accounts show an in-then-out hop inside 30 minutes —
  rent paid when the salary clears, sweeps between own accounts, same-day
  supplier settlement — so pass-through *speed* is not actionable either.
* Businesses out-fan-in the mules.

No single per-account feature reaches a false-positive rate below 15% at 80%
recall in the single-bank view. The separation only appears once structure,
timing and identity reuse are joined *across banks*.

Only the join of *graph structure plus timing plus identity reuse across
banks* isolates the rings. That is the result the project is trying to show.

## Synthetic KYC documents

`simulator/kyc.py` renders flat-colour PNGs with drawn text. There are no
likenesses — the photo box is a grey rectangle reading `NO PHOTO`. ID numbers
use the obviously-fake `SYN-xxxx-xxxx-xxxx` shape with no checksum, holders
are `SYNTHETIC HOLDER <hex>`, and every image carries a
`SYNTHETIC SAMPLE - NOT A REAL DOCUMENT` banner.

Images are rendered per *identity*, not per account, so a reused identity is
literally the same file at two banks — that is what `docai/` detects. To keep
the run fast, images are rendered for every cohort account plus
`--kyc-image-sample` ordinary accounts (default 300); other accounts get
document rows with a null `photo_embedding_ref`.

## Tests

```bash
python -m pytest simulator -q
```

Tests build a 1 200-account world (`SimParams(accounts_per_bank=400)`) so the
suite runs in a few seconds; `SimParams()` defaults stay at the full 30 000
accounts and `test_default_params_are_full_scale` guards that.
