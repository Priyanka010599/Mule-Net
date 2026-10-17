# Where MuleNet runs, and where its data lives

Written because "we moved to Mumbai" is easy to say and only half true.

## Update, 2026-10-05: the data moved too

Everything below this point describes the state where inference had
moved to `asia-south1` and the data had not -- that gap is now closed.
A parallel `asia-south1` stack (new bucket, new BigQuery dataset, `ui`
and `ingest` redeployed there) now runs the live app; `mulenet-agents`
did not need to move, because it was already in `asia-south1`. See
`infra/asia_south1_migration.md` for the inventory and the deploy
script. The rest of this document is kept as the record of how that
gap was found and why it mattered -- read it for the history, not for
where things currently run.

## What moved

**Gemini inference: `us-central1` -> `asia-south1`.** One config value,
`agents.config.DEFAULT_LOCATION`, overridable by `GOOGLE_CLOUD_LOCATION`.

Two reasons, both measured on 2026-09-27 with live `generate_content`
probes against project `mulenet` — not read off a docs page:

```
model                 us-central1   asia-south1   us      global
gemini-2.5-flash           SERVES        SERVES    404    SERVES
gemini-2.5-flash-lite      SERVES           404    404    SERVES
gemini-3.5-flash              404        SERVES SERVES    SERVES
gemini-3.8-flash              404           404 SERVES    SERVES
gemini-3.1-flash-lite         404           404 SERVES    SERVES
```

1. **`us-central1` is a dead end with a date on it.** It serves no Gemini
   3.x at all, and `gemini-2.5-flash` retires 16–20 October 2026. Staying
   there means the region and the model must change in the same step,
   under a deadline, with no fallback.
2. **`asia-south1` serves both.** So the region moves now, on the model
   already measured, and the 3.x choice stays a separate decision due by
   10 October. For an India BFSI entry, Mumbai is also a better answer
   than Iowa on its own merits.

The `us` multi-region serves every 3.x model but **404s on
`gemini-2.5-flash`**, so it cannot be adopted incrementally either.

A note on method: the publisher catalogue (`models.list`, `models.get`)
returns every Gemini 3.x id in every region, including ones that refuse
to serve. Only `generate_content` tells the truth. Probe before
believing — this caught us twice, from opposite directions.

## What did NOT move

**The data.** Verified with `gcloud`:

```
GCS bucket mulenet-mulenet-data  US-CENTRAL1
Cloud Run  mulenet-ingest        us-central1
Cloud Run  mulenet-ui            us-central1
BigQuery   dataset mulenet       us-central1
```

Transactions, KYC documents, scores, rings and the published tables are
all still in Iowa. So is every service that reads them.

**This matters for how the change is described.** Moving inference
relocates where a prompt and a tool result are *processed*. It does not
relocate the records. For a residency requirement about where data is
*stored*, this change on its own does not satisfy it, and saying "MuleNet
runs in Mumbai" would be untrue.

What a full move would require, none of which is done:

- a bucket in `asia-south1` and a re-upload
- a BigQuery dataset in `asia-south1` (datasets cannot be relocated in
  place; tables are recreated and reloaded)
- both Cloud Run services redeployed to `asia-south1`
- `infra/deploy.sh`, `infra/load_bigquery.py` and `infra/setup.sh`
  updated, plus the IAM bindings re-checked
- the agents service deployed there rather than to `us-central1`

## What this buys today

The `agents/` service is now deployed, and the UI's chat
(`MULENET_UI_CHAT_BACKEND=agents`) runs on it — `gemini-3.5-flash` in
`asia-south1`, not a mock. The region change took effect with that
deploy.

Cross-region latency from an `asia-south1` model call to `us-central1`
data is not a factor either, for the same reason — the tools read local
parquet inside the container, not the bucket, at request time.

## Verified on the new region

Four live chat questions, `gemini-2.5-flash` @ `asia-south1`, all `200`:

```
How many banks does ring-...-00148 span?            2 banks                    22.3s
How many alertable members does it have?            3 alertable members         8.8s
Which evidence kinds are attached?                  cross_bank_shared_device    9.2s
Total intra-ring transfer amount, to the cent?      3500.80 USD                17.9s
```

Every one carried real provenance (`get_ring_summary`,
`get_ring_transfer_summary`), so the answers were grounded in tool calls
rather than produced from nothing.

## The 3.x model choice: decided, 2026-10-03

`gemini-3.5-flash` is now `agents.config.DEFAULT_MODEL`. Two things made
this the only real option, both verified live, not read off a docs page:

1. **The 2.5 family retires 16-20 Oct 2026** (Google's own release notes;
   one Google page says the 20th, release notes say the 16th — planned
   against the earlier date). `gemini-3.5-flash` is Google's own named
   successor and does not retire before 2027-05-19.
2. **Region re-probed on 2026-10-03, after the first pass on
   2026-09-27 missed a model.** `gemini-3.5-flash` is the *only* Gemini
   3.x tier offered region-pinned in `asia-south1` today -- 3.6, 3.7,
   3.8-flash and 3.5-flash-lite all 404 there and only answer on the
   `global` endpoint. `agents/config.py` has the full probe table. This
   almost led to an unnecessary move to the global endpoint before the
   gap in the first probe was caught.

### Fresh eval, same 25 questions, 2026-10-03

Re-run live against `gemini-3.5-flash` in `asia-south1` after the switch.
Full records: `agents/out/eval/gemini-3.5-flash__asia-south1__20261003T144414Z.jsonl`.

```
                    score    detection    declines    avg cost/q   avg latency
gemini-3.5-flash    22/25    14/17 82.4%  8/8 100%    $0.0350      14.2s
gemini-2.5-flash *   23/25    15/17 88.2%  8/8 100%    $0.0027      12.7s
gemini-2.5-flash †   20/25    13/17 76.5%  7/8  87.5%  (not recorded)  --
```

\* best of two recorded 2.5-flash baseline runs (`...20260927T133758Z`)
† the other recorded run (`...20260927T134636Z`)

**Correcting this doc's earlier claim:** the "88.2% vs 76.5%" comparison
written here on 2026-09-27 was real, but it compared one 3.5-flash run
against the *worse* of two 2.5-flash baseline runs, not the better one.
Both 2.5-flash numbers are genuine recorded runs on the same question
set -- the 2.5 baseline itself varies 20/25 to 23/25 run to run, which
the single-run comparison hid. Measured today, 3.5-flash (22/25) sits
between the two, not clearly better or worse than 2.5-flash was. The
model switch is happening anyway, because 2.5 retires regardless of
which run you compare against.

The real, consistent cost is ~13x 2.5-flash's ($0.035 vs $0.0027 per
question) -- small in absolute terms (a 25-question eval cost $0.88
total; see `agents/limits.py`'s daily cap discussion), but a real
multiple, not the "~15x, partly from looping" figure in the September
run, which does not reproduce in this run: declines were 8/8 both times,
so the looping-as-decline fix already mentioned here seems to have held.

## Open

- **Whether data residency is actually required.** The work the list
  above called for is now done (see the 2026-10-05 update above) --
  this question no longer gates anything technical, but a legal review
  of whether it was actually *required* has still not been done (see
  the DPDP Act review item in the README's limitations).
