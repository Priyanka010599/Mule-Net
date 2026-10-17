# docai

KYC document parsing via Document AI and cross-bank document reuse detection,
emitting risk signals for scoring. Owned by the **docai-engineer** agent.
Signal shapes come from `shared/`.

## What reuse detection is worth

Document reuse **is evidence, not a detection**. On the seed-42 corpus, reuse of
any kind flags 2,178 accounts, of which only 335 are mules. Legitimate accounts
reuse documents constantly: joint accounts, business owners, families, and
people holding two accounts at one bank. The load-bearing distinction is
**scope**, not reuse:

| cohort                  | same-bank reuse | cross-bank reuse |
| ----------------------- | --------------: | ---------------: |
| hard_negative_business  |          28.00% |            0.00% |
| hard_negative_family    |           8.01% |            0.00% |
| normal                  |           4.90% |            0.00% |
| mule                    |          32.48% |           27.70% |
| hard_negative_dual_bank |           0.00% |          100.00% |

Same-bank reuse is mostly noise — mules reuse within a bank at about the rate
businesses do. Cross-bank reuse is far sharper but still not clean: 400
dual-bank customers legitimately present one identity at two banks, so
cross-bank reuse is only ~38% mules. Nothing here may be used as a firing
decision on its own.

## Modes

One `config.py`, selected by `MULENET_DOCAI_MODE` (`mock` | `cloud`).
**Mock is the default. Cloud is opt-in and is never inferred** — in particular,
having Application Default Credentials on the machine does *not* select it.

    # offline (default) -- nothing to set
    python -m docai.reuse

    # cloud -- explicit opt-in, plus the ids that path needs
    set MULENET_DOCAI_MODE=cloud
    set GOOGLE_CLOUD_PROJECT=...
    set MULENET_DOCAI_PROCESSOR_ID=...        # Document AI parsing
    set MULENET_REGULATIONS_DATASTORE=...     # Vertex AI Search data store
    set MULENET_REGULATIONS_ENGINE=...        # Vertex AI Search app

Credentials are checked only *after* the operator has asked for cloud: if
`MULENET_DOCAI_MODE=cloud` is set but an identifier is missing, or ADC does not
resolve, the run fails loudly with an actionable message rather than quietly
falling back to mock.

Why it is not inferred: credentials being *available* is not the same as the
operator *intending* to spend an API call. Inferring intent from ambient
credentials makes the same command do different things on different machines —
a suite that passes on a laptop breaks the moment someone runs
`gcloud auth login`, which is exactly what happened to `agents/`.

| component  | mock                                   | cloud                            |
| ---------- | -------------------------------------- | -------------------------------- |
| parsing    | fields from the intake parquet, no OCR | Document AI Form Parser (asia-south1) |
| embeddings | Pillow difference hash (1024-bit)      | Vertex AI `multimodalembedding`  |
| regulations| bundled placeholder passages           | Vertex AI Search (Discovery Eng.)|

Both cloud paths are **live and measured** — the regulation index under
"Regulations" below, Document AI and the Vertex embedder under "Measured".
The mock paths are what the tests exercise; no test spends money.

### Environment variables

`MULENET_DOCAI_MODE`, `MULENET_DATA_DIR`, `MULENET_DOCAI_OUT_DIR`,
`MULENET_REGULATIONS_DIR`, `MULENET_DOCAI_HASH_SIZE`,
`MULENET_PHOTO_SIMILARITY_THRESHOLD`, `GOOGLE_CLOUD_PROJECT`,
`MULENET_DOCAI_LOCATION`, `MULENET_DOCAI_PROCESSOR_ID`,
`MULENET_VERTEX_LOCATION`, `MULENET_EMBEDDING_MODEL`,
`MULENET_EMBEDDING_DIMENSION`, `MULENET_SEARCH_LOCATION`,
`MULENET_REGULATIONS_DATASTORE`, `MULENET_REGULATIONS_ENGINE`,
`MULENET_REGULATIONS_PDF_DIR`, `MULENET_REGULATIONS_GCS_URI`.

## Outputs

    python -m docai.parsing    # -> docai/out/parsed_documents.parquet
    python -m docai.reuse      # -> docai/out/reuse_pairs.parquet
                               #    docai/out/account_evidence.parquet

`account_evidence.parquet` is `shared.contracts.AccountEvidence`, the same
schema `ml/out/account_evidence.parquet` uses — this file is the
`source_table="kyc_documents"` half of it. `scope` is computed purely from
whether the two accounts' banks differ; `detector_version` is always `None`,
because this evidence is produced independently of any detector run.

Evidence types added by docai/: `{cross_bank,same_bank}_reused_id_number`,
`..._reused_address`, `..._reused_photo`. `reused_address` is new — the open
`evidence_type` taxonomy on `AccountEvidence` anticipates it.

## Measured: the cloud path against the offline one

`python -m docai.eval_cloud --limit 50` runs both cloud backends over a
bounded sample and prints what it spent. The sample is capped at 50 documents
because every one of them is a billed page. Ground truth is
`simulator.kyc`'s own renderer functions, imported rather than reimplemented,
so the expected values are the ones actually drawn on the image.

Run of 2026-09-27: 50 documents (24 ID cards, 26 address proofs), 50 distinct
images, Form Parser `pretrained-form-parser-v2.0-2022-11-10` in `asia-south1`,
`multimodalembedding@001` in `us-central1`. Report at
`docai/out/cloud_eval.json`.

### Extraction

| field | accuracy | note |
| --- | ---: | --- |
| address | **50/50 (100%)** | the field production actually consumes |
| document number | 48/50 (96%) | 22/24 on ID cards; address proofs correctly carry none |
| holder name | 46/50 (92%) | |
| all three | 44/50 (88%) | |

**Every one of the six failures is the same single character: `O` returned
where `0` was drawn.** Not one other character differed anywhere in the run.
The rendered alphabet after `SYN-` is hexadecimal and so is the identity key
in the holder name, and `O` is not a hex digit, so the confusion is detectable
and correctable without ambiguity. Applying `O` -> `0` to the hex portion only
takes all three fields to **50/50**.

That normalisation is **not** in the code, deliberately. Production reads only
the address off the document, and the address was already perfect; adding a
corrector that exists solely to improve a number this module also reports
would be marking its own homework. It matters for one specific future: the
extracted `SYN-xxxx-xxxx-xxxx` is a projection of the first 12 hex characters
of `hashed_id_number`, so it could become a *document-derived* reuse key
rather than an intake-derived one — which is strictly better evidence. If that
is built, the normalisation is mandatory, or roughly 8% of ID cards get a key
that matches nothing.

### Latency, per document

| p50 | p90 | p95 | max | mean |
| ---: | ---: | ---: | ---: | ---: |
| 1.85s | 6.36s | 24.96s | 60.13s | 4.73s |

**Quote the median, not the mean.** The typical call is 1.85s and 45 of 50
finished under 7s, but four calls took 14.1s, 25.0s, 40.9s and 60.1s, and
those four drag the mean to 4.73s. The first call was 7.12s, so this is not a
cold start — it is a tail. At 60s a request is well past any interactive
budget, so a deployed caller needs a timeout and a retry rather than an
assumption that a page takes two seconds.

### Reuse detection: does the cloud path beat the difference hash?

The offline backend is a *layout* hash and the synthetic documents share a
layout. On the full corpus that produces exactly one collision: two unrelated
address proofs hash identically, giving 2 false cross-bank photo pairs out of
1,792 (0.11%) and pulling in one account no other evidence supports. Both
colliding images are always in the sample, because a sample without them could
not answer the question.

Over all 1,225 pairs of the 50 distinct images — every one of which is a
different identity, so **every pair is a true negative**:

| | difference hash | Vertex multimodal |
| --- | ---: | ---: |
| same file against itself | 1.0 | 0.99999999991 |
| the known colliding pair | **1.0** | **0.99551** |
| highest unrelated pair | 1.0 | 0.99748 |
| median unrelated pair | 0.713 | 0.614 |
| false pairs at the operating threshold | 1 of 1,225 | 0 of 1,225 at 0.999 |

**Yes — but by 0.0025, and only on this sample.** The difference hash gives
the colliding pair and a genuine duplicate the *same* value, 1.0, so no
threshold can separate them: the false pair is unreachable, not merely
likely. Vertex gives the colliding pair 0.99551 and a genuine duplicate
0.99999999991, so a threshold anywhere in (0.99748, 0.9999) separates
everything in this sample cleanly. That is a real improvement on a real
failure.

It is also a thin one, and worth saying plainly rather than selling:

- The headroom is **0.0025 of cosine**. 56 unrelated pairs already sit above
  0.99 and 601 of 1,225 (49%) sit above 0.95. The model is not confidently
  telling these documents apart either; it is ordering them slightly better.
- Most of the win is a *determinism* property, not a content-discrimination
  one. A recycled identity in this simulator points at literally the same
  file, so byte-identical input gives an identical vector. The difference hash
  has that property too. The only place they differ is the collider.
- **1,225 pairs is not 7.5 million.** The 0.11% figure comes from the full
  3,890-image corpus. The highest unrelated similarity over 1,225 pairs is
  0.99748; over every pair of the full corpus it would almost certainly be
  higher, and it only has 0.0025 to climb before the threshold stops working.
  So "beats 0.11%" is established on this sample and **not** established at
  corpus scale.

The honest next step is cheap and was not taken here only because it exceeds
the 50-document cap this run was held to: embedding all 3,890 distinct images
costs **3,890 x $0.0001 = $0.39** and would settle the question outright by
measuring the real maximum-unrelated similarity. Until that is done, a photo
match should still not be the sole basis for a cross-bank claim — corroborate
it with `hashed_id_number`, exactly as in mock mode.

Note also that `vertexai.vision_models`, which the embedder uses, emits a
deprecation warning naming a removal date of 24 June 2026. It still served
this run on 2026-09-27, but it is living on borrowed time and the call will
need moving to the current SDK.

### What the run cost

Usage counted exactly; rates are list prices read from the Cloud Billing
Catalog API for this project on 2026-09-27.

| item | units | rate | cost |
| --- | ---: | --- | ---: |
| Document AI Form Parser | 50 pages | $0.03/page ("general" type pages, first 1M) | $1.50 |
| `multimodalembedding@001` | 51 images | $0.0001/image | $0.0051 |
| **total** | | | **$1.5051** |
| **per document** | | | **$0.0301** |

The 51st embedding is one sampled image embedded a second time, to check that
the model returns the same vector for the same bytes — which is what makes a
recycled document detectable at all.

Cost is dominated entirely by Document AI. The embedding half of this — the
part that actually improves reuse detection — is **0.3% of the bill**. If only
one cloud path is adopted, the embeddings are the one that pays for itself.

## Known limitation: the mock photo embedder

The offline difference hash is exact on the case that matters — a recycled
identity points at literally the same file, so it always matches — but it is a
*layout* hash, and the synthetic documents share a layout. Measured on the full
seed-42 run:

* all 2,991 distinct ID cards hash uniquely at the default 32-grid;
* one pair of unrelated **address proofs** collides, producing 2 false
  cross-bank photo pairs out of 1,792 (0.11%) and pulling in 1 account that no
  id-number or address match supports.

So in mock mode a photo match should not be the sole basis for a cross-bank
claim; corroborate it with `hashed_id_number`. An 8-grid hash is far worse
(collides on ~two thirds of cards) — hence the 32 default. Vertex multimodal
embeddings do separate that collision; see the measurement above for how
narrowly.

## Proposed contract addition: `ReusePair`

`shared/contracts.py` has no reuse-pair model, so `ReusePair` is defined locally
in `docai/reuse.py`. `AccountEvidence` is the per-account roll-up; `ReusePair`
is the document-level evidence underneath it, which an analyst needs to see
*which two documents* matched. **Proposed for promotion into `shared/`:**

| field                       | type       | notes                                        |
| --------------------------- | ---------- | -------------------------------------------- |
| `pair_id`                   | `str`      | sha256 of (match_type, left_doc_id, right_doc_id) |
| `left_hashed_account_id`    | `str`      | lexically smaller of the two                 |
| `right_hashed_account_id`   | `str`      |                                              |
| `left_bank_id`              | `str`      | aligned with `left_hashed_account_id`        |
| `right_bank_id`             | `str`      |                                              |
| `bank_pair`                 | `str`      | both banks sorted, `"BANK_A\|BANK_B"`        |
| `cross_bank`                | `bool`     | the signal; same-bank is mostly noise        |
| `match_type`                | `Literal`  | `hashed_id_number` / `normalized_address` / `photo_embedding_ref` |
| `match_value`               | `str`      | the shared value; never a raw identifier     |
| `similarity`                | `float`    | 1.0 for exact; embedding similarity for photos |
| `left_doc_id`               | `str`      |                                              |
| `right_doc_id`              | `str`      |                                              |
| `doc_type`                  | `str`      | both sides share it                          |
| `backend`                   | `str`      | `exact` / `dhash` / `vertex-multimodal`      |
| `detected_at`               | `datetime` |                                              |

## Regulations: the live Vertex AI Search index

The four official AML/KYC texts listed in the repo README are indexed into
Vertex AI Search and queried live by `search_regulations` when
`MULENET_DOCAI_MODE=cloud`. **Mock stays the default**, so the test suite
runs offline and free over the labelled placeholder corpus.

| resource | id |
| --- | --- |
| data store | `mulenet-aml-regulations` (location `global`, `CONTENT_REQUIRED`, digital parser) |
| search app | `mulenet-aml-regulations-app` (Enterprise tier) |
| corpus in GCS | `gs://mulenet-mulenet-data/regulations/` (4 PDFs, 8,867,736 B) |
| import metadata | `gs://mulenet-mulenet-data/regulations_metadata/regulations_metadata.jsonl` |

### Why the index is not in India

The rest of MuleNet moved its inference to `asia-south1` (see
`docs/residency.md`), and this index did not, because **Discovery Engine has no
Asia location**. Its supported locations are `global`, `us` and `eu` only —
read off the current locations page on 2026-09-27, which lists exactly `eu`
("multiple regions in the European Union"), `us` ("multiple regions in the
United States") and the `global` default, with no Asia-Pacific entry. There is
no asia-south1 data store to create, so the corpus PDFs were uploaded to
`gs://mulenet-mulenet-data` (us-central1), the bucket that existed at the
time this index was built -- MuleNet's data has since moved to
`asia-south1` too (see `infra/asia_south1_migration.md`), but this corpus
was not re-uploaded alongside it: there is still no asia-south1 Discovery
Engine location for the index itself to move to, so moving only the PDFs
would gain nothing.

This is a capability limit, not a residency decision that was available and
declined. It also costs nothing here: the corpus is public Government of India
and RBI text, published for anyone to download, so where a copy is indexed
raises no residency question. Nothing customer-derived goes into this index.

Document AI went the other way — see "Region" below.

### Every passage carries provenance, by construction

`search_regulations` previously returned placeholders, and an early live run
had the report agent emit five citations that all pointed at a FinCEN URL the
tool had never returned. That is the failure this path is built against.

Each result carries the document title, a section (the page the extractive
segment came from) and the issuing authority's URL:

```json
{
  "citation": "PMLA, 2002, p. 14",
  "doc_name": "The Prevention of Money-Laundering Act, 2002",
  "section": "p. 14",
  "source_uri": "https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf",
  "excerpt": "[12. Reporting entity to maintain records.-(1) Every reporting entity shall- ..."
}
```

Three rules make a fabricated citation unable to pass as real:

1. **Nothing is composed.** Every field of a passage comes from the API
   response or from the manifest entry that response resolved to. No part of a
   citation is built from the query, from a model, or from a template a caller
   supplies. A citation string exists only if a retrieval produced it.
2. **The corpus is a closed set.** `regulations/corpus_manifest.json` is the
   only list of documents a citation may name, and it is read from the repo,
   not from the index — so an unexpected document appearing in the data store
   cannot widen what a report may cite. A result whose document id is not in
   the manifest is dropped, not rendered.
3. **No text, no citation.** A result with no extractive segment, or an empty
   one, is dropped rather than returned with an empty excerpt. There is nothing
   to quote, so there is nothing to cite.

`agents/regulations.verify_citation()` then checks a model's citation by exact
string match against what the corpus actually returns, and it routes through
this module whenever `docai` is importable. The two halves compose: `agents/`
will not accept a citation the corpus did not return, and this module will not
return one the API and the manifest did not jointly produce.

**`source_uri` is the authority's URL, not the `gs://` object.** Discovery
Engine's own `derivedStructData.link` is
`gs://mulenet-mulenet-data/regulations/pmla_act_2002.pdf`, which nobody can
open and which is not a citation. The manifest's `source_url` is.

**What `section` is, honestly.** It is the page the extractive segment was
taken from — `p. 14` — not a legal section number like `s. 12`. The page comes
straight from the API's `pageNumber` and is exact. A section number would have
to be inferred from the passage text, and a *guessed* section number in a
compliance citation is the same class of error as a fabricated one, so it is
not guessed. An analyst who opens the source URL at that page finds the quoted
text.

### Standing the index up

The data store and the app are created once, by hand. `index_regulations.py`
only uploads and imports, which is the part that re-runs when the corpus
changes. Every command passes `--project mulenet` explicitly, because gcloud's
default project on the build machine is something else.

```bash
gcloud services enable discoveryengine.googleapis.com --project mulenet

# 1. data store: global, unstructured, default (free) digital parser
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token --project mulenet)" \
  -H "Content-Type: application/json" -H "X-Goog-User-Project: mulenet" \
  "https://discoveryengine.googleapis.com/v1/projects/mulenet/locations/global/collections/default_collection/dataStores?dataStoreId=mulenet-aml-regulations" \
  -d '{"displayName":"MuleNet AML/KYC regulations","industryVertical":"GENERIC","solutionTypes":["SOLUTION_TYPE_SEARCH"],"contentConfig":"CONTENT_REQUIRED"}'

# 2. search app: Enterprise tier, or extractive segments are refused
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token --project mulenet)" \
  -H "Content-Type: application/json" -H "X-Goog-User-Project: mulenet" \
  "https://discoveryengine.googleapis.com/v1/projects/mulenet/locations/global/collections/default_collection/engines?engineId=mulenet-aml-regulations-app" \
  -d '{"displayName":"MuleNet AML/KYC regulations search","solutionType":"SOLUTION_TYPE_SEARCH","industryVertical":"GENERIC","dataStoreIds":["mulenet-aml-regulations"],"searchEngineConfig":{"searchTier":"SEARCH_TIER_ENTERPRISE"}}'
```

Then upload and import:

```bash
set MULENET_DOCAI_MODE=cloud
set GOOGLE_CLOUD_PROJECT=mulenet
set MULENET_REGULATIONS_DATASTORE=mulenet-aml-regulations
set MULENET_REGULATIONS_GCS_URI=gs://mulenet-mulenet-data/regulations
set MULENET_REGULATIONS_PDF_DIR=D:\LLM\MuleNet\docai\regulations_corpus
python -m docai.index_regulations --mode cloud
```

And to query:

```bash
set MULENET_REGULATIONS_ENGINE=mulenet-aml-regulations-app
python -c "from docai.regulations import search_regulations as s; print(s('records of client identity', 2))"
```

### Three things that will waste your afternoon

**The engine id is required, not optional.** A search against a *data store*
serving config is refused with `HTTP 400: Cannot use enterprise edition
features (website search, multi-modal search, extractive answers/segments,
etc.) in a standard edition search engine`. Enterprise edition is enabled at
the app level only, and extractive segments are what carry the page number. So
there is no usable data-store path, and `MULENET_REGULATIONS_ENGINE` is
checked before the first call rather than failing at query time.

**The import reports nothing until it is done.** The first import of these four
PDFs took **10m59s** (`createTime` 06:39:58Z, `updateTime` 06:50:57Z). For all
of that time the long-running operation had `done` unset, no `successCount`,
and an `updateTime` 48 ms after its `createTime` — indistinguishable from a
stuck job — and the branch listed zero documents, then four at once. On
completion it did report honestly: `done: true`, `successCount: 4`,
`totalCount: 4`, no error samples. So the operation is trustworthy once it
finishes; it just tells you nothing while it runs. A 200 on
`documents:import` means the request was accepted and nothing more. Verify by
listing the branch's documents and by running a query that returns a real
passage.

**`dataSchema` must be `"document"`, not `"content"`.** `"content"` imports the
bare PDFs and leaves the title to whatever the file's own metadata says.
`"document"` reads the JSONL's `structData`, which is where the title and the
official source URL come from.

### What this costs

List prices read from the Cloud Billing Catalog API for this project on
2026-09-27, so they are the current published rates rather than remembered
ones:

| SKU | rate | this corpus |
| --- | --- | --- |
| Vertex AI Search: Search API Request Count - Enterprise | first 10,000/month free, then $0.004 each | $0.00 |
| Vertex AI Search and Conversation: Data Index | first 10 GiB-month free, then $5/GiB-month | $0.00 (8.46 MiB) |
| digital parsing at ingest | no SKU — not billed | $0.00 |
| Cloud Storage standard, us-central1 | $0.020/GiB-month | ~$0.0002/month |

So the whole index, including every query run to verify it, is inside the free
tier. The paid alternative worth knowing about is `Doc AI Layout Parser for
parsing` at **$0.01/page**; on this 825-page corpus that is ~$8.25 per
re-index. It would give layout-aware chunking rather than page numbers. It was
not used: the page locator is already exact, and the digital parser is free.

## Regulation corpus on disk

`docai/regulations_corpus/` is gitignored — these are third-party Government of
India documents and they are reproducible from the URLs in the repo README
under "Regulation corpus". `regulations/corpus_manifest.json` is committed: it
carries the title, version, authority, official URL and byte size of each PDF,
and it is both the import metadata and the closed set of citable documents.
The repo is public: do not commit PDFs whose licence forbids redistribution.
