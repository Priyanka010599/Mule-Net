# Submission checklist — pass/fail with evidence

| Check | Result | Evidence |
|---|---|---|
| Deployed URL reachable from a clean request | **PASS** | `curl -s -o /dev/null -w "%{http_code}"` against the deployed Cloud Run URL (withheld from this public release) → `200` (cold request, ~5s). `/health` → `200`. `/openapi.json` → `200`, 94,732 bytes (FastAPI's real schema, not a stub page). |
| Repo has no committed secrets | **PASS** | Grepped all tracked `.py/.md/.json/.yaml/.yml/.sh/.js/.html` files for common key shapes (Google/AWS API key patterns, PEM private-key headers, `sk-`/`xox` tokens) — zero matches. `.gitignore` excludes `.env*`, `*-key.json`, `service-account*.json`, `secrets/`, `*.pem`, `*.key`. |
| No committed raw transaction/KYC data or regulation PDFs | **PASS** | `git ls-files \| grep -iE '\.pdf$'` → no matches. Only tracked data-shaped files: 3 tiny synthetic sample CSVs (`ui/static/samples/sample_bank_{a,b,c}.csv`, for the upload-demo feature) and `docai/regulations/placeholder_corpus.json`, which opens with an explicit `"THESE PASSAGES ARE PLACEHOLDERS… not quotations from any real law"` disclaimer. `ml/features.parquet` sitting on disk locally is **not** git-tracked (`git ls-files ml/features.parquet` → empty) — the `.gitignore`'s `*.parquet` rule is actually working. |
| README present and substantive | **PASS** | `README.md`, 27,423 bytes, includes architecture diagram, reproduce steps, a self-authored "what went wrong" postmortem section, and a limitations section. |
| LICENSE present | **PASS** | `LICENSE`, 11,548 bytes, Apache License 2.0 — matches the README's own licence footer. |
| Reproduce steps present | **PASS, with caveats** | Every module named in the README's reproduce block (`simulator.generator`, `ml.early`, `ml.leadtime`, `ml.publish_all`, `ml.claims`, `ml.coalition`, `pytest`) is a real, present module. **Caveats:** (1) README pins Python 3.11; this audit ran the suite successfully under Python 3.14.5 / pytest 9.1.1 instead — the 3.11 pin was not itself exercised. `pyproject.toml:3-7` already explains why the pin exists (`torch`/`torch_geometric`, used only by the optional GNN experiment, won't build from source on newer interpreters) and gives the `py -3.11 -m venv` command for local work, so this is documented behavior, not an undocumented gap. (2) This audit ran `pytest` against the repo's *existing* local state (pre-built `ml/out/`, demo snapshot, etc.), not a from-scratch clone with nothing pre-generated — so full zero-to-green reproducibility was not independently timed end-to-end, only each individual command's presence and the suite's pass rate against what was already there. |
| Test suite runs and is (mostly) green | **MOSTLY PASS** | `python -m pytest -q` (testpaths: `enclave simulator ingest ml docai agents ui`): **1,695 collected, 1 failed, ~6 skipped, ~1,688 passed.** The one failure (`enclave/test_live_proof.py::test_build_instance_is_confidential_sev_with_the_workload_sa`) is a stale assertion (expected VM labels dict doesn't include a `mulenet-run` label the code now sets) — trivial, not security-relevant, but it is a real FAILED line on `main` today. |

## What this audit could **not** verify

- **A demo video under 3 minutes.** Not present in the repo (`git ls-files` for `.mp4/.mov/.webm` → no matches), and no link to one found in README.md or docs/*.md. **Cannot confirm it exists, its length, or its content.** If the submission requires one, it must live outside this repo (e.g. the contest platform) — say so rather than guess.
- **A pitch deck.** Same: no `.pptx`/`.pdf`/`.key` tracked, no link found. **Cannot confirm.**
- **Whether `infra/cloudsql_deploy.sh` has genuinely never been run** (the README claims Cloud SQL/AlloyDB are unprovisioned) — this would require live GCP console/API access this audit doesn't have. Plausible given the file's own "unrun" framing, not independently confirmed.
- **A from-scratch, zero-artifacts reproduce run.** See the caveat above — this audit verified the named commands exist and that the suite passes against the repo's current (partially pre-built) state, not a timed clean-room run from `git clone` with an empty venv.

## Timed 3-minute demo script, based on what actually works today

Built from what this audit directly confirmed live — the live chat
backend field, the live metrics dashboard, and the live `/healthz` /
`/health` split — not from README copy. Each beat is tagged with whether
it depends on anything not confirmed live in this audit.

| Time | Beat | Depends on |
|---|---|---|
| 0:00–0:20 | Open the live URL. Point at tab 1 ("Three banks, three fragments") — each bank's isolated view of the same ring. **Confirmed live: loads in ~4-5s, clear instantly.** | Nothing unconfirmed. |
| 0:20–0:45 | Click "Join the banks." The ring completes across all 3 banks. Hit "Replay" to show the money moving. | Nothing unconfirmed — this is client-side rendering of already-loaded data. |
| 0:45–1:15 | Switch to tab 3 ("What joining up buys"). Point at the "Rings detected, by banks joined" bars (0% → 60.3% → 100%) and "False alerts per mule caught" (6.23 → 1.97). **Confirmed live — this audit's own browser walkthrough saw these exact numbers on the deployed page, matching `docs/claims.md`.** | Nothing unconfirmed. |
| 1:15–1:30 | Point at "Attacks tried against the rule: 3 tried · 0 caught · 3 got through." Say one sentence: we red-teamed our own detector and are showing the losses, not hiding them. | Nothing unconfirmed — visible on the live dashboard. |
| 1:30–2:15 | Click into a ring, hit "Report." Gemini drafts an STR citing a real regulation page. **If time allows, ask the chat a real regulatory question live** (this audit did exactly this and got a correctly-cited RBI KYC answer in ~9s). | **Live Gemini latency is ~2–9s per the calls made during this audit — budget for it, and have a backup screenshot in case the 5-question/60s session rate limit is still cooling down from a prior demo run (this audit hit that limit itself during testing).** |
| 2:15–2:45 | Point at the "Live proof" card: a real Confidential Space VM ran, got the key, joined 349,524 edges. **Do not click "Run live proof" live on stage** unless you have budgeted the ~2–3 minutes and the cost of a real VM — the card already shows a captured, verified prior run; clicking it live is optional and risky against a 3-minute clock. | **This audit did not trigger a fresh live-proof run (cost + time); the card's number is confirmed to be a captured snapshot, by the endpoint's own code comment, not a fresh live call.** |
| 2:45–3:00 | Close on the one honest caveat: all numbers are synthetic-data results; real-world accuracy is unknown until a pilot with real banks. Say it before a judge has to ask. | Nothing unconfirmed — this is exactly what the README already says, and saying it yourself pre-empts the single biggest credibility risk this audit found (see `audit.md`, FINISH section). |

**What NOT to promise live, based on this audit's findings:** don't claim the Document AI KYC parser is "live" (it isn't wired in); don't lead with the 44%-ring-level-recall figure unprompted, but have the number ready if asked (it's real, just not on the dashboard — see `claims_vs_reality.md` #23); don't run `pytest` live on stage without first fixing the one failing enclave test.
