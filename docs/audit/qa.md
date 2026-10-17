# The 15 hardest questions a judge could ask

Each answer is built only from evidence gathered in this audit (code read,
commands run, or live HTTP/browser responses) — not from taking the
project's own framing on faith.

---

**1. "Your headline numbers are 100% recall and 1.97 false alerts — but that's on data you generated yourselves. Why should we believe any of this generalizes to real banks?"**

*Confident answer supported.* It doesn't, and the project says so itself
(`README.md:70-77,313-316`: "Real-world accuracy is unknown until a pilot
with real banks"). What *is* independently defensible is the experimental
design: a built-in negative control (span-1 rings, which are fully visible
to a single bank by construction, show the cross-bank "lift" is slightly
*negative* — `-0.009` — exactly where it should be zero-to-negative, which
is why the `+0.174`/`+0.157` lifts on span-2/3 rings are believable rather
than an artifact of measuring the wrong thing). I verified this table is
real (traced to `docs/claims.md` with JSON key paths, drift-tested by
`ml/test_claims.py`, which passed in this audit's own pytest run) and that
the live dashboard shows the identical numbers. The design is sound; the
external-validity question is honestly unanswered, not hidden.

---

**2. "Your dashboard prominently shows 100% ring recall, but your own README says ring-level recall collapses to 44%. Which number is real?"**

*Confident answer supported — I traced both.* Both are real and measure
different things. The `100%`/`92.5%` pair is **account/ring-detection
recall at a time horizon** (did the system flag the ring in time, via
`ring_catch.cross_bank.ring_recall` at T=180m). The `44%` figure is a
**downstream clustering yield**: 87.8% of individual accounts get flagged
correctly, but ring membership is reconstructed by clustering flagged
accounts together, and only 53/120 true rings (44.2%, `ml/stage2.py:10`)
end up as a *clean, fully-reconstructed cluster* a judge can click into in
the UI. I verified the `100%` figure is on the drift-tested `docs/claims.md`
ledger; the `44%` figure is **not** — it's real but traced only to a code
comment and prose, which is itself worth raising with the team (see
`claims_vs_reality.md` #23).

---

**3. "Your rate limits and daily spend caps are in-process memory. What actually stops costs from blowing up if Cloud Run autoscales past one instance, or if an instance crash-loops and resets the counters?"**

*Confident answer supported — and it's a real gap, not a solved problem.*
`agents/limits.py`'s own module docstring says exactly this: the counters
are per-instance, so caps are only correct "regardless of auth" at the
currently-pinned `--max-instances=1` (confirmed at `infra/deploy.sh:705`).
Nothing in the code itself prevents a future deploy from raising that
flag and silently multiplying the effective budget. A crash-restart loop
would also reset the in-memory daily counter early. This is disclosed in
comments, not hidden, but it is a genuine single point of failure in the
cost story, with no code-level guardrail against it today.

---

**4. "Can a malicious or compromised bank poison the enclave's cross-bank join with fabricated data attributed to another bank?"**

*Partially supported — I read the claim and the test file names, not the
full cryptographic argument.* `docs/data_sharing.md` and the README both
claim the enclave "will refuse a bank that tries to poison or misattribute
another bank's data," and `enclave/test_negative.py` is a real, substantial
test file aimed at exactly this. I confirmed the file exists and that the
attested-credential design (`enclave/kms.py`) is real, non-trivial code —
but I did not trace the specific policy/cryptographic enforcement path
(`enclave/policy.py`, `enclave/rings.py`) line-by-line to independently
confirm the negative-test claim holds for every poisoning strategy, nor did
I re-run the negative tests live against a real deployed enclave. **I
cannot give this one full confidence; I can say the claim is backed by a
real, named test rather than being asserted with no artifact at all.**

---

**5. "Is this compliant with India's Digital Personal Data Protection (DPDP) Act, given you're processing account and transaction data?"**

*Cannot answer well — and the project doesn't claim to.* The README's own
roadmap (`README.md:112`) lists "DPDP Act review" as a **not-yet-done**
item, explicitly under "Roadmap (not production-ready yet)." This audit
found no DPDP-specific compliance documentation anywhere in `docs/`. The
honest answer a judge should get is: this is explicitly future work, named
as such by the team, not something this system currently claims to satisfy.

---

**6. "Why XGBoost and not a graph neural network, given this is fundamentally a graph/network problem?"**

*Confident answer supported.* The team built a GraphSAGE GNN (`ml/gnn.py`
is real, non-trivial code), found it was seeing test accounts' graph
structure during training (a leakage mode specific to GNNs on
transductive graphs), and disclosed that rather than ship it
(`README.md:101-104`). I did not independently re-run the leakage
diagnosis myself, but the artifact (the GNN code) and the stated reasoning
are both real, and the decision to disclose a negative result rather than
suppress it is a credible engineering signal in itself.

---

**7. "If I ask your chatbot for another bank's customer's address or phone number, what actually stops it from telling me?"**

*Confident answer supported — live-tested in this audit.* I asked exactly
this live (`"Can you give me the full home address and phone number of the
account holder in ring R1?"`) and the deployed system correctly declined:
`"cannot be answered from the allowed tables. Phone numbers and account
holder names are not stored in the schema."` The protection is
structural — the database schema itself has no PII columns (account IDs
are HMAC-hashed everywhere, per `README.md:458-459`, independently
confirmed via secrets/PII grep) — not a content filter that could be
prompted around.

---

**8. "What stops Gemini from hallucinating a citation to a regulation page that doesn't actually say what you claim?"**

*Confident answer supported — live-tested.* `agents/verify.py` verifies
citations by **retrieval identity** (did `search()` actually return this
exact citation, not "does re-querying confirm it") — the README documents
a real historical bug where the opposite approach (re-querying to confirm)
failed on 100% of real citations because page text has little lexical
overlap with its own citation string. I independently live-tested a real
regulatory question and got a citation that, when I read the quoted
excerpt, genuinely matches the real RBI KYC periodicity rule (2/8/10 years
by risk tier) — not a plausible-sounding fabrication.

---

**9. "Of everything in your 'Google Cloud services actually used' table, how much is really load-bearing in the live user path today, versus just a logged artifact?"**

*Confident answer supported — this audit built the list independently.*
**Load-bearing today, confirmed live or by direct code inspection:**
Gemini on Vertex AI (live chat), Vertex AI Search (live, real citations),
Cloud Run (all three services, confirmed reachable), rate limiting/IAM
(code-confirmed). **Real but not on the live serving path:** Vertex AI
Model Registry (`servable: false`, self-disclosed and confirmed — no model
endpoint call found anywhere in the chat path), BigQuery `features` table
(schema exists, no writer found anywhere in the codebase), Document AI KYC
parser (built, explicitly not wired into the live app). **Captured
snapshot, not re-verified live by this audit:** the Confidential Space
enclave join (real code, but the dashboard serves a frozen prior result by
design, not a fresh call per page load).

---

**10. "What's your unit economics, and what actually caps the bill if this goes viral on a public, unauthenticated URL?"**

*Partially supported.* The README states per-unit costs for some paths
(~$0.0039/Gemini question, ~$0.05/Vertex training job, $0.0301/Document AI
doc) but this audit found no stated dollar figure for a live-proof VM run
beyond "real paid VM." The cap mechanism is real and three-tiered
(session/IP/daily, `agents/limits.py`, live-tested in this audit via a
genuine 429/rate-limit response with the exact hard-coded message), but it
inherits the weakness in question #3 — it's an in-memory, per-instance
guard pinned to `--max-instances=1` by convention, not enforced
structurally.

---

**11. "Does this actually scale past 3 banks, or is 3 a hard-coded assumption?"**

*Partially supported.* `ml/coalition.py` is real, general code (confirmed
by reading it — it tests combinations programmatically, not a lookup
table), and `ui/static/app.js:1919-1928` cites an `ml/experiments/n_banks/`
result claiming flat-to-improving false-alert ratios at N=5 and N=6. I did
not independently re-derive those N=5/N=6 numbers in this audit. The same
UI card self-hedges "still synthetic-world only; not yet validated against
real additional banks" — so the honest answer is: the *architecture*
doesn't hard-code 3, and there's a real (if self-reported, not
independently re-verified by me) experiment suggesting it holds at 5–6,
but nothing has been tested at the scale of a real national consortium
(dozens of banks).

---

**12. "Is this actually novel, or is it XGBoost plus an LLM wrapper that any team could build in a weekend?"**

*Confident answer supported, with nuance.* The ML model itself (gradient-
boosted trees on hand-engineered features) is not novel. What this audit
found to be genuinely substantial, real engineering — not a wrapper —
is: (a) the Confidential Space cross-bank join (real attested-credential
code, `enclave/kms.py`, not a mock), (b) the deterministic verifier that
re-derives claims from data rather than trusting the LLM's own arithmetic
(`agents/verify.py`, demonstrated live suppressing an ungrounded answer),
and (c) the team's own red-teaming of its detector (live-visible: "3
attacks tried, 0 caught"). None of that is a weekend build. The underlying
detection model, by contrast, is intentionally unglamorous (the README
says so: "XGBoost is explainable, honestly evaluated and servable live").

---

**13. "Have you load-tested the live Gemini chat path, and what's the latency at scale?"**

*Cannot answer well — not demonstrated.* This audit observed single-request
latencies of roughly 2.7-9.5 seconds across 5 live chat calls, and
triggered the real session rate limiter (5 requests/60s) simply by testing
normally. No load-test scripts, results, or k6/locust-style artifacts were
found anywhere in the repo. The honest answer: latency-under-concurrency is
untested, and the architecture (`--max-instances=1` for agents) means
concurrent judges hitting the demo simultaneously would compete for the
same rate-limit budget.

---

**14. "RBI already has MuleHunter.AI — why would a bank or regulator adopt this instead?"**

*Confident answer supported, with one correction.* I externally verified
(web search) that MuleHunter.AI is real: an RBI Innovation Hub (RBIH)
machine-learning tool, announced December 2024, piloted at individual
public-sector banks, analyzing single-institution transaction data. The
project's own positioning (`docs/data_sharing.md:189-193`) — that MuleNet's
specific addition is **cross-bank ring structure**, which a
single-institution tool structurally cannot see — is accurate and
defensible. The one correction: describing MuleHunter.AI as sitting "on
the regulator side" slightly overstates how centrally RBI runs it; it is
RBI-affiliated (via RBIH) but piloted bank-by-bank, not operated by RBI on
pooled data today.

---

**15. "Your test suite has a failing test on `main` right now — why should we trust anything else you're showing us?"**

*Confident answer supported — this audit found and diagnosed it directly.*
`enclave/test_live_proof.py::test_build_instance_is_confidential_sev_with_the_workload_sa`
asserts the Confidential VM's labels dict equals exactly
`{"purpose": "live-proof"}`; the actual code
(`enclave/live_proof.py:259`) now also sets `"mulenet-run": run_id` — a
label added on purpose (for a cleanup-reaper filter, per the adjacent
comment) that the test was never updated to expect. It is a stale
assertion, not a logic or security defect, and it is the **only** failure
in 1,695 collected tests (~1,688 passing, ~6 skipped for missing optional
browser-automation dependencies). A judge who runs `pytest` and sees one
red line, out of nearly 1,700 tests, with an obvious and tiny root cause,
is seeing evidence of a generally well-tested codebase with one
housekeeping miss — not evidence the rest is unreliable. That said, it is
real, it is on `main`, and it should be fixed before judging (10-minute
fix).
