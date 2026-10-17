# The MuleNet enclave: runbook and honest status

The design is in [`data_sharing.md`](data_sharing.md). This page is the
operational half: what exists, what has never been run, the exact commands for
each stage, and what output proves a stage passed.

## Status in one line

**Stage 1 has passed, live, on Google Cloud.** Self-check, a real production run
(`key_release_ok: true`, `running_debug_image: false`), and both negative tests
(`kms_decrypt_refused` for a tampered image, `sts_exchange_refused` for the
unmodified image on the debug family) all passed against evidence objects read
from GCS, not exit codes. Stages 2–4 have not run yet. Sections below written
before that run are marked; the rest still describes the design, not a result.

**One limitation found running Stage 1 for real, not caught by any offline
review:** the KMS policy grants `cryptoKeyDecrypter` only to the attested image
digest, but the *project owner's own account* can still decrypt both keys — the
basic Owner IAM role includes KMS decrypt, and the digest-only grant cannot
override that at the project level. Confirmed live: the operator account
decrypted `hash-rekey.bin` directly, no enclave involved. So today the claim "only
the attested image can read the joined data" holds against every other bank and
every other identity, but **not against whoever owns the `mulenet` project.**
Closing this needs an IAM deny policy scoped to KMS decrypt with an exception for
the digest principal — deliberately not applied yet, given the time left before
judging; the deny policy is a real access change, not a config file, and undoing
it needs a role most projects don't grant by default. A real bank consortium
would need this closed, or the keys held in a project none of the participating
banks (including the operator) has Owner on.

## Why nothing ran

Three independent blockers in the build session, each verified rather than
assumed:

1. **No Google Cloud credentials.** `CLOUDSDK_AUTH_ACCESS_TOKEN` was set to the
   literal placeholder `proxy-injected`, and unauthenticated calls to
   `cloudresourcemanager.googleapis.com` returned
   `401 UNAUTHENTICATED / CREDENTIALS_MISSING`. Service-account key creation is
   blocked by org policy (`iam.disableServiceAccountKeyCreation`), which is the
   right policy — hence the impersonation-based approach throughout.
2. **The Google Cloud documentation site was unreachable.** Every
   `cloud.google.com/...` docs path now 301s to `docs.cloud.google.com`, which
   the egress policy blocks (403 on CONNECT). API shapes were therefore taken
   from primary sources that *were* reachable, listed under
   [Where the API shapes came from](#where-the-api-shapes-came-from).
3. **`dl.google.com` was blocked**, so the Cloud SDK could not be installed from
   its usual host. It was eventually installed from
   `storage.googleapis.com/cloud-sdk-release/`, which is reachable.

## Region decision

`asia-south1` (Mumbai) is the configured default: Gemini and Document AI are
already there (`docs/residency.md`), and for an India BFSI entry Mumbai is the
defensible answer.

**This is a preference, not a verified capability.** Confidential VM support is
a property of a machine type in a specific *zone*, and it was not possible to
check it live. `infra/enclave_preflight.sh` performs that check and refuses to
continue if the zone cannot run the machine type; `infra/enclave_run.sh` surfaces
the instance-create API's own refusal verbatim if `SEV` is unavailable there. If
`asia-south1-a` turns out not to support it, set `ENCLAVE_ZONE` and
`ENCLAVE_REGION` to a zone that does **and record the reason here** — the scripts
deliberately do not relocate silently.

Note the KMS key and the VM share a region (`ENCLAVE_KMS_LOCATION` defaults to
`ENCLAVE_REGION`) so the key does not sit in a different jurisdiction from the
workload. The workload identity pool is `global`, as in every Google sample.

## Cost

**Nothing has run, so the spend so far is zero.** No hourly rate is hardcoded
anywhere: the pricing pages were unreachable, and an unverified price in a script
is worse than none. `infra/enclave_run.sh` measures real VM uptime and converts
it to an estimate only if you supply `ENCLAVE_VM_HOURLY_USD`.

Budget is ~INR 26,000 of trial credit. The cost shape to expect: a
Confidential VM is billed per second while it exists, plus a Confidential
Computing premium on top of the base machine type; KMS charges per key version
per month plus per operation, both negligible here. Every stage below ends with
the VM deleted, so the recurring cost between stages is the KMS key versions and
three near-empty buckets.

## The key-release policy

The key is released to **one container image digest, running a non-debug, stable
Confidential Space image, on a Secure Boot GCP confidential platform.** The
condition `infra/enclave_setup.sh` installs on the workload identity pool
provider is:

```
assertion.swname == 'CONFIDENTIAL_SPACE'
&& assertion.dbgstat == 'disabled-since-boot'
&& 'STABLE' in assertion.submods.confidential_space.support_attributes
&& assertion.secboot
&& assertion.hwmodel.startsWith('GCP_')
```

Each term earns its place:

| term | what it stops |
|---|---|
| `swname` | some other COS image impersonating Confidential Space |
| **`dbgstat`** | **a debug-family VM, which the operator can SSH into and read memory from** |
| `STABLE` | the same thing from the other side (see below) |
| `secboot` | a boot chain that does not actually measure the image the digest claim rests on |
| `hwmodel` | anything not a real GCP confidential platform |

### Why `dbgstat` is the term that matters

`confidential-space-debug` permits operator SSH. A VM running it while holding
three banks' joined transaction graph hands the operator exactly the plaintext
this design exists to withhold — while running the blessed, correctly attested
code. Without this term the whole attestation story reduces to "the operator
promises to boot the right family".

This was a real flaw in the first version of these scripts: the condition checked
only `swname`, **and the debug family was the default**, because serial-console
log redirect was how the demo showed the KMS refusal. Fixing it meant replacing
the evidence mechanism, which is what the output bucket below is for.

`enclave_setup.sh` reads the condition back from the API and **refuses to
continue** unless it contains `disabled-since-boot`. That check is separate from
the `CONFIDENTIAL_SPACE` one on purpose: a missing `CONFIDENTIAL_SPACE` term
breaks loudly, whereas a missing `dbgstat` term breaks nothing and protects
nothing. Everything would work, the demo would pass, and the key would go to a
readable VM.

### The `STABLE` term — now verified live

Google's own image database marks a hardened (production) entry
`[LATEST, STABLE, USABLE]` and a debug entry with **no labels at all**, so a
debug image cannot carry `STABLE`. That makes it a second, independent expression
of the same requirement.

**Confirmed against real evidence, not predicted.** Every production evidence
object from Stage 1's live run carries `"support_attributes_raw": ["LATEST",
"STABLE", "USABLE"]` and `"support_attributes_type": "list"` — the list shape,
one of three the documentation described as possible (see
`ENCLAVE_SUPPORT_ATTR_CHECK` in `infra/enclave_config.sh` for the alternatives
that turned out not to be needed). The debug-family evidence object shows
`"stable_support_attribute_present": None`, confirming the debug image really
does carry no `support_attributes` term at all.

`dbgstat` alone would have blocked the debug image even if `STABLE` had to be
dropped, so this was never a risk — only a loose end, and it is closed now.

## Evidence

**Evidence is a JSON object in `gs://<project>-enclave-evidence`, not a log line
and not an exit code.** The workload writes exactly one object per run to
`runs/<run-id>/<attempt>.json`, on **every** terminal path, including runs where
KMS refused the key. It records:

| field | why |
|---|---|
| `outcome` | `completed`, `key_release_refused`, `no_bundles_found`, `self_check_passed`, … Explicit, so a reader never infers "refused" from absent results — absence is also what a crashed VM looks like |
| `refusal_message` | the backend's error **verbatim**, never reworded or truncated |
| `refusal_kind` | which gate refused: `sts_exchange_refused` or `kms_decrypt_refused` |
| `attestation` | the claims the VM presented, including `dbgstat` and `support_attributes` with its raw type |
| `running_debug_image` | `true`/`false`/`null` — `null` means "could not tell", which must not read as "production" |
| `key_release_ok` | whether the key was actually obtained |
| `wrote_evidence_as` | the service account that wrote this object — distinct from the attested principal that asks KMS for the key, recorded so an auditor need not assume they are the same |

### Why a GCS write survives a KMS refusal

Because the two use **different identities**, and that separation is a property
of the design rather than a happy accident:

- **Key release** uses the *attested* federated principal built from the
  attestation token, authorised by an IAM binding on the key naming
  `attribute.image_digest/<digest>`.
- **The evidence write** uses the VM's *attached service account*, resolved
  through the instance metadata server by Application Default Credentials. It
  does not touch the attestation exchange at all.

So when attestation fails, the KMS call is refused and the GCS write still
succeeds. `enclave/workload.py` builds the storage client **before** the gated
step for exactly this reason; building it after would leave a refused run unable
to report itself.

This also fixed a second bug found while doing the work. `credential_config()`
originally included `service_account_impersonation_url`, which would have
exchanged the attestation for the workload service account — making *that
account* the caller at KMS, so the digest-scoped binding would never have been
consulted and "the key follows the code" would have quietly become "whoever holds
this service account can decrypt". The impersonation is gone; the federated
principal is used directly.

**The honest limitation:** the workload service account can write to the evidence
bucket without any attestation, so an evidence object proves *a VM ran*, not that
it was attested. What proves attestation is the object's **content** — a released
key, or a refusal naming the gate that refused it. The workload holds
`objectCreator` and not `objectAdmin`, and passes `if_generation_match=0`, so it
can add a result and cannot overwrite or delete an earlier one: an enclave that
can rewrite its own audit trail is not an audit trail.

## The stages

Run them in order. Each stage stands alone and ends with no VM running.

### Stage 0 — preflight (creates nothing)

```bash
./infra/enclave_preflight.sh
```

**Proves it passed:** `OK` on the active account, the operator project number,
`billingEnabled=True` for `mulenet`, a resolved Confidential Space image name,
zone `UP`, the machine type present in the zone, and a printed vCPU quota table.
Exits non-zero on any blocking failure. An *unbilled* bank project is reported
as `OK`, not a failure — that is the intended design.

### Stage 1 — attested key release

```bash
./infra/enclave_setup.sh                              # keys, pool, provider, SA, buckets
./infra/enclave_build.sh v1                           # prints the image DIGEST
./infra/enclave_pin_digest.sh sha256:<digest>         # grant decrypt to that image only
./infra/enclave_run.sh v1 stage1 selfcheck --self-check   # boot check, no KMS
./infra/enclave_run.sh v1 stage1 production           # real run
```

**Proves it passed:**

- `enclave_setup.sh` echoes the provider's `attributeCondition` read back from
  the API, and prints both `condition includes the non-debug check
  (disabled-since-boot)` and `condition includes the STABLE support-attribute
  check`. It refuses to continue without the first.
- `enclave_pin_digest.sh --list` shows `roles/cloudkms.cryptoKeyDecrypter` held
  **only** by `attribute.image_digest/...` principals. A plain `serviceAccount:`
  there voids the whole Stage 1 claim.
- the self-check run's evidence object has `"outcome": "self_check_passed"` — the
  image boots and the join, clustering and output-policy logic all run inside the
  Confidential VM. It also carries `"key_release_ok": false`, because KMS was not
  contacted; a boot test must not read as a released key.
- the real run's evidence object has `"outcome": "completed"`,
  `"key_release_ok": true`, and `"running_debug_image": false`.
- the VM is confirmed deleted (`instance deleted (verified absent from the API)`).

`enclave_run.sh` **fails** a run that completed while reporting
`running_debug_image: true`, and says why that is a security failure rather than a
pass.

#### The negative tests (the showpiece)

```bash
./infra/enclave_negative_test.sh
```

This runs a positive baseline and then **two** negative tests, which prove
different things. A demo showing only the first would be claiming more than it
proved.

| test | what changes | expected `refusal_kind` | property proved |
|---|---|---|---|
| **A** tampered image | one line of the scorer, rebuilt → new digest, **not** pinned | `kms_decrypt_refused` | **the key follows the code** |
| **B** debug image | nothing in the code; the **same pinned digest**, booted on `confidential-space-debug` | `sts_exchange_refused` | **the key requires no operator access** |

Test B is the one that was impossible to state honestly before: the debug family
used to be the default. It boots the *unchanged, still-pinned* image, so the only
variable is the image family — and a family the operator can SSH into must be
refused even while running blessed code.

**Proves they passed:** the evidence objects under
`gs://<project>-enclave-evidence/runs/<run-id>/`, mirrored locally into
`.enclave-evidence/`:

| file | what it proves |
|---|---|
| `production.json` | the pinned production image **does** get the key — without this neither refusal proves anything |
| `tamper.diff` | the one-line change made for test A |
| `kms-decrypters.txt` | the live KMS policy named v1 and **not** the tampered digest |
| `tampered.json` | test A's `outcome: key_release_refused`, `refusal_kind: kms_decrypt_refused`, and the verbatim refusal |
| `debug.json` | test B's `outcome: key_release_refused`, `refusal_kind: sts_exchange_refused`, the verbatim refusal, and the `dbgstat` the debug VM presented |

and the script prints `PASSED` for each, plus `the two refusals came from
DIFFERENT gates, so they prove two properties`. **If both refusals classify the
same way, only one property is demonstrated and the script says so** — do not
report two.

Both verdicts are read from the evidence object's `outcome` field with `python3`,
never with `grep` (a substring match on JSON would find `key_release_refused`
inside a refusal *message* and misreport a successful run) and never from an exit
code. A missing evidence object is reported **INCONCLUSIVE** — not a refusal and
not a pass.

**The real refusal texts are not in this PR.** The refusal *logic* for both tests
is covered offline by `enclave/test_negative.py`, whose fake models both gates:
production+pinned → released, tampered digest → refused by KMS, debug image with
the correct digest → refused by STS. Each fake changes exactly one thing from a
production baseline, so a refusal can only be attributed to that change.

### Stage 2 — one bank sends encrypted edges

**Run the unbilled probe first. It is the assumption everything rests on.**

```bash
./infra/enclave_bank_identity.sh bank-c mulenet-bank-c   # the UNBILLED one, first
./infra/enclave_bank_identity.sh bank-a mulenet-bank-a
python -m infra.enclave_send_edges --bank-id bank-a \
    --bank-project mulenet-bank-a --run-id stage2 --dry-run   # review first
source infra/enclave_config.sh
python -m infra.enclave_send_edges --bank-id bank-a \
    --bank-project mulenet-bank-a --run-id stage2 \
    --kms-key "$ENCLAVE_KMS_EDGE_KEY_RESOURCE" \
    --inbox-bucket "$ENCLAVE_INBOX_BUCKET" \
    --impersonate "mulenet-bank-a-sender@mulenet-bank-a.iam.gserviceaccount.com"
./infra/enclave_run.sh v1 stage2 production
```

**Proves it passed:**

- `VERIFIED: a service account in mulenet-bank-c (billingEnabled=False)
  authenticated and wrote gs://...` — this is the line that settles the open
  question. If instead the script prints `STOP`, the identity-only model does
  not hold: **report it and stop**; do not add billing to bank-c.
- `mulenet-bank-c-sender@... cannot write outside inbox/bank-c/`.
- the sender prints `no intra-bank edge included (checked, not assumed)`.
- the run manifest lists `"contributing_banks": ["bank-a"]` and names the absent
  banks in `notes`, so a thin run cannot be mistaken for a complete one.

Bank C is deliberately probed first: it is the case that can fail.

### Stage 3 — three banks, per-bank alerts

```bash
for b in a b c; do
  ./infra/enclave_bank_identity.sh "bank-$b" "mulenet-bank-$b"
  python -m infra.enclave_send_edges --bank-id "bank-$b" \
      --bank-project "mulenet-bank-$b" --run-id stage3 \
      --kms-key "$ENCLAVE_KMS_EDGE_KEY_RESOURCE" \
      --inbox-bucket "$ENCLAVE_INBOX_BUCKET" \
      --impersonate "mulenet-bank-$b-sender@mulenet-bank-$b.iam.gserviceaccount.com"
done
export MULENET_ENCLAVE_FLAGGED_BANK_A=... MULENET_ENCLAVE_FLAGGED_BANK_B=... MULENET_ENCLAVE_FLAGGED_BANK_C=...
./infra/enclave_run.sh v1 stage3 production
```

**Proves it passed:** the manifest shows `"contributing_banks"` with all three,
`"absent_banks": []`, a non-zero `ring_count`, and one outbox object per bank per
ring. Then, for the isolation claim:

```bash
gcloud storage cat "gs://$ENCLAVE_OUTBOX_BUCKET/outbox/bank-a/stage3/<ring>.json"
```

Bank A's bundle must contain its own enclave ids, `bank_span`, counts for other
banks, and **every foreign counterparty as `other-bank:bank-b:1`** — no foreign
enclave id anywhere. The workload refuses to write *any* output if that check
fails, so a completed run is itself part of the evidence. The property is
asserted directly in `enclave/test_policy.py` over the serialised bundle.

### Stage 4 — retire the shared `demo-key`

Nothing extra to run: Stage 3 already re-keys inside the enclave, using the HMAC
key `enclave_setup.sh` generated, wrapped by KMS and released only to the
attested digest.

**Proves it passed:** `"key_release_ok": true` in the manifest (the HMAC key was
unwrapped inside the enclave), and the enclave ids in the outbox match neither
the input hashes nor anything derivable from `"demo-key"`.

**Read `enclave/rekey.py`'s module docstring before describing this stage.** It
retires the shared key *as the joining mechanism*. It does **not** retroactively
protect data already hashed under the published `"demo-key"`, and it is not yet
the production ingest path (the bank would seal the raw identifier). Saying
Stage 4 "removed `demo-key`" without that qualification overstates it.

### Teardown

```bash
./infra/enclave_teardown.sh          # VMs only — the safe default
./infra/enclave_teardown.sh --all    # asks for confirmation
```

`--all` destroys the KMS hash key version, which makes **every enclave
identifier ever computed unreproducible**. Any alert a bank already holds becomes
permanently unjoinable to a future run. It also deletes the evidence bucket,
which holds the negative tests' captured refusals — **export anything you need to
keep first**, because those objects are the Stage 1 proof.

## Rebuilding on a new account

KMS keys, workload identity pools and attestation policies do not migrate. To
rebuild from nothing:

```bash
export ENCLAVE_PROJECT=<new-operator-project>
export BANK_A_PROJECT=... BANK_B_PROJECT=... BANK_C_PROJECT=...
./infra/enclave_preflight.sh && ./infra/enclave_setup.sh
./infra/enclave_build.sh v1 && ./infra/enclave_pin_digest.sh sha256:<digest>
for b in a b c; do ./infra/enclave_bank_identity.sh "bank-$b" "$(eval echo \$BANK_${b^^}_PROJECT)"; done
```

Every name and region lives in `infra/enclave_config.sh` and is
environment-overridable. There are **no console-only steps**. Two caveats:

- A deleted workload identity pool name is unavailable for about 30 days, so
  reusing a name from a torn-down account may need a new one.
- The Dockerfile pins its base image by digest (resolved 2026-09-27). Re-resolve
  it deliberately with `docker buildx imagetools inspect python:3.11-slim`, and
  re-pin the KMS binding afterwards: changing the base changes the workload's
  digest and therefore revokes its key release.

## Every IAM binding the scripts create

Recorded because "report every IAM binding" is a requirement, and because the
least-privilege argument is only checkable if the list is explicit. The scripts
also append each binding they actually make to `$IAM_LOG_FILE` and print it.

| scope | member | role | why this and not more |
|---|---|---|---|
| project `mulenet` | workload SA | `confidentialcomputing.workloadUser` | required to obtain an attestation token at all |
| project `mulenet` | workload SA | `logging.logWriter` | a custom runtime SA gets no logging by default, and without logs the negative test has nothing to read |
| artifact repo | workload SA | `artifactregistry.reader` | pull the image; cannot push, so a compromised VM cannot replace it |
| inbox bucket | workload SA | `storage.objectViewer` | read bundles; **not** objectAdmin, so a bug cannot destroy a bank's evidence |
| outbox bucket | workload SA | `storage.objectCreator` | write alerts; cannot read or overwrite prior runs |
| keys bucket | workload SA | `storage.objectViewer` | read the *wrapped* HMAC key |
| evidence bucket | workload SA | `storage.objectCreator` | add a run's result; **cannot** overwrite or delete an earlier one, so the enclave cannot rewrite its own audit trail. Deliberately independent of attestation, which is what lets a refused run report itself |
| KMS `edge-wrap` | `attribute.image_digest/<digest>` | `cloudkms.cryptoKeyDecrypter` | **an attestation attribute, not an account** — this is the Stage 1 claim |
| KMS `hash-rekey` | `attribute.image_digest/<digest>` | `cloudkms.cryptoKeyDecrypter` | same, for the Stage 4 key |
| KMS `edge-wrap` | each bank sender SA | `cloudkms.cryptoKeyEncrypter` | seal a bundle; **never** Decrypter, so a bank cannot read another's |
| inbox bucket | each bank sender SA | `storage.objectCreator` *(IAM condition: own prefix)* | objectAdmin would let a bank overwrite another's bundle and hide its own mules |
| bank sender SA | the operator user | `iam.serviceAccountTokenCreator` | key-free impersonation for the Stage 2 probe; scoped to one SA, not project-wide |

The workload SA notably has **no** BigQuery or AlloyDB role, and the image ships
no client for either: a workload that could write the operator's analytics store
could exfiltrate the joined graph it was trusted to keep private.

## Where the API shapes came from

The docs site was blocked, so nothing here was written from memory alone.

| shape | source |
|---|---|
| attestation claim names (`submods.container.image_digest`, `hwmodel`, `swname`, `secboot`) | `GoogleCloudPlatform/confidential-space`, `codelabs/health_data_analysis_codelab/src/uwear/` (OPA policy + `opa_validation_values.json`) |
| WIF credential-config JSON, attestation token path | same repo, `codelabs/digital_asset_transaction_codelab/src/credential-config.js` |
| `tee-image-reference`, `tee-container-log-redirect`, `--confidential-compute-type=SEV`, `--shielded-secure-boot`, `--image-family=...`, `n2d-standard-2` | same repo, `codelabs/mig_cs_codelab/create_launch_mig.sh` (note: that sample uses the **debug** family; MuleNet defaults to production for the reason given above) |
| `dbgstat` values `disabled-since-boot` (production) / `enable` (debug) | Google Cloud docs pages *Create and grant access to confidential resources* and *Attestation token claims*, read via web search — the docs site itself is egress-blocked from this container |
| `STABLE` is absent from a debug image | `GoogleCloudPlatform/confidential-space`, `server/proto/image_database.proto` (the `AttributeLabel` enum) and `server/image/image_test.go`, whose fixture marks a hardened entry `[LATEST, STABLE, USABLE]` and a debug entry with no labels |
| direct federation works with no `service_account_impersonation_url` | constructed `identity_pool.Credentials` locally and confirmed `service_account_email` is `None` |
| `roles/confidentialcomputing.workloadUser` | same repo, `codelabs/ml_model_protection/scripts/create_workload_service_account.sh` |
| KMS `additional_authenticated_data` field name | introspected from the installed `google-cloud-kms` protobuf descriptors |
| the credential config is accepted by `google-auth` | constructed a real `identity_pool.Credentials` from it locally |
| `python:3.11-slim` base digest | resolved from `registry-1.docker.io` on 2026-09-27 |

**Still unverified by any source**, and the most likely places for this to be
wrong on first run:

- **the CEL shape of the `STABLE` support-attribute check** — list vs
  concatenated string vs `.stable == true`. Isolated in
  `ENCLAVE_SUPPORT_ATTR_CHECK`; `dbgstat` still blocks the debug image if this
  term has to be dropped, and the first run's evidence object settles it;
- the exact `gcloud iam workload-identity-pools providers create-oidc` flag
  spelling, and whether `--allowed-audiences` and the `issuer-uri` trailing
  slash are right;
- whether `principalSet://.../attribute.image_digest/<digest>` is accepted
  verbatim as a KMS IAM member;
- whether an unbilled project can enable `iamcredentials` and mint an
  impersonation token — the Stage 2 probe exists to answer exactly this;
- Confidential VM (`SEV`) availability in `asia-south1-a`.

Expect the first credentialed run to fail on one of these. They are named so
that a failure is recognised as a known unknown rather than a surprise.
