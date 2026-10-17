# enclave

The Confidential Space workload that lets three banks score a shared
transaction graph jointly, with Cloud KMS releasing keys only to the attested
image. Owned by the **platform-engineer** agent.

The design is [`docs/data_sharing.md`](../docs/data_sharing.md). The runbook,
the honest status, and the exact commands per stage are
[`docs/enclave.md`](../docs/enclave.md).

**Status:** the workload is built and tested; **nothing has been executed
against Google Cloud.** No KMS key, no attestation policy, no Confidential VM.
Treat every attestation claim as a prediction until a credentialed run confirms
it.

The key is released only to a **non-debug** Confidential Space image: the
`confidential-space-debug` family allows operator SSH, so a VM running it must
never hold the banks' joined graph. That is enforced by the provider condition,
not by convention — see `docs/enclave.md`.

## Modules

| file | role |
|---|---|
| `config.py` | env-var config; on a Confidential VM these arrive as `tee-env-*` metadata, so nothing secret travels this way |
| `envelope.py` | AES-256-GCM per-bundle data key, wrapped by KMS; sending bank bound into the AEAD |
| `bundle.py` | what a bank seals — carried as existing `shared/` contracts, no new schema |
| `kms.py` | the only module that talks to Cloud KMS; turns attestation into a credential and a refusal into `KeyWrapperError`. Read its docstring on the **two identities**: KMS uses the attested principal, GCS uses the VM's attached service account, which is why a refused run can still write down why |
| `rekey.py` | Stage 4 re-keying inside the enclave. **Read its docstring before describing what Stage 4 proves** |
| `graph.py` | joins the three contributions; feature names derived from `ml.features.FEATURE_NAMES`, checked at import |
| `rings.py` | cross-bank connected components among flagged accounts |
| `policy.py` | Stage 3: each bank receives its own accounts only |
| `inbox.py` | the operator-owned inbox/outbox layout |
| `workload.py` | entrypoint; `--self-check` runs the whole join with no cloud calls. Writes an evidence object on **every** run, refusals included |

## Try it without Google Cloud

```bash
python -m enclave.workload --self-check   # 3-bank join, clustering, output policy
python -m pytest enclave/ -q              # 85 tests
```

`--self-check` exists so that a failure on a real Confidential VM can be
attributed to key release rather than to this code.
