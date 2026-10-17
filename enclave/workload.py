"""The Confidential Space workload entrypoint.

One run does this, in order:

1. read this image's attestation claims (for the manifest, never as a check),
2. ask KMS to release the edge-wrapping key -- **this is the step attestation
   gates**, and the step the Stage 1 negative test expects to fail,
3. unwrap the Stage 4 in-enclave HMAC key,
4. read each bank's sealed bundle from the inbox and decrypt it,
5. re-key every identifier into the enclave's identifier space,
6. join the cross-bank graph and compute the cross-bank features,
7. cluster cross-bank rings among the accounts the banks flagged,
8. write one redacted alert bundle per bank to the outbox.

Every run emits a **manifest**: which banks arrived, which were absent, how many
edges survived, and the image digest from the attestation. A thin run and a
complete run otherwise look identical in the outbox, and a bank reading two
alerts has no way to tell that the second one was computed without Bank C.

``--self-check`` runs steps 5 through 8 on a synthetic in-memory graph with a
local key and no Google Cloud at all. It exists so that a Confidential VM boot
can be distinguished from a workload bug: if ``--self-check`` passes on the VM
but the real run fails at step 2, the failure is key release, which is exactly
what the negative test wants to show.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from enclave.bundle import BankContribution
from enclave.config import EnclaveConfig
from enclave.envelope import KeyWrapperError, unseal
from enclave.graph import build_graph, cross_bank_features
from enclave.inbox import read_inbox, write_evidence, write_outbox
from enclave.outbox_crypto import seal_for_bank
from enclave.policy import bundles_for_ring, foreign_identifiers_in
from enclave.rekey import AccountRekeyer
from enclave.rings import find_candidates, to_ring

LOG = logging.getLogger("mulenet.enclave")


#: Terminal outcomes recorded in the evidence object. A run lands on exactly one.
OUTCOME_COMPLETED = "completed"
OUTCOME_KEY_REFUSED = "key_release_refused"
OUTCOME_NO_BUNDLES = "no_bundles_found"
OUTCOME_ERROR = "error"
OUTCOME_SELF_CHECK_PASSED = "self_check_passed"
OUTCOME_SELF_CHECK_FAILED = "self_check_failed"


@dataclass
class RunManifest:
    """What a run did, so a thin result is never read as a complete one.

    Serialised to the evidence bucket on **every** run, including refused ones.
    ``outcome`` is explicit rather than inferred: a reader must never have to
    deduce "refused" from the absence of results, because absence is also what a
    crashed VM and a lost log look like.
    """

    run_id: str
    started_at: str
    outcome: str = OUTCOME_ERROR
    image_digest: str | None = None
    contributing_banks: list[str] = field(default_factory=list)
    absent_banks: list[str] = field(default_factory=list)
    edge_count: int = 0
    account_count: int = 0
    flagged_count: int = 0
    ring_count: int = 0
    bundles_written: list[str] = field(default_factory=list)
    key_release_ok: bool = False
    #: Verbatim backend error when key release was refused. Never reworded or
    #: truncated: for the negative tests this string IS the evidence.
    refusal_message: str | None = None
    #: Which gate refused -- an STS-exchange failure (the provider's attribute
    #: condition rejected the attestation, what a debug image hits) or a KMS
    #: decrypt failure (the key's binding does not name this digest, what a
    #: tampered image hits). They prove different properties.
    refusal_kind: str | None = None
    #: The attestation claims this VM presented, recorded for audit and to settle
    #: the support_attributes shape question. Observation, never enforcement.
    attestation: dict = field(default_factory=dict)
    #: The service account the VM ran as, i.e. the identity that wrote this
    #: evidence object. Distinct from the attested principal that asks KMS for the
    #: key, and recorded so an auditor does not have to assume they are the same.
    wrote_evidence_as: str | None = None
    running_debug_image: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def _flagged_from_env(
    rekeyer: AccountRekeyer, contributions: dict[str, BankContribution]
) -> set[str]:
    """Resolve which accounts each bank flagged.

    Each bank names the accounts *it* flagged, by the hashed id it already uses,
    via ``MULENET_ENCLAVE_FLAGGED_<BANK>`` (comma-separated). A bank can only
    flag accounts it owns: a bank naming another bank's account is ignored, with
    a note, because honouring it would let one bank drag another's customers
    into a ring.

    In a deployment this list would arrive inside the sealed bundle rather than
    as metadata. It is an env var here because the bundle schema lives in
    ``shared/``, which this change does not own -- see ``docs/data_sharing.md``.
    """
    flagged: set[str] = set()
    for bank_id, contribution in contributions.items():
        owned = {a.hashed_account_id for a in contribution.accounts}
        key = f"MULENET_ENCLAVE_FLAGGED_{bank_id.upper().replace('-', '_')}"
        for hashed in (h.strip() for h in os.environ.get(key, "").split(",")):
            if not hashed:
                continue
            if hashed not in owned:
                LOG.warning("%s flagged %s which it does not own; ignoring", bank_id, hashed[:12])
                continue
            flagged.add(rekeyer.rekey_counterparty(bank_id, hashed))
    return flagged


def run(cfg: EnclaveConfig, run_id: str, attempt: str = "run") -> RunManifest:
    """Execute one joint scoring run. Requires a Confidential Space VM.

    Returns a manifest in every case, including when KMS refuses the key. The
    refusal is recorded on the manifest and written to the evidence bucket rather
    than raised, because a refusal is a *result* of this run, not an error in it.
    """
    from google.cloud import storage

    from enclave.kms import (
        KmsKeyWrapper,
        attestation_claims,
        attestation_posture,
        classify_refusal,
        looks_like_debug_image,
    )

    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(),
    )
    claims = attestation_claims()
    manifest.attestation = attestation_posture(claims)
    manifest.image_digest = manifest.attestation.get("image_digest")
    manifest.running_debug_image = looks_like_debug_image(manifest.attestation)
    manifest.wrote_evidence_as = cfg.workload_service_account

    # The GCS client is built BEFORE the gated step, and on purpose. It uses the
    # VM's attached service account via the metadata server, which does not
    # depend on the attestation exchange -- so if key release is refused below,
    # this client still works and the refusal can be written down. Building it
    # after the KMS call would mean a refused run had no way to report itself.
    storage_client = storage.Client(project=cfg.operator_project)

    def record(outcome: str) -> RunManifest:
        """Write the evidence object and return the manifest."""
        manifest.outcome = outcome
        try:
            name = write_evidence(
                storage_client, cfg.evidence_bucket, run_id, attempt, manifest.to_json()
            )
            LOG.info("evidence written to gs://%s/%s", cfg.evidence_bucket, name)
        except Exception as exc:  # noqa: BLE001
            # Losing the evidence must not also destroy the result it described.
            LOG.error("could not write evidence: %s", exc)
            manifest.notes.append(f"evidence write failed: {exc}")
        return manifest

    # Step 2: the gated step. A refusal here is the Stage 1 result, recorded as
    # data with the backend's own message intact.
    try:
        wrapper = KmsKeyWrapper.from_attestation(
            key_name=cfg.kms_key,
            wip_provider_audience=cfg.wip_provider_audience,
        )
        # Step 3: the in-enclave HMAC key, itself only decryptable by this image.
        # Fetching it is what actually exercises key release: building the
        # wrapper above only constructs a credential.
        bucket_name, _, blob_path = (
            cfg.hash_key_ciphertext_uri.removeprefix("gs://").partition("/")
        )
        wrapped_hash_key = (
            storage_client.bucket(bucket_name).blob(blob_path).download_as_bytes()
        )
        # Its own key, not cfg.kms_key: observed live, decrypting it under the
        # edge-wrap key got INDECIPHERABLE_CIPHERTEXT_FOR_CRYPTOKEY.
        hash_key_wrapper = KmsKeyWrapper.from_attestation(
            key_name=cfg.hash_kms_key,
            wip_provider_audience=cfg.wip_provider_audience,
        )
        enclave_hash_key = hash_key_wrapper.decrypt_blob(wrapped_hash_key)
    except KeyWrapperError as exc:
        manifest.refusal_message = str(exc)
        manifest.refusal_kind = classify_refusal(str(exc))
        manifest.notes.append(
            "KMS did not release the key to this image. Nothing was decrypted and "
            "no bank data was read."
        )
        return record(OUTCOME_KEY_REFUSED)
    manifest.key_release_ok = True

    # Step 4: read and decrypt what arrived.
    entries = read_inbox(storage_client, cfg.inbox_bucket, run_id, cfg.bank_ids)
    contributions: dict[str, BankContribution] = {}
    for entry in entries:
        plaintext = unseal(entry.bundle, wrapper)
        contributions[entry.bank_id] = BankContribution.from_bytes(plaintext, entry.bank_id)
    manifest.contributing_banks = sorted(contributions)
    manifest.absent_banks = sorted(set(cfg.bank_ids) - set(contributions))
    if manifest.absent_banks:
        manifest.notes.append(
            f"scored without {', '.join(manifest.absent_banks)}; "
            "cross-bank findings involving those banks cannot appear in this run"
        )
    if not contributions:
        manifest.notes.append("no bundles found; nothing scored")
        return record(OUTCOME_NO_BUNDLES)

    # Steps 5-7.
    rekeyer = AccountRekeyer(enclave_hash_key)
    hints = {
        bank_id: {
            a.hashed_account_id: a.bank_id
            for other, other_contribution in contributions.items()
            if other != bank_id
            for a in other_contribution.accounts
        }
        for bank_id in contributions
    }
    graph = build_graph(contributions, rekeyer, hints)
    manifest.edge_count = len(graph.edges)
    manifest.account_count = len(graph.account_ids)
    features = cross_bank_features(graph)
    LOG.info("computed %d cross-bank feature rows", len(features))

    flagged = _flagged_from_env(rekeyer, contributions)
    manifest.flagged_count = len(flagged)
    candidates = find_candidates(graph, flagged)
    manifest.ring_count = len(candidates)

    # Step 8: release, with the isolation property asserted before writing.
    for candidate in candidates:
        ring = to_ring(candidate)
        per_bank = bundles_for_ring(ring, graph, len(candidate.connector_ids))
        for bank_id, bundle in per_bank.items():
            leaked = foreign_identifiers_in(bundle, graph)
            if leaked:
                # Refuse the whole run rather than publish a leaking bundle. A
                # partial write would already have disclosed it.
                raise RuntimeError(
                    f"output policy violation: {bank_id}'s bundle for {ring.ring_id} "
                    f"contains {len(leaked)} identifier(s) owned by another bank; "
                    "refusing to write any further output"
                )
            # Encrypted to the bank's own key before it ever reaches the
            # operator-owned outbox bucket -- see enclave/outbox_crypto.py.
            # The operator can list and read this object, same as the inbox,
            # but not open it: it holds no bank's private key.
            sealed = seal_for_bank(
                bank_id, bundle.to_json().encode(), cfg.outbox_public_keys[bank_id]
            )
            name = write_outbox(
                storage_client,
                cfg.outbox_bucket,
                bank_id,
                run_id,
                ring.ring_id,
                sealed.to_json(),
            )
            manifest.bundles_written.append(name)
    return record(OUTCOME_COMPLETED)


def self_check() -> int:
    """Run the join, clustering and policy locally, with no Google Cloud.

    Returns a process exit status: ``0`` if the isolation property held.
    """
    from datetime import timedelta
    from decimal import Decimal

    from shared.contracts import Account, Transfer

    key = b"s" * 32
    t0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

    def account(hashed: str, bank: str) -> Account:
        return Account(
            hashed_account_id=hashed,
            bank_id=bank,
            opened_at=t0 - timedelta(days=100),
            account_type="savings",
        )

    def transfer(tid: str, src: str, dst: str, amount: str, minutes: int) -> Transfer:
        return Transfer(
            transfer_id=tid,
            from_hashed_account_id=src,
            to_hashed_account_id=dst,
            amount=Decimal(amount),
            currency="USD",
            timestamp=t0 + timedelta(minutes=minutes),
            channel="neft",
        )

    t1 = transfer("t1", "a1", "b1", "5000.00", 0)
    t2 = transfer("t2", "b1", "c1", "4900.00", 5)
    contributions = {
        "bank-a": BankContribution("bank-a", (account("a1", "bank-a"),), (t1,)),
        "bank-b": BankContribution(
            "bank-b", (account("b1", "bank-b"),), (t1, t2)
        ),
        "bank-c": BankContribution("bank-c", (account("c1", "bank-c"),), (t2,)),
    }
    hints = {
        "bank-a": {"b1": "bank-b"},
        "bank-b": {"a1": "bank-a", "c1": "bank-c"},
        "bank-c": {"b1": "bank-b"},
    }
    rekeyer = AccountRekeyer(key)
    graph = build_graph(contributions, rekeyer, hints)
    flagged = {
        rekeyer.rekey_counterparty("bank-a", "a1"),
        rekeyer.rekey_counterparty("bank-b", "b1"),
        rekeyer.rekey_counterparty("bank-c", "c1"),
    }
    candidates = find_candidates(graph, flagged)
    if len(candidates) != 1 or candidates[0].bank_span != 3:
        print("SELF-CHECK FAIL: expected exactly one 3-bank ring", file=sys.stderr)
        return 1
    ring = to_ring(candidates[0])
    per_bank = bundles_for_ring(ring, graph, len(candidates[0].connector_ids))
    for bank_id, bundle in sorted(per_bank.items()):
        leaked = foreign_identifiers_in(bundle, graph)
        status = "OK" if not leaked else f"LEAKED {sorted(leaked)}"
        print(
            f"  {bank_id}: own_members={len(bundle.own_members)} "
            f"bank_span={bundle.bank_span} edges={len(bundle.edges)} -> {status}"
        )
        if leaked:
            return 1
    print(f"SELF-CHECK PASS: ring {ring.ring_id} spans {len(ring.banks_involved)} banks, "
          f"{len(graph.edges)} joined edges, no cross-bank identifier leaked")
    return 0


def _write_self_check_evidence(
    cfg: EnclaveConfig, run_id: str, attempt: str, status: int
) -> None:
    """Record a self-check result in the evidence bucket.

    Deliberately reports ``key_release_ok`` as ``False``: the self-check does not
    touch KMS, and an evidence object claiming otherwise would let a boot test be
    mistaken for a released key.
    """
    from google.cloud import storage

    from enclave.kms import attestation_claims, attestation_posture, looks_like_debug_image

    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(),
        outcome=(
            OUTCOME_SELF_CHECK_PASSED if status == 0 else OUTCOME_SELF_CHECK_FAILED
        ),
    )
    manifest.attestation = attestation_posture(attestation_claims())
    manifest.image_digest = manifest.attestation.get("image_digest")
    manifest.running_debug_image = looks_like_debug_image(manifest.attestation)
    manifest.wrote_evidence_as = cfg.workload_service_account
    manifest.notes.append(
        "self-check only: the join, clustering and output policy ran locally. "
        "KMS was NOT contacted, so this says nothing about key release."
    )
    try:
        storage_client = storage.Client(project=cfg.operator_project)
        write_evidence(
            storage_client, cfg.evidence_bucket, run_id, attempt, manifest.to_json()
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("could not write self-check evidence: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MuleNet Confidential Space workload")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the join/cluster/policy logic locally with no Google Cloud calls",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("MULENET_ENCLAVE_RUN_ID", "run-1"),
        help="inbox/outbox run id",
    )
    parser.add_argument(
        "--attempt",
        default=os.environ.get("MULENET_ENCLAVE_ATTEMPT", "run"),
        help=(
            "label for this attempt within the run id, used as the evidence "
            "object name. The negative tests use 'production' and 'debug' so each "
            "leaves its own object instead of overwriting the other's."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.self_check:
        status = self_check()
        # On a Confidential VM the self-check must leave an evidence object like
        # any other run, or infra/enclave_run.sh waits for one that never comes
        # and reports a healthy boot as inconclusive. Off a VM there is no
        # config and no bucket, and the printed result is the whole point, so a
        # missing config is not an error here.
        try:
            cfg = EnclaveConfig.from_env()
        except ValueError:
            return status
        _write_self_check_evidence(cfg, args.run_id, args.attempt, status)
        return status

    try:
        cfg = EnclaveConfig.from_env()
    except ValueError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    # A refusal is a RESULT, not an exception: run() records it in the evidence
    # object and returns a manifest. Only a genuinely unexpected failure reaches
    # the handler below, and that is still written down where possible.
    try:
        manifest = run(cfg, args.run_id, args.attempt)
    except KeyWrapperError as exc:
        # Reachable only if a refusal escapes outside the gated block -- for
        # instance while unsealing a bank bundle after the key was released.
        print("KEY RELEASE REFUSED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        # Observed live: this branch printed to stderr and returned, but wrote
        # no evidence object at all -- on the production image, where stdout
        # and stderr are not redirected by design, that made every crash here
        # indistinguishable from a hung or lost VM: infra/enclave_run.sh's
        # wait loop ran its full timeout finding nothing, every time, with no
        # way to tell "crashed in 30 seconds" from "still running". OUTCOME_ERROR
        # already existed as RunManifest's default value, unused, exactly for
        # this case; this is what actually writes it. Best-effort: if even
        # this write fails (no Google Cloud reachable at all, say), the
        # original exception is what a human sees, not a masking one.
        print(f"RUN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            import traceback

            from google.cloud import storage

            storage_client = storage.Client(project=cfg.operator_project)
            crash_manifest = RunManifest(
                run_id=args.run_id,
                started_at=datetime.now(UTC).isoformat(),
                outcome=OUTCOME_ERROR,
                notes=[
                    f"{type(exc).__name__}: {exc}",
                    "".join(traceback.format_exception(exc)),
                ],
            )
            name = write_evidence(
                storage_client,
                cfg.evidence_bucket,
                args.run_id,
                args.attempt,
                crash_manifest.to_json(),
            )
            print(f"crash evidence written to gs://{cfg.evidence_bucket}/{name}", file=sys.stderr)
        except Exception as write_exc:  # noqa: BLE001
            print(f"(could not write crash evidence: {write_exc})", file=sys.stderr)
        return 1

    print(manifest.to_json())

    # The exit code is a convenience for a human at a terminal. It is NOT the
    # evidence: infra/enclave_run.sh and the negative tests read the evidence
    # object's `outcome` field, because an exit status can come from a trailing
    # pipe stage and this project has been misled by exactly that.
    if manifest.outcome == OUTCOME_KEY_REFUSED:
        print("KEY RELEASE REFUSED", file=sys.stderr)
        print(manifest.refusal_message or "(no message)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
