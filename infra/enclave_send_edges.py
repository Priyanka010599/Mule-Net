#!/usr/bin/env python3
"""Seal one bank's cross-bank edges and upload them to the operator's inbox.

This is the **bank side** of the design: it runs at the bank, on the bank's own
data, under the bank's own identity, and the only thing it hands the operator is
ciphertext. For the demo each bank's data is a local slice of the simulator
parquet, which stands in for "raw data that stays at the bank".

Usage (per bank):

    python -m infra.enclave_send_edges \\
        --bank-id bank-a \\
        --bank-project mulenet-bank-a \\
        --run-id run-1

Selection is deliberately narrow. Only transfers with **exactly one endpoint at
this bank** are sent: a transfer inside the bank is nobody else's business, and
the cross-bank features in ``ml/features.py`` do not use it. Devices, KYC
documents, balances and ground truth are never read by this script at all.

``--dry-run`` writes the sealed bundle locally and uploads nothing, so the
selection can be reviewed before any data leaves. It also prints what a bank's
compliance reviewer would want: how many accounts and edges, and confirmation
that no same-bank edge was included.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enclave.bundle import BankContribution  # noqa: E402
from enclave.envelope import seal  # noqa: E402
from enclave.inbox import inbox_object_name  # noqa: E402
from shared.contracts import Account, Transfer  # noqa: E402


def load_contribution(data_dir: Path, bank_id: str) -> BankContribution:
    """Build this bank's contribution from the simulator tables.

    Reads only ``accounts`` and ``transfers``. Raises if a table is missing
    rather than sending a thin bundle: a bank that silently contributed nothing
    would make the joint result quietly incomplete.
    """
    import pandas as pd

    accounts_path = data_dir / "accounts.parquet"
    transfers_path = data_dir / "transfers.parquet"
    for path in (accounts_path, transfers_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Fetch the data first: python infra/fetch_data.py"
            )

    accounts = pd.read_parquet(accounts_path)
    transfers = pd.read_parquet(transfers_path)

    # Two separate bank-id namespaces meet here, and this is the only place
    # they touch. simulator/generator.py labels accounts "BANK_A"/"BANK_B"/
    # "BANK_C" (shared.contracts.Account's own convention, used everywhere in
    # ml/ and the UI). The enclave's own world -- GCP project names, the
    # inbox path, MULENET_ENCLAVE_BANK_IDS, BankContribution.bank_id -- uses
    # lowercase "bank-a"/"bank-b"/"bank-c" throughout. Observed live: without
    # this translation, `own = accounts[accounts["bank_id"] == bank_id]` (the
    # enclave's lowercase form) matched nothing against the data's uppercase
    # labels, and even fixing just the filter would leave BankContribution's
    # own self-check failing the other way (accounts' bank_id would be
    # "BANK_A" while the contribution's own bank_id stayed "bank-a").
    # enclave/workload.py's docstring calls this "re-keying into the
    # enclave's identifier space" for account ids; the bank label needs the
    # same translation at the same boundary, for the same reason: nothing
    # inside a sealed bundle should carry a label from outside the enclave's
    # own namespace.
    data_label = bank_id.upper().replace("-", "_")
    own = accounts[accounts["bank_id"] == data_label]
    if own.empty:
        raise ValueError(
            f"no accounts with bank_id={data_label!r} (from --bank-id {bank_id!r}) "
            f"in {accounts_path}. Known banks: {sorted(accounts['bank_id'].unique())}"
        )
    own_ids = set(own["hashed_account_id"])

    from_mine = transfers["from_hashed_account_id"].isin(own_ids)
    to_mine = transfers["to_hashed_account_id"].isin(own_ids)
    # Exactly one endpoint here: `^` excludes intra-bank transfers, which is the
    # whole point. Using `|` would ship this bank's internal activity.
    cross = transfers[from_mine ^ to_mine]

    endpoints = set(cross["from_hashed_account_id"]) | set(cross["to_hashed_account_id"])
    # Only the accounts that actually appear on a shared edge are declared, not
    # the bank's whole book: an account with no cross-bank activity is not
    # relevant to the joint graph and disclosing it would widen exposure.
    relevant = own[own["hashed_account_id"].isin(endpoints)]

    contribution = BankContribution(
        bank_id=bank_id,
        accounts=tuple(
            # bank_id=bank_id (the enclave's lowercase form), not row.bank_id
            # (the data's "BANK_A"): every account in a sealed bundle must
            # agree with the bundle's own bank_id, which is what
            # BankContribution.validate()'s no-foreign-accounts check and
            # enclave/inbox.py's read_inbox() both compare against.
            Account(
                hashed_account_id=row.hashed_account_id,
                bank_id=bank_id,
                opened_at=row.opened_at,
                account_type=row.account_type,
                status=row.status,
            )
            for row in relevant.itertuples()
        ),
        transfers=tuple(
            Transfer(
                transfer_id=str(row.transfer_id),
                from_hashed_account_id=row.from_hashed_account_id,
                to_hashed_account_id=row.to_hashed_account_id,
                amount=row.amount,
                currency=row.currency,
                timestamp=row.timestamp,
                channel=row.channel,
            )
            for row in cross.itertuples()
        ),
    )
    contribution.validate()
    return contribution


def assert_no_same_bank_edges(contribution: BankContribution) -> None:
    """Fail if any edge has both endpoints at this bank.

    Checked here, at the point of sending, rather than trusted from the selection
    above: this is the last moment the bank controls, and an intra-bank edge that
    slips through is data the bank never agreed to share.
    """
    owned = {a.hashed_account_id for a in contribution.accounts}
    for transfer in contribution.transfers:
        if (
            transfer.from_hashed_account_id in owned
            and transfer.to_hashed_account_id in owned
        ):
            raise ValueError(
                f"transfer {transfer.transfer_id} has both endpoints at "
                f"{contribution.bank_id}; refusing to send an intra-bank edge"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-id", required=True, help="logical bank id, e.g. bank-a")
    parser.add_argument("--bank-project", required=True, help="the bank's GCP project")
    parser.add_argument("--run-id", default="run-1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "simulator" / "out",
        help="where the simulator parquet lives (stands in for the bank's own store)",
    )
    parser.add_argument("--kms-key", default=None, help="enclave edge-wrap key resource")
    parser.add_argument("--inbox-bucket", default=None)
    parser.add_argument(
        "--impersonate",
        default=None,
        help="the bank's sender service account, e.g. mulenet-bank-a-sender@...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="seal locally, upload nothing; prints what would be sent",
    )
    args = parser.parse_args(argv)

    contribution = load_contribution(args.data_dir, args.bank_id)
    assert_no_same_bank_edges(contribution)

    print(f"==> {args.bank_id}: {len(contribution.accounts)} accounts, "
          f"{len(contribution.transfers)} cross-bank edges selected")
    print("    no intra-bank edge included (checked, not assumed)")
    payload = contribution.to_bytes()
    print(f"    plaintext payload: {len(payload)} bytes")

    if args.dry_run:
        out = REPO_ROOT / f".enclave-bundle-{args.bank_id}-{args.run_id}.plaintext.json"
        out.write_bytes(payload)
        print(f"    DRY RUN: wrote plaintext to {out} and uploaded nothing.")
        print("    Review it, then re-run without --dry-run to seal and upload.")
        return 0

    if not args.kms_key or not args.inbox_bucket:
        print(
            "ERROR: --kms-key and --inbox-bucket are required unless --dry-run.\n"
            "       Source infra/enclave_config.sh and pass\n"
            "         --kms-key \"$ENCLAVE_KMS_EDGE_KEY_RESOURCE\"\n"
            "         --inbox-bucket \"$ENCLAVE_INBOX_BUCKET\"",
            file=sys.stderr,
        )
        return 2

    import google.auth
    from google.cloud import storage

    from enclave.kms import KmsKeyWrapper

    operator_credentials, operator_project = google.auth.default()
    credentials = operator_credentials
    if args.impersonate:
        # Impersonation rather than a downloaded key: service-account key
        # creation is blocked by org policy, and a key checked out anywhere near
        # a public repo would be the wrong answer regardless.
        from google.auth import impersonated_credentials

        credentials = impersonated_credentials.Credentials(
            source_credentials=operator_credentials,
            target_principal=args.impersonate,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        # The real cause of a KMS encrypt refusal for an impersonated bank
        # identity is not a quota-project mismatch (checked directly: this
        # credential's quota_project_id is already unset, before and after
        # any attempt to clear it) -- it is that cloudkms.googleapis.com
        # must be enabled on the BANK's own project for KMS to attribute
        # usage to a caller with no quota project set, regardless of where
        # the key itself lives. See infra/enclave_bank_identity.sh step 1.

    wrapper = KmsKeyWrapper(args.kms_key, credentials=credentials)
    bundle = seal(args.bank_id, payload, wrapper)
    print(f"    sealed: DEK wrapped by {args.kms_key}")

    object_name = inbox_object_name(args.bank_id, args.run_id)
    client = storage.Client(project=args.bank_project, credentials=credentials)
    blob = client.bucket(args.inbox_bucket).blob(object_name)
    # Written to a temp file and uploaded from there, not upload_from_string().
    # Observed live: a real bank's base64-encoded ciphertext is well over 100MB
    # as a Python string, and upload_from_string() needs that string held in
    # memory a SECOND time while google-cloud-storage's resumable-upload
    # chunker reads it -- two copies of a >100MB string on a machine with
    # ~160MB free hit MemoryError, once inside json.dumps() itself and once
    # inside the upload's own chunk reader on a later run. upload_from_filename
    # streams from disk in fixed-size chunks instead, so peak memory is the
    # one transient string from bundle.to_json() (freed immediately below),
    # never that string plus a second in-memory read buffer of it.
    #
    # The google-cloud-storage default timeout (60s) is tuned for small
    # objects. Observed live: a real bank's sealed bundle is tens of MB of
    # ciphertext, and the default timed out mid-upload with the seal (the
    # expensive KMS step) already paid for. 600s is generous headroom, not a
    # tuned value -- there is no per-stage size limit to tune it against.
    tmp_path = REPO_ROOT / f".enclave-sealed-{args.bank_id}-{args.run_id}.json"
    tmp_path.write_text(bundle.to_json(), encoding="utf-8")
    try:
        blob.upload_from_filename(
            str(tmp_path), content_type="application/json", timeout=600
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Verified by reading the object back with the OPERATOR's own credentials,
    # not the sender's. The sender identity holds only roles/storage.objectCreator
    # on its own prefix -- create, not get -- by design, so an account that could
    # read what it just wrote could also read every other bank's ciphertext in
    # the same bucket. Observed live: reloading with the impersonated sender
    # credential 403'd with "does not have storage.objects.get access", which
    # is the permission model working as intended, not a failure to work around.
    # The operator's own quota project, not the bank's: this client
    # authenticates as the operator, so billing a bank project for it would
    # be the wrong identity's quota charged for the wrong identity's call.
    operator_client = storage.Client(
        project=operator_project, credentials=operator_credentials
    )
    operator_blob = operator_client.bucket(args.inbox_bucket).blob(object_name)
    if not operator_blob.exists():
        print(f"ERROR: gs://{args.inbox_bucket}/{object_name} is not there after upload",
              file=sys.stderr)
        return 1
    operator_blob.reload()
    blob = operator_blob
    print(f"==> uploaded gs://{args.inbox_bucket}/{object_name} ({blob.size} bytes ciphertext)")
    print("    the operator now holds ciphertext it has no key for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
