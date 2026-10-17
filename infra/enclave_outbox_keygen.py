#!/usr/bin/env python3
"""Generate one bank's outbox keypair. Run once, offline, by the bank.

This is the **bank side** of per-bank outbox encryption (see
``enclave/outbox_crypto.py`` and ``docs/data_sharing.md``): the private key
never leaves the machine this script runs on and is never uploaded anywhere.
Only the public key is handed to the enclave operator, who passes it to the
enclave as ``MULENET_ENCLAVE_OUTBOX_PUBKEY_<BANK>`` instance metadata -- safe
to pass that way because a public key is not a secret.

Usage:

    python -m infra.enclave_outbox_keygen --bank-id bank-a

Writes the private key to ``.enclave-outbox-key-<bank-id>.bin`` in the repo
root (gitignored; never commit it) and prints the base64 public key to stdout
for the operator to set as the matching env var.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enclave.outbox_crypto import generate_keypair  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-id", required=True, help="e.g. bank-a")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT),
        help="Where to write the private key file (default: repo root).",
    )
    args = parser.parse_args(argv)

    private_key, public_key = generate_keypair()
    key_path = Path(args.out_dir) / f".enclave-outbox-key-{args.bank_id}.bin"
    if key_path.exists():
        print(
            f"REFUSING to overwrite an existing key at {key_path}. "
            "Delete it first if you really mean to rotate.",
            file=sys.stderr,
        )
        return 1
    key_path.write_bytes(private_key)

    env_name = f"MULENET_ENCLAVE_OUTBOX_PUBKEY_{args.bank_id.upper().replace('-', '_')}"
    public_b64 = base64.b64encode(public_key).decode("ascii")
    print(f"private key written to {key_path} -- keep it at the bank, never upload it")
    print(f"public key (give this to the enclave operator):")
    print(f"  {env_name}={public_b64}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
