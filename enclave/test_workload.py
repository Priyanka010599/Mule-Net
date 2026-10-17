"""Tests for the workload entrypoint's non-cloud paths.

The cloud path (:func:`enclave.workload.run`) is not covered here: it needs a
Confidential Space VM and a released key, and a mock of it would test the mock.
What is covered is everything that must work *before* the credentialed run is
worth attempting -- so a failure on the VM can be attributed to key release
rather than to this code.
"""

from __future__ import annotations

import pytest

from enclave.config import EnclaveConfig
from enclave.workload import main, self_check


def test_self_check_passes():
    """The 3-bank join, clustering and output policy, end to end, no cloud."""
    assert self_check() == 0


def test_self_check_via_cli(capsys):
    assert main(["--self-check"]) == 0
    out = capsys.readouterr().out
    assert "SELF-CHECK PASS" in out
    assert "spans 3 banks" in out


def test_missing_config_exits_two_not_crash(monkeypatch, capsys):
    """A missing variable must fail closed with a clear message, not traceback."""
    monkeypatch.delenv("MULENET_ENCLAVE_BANK_IDS", raising=False)
    assert main(["--run-id", "run-1"]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err


class TestConfig:
    def test_bank_ids_are_required(self, monkeypatch):
        monkeypatch.delenv("MULENET_ENCLAVE_BANK_IDS", raising=False)
        with pytest.raises(ValueError, match="MULENET_ENCLAVE_BANK_IDS"):
            EnclaveConfig.from_env()

    def test_missing_project_is_reported_with_the_metadata_hint(self, monkeypatch):
        """The error must say how to pass it on a Confidential VM."""
        monkeypatch.setenv("MULENET_ENCLAVE_BANK_IDS", "bank-a,bank-b")
        monkeypatch.delenv("MULENET_ENCLAVE_PROJECT", raising=False)
        with pytest.raises(ValueError, match="tee-env-MULENET_ENCLAVE_PROJECT"):
            EnclaveConfig.from_env()

    def test_full_config_resolves(self, monkeypatch):
        import base64

        env = {
            "MULENET_ENCLAVE_BANK_IDS": "bank-a, bank-b ,bank-c",
            "MULENET_ENCLAVE_PROJECT": "mulenet",
            "MULENET_ENCLAVE_INBOX_BUCKET": "mulenet-enclave-inbox",
            "MULENET_ENCLAVE_OUTBOX_BUCKET": "mulenet-enclave-outbox",
            "MULENET_ENCLAVE_EVIDENCE_BUCKET": "mulenet-enclave-evidence",
            "MULENET_ENCLAVE_KMS_KEY": (
                "projects/mulenet/locations/asia-south1/keyRings/kr/cryptoKeys/k"
            ),
            "MULENET_ENCLAVE_HASH_KMS_KEY": (
                "projects/mulenet/locations/asia-south1/keyRings/kr/cryptoKeys/h"
            ),
            "MULENET_ENCLAVE_HASH_KEY_URI": "gs://mulenet-enclave-keys/hash.bin",
            "MULENET_ENCLAVE_WIP_AUDIENCE": "//iam.googleapis.com/projects/1/x",
            "MULENET_ENCLAVE_WORKLOAD_SA": "wl@mulenet.iam.gserviceaccount.com",
            "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A": base64.b64encode(b"a" * 32).decode(),
            "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_B": base64.b64encode(b"b" * 32).decode(),
            "MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_C": base64.b64encode(b"c" * 32).decode(),
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        cfg = EnclaveConfig.from_env()
        assert cfg.bank_ids == ("bank-a", "bank-b", "bank-c")
        assert cfg.operator_project == "mulenet"
        assert cfg.evidence_bucket == "mulenet-enclave-evidence"
        assert cfg.outbox_public_keys == {
            "bank-a": b"a" * 32,
            "bank-b": b"b" * 32,
            "bank-c": b"c" * 32,
        }

    def test_missing_outbox_pubkey_is_reported_with_the_metadata_hint(self, monkeypatch):
        monkeypatch.setenv("MULENET_ENCLAVE_BANK_IDS", "bank-a")
        monkeypatch.setenv("MULENET_ENCLAVE_PROJECT", "mulenet")
        monkeypatch.setenv("MULENET_ENCLAVE_INBOX_BUCKET", "inbox")
        monkeypatch.setenv("MULENET_ENCLAVE_OUTBOX_BUCKET", "outbox")
        monkeypatch.setenv("MULENET_ENCLAVE_EVIDENCE_BUCKET", "evidence")
        monkeypatch.setenv("MULENET_ENCLAVE_KMS_KEY", "k")
        monkeypatch.setenv("MULENET_ENCLAVE_HASH_KMS_KEY", "h")
        monkeypatch.setenv("MULENET_ENCLAVE_HASH_KEY_URI", "gs://b/hash.bin")
        monkeypatch.setenv("MULENET_ENCLAVE_WIP_AUDIENCE", "//iam.googleapis.com/x")
        monkeypatch.setenv("MULENET_ENCLAVE_WORKLOAD_SA", "wl@mulenet.iam.gserviceaccount.com")
        monkeypatch.delenv("MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A", raising=False)
        with pytest.raises(ValueError, match="tee-env-MULENET_ENCLAVE_OUTBOX_PUBKEY_BANK_A"):
            EnclaveConfig.from_env()
