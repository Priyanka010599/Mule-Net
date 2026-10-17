"""The Stage 1 negative tests, exercised against a fake KMS.

Real Cloud KMS is not reachable from a test run, so the *logic* of both negative
tests is pinned here: a key holder that models **both gates** the real design
uses, wired to the same :class:`~enclave.envelope.KeyWrapper` protocol the real
:class:`~enclave.kms.KmsKeyWrapper` implements.

The two gates, and why modelling both matters:

1. **The workload identity pool provider's attribute condition** (gate 1).
   Rejects an attestation that is not a production Confidential Space image --
   wrong ``swname``, Secure Boot off, or **debugging enabled**. Failing this
   means no credential is ever minted, so the refusal surfaces from the STS
   exchange.
2. **The KMS key's IAM binding** (gate 2). Names one container image digest, so
   a rebuilt image is a different principal. Failing this means the refusal comes
   from KMS and names the decrypt permission.

Three outcomes are therefore asserted, and they are genuinely different claims:

| attestation | digest pinned? | result | gate |
|---|---|---|---|
| production image | yes | key released | -- |
| production image | **no** (tampered) | refused | KMS |
| **debug image** | yes | refused | STS |

The tampered case proves *the key follows the code*. The debug case proves *the
key requires no operator access* -- a debug-family VM allows SSH, so releasing
bank data to one would hand the operator the plaintext the whole design exists
to withhold. Neither test implies the other.

**Not proved here:** that real KMS and real STS behave this way. That depends on
the provider condition and IAM binding created by ``infra/enclave_setup.sh`` and
``infra/enclave_pin_digest.sh``, and can only be established by a credentialed
run. The fakes mimic the *shape* of each refusal so the assertions made here are
the same assertions the live run makes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from enclave.bundle import BankContribution
from enclave.envelope import (
    KeyWrapperError,
    aad_for,
    seal,
    unseal,
    unseal_is_refused,
)
from enclave.kms import (
    DBGSTAT_PRODUCTION,
    REFUSAL_KMS,
    REFUSAL_STS,
    attestation_posture,
    classify_refusal,
    looks_like_debug_image,
)
from shared.contracts import Account, Transfer

KEY_RESOURCE = (
    "projects/mulenet/locations/asia-south1/keyRings/mulenet-enclave/cryptoKeys/edge-wrap"
)
GOOD_DIGEST = "sha256:" + "a" * 64
TAMPERED_DIGEST = "sha256:" + "b" * 64

#: What the debug image reports for dbgstat. The production image reports
#: "disabled-since-boot" (enclave.kms.DBGSTAT_PRODUCTION).
DBGSTAT_DEBUG = "enable"


@dataclass(frozen=True)
class FakeAttestation:
    """The attestation claims a fake VM presents.

    Defaults describe a **production** Confidential Space VM. Each test changes
    exactly one thing, so a refusal can only be attributed to that change.
    """

    image_digest: str = GOOD_DIGEST
    dbgstat: str = DBGSTAT_PRODUCTION
    support_attributes: tuple[str, ...] = ("LATEST", "STABLE", "USABLE")
    swname: str = "CONFIDENTIAL_SPACE"
    secboot: bool = True
    hwmodel: str = "GCP_AMD_SEV"

    def as_claims(self) -> dict:
        """Render as an attestation token payload, in the real claim layout."""
        return {
            "swname": self.swname,
            "hwmodel": self.hwmodel,
            "secboot": self.secboot,
            "dbgstat": self.dbgstat,
            "submods": {
                "container": {
                    "image_digest": self.image_digest,
                    "image_reference": "example/scorer:v1",
                },
                "confidential_space": {
                    "support_attributes": list(self.support_attributes),
                },
            },
        }


def debug_image_attestation(digest: str = GOOD_DIGEST) -> FakeAttestation:
    """A debug-family VM: debugging on, and no STABLE support attribute.

    Both facts are set because the real debug image exhibits both -- Google's
    image database marks a hardened entry ``[LATEST, STABLE, USABLE]`` and a
    debug entry with no labels at all. A fake that changed only one of them
    would let a condition checking the other pass by accident.
    """
    return FakeAttestation(
        image_digest=digest,
        dbgstat=DBGSTAT_DEBUG,
        support_attributes=(),
    )


class FakeProviderCondition:
    """Models the WIP provider's ``--attribute-condition`` (gate 1).

    Mirrors the expression ``infra/enclave_setup.sh`` installs. It is a fake, but
    it is the same set of checks in the same order, so a test that passes here
    describes the condition the script actually writes.
    """

    def evaluate(self, attestation: FakeAttestation) -> tuple[bool, str]:
        """Return ``(accepted, reason)``. Reason is empty when accepted."""
        if attestation.swname != "CONFIDENTIAL_SPACE":
            return False, f"swname is {attestation.swname!r}, not 'CONFIDENTIAL_SPACE'"
        if attestation.dbgstat != DBGSTAT_PRODUCTION:
            return False, (
                f"dbgstat is {attestation.dbgstat!r}, expected "
                f"{DBGSTAT_PRODUCTION!r} (a debug image allows operator SSH)"
            )
        if "STABLE" not in attestation.support_attributes:
            return False, "support_attributes does not contain 'STABLE'"
        if not attestation.secboot:
            return False, "secboot is false"
        if not attestation.hwmodel.startswith("GCP_"):
            return False, f"hwmodel {attestation.hwmodel!r} is not a GCP platform"
        return True, ""


@dataclass
class FakeAttestedKms:
    """A KMS stand-in gated on BOTH the provider condition and the digest binding.

    Encrypt is deliberately **not** gated, matching the real split: banks hold
    ``cryptoKeyEncrypter`` and can always seal a bundle. Only decrypt is gated.
    """

    allowed_digests: set[str]
    attestation: FakeAttestation = field(default_factory=FakeAttestation)
    condition: FakeProviderCondition = field(default_factory=FakeProviderCondition)
    decrypt_attempts: int = 0
    _store: dict[bytes, tuple[bytes, bytes]] = field(default_factory=dict)

    def wrap(self, dek: bytes, aad: bytes) -> bytes:
        handle = os.urandom(16)
        self._store[handle] = (dek, aad)
        return handle

    def unwrap(self, wrapped_dek: bytes, aad: bytes) -> bytes:
        self.decrypt_attempts += 1

        # Gate 1: the STS exchange. Shaped like a federation failure, because
        # that is where a rejected attestation actually surfaces.
        accepted, reason = self.condition.evaluate(self.attestation)
        if not accepted:
            raise KeyWrapperError(
                "could not build a federated credential from the attestation token: "
                "400 Bad Request from sts.googleapis.com: the attestation does not "
                f"satisfy the workload identity pool provider's attribute condition "
                f"({reason})"
            )

        # Gate 2: the KMS IAM binding, which names one image digest.
        if self.attestation.image_digest not in self.allowed_digests:
            raise KeyWrapperError(
                "403 Permission 'cloudkms.cryptoKeyVersions.useToDecrypt' denied on "
                f"resource '{KEY_RESOURCE}' (or it may not exist)."
            )

        dek, bound_aad = self._store[wrapped_dek]
        if bound_aad != aad:
            raise KeyWrapperError("AAD mismatch")
        return dek

    def clone_with(self, attestation: FakeAttestation) -> FakeAttestedKms:
        """Same key, same policy, same sealed data -- a different running VM.

        Used so a test changes only the attestation. Sharing ``_store`` is the
        point: the bundle really is the same bundle.
        """
        twin = FakeAttestedKms(
            allowed_digests=self.allowed_digests,
            attestation=attestation,
            condition=self.condition,
        )
        twin._store = self._store
        return twin


@pytest.fixture
def cross_bank_payload() -> bytes:
    """A real cross-bank contribution, serialised the way a bank would send it."""
    t0 = datetime(2026, 3, 1, 9, 0)
    contribution = BankContribution(
        bank_id="bank-a",
        accounts=(
            Account(
                hashed_account_id="a-hash-1",
                bank_id="bank-a",
                opened_at=t0 - timedelta(days=90),
                account_type="savings",
            ),
        ),
        transfers=(
            Transfer(
                transfer_id="t1",
                from_hashed_account_id="a-hash-1",
                to_hashed_account_id="b-hash-1",
                amount=Decimal("5000.00"),
                currency="USD",
                timestamp=t0,
                channel="neft",
            ),
        ),
    )
    return contribution.to_bytes()


@pytest.fixture
def production_kms() -> FakeAttestedKms:
    """A production VM whose digest is pinned: the happy path."""
    return FakeAttestedKms(allowed_digests={GOOD_DIGEST})


class TestProductionImageSucceeds:
    """The positive half. Without it, neither refusal proves anything."""

    def test_pinned_production_image_can_decrypt(self, cross_bank_payload, production_kms):
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        assert unseal(bundle, production_kms) == cross_bank_payload

    def test_decrypted_edges_survive_the_round_trip_intact(
        self, cross_bank_payload, production_kms
    ):
        """Amounts must come back exact: a Decimal mangled to float is a bug."""
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        restored = BankContribution.from_bytes(unseal(bundle, production_kms), "bank-a")
        assert restored.transfers[0].amount == Decimal("5000.00")
        assert restored.transfers[0].to_hashed_account_id == "b-hash-1"
        assert restored.accounts[0].bank_id == "bank-a"


class TestTamperedImageIsRefused:
    """Negative test 1: the key follows the code."""

    def test_changed_digest_is_refused_by_kms(self, cross_bank_payload, production_kms):
        bundle = seal("bank-a", cross_bank_payload, production_kms)

        # Same bundle, same key, same policy, still a production image -- only
        # the digest changed, which is what rebuilding one line produces.
        tampered = production_kms.clone_with(FakeAttestation(image_digest=TAMPERED_DIGEST))
        with pytest.raises(KeyWrapperError) as excinfo:
            unseal(bundle, tampered)

        message = str(excinfo.value)
        assert "cloudkms.cryptoKeyVersions.useToDecrypt" in message
        assert KEY_RESOURCE in message
        assert classify_refusal(message) == REFUSAL_KMS

    def test_re_pinning_the_new_digest_restores_access(self, cross_bank_payload):
        """Shows the refusal was about the digest and nothing incidental."""
        kms = FakeAttestedKms(allowed_digests={GOOD_DIGEST})
        bundle = seal("bank-a", cross_bank_payload, kms)
        repinned = FakeAttestedKms(
            allowed_digests={GOOD_DIGEST, TAMPERED_DIGEST},
            attestation=FakeAttestation(image_digest=TAMPERED_DIGEST),
        )
        repinned._store = kms._store
        assert unseal(bundle, repinned) == cross_bank_payload


class TestDebugImageIsRefused:
    """Negative test 2: the key requires no operator access.

    The image is byte-identical to the attested one and its digest **is** pinned
    in KMS. The only difference is that it boots the debug Confidential Space
    family, which allows the operator to SSH in and read memory. Releasing bank
    data to such a VM would hand the operator the plaintext the entire design
    exists to withhold, so it must be refused even though the code is correct.
    """

    def test_debug_image_with_pinned_digest_is_still_refused(
        self, cross_bank_payload, production_kms
    ):
        bundle = seal("bank-a", cross_bank_payload, production_kms)

        debug = production_kms.clone_with(debug_image_attestation(GOOD_DIGEST))
        # The digest is pinned -- this is not a digest failure.
        assert debug.attestation.image_digest in debug.allowed_digests

        with pytest.raises(KeyWrapperError) as excinfo:
            unseal(bundle, debug)
        message = str(excinfo.value)
        assert "dbgstat" in message
        assert DBGSTAT_PRODUCTION in message
        assert classify_refusal(message) == REFUSAL_STS

    def test_debug_refusal_is_a_different_gate_from_the_tampered_one(
        self, cross_bank_payload, production_kms
    ):
        """The two negative tests must not collapse into one finding.

        If both refusals classified the same way, one test would be redundant and
        the report would be claiming two properties on the evidence of one.
        """
        bundle = seal("bank-a", cross_bank_payload, production_kms)

        _, debug_message = unseal_is_refused(
            bundle, production_kms.clone_with(debug_image_attestation(GOOD_DIGEST))
        )
        _, tampered_message = unseal_is_refused(
            bundle, production_kms.clone_with(FakeAttestation(image_digest=TAMPERED_DIGEST))
        )
        assert classify_refusal(debug_message) == REFUSAL_STS
        assert classify_refusal(tampered_message) == REFUSAL_KMS
        assert classify_refusal(debug_message) != classify_refusal(tampered_message)

    def test_debug_image_yields_no_plaintext(self, cross_bank_payload, production_kms):
        """Fail closed: a debug VM must not end up holding bank data."""
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        debug = production_kms.clone_with(debug_image_attestation(GOOD_DIGEST))
        result = None
        try:
            result = unseal(bundle, debug)
        except KeyWrapperError:
            pass
        assert result is None

    def test_the_key_was_actually_asked_for(self, cross_bank_payload, production_kms):
        """Guards against a vacuous pass: the refusal must follow a real attempt."""
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        debug = production_kms.clone_with(debug_image_attestation(GOOD_DIGEST))
        unseal_is_refused(bundle, debug)
        assert debug.decrypt_attempts == 1

    def test_stable_attribute_alone_also_refuses(self, cross_bank_payload, production_kms):
        """Defence in depth: either check on its own must be enough.

        dbgstat is the better-documented gate, but if a future image reported an
        unexpected dbgstat value, the missing STABLE attribute must still refuse.
        """
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        no_stable = production_kms.clone_with(
            FakeAttestation(image_digest=GOOD_DIGEST, support_attributes=("USABLE",))
        )
        refused, message = unseal_is_refused(bundle, no_stable)
        assert refused is True
        assert "STABLE" in message


class TestSecureBootAndPlatform:
    """The remaining condition terms, so none of them is decorative."""

    def test_secure_boot_off_is_refused(self, cross_bank_payload, production_kms):
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        insecure = production_kms.clone_with(FakeAttestation(secboot=False))
        refused, message = unseal_is_refused(bundle, insecure)
        assert refused is True
        assert "secboot" in message

    def test_non_confidential_space_is_refused(self, cross_bank_payload, production_kms):
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        impostor = production_kms.clone_with(FakeAttestation(swname="COS"))
        refused, message = unseal_is_refused(bundle, impostor)
        assert refused is True
        assert "swname" in message


class TestAttestationPosture:
    """The claim reader, including the shape ambiguity it exists to settle."""

    def test_production_posture_is_read_correctly(self):
        posture = attestation_posture(FakeAttestation().as_claims())
        assert posture["dbgstat"] == DBGSTAT_PRODUCTION
        assert posture["debugging_disabled_since_boot"] is True
        assert posture["stable_support_attribute_present"] is True
        assert posture["image_digest"] == GOOD_DIGEST
        assert looks_like_debug_image(posture) is False

    def test_debug_posture_is_read_correctly(self):
        posture = attestation_posture(debug_image_attestation().as_claims())
        assert posture["debugging_disabled_since_boot"] is False
        assert posture["stable_support_attribute_present"] is False
        assert looks_like_debug_image(posture) is True

    def test_concatenated_support_attributes_are_understood(self):
        """The docs describe support_attributes as 'LATEST=STABLE=USABLE' too.

        The shape could not be verified offline, so both are handled and the raw
        value plus its type go into the evidence object for the first real run to
        settle. This test pins the string form specifically.
        """
        claims = FakeAttestation().as_claims()
        claims["submods"]["confidential_space"]["support_attributes"] = "LATEST=STABLE=USABLE"
        posture = attestation_posture(claims)
        assert posture["stable_support_attribute_present"] is True
        assert posture["support_attributes_type"] == "str"

    def test_concatenated_form_without_stable_is_detected(self):
        claims = FakeAttestation().as_claims()
        claims["submods"]["confidential_space"]["support_attributes"] = "USABLE"
        posture = attestation_posture(claims)
        assert posture["stable_support_attribute_present"] is False

    def test_substring_match_does_not_produce_a_false_stable(self):
        """'UNSTABLE' must not satisfy a check for 'STABLE'.

        A naive substring test would pass here and silently accept an image the
        policy meant to exclude.
        """
        claims = FakeAttestation().as_claims()
        claims["submods"]["confidential_space"]["support_attributes"] = "UNSTABLE"
        posture = attestation_posture(claims)
        assert posture["stable_support_attribute_present"] is False

    def test_dict_support_attributes_form_is_understood(self):
        """Some examples show `support_attributes.stable == true`."""
        claims = FakeAttestation().as_claims()
        claims["submods"]["confidential_space"]["support_attributes"] = {"stable": True}
        posture = attestation_posture(claims)
        assert posture["stable_support_attribute_present"] is True

    def test_absent_claims_are_unknown_not_false(self):
        """'We could not tell' must not read as 'it is production'."""
        posture = attestation_posture({})
        assert posture["debugging_disabled_since_boot"] is None
        assert posture["stable_support_attribute_present"] is None
        assert looks_like_debug_image(posture) is None

    def test_unclassifiable_refusal_is_not_guessed(self):
        """A mislabelled refusal would let one test claim the other's property."""
        assert classify_refusal("something went wrong") not in (REFUSAL_KMS, REFUSAL_STS)


class TestOperatorCannotRead:
    """The operator holds ciphertext and no decrypt right."""

    def test_operator_outside_confidential_space_is_refused(
        self, cross_bank_payload, production_kms
    ):
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        operator = production_kms.clone_with(
            FakeAttestation(swname="NONE", dbgstat="unknown", support_attributes=())
        )
        refused, _ = unseal_is_refused(bundle, operator)
        assert refused is True

    def test_what_the_operator_can_see_is_only_ciphertext(
        self, cross_bank_payload, production_kms
    ):
        """The inbox object must not reveal account ids or amounts."""
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        on_the_wire = bundle.to_json()
        for secret in (b"a-hash-1", b"b-hash-1", b"5000.00"):
            assert secret not in bundle.ciphertext
            assert secret.decode() not in on_the_wire

    def test_bank_id_is_visible_metadata_and_that_is_acknowledged(
        self, cross_bank_payload, production_kms
    ):
        """The operator does learn which bank uploaded, by design.

        Pinned as a test so the limitation cannot quietly stop being true, or
        quietly start being denied: it is documented in docs/data_sharing.md as
        traffic metadata the inbox model exposes.
        """
        bundle = seal("bank-a", cross_bank_payload, production_kms)
        assert "bank-a" in bundle.to_json()
        assert aad_for("bank-a") == aad_for(bundle.bank_id)
