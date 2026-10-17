"""Shared fixtures for the ``docai/`` tests.

Everything here runs offline. An autouse fixture pins mock mode and
strips any ambient Google Cloud credentials for the whole session, so no
test can reach Document AI, Vertex AI embeddings or Vertex AI Search
even on a machine where credentials happen to be configured.

Two corpora are available:

``tiny_corpus``
    A five-account, three-bank fixture built from scratch in a temp
    directory, with real PNGs on disk. Fast, and its expected pairs are
    obvious by construction.
``seed42_*``
    The real 60,000-row corpus at ``simulator/out/``. Skipped when it
    has not been generated. The cohort assertions need it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from docai.config import DocaiConfig
from docai.parsing import parse_kyc_documents
from docai.reuse import load_bank_map
from shared.contracts import KycDocument

DOC_TYPES: tuple[str, ...] = ("id_card", "address_proof")

#: Environment variables that could pull a test onto a cloud path.
_CLOUD_ENV: tuple[str, ...] = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "MULENET_DOCAI_PROCESSOR_ID",
    "MULENET_REGULATIONS_DATASTORE",
    "MULENET_DATA_DIR",
    "MULENET_DOCAI_OUT_DIR",
    "MULENET_REGULATIONS_DIR",
    "MULENET_DOCAI_HASH_SIZE",
    "MULENET_PHOTO_SIMILARITY_THRESHOLD",
)


@pytest.fixture(scope="session", autouse=True)
def _force_mock_mode() -> Iterator[None]:
    """Pin mock mode for the whole session and hide cloud configuration.

    Session-scoped and autouse because the seed-42 fixtures below are
    session-scoped too; ``monkeypatch`` is function-scoped and could not
    cover them.
    """
    saved = {name: os.environ.get(name) for name in (*_CLOUD_ENV, "MULENET_DOCAI_MODE")}
    for name in _CLOUD_ENV:
        os.environ.pop(name, None)
    os.environ["MULENET_DOCAI_MODE"] = "mock"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class TinyCorpus:
    """A hand-built KYC corpus with known reuse, for fast unit tests."""

    config: DocaiConfig
    #: hashed account id -> bank id, as the submissions parquet records it.
    banks: dict[str, str]
    #: The two accounts sharing an identity across banks.
    cross_bank_accounts: tuple[str, str]
    #: The two accounts sharing an identity inside one bank.
    same_bank_accounts: tuple[str, str]
    #: The account that reuses nothing.
    unique_account: str
    #: Image refs, relative to the data dir, keyed by (identity, doc_type).
    refs: dict[tuple[str, str], str]


def _render(path: Path, label: str, colour: tuple[int, int, int]) -> None:
    """Draw a small placeholder document image.

    Deliberately crude and deliberately fake, like the simulator's: a
    flat colour block with drawn text. No likeness, no real identifiers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 100), colour)
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 151, 91], outline=(255, 255, 255), width=2)
    draw.text((16, 20), "SYNTHETIC TEST DOC", fill=(255, 255, 255))
    draw.text((16, 44), label, fill=(255, 255, 255))
    image.save(path, format="PNG")


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory: pytest.TempPathFactory) -> TinyCorpus:
    """Build a five-account corpus with one cross-bank and one same-bank reuse.

    Accounts ``acct_0``/``acct_1`` share identity ``ident_x`` at BANK_A and
    BANK_B (cross-bank). ``acct_2``/``acct_3`` share ``ident_y``, both at
    BANK_A (same-bank). ``acct_4`` has its own identity at BANK_C.
    Reused identities point at *the same file on disk*, exactly as the
    simulator models a recycled document.
    """
    data_dir = tmp_path_factory.mktemp("tiny_corpus")
    accounts = [f"acct_{i}" for i in range(5)]
    banks = {
        "acct_0": "BANK_A",
        "acct_1": "BANK_B",
        "acct_2": "BANK_A",
        "acct_3": "BANK_A",
        "acct_4": "BANK_C",
    }
    identity_of = {
        "acct_0": "ident_x",
        "acct_1": "ident_x",
        "acct_2": "ident_y",
        "acct_3": "ident_y",
        "acct_4": "ident_z",
    }
    colours = {
        "ident_x": (36, 78, 140),
        "ident_y": (140, 42, 58),
        "ident_z": (32, 110, 88),
    }

    refs: dict[tuple[str, str], str] = {}
    for identity, colour in colours.items():
        for doc_type in DOC_TYPES:
            ref = f"kyc_images/{identity}_{doc_type}.png"
            refs[(identity, doc_type)] = ref
            _render(data_dir / ref, f"{identity} {doc_type}", colour)

    doc_rows = []
    submission_rows = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for index, account in enumerate(accounts):
        identity = identity_of[account]
        doc_ids = []
        for doc_type in DOC_TYPES:
            doc_id = f"doc_{account}_{doc_type}"
            doc_ids.append(doc_id)
            doc_rows.append(
                {
                    "doc_id": doc_id,
                    "hashed_account_id": account,
                    "doc_type": doc_type,
                    "hashed_id_number": f"idnum_{identity}",
                    "normalized_address": f"1 {identity.upper()} ST, TESTVILLE ZZ-0001",
                    "photo_embedding_ref": refs[(identity, doc_type)],
                }
            )
        submission_rows.append(
            {
                "submission_id": f"sub_{account}",
                "hashed_account_id": account,
                "bank_id": banks[account],
                "submitted_at": start + timedelta(days=index),
                "doc_ids": doc_ids,
            }
        )

    pd.DataFrame(doc_rows).to_parquet(data_dir / "kyc_documents.parquet", index=False)
    pd.DataFrame(submission_rows).to_parquet(
        data_dir / "kyc_submissions.parquet", index=False
    )

    config = replace(DocaiConfig.from_env(), data_dir=data_dir, out_dir=data_dir / "out")
    return TinyCorpus(
        config=config,
        banks=banks,
        cross_bank_accounts=("acct_0", "acct_1"),
        same_bank_accounts=("acct_2", "acct_3"),
        unique_account="acct_4",
        refs=refs,
    )


@pytest.fixture(scope="session")
def seed42_config() -> DocaiConfig:
    """Config pointing at the real ``simulator/out/`` corpus."""
    config = DocaiConfig.from_env()
    if not config.documents_path.is_file() or not config.submissions_path.is_file():
        pytest.skip(
            f"seed-42 corpus not generated at {config.data_dir}; "
            "run simulator/generator.py first"
        )
    return config


@pytest.fixture(scope="session")
def seed42_documents(seed42_config: DocaiConfig) -> list[KycDocument]:
    """All 60,000 seed-42 documents, parsed once for the whole session."""
    return parse_kyc_documents(seed42_config)


@pytest.fixture(scope="session")
def seed42_banks(seed42_config: DocaiConfig) -> dict[str, str]:
    """Hashed account id -> bank id for the seed-42 corpus."""
    return load_bank_map(seed42_config)


@pytest.fixture(scope="session")
def ground_truth(seed42_config: DocaiConfig) -> pd.DataFrame:
    """Seed-42 labels: cohort, is_mule, ring id, pattern and ring bank span."""
    path = seed42_config.data_dir / "ground_truth.parquet"
    if not path.is_file():
        pytest.skip(f"ground truth not found at {path}")
    return pd.read_parquet(path)
