"""Tests for the offline image embedder used in photo reuse detection.

The mock embedder is a Pillow difference hash. Its one load-bearing
property is that two copies of the same file hash identically -- that is
how a recycled identity document is found -- while two different
documents do not. Both are asserted here, on the fixture corpus and on
the real seed-42 images.

The Vertex multimodal path is not exercised: no credentials exist here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from docai.config import DocaiConfig
from docai.conftest import TinyCorpus
from docai.embeddings import (
    HAMMING,
    DifferenceHashEmbedder,
    bits_to_hex,
    build_embedder,
    difference_hash_bits,
    embed_refs,
    find_near_duplicates,
)


def test_build_embedder_returns_the_offline_backend_in_mock_mode(
    tiny_corpus: TinyCorpus,
) -> None:
    embedder = build_embedder(tiny_corpus.config)
    assert isinstance(embedder, DifferenceHashEmbedder)
    assert embedder.backend == "dhash"
    assert embedder.metric == HAMMING


def test_identical_files_hash_identically(tiny_corpus: TinyCorpus, tmp_path: Path) -> None:
    """A copied file is a different path with the same bytes: same hash."""
    source = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_x", "id_card")]
    copy = tmp_path / "copied_under_another_name.png"
    shutil.copyfile(source, copy)

    size = tiny_corpus.config.hash_size
    assert difference_hash_bits(source, size) == difference_hash_bits(copy, size)
    assert bits_to_hex(difference_hash_bits(source, size)) == bits_to_hex(
        difference_hash_bits(copy, size)
    )


def test_different_images_do_not_hash_identically(tiny_corpus: TinyCorpus) -> None:
    size = tiny_corpus.config.hash_size
    left = difference_hash_bits(
        tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_x", "id_card")], size
    )
    right = difference_hash_bits(
        tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_y", "id_card")], size
    )
    assert left != right


def test_hashing_is_deterministic_across_calls(tiny_corpus: TinyCorpus) -> None:
    """No randomness, no timestamps: the same file always gives the same bits."""
    path = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_z", "address_proof")]
    size = tiny_corpus.config.hash_size
    assert difference_hash_bits(path, size) == difference_hash_bits(path, size)


def test_hash_width_matches_the_configured_grid(tiny_corpus: TinyCorpus) -> None:
    path = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_x", "id_card")]
    assert len(difference_hash_bits(path, 8)) == 64
    assert len(difference_hash_bits(path, 32)) == 1024


def test_a_degenerate_grid_is_rejected(tiny_corpus: TinyCorpus) -> None:
    path = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_x", "id_card")]
    with pytest.raises(ValueError, match="hash_size"):
        difference_hash_bits(path, 1)


def test_embed_refs_embeds_each_ref_once_and_skips_missing_files(
    tiny_corpus: TinyCorpus,
) -> None:
    embedder = build_embedder(tiny_corpus.config)
    refs = [*(ref for ref in tiny_corpus.refs.values()), "kyc_images/absent.png"]
    # Deliberately pass one ref twice: distinct refs are embedded once.
    refs.append(next(iter(tiny_corpus.refs.values())))

    embeddings = embed_refs(refs, embedder, tiny_corpus.config.images_dir)

    assert len(embeddings) == len(set(tiny_corpus.refs.values()))
    assert all(e.backend == "dhash" for e in embeddings)
    assert "kyc_images/absent.png" not in {e.ref for e in embeddings}


def test_near_duplicate_search_pairs_copies_and_not_distinct_images(
    tiny_corpus: TinyCorpus, tmp_path: Path
) -> None:
    """At threshold 1.0 only byte-identical images pair up."""
    images_dir = tmp_path
    original = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_x", "id_card")]
    other = tiny_corpus.config.images_dir / tiny_corpus.refs[("ident_y", "id_card")]
    shutil.copyfile(original, images_dir / "a.png")
    shutil.copyfile(original, images_dir / "b.png")
    shutil.copyfile(other, images_dir / "c.png")

    embedder = DifferenceHashEmbedder(tiny_corpus.config.hash_size)
    embeddings = embed_refs(["a.png", "b.png", "c.png"], embedder, images_dir)
    matches = find_near_duplicates(embeddings, 1.0, HAMMING)

    assert [(m.left_ref, m.right_ref) for m in matches] == [("a.png", "b.png")]
    assert matches[0].similarity == pytest.approx(1.0)


def test_near_duplicate_search_needs_at_least_two_embeddings(
    tiny_corpus: TinyCorpus,
) -> None:
    embedder = build_embedder(tiny_corpus.config)
    one = embed_refs(
        [tiny_corpus.refs[("ident_x", "id_card")]], embedder, tiny_corpus.config.images_dir
    )
    assert find_near_duplicates(one, 1.0, HAMMING) == []


def test_an_unknown_metric_is_rejected(tiny_corpus: TinyCorpus) -> None:
    embedder = build_embedder(tiny_corpus.config)
    embeddings = embed_refs(
        list(tiny_corpus.refs.values()), embedder, tiny_corpus.config.images_dir
    )
    with pytest.raises(ValueError, match="metric"):
        find_near_duplicates(embeddings, 1.0, "euclidean")


def test_seed42_reused_identity_shares_one_file_on_disk(seed42_config: DocaiConfig) -> None:
    """The real corpus reuses documents by pointing at the same file.

    Two accounts on a recycled identity carry the *same*
    ``photo_embedding_ref``, so the file they resolve to is literally one
    file. Hashing it under two names must agree exactly, and a different
    identity's card must not collide with it.
    """
    frame = pd.read_parquet(
        seed42_config.documents_path,
        columns=["hashed_account_id", "doc_type", "photo_embedding_ref"],
    )
    cards = frame[(frame["doc_type"] == "id_card") & frame["photo_embedding_ref"].notna()]
    per_ref = cards.groupby("photo_embedding_ref")["hashed_account_id"].nunique()

    shared_refs = sorted(per_ref[per_ref >= 2].index)
    solo_refs = sorted(per_ref[per_ref == 1].index)
    assert shared_refs, "expected at least one image reused by two accounts"
    assert solo_refs

    size = seed42_config.hash_size
    shared_path = seed42_config.images_dir / shared_refs[0]
    solo_path = seed42_config.images_dir / solo_refs[0]
    if not shared_path.is_file() or not solo_path.is_file():
        pytest.skip("seed-42 KYC images were not rendered")

    assert difference_hash_bits(shared_path, size) == difference_hash_bits(shared_path, size)
    assert difference_hash_bits(shared_path, size) != difference_hash_bits(solo_path, size)


def test_seed42_distinct_cards_do_not_collide(seed42_config: DocaiConfig) -> None:
    """A sample of real cards hashes to distinct values at the default grid.

    This is what justifies ``DEFAULT_HASH_SIZE = 32``: the placeholder
    cards share a layout, so a coarser hash cannot separate them.
    """
    images = sorted((seed42_config.images_dir / "kyc_images").glob("*_id_card.png"))[:200]
    if len(images) < 2:
        pytest.skip("seed-42 KYC images were not rendered")

    hashes = {difference_hash_bits(path, seed42_config.hash_size) for path in images}
    assert len(hashes) == len(images)
