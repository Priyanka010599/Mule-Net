"""``docs/claims.md`` must never drift from the metrics files.

This is the test the session's four transcription errors would each have
failed. It re-resolves every published claim from its source and compares
the value, so the file cannot rot in either direction: not if someone
edits a number in the markdown, and not if a re-run moves a number in a
metrics file.

It never skips silently. If the metrics files are absent it uses the
``require_artefact`` convention, so ``MULENET_REQUIRE_OOF=1`` turns a
missing artefact into a failure and CI cannot pass vacuously.
"""

from __future__ import annotations

import pytest

from ml.claims import (
    CLAIMS,
    OUTPUT,
    RETIRED,
    SOURCES,
    build_payload,
    load_sources,
    parse_payload,
    render_markdown,
    resolve,
)
from ml.config import MlConfig
from ml.conftest import require_artefact


@pytest.fixture(scope="module")
def cfg() -> MlConfig:
    return MlConfig.from_env()


@pytest.fixture(scope="module")
def docs(cfg: MlConfig) -> dict:
    """The metrics files, or a loud failure under MULENET_REQUIRE_OOF=1."""
    for relative in SOURCES.values():
        require_artefact(
            cfg.ml_dir.parent / relative,
            "python -m ml.early && python -m ml.leadtime",
        )
    return load_sources(cfg)


@pytest.fixture(scope="module")
def published(cfg: MlConfig) -> dict:
    path = cfg.ml_dir.parent / OUTPUT
    require_artefact(path, "python -m ml.claims")
    return parse_payload(path.read_text("utf-8"))


def _all_claims() -> list:
    return [*CLAIMS, *(c for item in RETIRED for c in item.evidence)]


def test_claim_ids_are_unique() -> None:
    """An id is how a claim is cited; two claims sharing one is ambiguous."""
    ids = [claim.id for claim in _all_claims()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("claim", _all_claims(), ids=lambda c: c.id)
def test_every_claim_still_resolves_to_its_published_value(
    claim, docs: dict, published: dict
) -> None:
    """The whole point: published value == value re-derived from source.

    Compared at the claim's own published precision. A path that no longer
    resolves raises out of ``resolve`` rather than degrading to a default,
    so a renamed key fails here instead of silently reading as zero.
    """
    records = {
        r["id"]: r
        for r in [*published["claims"], *(e for i in published["retired"] for e in i["evidence"])]
    }
    assert claim.id in records, f"{claim.id} is missing from {OUTPUT}; regenerate it"
    record = records[claim.id]
    assert record["value"] == pytest.approx(
        round(claim.value(docs), claim.decimals), abs=10**-claim.decimals / 2
    ), (
        f"{claim.id} has drifted: {OUTPUT} says {record['value']}, "
        f"{record['source_file']}:{record['path']} now gives {claim.value(docs)}"
    )


@pytest.mark.parametrize("claim", _all_claims(), ids=lambda c: c.id)
def test_every_published_path_is_the_one_the_code_uses(claim, published: dict) -> None:
    """Provenance must be accurate, not merely present."""
    records = {
        r["id"]: r
        for r in [*published["claims"], *(e for i in published["retired"] for e in i["evidence"])]
    }
    record = records[claim.id]
    assert record["path"] == claim.path
    assert record["source_file"] == SOURCES[claim.source]
    if claim.minus is not None:
        assert record["minus_path"] == claim.minus


def test_published_file_is_exactly_what_the_generator_emits(cfg: MlConfig, docs: dict) -> None:
    """Catches a hand edit anywhere in the file, prose included.

    The generated block is what the other tests verify, but the rendered
    table beside it is also derived, and a number edited only there would
    otherwise pass while misleading every human reader.
    """
    path = cfg.ml_dir.parent / OUTPUT
    require_artefact(path, "python -m ml.claims")
    expected = render_markdown(build_payload(docs))
    actual = path.read_text("utf-8")
    assert actual == expected, (
        f"{OUTPUT} is not what `python -m ml.claims` produces -- it was edited by "
        "hand or its sources moved. Regenerate it."
    )


def test_resolve_rejects_positional_and_ambiguous_paths(docs: dict) -> None:
    """The path language must fail loudly, which is why it is trusted.

    Position-keyed paths are what broke this session's first gate run:
    inserting three horizons moved ``full`` from index 5 to index 8.
    """
    with pytest.raises(KeyError):
        resolve(docs["early_detection"], "horizons.summary")
    with pytest.raises(KeyError):
        resolve(docs["early_detection"], "horizons[cutoff_minutes=999].summary")
    with pytest.raises(KeyError):
        resolve(docs["early_detection"], "horizons[cutoff_minutes=5].summary.no_such_scope")


def test_the_scope_pair_is_not_swapped(docs: dict) -> None:
    """Cross-bank beats single-bank at T=5m -- the error that started this.

    A brief quoted 0.6220 as single-bank and 0.4282 as cross-bank, which
    inverts the project's central finding. Asserted as a direction, so
    swapping the two paths in the claim table fails even though both
    numbers would still be real.
    """
    cross = resolve(
        docs["early_detection"], "horizons[cutoff_minutes=5].summary.cross_bank.pr_auc_mean"
    )
    single = resolve(
        docs["early_detection"], "horizons[cutoff_minutes=5].summary.single_bank.pr_auc_mean"
    )
    assert cross > single
    by_id = {claim.id: claim for claim in CLAIMS}
    assert by_id["pr_auc_t5_cross_bank"].value(docs) == cross
    assert by_id["pr_auc_t5_single_bank"].value(docs) == single


def test_retired_claim_keeps_the_evidence_that_retired_it(docs: dict) -> None:
    """share_caught_before_exit must stay demonstrably degenerate.

    If a future change made the share differ from ring_recall at T=5m, or
    non-zero at T=180m, the stated reason for retiring it would no longer
    hold and this file would be arguing from numbers that had moved.
    """
    by_id = {c.id: c for item in RETIRED for c in item.evidence}
    assert by_id["retired_share_t5_cross_bank"].value(docs) == pytest.approx(
        by_id["retired_recall_t5_cross_bank"].value(docs)
    ), "the T=5m share no longer equals ring_recall; revisit why it was retired"
    assert by_id["retired_share_t180_cross_bank"].value(docs) == 0.0
    assert by_id["retired_recall_t180_cross_bank"].value(docs) == 1.0
    assert by_id["retired_rings_in_flight_t180"].value(docs) == 0.0


def test_qualifiers_survive_onto_the_numbers_that_need_them(published: dict) -> None:
    """Numbers that are wrong when quoted bare must carry their qualifier.

    The "12 of 27" incident was a correct number stripped of the context
    that made it mean anything. These particular claims are conditional
    or unconditional in ways a reader cannot infer from the value.
    """
    records = {r["id"]: r for r in published["claims"]}
    for claim_id in (
        "lead_median_minutes",
        "coverage_span1_cross_bank",
        "coverage_span3_single_bank",
        "ring_recall_t180_cross_bank",
    ):
        assert records[claim_id].get("qualifier"), f"{claim_id} has no qualifier"
    assert "CONDITIONAL" in records["lead_median_minutes"]["qualifier"]
    assert "UNCONDITIONAL" in records["coverage_span1_cross_bank"]["qualifier"]
