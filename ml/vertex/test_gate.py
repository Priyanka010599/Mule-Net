"""Tests for the cloud-vs-local reproduction gate.

The gate is the only thing standing between "the cloud job ran" and "the
cloud job produced the published numbers", so its failure modes are
tested explicitly: a gate that cannot fail is worth nothing.
"""

from __future__ import annotations

import json

import pytest

from ml.vertex.gate import compare, compare_bytes, compare_files, flatten, main

CANONICAL = {
    "model": "early_detection",
    "generated_at": "2026-09-26T14:27:46.891447+00:00",
    "horizons": [{"label": "T=5m", "summary": {"pr_auc_mean": 0.8694}}],
}


def test_flatten_indexes_list_elements() -> None:
    flat = flatten({"a": [{"b": 1}, {"b": 2}]})
    assert flat == {"a[0].b": 1, "a[1].b": 2}


def test_identical_but_for_generated_at_passes() -> None:
    cloud = dict(CANONICAL, generated_at="2026-09-27T09:00:00+00:00")
    report = compare(CANONICAL, cloud)
    assert report["passed"]
    assert report["leaves_ignored"] == ["generated_at"]
    assert report["leaves_compared"] == 3


def test_generated_at_is_the_only_ignored_leaf() -> None:
    """A ``generated_at`` anywhere but the top level is still compared."""
    local = {"a": {"generated_at": "x"}}
    cloud = {"a": {"generated_at": "y"}}
    report = compare(local, cloud)
    assert not report["passed"]
    assert [d["path"] for d in report["differences"]] == ["a.generated_at"]


def test_last_bit_float_difference_fails_with_a_delta() -> None:
    """No tolerance: one ULP is a failure and is reported as one."""
    cloud = json.loads(json.dumps(CANONICAL))
    cloud["horizons"][0]["summary"]["pr_auc_mean"] = 0.8694 + 1e-16
    report = compare(CANONICAL, cloud)
    assert not report["passed"]
    (diff,) = report["differences"]
    assert diff["path"] == "horizons[0].summary.pr_auc_mean"
    assert diff["local"] == 0.8694
    assert float(diff["delta"]) != 0.0


def test_int_and_float_of_equal_value_still_differ() -> None:
    """``1`` and ``1.0`` are compared by repr, so a type change is caught."""
    report = compare({"n": 1}, {"n": 1.0})
    assert not report["passed"]


def test_missing_leaf_fails() -> None:
    report = compare(CANONICAL, {"model": "early_detection", "generated_at": "z"})
    assert not report["passed"]
    assert report["missing_in_cloud"] == ["horizons[0].label", "horizons[0].summary.pr_auc_mean"]


def test_extra_leaf_fails() -> None:
    report = compare(CANONICAL, dict(CANONICAL, surprise=1))
    assert not report["passed"]
    assert report["extra_in_cloud"] == ["surprise"]


def test_empty_comparison_does_not_pass() -> None:
    """Two empty documents are not evidence of reproduction."""
    assert not compare({}, {})["passed"]


def test_compare_files_round_trip(tmp_path) -> None:
    local = tmp_path / "local.json"
    cloud = tmp_path / "cloud.json"
    local.write_text(json.dumps(CANONICAL), "utf-8")
    cloud.write_text(json.dumps(dict(CANONICAL, generated_at="later")), "utf-8")
    assert compare_files(local, cloud)["passed"]


def _oof_pair(tmp_path, cloud_bytes: bytes) -> tuple:
    local_dir = tmp_path / "local"
    cloud_dir = tmp_path / "cloud"
    local_dir.mkdir()
    cloud_dir.mkdir()
    (local_dir / "T5_cross_bank_seed42.parquet").write_bytes(b"PAR1payload")
    (cloud_dir / "T5_cross_bank_seed42.parquet").write_bytes(cloud_bytes)
    return local_dir, cloud_dir


def test_compare_bytes_passes_on_identical_files(tmp_path) -> None:
    local_dir, cloud_dir = _oof_pair(tmp_path, b"PAR1payload")
    report = compare_bytes(local_dir, cloud_dir, "*.parquet")
    assert report["passed"]
    assert report["leaves_compared"] == 1
    assert report["leaves_ignored"] == []


def test_compare_bytes_fails_on_one_changed_byte(tmp_path) -> None:
    local_dir, cloud_dir = _oof_pair(tmp_path, b"PAR1payloaD")
    report = compare_bytes(local_dir, cloud_dir, "*.parquet")
    assert not report["passed"]
    assert report["differences"][0]["local"] != report["differences"][0]["cloud"]


def test_compare_bytes_fails_when_cloud_wrote_nothing(tmp_path) -> None:
    local_dir = tmp_path / "local"
    cloud_dir = tmp_path / "cloud"
    local_dir.mkdir()
    cloud_dir.mkdir()
    (local_dir / "a.parquet").write_bytes(b"x")
    report = compare_bytes(local_dir, cloud_dir, "*.parquet")
    assert not report["passed"]
    assert report["missing_in_cloud"] == ["a.parquet"]


def test_cli_exit_code_follows_the_verdict(tmp_path, capsys) -> None:
    local = tmp_path / "local.json"
    cloud = tmp_path / "cloud.json"
    local.write_text(json.dumps(CANONICAL), "utf-8")
    cloud.write_text(json.dumps(dict(CANONICAL, generated_at="later")), "utf-8")
    assert main(["--pair", "early", str(local), str(cloud)]) == 0
    assert "GATE: PASSED" in capsys.readouterr().out

    cloud.write_text(json.dumps(dict(CANONICAL, model="something-else")), "utf-8")
    assert main(["--pair", "early", str(local), str(cloud)]) == 1
    assert "GATE: FAILED" in capsys.readouterr().out


def test_cli_requires_at_least_one_pair() -> None:
    with pytest.raises(SystemExit):
        main([])
