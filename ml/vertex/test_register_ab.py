"""Tests for ml/vertex/register_ab.py's gcloud executable resolution.

Windows installs gcloud as gcloud.cmd; the bare string "gcloud" does not
resolve via subprocess without shell=True. _gcloud_executable() exists
to fix exactly that -- see its own docstring.
"""

from __future__ import annotations

from ml.vertex.register_ab import _gcloud_executable


def test_returns_a_string():
    assert isinstance(_gcloud_executable(), str)


def test_uses_shutil_which_result_when_found(monkeypatch):
    monkeypatch.setattr(
        "ml.vertex.register_ab.shutil.which", lambda name: "C:\\fake\\gcloud.cmd"
    )
    assert _gcloud_executable() == "C:\\fake\\gcloud.cmd"


def test_falls_back_to_bare_name_when_not_found(monkeypatch):
    monkeypatch.setattr("ml.vertex.register_ab.shutil.which", lambda name: None)
    assert _gcloud_executable() == "gcloud"
