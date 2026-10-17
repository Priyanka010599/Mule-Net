"""Tests for mode selection.

The rule these pin: **cloud is opt-in and is never inferred.** Only an
explicit ``MULENET_DOCAI_MODE=cloud`` selects it.

This exists because ambient Application Default Credentials used to
select cloud mode on their own. That made the same command behave
differently on different machines: logging into gcloud for an unrelated
reason silently flipped docai into cloud mode and broke six tests in
``agents/``, which reaches docai through ``search_regulations``.
Credentials being *available* is not the operator *intending* to use
them.
"""

from __future__ import annotations

import pytest

from docai import config as config_module
from docai.config import DocaiConfig, credentials_available, resolve_mode


def test_mode_is_mock_when_the_env_var_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MULENET_DOCAI_MODE", raising=False)
    assert resolve_mode() == "mock"
    assert DocaiConfig.from_env().mode == "mock"


def test_present_credentials_do_not_select_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression. ADC present + no explicit mode must still be mock.

    ``credentials_available`` is forced true so this pins the rule on any
    machine, whether or not ADC actually resolves there.
    """
    monkeypatch.delenv("MULENET_DOCAI_MODE", raising=False)
    monkeypatch.setattr(config_module, "credentials_available", lambda: True)

    assert resolve_mode() == "mock"
    assert DocaiConfig.from_env().mode == "mock"


def test_credentials_are_not_consulted_to_pick_a_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode selection must not call the credential check at all."""
    monkeypatch.delenv("MULENET_DOCAI_MODE", raising=False)

    def _fail() -> bool:
        raise AssertionError("resolve_mode must not consult credentials_available")

    monkeypatch.setattr(config_module, "credentials_available", _fail)
    assert resolve_mode() == "mock"


@pytest.mark.parametrize("value", ["cloud", "CLOUD", "  Cloud  "])
def test_cloud_is_selected_only_by_an_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MULENET_DOCAI_MODE", value)
    assert resolve_mode() == "cloud"


@pytest.mark.parametrize("value", ["mock", "MOCK", "  mock  "])
def test_mock_can_be_requested_explicitly(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MULENET_DOCAI_MODE", value)
    assert resolve_mode() == "mock"


def test_an_unknown_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not quietly fall back to either mode."""
    monkeypatch.setenv("MULENET_DOCAI_MODE", "clould")
    with pytest.raises(ValueError, match="MULENET_DOCAI_MODE"):
        resolve_mode()


def test_explicit_cloud_without_identifiers_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit intent plus missing config stays a loud, actionable error."""
    monkeypatch.setenv("MULENET_DOCAI_MODE", "cloud")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("MULENET_REGULATIONS_DATASTORE", raising=False)

    config = DocaiConfig.from_env()
    assert config.mode == "cloud"
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        config.require_cloud_ids(datastore=True)


def test_explicit_cloud_without_credentials_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials are checked here -- after the operator asked for cloud."""
    monkeypatch.setenv("MULENET_DOCAI_MODE", "cloud")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    monkeypatch.setattr(config_module, "credentials_available", lambda: False)

    with pytest.raises(RuntimeError, match="Application Default Credentials"):
        DocaiConfig.from_env().require_cloud_ids()


def test_credentials_available_returns_a_bool() -> None:
    """Whatever this machine has, the probe must answer without raising."""
    assert isinstance(credentials_available(), bool)
