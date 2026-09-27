"""Configuration loading, paths, and the headless CLI."""

from __future__ import annotations

import pytest

from app.main import build_parser, main
from core.config import load_settings
from core.config.paths import data_dir
from core.errors import ConfigurationError
from core.safety import RiskLevel


# -- configuration -------------------------------------------------------


def test_defaults_work_with_no_config_file(settings):
    assert settings.runtime.runtime == "onnxruntime"
    assert settings.safety.allow_mutating_actions is False
    assert settings.ai.models == ()


def test_data_dir_follows_the_environment_override(isolated_data_dir):
    assert data_dir() == isolated_data_dir.resolve()


def test_malformed_toml_is_an_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(bad)


def test_a_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_settings(tmp_path / "nope.toml")


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "extra.toml"
    path.write_text('[ui]\nnot_a_real_setting = 1\n', encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(path, use_environment=False)


def test_approval_cannot_be_disabled_by_configuration(tmp_path):
    path = tmp_path / "unsafe.toml"
    path.write_text("[safety]\nrequire_approval = false\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(path, use_environment=False)


def test_max_risk_and_mutating_switch_round_trip(tmp_path):
    path = tmp_path / "risk.toml"
    path.write_text(
        '[safety]\nallow_mutating_actions = true\nmax_risk = "high"\n', encoding="utf-8"
    )
    settings = load_settings(path, use_environment=False)
    assert settings.safety.allow_mutating_actions is True
    assert settings.safety.max_risk is RiskLevel.HIGH


def test_overrides_beat_the_file(tmp_path):
    path = tmp_path / "level.toml"
    path.write_text('[logging]\nlevel = "INFO"\n', encoding="utf-8")
    settings = load_settings(
        path, overrides={"logging": {"level": "DEBUG"}}, use_environment=False
    )
    assert settings.logging.level == "DEBUG"


def test_to_policy_round_trip(settings):
    policy = settings.safety.to_policy()
    assert policy.max_risk is settings.safety.max_risk
    assert not policy.permits_remediation, "a fresh install proposes only read-only work"
    assert policy.require_approval is True


# -- CLI -----------------------------------------------------------------


def test_the_parser_exposes_the_documented_flags():
    parser = build_parser()
    args = parser.parse_args(["--diagnose", "x.log", "--log-level", "DEBUG"])
    assert args.diagnose == "x.log"
    assert args.log_level == "DEBUG"
    assert parser.parse_args(["--diagnose"]).diagnose == "-"


def test_headless_diagnosis_prints_a_report(capsys, tmp_path, log_file):
    code = main(["--diagnose", str(log_file), "--question", "why does it fail?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "outcome=" in out
    assert "Stages" in out
    assert "diagnosis" in out
    assert "rules_only" in out
    assert "Proposed actions" in out
    assert "nothing has been approved or run" in out


def test_headless_diagnosis_says_so_when_there_is_no_evidence(capsys, tmp_path):
    quiet = tmp_path / "quiet.log"
    quiet.write_text("everything is fine\n", encoding="utf-8")
    code = main(["--diagnose", str(quiet)])
    out = capsys.readouterr().out
    assert code == 1, "an unresolved run is reported as such"
    assert "No diagnosis" in out


def test_headless_diagnosis_reports_a_missing_file(capsys, tmp_path):
    code = main(["--diagnose", str(tmp_path / "nope.log")])
    assert code == 2
    assert "no such file" in capsys.readouterr().err


def test_a_broken_config_file_exits_with_a_message(capsys, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("nope = = 1", encoding="utf-8")
    code = main(["--config", str(bad)])
    assert code == 2
    assert "configuration error" in capsys.readouterr().err
