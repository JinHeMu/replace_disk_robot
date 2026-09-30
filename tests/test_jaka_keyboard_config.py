"""Explicit boolean config/CLI semantics and their hardware-free effects."""
import json
import sys

import numpy as np
import pytest

from replace_disk_robot.applications.jaka_keyboard import config, node, jaka_keyboard_servo
from replace_disk_robot.adapters.jaka import JakaWristFTAdapter
from test_jaka_keyboard_admittance import DummyClient, LinearKinematics

PARSERS = (config._parse_args, jaka_keyboard_servo._parse_args)
SWITCHES = ("gravity_compensation_enable", "tare_enable", "plot_wrench_enable")


def parse_profile(monkeypatch, tmp_path, parser, values, *cli):
    profile = tmp_path / "profile.yaml"
    # JSON is valid YAML, including quoted-string vs boolean distinctions.
    profile.write_text(json.dumps(values))
    monkeypatch.setattr(sys, "argv", ["keyboard", "--config", str(profile), *cli])
    return parser()


@pytest.mark.parametrize("parser", PARSERS)
def test_positive_switches_default_enabled(monkeypatch, tmp_path, parser):
    args = parse_profile(monkeypatch, tmp_path, parser, {})
    assert all(getattr(args, name) is True for name in SWITCHES)
    assert not any(name.startswith("no_") for name in vars(args))


@pytest.mark.parametrize("parser", PARSERS)
def test_yaml_false_really_disables_switches(monkeypatch, tmp_path, parser):
    args = parse_profile(monkeypatch, tmp_path, parser, dict.fromkeys(SWITCHES, False))
    assert all(getattr(args, name) is False for name in SWITCHES)


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("enabled", [True, False])
def test_cli_explicit_bool_overrides_yaml(monkeypatch, tmp_path, parser, enabled):
    cli = [item for name in SWITCHES for item in
           ("--"+name.replace("_", "-"), str(enabled).lower())]
    args = parse_profile(monkeypatch, tmp_path, parser,
                         dict.fromkeys(SWITCHES, not enabled), *cli)
    assert all(getattr(args, name) is enabled for name in SWITCHES)


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("name", SWITCHES)
def test_quoted_yaml_false_is_rejected(monkeypatch, tmp_path, parser, name):
    with pytest.raises(SystemExit):
        parse_profile(monkeypatch, tmp_path, parser, {name: "false"})


@pytest.mark.parametrize("parser", PARSERS)
def test_ambiguous_cli_bool_is_rejected(monkeypatch, tmp_path, parser):
    with pytest.raises(SystemExit):
        parse_profile(monkeypatch, tmp_path, parser, {},
                      "--gravity-compensation-enable", "off")


@pytest.mark.parametrize("retained", [True, False])
@pytest.mark.parametrize("tare_enabled", [True, False])
def test_tare_switch_controls_actual_initialization(monkeypatch, tmp_path, retained, tare_enabled):
    args = parse_profile(monkeypatch, tmp_path, config._parse_args,
                         {"gravity_compensation_enable": False, "tare_enable": tare_enabled})
    calls = []
    def tare(adapter, **kwargs):
        calls.append(kwargs)
        return np.zeros(6)
    monkeypatch.setattr(JakaWristFTAdapter, "tare", tare)
    client, kinematics = DummyClient(), LinearKinematics()
    if retained:
        monkeypatch.setattr(jaka_keyboard_servo, "JakaKinematics", lambda: kinematics)
        app = jaka_keyboard_servo.JakaKeyboardServo(client, args)
    else:
        app = node.JakaKeyboardServo(client, args, kinematics=kinematics)
    app.initialize()
    assert len(calls) == int(tare_enabled)
    assert app.gravity_compensator is None
    app.shutdown()
