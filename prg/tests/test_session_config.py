"""Regression tests for prg.utils.session.

The session-config implementation lives in the *tracked* prg.utils.session so
the shipped ``--config`` CLI flag does not depend on the gitignored prg.gui
package. These tests pin that contract.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from prg.utils.session import SessionConfig, load_config, save_config

try:
    import tomli_w  # noqa: F401

    _HAS_TOMLI_W = True
except ModuleNotFoundError:
    _HAS_TOMLI_W = False

needs_writer = pytest.mark.skipif(not _HAS_TOMLI_W, reason="tomli-w not installed")


@needs_writer
def test_round_trip(tmp_path: Path) -> None:
    cfg = SessionConfig(
        filter_name="epkf",
        model_name="model_x1_y1_multiplicative",
        overrides={"q_y": 0.5},
        label="round-trip",
    )
    p = tmp_path / "s.toml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.filter_name == cfg.filter_name
    assert loaded.model_name == cfg.model_name
    assert loaded.overrides == cfg.overrides
    assert loaded.label == cfg.label


@needs_writer
def test_smoother_joseph_round_trip(tmp_path: Path) -> None:
    cfg = SessionConfig(
        filter_name="pkf",
        model_name="model_x1_y1_AQ_pairwise",
        smoother=True,
        joseph=True,
    )
    p = tmp_path / "s.toml"
    cfg.save(p)  # also exercises SessionConfig.save -> save_config
    loaded = load_config(p)
    assert loaded.smoother is True
    assert loaded.joseph is True


def test_invalid_filter_rejected() -> None:
    with pytest.raises(ValueError):
        SessionConfig(filter_name="not_a_filter", model_name="model_x1_y1_pairwise")


def test_missing_keys_rejected() -> None:
    with pytest.raises(ValueError):
        SessionConfig.from_dict({"model": {"name": "m"}})  # no filter.name


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.toml")


def test_cli_config_does_not_depend_on_gui() -> None:
    """The --config path in run_filter must load from prg.utils.session,
    never from the gitignored prg.gui (which is absent on a fresh install)."""
    import prg.run_filter as rf

    src = inspect.getsource(rf)
    assert "from prg.utils.session import load_config" in src
    assert "prg.gui" not in src
