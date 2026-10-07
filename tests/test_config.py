"""Tests for configuration loading and validation."""

import pytest
from ei.config import load_config


def test_load_config(cfg):
    """Test that config loads and fixture overrides are applied."""
    assert cfg["seed"] == 42
    assert cfg["estimation_window"] == 120
    assert cfg["warmup_days"] == 120
    assert "data" in cfg
    assert "optimizer" in cfg
    assert "costs" in cfg
    assert "signals" in cfg
    assert "covariance" in cfg


def test_missing_config_file():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")
