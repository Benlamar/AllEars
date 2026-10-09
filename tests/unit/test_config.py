"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path
from audio_agent.config import load_config, AppConfig
from audio_agent.exceptions import ConfigurationError


def test_load_default_config():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.defaults.application.name == "audio-agent"
    assert config.defaults.pipeline.language == "en"
    assert config.privacy.execution.get("mode") == "local_only"
    assert config.privacy.cloud.get("allowed") is False


def test_missing_config_uses_defaults(tmp_path: Path):
    # Empty directory should load safe default models without crashing
    config = load_config(tmp_path)
    assert isinstance(config, AppConfig)
    assert config.privacy.cloud.get("allowed") is False

