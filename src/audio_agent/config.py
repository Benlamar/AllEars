"""Configuration loading, merging, and schema validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field

from audio_agent.exceptions import ConfigurationError


class ApplicationConfig(BaseModel):
    name: str = "audio-agent"
    version: str = "0.1.0"
    log_level: str = "INFO"


class RecordingConfig(BaseModel):
    max_duration_seconds: int = 3600
    sample_rate: int = 16000
    channels: int = 1
    input_device: Optional[str] = None


class PipelineConfig(BaseModel):
    language: str = "en"
    vad_enabled: bool = True
    asr_enabled: bool = True
    diarization_enabled: bool = True
    sound_events_enabled: bool = True
    acoustic_features_enabled: bool = True
    tts_enabled: bool = True
    chunk_duration_seconds: int = 300


class PathsConfig(BaseModel):
    data_dir: str = "data"
    recordings_dir: str = "data/recordings"
    analyses_dir: str = "data/analyses"
    database_path: str = "data/database/audio_agent.db"


class DefaultsConfig(BaseModel):
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


class ProviderModelConfig(BaseModel):
    provider: str = "mock"
    model: str = "default"
    device: Optional[str] = "cuda"
    compute_type: Optional[str] = "float16"
    endpoint: Optional[str] = None
    temperature: Optional[float] = 0.1
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.15
    threshold: Optional[float] = 0.5
    pitch_floor_hz: Optional[float] = 75.0
    pitch_ceiling_hz: Optional[float] = 600.0
    beam_size: Optional[int] = 1
    max_tokens: Optional[int] = 384
    voice: Optional[str] = "af_sarah"
    speed: Optional[float] = 1.0
    task_type: Optional[str] = "VoiceDesign"
    instructions: Optional[str] = "A calm, friendly female voice with clear articulation."


class ModelsConfig(BaseModel):
    asr: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="base.en"))
    vad: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="silero-vad"))
    diarization: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="default"))
    sound_events: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="yamnet"))
    acoustic: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="parselmouth_librosa"))
    llm: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="qwen2.5:7b-instruct-q4_K_M"))
    tts: ProviderModelConfig = Field(default_factory=lambda: ProviderModelConfig(provider="mock", model="kokoro-v0_19.onnx", voice="af_sarah"))


class RetentionConfig(BaseModel):
    retain_audio: bool = True
    retain_transcripts: bool = True
    retain_analysis: bool = True
    auto_delete_audio_post_analysis: bool = False
    max_retention_days: Optional[int] = None


class PrivacyConfig(BaseModel):
    execution: Dict[str, str] = Field(default_factory=lambda: {"mode": "local_only"})
    cloud: Dict[str, bool] = Field(default_factory=lambda: {"allowed": False, "automatic_fallback": False})
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    telemetry: Dict[str, bool] = Field(default_factory=lambda: {"enabled": False})


class AppConfig(BaseModel):
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}") from e


def load_config(config_dir: Optional[Path] = None) -> AppConfig:
    """Loads configuration from YAML files in config_dir, with fallback defaults."""
    if config_dir is None:
        # Check standard config directory relative to repo root
        env_config = os.environ.get("AUDIO_AGENT_CONFIG_DIR")
        if env_config:
            config_dir = Path(env_config)
        else:
            config_dir = Path("config")

    defaults_data = _load_yaml(config_dir / "defaults.yaml")
    models_data = _load_yaml(config_dir / "models.yaml")
    privacy_data = _load_yaml(config_dir / "privacy.yaml")

    try:
        defaults = DefaultsConfig.model_validate(defaults_data) if defaults_data else DefaultsConfig()
        models = ModelsConfig.model_validate(models_data) if models_data else ModelsConfig()
        privacy = PrivacyConfig.model_validate(privacy_data) if privacy_data else PrivacyConfig()

        return AppConfig(defaults=defaults, models=models, privacy=privacy)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e

