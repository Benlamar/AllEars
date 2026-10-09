"""Unit tests for TTS providers."""

import pytest
import numpy as np
from audio_agent.providers.mocks import MockTTSProvider
from audio_agent.providers.tts.kokoro import KokoroTTSProvider
from audio_agent.providers.tts.qwen import QwenTTSProvider
from audio_agent.providers.registry import ProviderRegistry


def test_mock_tts_provider_synthesize():
    provider = MockTTSProvider()
    assert provider.is_available() is True
    assert provider.sample_rate == 24000

    samples = provider.synthesize("Hello world testing mock TTS.")
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 0
    assert samples.dtype == np.float32


def test_mock_tts_provider_stream_synthesize():
    provider = MockTTSProvider()
    stream = iter(["First phrase.", "Second phrase."])
    chunks = list(provider.stream_synthesize(stream))

    assert len(chunks) == 2
    for chunk, sr in chunks:
        assert isinstance(chunk, np.ndarray)
        assert sr == 24000


def test_kokoro_tts_provider_availability():
    provider = KokoroTTSProvider()
    assert provider.is_available() is True
    assert provider.sample_rate == 24000
    assert provider.voice == "af_sarah"


def test_qwen_tts_provider_initialization():
    provider = QwenTTSProvider(
        endpoint="http://192.168.1.6:8091/v1",
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        task_type="VoiceDesign",
        instructions="A calm voice.",
    )
    assert provider.model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert provider.speech_url == "http://192.168.1.6:8091/v1/audio/speech"


def test_registry_resolves_tts():
    mock_prov = ProviderRegistry.get("tts", "mock")
    assert isinstance(mock_prov, MockTTSProvider)

    kokoro_prov = ProviderRegistry.get("tts", "kokoro")
    assert isinstance(kokoro_prov, KokoroTTSProvider)

    qwen_prov = ProviderRegistry.get("tts", "qwen")
    assert isinstance(qwen_prov, QwenTTSProvider)


def test_qwen_tts_customvoice_settings():
    provider = QwenTTSProvider(
        endpoint="http://192.168.1.6:8091/v1",
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        task_type="CustomVoice",
        voice="vivian",
    )
    assert provider.model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert provider.task_type == "CustomVoice"
    assert provider.voice == "vivian"
    assert "vivian" in provider.PRESET_VOICES



