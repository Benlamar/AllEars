"""Unit tests for FasterWhisperASRProvider."""

from unittest.mock import MagicMock, patch
from pathlib import Path
from audio_agent.providers.asr.faster_whisper import FasterWhisperASRProvider
from audio_agent.schemas import Word


def test_faster_whisper_provider_availability():
    provider = FasterWhisperASRProvider()
    assert provider.is_available() is True
    assert provider.capability == "asr"
    assert provider.provider_name == "faster_whisper"


def test_faster_whisper_transcribe_parsing(tmp_path: Path):
    dummy_wav = tmp_path / "test.wav"
    dummy_wav.write_bytes(b"RIFFdummydata")

    provider = FasterWhisperASRProvider(model_name="base.en", device="cpu")

    # Mock WhisperModel segment output
    mock_word = MagicMock()
    mock_word.word = "hello"
    mock_word.start = 0.1
    mock_word.end = 0.5
    mock_word.probability = 0.95

    mock_segment = MagicMock()
    mock_segment.text = "hello"
    mock_segment.words = [mock_word]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
    provider._model = mock_model

    transcript, words = provider.transcribe(dummy_wav)
    assert transcript == "hello"
    assert len(words) == 1
    assert words[0].text == "hello"
    assert words[0].start == 0.1
    assert words[0].end == 0.5
    assert words[0].confidence == 0.95

