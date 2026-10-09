"""Unit tests for Silero VAD provider."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from audio_agent.providers.vad.silero import SileroVADProvider


def test_silero_vad_provider_availability():
    provider = SileroVADProvider()
    assert provider.is_available() is True
    assert provider.capability == "vad"
    assert provider.provider_name == "silero"


def test_silero_vad_detect_speech_regions(tmp_path: Path):
    import numpy as np
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummy")

    provider = SileroVADProvider()
    mock_model = MagicMock()
    provider._model = mock_model

    with patch("soundfile.read", return_value=(np.zeros(16000, dtype=np.float32), 16000)):
        with patch("silero_vad.get_speech_timestamps", return_value=[{"start": 0.5, "end": 2.1}]):
            regions = provider.detect_speech_regions(dummy_wav)
            assert len(regions) == 1
            assert regions[0].start == 0.5
            assert regions[0].end == 2.1
