"""Unit tests for Nemotron Diarization provider."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from audio_agent.providers.diarization.nemotron import NemotronDiarizationProvider


def test_nemotron_provider_availability():
    provider = NemotronDiarizationProvider()
    assert provider.is_available() is True
    assert provider.capability == "diarization"
    assert provider.provider_name == "nemotron"


def test_nemotron_diarize_mocked(tmp_path: Path):
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummy")

    provider = NemotronDiarizationProvider(threshold=0.3)

    # Mock session
    mock_session = MagicMock()
    # Logits: (1, 100, 8), speaker 0 active with high logit (+3.0), speaker 1 active in second half
    logits = np.full((1, 100, 8), -3.0, dtype=np.float32)
    logits[0, :50, 0] = 3.0   # speaker 0
    logits[0, 50:, 1] = 3.0   # speaker 1

    mock_session.run.return_value = (logits, np.zeros((1, 100, 512), dtype=np.float32), None)
    provider._session = mock_session

    # Mock audio (1.0s at 16000Hz)
    audio_data = np.zeros(16000, dtype=np.float32)

    with patch("soundfile.read", return_value=(audio_data, 16000)):
        segments = provider.diarize(dummy_wav)
        assert len(segments) >= 1
        speakers = {s.speaker_id for s in segments}
        assert "speaker_1" in speakers or "speaker_2" in speakers

