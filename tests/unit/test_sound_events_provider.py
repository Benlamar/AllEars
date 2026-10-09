"""Unit tests for YAMNet Sound Event Detection provider."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from audio_agent.providers.sound_events.yamnet import YAMNetSoundEventProvider


def test_yamnet_provider_availability():
    provider = YAMNetSoundEventProvider()
    assert provider.is_available() is True
    assert provider.capability == "sound_events"
    assert provider.provider_name == "yamnet"


def test_yamnet_detect_events_mocked(tmp_path: Path):
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummy")

    provider = YAMNetSoundEventProvider(score_threshold=0.2)

    # Mock session and labels
    mock_session = MagicMock()
    # 2 frames, 3 classes: [Speech, Cough, Music]
    mock_scores = np.array([
        [0.85, 0.05, 0.10],
        [0.10, 0.90, 0.02],
    ], dtype=np.float32)
    mock_session.run.return_value = (mock_scores, None, None)
    mock_labels = ["Speech", "Cough", "Music"]

    provider._session = mock_session
    provider._labels = mock_labels

    with patch("soundfile.read", return_value=(np.zeros(16000, dtype=np.float32), 16000)):
        events = provider.detect_events(dummy_wav)
        assert len(events) >= 2
        labels = [e.label for e in events]
        assert "Speech" in labels
        assert "Cough" in labels

