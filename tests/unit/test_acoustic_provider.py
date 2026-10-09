"""Unit tests for Parselmouth acoustic provider."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from audio_agent.providers.acoustic.parselmouth_extractor import ParselmouthAcousticProvider


def test_parselmouth_provider_availability():
    provider = ParselmouthAcousticProvider()
    assert provider.is_available() is True
    assert provider.capability == "acoustic"
    assert provider.provider_name == "parselmouth"


def test_parselmouth_extract_features_mocked(tmp_path: Path):
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummy")

    provider = ParselmouthAcousticProvider()

    mock_sound = MagicMock()
    mock_sound.get_total_duration.return_value = 3.0
    mock_pitch = MagicMock()
    mock_pitch.selected_array = {"frequency": [120.0, 122.0, 0.0, 0.0]}
    mock_sound.to_pitch.return_value = mock_pitch
    mock_sound.to_intensity.return_value = MagicMock()
    mock_sound.to_harmonicity.return_value = MagicMock()

    with patch("parselmouth.Sound", return_value=mock_sound):
        with patch("parselmouth.praat.call", return_value=125.4):
            features = provider.extract_features(dummy_wav)
            assert features.pitch_mean_hz == 125.4
            assert features.intensity_mean_db == 125.4
            assert features.hnr_db == 125.4

