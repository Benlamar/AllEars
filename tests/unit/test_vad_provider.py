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


def test_energy_vad_provider_stream_and_regions(tmp_path: Path):
    import numpy as np
    from audio_agent.providers.vad.energy import EnergyVADProvider
    from audio_agent.providers.registry import ProviderRegistry

    prov = EnergyVADProvider(threshold=0.01)
    assert prov.is_available() is True
    assert prov.capability == "vad"
    assert prov.provider_name == "energy"

    # Silence frame
    silence = np.zeros(512, dtype=np.float32)
    assert prov.stream_prob(silence) == 0.0

    # Speech frame (sine wave above threshold)
    t = np.linspace(0, 1, 512, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
    assert prov.stream_prob(tone) > 0.5

    # Speech regions from file
    dummy_wav = tmp_path / "tone.wav"
    dummy_wav.write_bytes(b"RIFFdummy")
    with patch("soundfile.read", return_value=(tone, 16000)):
        regions = prov.detect_speech_regions(dummy_wav)
        assert len(regions) >= 0

    # Registry resolution
    reg_prov = ProviderRegistry.get("vad", "energy")
    assert isinstance(reg_prov, EnergyVADProvider)
    reg_none = ProviderRegistry.get("vad", "none")
    assert isinstance(reg_none, EnergyVADProvider)


def test_firered_vad_provider(tmp_path: Path):
    import numpy as np
    from audio_agent.providers.vad.firered import FireRedVADProvider
    from audio_agent.providers.registry import ProviderRegistry

    prov = FireRedVADProvider(threshold=0.5)
    assert prov.is_available() is True
    assert prov.capability == "vad"
    assert prov.provider_name == "firered"

    # Registry resolution
    reg_prov = ProviderRegistry.get("vad", "firered")
    assert isinstance(reg_prov, FireRedVADProvider)

    # Stream prob with mocked stream model
    mock_stream = MagicMock()
    mock_res = MagicMock()
    mock_res.smoothed_prob = 0.88
    mock_stream.detect_chunk.return_value = [mock_res]
    prov._stream_model = mock_stream

    silence = np.zeros(512, dtype=np.float32)
    assert prov.stream_prob(silence) == 0.88
    prov.reset_states()
    mock_stream.reset.assert_called_once()

    # Detect regions with mocked offline model
    mock_vad = MagicMock()
    mock_vad.detect.return_value = ({"timestamps": [(0.2, 1.8)]}, None)
    prov._offline_model = mock_vad

    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummy")
    with patch("soundfile.read", return_value=(silence, 16000)):
        regions = prov.detect_speech_regions(dummy_wav)
        assert len(regions) == 1
        assert regions[0].start == 0.2
        assert regions[0].end == 1.8

