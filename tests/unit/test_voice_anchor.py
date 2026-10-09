"""Tests for UserVoiceAnchor and anti-echo pitch/harmonicity matching."""

import numpy as np
import pytest
from audio_agent.audio.voice_anchor import UserVoiceAnchor


def generate_tone(freq_hz: float, duration_s: float = 0.5, sr: int = 16000) -> np.ndarray:
    """Generates a clean harmonic tone with fundamental frequency and harmonics."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Fundamental + 2 harmonics for rich pitch detection
    signal = (
        0.6 * np.sin(2 * np.pi * freq_hz * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq_hz * t)
        + 0.1 * np.sin(2 * np.pi * 3 * freq_hz * t)
    )
    return signal.astype(np.float32)


def generate_white_noise(duration_s: float = 0.5, sr: int = 16000) -> np.ndarray:
    """Generates unvoiced random white noise (representative of fan/air conditioning)."""
    return (np.random.normal(0, 0.2, int(sr * duration_s))).astype(np.float32)


def test_voice_anchor_initial_state():
    anchor = UserVoiceAnchor()
    assert not anchor.is_enrolled
    assert anchor.user_pitch_mean is None
    assert anchor.agent_pitch_mean is None


def test_voice_anchor_enroll_user_tone():
    anchor = UserVoiceAnchor()
    tone_150hz = generate_tone(150.0)

    enrolled = anchor.enroll_user(tone_150hz, sr=16000)
    assert enrolled is True
    assert anchor.is_enrolled is True
    assert anchor.user_pitch_mean is not None
    # Detected pitch should be close to 150 Hz
    assert abs(anchor.user_pitch_mean - 150.0) < 15.0


def test_voice_anchor_rejects_white_noise_enrollment():
    anchor = UserVoiceAnchor()
    noise = generate_white_noise()
    # Noise has no valid pitch/harmonicity
    # Even if an autocorrelation artifact is found, HNR should be very low
    _ = anchor.enroll_user(noise, sr=16000)
    # If it fails or returns pitch, user matching should reject white noise
    assert anchor.matches(noise, sr=16000) is False


def test_voice_anchor_user_matching():
    anchor = UserVoiceAnchor(pitch_tolerance_ratio=0.30)
    user_voice = generate_tone(140.0)
    anchor.enroll_user(user_voice, sr=16000)

    # Similar pitch should match
    similar_voice = generate_tone(145.0)
    assert anchor.matches(similar_voice, sr=16000) is True

    # Very different pitch should NOT match
    different_voice = generate_tone(350.0)
    assert anchor.matches(different_voice, sr=16000) is False


def test_voice_anchor_agent_echo_rejection():
    anchor = UserVoiceAnchor(pitch_tolerance_ratio=0.40)
    user_voice = generate_tone(130.0)
    anchor.enroll_user(user_voice, sr=16000)

    # Enroll agent with pitch 210 Hz
    agent_synth = generate_tone(210.0)
    anchor.enroll_agent(agent_synth, sr=16000)
    assert anchor.agent_pitch_mean is not None

    # Bleed audio from agent speaker
    agent_bleed = generate_tone(210.0)
    assert anchor.matches(agent_bleed, sr=16000) is False

    # User audio still matches
    assert anchor.matches(user_voice, sr=16000) is True


def test_voice_anchor_barge_in_modes():
    from audio_agent.audio.live_session import LiveConversationSession
    from unittest.mock import MagicMock

    mock_asr = MagicMock()
    mock_vad = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()

    # Test auto mode
    session_auto = LiveConversationSession(
        asr_provider=mock_asr,
        vad_provider=mock_vad,
        llm_provider=mock_llm,
        tts_provider=mock_tts,
        barge_in_mode="auto",
    )
    assert session_auto.barge_in_mode == "auto"

    # Test tap mode
    session_tap = LiveConversationSession(
        asr_provider=mock_asr,
        vad_provider=mock_vad,
        llm_provider=mock_llm,
        tts_provider=mock_tts,
        barge_in_mode="tap",
    )
    assert session_tap.barge_in_mode == "tap"

    # Test manual interrupt method
    interrupted_called = False
    def on_barge():
        nonlocal interrupted_called
        interrupted_called = True

    session_tap.on_barge_in = on_barge
    from audio_agent.audio.live_session import SessionState
    session_tap.state = SessionState.AGENT_SPEAKING
    session_tap.interrupt()
    assert interrupted_called is True
    assert session_tap.state == SessionState.LISTENING

