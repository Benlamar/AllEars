"""Unit tests for LiveConversationSession turn-taking, pause endpointing, and barge-in."""

import time
import numpy as np
import pytest

from audio_agent.audio.live_session import (
    LiveConversationSession,
    SessionState,
    compute_audio_level_and_bands,
)
from audio_agent.providers.mocks import (
    MockASRProvider,
    MockLLMProvider,
    MockSoundEventProvider,
    MockTTSProvider,
    MockVADProvider,
)
from audio_agent.schemas import SoundEvent


def test_audio_level_computation():
    silence = np.zeros(512, dtype=np.float32)
    lvl_silence, bands_silence = compute_audio_level_and_bands(silence)
    assert lvl_silence == 0.0
    assert len(bands_silence) == 16

    # Sine wave
    t = np.linspace(0, 1, 512, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    lvl_tone, bands_tone = compute_audio_level_and_bands(tone)
    assert lvl_tone > 0.1
    assert len(bands_tone) == 16


def test_live_session_turn_taking_and_pause_endpointing():
    vad = MockVADProvider()
    asr = MockASRProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=vad,
        llm_provider=llm,
        tts_provider=tts,
        pause_threshold_seconds=0.1, # fast for unit test
        sample_rate=16000,
        frame_size=512,
    )

    user_transcripts = []
    session.on_user_transcript = lambda txt: user_transcripts.append(txt)

    session.start()
    assert session.state == SessionState.LISTENING

    frame = np.ones(512, dtype=np.float32) * 0.1

    # 1. Feed speech frames (mock returns prob 0.8)
    vad._mock_prob = 0.9
    for _ in range(5):
        session.feed_audio_chunk(frame)
        time.sleep(0.01)

    assert session.state in {SessionState.USER_SPEAKING, SessionState.PROCESSING, SessionState.AGENT_SPEAKING}

    # 2. Feed silence frames (mock returns prob 0.05) to exceed pause_threshold
    vad._mock_prob = 0.05
    for _ in range(6):
        session.feed_audio_chunk(frame * 0.0)
        time.sleep(0.02)

    # Wait for turn processing thread
    time.sleep(0.15)
    session.stop()
    assert session.state == SessionState.IDLE


def test_live_session_barge_in_interruption():
    vad = MockVADProvider()
    asr = MockASRProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()
    sed = MockSoundEventProvider()
    # Ensure SED does not report cough for this test
    sed._mock_events = [SoundEvent(label="Speech", start=0.0, end=1.0, score=0.95)]

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=vad,
        llm_provider=llm,
        tts_provider=tts,
        sound_event_provider=sed,
        min_interruption_duration_seconds=0.06, # 2 frames
        interruption_vad_threshold=0.6,
        sample_rate=16000,
        frame_size=512,
    )

    barge_in_called = []
    session.on_barge_in = lambda: barge_in_called.append(True)

    session.start()
    # Simulate agent speaking state
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    vad._mock_prob = 0.85 # strong speech

    # Feed enough frames to trigger interruption
    for _ in range(4):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    assert len(barge_in_called) >= 1


def test_live_session_filters_coughs_and_chuckles():
    vad = MockVADProvider()
    asr = MockASRProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()
    sed = MockSoundEventProvider()
    # Explicitly configure mock SED to report Cough
    sed._mock_events = [SoundEvent(label="Cough", start=0.0, end=0.4, score=0.88)]

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=vad,
        llm_provider=llm,
        tts_provider=tts,
        sound_event_provider=sed,
        filter_non_speech=True,
        min_interruption_duration_seconds=0.06,
        interruption_vad_threshold=0.6,
        sample_rate=16000,
        frame_size=512,
    )

    barge_in_called = []
    filter_ignored = []
    session.on_barge_in = lambda: barge_in_called.append(True)
    session.on_filter_ignored = lambda lbl: filter_ignored.append(lbl)

    session.start()
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    vad._mock_prob = 0.85 # looks like speech energy, but classifier flags Cough

    for _ in range(4):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    # Barge-in should be prevented because it's a cough!
    assert len(barge_in_called) == 0
    assert "Cough" in filter_ignored


def test_live_session_multi_turn_context_maintenance():
    vad = MockVADProvider()
    asr = MockASRProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    received_prompts = []
    def mock_generate_stream(prompt, **kwargs):
        received_prompts.append(prompt)
        yield "Mock response"

    llm.generate_stream = mock_generate_stream

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=vad,
        llm_provider=llm,
        tts_provider=tts,
        max_history_turns=2,
    )

    # Turn 1
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)
    assert len(session.messages) == 3 # system, user, assistant
    assert session.messages[0]["role"] == "system"
    assert session.messages[1]["role"] == "user"
    assert session.messages[2]["role"] == "assistant"

    # Turn 2
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)
    assert len(session.messages) == 5 # system, user1, asst1, user2, asst2
    # Check that second call received full prior turn context
    assert len(received_prompts[1]) == 4 # system, user1, asst1, user2


