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


def test_live_session_repair_aware_prompt():
    session = LiveConversationSession(
        asr_provider=MockASRProvider(),
        vad_provider=MockVADProvider(),
        llm_provider=MockLLMProvider(),
        tts_provider=MockTTSProvider(),
    )
    # Check system prompt includes guidance for phonetic corrections / repair
    system_content = session.messages[0]["content"]
    assert "correction" in system_content or "Calligraphy" in system_content
    assert "phonetic" in system_content


def test_live_session_backchannel_and_sigh_filtering():
    from audio_agent.audio.sanitizer import SpeechTextSanitizer

    # Test sanitizer directly
    assert SpeechTextSanitizer.is_passive_backchannel("okay") is True
    assert SpeechTextSanitizer.is_passive_backchannel("ok ok") is True
    assert SpeechTextSanitizer.is_passive_backchannel("mhm") is True
    assert SpeechTextSanitizer.is_passive_backchannel("hmm") is True
    assert SpeechTextSanitizer.is_passive_backchannel("[sigh]") is True
    assert SpeechTextSanitizer.is_passive_backchannel("yeah") is True
    assert SpeechTextSanitizer.is_passive_backchannel("pizza.") is False
    assert SpeechTextSanitizer.is_passive_backchannel("Are you gonna do it?") is False

    # Test live session turn handling
    asr = MockASRProvider()
    asr._mock_transcript = "okay"
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=MockVADProvider(),
        llm_provider=llm,
        tts_provider=tts,
    )
    backchannels = []
    session.on_backchannel = lambda txt: backchannels.append(txt)

    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)
    # LLM and assistant message should NOT be added
    assert len(session.messages) == 1  # only system message remains
    assert len(backchannels) == 1
    assert backchannels[0] == "okay"


def test_live_session_personal_mode_speaker_filtering():
    asr = MockASRProvider()
    asr._mock_transcript = "Tell me about pizza."
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=MockVADProvider(),
        llm_provider=llm,
        tts_provider=tts,
        mode="personal",
    )
    ignored = []
    session.on_filter_ignored = lambda lbl: ignored.append(lbl)

    # 1. First turn: user enrolls with voiced speech audio (e.g. 150 Hz pitch)
    t = np.linspace(0, 1, 16000, endpoint=False)
    turn1_audio = (np.sin(2 * np.pi * 150 * t) * 0.3).astype(np.float32)
    session._process_user_turn(turn1_audio)
    assert session.voice_anchor.is_enrolled is True
    assert len(session.messages) == 3  # system, user, assistant

    # 2. Second turn: background voice that does not match
    session.voice_anchor.matches = lambda audio, sr: False
    turn2_background = np.ones(16000, dtype=np.float32) * 0.05
    session._process_user_turn(turn2_background)
    assert len(ignored) == 1
    assert "Background Speaker" in ignored[0]
    # Messages should not have increased
    assert len(session.messages) == 3


def test_live_session_explicit_stop_command():
    from audio_agent.audio.sanitizer import SpeechTextSanitizer

    assert SpeechTextSanitizer.is_explicit_stop_command("Stop talking") is True
    assert SpeechTextSanitizer.is_explicit_stop_command("Be quiet") is True
    assert SpeechTextSanitizer.is_explicit_stop_command("Stop and listen to me") is True
    assert SpeechTextSanitizer.is_explicit_stop_command("Shut up") is True
    assert SpeechTextSanitizer.is_explicit_stop_command("How do you make pizza?") is False

    asr = MockASRProvider()
    asr._mock_transcript = "Stop talking"
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=MockVADProvider(),
        llm_provider=llm,
        tts_provider=tts,
    )
    stopped_cmd = []
    session.on_stop_command = lambda txt: stopped_cmd.append(txt)

    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)
    assert len(stopped_cmd) == 1
    assert stopped_cmd[0] == "Stop talking"
    assert len(session.messages) == 1  # No LLM answer generated
    assert session.state == SessionState.LISTENING


def test_live_session_floor_holding_incomplete_sentence():
    from audio_agent.audio.sanitizer import SpeechTextSanitizer

    assert SpeechTextSanitizer.is_syntactically_incomplete("how do you make a pizza in...") is True
    assert SpeechTextSanitizer.is_syntactically_incomplete("I want to know about") is True
    assert SpeechTextSanitizer.is_syntactically_incomplete("how do you make a pizza?") is False

    asr = MockASRProvider()
    asr._mock_transcript = "how do you make a pizza in..."
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=MockVADProvider(),
        llm_provider=llm,
        tts_provider=tts,
    )
    floor_held = []
    session.on_floor_held = lambda txt: floor_held.append(txt)

    turn1 = np.ones(16000, dtype=np.float32) * 0.1
    session._process_user_turn(turn1)
    # Turn 1 held floor!
    assert len(floor_held) == 1
    assert "pizza in..." in floor_held[0]
    assert session._pending_turn_audio is not None
    assert len(session.messages) == 1  # waiting for continuation

    # Turn 2: User completes sentence
    asr._mock_transcript = "how do you make a pizza in a home oven?"
    turn2 = np.ones(16000, dtype=np.float32) * 0.1
    session._process_user_turn(turn2)
    # Now complete and answered!
    assert len(session.messages) == 3




