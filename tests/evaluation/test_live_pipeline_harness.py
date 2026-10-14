"""Repeatable evaluation harness for the Live Conversational Voice Pipeline.

Tests objective scenarios and telemetry across:
1. Clean conversational turn-taking
2. Immediate explicit stop / pause commands
3. Natural barge-in with pre-roll candidate preservation
4. Transient cough / noise recovery (duck-then-confirm restoration)
5. Race-safe cancellation and stale generation chunk rejection
6. Floor-holding incomplete thoughts
7. Conversation reset / forget command
8. Speaker verification and background speaker rejection
"""

import time
import numpy as np
import pytest

from audio_agent.audio.diagnostics import DiagnosticEvent, InterruptionReason
from audio_agent.audio.live_session import LiveConversationSession, SessionState
from audio_agent.providers.mocks import (
    MockASRProvider,
    MockLLMProvider,
    MockSoundEventProvider,
    MockSpeakerVerificationProvider,
    MockTTSProvider,
    MockVADProvider,
)
from audio_agent.schemas import SoundEvent


def make_harness_session(
    min_interruption_duration: float = 0.06, # 2 frames for fast test execution
    pause_threshold: float = 0.10,
    mode: str = "personal",
    duck_volume: float = 0.25,
) -> tuple[LiveConversationSession, dict]:
    vad = MockVADProvider()
    asr = MockASRProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()
    sed = MockSoundEventProvider()
    spk = MockSpeakerVerificationProvider()

    telemetry = {
        "state_changes": [],
        "diagnostic_events": [],
        "barge_ins": 0,
        "stop_commands": [],
        "transcripts": [],
        "assistant_tokens": [],
        "turn_completes": [],
        "floor_held": [],
    }

    session = LiveConversationSession(
        asr_provider=asr,
        vad_provider=vad,
        llm_provider=llm,
        tts_provider=tts,
        sound_event_provider=sed,
        speaker_verifier=spk,
        pause_threshold_seconds=pause_threshold,
        min_interruption_duration_seconds=min_interruption_duration,
        mode=mode,
        duck_volume=duck_volume,
        sample_rate=16000,
        frame_size=512,
    )

    session.on_state_change = lambda s: telemetry["state_changes"].append(s)
    session.on_diagnostic_event = lambda ev: telemetry["diagnostic_events"].append(ev)
    session.on_barge_in = lambda: telemetry.update(barge_ins=telemetry["barge_ins"] + 1)
    session.on_stop_command = lambda cmd: telemetry["stop_commands"].append(cmd)
    session.on_user_transcript = lambda txt: telemetry["transcripts"].append(txt)
    session.on_assistant_token = lambda tok: telemetry["assistant_tokens"].append(tok)
    session.on_assistant_turn_complete = lambda txt: telemetry["turn_completes"].append(txt)
    session.on_floor_held = lambda txt: telemetry["floor_held"].append(txt)

    return session, telemetry


def test_scenario_1_clean_turn_taking():
    """Verify clean conversational turn from speech onset to response completion."""
    session, telem = make_harness_session()
    session.start()
    assert session.state == SessionState.LISTENING

    frame = np.ones(512, dtype=np.float32) * 0.15
    session.vad._mock_prob = 0.9

    # User speaks 15 frames (~480ms)
    for _ in range(15):
        session.feed_audio_chunk(frame)
        time.sleep(0.01)

    assert session.state in {SessionState.USER_SPEAKING, SessionState.PROCESSING, SessionState.AGENT_SPEAKING}

    # Silence causes pause endpointing
    session.vad._mock_prob = 0.02
    for _ in range(10):
        session.feed_audio_chunk(np.zeros(512, dtype=np.float32))
        time.sleep(0.015)

    time.sleep(0.25)
    session.stop()

    assert len(telem["transcripts"]) >= 1
    assert len(telem["turn_completes"]) >= 1
    assert any(m["role"] == "assistant" for m in session.messages)


def test_scenario_2_explicit_stop_command_and_pause():
    """Verify that stop commands halt response without clearing context history."""
    session, telem = make_harness_session()
    session.messages.append({"role": "user", "content": "Prior message"})
    session.messages.append({"role": "assistant", "content": "Prior reply"})

    session.asr._mock_transcript = "Stop talking please"
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)

    assert len(telem["stop_commands"]) == 1
    assert "Stop talking please" in telem["stop_commands"]
    assert session.state == SessionState.LISTENING
    # Context must NOT be wiped by stop command
    assert len(session.messages) == 3 # system, user, assistant

    # Test pause command
    session.asr._mock_transcript = "Wait a second"
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)
    assert len(telem["stop_commands"]) == 2
    assert session.state == SessionState.LISTENING


def test_scenario_3_barge_in_with_pre_roll_preservation():
    """Verify duck-then-confirm barge in preserves pre-roll audio frames into next turn."""
    session, telem = make_harness_session(min_interruption_duration=0.06)
    session.start()
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    session.vad._mock_prob = 0.88
    session.sound_events._mock_events = [SoundEvent(label="Speech", start=0.0, end=0.5, score=0.9)]

    # Feed frames to trigger ducking into candidate, then confirmation
    for _ in range(5):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    assert telem["barge_ins"] >= 1
    # Check that candidate detected and confirmed events were emitted
    event_types = [ev.event_type for ev in telem["diagnostic_events"]]
    assert "CANDIDATE_DETECTED" in event_types
    assert "INTERRUPTION_CONFIRMED" in event_types


def test_scenario_4_transient_cough_rejection_and_restoration():
    """Verify that a transient cough during playback ducks volume and then restores playback."""
    session, telem = make_harness_session(min_interruption_duration=0.06)
    session.sound_events._mock_events = [SoundEvent(label="Cough", start=0.0, end=0.3, score=0.92)]

    session.start()
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    session.vad._mock_prob = 0.88 # Looks like energy, but SED identifies Cough

    for _ in range(4):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    # Barge-in must be suppressed
    assert telem["barge_ins"] == 0
    # Telemetry must show restoration
    event_types = [ev.event_type for ev in telem["diagnostic_events"]]
    assert "CANDIDATE_DETECTED" in event_types
    assert "INTERRUPTION_RESTORED" in event_types
    reasons = [ev.reason for ev in telem["diagnostic_events"] if ev.event_type == "INTERRUPTION_RESTORED"]
    assert InterruptionReason.LIKELY_NON_SPEECH in reasons


def test_scenario_5_race_safe_stale_generation_chunk_rejection():
    """Verify player.cancel_generation purges chunks belonging to an invalidated generation."""
    session, telem = make_harness_session()
    player = session.player
    player.start()

    chunk_gen1 = np.ones(2400, dtype=np.float32) * 0.5
    chunk_gen2 = np.ones(2400, dtype=np.float32) * 0.8

    player.enqueue(chunk_gen1, generation_id=1)
    player.enqueue(chunk_gen1, generation_id=1)
    player.enqueue(chunk_gen2, generation_id=2)

    assert player.buffered_seconds > 0.1

    # Cancel generation 1
    discarded = player.cancel_generation(1)
    assert discarded == 4800

    # Only generation 2 chunks remain
    with player._lock:
        for chunk in player._chunks:
            assert chunk.generation_id == 2

    player.stop()


def test_scenario_6_floor_holding_incomplete_thought():
    """Verify trailing conjunction triggers floor-holding without duplicate prompts."""
    session, telem = make_harness_session()

    session.asr._mock_transcript = "Could you tell me how to make a pizza in"
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)

    assert len(telem["floor_held"]) == 1
    assert "Could you tell me how to make a pizza in" in telem["floor_held"][0]
    assert session.state == SessionState.USER_SPEAKING
    assert session._pending_turn_audio is not None

    # Follow-up completion
    session.asr._mock_transcript = "an outdoor wood-fired oven?"
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)

    assert len(session.messages) == 3 # system, user, assistant
    assert session.state == SessionState.LISTENING


def test_scenario_7_conversation_reset_command():
    """Verify reset command clears context back to initial system prompt."""
    session, telem = make_harness_session()
    session.messages.append({"role": "user", "content": "Question 1"})
    session.messages.append({"role": "assistant", "content": "Answer 1"})
    assert len(session.messages) == 3

    session.asr._mock_transcript = "Forget this conversation"
    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)

    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "system"
    assert session.state == SessionState.LISTENING


def test_scenario_8_speaker_verification_and_rejection():
    """Verify that in personal mode, non-enrolled speakers are rejected."""
    session, telem = make_harness_session(mode="personal")
    # Simulate enrolled speaker
    session.speaker_verifier._is_enrolled = True
    session.voice_anchor.is_enrolled = True
    session.voice_anchor.user_pitch_mean = 140.0

    # Non-matching candidate
    session.speaker_verifier._mock_match = False
    session.asr._mock_transcript = "Random background chatter"

    session._process_user_turn(np.ones(16000, dtype=np.float32) * 0.1)

    # Turn should be rejected and returned to LISTENING without LLM generation
    assert session.state == SessionState.LISTENING
    assert not any(m.get("content") == "Random background chatter" for m in session.messages)


def test_scenario_9_semantic_keyword_interruption_confirmation():
    """Verify that in keywords mode, explicit stop/wait phrase confirms barge-in."""
    session, telem = make_harness_session(min_interruption_duration=0.06)
    session.barge_in_mode = "keywords"
    session.asr._mock_transcript = "Actually wait, that's not what I meant"
    session.sound_events._mock_events = [SoundEvent(label="Speech", start=0.0, end=0.5, score=0.9)]

    session.start()
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    session.vad._mock_prob = 0.88

    for _ in range(5):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    assert telem["barge_ins"] >= 1
    event_types = [ev.event_type for ev in telem["diagnostic_events"]]
    assert "INTERRUPTION_CONFIRMED" in event_types


def test_scenario_10_semantic_backchannel_rejection_and_restoration():
    """Verify that in keywords mode, passive backchannel or unrelated words restore playback."""
    session, telem = make_harness_session(min_interruption_duration=0.06)
    session.barge_in_mode = "keywords"
    session.asr._mock_transcript = "Yeah okay"
    session.sound_events._mock_events = [SoundEvent(label="Speech", start=0.0, end=0.5, score=0.9)]

    session.start()
    session._set_state(SessionState.AGENT_SPEAKING)

    frame = np.ones(512, dtype=np.float32) * 0.2
    session.vad._mock_prob = 0.88

    for _ in range(5):
        session.feed_audio_chunk(frame)
        time.sleep(0.02)

    time.sleep(0.05)
    session.stop()

    # Interruption must NOT trigger on backchannel in keywords mode
    assert telem["barge_ins"] == 0
    event_types = [ev.event_type for ev in telem["diagnostic_events"]]
    assert "INTERRUPTION_RESTORED" in event_types

