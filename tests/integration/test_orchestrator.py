"""Integration tests for the complete Agent Orchestrator pipeline."""

import wave
from pathlib import Path
from audio_agent.config import load_config
from audio_agent.orchestrator import AgentOrchestrator
from audio_agent.storage.database import DatabaseManager


def _create_dummy_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 4 seconds of audio frames
        wf.writeframes(b"\x00\x00" * (16000 * 4))
    return path


def test_orchestrator_full_pipeline(tmp_path: Path):
    audio_file = _create_dummy_wav(tmp_path / "sample.wav")
    db_file = tmp_path / "integration_test.db"

    config = load_config()
    config.models.asr.provider = "mock"
    config.models.vad.provider = "mock"
    config.models.diarization.provider = "mock"
    config.models.sound_events.provider = "mock"
    config.models.acoustic.provider = "mock"
    config.models.llm.provider = "mock"
    db = DatabaseManager(db_file)
    orchestrator = AgentOrchestrator(config, db)

    # 1. Run full analysis
    evidence = orchestrator.analyze_audio(audio_file)
    assert evidence.recording_id.startswith("rec_")
    assert "meeting" in evidence.transcript.lower()
    assert len(evidence.words) > 0
    assert len(evidence.speaker_segments) == 2
    assert len(evidence.sound_events) > 0
    assert evidence.acoustic_features.pitch_mean_hz is not None
    assert evidence.summary is not None
    assert "Speaker 1" in evidence.summary

    # 2. Check persistence in DB
    stored_evidence = db.get_analysis(evidence.recording_id)
    assert stored_evidence is not None
    assert stored_evidence.recording_id == evidence.recording_id

    # 3. Ask grounded follow-up question
    answer = orchestrator.ask_question(evidence.recording_id, "What are the action items discussed?")
    assert answer is not None
    assert len(answer) > 0


def test_orchestrator_pipeline_with_vad_disabled(tmp_path: Path):
    audio_file = _create_dummy_wav(tmp_path / "sample_novad.wav")
    db_file = tmp_path / "integration_test_novad.db"

    config = load_config()
    config.models.asr.provider = "mock"
    config.models.vad.enabled = False
    config.models.vad.provider = "none"
    config.models.diarization.provider = "mock"
    config.models.sound_events.provider = "mock"
    config.models.acoustic.provider = "mock"
    config.models.llm.provider = "mock"
    db = DatabaseManager(db_file)
    orchestrator = AgentOrchestrator(config, db)

    evidence = orchestrator.analyze_audio(audio_file)
    assert evidence.recording_id.startswith("rec_")
    # VAD was disabled, so speech_regions should be empty, but transcript and summary succeed
    assert len(evidence.speech_regions) == 0
    assert "meeting" in evidence.transcript.lower()
    assert evidence.summary is not None

