"""Unit tests for SQLite database manager."""

from pathlib import Path
from audio_agent.schemas import AnalysisEvidence, RecordingMetadata, Word, SpeakerSegment
from audio_agent.storage.database import DatabaseManager


def test_database_crud(tmp_path: Path):
    db_file = tmp_path / "test.db"
    db = DatabaseManager(db_file)

    # 1. Save recording
    rec = RecordingMetadata(
        id="rec_001",
        filepath=str(tmp_path / "sample.wav"),
        filename="sample.wav",
        duration_seconds=12.5,
        filesize_bytes=4096,
    )
    db.save_recording(rec)
    recs = db.list_recordings()
    assert len(recs) == 1
    assert recs[0].id == "rec_001"

    # 2. Save analysis
    evidence = AnalysisEvidence(
        recording_id="rec_001",
        filename="sample.wav",
        duration_seconds=12.5,
        transcript="Test transcript",
        words=[Word(text="Test", start=0.0, end=0.5, speaker_id="speaker_1")],
        speaker_segments=[SpeakerSegment(speaker_id="speaker_1", start=0.0, end=1.0)],
        summary="Test executive summary",
    )
    db.save_analysis(evidence)

    # 3. Fetch analysis
    retrieved = db.get_analysis("rec_001")
    assert retrieved is not None
    assert retrieved.transcript == "Test transcript"
    assert retrieved.summary == "Test executive summary"

    # 4. Alias speaker
    db.set_speaker_alias("rec_001", "speaker_1", "Charlie")
    aliased = db.get_analysis("rec_001")
    assert aliased.speaker_aliases["speaker_1"] == "Charlie"

    # 5. Delete recording
    deleted = db.delete_recording("rec_001", delete_audio_file=False)
    assert deleted is True
    assert len(db.list_recordings()) == 0
    assert db.get_analysis("rec_001") is None

