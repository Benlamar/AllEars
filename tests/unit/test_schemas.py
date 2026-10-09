"""Unit tests for Pydantic schemas and serialization."""

from audio_agent.schemas import (
    AnalysisEvidence,
    AcousticFeatures,
    SoundEvent,
    SpeakerSegment,
    Word,
)


def test_word_and_segment_validation():
    word = Word(text="hello", start=0.0, end=0.5, speaker_id="speaker_1", confidence=0.99)
    assert word.text == "hello"
    assert word.speaker_id == "speaker_1"

    segment = SpeakerSegment(speaker_id="speaker_1", start=0.0, end=1.5, confidence=0.95)
    assert segment.start == 0.0
    assert segment.end == 1.5


def test_evidence_attributed_transcript():
    words = [
        Word(text="Hello", start=0.1, end=0.4, speaker_id="speaker_1"),
        Word(text="world", start=0.5, end=0.8, speaker_id="speaker_1"),
        Word(text="Hi", start=1.0, end=1.3, speaker_id="speaker_2"),
    ]
    segments = [
        SpeakerSegment(speaker_id="speaker_1", start=0.0, end=0.9),
        SpeakerSegment(speaker_id="speaker_2", start=0.95, end=1.5),
    ]
    evidence = AnalysisEvidence(
        recording_id="rec_test",
        filename="test.wav",
        duration_seconds=2.0,
        transcript="Hello world Hi",
        words=words,
        speaker_segments=segments,
        speaker_aliases={"speaker_1": "Alice", "speaker_2": "Bob"},
    )

    attributed = evidence.get_attributed_transcript()
    assert "Alice: Hello world" in attributed
    assert "Bob: Hi" in attributed

