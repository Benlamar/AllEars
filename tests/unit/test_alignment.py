"""Unit tests for word/speaker timeline alignment."""

from audio_agent.analysis.alignment import align_words_with_speakers
from audio_agent.schemas import SpeakerSegment, Word


def test_align_words_clean_assignment():
    words = [
        Word(text="Start", start=0.1, end=0.4),
        Word(text="meeting", start=0.5, end=0.9),
        Word(text="Agreed", start=1.5, end=1.9),
    ]
    segments = [
        SpeakerSegment(speaker_id="speaker_1", start=0.0, end=1.0),
        SpeakerSegment(speaker_id="speaker_2", start=1.4, end=2.0),
    ]

    aligned, uncertainties = align_words_with_speakers(words, segments)
    assert aligned[0].speaker_id == "speaker_1"
    assert aligned[1].speaker_id == "speaker_1"
    assert aligned[2].speaker_id == "speaker_2"
    assert len(uncertainties) == 0


def test_align_words_overlapping_speech():
    words = [
        Word(text="Overlap", start=1.0, end=1.5),
    ]
    segments = [
        SpeakerSegment(speaker_id="speaker_1", start=0.8, end=1.6),
        SpeakerSegment(speaker_id="speaker_2", start=0.9, end=1.7),
    ]

    aligned, uncertainties = align_words_with_speakers(words, segments)
    assert len(uncertainties) == 1
    assert "Overlapping speech" in uncertainties[0].message

