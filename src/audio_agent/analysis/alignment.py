"""Alignment of ASR word timestamps with speaker diarization segments."""

from typing import List, Tuple
from audio_agent.schemas import SpeakerSegment, UncertaintyNotice, Word


def align_words_with_speakers(
    words: List[Word],
    segments: List[SpeakerSegment],
    boundary_tolerance: float = 0.15,
) -> Tuple[List[Word], List[UncertaintyNotice]]:
    """Aligns timestamped words with speaker segments based on temporal overlap.
    
    Returns:
        Tuple of (attributed_words, uncertainty_notices)
    """
    attributed_words: List[Word] = []
    uncertainties: List[UncertaintyNotice] = []

    if not segments:
        return words, uncertainties

    for word in words:
        word_midpoint = (word.start + word.end) / 2.0
        
        # Check overlapping segments
        candidates = []
        for seg in segments:
            # Overlap condition: segment covers word midpoint or has significant span overlap
            if (seg.start - boundary_tolerance) <= word_midpoint <= (seg.end + boundary_tolerance):
                overlap_start = max(word.start, seg.start)
                overlap_end = min(word.end, seg.end)
                overlap_duration = max(0.0, overlap_end - overlap_start)
                candidates.append((seg.speaker_id, overlap_duration))

        if len(candidates) == 1:
            assigned_speaker = candidates[0][0]
        elif len(candidates) > 1:
            # Sort by overlap duration
            candidates.sort(key=lambda x: x[1], reverse=True)
            assigned_speaker = candidates[0][0]
            # Flag overlapping speaker ambiguity
            uncertainties.append(
                UncertaintyNotice(
                    stage="alignment",
                    message=f"Overlapping speech detected for word '{word.text}'; assigned to {assigned_speaker}",
                    start=word.start,
                    end=word.end,
                )
            )
        else:
            # Word fell between speaker intervals - look for nearest segment within 0.5s
            nearest_seg = None
            min_dist = float("inf")
            for seg in segments:
                dist = min(abs(word.start - seg.end), abs(word.end - seg.start))
                if dist < min_dist:
                    min_dist = dist
                    nearest_seg = seg

            if nearest_seg and min_dist <= 0.5:
                assigned_speaker = nearest_seg.speaker_id
            else:
                assigned_speaker = "unknown_speaker"
                uncertainties.append(
                    UncertaintyNotice(
                        stage="alignment",
                        message=f"Word '{word.text}' did not match any speaker interval; marked unknown",
                        start=word.start,
                        end=word.end,
                    )
                )

        attributed_words.append(
            Word(
                text=word.text,
                start=word.start,
                end=word.end,
                speaker_id=assigned_speaker,
                confidence=word.confidence,
            )
        )

    return attributed_words, uncertainties

