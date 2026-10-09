"""Core Pydantic data models for Audio Agent evidence, metadata, and results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Word(BaseModel):
    """Timestamped word with optional speaker attribution and confidence."""
    text: str
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    speaker_id: Optional[str] = Field(default=None, description="Speaker identifier, e.g. speaker_1")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="ASR confidence score")


class SpeakerSegment(BaseModel):
    """Temporal speech segment attributed to a speaker."""
    speaker_id: str
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SpeechRegion(BaseModel):
    """Voice Activity Detection (VAD) speech region."""
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SoundEvent(BaseModel):
    """A detected non-speech or environmental sound event."""
    label: str = Field(..., description="Sound event class (e.g. Cough, Laughter, Music)")
    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    score: float = Field(..., ge=0.0, le=1.0, description="Model classification probability")
    description: Optional[str] = None


class AcousticFeatures(BaseModel):
    """Extracted acoustic and voice quality measurements."""
    pitch_mean_hz: Optional[float] = None
    pitch_min_hz: Optional[float] = None
    pitch_max_hz: Optional[float] = None
    pitch_std_hz: Optional[float] = None
    intensity_mean_db: Optional[float] = None
    speaking_rate_syllables_per_sec: Optional[float] = None
    pause_count: Optional[int] = None
    pause_duration_ratio: Optional[float] = None
    jitter_local: Optional[float] = Field(default=None, description="Pitch perturbation quotient")
    shimmer_local: Optional[float] = Field(default=None, description="Amplitude perturbation quotient")
    hnr_db: Optional[float] = Field(default=None, description="Harmonics-to-Noise Ratio in dB")
    notes: Optional[str] = None


class Provenance(BaseModel):
    """Provenance tracking for models, versions, and execution modes."""
    stage: str
    provider_name: str
    model_name: str
    execution_mode: str = "local" # "local" or "cloud"
    version: Optional[str] = None
    latency_seconds: Optional[float] = None


class UncertaintyNotice(BaseModel):
    """Preserves pipeline caveats, uncertain speaker overlap, or measurement failures."""
    stage: str
    message: str
    start: Optional[float] = None
    end: Optional[float] = None


class AnalysisEvidence(BaseModel):
    """Complete structured evidence object representing analyzed audio."""
    recording_id: str
    filename: str
    duration_seconds: float = Field(..., ge=0.0)
    channels: int = 1
    sample_rate: int = 16000
    
    # Core outputs
    transcript: str = ""
    words: List[Word] = Field(default_factory=list)
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list)
    speech_regions: List[SpeechRegion] = Field(default_factory=list)
    sound_events: List[SoundEvent] = Field(default_factory=list)
    acoustic_features: AcousticFeatures = Field(default_factory=AcousticFeatures)
    
    # User tags / metadata
    speaker_aliases: Dict[str, str] = Field(default_factory=dict, description="Mapping e.g. speaker_1 -> Alice")
    uncertainties: List[UncertaintyNotice] = Field(default_factory=list)
    provenance: Dict[str, Provenance] = Field(default_factory=dict)
    summary: Optional[str] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_attributed_transcript(self) -> str:
        """Returns transcript formatted with speaker turns and aliases."""
        if not self.speaker_segments:
            return self.transcript
        
        # Group words by speaker turns or use speaker_segments
        lines: List[str] = []
        for segment in self.speaker_segments:
            alias = self.speaker_aliases.get(segment.speaker_id, segment.speaker_id)
            seg_words = [
                w.text for w in self.words
                if w.start >= segment.start - 0.15 and w.end <= segment.end + 0.15
            ]
            if seg_words:
                text = " ".join(seg_words)
                lines.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {alias}: {text}")
        
        return "\n".join(lines) if lines else self.transcript


class RecordingMetadata(BaseModel):
    """Metadata record for a stored recording."""
    id: str
    filepath: str
    filename: str
    duration_seconds: float
    filesize_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_retained: bool = True

