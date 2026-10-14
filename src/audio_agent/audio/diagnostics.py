"""Structured diagnostics and event telemetry for the live conversational session."""

from dataclasses import dataclass, field
import enum
import time
from typing import Any, Dict, Optional


class InterruptionReason(str, enum.Enum):
    LOW_SPEECH_CONFIDENCE = "LOW_SPEECH_CONFIDENCE"
    LOW_SIGNAL_LEVEL = "LOW_SIGNAL_LEVEL"
    LIKELY_BACKCHANNEL = "LIKELY_BACKCHANNEL"
    SPEAKER_MISMATCH = "SPEAKER_MISMATCH"
    LIKELY_NON_SPEECH = "LIKELY_NON_SPEECH"
    SPEAKER_ECHO_DETECTED = "SPEAKER_ECHO_DETECTED"
    INTERRUPTION_CONFIRMED = "INTERRUPTION_CONFIRMED"
    INTERRUPTION_RESTORED = "INTERRUPTION_RESTORED"
    CANDIDATE_TIMEOUT = "CANDIDATE_TIMEOUT"
    STALE_GENERATION_REJECTED = "STALE_GENERATION_REJECTED"
    EXPLICIT_STOP_COMMAND = "EXPLICIT_STOP_COMMAND"
    KEYWORD_TRIGGERED = "KEYWORD_TRIGGERED"
    NON_INTERRUPTION_INTENT = "NON_INTERRUPTION_INTENT"


@dataclass
class DiagnosticEvent:
    """Represents a structured operational telemetry event in the live pipeline."""

    event_type: str
    timestamp: float = field(default_factory=time.time)
    turn_id: Optional[int] = None
    generation_id: Optional[int] = None
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    reason: Optional[str] = None
    vad_confidence: Optional[float] = None
    candidate_duration_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "event_type": self.event_type,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "reason": self.reason,
            "vad_confidence": round(self.vad_confidence, 4) if self.vad_confidence is not None else None,
            "candidate_duration_ms": round(self.candidate_duration_ms, 2) if self.candidate_duration_ms is not None else None,
            "details": self.details,
        }
