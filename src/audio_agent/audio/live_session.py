"""Full-duplex live conversational voice session with pause endpointing, barge-in, and acoustic filtering."""

from collections import deque
from dataclasses import dataclass
import enum
import math
import queue
import threading
import time
from typing import Callable, Deque, Dict, List, Optional
import numpy as np

from audio_agent.audio.diagnostics import DiagnosticEvent, InterruptionReason
from audio_agent.audio.player import AudioStreamPlayer
from audio_agent.audio.speech_chunker import ChunkerConfig
from audio_agent.audio.speech_pipeline import StreamingSpeechPipeline
from audio_agent.audio.sanitizer import SpeechTextSanitizer
from audio_agent.audio.voice_anchor import UserVoiceAnchor
from audio_agent.providers.base import (
    ASRProvider,
    LLMProvider,
    SoundEventProvider,
    SpeakerVerificationProvider,
    TTSProvider,
    VADProvider,
)
from audio_agent.providers.speaker_verification.acoustic_baseline import AcousticSpeakerVerificationProvider


class SessionState(str, enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    INTERRUPTION_CANDIDATE = "INTERRUPTION_CANDIDATE"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class TurnContext:
    turn_id: int
    generation_id: int


NON_SPEECH_VOCALIZATIONS = {
    "Cough",
    "Throat clearing",
    "Laughter",
    "Chuckle, chortle",
    "Giggle",
    "Snicker",
    "Belly laugh",
    "Sigh",
    "Snort",
    "Sneeze",
    "Burping, eructation",
    "Hiccup",
    "Yawn",
}

BANDS = 16
EDGES = [2]
for _i in range(1, BANDS + 1):
    EDGES.append(max(EDGES[-1] + 1, int(2 * 128 ** (_i / BANDS))))


def compute_audio_level_and_bands(samples: np.ndarray) -> tuple[float, list[float]]:
    """Calculates RMS level and 16 logarithmic frequency bands from an audio array."""
    if samples is None or len(samples) == 0:
        return 0.0, [0.0] * BANDS
    x = samples.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x)))
    level = min(1.0, rms * 12.0) ** 0.7
    try:
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        bands = [min(1.0, float(spec[EDGES[i]:EDGES[i + 1]].mean() / 6.0) ** 0.6) for i in range(BANDS)]
    except Exception:
        bands = [level] * BANDS
    return level, bands


class LiveConversationSession:
    """Orchestrates hands-free full-duplex conversational voice turn-taking with duck-then-confirm barge-in."""

    def __init__(
        self,
        asr_provider: ASRProvider,
        vad_provider: VADProvider,
        llm_provider: LLMProvider,
        tts_provider: TTSProvider,
        sound_event_provider: Optional[SoundEventProvider] = None,
        speaker_verifier: Optional[SpeakerVerificationProvider] = None,
        pause_threshold_seconds: float = 1.8,
        vad_threshold: float = 0.5,
        interruption_vad_threshold: float = 0.60,
        min_interruption_duration_seconds: float = 0.30,
        filter_non_speech: bool = True,
        barge_in_mode: str = "auto",
        system_prompt: Optional[str] = None,
        max_history_turns: int = 10,
        sample_rate: int = 16000,
        frame_size: int = 512,  # 32ms at 16kHz
        chunker_config: Optional[ChunkerConfig] = None,
        debug_dump_dir: Optional[str] = None,
        max_turn_duration_seconds: float = 12.0,
        mode: str = "personal",
        duck_volume: float = 0.25,
    ):
        self.debug_dump_dir = debug_dump_dir
        self.max_turn_duration = max_turn_duration_seconds
        self.mode = mode.lower()
        self._turn_counter = 0
        self._turn_id = 0
        self._generation_id = 0
        self.chunker_config = chunker_config or ChunkerConfig()
        self._speech: Optional[StreamingSpeechPipeline] = None
        self.asr = asr_provider
        self.vad = vad_provider
        self.llm = llm_provider
        self.tts = tts_provider
        self.sound_events = sound_event_provider
        self.speaker_verifier = speaker_verifier or AcousticSpeakerVerificationProvider()
        self.pause_threshold = pause_threshold_seconds
        self.vad_threshold = vad_threshold
        self.interruption_vad_threshold = interruption_vad_threshold
        self.min_interruption_duration = min_interruption_duration_seconds
        self.filter_non_speech = filter_non_speech
        self.barge_in_mode = barge_in_mode.lower()
        self.max_history_turns = max_history_turns
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.duck_volume = duck_volume

        # Praat voice anchor for agent synthetic echo matching
        self.voice_anchor = UserVoiceAnchor()
        self._noise_floor_rms = 0.005
        self._playback_mic_bleed_rms = 0.015

        # Rolling pre-roll buffer (~256ms) to prevent clipping onset of speech
        self._preroll_buffer: Deque[np.ndarray] = deque(maxlen=8)

        self.state = SessionState.IDLE
        self.messages: List[Dict[str, str]] = []
        self._system_prompt = system_prompt or (
            "You are an intelligent, highly articulate, and helpful conversational voice assistant speaking with a user via microphone.\n"
            "Provide clear, informative, and well-structured answers with substantive depth. Tailor your response length naturally to the request: "
            "give concise answers for brief inquiries, but provide thorough explanations, context, and practical details when discussing complex topics, questions, or ideas.\n"
            "Speak in fluent, spoken-style conversational prose. Never use visual markdown formatting like bolding, asterisks, hashtags, bullet point symbols, table grids, emojis, or code blocks, as these sound jarring when read aloud by text-to-speech engines. Instead, structure your thoughts with spoken transitions such as 'First', 'Next', 'Additionally', or 'For example'.\n"
            "If speech recognition produces phonetic oddities or if the user repeats or emphasizes a short correction "
            "(e.g. 'Calligraphy!'), immediately recognize that they are correcting a prior misunderstanding, "
            "discard the misrecognized words, and seamlessly address their intended topic."
        )
        self.messages.append({"role": "system", "content": self._system_prompt})

        self.player = AudioStreamPlayer(sample_rate=getattr(tts_provider, "sample_rate", 24000))
        self._is_active = False
        self._input_queue: queue.Queue = queue.Queue()
        self._loop_thread: Optional[threading.Thread] = None
        self._cancel_llm_event = threading.Event()

        # Callbacks
        self.on_state_change: Optional[Callable[[SessionState], None]] = None
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_assistant_token: Optional[Callable[[str], None]] = None
        self.on_assistant_turn_complete: Optional[Callable[[str], None]] = None
        self.on_audio_level: Optional[Callable[[float, List[float], bool], None]] = None
        self.on_barge_in: Optional[Callable[[], None]] = None
        self.on_barge_in_detailed: Optional[Callable[[str, str, Optional[str]], None]] = None
        self.on_filter_ignored: Optional[Callable[[str], None]] = None
        self.on_candidate_ignored: Optional[Callable[[str, str], None]] = None
        self.on_backchannel: Optional[Callable[[str], None]] = None
        self.on_stop_command: Optional[Callable[[str], None]] = None
        self.on_floor_held: Optional[Callable[[str], None]] = None
        self.on_timing: Optional[Callable[[Dict[str, float]], None]] = None
        self.on_diagnostic_event: Optional[Callable[[DiagnosticEvent], None]] = None

        # Turn continuation / floor-holding state & interruption cooldown
        self._pending_turn_audio: Optional[np.ndarray] = None
        self._in_incomplete_continuation: bool = False
        self._interruption_cooldown_frames: int = 0

    def _emit_diagnostic(self, event_type: str, reason: Optional[str] = None, **kwargs) -> None:
        if self.on_diagnostic_event:
            ev = DiagnosticEvent(
                event_type=event_type,
                turn_id=self._turn_id,
                generation_id=self._generation_id,
                state_before=kwargs.pop("state_before", None),
                state_after=kwargs.pop("state_after", self.state.value if isinstance(self.state, SessionState) else str(self.state)),
                reason=reason,
                details=kwargs,
            )
            try:
                self.on_diagnostic_event(ev)
            except Exception:
                pass

    def _set_state(self, new_state: SessionState) -> None:
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            if self.on_state_change:
                self.on_state_change(new_state)

    def start(self) -> None:
        """Starts the live conversational background loop."""
        if self._is_active:
            return
        self._is_active = True
        self.player.start()
        self._set_state(SessionState.LISTENING)

        # Pre-warm agent pitch from TTS in background for anti-echo guard
        if self.voice_anchor.agent_pitch_mean is None and hasattr(self.tts, "synthesize"):
            def _warmup_tts_pitch():
                try:
                    pcm = self.tts.synthesize("Ready.")
                    if pcm is not None and len(pcm) > 0:
                        sr = getattr(self.tts, "sample_rate", 24000)
                        self.voice_anchor.enroll_agent(pcm, sr)
                except Exception:
                    pass
            threading.Thread(target=_warmup_tts_pitch, daemon=True).start()

        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Stops live conversation session and shuts down audio."""
        self._is_active = False
        self._cancel_llm_event.set()
        if self._speech:
            self._speech.cancel()
        self.player.stop()
        self._set_state(SessionState.IDLE)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

    def interrupt(self) -> None:
        """Instantly interrupts agent playback manually (via keypress, tap, or UI)."""
        if self.state in {SessionState.AGENT_SPEAKING, SessionState.INTERRUPTION_CANDIDATE}:
            self.player.cancel_generation(self._generation_id)
            self.player.stop()
            self._cancel_llm_event.set()
            if self._speech:
                self._speech.cancel()
            self._set_state(SessionState.INTERRUPTED)
            if self.on_barge_in:
                self.on_barge_in()
            self._emit_diagnostic("MANUAL_INTERRUPT", reason=InterruptionReason.INTERRUPTION_CONFIRMED)
            self._set_state(SessionState.LISTENING)

    def feed_audio_chunk(self, chunk: np.ndarray) -> None:
        """Enqueues microphone audio block from InputStream callback."""
        if self._is_active and chunk is not None and len(chunk) > 0:
            self._input_queue.put(chunk.copy())

    def _run_event_loop(self) -> None:
        """Main event loop processing streaming audio frames and state transitions."""
        user_speech_frames: List[np.ndarray] = []
        silence_frame_count = 0
        speech_frame_count = 0
        consecutive_speech_frames = 0
        candidate_interruption_frames: List[np.ndarray] = []
        interruption_silence_gap = 0
        seconds_per_frame = self.frame_size / float(self.sample_rate)

        while self._is_active:
            try:
                frame = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Always maintain rolling pre-roll buffer
            self._preroll_buffer.append(frame.copy())

            # Level calculation for UI visualization
            if self.on_audio_level and self.state != SessionState.AGENT_SPEAKING:
                lvl, bnds = compute_audio_level_and_bands(frame)
                self.on_audio_level(lvl, bnds, False)

            frame_rms = float(np.sqrt(np.mean(frame * frame)))

            # Evaluate VAD
            try:
                prob = self.vad.stream_prob(frame) if hasattr(self.vad, "stream_prob") else 0.0
            except Exception:
                prob = 0.0

            # Noise-gated speech detection: requires VAD probability AND energy above ambient floor
            min_speech_energy = max(0.009, self._noise_floor_rms * 2.0)
            is_speech = (prob >= self.vad_threshold) and (frame_rms >= min_speech_energy)

            # Adaptive ambient noise floor tracking on non-speech frames
            if not is_speech and self.state != SessionState.AGENT_SPEAKING:
                self._noise_floor_rms = 0.95 * self._noise_floor_rms + 0.05 * frame_rms

            # -------------------------------------------------------------
            # STATE: LISTENING
            # -------------------------------------------------------------
            if self.state == SessionState.LISTENING:
                if is_speech:
                    speech_frame_count += 1
                    if speech_frame_count >= 2:  # ~64ms speech
                        self._set_state(SessionState.USER_SPEAKING)
                        # Prepend pre-roll buffer to preserve first syllables
                        user_speech_frames = list(self._preroll_buffer)
                        silence_frame_count = 0
                        consecutive_speech_frames = 2
                else:
                    speech_frame_count = 0

            # -------------------------------------------------------------
            # STATE: USER_SPEAKING
            # -------------------------------------------------------------
            elif self.state == SessionState.USER_SPEAKING:
                user_speech_frames.append(frame)
                turn_duration = len(user_speech_frames) * seconds_per_frame
                if is_speech:
                    consecutive_speech_frames += 1
                    if consecutive_speech_frames >= 2:
                        silence_frame_count = 0
                else:
                    consecutive_speech_frames = 0
                    silence_frame_count += 1

                silence_duration = silence_frame_count * seconds_per_frame
                # Dynamic pause tolerance: allow longer pause if holding an incomplete thought
                effective_pause_thresh = (
                    max(1.8, self.pause_threshold * 1.25)
                    if self._in_incomplete_continuation
                    else self.pause_threshold
                )

                if (silence_duration >= effective_pause_thresh) or (turn_duration >= self.max_turn_duration):
                    # User paused long enough OR max turn duration reached!
                    self._set_state(SessionState.PROCESSING)
                    if hasattr(self.vad, "reset_states"):
                        self.vad.reset_states()

                    pad_frames = int(0.32 / seconds_per_frame)
                    keep = max(1, len(user_speech_frames) - max(0, silence_frame_count - pad_frames))
                    turn_audio = np.concatenate(user_speech_frames[:keep], axis=0)
                    user_speech_frames = []
                    silence_frame_count = 0
                    speech_frame_count = 0
                    consecutive_speech_frames = 0
                    threading.Thread(
                        target=self._process_user_turn,
                        args=(turn_audio,),
                        daemon=True,
                    ).start()

            # -------------------------------------------------------------
            # STATE: AGENT_SPEAKING (Barge-In / Duck-then-Confirm Entry)
            # -------------------------------------------------------------
            elif self.state == SessionState.AGENT_SPEAKING:
                if self.barge_in_mode in {"off", "none", "disabled", "tap"}:
                    candidate_interruption_frames = []
                    interruption_silence_gap = 0
                    continue

                if self._interruption_cooldown_frames > 0:
                    self._interruption_cooldown_frames -= 1
                    continue

                is_currently_playing = self.player.is_playing
                if is_currently_playing:
                    if prob < self.vad_threshold:
                        self._playback_mic_bleed_rms = (
                            0.95 * self._playback_mic_bleed_rms + 0.05 * frame_rms
                        )
                    min_barge_energy = max(0.015, self._playback_mic_bleed_rms * 1.25)
                    is_interruption_speech = (
                        prob >= self.interruption_vad_threshold
                        and frame_rms >= min_barge_energy
                    )
                else:
                    is_interruption_speech = prob >= self.interruption_vad_threshold

                if is_interruption_speech:
                    # CANDIDATE DETECTED: Duck playback volume and transition to INTERRUPTION_CANDIDATE
                    self.player.duck(target_volume=self.duck_volume, ramp_ms=40.0)
                    self._set_state(SessionState.INTERRUPTION_CANDIDATE)
                    # Include pre-roll frames in candidate
                    candidate_interruption_frames = list(self._preroll_buffer)
                    interruption_silence_gap = 0
                    self._emit_diagnostic(
                        "CANDIDATE_DETECTED",
                        vad_confidence=prob,
                        frame_rms=frame_rms,
                    )

            # -------------------------------------------------------------
            # STATE: INTERRUPTION_CANDIDATE (Evaluate Signals While Ducked)
            # -------------------------------------------------------------
            elif self.state == SessionState.INTERRUPTION_CANDIDATE:
                is_currently_playing = self.player.is_playing
                if is_currently_playing:
                    min_barge_energy = max(0.015, self._playback_mic_bleed_rms * 1.25)
                    is_interruption_speech = (
                        prob >= self.interruption_vad_threshold
                        and frame_rms >= min_barge_energy
                    )
                else:
                    is_interruption_speech = prob >= self.interruption_vad_threshold

                if is_interruption_speech:
                    candidate_interruption_frames.append(frame)
                    interruption_silence_gap = 0
                else:
                    interruption_silence_gap += 1
                    # Allow up to 4 frames (~128ms) of brief unvoiced dips
                    if interruption_silence_gap <= 4:
                        candidate_interruption_frames.append(frame)
                    else:
                        # Candidate speech evaporated / brief blip -> RESTORE PLAYBACK!
                        self.player.restore(ramp_ms=60.0)
                        candidate_interruption_frames = []
                        interruption_silence_gap = 0
                        self._interruption_cooldown_frames = 15
                        self._set_state(SessionState.AGENT_SPEAKING)
                        self._emit_diagnostic(
                            "INTERRUPTION_RESTORED",
                            reason=InterruptionReason.LOW_SIGNAL_LEVEL,
                        )
                        continue

                candidate_duration = len(candidate_interruption_frames) * seconds_per_frame

                # Confirm or reject once sufficient candidate evidence is gathered
                if candidate_duration >= self.min_interruption_duration:
                    candidate_audio = np.concatenate(candidate_interruption_frames, axis=0)

                    # 1. Anti-Echo Check: Verify candidate pitch does not match synthetic agent pitch
                    if self.voice_anchor.agent_pitch_mean is not None:
                        cand_pitch, _ = self.voice_anchor._extract_pitch_and_hnr(candidate_audio, self.sample_rate)
                        if cand_pitch is not None:
                            diff = abs(cand_pitch - self.voice_anchor.agent_pitch_mean) / self.voice_anchor.agent_pitch_mean
                            if diff < 0.22:
                                # Speaker echo! Restore playback
                                self.player.restore(ramp_ms=60.0)
                                candidate_interruption_frames = []
                                interruption_silence_gap = 0
                                self._interruption_cooldown_frames = 20
                                self._set_state(SessionState.AGENT_SPEAKING)
                                self._emit_diagnostic(
                                    "INTERRUPTION_RESTORED",
                                    reason=InterruptionReason.SPEAKER_ECHO_DETECTED,
                                )
                                continue

                    # 2. Sound Event Check: Filter coughs, laughter, throat-clearing
                    if self.filter_non_speech and self.sound_events is not None:
                        try:
                            detected = self.sound_events.detect_events_waveform(candidate_audio, self.sample_rate)
                            non_speech_hits = [
                                ev for ev in detected 
                                if ev.label in NON_SPEECH_VOCALIZATIONS and ev.score >= 0.20
                            ]
                            if non_speech_hits:
                                top_hit = max(non_speech_hits, key=lambda x: x.score)
                                if self.on_filter_ignored:
                                    self.on_filter_ignored(top_hit.label)
                                self.player.restore(ramp_ms=60.0)
                                candidate_interruption_frames = []
                                interruption_silence_gap = 0
                                self._interruption_cooldown_frames = 20
                                self._set_state(SessionState.AGENT_SPEAKING)
                                self._emit_diagnostic(
                                    "INTERRUPTION_RESTORED",
                                    reason=InterruptionReason.LIKELY_NON_SPEECH,
                                    label=top_hit.label,
                                )
                                continue
                        except Exception:
                            pass

                    # 3. Speaker Verification Check (Personal Mode)
                    if self.mode == "personal" and self.speaker_verifier.is_enrolled:
                        is_match, score = self.speaker_verifier.verify(candidate_audio, self.sample_rate)
                        if not is_match:
                            if self.on_filter_ignored:
                                self.on_filter_ignored("Background Speaker (Non-Primary Voice)")
                            self.player.restore(ramp_ms=60.0)
                            candidate_interruption_frames = []
                            interruption_silence_gap = 0
                            self._interruption_cooldown_frames = 20
                            self._set_state(SessionState.AGENT_SPEAKING)
                            self._emit_diagnostic(
                                "INTERRUPTION_RESTORED",
                                reason=InterruptionReason.SPEAKER_MISMATCH,
                                score=score,
                            )
                            continue

                    # 4. Semantic ASR & Intent / Keyword Gate
                    cand_transcript = ""
                    try:
                        cand_transcript, _ = self.asr.transcribe(candidate_audio)
                        cand_transcript = cand_transcript.strip()
                    except Exception:
                        cand_transcript = ""

                    decision = SpeechTextSanitizer.classify_interruption_intent(
                        cand_transcript, mode=self.barge_in_mode
                    )

                    if not decision.is_interruption:
                        # Candidate speech was NOT an intentional interruption:
                        # Restore playback and let the agent continue speaking smoothly!
                        self.player.restore(ramp_ms=80.0)
                        candidate_interruption_frames = []
                        interruption_silence_gap = 0
                        self._interruption_cooldown_frames = 25
                        self._set_state(SessionState.AGENT_SPEAKING)

                        if self.on_candidate_ignored:
                            self.on_candidate_ignored(cand_transcript, decision.reason)
                        elif self.on_filter_ignored and cand_transcript:
                            self.on_filter_ignored(f"'{cand_transcript}' ({decision.reason})")

                        self._emit_diagnostic(
                            "INTERRUPTION_RESTORED",
                            reason=InterruptionReason.NON_INTERRUPTION_INTENT,
                            details={"transcript": cand_transcript, "reason": decision.reason},
                        )
                        continue

                    # 5. INTERRUPTION CONFIRMED!
                    active_gen = self._generation_id
                    self.player.cancel_generation(active_gen)
                    self.player.stop()
                    self._cancel_llm_event.set()
                    if self._speech is not None:
                        self._speech.cancel()

                    self._set_state(SessionState.INTERRUPTED)
                    if self.on_barge_in_detailed:
                        self.on_barge_in_detailed(cand_transcript, decision.trigger_type, decision.matched_phrase)
                    elif self.on_barge_in:
                        self.on_barge_in()

                    self._emit_diagnostic(
                        "INTERRUPTION_CONFIRMED",
                        reason=InterruptionReason.KEYWORD_TRIGGERED if decision.matched_phrase else InterruptionReason.INTERRUPTION_CONFIRMED,
                        candidate_duration_ms=candidate_duration * 1000.0,
                        details={
                            "transcript": cand_transcript,
                            "trigger": decision.trigger_type,
                            "phrase": decision.matched_phrase,
                        },
                    )

                    # Seamlessly transfer preserved candidate frames into the new user speech turn
                    user_speech_frames = list(candidate_interruption_frames)
                    candidate_interruption_frames = []
                    interruption_silence_gap = 0
                    silence_frame_count = 0
                    speech_frame_count = 0
                    self._set_state(SessionState.USER_SPEAKING)


    def _process_user_turn(self, turn_audio: np.ndarray) -> None:
        """Transcribes user speech, queries LLM, and streams synthesized speech."""
        t_turn_start = time.perf_counter()
        self._turn_id += 1

        # If continuing from an incomplete thought (held floor), prepend previously saved audio
        if self._pending_turn_audio is not None:
            if turn_audio is not None and len(turn_audio) > 0 and (len(turn_audio) / float(self.sample_rate)) >= 0.15:
                turn_audio = np.concatenate([self._pending_turn_audio, turn_audio], axis=0)
            else:
                turn_audio = self._pending_turn_audio
            self._pending_turn_audio = None

        duration_s = len(turn_audio) / float(self.sample_rate)

        if duration_s < 0.4:
            self._in_incomplete_continuation = False
            self._set_state(SessionState.LISTENING)
            return

        words = []
        asr_duration = 0.0
        try:
            t_asr = time.perf_counter()
            transcript, words = self.asr.transcribe(turn_audio)
            asr_duration = time.perf_counter() - t_asr
            transcript = transcript.strip()
        except Exception:
            transcript = ""

        self._dump_turn_debug(turn_audio, transcript, words)

        if not transcript:
            self._in_incomplete_continuation = False
            self._set_state(SessionState.LISTENING)
            return


        # Classify user conversational intent
        intent = SpeechTextSanitizer.classify_intent(transcript)

        # 1. Reset / Forget Command ("forget this conversation", "clear history", "start over")
        if intent == "RESET":
            self.player.cancel_generation(self._generation_id)
            self.player.stop()
            self._cancel_llm_event.set()
            sys_msg = [m for m in self.messages if m.get("role") == "system"]
            self.messages = sys_msg or [{"role": "system", "content": self._system_prompt}]
            self._in_incomplete_continuation = False
            self._pending_turn_audio = None
            self._emit_diagnostic("COMMAND_EXECUTED", reason="RESET")
            self._set_state(SessionState.LISTENING)
            return

        # 2. Stop or Pause Command ("stop talking", "be quiet", "wait", "hold on")
        if intent in {"STOP", "PAUSE"}:
            self.player.cancel_generation(self._generation_id)
            self.player.stop()
            self._cancel_llm_event.set()
            if self.on_stop_command:
                self.on_stop_command(transcript)
            self._in_incomplete_continuation = False
            self._pending_turn_audio = None
            self._emit_diagnostic("COMMAND_EXECUTED", reason=intent)
            self._set_state(SessionState.LISTENING)
            return

        # 3. Passive Backchannel Filter ("okay", "mhm", "yeah" without trailing clause)
        if intent == "BACKCHANNEL":
            if self.on_backchannel:
                self.on_backchannel(transcript)
            self._in_incomplete_continuation = False
            self._pending_turn_audio = None
            self._emit_diagnostic("FILTERED_BACKCHANNEL", reason=InterruptionReason.LIKELY_BACKCHANNEL)
            self._set_state(SessionState.LISTENING)
            return

        # 4. Speaker Verification (Personal Mode vs Meeting Mode)
        if self.mode == "personal":
            if not self.speaker_verifier.is_enrolled:
                self.speaker_verifier.enroll([turn_audio], self.sample_rate)
                self.voice_anchor.enroll_user(turn_audio, self.sample_rate)
            else:
                is_match, score = self.speaker_verifier.verify(turn_audio, self.sample_rate)
                acoustic_match = self.voice_anchor.matches(turn_audio, self.sample_rate)
                if not is_match or not acoustic_match:
                    if self.on_filter_ignored:
                        self.on_filter_ignored("Background Speaker (Non-Primary Voice)")
                    self._in_incomplete_continuation = False
                    self._pending_turn_audio = None
                    self._emit_diagnostic("SPEAKER_REJECTED", reason=InterruptionReason.SPEAKER_MISMATCH)
                    self._set_state(SessionState.LISTENING)
                    return

        # 5. Syntactically Incomplete Thought / Floor Holding (e.g. "how do you make a pizza in...")
        if intent == "INCOMPLETE" and not self._in_incomplete_continuation:
            self._in_incomplete_continuation = True
            self._pending_turn_audio = turn_audio
            if self.on_floor_held:
                self.on_floor_held(transcript)
            self._emit_diagnostic("FLOOR_HELD", text=transcript)
            self._set_state(SessionState.USER_SPEAKING)
            return

        self._in_incomplete_continuation = False
        self._pending_turn_audio = None

        if self.on_user_transcript:
            self.on_user_transcript(transcript)

        self.messages.append({"role": "user", "content": transcript})

        # Generate unique generation ID for race-safe cancellation
        self._generation_id += 1
        current_gen = self._generation_id

        # Drop any pipeline left over from a previous turn
        if self._speech is not None:
            self._speech.cancel()
            self._speech = None

        self._cancel_llm_event = threading.Event()
        self._set_state(SessionState.AGENT_SPEAKING)

        first_audio_time = None
        def handle_first_audio(audio_data: np.ndarray, sr: int) -> None:
            nonlocal first_audio_time
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - t_turn_start
            self._on_first_agent_audio(audio_data, sr)

        speech = StreamingSpeechPipeline(
            tts=self.tts,
            player=self.player,
            chunker_config=self.chunker_config,
            on_first_audio=handle_first_audio,
            cancel_event=self._cancel_llm_event,
            on_error=lambda e: print(f"\n[Agent TTS Error]: {e}"),
            generation_id=current_gen,
        )
        self._speech = speech
        full_assistant_response: List[str] = []
        first_token_time = None

        try:
            # Maintain rolling conversational context with system prompt + last N turns
            system_msg = [self.messages[0]] if self.messages and self.messages[0].get("role") == "system" else []
            history_msgs = self.messages[1:] if system_msg else self.messages
            max_msgs = max(1, self.max_history_turns * 2)
            trimmed_messages = system_msg + history_msgs[-max_msgs:]

            t_llm_req = time.perf_counter()
            token_stream = self.llm.generate_stream(trimmed_messages, temperature=0.7)
            for token in token_stream:
                if self._cancel_llm_event.is_set():
                    break

                if first_token_time is None:
                    first_token_time = time.perf_counter() - t_llm_req

                full_assistant_response.append(token)
                if self.on_assistant_token:
                    self.on_assistant_token(token)

                speech.push_text(token)
                speech.poll_stall()

            if not self._cancel_llm_event.is_set():
                speech.finish(wait=True)

        except Exception as e:
            print(f"\n[Agent Processing Error]: {e}")
        finally:
            # Context reflection: If interrupted, only store what was actually spoken
            if self._cancel_llm_event.is_set():
                spoken_text = " ".join(speech.synthesized_texts)
                complete_text = SpeechTextSanitizer.sanitize(spoken_text)
                if complete_text:
                    complete_text += " [interrupted]"
            else:
                complete_text = SpeechTextSanitizer.sanitize("".join(full_assistant_response))

            if complete_text:
                self.messages.append({"role": "assistant", "content": complete_text})
                if self.on_assistant_turn_complete:
                    self.on_assistant_turn_complete(complete_text)

            if self.on_timing and first_token_time is not None:
                first_audio_ms = (first_audio_time * 1000.0) if first_audio_time else 0.0
                total_turn_ms = (time.perf_counter() - t_turn_start) * 1000.0
                spoken_ms = max(0.0, total_turn_ms - first_audio_ms)
                self.on_timing({
                    "asr_ms": asr_duration * 1000.0,
                    "llm_ttft_ms": first_token_time * 1000.0,
                    "tts_first_audio_ms": first_audio_ms,
                    "spoken_ms": spoken_ms,
                    "total_ms": total_turn_ms,
                    "pause_threshold_s": self.pause_threshold,
                })

            # If not interrupted during playback, return to LISTENING
            if not self._cancel_llm_event.is_set() and self.state in {SessionState.AGENT_SPEAKING, SessionState.INTERRUPTION_CANDIDATE}:
                self._set_state(SessionState.LISTENING)

    def _dump_turn_debug(self, turn_audio: np.ndarray, transcript: str, words: list) -> None:
        """Saves the user's turn audio and raw ASR output for offline transcription audits."""
        if not self.debug_dump_dir:
            return
        try:
            import json
            from pathlib import Path
            import soundfile as sf

            self._turn_counter += 1
            out_dir = Path(self.debug_dump_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"turn_{int(time.time())}_{self._turn_counter:03d}"
            sf.write(str(out_dir / f"{stem}.wav"), turn_audio, self.sample_rate)
            low_conf = [w.text for w in words if getattr(w, "confidence", None) is not None and w.confidence < 0.5]
            record = {
                "duration_s": round(len(turn_audio) / float(self.sample_rate), 2),
                "rms": round(float(np.sqrt(np.mean(turn_audio * turn_audio))), 4),
                "raw_asr_text": transcript,
                "sent_to_llm": transcript,
                "low_confidence_words": low_conf,
                "words": [
                    {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in words
                ],
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"\n[Debug dump error]: {e}")

    def _on_first_agent_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Enrolls the agent's voice pitch for the anti-echo guard from its first audio chunk."""
        if self.voice_anchor.agent_pitch_mean is None:
            self.voice_anchor.enroll_agent(audio_data, sample_rate)
