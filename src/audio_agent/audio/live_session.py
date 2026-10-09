"""Full-duplex live conversational voice session with pause endpointing, barge-in, and acoustic filtering."""

import enum
import math
import queue
import threading
import time
from typing import Callable, Dict, List, Optional
import numpy as np

from audio_agent.audio.chunker import StreamingClauseChunker
from audio_agent.audio.player import AudioStreamPlayer
from audio_agent.audio.sanitizer import SpeechTextSanitizer
from audio_agent.audio.voice_anchor import UserVoiceAnchor
from audio_agent.providers.base import ASRProvider, LLMProvider, SoundEventProvider, TTSProvider, VADProvider



class SessionState(str, enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    INTERRUPTED = "INTERRUPTED"


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
    """Orchestrates hands-free full-duplex conversational voice turn-taking."""

    def __init__(
        self,
        asr_provider: ASRProvider,
        vad_provider: VADProvider,
        llm_provider: LLMProvider,
        tts_provider: TTSProvider,
        sound_event_provider: Optional[SoundEventProvider] = None,
        pause_threshold_seconds: float = 1.8,
        vad_threshold: float = 0.5,
        interruption_vad_threshold: float = 0.82,
        min_interruption_duration_seconds: float = 0.65,
        filter_non_speech: bool = True,
        barge_in_mode: str = "auto",
        system_prompt: Optional[str] = None,
        max_history_turns: int = 10,
        sample_rate: int = 16000,
        frame_size: int = 512,  # 32ms at 16kHz
    ):
        self.asr = asr_provider
        self.vad = vad_provider
        self.llm = llm_provider
        self.tts = tts_provider
        self.sound_events = sound_event_provider
        self.pause_threshold = pause_threshold_seconds
        self.vad_threshold = vad_threshold
        self.interruption_vad_threshold = interruption_vad_threshold
        self.min_interruption_duration = min_interruption_duration_seconds
        self.filter_non_speech = filter_non_speech
        self.barge_in_mode = barge_in_mode.lower()
        self.max_history_turns = max_history_turns
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        self.voice_anchor = UserVoiceAnchor()
        self._noise_floor_rms = 0.005
        self._playback_mic_bleed_rms = 0.015

        self.state = SessionState.IDLE
        self.messages: List[Dict[str, str]] = []
        self._system_prompt = system_prompt or (
            "You are a helpful, conversational voice assistant. Speak naturally in concise sentences. "
            "Avoid markdown headers, asterisks, bullet points, and code blocks."
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
        self.on_filter_ignored: Optional[Callable[[str], None]] = None

    def _set_state(self, new_state: SessionState) -> None:
        if self.state != new_state:
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
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Stops live conversation session and shuts down audio."""
        self._is_active = False
        self._cancel_llm_event.set()
        self.player.stop()
        self._set_state(SessionState.IDLE)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

    def interrupt(self) -> None:
        """Instantly interrupts agent playback manually (via keypress, tap, or UI)."""
        if self.state == SessionState.AGENT_SPEAKING:
            self.player.stop()
            self._cancel_llm_event.set()
            self._set_state(SessionState.INTERRUPTED)
            if self.on_barge_in:
                self.on_barge_in()
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
        candidate_interruption_frames: List[np.ndarray] = []
        seconds_per_frame = self.frame_size / float(self.sample_rate)

        while self._is_active:
            try:
                frame = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

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

            # Adaptive noise floor tracking on non-speech frames
            if prob < self.vad_threshold:
                self._noise_floor_rms = 0.95 * self._noise_floor_rms + 0.05 * frame_rms

            # Noise-gated speech detection: requires VAD probability AND energy above ambient floor
            min_speech_energy = max(0.007, self._noise_floor_rms * 2.0)
            is_speech = (prob >= self.vad_threshold) and (frame_rms >= min_speech_energy)

            # -------------------------------------------------------------
            # STATE: LISTENING
            # -------------------------------------------------------------
            if self.state == SessionState.LISTENING:
                if is_speech:
                    speech_frame_count += 1
                    if speech_frame_count >= 2:  # ~64ms speech
                        self._set_state(SessionState.USER_SPEAKING)
                        user_speech_frames = [frame]
                        silence_frame_count = 0
                else:
                    speech_frame_count = 0

            # -------------------------------------------------------------
            # STATE: USER_SPEAKING
            # -------------------------------------------------------------
            elif self.state == SessionState.USER_SPEAKING:
                user_speech_frames.append(frame)
                if is_speech:
                    silence_frame_count = 0
                else:
                    silence_frame_count += 1
                    silence_duration = silence_frame_count * seconds_per_frame
                    if silence_duration >= self.pause_threshold:
                        # User paused long enough — turn finished!
                        self._set_state(SessionState.PROCESSING)
                        if hasattr(self.vad, "reset_states"):
                            self.vad.reset_states()

                        # Dispatch ASR and LLM response in a separate thread so audio queue keeps reading
                        turn_audio = np.concatenate(user_speech_frames, axis=0)
                        user_speech_frames = []
                        silence_frame_count = 0
                        speech_frame_count = 0
                        threading.Thread(
                            target=self._process_user_turn,
                            args=(turn_audio,),
                            daemon=True,
                        ).start()

            # -------------------------------------------------------------
            # STATE: AGENT_SPEAKING (Barge-In / Anti-Echo Guard)
            # -------------------------------------------------------------
            elif self.state == SessionState.AGENT_SPEAKING:
                # If barge-in mode is disabled or tap-only, mic does not interrupt
                if self.barge_in_mode in {"off", "none", "disabled", "tap"}:
                    candidate_interruption_frames = []
                    continue

                # Auto Barge-in with Echo & Voice Profile Guard
                is_currently_playing = self.player.is_playing
                if is_currently_playing:
                    # Speaker bleed adaptation
                    self._playback_mic_bleed_rms = (
                        0.90 * self._playback_mic_bleed_rms + 0.10 * frame_rms
                    )
                    # Interruption speech must exceed both high VAD threshold and speaker bleed surge
                    is_interruption_speech = (
                        prob >= self.interruption_vad_threshold
                        and frame_rms >= max(0.015, self._playback_mic_bleed_rms * 1.5)
                    )
                else:
                    is_interruption_speech = prob >= self.interruption_vad_threshold

                if is_interruption_speech:
                    candidate_interruption_frames.append(frame)
                    candidate_duration = len(candidate_interruption_frames) * seconds_per_frame

                    if candidate_duration >= self.min_interruption_duration:
                        candidate_audio = np.concatenate(candidate_interruption_frames, axis=0)

                        # Voice profile anchor check: verify candidate matches user rather than echo or noise
                        should_barge_in = True
                        if self.voice_anchor.is_enrolled:
                            if not self.voice_anchor.matches(candidate_audio, self.sample_rate):
                                should_barge_in = False
                                candidate_interruption_frames = []
                                continue

                        # Acoustic filter check (coughs, chuckles, laughter)
                        if self.filter_non_speech and self.sound_events is not None:
                            try:
                                detected = self.sound_events.detect_events_waveform(candidate_audio, self.sample_rate)
                                non_speech_hits = [
                                    ev for ev in detected 
                                    if ev.label in NON_SPEECH_VOCALIZATIONS and ev.score >= 0.20
                                ]
                                if non_speech_hits:
                                    should_barge_in = False
                                    top_hit = max(non_speech_hits, key=lambda x: x.score)
                                    if self.on_filter_ignored:
                                        self.on_filter_ignored(top_hit.label)
                                    candidate_interruption_frames = []
                            except Exception:
                                pass

                        if should_barge_in:
                            # BARGE-IN TRIGGERED!
                            self.player.stop()
                            self._cancel_llm_event.set()
                            self._set_state(SessionState.INTERRUPTED)
                            if self.on_barge_in:
                                self.on_barge_in()

                            # Transfer candidate frames to new user speech buffer
                            user_speech_frames = list(candidate_interruption_frames)
                            candidate_interruption_frames = []
                            silence_frame_count = 0
                            speech_frame_count = 0
                            self._set_state(SessionState.USER_SPEAKING)
                else:
                    if candidate_interruption_frames:
                        # Dropped below threshold before reaching minimum duration
                        candidate_interruption_frames = []

    def _process_user_turn(self, turn_audio: np.ndarray) -> None:
        """Transcribes user speech, queries LLM, and streams synthesized speech."""
        duration_s = len(turn_audio) / float(self.sample_rate)
        if duration_s < 0.4:
            # Too short to be a valid utterance
            self._set_state(SessionState.LISTENING)
            return

        try:
            transcript, _ = self.asr.transcribe(turn_audio)
            transcript = transcript.strip()
        except Exception:
            transcript = ""

        if not transcript:
            self._set_state(SessionState.LISTENING)
            return

        # Enroll user voice anchor on first utterance
        if not self.voice_anchor.is_enrolled:
            self.voice_anchor.enroll_user(turn_audio, self.sample_rate)

        if self.on_user_transcript:
            self.on_user_transcript(transcript)

        self.messages.append({"role": "user", "content": transcript})

        self._cancel_llm_event.clear()
        self._set_state(SessionState.AGENT_SPEAKING)

        chunker = StreamingClauseChunker()
        full_assistant_response: List[str] = []

        try:
            # Maintain rolling conversational context with system prompt + last N turns
            system_msg = [self.messages[0]] if self.messages and self.messages[0].get("role") == "system" else []
            history_msgs = self.messages[1:] if system_msg else self.messages
            max_msgs = max(1, self.max_history_turns * 2)
            trimmed_messages = system_msg + history_msgs[-max_msgs:]

            token_stream = self.llm.generate_stream(trimmed_messages, temperature=0.7)
            for token in token_stream:
                if self._cancel_llm_event.is_set():
                    break

                full_assistant_response.append(token)
                if self.on_assistant_token:
                    self.on_assistant_token(token)

                completed_clause = chunker.push(token)
                if completed_clause and not self._cancel_llm_event.is_set():
                    self._synthesize_and_enqueue_clause(completed_clause)

            # Flush any remaining tail clause
            tail = chunker.flush()
            if tail and not self._cancel_llm_event.is_set():
                self._synthesize_and_enqueue_clause(tail)

            # Wait for audio to finish playing if not interrupted
            if not self._cancel_llm_event.is_set():
                self.player.finish(wait=True)

        except Exception as e:
            print(f"\n[Agent Processing Error]: {e}")
        finally:
            complete_text = "".join(full_assistant_response).strip()
            if complete_text:
                self.messages.append({"role": "assistant", "content": complete_text})
                if self.on_assistant_turn_complete:
                    self.on_assistant_turn_complete(complete_text)

            # If not interrupted during playback, return to LISTENING
            if not self._cancel_llm_event.is_set() and self.state == SessionState.AGENT_SPEAKING:
                self._set_state(SessionState.LISTENING)

    def _synthesize_and_enqueue_clause(self, clause_text: str) -> None:
        """Sanitizes text clause, generates TTS audio, and pushes to player queue."""
        clean_text = SpeechTextSanitizer.clean_for_speech(clause_text)
        if not clean_text:
            return

        try:
            res = self.tts.synthesize(clean_text)
            if isinstance(res, tuple):
                audio_data = res[0]
            else:
                audio_data = res

            if self._cancel_llm_event.is_set():
                return
            if audio_data is not None and len(audio_data) > 0:
                if self.voice_anchor.agent_pitch_mean is None:
                    tts_sr = getattr(self.tts, "sample_rate", 24000)
                    self.voice_anchor.enroll_agent(audio_data, tts_sr)
                self.player.enqueue(audio_data)
        except Exception as e:
            print(f"\n[Agent TTS Error]: {e}")


