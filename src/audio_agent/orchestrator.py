"""Main Agent Orchestrator coordinating audio pipelines, privacy enforcement, and LLM reasoning."""

import concurrent.futures
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from audio_agent.analysis.evidence_builder import EvidenceBuilder
from audio_agent.audio.validation import validate_audio_file
from audio_agent.config import AppConfig
from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import (
    ASRProvider,
    AcousticProvider,
    DiarizationProvider,
    LLMProvider,
    SoundEventProvider,
    TTSProvider,
    VADProvider,
)
from audio_agent.providers.registry import ProviderRegistry
from audio_agent.schemas import AnalysisEvidence, RecordingMetadata
from audio_agent.security.privacy_policy import PrivacyPolicyEnforcer
from audio_agent.storage.database import DatabaseManager


class AgentOrchestrator:
    """Coordinates the end-to-end local-first audio intelligence workflow."""

    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        privacy_enforcer: Optional[PrivacyPolicyEnforcer] = None,
    ):
        self.config = config
        self.db = db
        self.privacy = privacy_enforcer or PrivacyPolicyEnforcer(config.privacy)
        self._provider_cache = {}

    def _resolve_provider(self, capability: str) -> any:
        if capability in self._provider_cache:
            return self._provider_cache[capability]

        model_cfg = getattr(self.config.models, capability)
        provider_name = model_cfg.provider
        
        # Enforce privacy check before creating provider
        # Local network and local model providers are 'local'
        local_providers = {
            "mock", "faster_whisper", "faster-whisper", "yamnet", 
            "silero", "silero_vad", "silero-vad", "pyannote", 
            "nemotron", "parselmouth", "parselmouth_librosa", 
            "openai_compatible", "tensorrt_llm", "vllm", "ollama", "llama_cpp",
            "kokoro", "qwen", "qwen_tts"
        }
        exec_mode = "local" if provider_name in local_providers else "cloud"
        self.privacy.validate_provider_execution(capability, provider_name, exec_mode)

        kwargs = {}
        if capability == "asr" and provider_name in {"faster_whisper", "faster-whisper"}:
            if model_cfg.model:
                kwargs["model_name"] = model_cfg.model
            if model_cfg.device:
                kwargs["device"] = model_cfg.device
            if model_cfg.compute_type:
                kwargs["compute_type"] = model_cfg.compute_type
            if getattr(model_cfg, "beam_size", None) is not None:
                kwargs["beam_size"] = model_cfg.beam_size
        elif capability == "vad" and provider_name in {"silero", "silero_vad", "silero-vad"}:
            if model_cfg.threshold is not None:
                kwargs["threshold"] = model_cfg.threshold
            if model_cfg.device:
                kwargs["device"] = model_cfg.device
        elif capability == "diarization" and provider_name in {"nemotron"}:
            if model_cfg.threshold is not None:
                kwargs["threshold"] = model_cfg.threshold
            if model_cfg.device:
                kwargs["device"] = model_cfg.device
        elif capability == "sound_events" and provider_name in {"yamnet"}:
            if model_cfg.score_threshold is not None:
                kwargs["score_threshold"] = model_cfg.score_threshold
            if model_cfg.top_k is not None:
                kwargs["top_k"] = model_cfg.top_k
            if model_cfg.device:
                kwargs["device"] = model_cfg.device
        elif capability == "acoustic" and provider_name in {"parselmouth", "parselmouth_librosa"}:
            if model_cfg.pitch_floor_hz is not None:
                kwargs["pitch_floor_hz"] = model_cfg.pitch_floor_hz
            if model_cfg.pitch_ceiling_hz is not None:
                kwargs["pitch_ceiling_hz"] = model_cfg.pitch_ceiling_hz
        elif capability == "llm" and provider_name in {"openai_compatible", "tensorrt_llm", "vllm"}:
            if model_cfg.endpoint:
                kwargs["endpoint"] = model_cfg.endpoint
            if model_cfg.model:
                kwargs["model_name"] = model_cfg.model
            if model_cfg.temperature is not None:
                kwargs["temperature"] = model_cfg.temperature
            if getattr(model_cfg, "max_tokens", None) is not None:
                kwargs["max_tokens"] = model_cfg.max_tokens
        elif capability == "tts" and provider_name in {"kokoro"}:
            if getattr(model_cfg, "voice", None):
                kwargs["voice"] = model_cfg.voice
            if getattr(model_cfg, "speed", None) is not None:
                kwargs["speed"] = model_cfg.speed
            if getattr(model_cfg, "device", None):
                kwargs["device"] = model_cfg.device
        elif capability == "tts" and provider_name in {"qwen", "qwen_tts", "vllm"}:
            if model_cfg.endpoint:
                kwargs["endpoint"] = model_cfg.endpoint
            if model_cfg.model and model_cfg.model != "default":
                kwargs["model_name"] = model_cfg.model
            if getattr(model_cfg, "task_type", None):
                kwargs["task_type"] = model_cfg.task_type
            if getattr(model_cfg, "instructions", None):
                kwargs["instructions"] = model_cfg.instructions

        provider = ProviderRegistry.get(capability, provider_name, **kwargs)
        self._provider_cache[capability] = provider
        return provider

    def analyze_audio(
        self,
        audio_path: Path,
        recording_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AnalysisEvidence:
        """Executes full analysis pipeline on audio file and persists structured evidence."""
        validated_path = validate_audio_file(audio_path)
        rec_id = recording_id or f"rec_{uuid.uuid4().hex[:12]}"

        # Estimate duration (or default for mock/wav)
        duration_seconds = 4.5  # placeholder or read from header
        try:
            import wave
            with wave.open(str(validated_path), "rb") as wf:
                duration_seconds = wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

        # Save metadata record
        filesize = validated_path.stat().st_size
        rec_meta = RecordingMetadata(
            id=rec_id,
            filepath=str(validated_path.resolve()),
            filename=validated_path.name,
            duration_seconds=duration_seconds,
            filesize_bytes=filesize,
            is_retained=self.config.privacy.retention.retain_audio,
        )
        self.db.save_recording(rec_meta)

        builder = EvidenceBuilder(rec_id, validated_path, duration_seconds)

        # 1-5. Execute all independent extraction stages concurrently
        def _run_vad():
            vad_prov: VADProvider = self._resolve_provider("vad")
            t0 = time.perf_counter()
            regions = vad_prov.detect_speech_regions(validated_path)
            return regions, vad_prov.get_provenance(time.perf_counter() - t0)

        def _run_asr():
            asr_prov: ASRProvider = self._resolve_provider("asr")
            t0 = time.perf_counter()
            lang = language or self.config.defaults.pipeline.language
            transcript, words = asr_prov.transcribe(validated_path, language=lang)
            return (transcript, words), asr_prov.get_provenance(time.perf_counter() - t0)

        def _run_diarization():
            diar_prov: DiarizationProvider = self._resolve_provider("diarization")
            t0 = time.perf_counter()
            segments = diar_prov.diarize(validated_path)
            return segments, diar_prov.get_provenance(time.perf_counter() - t0)

        def _run_sound_events():
            sed_prov: SoundEventProvider = self._resolve_provider("sound_events")
            t0 = time.perf_counter()
            events = sed_prov.detect_events(validated_path)
            return events, sed_prov.get_provenance(time.perf_counter() - t0)

        def _run_acoustics():
            ac_prov: AcousticProvider = self._resolve_provider("acoustic")
            t0 = time.perf_counter()
            features = ac_prov.extract_features(validated_path)
            return features, ac_prov.get_provenance(time.perf_counter() - t0)

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            if self.config.defaults.pipeline.vad_enabled:
                futures["vad"] = executor.submit(_run_vad)
            if self.config.defaults.pipeline.asr_enabled:
                futures["asr"] = executor.submit(_run_asr)
            if self.config.defaults.pipeline.diarization_enabled:
                futures["diarization"] = executor.submit(_run_diarization)
            if self.config.defaults.pipeline.sound_events_enabled:
                futures["sound_events"] = executor.submit(_run_sound_events)
            if self.config.defaults.pipeline.acoustic_features_enabled:
                futures["acoustic"] = executor.submit(_run_acoustics)

        # Collect stage results and record into evidence builder
        if "vad" in futures:
            regions, prov = futures["vad"].result()
            builder.set_vad(regions, prov)

        if "asr" in futures:
            (transcript, words), prov = futures["asr"].result()
            builder.set_transcript(transcript, words, prov)

        if "diarization" in futures:
            segments, prov = futures["diarization"].result()
            builder.set_diarization(segments, prov)

        if "sound_events" in futures:
            events, prov = futures["sound_events"].result()
            builder.set_sound_events(events, prov)

        if "acoustic" in futures:
            features, prov = futures["acoustic"].result()
            builder.set_acoustics(features, prov)

        evidence = builder.build()

        # 6. Generate Executive Summary with Local LLM
        llm_prov: LLMProvider = self._resolve_provider("llm")
        context_prompt = (
            "You are an objective local audio intelligence assistant. "
            "Based strictly on the provided structured audio evidence, write a concise executive summary "
            "highlighting speakers, key statements, sound events, and acoustic properties.\n\n"
            f"{builder.format_llm_context(evidence)}"
        )
        summary = llm_prov.generate(context_prompt)
        evidence.summary = summary

        # 7. Persist analysis to database
        self.db.save_analysis(evidence)

        # 8. Check auto-delete audio retention policy
        if self.config.privacy.retention.auto_delete_audio_post_analysis:
            if validated_path.exists():
                try:
                    validated_path.unlink()
                except OSError:
                    pass

        return evidence

    def ask_question(self, recording_id: str, question: str) -> str:
        """Answers a question about an analyzed recording grounded strictly in the evidence."""
        evidence = self.db.get_analysis(recording_id)
        if not evidence:
            raise AudioProcessingError(f"No analysis found for recording ID '{recording_id}'.")

        llm_prov: LLMProvider = self._resolve_provider("llm")
        context = EvidenceBuilder.format_llm_context(evidence)
        prompt = (
            "You are an objective, grounded audio intelligence agent.\n"
            "Answer the user's question relying ONLY on the facts, timestamps, speaker attributions, "
            "sound events, and acoustic features present in the evidence below.\n"
            "If the answer cannot be determined from the evidence, state clearly that the evidence does not provide that information.\n"
            "Do not invent facts, confidence values, or unverified events.\n\n"
            f"{context}\n\n"
            f"User Question: {question}\n"
            "Grounded Answer:"
        )
        return llm_prov.generate(prompt)

    def ask_question_stream(self, recording_id: str, question: str) -> Iterator[str]:
        """Streams grounded answer tokens about an analyzed recording."""
        evidence = self.db.get_analysis(recording_id)
        if not evidence:
            raise AudioProcessingError(f"No analysis found for recording ID '{recording_id}'.")

        llm_prov: LLMProvider = self._resolve_provider("llm")
        context = EvidenceBuilder.format_llm_context(evidence)
        prompt = (
            "You are an objective, grounded audio intelligence agent.\n"
            "Answer the user's question relying ONLY on the facts, timestamps, speaker attributions, "
            "sound events, and acoustic features present in the evidence below.\n"
            "If the answer cannot be determined from the evidence, state clearly that the evidence does not provide that information.\n"
            "Do not invent facts, confidence values, or unverified events.\n\n"
            f"{context}\n\n"
            f"User Question: {question}\n"
            "Grounded Answer:"
        )
        return llm_prov.generate_stream(prompt)

    def speak_stream(
        self,
        token_stream: Iterator[str],
        play_audio: bool = True,
    ) -> Iterator[Tuple[Any, int]]:
        """Consumes a stream of tokens, synthesizes natural speech in real-time, and plays it."""
        from audio_agent.audio.player import AudioStreamPlayer

        tts_prov: TTSProvider = self._resolve_provider("tts")
        player = AudioStreamPlayer(sample_rate=tts_prov.sample_rate) if play_audio else None

        if player:
            player.start()

        try:
            for audio_chunk, sr in tts_prov.stream_synthesize(token_stream):
                if player:
                    player.enqueue(audio_chunk)
                yield audio_chunk, sr
        finally:
            if player:
                player.finish(wait=True)

    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Path] = None,
        play_audio: bool = False,
    ) -> Any:
        """Synthesizes text to audio and optionally plays or saves to file."""
        from audio_agent.audio.player import AudioStreamPlayer

        tts_prov: TTSProvider = self._resolve_provider("tts")
        samples = tts_prov.synthesize(text, output_path=output_path)

        if play_audio and samples is not None and len(samples) > 0:
            player = AudioStreamPlayer(sample_rate=tts_prov.sample_rate)
            player.start()
            player.enqueue(samples)
            player.finish(wait=True)

        return samples


