"""Nemotron-3 Diarization provider using ONNX Runtime with streaming chunk support."""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import scipy.signal as signal
import soundfile as sf

from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import DiarizationProvider
from audio_agent.schemas import SpeakerSegment


class NemotronDiarizationProvider(DiarizationProvider):
    """Local Speaker Diarization using NVIDIA Nemotron-3 Diarization ONNX model."""

    provider_name: str = "nemotron"
    execution_mode: str = "local"

    def __init__(
        self,
        model_dir: str = "models/nemotron_diarization",
        repo_id: str = "onnx-community/Nemotron-3-Diarization-ONNX",
        threshold: float = 0.35,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.25,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.repo_id = repo_id
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.device = device
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from huggingface_hub import snapshot_download

                onnx_path = self.model_dir / "onnx" / "model_quantized.onnx"
                if not onnx_path.exists():
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        self.repo_id,
                        local_dir=str(self.model_dir),
                        allow_patterns=["*quantized*"],
                    )

                providers = ["CPUExecutionProvider"]
                if (self.device == "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 2
                sess_options.inter_op_num_threads = 1
                self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
            except Exception as e:
                raise ProviderError(f"Failed to load Nemotron Diarization ONNX model: {e}") from e

        return self._session

    def is_available(self) -> bool:
        try:
            import onnxruntime
            import huggingface_hub
            import librosa
            return True
        except ImportError:
            return False

    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        session = self._get_session()

        try:
            import librosa

            wav, sr = sf.read(str(path))
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)

            target_sr = 16000
            if sr != target_sr:
                num_samples = int(len(wav) * target_sr / sr)
                wav = signal.resample(wav, num_samples).astype(np.float32)
            else:
                wav = wav.astype(np.float32)

            total_duration = len(wav) / target_sr
            if total_duration < 0.2:
                return []

            # 1. Feature Extraction: 128-dim log Mel filterbank
            wav_pre = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
            mel_basis = librosa.filters.mel(sr=target_sr, n_fft=512, n_mels=128)
            stft = np.abs(librosa.stft(wav_pre, n_fft=512, hop_length=160, win_length=400, window="hann"))
            log_mel = np.log(np.dot(mel_basis, stft) + 1e-5).T
            # Mean-variance normalization
            std = np.std(log_mel, axis=0) + 1e-5
            log_mel = (log_mel - np.mean(log_mel, axis=0)) / std

            total_mel_frames = log_mel.shape[0]

            # 2. Streaming Chunk Inference
            chunk_len = 384 # 3.84s window
            stride = 340    # 3.40s advance
            cached_len = 264
            subsample = 8

            cached_embeds = np.zeros((1, cached_len, 512), dtype=np.float32)
            speaker_probs_timeline = []

            for start_frame in range(0, total_mel_frames, stride):
                end_frame = min(start_frame + chunk_len, total_mel_frames)
                chunk = log_mel[start_frame:end_frame]
                curr_chunk_len = chunk.shape[0]

                if curr_chunk_len < chunk_len:
                    # Pad to chunk_len
                    chunk = np.pad(chunk, ((0, chunk_len - curr_chunk_len), (0, 0)))

                chunk_input = np.expand_dims(chunk, axis=0).astype(np.float32)
                mask = np.ones((1, cached_len + chunk_len // subsample), dtype=np.int64)

                outputs = session.run(
                    None,
                    {
                        "input_features": chunk_input,
                        "cached_embeds": cached_embeds,
                        "attention_mask": mask,
                    },
                )
                logits = outputs[0][0] # (num_logit_frames, 8)
                chunk_embeds = outputs[1]
                # Update FIFO cache (roll and append new frames to preserve 264 length)
                new_frames_count = chunk_embeds.shape[1]
                cached_embeds = np.concatenate([cached_embeds[:, new_frames_count:, :], chunk_embeds], axis=1)

                # Sigmoid probabilities
                chunk_probs = 1.0 / (1.0 + np.exp(-logits))

                # Number of valid logit frames corresponding to curr_chunk_len
                valid_logits_count = int((curr_chunk_len / chunk_len) * chunk_probs.shape[0])
                stride_logits_count = int((min(stride, curr_chunk_len) / chunk_len) * chunk_probs.shape[0])

                if start_frame + stride >= total_mel_frames:
                    # Last chunk: take all valid logits
                    speaker_probs_timeline.append(chunk_probs[:valid_logits_count])
                else:
                    # Intermediate chunk: take stride logits
                    speaker_probs_timeline.append(chunk_probs[:stride_logits_count])

            if not speaker_probs_timeline:
                return []

            all_probs = np.concatenate(speaker_probs_timeline, axis=0)
            total_prob_frames = all_probs.shape[0]
            sec_per_frame = total_duration / max(1, total_prob_frames)

            # 3. Segment Extraction per Speaker
            segments: List[SpeakerSegment] = []
            max_speakers = num_speakers or 8

            # Sort speakers by maximum activity
            speaker_peaks = [(s, float(np.max(all_probs[:, s]))) for s in range(8)]
            speaker_peaks.sort(key=lambda x: x[1], reverse=True)

            active_speaker_indices = [
                s for s, peak in speaker_peaks[:max_speakers]
                if peak >= self.threshold
            ]

            if not active_speaker_indices:
                # Default to top speaker if speech exists
                active_speaker_indices = [speaker_peaks[0][0]]

            # Map internal model speaker index to clean 1-based speaker_id
            speaker_label_map = {orig: f"speaker_{idx + 1}" for idx, orig in enumerate(active_speaker_indices)}

            for orig_idx, spk_label in speaker_label_map.items():
                probs_s = all_probs[:, orig_idx]
                is_active = probs_s >= self.threshold

                # Find continuous intervals
                in_seg = False
                seg_start = 0.0
                seg_scores = []

                for i, active in enumerate(is_active):
                    t = i * sec_per_frame
                    if active and not in_seg:
                        in_seg = True
                        seg_start = t
                        seg_scores = [probs_s[i]]
                    elif active and in_seg:
                        seg_scores.append(probs_s[i])
                    elif not active and in_seg:
                        in_seg = False
                        seg_end = t
                        if (seg_end - seg_start) >= self.min_speech_duration:
                            segments.append(
                                SpeakerSegment(
                                    speaker_id=spk_label,
                                    start=round(seg_start, 2),
                                    end=round(seg_end, 2),
                                    confidence=round(float(np.mean(seg_scores)), 2),
                                )
                            )

                if in_seg:
                    seg_end = total_duration
                    if (seg_end - seg_start) >= self.min_speech_duration:
                        segments.append(
                            SpeakerSegment(
                                speaker_id=spk_label,
                                start=round(seg_start, 2),
                                end=round(seg_end, 2),
                                confidence=round(float(np.mean(seg_scores)), 2),
                            )
                        )

            # 4. Bridge tiny gaps (< min_silence_duration) for the same speaker
            segments.sort(key=lambda x: (x.speaker_id, x.start))
            merged_segments: List[SpeakerSegment] = []

            for seg in segments:
                if not merged_segments:
                    merged_segments.append(seg)
                    continue

                prev = merged_segments[-1]
                if prev.speaker_id == seg.speaker_id and (seg.start - prev.end) <= self.min_silence_duration:
                    # Merge
                    prev_conf = prev.confidence or 0.5
                    curr_conf = seg.confidence or 0.5
                    merged_segments[-1] = SpeakerSegment(
                        speaker_id=prev.speaker_id,
                        start=prev.start,
                        end=seg.end,
                        confidence=round((prev_conf + curr_conf) / 2.0, 2),
                    )
                else:
                    merged_segments.append(seg)

            # Sort globally by start time
            merged_segments.sort(key=lambda x: x.start)
            return merged_segments

        except Exception as e:
            raise ProviderError(f"Nemotron diarization processing failed: {e}") from e

