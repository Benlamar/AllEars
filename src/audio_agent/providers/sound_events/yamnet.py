"""YAMNet Sound Event Detection (SED) using ONNX Runtime and AudioSet ontology."""

import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import scipy.signal as signal
import soundfile as sf

from audio_agent.exceptions import AudioProcessingError, ProviderError
from audio_agent.providers.base import SoundEventProvider
from audio_agent.schemas import SoundEvent


class YAMNetSoundEventProvider(SoundEventProvider):
    """Detects 521 AudioSet sound classes (Cough, Laughter, Speech, Music, etc.) using YAMNet ONNX."""

    provider_name: str = "yamnet"
    execution_mode: str = "local"

    def __init__(
        self,
        repo_id: str = "zeropointnine/yamnet-onnx",
        score_threshold: float = 0.15,
        top_k: int = 5,
        device: str = "cuda",
    ):
        self.repo_id = repo_id
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.device = device
        self._session = None
        self._labels: List[str] = []

    def _get_session_and_labels(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(self.repo_id, "yamnet.onnx")
                class_map_path = hf_hub_download(self.repo_id, "yamnet_class_map.csv")

                # Load labels
                labels = []
                with open(class_map_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader) # skip header
                    for row in reader:
                        if len(row) >= 3:
                            labels.append(row[2].strip())
                self._labels = labels

                # Initialize ONNX inference session with CUDA provider if requested
                providers = ["CPUExecutionProvider"]
                if (self.device == "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 2
                sess_options.inter_op_num_threads = 1
                self._session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            except Exception as e:
                raise ProviderError(f"Failed to load YAMNet ONNX model from {self.repo_id}: {e}") from e

        return self._session, self._labels

    def is_available(self) -> bool:
        try:
            import onnxruntime
            import huggingface_hub
            return True
        except ImportError:
            return False

    def detect_events(self, audio_path: Path) -> List[SoundEvent]:
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        session, labels = self._get_session_and_labels()

        try:
            wav, sr = sf.read(str(path))
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)

            target_sr = 16000
            if sr != target_sr:
                num_samples = int(len(wav) * target_sr / sr)
                wav = signal.resample(wav, num_samples).astype(np.float32)
            else:
                wav = wav.astype(np.float32)

            if len(wav) < target_sr * 0.5:
                # Pad to at least 0.5s if too short
                wav = np.pad(wav, (0, int(target_sr * 0.5) - len(wav)))

            # YAMNet inference: inputs: {'waveform': wav} -> outputs: [scores, embeddings, spectrogram]
            scores, _, _ = session.run(None, {"waveform": wav})
            num_frames = scores.shape[0]

            # YAMNet uses 0.96s window with 0.48s hop
            hop_size = 0.48
            window_size = 0.96

            raw_events: List[Dict] = []
            for i in range(num_frames):
                frame_start = round(i * hop_size, 2)
                frame_end = round(frame_start + window_size, 2)
                frame_scores = scores[i]

                # Top scores above threshold
                top_idx = np.argsort(frame_scores)[::-1][: self.top_k]
                for idx in top_idx:
                    score = float(frame_scores[idx])
                    if score >= self.score_threshold:
                        raw_events.append({
                            "label": labels[idx],
                            "start": frame_start,
                            "end": frame_end,
                            "score": score,
                        })

            if not raw_events:
                # If nothing exceeded threshold, pick top overall class
                mean_scores = np.mean(scores, axis=0)
                top_idx = int(np.argmax(mean_scores))
                return [
                    SoundEvent(
                        label=labels[top_idx],
                        start=0.0,
                        end=round(num_frames * hop_size, 2),
                        score=round(float(mean_scores[top_idx]), 2),
                        description="Dominant acoustic background class",
                    )
                ]

            # Merge contiguous frames with the same label
            merged_events: List[SoundEvent] = []
            # Group by label
            from collections import defaultdict
            label_groups = defaultdict(list)
            for ev in raw_events:
                label_groups[ev["label"]].append(ev)

            for label, evs in label_groups.items():
                evs.sort(key=lambda x: x["start"])
                curr_start = evs[0]["start"]
                curr_end = evs[0]["end"]
                curr_scores = [evs[0]["score"]]

                for next_ev in evs[1:]:
                    if next_ev["start"] <= curr_end + 0.1: # contiguous or overlapping
                        curr_end = max(curr_end, next_ev["end"])
                        curr_scores.append(next_ev["score"])
                    else:
                        merged_events.append(
                            SoundEvent(
                                label=label,
                                start=curr_start,
                                end=curr_end,
                                score=round(float(np.mean(curr_scores)), 2),
                            )
                        )
                        curr_start = next_ev["start"]
                        curr_end = next_ev["end"]
                        curr_scores = [next_ev["score"]]

                merged_events.append(
                    SoundEvent(
                        label=label,
                        start=curr_start,
                        end=curr_end,
                        score=round(float(np.mean(curr_scores)), 2),
                    )
                )

            # Sort by start time
            merged_events.sort(key=lambda x: (x.start, -x.score))
            return merged_events

        except Exception as e:
            raise ProviderError(f"YAMNet sound event detection failed: {e}") from e

