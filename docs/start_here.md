# START HERE — Agent & Developer Navigation Guide

> **For Future AI Agents & Developers:**  
> Read this document **first**. It provides a direct map of the codebase, pinpointing the exact files, configurations, and isolated test commands for each individual pipeline component. **Do not read irrelevant modules**; use the component guide below to conserve context tokens and make targeted modifications.

---

## 1. Documentation Index

- **Full Architecture & Usage Manual:** [`docs/summary_and_usage_guide.md`](file:///c:/Users/verge/SourceMaker/AudioAgent/docs/summary_and_usage_guide.md)
- **Original Implementation Plan:** [`docs/local_first_audio_intelligence_agent_implementation_plan.md`](file:///c:/Users/verge/SourceMaker/AudioAgent/docs/local_first_audio_intelligence_agent_implementation_plan.md)

---

## 2. Fast Architecture Overview

The system is a **modular monolith** with capability-based provider adapters. The orchestrator depends only on abstract interfaces, allowing any model or provider to be modified, replaced, or debugged in total isolation.

```text
src/audio_agent/
├── cli.py                     # Typer CLI entrypoint & commands
├── orchestrator.py            # Pipeline coordinator & LLM prompt assembler
├── schemas.py                 # Core Pydantic v2 data models (AnalysisEvidence, Word, etc.)
├── config.py                  # YAML loader with Pydantic validation
│
├── providers/                 # Pluggable ML/DSP adapters
│   ├── base.py                # Abstract Base Classes (ASRProvider, DiarizationProvider, etc.)
│   ├── registry.py            # Central registry mapping (capability, name) -> Provider Class
│   ├── asr/faster_whisper.py  # Faster-Whisper on CUDA (FP16, word timestamps)
│   ├── diarization/nemotron.py# Nemotron-3 Diarization ONNX (streaming chunked, FIFO cache)
│   ├── vad/silero.py          # Silero VAD ONNX (speech region detection)
│   ├── sound_events/yamnet.py # YAMNet ONNX (521 AudioSet sound classes)
│   ├── acoustic/parselmouth_extractor.py # Praat/Parselmouth DSP (pitch, intensity, voice quality)
│   └── llm.py                 # OpenAI-compatible client for remote vLLM (gemma4_4b)
│
├── analysis/
│   ├── alignment.py           # Matches ASR words with Diarization speaker intervals
│   └── evidence_builder.py    # Formats structured evidence into LLM prompt context
│
├── storage/
│   └── database.py            # SQLite persistence (analyses, recordings, speaker aliases)
└── security/
    └── privacy_policy.py      # Strictly enforces zero unapproved cloud calls
```

---

## 3. Targeted Debugging & Extension Cheat Sheet

When working on or debugging a specific feature, **inspect only the files listed for that feature**.

---

### A. Automatic Speech Recognition (ASR)
- **Primary Code:** [`src/audio_agent/providers/asr/faster_whisper.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/asr/faster_whisper.py)
- **Config Key:** `asr:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_asr_provider.py -v
  ```
- **Quick Verification Script:**
  ```bash
  uv run python -c "from audio_agent.providers.asr import FasterWhisperASRProvider; p = FasterWhisperASRProvider(); t, w = p.transcribe('data/speech_sample.wav'); print('Transcript:', t); print('Words count:', len(w))"
  ```
- **Key Notes:** Runs on CUDA with `float16`. Uses `vad_filter=True` to ignore pure silence. Outputs `List[Word]` with `start`, `end`, and `confidence`.

---

### B. Speaker Diarization
- **Primary Code:** [`src/audio_agent/providers/diarization/nemotron.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/diarization/nemotron.py)
- **Config Key:** `diarization:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Model Files:** `models/nemotron_diarization/onnx/model_quantized.onnx` (~115 MB)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_diarization_provider.py -v
  ```
- **Quick Verification Script:**
  ```bash
  uv run python -c "from audio_agent.providers.diarization import NemotronDiarizationProvider; p = NemotronDiarizationProvider(); segs = p.diarize('data/speech_sample.wav'); print([(s.speaker_id, s.start, s.end) for s in segs])"
  ```
- **Key Notes:** Uses a streaming chunk size of 384 Mel frames with 340 stride. Preserves a 264-frame FIFO `cached_embeds` across chunks.

---

### C. Voice Activity Detection (VAD)
- **Primary Code:** [`src/audio_agent/providers/vad/silero.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/vad/silero.py)
- **Config Key:** `vad:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_vad_provider.py -v
  ```
- **Quick Verification Script:**
  ```bash
  uv run python -c "from audio_agent.providers.vad import SileroVADProvider; p = SileroVADProvider(); print(p.detect_speech_regions('data/speech_sample.wav'))"
  ```
- **Key Notes:** Input audio is automatically resampled to 16 kHz mono using `scipy.signal.resample` before ONNX inference.

---

### D. Sound Event Detection (SED)
- **Primary Code:** [`src/audio_agent/providers/sound_events/yamnet.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/sound_events/yamnet.py)
- **Config Key:** `sound_events:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_sound_events_provider.py -v
  ```
- **Quick Verification Script:**
  ```bash
  uv run python -c "from audio_agent.providers.sound_events import YAMNetSoundEventProvider; p = YAMNetSoundEventProvider(); print([(e.label, e.start, e.end, e.score) for e in p.detect_events('data/speech_sample.wav')])"
  ```
- **Key Notes:** Evaluates 521 AudioSet sound classes. Automatically merges contiguous frames with the same label.

---

### E. Acoustic Feature Extraction (DSP)
- **Primary Code:** [`src/audio_agent/providers/acoustic/parselmouth_extractor.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/acoustic/parselmouth_extractor.py)
- **Config Key:** `acoustic:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_acoustic_provider.py -v
  ```
- **Quick Verification Script:**
  ```bash
  uv run python -c "from audio_agent.providers.acoustic import ParselmouthAcousticProvider; p = ParselmouthAcousticProvider(); f = p.extract_features('data/speech_sample.wav'); print(f.model_dump())"
  ```
- **Key Notes:** Runs Praat C algorithms on CPU (0 MB VRAM). Computes pitch ($F_0$), intensity (dB), speaking rate (syll/s), jitter, shimmer, and Harmonics-to-Noise Ratio (HNR).

---

### F. Local / Remote LLM Reasoning
- **Primary Code:** [`src/audio_agent/providers/llm.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/llm.py)
- **Config Key:** `llm:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Current Endpoint:** `http://192.168.1.11:8001/v1` (`gemma4_4b` hosted on vLLM)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_llm_provider.py -v
  ```
- **Quick Connectivity Check:**
  ```bash
  uv run python -c "from audio_agent.providers.llm import OpenAICompatibleLLMProvider; p = OpenAICompatibleLLMProvider(endpoint='http://192.168.1.11:8001/v1', model_name='gemma4_4b'); print('Connected:', p.is_available()); print('Response:', p.generate('Say test'))"
  ```

---

### G. Timeline Alignment (Words to Speakers)
- **Primary Code:** [`src/audio_agent/analysis/alignment.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/analysis/alignment.py)
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_alignment.py -v
  ```
- **Key Notes:** Aligns ASR word timestamps with diarization intervals. If overlapping speech is present, it assigns the best match and records an `UncertaintyNotice` instead of failing or hallucinating.

---

### H. SQLite Database & Storage
- **Primary Code:** [`src/audio_agent/storage/database.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/storage/database.py)
- **Database File:** `data/database/audio_agent.db`
- **Isolated Unit Test:**
  ```bash
  uv run pytest tests/unit/test_database.py -v
  ```
- **Key Notes:** Handles CRUD operations, speaker alias mapping, and cascading deletions.

---

### I. CLI & Orchestration
- **CLI Commands:** [`src/audio_agent/cli.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/cli.py)
- **Orchestrator Pipeline:** [`src/audio_agent/orchestrator.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/orchestrator.py)
- **End-to-End Test:**
  ```bash
  uv run pytest tests/integration/test_orchestrator.py -v
  ```

---

### J. Streaming Text-to-Speech (TTS) & Speech Sanitizer
- **TTS Providers:**
  - [`src/audio_agent/providers/tts/kokoro.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/tts/kokoro.py) (Kokoro-v0.19 82M ONNX local CPU/CUDA)
  - [`src/audio_agent/providers/tts/qwen.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/providers/tts/qwen.py) (Qwen3-TTS 1.7B VoiceDesign via local vLLM HTTP API at `http://192.168.1.6:8091/v1`)
- **Speech Text Sanitizer:** [`src/audio_agent/audio/sanitizer.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/audio/sanitizer.py) (cleans `<turn|>`, `<think>`, markdown `**`, bullets, emojis)
- **Streaming Chunker:** [`src/audio_agent/audio/chunker.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/audio/chunker.py)
- **Audio Stream Player:** [`src/audio_agent/audio/player.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/audio/player.py)
- **Config Key:** `tts:` in [`config/models.yaml`](file:///c:/Users/verge/SourceMaker/AudioAgent/config/models.yaml)
- **Isolated Unit Tests:**
  ```bash
  uv run pytest tests/unit/test_speech_sanitizer.py -v
  uv run pytest tests/unit/test_streaming_chunker.py -v
  uv run pytest tests/unit/test_tts_provider.py -v
  ```
- **CLI Voice Commands:**
  ```bash
  uv run audio-agent speak "Speech test without control tags."
  uv run audio-agent ask <recording_id> "What was discussed?" --voice
  uv run audio-agent chat <recording_id> --voice
  ```

---

### K. Full-Duplex Live Voice & 3D Audio-Reactive Sphere TUI
- **Live Session Engine:** [`src/audio_agent/audio/live_session.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/audio/live_session.py)
  - Hands-free pause endpointing (~1.8s silence) using streaming Silero VAD frames.
  - In-memory Faster-Whisper ASR and streaming LLM / TTS.
  - Sub-30ms barge-in interruption (`player.stop()` + LLM cancellation event).
  - Acoustic non-speech filter via YAMNet to reject coughs, chuckles, laughter, and throat-clearing.
- **3D Sphere TUI Application:** [`src/audio_agent/tui/live_sphere_app.py`](file:///c:/Users/verge/SourceMaker/AudioAgent/src/audio_agent/tui/live_sphere_app.py)
  - Interactive ASCII shaded rotating sphere reacting to FFT frequency bands & RMS loudness.
  - Dynamic state themes: Aurora (listening), Spectrum (user speaking), Matrix (thinking), Ember (agent speaking), and Shockwave rings on interruption.
- **Isolated Unit Tests:**
  ```bash
  uv run pytest tests/unit/test_live_session.py -v
  ```
- **CLI Live Commands:**
  ```bash
  # Launch live voice conversation with 3D Sphere TUI
  uv run audio-agent live

  # Custom pause silence threshold (e.g. 2.0 seconds)
  uv run audio-agent live --pause 2.0

  # Console-only streaming mode without TUI
  uv run audio-agent live --no-tui
  ```

---

## 4. Full Test Suite Command

Before finishing any change, run the entire test suite:
```bash
uv run pytest -v
```
Ensure all **40 tests** pass without regressions.



