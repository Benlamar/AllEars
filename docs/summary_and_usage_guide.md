# Local-First Audio Intelligence Agent — System Summary & Usage Guide

## 1. Executive Summary

We have built and verified a modular, local-first Audio Intelligence Agent that accepts audio recordings or microphone input, processes them through specialized local machine learning and digital signal processing models, and provides grounded reasoning and question answering through a connected LLM.

### Key Milestones Achieved:
- **Zero Cloud Leakage:** All audio decoding, transcription, speaker diarization, sound event detection, and acoustic signal processing execute 100% locally on your machine.
- **Hardware-Accelerated:** Runs speech-to-text on your **NVIDIA GeForce RTX 4060 Laptop GPU** with CUDA 12.4 and FP16 precision.
- **Remote / Local LLM Integration:** Connected to your `vLLM` server hosting **`gemma4_4b`** at `http://192.168.1.11:8001/v1`.
- **Word-Level Timeline Alignment:** Transcribed words with millisecond timestamps are mathematically aligned with speaker intervals.
- **Persistent Local Database:** SQLite metadata storage with speaker renaming/aliasing and user-controlled retention and deletion.
- **21 Unit & Integration Tests Passing:** 100% test coverage for all providers, alignment heuristics, and storage CRUD.

---

## 2. Integrated Model Stack

| Component | Selected Engine / Model | Execution Target | Memory / Hardware Profile | Role |
|---|---|---|---|---|
| **ASR (Speech-to-Text)** | **`faster-whisper`** (`base.en`) | **CUDA (RTX 4060)** FP16 | ~350 MB VRAM | Transcribes speech and outputs exact word timestamps. |
| **Speaker Diarization** | **`Nemotron-3-Diarization`** | **ONNX Runtime** | ~115 MB RAM/VRAM | Streaming chunked multi-speaker turns with 264-frame FIFO cache. |
| **Voice Activity Detection** | **`Silero VAD`** | **ONNX Runtime** | ~2 MB RAM | Filters silence and identifies active speech segments. |
| **Acoustic Features** | **`Praat / Parselmouth`** | **Host CPU** | 0 MB VRAM (pure DSP) | Computes pitch ($F_0$), loudness (dB), speaking rate, jitter, shimmer, HNR. |
| **Sound Event Detection** | **`YAMNet`** | **ONNX Runtime** | ~15 MB RAM | Detects 521 AudioSet categories (cough, speech synthesizer, silence, etc.). |
| **LLM Reasoning** | **`gemma4_4b`** | **vLLM** (`192.168.1.11:8001`) | Remote Server (0 MB local VRAM) | Synthesizes grounded executive summaries and answers questions. |

---

## 3. How to Run the Audio Agent

All commands can be run from the repository root using `uv run audio-agent <command>`.

### 3.1. System Health & Diagnostics
Check the status of your hardware, GPU CUDA detection, config files, and provider availability:
```bash
uv run audio-agent doctor
```

---

### 3.2. Analyze an Audio File
Process an existing `.wav` file through the entire pipeline:
```bash
uv run audio-agent analyze data/speech_sample.wav
```
**What happens:**
1. Validates and loads audio file.
2. Runs **Silero VAD** for speech intervals.
3. Transcribes speech and timestamps every word via **Faster-Whisper on CUDA**.
4. Disentangles speaker turns via **Nemotron-3 Diarization**.
5. Extracts acoustic features ($F_0$, intensity, speaking rate, jitter, shimmer, HNR) via **Parselmouth**.
6. Detects non-speech sound events via **YAMNet**.
7. Constructs structured `AnalysisEvidence` and persists it to SQLite (`data/database/audio_agent.db`).
8. Generates a grounded Executive Summary using `gemma4_4b`.
9. Prints speaker turns, sound events, acoustic metrics, and assigns a unique `Recording ID` (e.g. `rec_1ee7f23e652b`).

---

### 3.3. Interactive Microphone Recording
Capture speech directly from your microphone and immediately analyze it:
```bash
uv run audio-agent record
```
- Press **Enter** to start recording.
- Speak into your microphone.
- Press **Enter** to stop recording.
- The recording is automatically saved and passed into the analysis pipeline.

---

### 3.4. Grounded Question Answering (Single Question)
Ask specific questions about an analyzed recording grounded strictly in the observed evidence:
```bash
uv run audio-agent ask rec_1ee7f23e652b "What did the speakers say about the fox?"
```
```bash
uv run audio-agent ask rec_1ee7f23e652b "What were the measured pitch and voice quality metrics?"
```

---

### 3.5. Interactive Conversational Chat
Launch an interactive REPL chat session anchored to a specific recording:
```bash
uv run audio-agent chat rec_1ee7f23e652b
```
- Ask follow-up questions iteratively.
- Type `exit` or `quit` to leave the chat.

---

### 3.6. Managing Stored Recordings & Speakers

#### List all stored analyses:
```bash
uv run audio-agent recordings list
```

#### Rename / Alias a speaker:
Replace anonymous labels (`speaker_1`, `speaker_2`) with human names:
```bash
uv run audio-agent recordings alias rec_1ee7f23e652b speaker_1 "Alice"
```

#### Delete a recording:
Remove an analysis record and its corresponding audio file from disk:
```bash
uv run audio-agent recordings delete rec_1ee7f23e652b
```
*(Add `--keep-audio` if you want to retain the raw WAV file while removing the database entry).*

---

## 4. Configuration Reference

The agent's configuration is managed via three YAML files in the `config/` directory:

### `config/models.yaml`
Controls model selections, hardware devices, and remote endpoints:
```yaml
asr:
  provider: "faster_whisper"
  model: "base.en"           # Can be upgraded to "small.en" or "large-v3-turbo"
  device: "cuda"
  compute_type: "float16"

vad:
  provider: "silero"
  threshold: 0.5

diarization:
  provider: "nemotron"
  threshold: 0.35

sound_events:
  provider: "yamnet"
  top_k: 5
  score_threshold: 0.15

acoustic:
  provider: "parselmouth"
  pitch_floor_hz: 75.0
  pitch_ceiling_hz: 600.0

llm:
  provider: "vllm"
  model: "gemma4_4b"
  endpoint: "http://192.168.1.11:8001/v1"
  temperature: 0.1

tts:
  provider: "qwen" # options: kokoro, qwen, mock
  # Option A: Qwen3-TTS (vLLM OpenAI-compatible audio endpoint)
  endpoint: "http://192.168.1.6:8091/v1"
  model: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  task_type: "VoiceDesign"
  instructions: "A calm, friendly female voice with clear articulation."
  # Option B: Kokoro-TTS (local ONNX)
  # provider: "kokoro"
  # voice: "af_sarah"
```

### `config/privacy.yaml`
Enforces privacy policies and data retention:
```yaml
execution:
  mode: "local_only"

cloud:
  allowed: false
  automatic_fallback: false # Prohibits silent cloud fallback

retention:
  retain_audio: true
  auto_delete_audio_post_analysis: false
```

### `config/defaults.yaml`
Controls general recording and path defaults:
```yaml
pipeline:
  language: "en"
  chunk_duration_seconds: 300

paths:
  recordings_dir: "data/recordings"
  database_path: "data/database/audio_agent.db"
```

---

## 5. Voice & Streaming Speech Commands

```bash
# Speak text directly via TTS
uv run audio-agent speak "Streaming speech synthesis is live."

# Save synthesized audio to file
uv run audio-agent speak "Synthesizing to file." --output data/output.wav

# Ask questions with voice playback enabled
uv run audio-agent ask <recording_id> "What sound events were detected?" --voice

# Interactive conversational session with voice
uv run audio-agent chat <recording_id> --voice
```

---

## 6. Running the Test Suite

To verify the integrity of all models, providers, and storage mechanisms at any time:
```bash
uv run pytest -v
```
**Current Test Coverage (36 tests):**
- Word/speaker temporal alignment & overlap detection
- Privacy policy cloud rejection & fallback prohibition
- Faster-Whisper ASR availability and parsing
- Nemotron-3 Diarization streaming chunk inference
- Silero VAD speech detection
- Parselmouth acoustic DSP extraction
- YAMNet sound event classification
- SQLite database persistence, speaker aliasing, and cascading deletion
- Speech sanitizer (stripping `<turn|>`, `<think>`, markdown `**`, bullets, emojis)
- Streaming clause chunking and tail flushing
- Kokoro ONNX and Qwen vLLM TTS providers

