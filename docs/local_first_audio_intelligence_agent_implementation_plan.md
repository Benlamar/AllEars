# Local-First Audio Intelligence Agent
## Implementation Plan & Brainstorming Document

**Status:** Draft — intended for brainstorming and iterative refinement  
**Implementation approach:** Modular Python application, CLI first  
**Primary goal:** Analyze audio locally, identify speakers, detect sound events, extract acoustic features, and let an LLM reason over the resulting evidence.

---

## 1. Vision and product scope

Build a local-first audio intelligence agent that accepts text and audio, analyzes audio with specialized components, and answers questions using structured evidence.

### Input modes

- **Text:** Ask a question or give an instruction to the agent.
- **Audio file:** Analyze an existing recording.
- **Microphone:** Record only after the user explicitly starts recording; stop when the user requests it or a configured limit is reached.

### Target capabilities

- Speech-to-text transcription.
- Word or segment timestamps where supported.
- Multi-speaker diarization: identify distinct speakers within a recording and estimate when they speak.
- Speaker-attributed transcription.
- Optional cross-recording speaker matching using explicitly enrolled profiles.
- Sound-event detection, such as coughing, laughter, throat clearing, and other useful events.
- Acoustic feature extraction: pitch, loudness-related measurements, speaking rate, pauses, and selected voice-quality measures.
- LLM-powered questions and summaries grounded in the analyzed evidence.
- Optional text-to-speech (TTS).
- Local persistence for analysis results and user-controlled recording retention.

### Non-goals for the first release

- Always-on microphone listening.
- Real-time streaming analysis.
- Automatic medical interpretation of coughs or other sounds.
- Emotion recognition presented as fact.
- Voice cloning.
- Distributed microservices.
- A web frontend before the core pipeline works.
- Automatic cloud fallback.

---

## 2. Product and privacy principles

1. **Local by default.** Processing stays on the user's machine unless a specific cloud route is explicitly enabled.
2. **No silent cloud fallback.** If a local provider fails, return an error unless the user has explicitly configured an allowed alternative.
3. **Explicit recording.** The microphone is activated only by a deliberate user action.
4. **Replaceable providers.** The orchestrator depends on capability interfaces, not vendor-specific APIs.
5. **Evidence before reasoning.** Specialized components produce observations; the LLM reasons over those observations.
6. **Preserve uncertainty.** Don't invent confidence values or force uncertain speaker/event assignments.
7. **Keep diarization and identity distinct.** `speaker_1` is a within-recording label, not a real-world identity.
8. **Minimize retained data.** Make audio retention configurable; support deletion of audio, analyses, and enrolled profiles.
9. **Evaluate before adopting.** Model-card benchmarks are hypotheses; test models on representative audio.
10. **Avoid unnecessary complexity.** Start with a modular monolith and add a UI/API only when needed.

---

## 3. Initial technology choices

These are starting candidates, not permanent commitments. Verify runtime compatibility, model access, hardware requirements, and licensing before installation.

| Area | Initial choice/candidate | Notes |
|---|---|---|
| Language | Python | Core application and ML integrations |
| Environment/package manager | `uv` | Reproducible environment and lockfile |
| CLI | Typer | Commands and interactive flow |
| Audio decoding | FFmpeg | Decode common audio formats |
| Audio arrays/files | NumPy, SoundFile | Loading and manipulation |
| ASR | Cactus Whistle candidate | Benchmark against alternatives |
| ASR alternatives | Phonon-2, Parakeet Redux, others | Choose using target-language tests |
| Diarization | Nemotron-3-Diarization candidate | Check supported NeMo runtime and hardware |
| Diarization alternative/component | pyannote pipeline/components | Segmentation alone is not a complete diarization system |
| VAD | Silero VAD candidate | Optional; some pipelines handle segmentation internally |
| Sound events | YAMNet baseline | Validate cough/event performance; replace if inadequate |
| Acoustic analysis | librosa + Parselmouth | Feature extraction and measurements |
| Local LLM runtime | Ollama | Select the actual LLM after hardware assessment |
| Metadata storage | SQLite | Recordings, analysis metadata, model runs |
| Audio storage | Local filesystem | Retention configurable |
| Configuration | YAML + environment variables | Validate schema and provider policy |
| Tests | pytest | Unit, integration, evaluation tests |

**Do not install every model at once.** First establish the environment and test one capability at a time.

---

## 4. Architecture: modular monolith first

Initially, the CLI, orchestration, storage, and model adapters run within one Python application. Logical modules are separate; they do not need to be separate services.

```text
CLI
 ├── Text questions
 ├── Analyze audio file
 └── Explicit microphone recording
          |
          v
Agent Orchestrator
          |
          v
Audio validation / preprocessing
          |
     +----+----------+----------------+
     |               |                |
     v               v                v
    VAD             ASR          Diarization
     |               |                |
     +---------------+----------------+
                     |
                     v
            Timeline alignment
                     |
          +----------+-----------+
          |                      |
          v                      v
   Sound-event detection   Acoustic analysis
          |                      |
          +----------+-----------+
                     |
                     v
              Evidence builder
                     |
                     v
                Model router
                     |
                     v
             LLM / optional TTS
                     |
                     v
                CLI response

SQLite + local files support the application.
```

### Why this architecture?

- No frontend is required for the first version.
- No separate backend server is required for the first version.
- Models can be replaced independently.
- A future API or desktop UI can call the same core.
- Pipeline behavior can be tested without the UI.
- Local/cloud policies can be enforced centrally.

---

## 5. Proposed repository structure

Create files as milestones require them; this is the intended shape, not a requirement to scaffold everything immediately.

```text
audio-agent/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── .env.example
├── config/
│   ├── defaults.yaml
│   ├── models.yaml
│   └── privacy.yaml
├── src/
│   └── audio_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── orchestrator.py
│       ├── schemas.py
│       ├── exceptions.py
│       ├── audio/
│       │   ├── recorder.py
│       │   ├── loader.py
│       │   ├── preprocessing.py
│       │   └── validation.py
│       ├── providers/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── asr/
│       │   ├── vad/
│       │   ├── diarization/
│       │   ├── sound_events/
│       │   ├── llm/
│       │   └── tts/
│       ├── analysis/
│       │   ├── alignment.py
│       │   ├── acoustic_features.py
│       │   └── evidence_builder.py
│       ├── storage/
│       │   ├── database.py
│       │   ├── recordings.py
│       │   └── speakers.py
│       └── security/
│           └── privacy_policy.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── evaluation/
│   └── fixtures/
├── scripts/
│   ├── check_environment.py
│   └── benchmark_models.py
└── data/  # local only; exclude from Git
    ├── recordings/
    ├── analyses/
    └── database/
```

---

## 6. Configuration strategy

### Guiding rule

**Make the architecture highly configurable, but keep user-facing settings simple.** Developer experimentation and everyday user preferences are different needs.

### 6.1 Model/provider configuration

Example only; provider names and model identifiers must match implemented adapters.

```yaml
asr:
  provider: local
  model: whistle

vad:
  enabled: true
  provider: local
  model: silero-vad

diarization:
  provider: local
  model: nemotron

sound_events:
  enabled: true
  provider: local
  model: yamnet

llm:
  provider: ollama
  model: REPLACE_WITH_INSTALLED_MODEL

tts:
  enabled: false
  provider: local
  model: REPLACE_WITH_SELECTED_MODEL
```

Each capability should have an independent provider selection. Switching ASR must not require changing the orchestrator.

### 6.2 Behavior configuration

Potential settings:

- Maximum recording duration.
- Audio device.
- Language preference or auto-detection.
- Sound-event analysis enabled/disabled.
- Acoustic analysis enabled/disabled.
- Speaker-count hints (optional).
- Recording retention.
- Transcript and analysis retention.
- TTS enabled/disabled.
- Logging verbosity.

Let the application automatically convert audio to a provider's required format. Avoid making users configure sample rates unless they are troubleshooting.

### 6.3 Privacy configuration

```yaml
execution:
  default: local

cloud:
  allowed: false
  automatic_fallback: false

retention:
  audio: true
  transcripts: true
  analysis: true
```

Further design decisions:

- Should raw audio, transcript text, embeddings, and derived acoustic features have separate sharing permissions?
- Should a cloud LLM be allowed to receive a transcript while raw audio remains local?
- How should the application display the provider receiving each request?
- How should secrets be stored?
- Should diagnostic telemetry be disabled by default?

### 6.4 Operational configuration

Possible advanced settings:

- Compute device: auto/CPU/GPU.
- Maximum concurrent jobs.
- Timeouts.
- Thread counts.
- Model cache locations.
- Diagnostic log level.

Keep internal orchestration heuristics private until a real need for user control appears.

### 6.5 Three configuration layers

- `defaults.yaml`: ordinary behavior and safe defaults.
- `models.yaml`: model and provider selection.
- `privacy.yaml`: cloud policy, permissions, and retention.

Validate configuration at startup. Never commit credentials or private audio. Define precedence between defaults, local config, and environment variables.

---

## 7. Capability registry and provider contracts

The application should know what each configured provider can actually do.

A provider's metadata may include:

- Supported languages.
- Word/segment timestamp support.
- Streaming support.
- Local or remote execution.
- CPU/GPU/runtime requirements.
- Speaker-count limitations.
- Overlap support.
- Confidence information available.
- Required audio formats.
- License and model-access conditions.

Reject incompatible configurations early. For example, if word-level speaker attribution is required but the ASR provider returns no word timestamps, either configure an alignment component or report the limitation.

### Conceptual interfaces

```python
class ASRProvider:
    def transcribe(self, audio):
        raise NotImplementedError


class VADProvider:
    def detect_speech_regions(self, audio):
        raise NotImplementedError


class DiarizationProvider:
    def diarize(self, audio):
        raise NotImplementedError


class SoundEventProvider:
    def detect_events(self, audio):
        raise NotImplementedError


class LLMProvider:
    def generate(self, messages):
        raise NotImplementedError


class TTSProvider:
    def synthesize(self, text):
        raise NotImplementedError
```

These are sketches. Implement explicit input/output schemas, time units, errors, cancellation, and resource cleanup before relying on them.

---

## 8. Core data model

The evidence schema is a critical contract between the specialized models and the LLM.

Potential entities:

- `Recording`
- `AnalysisRun`
- `Transcript`
- `Word`
- `Speaker`
- `SpeakerSegment`
- `SoundEvent`
- `AcousticFeature`
- `ModelRun`
- `SpeakerProfile` (optional later)

### Example analysis result

```json
{
  "recording_id": "recording_001",
  "transcript": "Let's start the meeting.",
  "words": [
    {
      "text": "Let's",
      "start": 0.0,
      "end": 0.25,
      "speaker_id": "speaker_1",
      "confidence": null
    }
  ],
  "speaker_segments": [
    {
      "speaker_id": "speaker_1",
      "start": 0.0,
      "end": 1.5
    }
  ],
  "sound_events": [],
  "acoustic_features": [],
  "uncertainties": []
}
```

This is illustrative, not model output. Use seconds consistently for timestamps. Preserve overlapping segments. Keep native model confidence separate if confidence scales differ. Never invent a unified confidence score.

The schema should support:

- Unknown speaker assignments.
- Multiple overlapping speakers.
- Unknown/other event labels.
- Failed or skipped pipeline stages.
- Provenance: which model produced each result.
- Model and configuration versions.
- Timestamp precision and alignment caveats.

---

## 9. Implementation roadmap

### Phase 0 — Hardware and environment assessment

**Tasks**
- Confirm OS, RAM, CPU, and GPU.
- Install Python, `uv`, Git, FFmpeg, and optionally Ollama.
- Check package/runtime compatibility.
- Check model license, access requirements, and storage size.
- Decide target languages.

**Deliverable:** Reproducible development environment and a documented hardware baseline.

**Exit criteria:** Environment checks pass and at least one viable runtime candidate is identified per initial capability.

### Phase 1 — Project foundation

**Tasks**
- Create the package and CLI.
- Add validated configuration loading.
- Define schemas and provider interfaces.
- Implement the orchestrator with mock providers.
- Set up pytest and basic logging.
- Add `.gitignore` for audio, databases, model weights, and secrets.

**Deliverable:** CLI runs and returns a structured mock result.

**Exit criteria:** Core flow and config validation are tested without any ML model installed.

### Phase 2 — Audio input and preprocessing

**Tasks**
- Analyze an existing audio file.
- Add explicit microphone recording.
- Implement start/stop state handling.
- Validate duration, size, format, and decoding.
- Normalize audio as required by each provider.
- Add playback and cancellation behavior.
- Clean up temporary files according to retention settings.

**Deliverable:** Validated audio input from file or deliberate recording.

**Exit criteria:** No microphone recording starts automatically; malformed inputs fail safely.

### Phase 3 — ASR

**Tasks**
- Implement first ASR adapter.
- Return transcript and word timestamps where available.
- Preserve supported language and confidence metadata.
- Benchmark alternatives on the same evaluation set.
- Record latency, memory, WER, and timestamp quality.

**Deliverable:** Model-independent transcript schema.

**Exit criteria:** Chosen model meets agreed quality and performance requirements on target audio.

### Phase 4 — Diarization

**Tasks**
- Integrate one diarization pipeline.
- Normalize speaker segments and timestamps.
- Preserve overlapping speech.
- Test short utterances and speaker changes.
- Track unknown/uncertain speaker assignments.
- Evaluate diarization error rate and speaker consistency.

**Deliverable:** Timestamped speaker segments.

**Exit criteria:** Multi-speaker recordings produce usable diarization with documented limitations.

### Phase 5 — Word/speaker alignment

**Tasks**
- Align ASR word timestamps with diarization intervals.
- Handle boundaries and overlapping speech.
- Represent ambiguous attribution rather than forcing a label.
- Keep ASR and diarization results available separately for debugging.

**Deliverable:** Speaker-attributed transcript.

**Exit criteria:** Evaluation includes word correctness and speaker-attribution correctness separately.

### Phase 6 — VAD, sound events, and acoustics

**Tasks**
- Evaluate whether separate VAD is beneficial for the selected pipeline.
- Implement a sound-event baseline.
- Evaluate cough, throat-clearing, laughter, and background-noise examples.
- Add acoustic feature extraction.
- Attach measurements to time intervals and speakers when defensible.
- Record failed/invalid measurements.

**Deliverable:** Timestamped event timeline and acoustic measurements.

**Exit criteria:** Precision/recall and false-positive rates are measured on labeled examples. Replace inadequate models rather than assuming baseline performance is sufficient.

### Phase 7 — Evidence builder

**Tasks**
- Normalize time bases.
- Merge transcript, diarization, event, and acoustic outputs.
- Preserve overlapping intervals.
- Attach provenance and uncertainty.
- Produce stable JSON output.
- Test serialization and edge cases.

**Deliverable:** Complete evidence object for one recording.

**Exit criteria:** The evidence can be inspected and tested independently of the LLM.

### Phase 8 — Local LLM reasoning

**Tasks**
- Integrate Ollama through an LLM provider interface.
- Select an actual model based on hardware and language needs.
- Pass the user's question and relevant evidence.
- Add follow-up questions about the same recording.
- Prevent the LLM from presenting unsupported events or measurements as facts.

**Deliverable:** Local question answering and summaries about analyzed recordings.

**Exit criteria:** Responses use evidence and clearly state when information is unavailable or uncertain.

### Phase 9 — Provider routing and privacy enforcement

**Tasks**
- Resolve providers per capability.
- Validate provider capabilities at startup.
- Enforce local-only mode.
- Reject cloud requests when disallowed.
- Prohibit automatic cloud fallback.
- Expose provider selection in diagnostic output.
- Prevent sensitive audio/transcripts from entering unintended logs.

**Deliverable:** Configurable routing with enforceable privacy policy.

**Exit criteria:** Tests prove that a failed local provider cannot cause an unapproved upload.

### Phase 10 — Storage and history

**Tasks**
- Add SQLite schema and migrations.
- Store analysis metadata and outputs.
- Store audio only under the configured retention policy.
- Retrieve prior analyses for follow-up questions.
- Implement deletion for recordings, transcripts, and analysis.
- Record model/version provenance.

**Deliverable:** Results persist between sessions.

**Exit criteria:** Restarting the app preserves retained results, and deletion removes the relevant data.

### Phase 11 — Optional cross-recording speaker recognition

**Tasks**
- Select an embedding model.
- Design explicit enrollment.
- Store profiles locally.
- Calibrate similarity thresholds on representative data.
- Support unknown/no-match outcomes.
- Add profile deletion and opt-in controls.

**Deliverable:** Optional matching against enrolled voice profiles.

**Exit criteria:** The feature reports uncertainty and is not treated as definitive identity verification.

### Phase 12 — Optional TTS, API, or UI

Only add these after the core pipeline works.

- TTS: local first; cloud only when enabled.
- API: FastAPI wrapper around the same core.
- UI: desktop or web interface calling the same core/API.
- Streaming: separate design and latency evaluation.

**Deliverable:** Optional interaction layers without rewriting the core.

### Phase 13 — Release hardening

**Tasks**
- Benchmark end-to-end latency and peak memory.
- Test corrupt files, no-speech clips, and provider failures.
- Improve cancellation and cleanup.
- Pin dependencies and model versions where practical.
- Document setup, limitations, privacy, and troubleshooting.
- Review licenses for intended use.

**Deliverable:** Repeatable local release.

---

## 10. CLI design

Potential commands:

```bash
audio-agent --help
audio-agent doctor
audio-agent chat
audio-agent analyze ./recordings/sample.wav
audio-agent record
audio-agent recordings list
audio-agent recordings show <recording-id>
audio-agent recordings delete <recording-id>
audio-agent models list
audio-agent models test
audio-agent config validate
```

Possible interactive flow:

```text
Audio Agent

1. Ask a text question
2. Analyze an audio file
3. Record audio
4. View previous analyses
5. Check model status
6. Exit
```

Keep CLI presentation separate from the orchestration and provider logic.

---

## 11. Testing and evaluation plan

### Unit tests

- Configuration validation.
- Audio input validation.
- Timestamp conversion.
- Transcript serialization.
- Word/speaker alignment.
- Overlap handling.
- Privacy policy.
- Storage, retention, and deletion.
- Provider failure handling.

### Integration tests

Use a small, appropriately licensed and locally stored dataset containing:

- Clear single-speaker speech.
- Two speakers alternating.
- Three or more speakers.
- Overlapping speech.
- Short interjections.
- Background noise.
- Coughs and laughter.
- Silence and no-speech recordings.
- Target languages and accents.

### Metrics

| Capability | Evaluation |
|---|---|
| ASR | Word error rate, latency |
| Diarization | Diarization error rate, speaker consistency |
| Speaker-attributed transcript | Word error rate plus speaker attribution accuracy |
| VAD | Speech-region precision/recall and boundary quality |
| Sound events | Per-class precision, recall, false-positive rate |
| Acoustic features | Valid-estimate rate and stability |
| LLM | Evidence grounding, factual accuracy, useful answers |
| Whole pipeline | End-to-end latency, peak memory, failure rate |

Use the same recordings for model comparisons. Record model version, runtime, hardware, settings, and test-set composition. Model-card benchmark results are not a substitute for evaluating the target use case.

---

## 12. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Model won't fit or runs too slowly | Check hardware first; benchmark small candidates |
| ASR and diarization timestamps disagree | Keep separate outputs and test alignment logic |
| Overlapping speech is mislabeled | Preserve overlap and allow ambiguous assignments |
| Cough detector produces false positives | Build labeled examples; evaluate precision/recall |
| VAD cuts off quiet words | Test VAD on/off; don't require it if the pipeline handles segmentation |
| Model confidence scores aren't comparable | Preserve native scores and avoid fabricated unified confidence |
| Silent cloud upload | Central privacy policy, explicit provider routing, automated tests |
| Too many configuration options | Sensible defaults; separate everyday, advanced, and developer settings |
| Cross-recording identity mistakes | Opt-in enrollment, calibrated thresholds, unknown/no-match outcome |
| Dependencies conflict | Pin compatible versions and test runtime installation early |
| LLM invents conclusions | Evidence-grounded prompts, structured input, evaluation tests |
| Sensitive data retained unintentionally | Retention policy, deletion tests, local-only defaults |

---

## 13. Decisions still open for brainstorming

Fill these in as the project becomes clearer.

### Product

- [ ] CLI only, or eventual desktop app?
- [ ] English only, Indian English, or multilingual?
- [ ] Typical recording length?
- [ ] Expected maximum number of speakers?
- [ ] Is near-real-time response important?
- [ ] Is the main use case conversations, personal voice notes, meetings, or general ambient sounds?

### Speaker handling

- [ ] Within-recording diarization only for the first release?
- [ ] Cross-recording speaker profiles later?
- [ ] Manual renaming of `speaker_1`, `speaker_2`, etc.?
- [ ] Should the system allow a speaker to remain unknown?
- [ ] How should it handle overlap?

### Audio analysis

- [ ] Which sound events matter most?
- [ ] Is cough detection a priority or a later experiment?
- [ ] Which acoustic features are useful for the actual use case?
- [ ] Should users be able to enable/disable analysis stages?

### Models and execution

- [ ] Which OS and hardware?
- [ ] What are the target languages?
- [ ] Which capabilities must run offline?
- [ ] Is cloud use allowed for any component?
- [ ] Is model download/storage size a concern?
- [ ] Is CPU-only operation required?

### Data and privacy

- [ ] Retain original recordings by default?
- [ ] Retain transcripts when deleting audio?
- [ ] How long should analysis results persist?
- [ ] Should speaker profiles be encrypted at rest?
- [ ] Should a cloud LLM be allowed to receive text evidence while audio remains local?

### Quality targets

- [ ] Acceptable transcription error rate?
- [ ] Acceptable speaker-attribution error rate?
- [ ] Acceptable event-detection false-positive rate?
- [ ] Maximum acceptable analysis latency?
- [ ] Maximum RAM or disk usage?

---

## 14. Suggested MVP acceptance checklist

MVP 1.0 is complete when:

- [ ] The project installs reproducibly.
- [ ] The CLI starts and reports environment/model status.
- [ ] Text questions can be sent to a local LLM.
- [ ] Existing audio files can be analyzed.
- [ ] Microphone recording requires explicit user action.
- [ ] Audio validation and preprocessing work.
- [ ] A local ASR provider returns a transcript.
- [ ] A diarization provider returns speaker segments.
- [ ] Word/speaker alignment handles overlap and uncertainty.
- [ ] Sound-event detection returns timestamped predictions.
- [ ] Acoustic analysis returns measured features where valid.
- [ ] Evidence is serialized in a stable schema.
- [ ] The LLM can answer questions about the evidence.
- [ ] Results persist locally when configured.
- [ ] Local-only mode prevents unapproved cloud calls.
- [ ] Tests cover errors, deletion, and privacy policy.
- [ ] Known limitations and evaluation results are documented.

---

## 15. Recommended immediate next steps

1. Confirm operating system, RAM, CPU, and GPU.
2. Decide initial language coverage.
3. Create the project and environment.
4. Implement schemas, provider interfaces, CLI, and mock orchestrator.
5. Test one ASR model.
6. Add diarization and alignment.
7. Add sound-event and acoustic analysis.
8. Add the evidence-grounded local LLM.
9. Add storage and provider policy.
10. Evaluate the full pipeline before adding UI/API/TTS.

**Keep this plan as a living document.** The architecture and privacy requirements should be stable; model selections and performance settings should change only when evidence from testing justifies it.
