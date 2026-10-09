# Audio Agent

Local-First Audio Intelligence Agent for transcription, multi-speaker diarization, sound event detection, acoustic analysis, and grounded local LLM reasoning.

## Requirements
- Python 3.11
- uv
- NVIDIA GPU with CUDA (optional but recommended, tested on RTX 4060)

## Quick Start
```bash
# Install package and dependencies
uv sync --extra dev

# Run CLI health check
uv run audio-agent doctor

# Run test suite
uv run pytest
```

