"""Audio file format, size, and validity checking."""

import os
from pathlib import Path
from typing import Set
from audio_agent.exceptions import AudioProcessingError

SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_FILESIZE_BYTES: int = 500 * 1024 * 1024 # 500 MB default sanity cap


def validate_audio_file(file_path: Path) -> Path:
    """Validates that the file exists, has a supported audio extension, and is not empty."""
    path = Path(file_path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file not found: {path}")

    if not path.is_file():
        raise AudioProcessingError(f"Path is not a regular file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioProcessingError(
            f"Unsupported audio format '{path.suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioProcessingError(f"Audio file is empty (0 bytes): {path}")

    if file_size > MAX_FILESIZE_BYTES:
        raise AudioProcessingError(
            f"Audio file exceeds maximum size limit ({file_size / (1024*1024):.1f} MB > 500 MB): {path}"
        )

    return path

