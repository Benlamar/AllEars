"""SQLite database manager for persisting recordings and analysis results."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_agent.exceptions import StorageError
from audio_agent.schemas import AnalysisEvidence, RecordingMetadata


class DatabaseManager:
    """Manages SQLite schema, migrations, records, and deletions."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recordings (
                        id TEXT PRIMARY KEY,
                        filepath TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        filesize_bytes INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        is_retained INTEGER NOT NULL DEFAULT 1
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analyses (
                        recording_id TEXT PRIMARY KEY,
                        evidence_json TEXT NOT NULL,
                        summary TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (recording_id) REFERENCES recordings (id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS speaker_aliases (
                        recording_id TEXT NOT NULL,
                        speaker_id TEXT NOT NULL,
                        alias TEXT NOT NULL,
                        PRIMARY KEY (recording_id, speaker_id),
                        FOREIGN KEY (recording_id) REFERENCES recordings (id) ON DELETE CASCADE
                    )
                """)
                conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to initialize database at {self.db_path}: {e}") from e

    def save_recording(self, metadata: RecordingMetadata) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO recordings 
                    (id, filepath, filename, duration_seconds, filesize_bytes, created_at, is_retained)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.id,
                    metadata.filepath,
                    metadata.filename,
                    metadata.duration_seconds,
                    metadata.filesize_bytes,
                    metadata.created_at.isoformat(),
                    1 if metadata.is_retained else 0,
                ))
                conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save recording metadata: {e}") from e

    def save_analysis(self, evidence: AnalysisEvidence) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analyses (recording_id, evidence_json, summary, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    evidence.recording_id,
                    evidence.model_dump_json(),
                    evidence.summary,
                    evidence.created_at.isoformat(),
                ))
                # Store any speaker aliases
                for spk_id, alias in evidence.speaker_aliases.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO speaker_aliases (recording_id, speaker_id, alias)
                        VALUES (?, ?, ?)
                    """, (evidence.recording_id, spk_id, alias))
                conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save analysis evidence: {e}") from e

    def get_analysis(self, recording_id: str) -> Optional[AnalysisEvidence]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT evidence_json FROM analyses WHERE recording_id = ?", (recording_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                data = json.loads(row["evidence_json"])
                evidence = AnalysisEvidence.model_validate(data)

                # Fetch any aliases
                cursor.execute("SELECT speaker_id, alias FROM speaker_aliases WHERE recording_id = ?", (recording_id,))
                for spk_row in cursor.fetchall():
                    evidence.speaker_aliases[spk_row["speaker_id"]] = spk_row["alias"]

                return evidence
        except Exception as e:
            raise StorageError(f"Failed to load analysis for recording {recording_id}: {e}") from e

    def list_recordings(self) -> List[RecordingMetadata]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM recordings ORDER BY created_at DESC")
                results = []
                for row in cursor.fetchall():
                    results.append(
                        RecordingMetadata(
                            id=row["id"],
                            filepath=row["filepath"],
                            filename=row["filename"],
                            duration_seconds=row["duration_seconds"],
                            filesize_bytes=row["filesize_bytes"],
                            created_at=datetime.fromisoformat(row["created_at"]),
                            is_retained=bool(row["is_retained"]),
                        )
                    )
                return results
        except Exception as e:
            raise StorageError(f"Failed to list recordings: {e}") from e

    def set_speaker_alias(self, recording_id: str, speaker_id: str, alias: str) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO speaker_aliases (recording_id, speaker_id, alias)
                    VALUES (?, ?, ?)
                """, (recording_id, speaker_id, alias))
                
                # Also update inside the json evidence
                cursor.execute("SELECT evidence_json FROM analyses WHERE recording_id = ?", (recording_id,))
                row = cursor.fetchone()
                if row:
                    evidence = AnalysisEvidence.model_validate(json.loads(row["evidence_json"]))
                    evidence.speaker_aliases[speaker_id] = alias
                    cursor.execute("""
                        UPDATE analyses SET evidence_json = ? WHERE recording_id = ?
                    """, (evidence.model_dump_json(), recording_id))
                conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to set speaker alias: {e}") from e

    def delete_recording(self, recording_id: str, delete_audio_file: bool = True) -> bool:
        """Deletes recording, its analysis, and optionally the audio file from disk."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filepath FROM recordings WHERE id = ?", (recording_id,))
                row = cursor.fetchone()
                if not row:
                    return False

                filepath = Path(row["filepath"])
                if delete_audio_file and filepath.exists():
                    try:
                        filepath.unlink()
                    except OSError:
                        pass

                cursor.execute("DELETE FROM speaker_aliases WHERE recording_id = ?", (recording_id,))
                cursor.execute("DELETE FROM analyses WHERE recording_id = ?", (recording_id,))
                cursor.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
                conn.commit()
                return True
        except Exception as e:
            raise StorageError(f"Failed to delete recording {recording_id}: {e}") from e

