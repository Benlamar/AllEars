"""Command-line interface (CLI) for Audio Agent using Typer and Rich."""

import sys
import time
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from audio_agent.analysis.evidence_builder import EvidenceBuilder
from audio_agent.audio.recorder import AudioRecorder
from audio_agent.audio.sanitizer import SpeechTextSanitizer
from audio_agent.config import load_config
from audio_agent.exceptions import AudioAgentError
from audio_agent.orchestrator import AgentOrchestrator
from audio_agent.providers.registry import ProviderRegistry
from audio_agent.storage.database import DatabaseManager

app = typer.Typer(
    name="audio-agent",
    help="Local-first audio intelligence agent for transcription, diarization, sound events, and grounded LLM reasoning.",
    no_args_is_help=True,
)
recordings_app = typer.Typer(help="Manage stored recordings and analyses.")
app.add_typer(recordings_app, name="recordings")

console = Console(highlight=False)


def _get_orchestrator() -> AgentOrchestrator:
    config = load_config()
    db_path = Path(config.defaults.paths.database_path)
    db = DatabaseManager(db_path)
    return AgentOrchestrator(config, db)


@app.command()
def doctor() -> None:
    """Check local environment, GPU/CUDA status, dependencies, and configuration."""
    rprint("[bold blue][*] Audio Agent Diagnostics[/bold blue]\n")
    
    # 1. Python & System
    table = Table(title="System & Runtime", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status / Details", style="green")
    
    table.add_row("Python Version", sys.version.split()[0])
    
    # Check PyTorch & CUDA if installed
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        table.add_row("PyTorch", f"{torch.__version__} (CUDA Available: {cuda_avail}, GPU: {device_name})")
    except ImportError:
        table.add_row("PyTorch", "[yellow]Not installed (will be required for local ML providers)[/yellow]")

    # Check FFmpeg
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    table.add_row("FFmpeg", "[green]Available[/green]" if ffmpeg_path else "[yellow]Not found in PATH (needed for non-WAV formats)[/yellow]")

    # Check Ollama
    ollama_path = shutil.which("ollama")
    table.add_row("Ollama CLI", "[green]Available[/green]" if ollama_path else "[yellow]Not found in PATH[/yellow]")

    # Check Configured LLM Endpoint
    try:
        config = load_config()
        if config.models.llm.provider in {"openai_compatible", "tensorrt_llm", "vllm"}:
            from audio_agent.providers.llm import OpenAICompatibleLLMProvider
            llm_prov = OpenAICompatibleLLMProvider(
                endpoint=config.models.llm.endpoint or "http://localhost:8001/v1",
                model_name=config.models.llm.model,
            )
            is_up = llm_prov.is_available()
            status_str = f"[green]Connected ({config.models.llm.model} at {config.models.llm.endpoint})[/green]" if is_up else f"[red]Unreachable at {config.models.llm.endpoint}[/red]"
            table.add_row(f"Local LLM ({config.models.llm.provider})", status_str)
    except Exception:
        pass

    console.print(table)
    console.print()

    # 2. Config & Registry
    try:
        config = load_config()
        rprint("[green][+] Configuration files loaded and validated successfully.[/green]")
        rprint(f"  * Execution Mode: [bold]{config.privacy.execution.get('mode', 'local')}[/bold]")
        rprint(f"  * Cloud Fallback Allowed: [bold]{config.privacy.cloud.get('allowed', False)}[/bold]")
        rprint(f"  * Storage DB: [bold]{config.defaults.paths.database_path}[/bold]")
    except Exception as e:
        rprint(f"[red][!] Configuration error: {e}[/red]")

    console.print()
    # 3. Available Providers
    prov_table = Table(title="Registered Capability Providers", show_header=True)
    prov_table.add_column("Capability", style="cyan")
    prov_table.add_column("Available Adapters", style="magenta")
    for cap, adapters in ProviderRegistry.list_available().items():
        prov_table.add_row(cap, ", ".join(adapters))
    console.print(prov_table)


@app.command()
def analyze(
    audio_path: Path = typer.Argument(..., help="Path to audio file to analyze"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g. 'en')"),
) -> None:
    """Analyze an audio file and display structured observations and summary."""
    orchestrator = _get_orchestrator()
    rprint(f"\n[bold cyan]Processing audio:[/bold cyan] {audio_path}")

    try:
        evidence = orchestrator.analyze_audio(audio_path, language=language)

        # Print Executive Summary
        if evidence.summary:
            console.print(Panel(evidence.summary, title="[bold green]Executive Summary[/bold green]", border_style="green"))

        # Print Speaker Turns
        if evidence.speaker_segments:
            rprint("\n[bold][*] Speaker Segments & Turns:[/bold]")
            spk_table = Table(show_header=True)
            spk_table.add_column("Start", style="cyan")
            spk_table.add_column("End", style="cyan")
            spk_table.add_column("Speaker", style="yellow")
            spk_table.add_column("Confidence", style="magenta")
            for seg in evidence.speaker_segments:
                alias = evidence.speaker_aliases.get(seg.speaker_id, seg.speaker_id)
                spk_table.add_row(f"{seg.start:.2f}s", f"{seg.end:.2f}s", alias, f"{seg.confidence or 0.0:.2f}")
            console.print(spk_table)

        # Print Sound Events
        if evidence.sound_events:
            rprint("\n[bold][*] Detected Sound Events:[/bold]")
            ev_table = Table(show_header=True)
            ev_table.add_column("Time Window", style="cyan")
            ev_table.add_column("Event Label", style="green")
            ev_table.add_column("Score", style="magenta")
            ev_table.add_column("Description", style="white")
            for ev in evidence.sound_events:
                ev_table.add_row(f"{ev.start:.2f}s - {ev.end:.2f}s", ev.label, f"{ev.score:.2f}", ev.description or "-")
            console.print(ev_table)

        # Print Acoustic Measurements
        ac = evidence.acoustic_features
        if ac.pitch_mean_hz:
            rprint(f"\n[bold][*] Acoustic Metrics:[/bold] Pitch (F0): {ac.pitch_mean_hz:.1f} Hz | Loudness: {ac.intensity_mean_db:.1f} dB | Speech Rate: {ac.speaking_rate_syllables_per_sec:.1f} syll/s")

        rprint(f"\n[green][+] Analysis saved to database under Recording ID:[/green] [bold]{evidence.recording_id}[/bold]")
        rprint(f"Use [bold cyan]audio-agent chat {evidence.recording_id}[/bold cyan] to ask follow-up questions.")

    except AudioAgentError as e:
        rprint(f"[red]Error during analysis: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def record(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output path for recording"),
) -> None:
    """Record audio interactively from microphone (Enter to start, Enter to stop) and analyze it."""
    config = load_config()
    recordings_dir = Path(config.defaults.paths.recordings_dir)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = output or (recordings_dir / f"recording_{int(time.time())}.wav")
    recorder = AudioRecorder(
        sample_rate=config.defaults.recording.sample_rate,
        channels=config.defaults.recording.channels,
    )
    
    rprint("[bold yellow]Press Enter to start microphone recording...[/bold yellow]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return

    saved_path = recorder.record_interactive(target_path, max_duration_seconds=config.defaults.recording.max_duration_seconds)
    rprint(f"[green]Recording saved to {saved_path}. Starting analysis...[/green]")
    
    orchestrator = _get_orchestrator()
    evidence = orchestrator.analyze_audio(saved_path)
    rprint(f"[green]Analysis complete! ID: {evidence.recording_id}[/green]")
    if evidence.summary:
        console.print(Panel(evidence.summary, title="[bold green]Executive Summary[/bold green]"))


def _stream_and_speak(orchestrator: AgentOrchestrator, token_stream, voice: bool = False) -> str:
    """Streams tokens to terminal console in real-time, optionally streaming audio via TTS."""
    full_text = []

    def _clean_token_display(tok: str) -> str:
        # Strip common control tags from screen display
        return SpeechTextSanitizer._CONTROL_TAGS_PATTERN.sub("", tok)

    if voice:
        console.print("[bold blue]Audio Agent > [/bold blue]", end="")

        def _tee_tokens():
            for token in token_stream:
                disp = _clean_token_display(token)
                if disp:
                    console.print(disp, end="", highlight=False)
                full_text.append(token)
                yield token
            console.print()

        for _ in orchestrator.speak_stream(_tee_tokens(), play_audio=True):
            pass
    else:
        console.print("[bold blue]Audio Agent > [/bold blue]", end="")
        for token in token_stream:
            disp = _clean_token_display(token)
            if disp:
                console.print(disp, end="", highlight=False)
            full_text.append(token)
        console.print()

    return "".join(full_text)


@app.command()
def speak(
    text: str = typer.Argument(..., help="Text to synthesize into spoken audio"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output WAV path"),
) -> None:
    """Synthesize text into speech and play aloud with neural TTS."""
    orchestrator = _get_orchestrator()
    try:
        clean = SpeechTextSanitizer.sanitize(text)
        rprint(f"[bold cyan]Speaking:[/bold cyan] {clean}")
        orchestrator.synthesize_speech(clean, output_path=output, play_audio=True)
        if output:
            rprint(f"[green][+] Saved synthesized audio to: {output}[/green]")
    except AudioAgentError as e:
        rprint(f"[red]Error during speech synthesis: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def ask(
    recording_id: str = typer.Argument(..., help="Recording ID to query"),
    question: str = typer.Argument(..., help="Question to ask about the recording"),
    voice: bool = typer.Option(False, "--voice", "-v", help="Stream response aloud with real-time neural TTS"),
) -> None:
    """Ask a single grounded question about an analyzed recording."""
    orchestrator = _get_orchestrator()
    try:
        if voice:
            token_stream = orchestrator.ask_question_stream(recording_id, question)
            _stream_and_speak(orchestrator, token_stream, voice=True)
        else:
            answer = orchestrator.ask_question(recording_id, question)
            console.print(Panel(answer, title="[bold blue]Grounded Answer[/bold blue]", border_style="blue"))
    except AudioAgentError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def chat(
    recording_id: str = typer.Argument(..., help="Recording ID to chat about"),
    voice: bool = typer.Option(False, "--voice", "-v", help="Stream responses aloud with real-time neural TTS"),
) -> None:
    """Start an interactive chat session to query an analyzed recording."""
    orchestrator = _get_orchestrator()
    evidence = orchestrator.db.get_analysis(recording_id)
    if not evidence:
        rprint(f"[red]No analysis found for recording ID '{recording_id}'.[/red]")
        raise typer.Exit(code=1)

    rprint(f"[bold green]Starting interactive Q&A for recording '{evidence.filename}' ({recording_id}).[/bold green]")
    if voice:
        rprint("[bold magenta][Voice Mode Enabled][/bold magenta] Agent responses will stream aloud.")
    rprint("Type 'exit' or 'quit' to end session.\n")

    while True:
        try:
            query = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                break
            if voice:
                token_stream = orchestrator.ask_question_stream(recording_id, query)
                _stream_and_speak(orchestrator, token_stream, voice=True)
            else:
                answer = orchestrator.ask_question(recording_id, query)
                console.print(Panel(answer, title="[bold blue]Audio Agent[/bold blue]", border_style="blue"))
        except (KeyboardInterrupt, EOFError):
            break
    rprint("\n[dim]Chat session closed.[/dim]")


@recordings_app.command("list")
def list_recordings() -> None:
    """List all stored recordings and analyses in the database."""
    orchestrator = _get_orchestrator()
    recs = orchestrator.db.list_recordings()
    if not recs:
        rprint("[yellow]No recordings stored yet.[/yellow]")
        return

    table = Table(title="Stored Recordings", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Filename", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Size", style="white")
    table.add_column("Date", style="magenta")

    for r in recs:
        table.add_row(
            r.id,
            r.filename,
            f"{r.duration_seconds:.1f}s",
            f"{r.filesize_bytes / 1024:.1f} KB",
            r.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@recordings_app.command("alias")
def alias_speaker(
    recording_id: str = typer.Argument(..., help="Recording ID"),
    speaker_id: str = typer.Argument(..., help="Speaker ID, e.g. speaker_1"),
    name: str = typer.Argument(..., help="Human name / alias to assign"),
) -> None:
    """Assign a human name / alias to a speaker ID within a recording."""
    orchestrator = _get_orchestrator()
    orchestrator.db.set_speaker_alias(recording_id, speaker_id, name)
    rprint(f"[green][+] Speaker '{speaker_id}' in recording '{recording_id}' aliased to '[bold]{name}[/bold]'.[/green]")


@recordings_app.command("delete")
def delete_recording(
    recording_id: str = typer.Argument(..., help="Recording ID to delete"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Retain raw audio file on disk"),
) -> None:
    """Delete a recording and its analysis from the database."""
    orchestrator = _get_orchestrator()
    deleted = orchestrator.db.delete_recording(recording_id, delete_audio_file=not keep_audio)
    if deleted:
        rprint(f"[green][+] Recording '{recording_id}' successfully deleted.[/green]")
    else:
        rprint(f"[red][-] Recording '{recording_id}' not found.[/red]")


if __name__ == "__main__":
    app()

