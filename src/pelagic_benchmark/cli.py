from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from .config import load_config
from .environment import provenance
from .runner import run
from .verify import verify

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def validate(config: Path) -> None:
    """Validate an experiment TOML without running it."""
    loaded = load_config(config)
    typer.echo(f"Valid: {loaded.source}")
    typer.echo(f"Dataset: {loaded.dataset_path}")
    typer.echo(f"Configuration SHA-256: {loaded.sha256}")


@app.command()
def inspect() -> None:
    """Print machine, dependency, and optional Pelagia integration information."""
    typer.echo(json.dumps(provenance(Path.cwd()), indent=2, sort_keys=True, default=str))


@app.command("run")
def run_experiment(config: Path = typer.Argument(..., metavar="CONFIG")) -> None:
    """Execute a new immutable benchmark run."""
    frame_task: TaskID | None = None
    treatment_task: TaskID | None = None
    system_task: TaskID | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        refresh_per_second=8,
    ) as display:
        def report(event: str, payload: dict[str, object]) -> None:
            nonlocal frame_task, treatment_task, system_task
            if event == "started":
                sources = payload["source_count"]
                frame_task = display.add_task(f"Frames ({sources} sources)", total=int(payload["estimated_frames"]))
                treatment_task = display.add_task("Codec treatments", total=int(payload["estimated_treatments"]))
                if int(payload["estimated_system_observations"]):
                    system_task = display.add_task("System context observations", total=int(payload["estimated_system_observations"]))
            elif event == "frame" and frame_task is not None:
                display.advance(frame_task)
            elif event == "treatment" and treatment_task is not None:
                display.advance(treatment_task)
            elif event == "system" and system_task is not None:
                display.update(system_task, total=int(payload["observations"]), completed=int(payload["observations"]), description="System context observations")
            elif event == "completed":
                if frame_task is not None:
                    display.update(frame_task, total=int(payload["frames"]), completed=int(payload["frames"]), description=f"Frames ({payload['frames']} processed)")
                if treatment_task is not None:
                    display.update(treatment_task, total=int(payload["treatments"]), completed=int(payload["treatments"]), description=f"Codec treatments ({payload['treatments']} completed)")
        output = run(load_config(config), progress=report)
    typer.echo(output)


@app.command("verify")
def verify_run(run_directory: Path = typer.Argument(..., metavar="RUN_DIRECTORY")) -> None:
    """Validate checksums, schemas, and row counts for a completed run."""
    checked = verify(run_directory)
    typer.echo(f"Verified {len(checked)} Parquet artifacts in {run_directory}")


@app.command()
def summarize(run_directory: Path = typer.Argument(..., metavar="RUN_DIRECTORY")) -> None:
    """Print simple per-treatment size and timing aggregates."""
    verify(run_directory)
    codec = pq.read_table(run_directory / "codec_results.parquet").to_pylist()
    timing = pq.read_table(run_directory / "benchmark_results.parquet").to_pylist()
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for row in codec:
        entry = groups.setdefault((row["preprocessing_id"], row["codec"], row["setting_id"], "codec_size"), {"bpp": [], "encode_ns": [], "decode_ns": [], "write_ns": [], "read_ns": []})
        entry["bpp"].append(row["bits_per_pixel"])
    for row in timing:
        entry = groups.setdefault((row["preprocessing_id"], row["codec"], row["setting_id"], row["benchmark_kind"]), {"bpp": [], "encode_ns": [], "decode_ns": [], "write_ns": [], "read_ns": []})
        entry[f"{row['operation']}_ns"].append(row["elapsed_ns"])
    result = {f"{preprocessing}/{codec}/{setting}/{kind}": {key: (sum(values) / len(values) if values else None) for key, values in values_by_metric.items()} for (preprocessing, codec, setting, kind), values_by_metric in groups.items()}
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    app(prog_name="pelagic-bench")
