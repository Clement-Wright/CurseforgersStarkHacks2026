from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from importlib.resources import files

from .specs import (
    TEMPLATE_NAME,
    SpecValidationError,
    emit_json_schemas,
    load_bundle,
    validate_bundle,
)
from .video_ingest import VideoIngestError, ingest_monocular_video

app = typer.Typer(help="Video Task Compiler utilities.")
spec_app = typer.Typer(help="Manage alpha spec bundles.")
video_app = typer.Typer(help="Run beta video ingest pipelines.")
app.add_typer(spec_app, name="spec")
app.add_typer(video_app, name="video")


def _template_dir(template: str) -> Path:
    if template != TEMPLATE_NAME:
        raise typer.BadParameter(
            f"unsupported template '{template}', expected '{TEMPLATE_NAME}'"
        )
    return Path(str(files("video_task_compiler").joinpath("templates", template)))


def _print_issues(exc: SpecValidationError) -> None:
    typer.echo("spec validation failed:")
    for issue in exc.issues:
        typer.echo(f"- {issue.file_path} :: {issue.field_path} :: {issue.reason}")


@spec_app.command("init")
def spec_init(
    template: str = typer.Option(
        TEMPLATE_NAME,
        "--template",
        help="Spec bundle template to materialize.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive robot.yaml, task.yaml, and capture.yaml.",
    ),
) -> None:
    """Create a new alpha spec bundle from the built-in template."""
    template_dir = _template_dir(template)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("robot.yaml", "task.yaml", "capture.yaml"):
        destination = output_dir / name
        if destination.exists():
            raise typer.BadParameter(f"refusing to overwrite existing file: {destination}")
        shutil.copyfile(template_dir / name, destination)

    typer.echo(f"initialized template '{template}' in {output_dir}")


@spec_app.command("validate")
def spec_validate(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory that contains robot.yaml, task.yaml, and capture.yaml.",
    )
) -> None:
    """Validate a spec bundle and exit non-zero on failure."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"spec validation passed for {spec_dir}")


@spec_app.command("schema")
def spec_schema(
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to write JSON Schemas into.",
    )
) -> None:
    """Emit JSON Schema files for editor support and future integrations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = emit_json_schemas()
    for name, schema in schemas.items():
        target = out_dir / name
        target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    typer.echo(f"wrote {len(schemas)} schema files to {out_dir}")


@video_app.command("ingest-monocular")
def ingest_monocular(
    spec_dir: Path = typer.Option(
        ...,
        "--spec-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory that contains the validated robot.yaml, task.yaml, and capture.yaml bundle.",
    ),
    video: Path = typer.Option(
        ...,
        "--video",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Input monocular video file.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory that will receive beta ingest artifacts.",
    ),
    colmap_bin: Optional[str] = typer.Option(
        None,
        "--colmap-bin",
        help="Optional explicit path to the COLMAP executable.",
    ),
    keep_workdir: bool = typer.Option(
        False,
        "--keep-workdir",
        help="Keep raw COLMAP work artifacts under the output directory.",
    ),
) -> None:
    """Decode a monocular demo video, calibrate it with COLMAP, and export dense poses."""
    try:
        bundle = load_bundle(spec_dir)
        validate_bundle(bundle)
        if bundle.capture.modality != "rgb_monocular":
            raise VideoIngestError("beta ingest-monocular only supports rgb_monocular capture bundles")
        ingest_monocular_video(
            bundle=bundle,
            video_path=video,
            out_dir=out_dir,
            colmap_bin=colmap_bin,
            keep_workdir=keep_workdir,
        )
    except SpecValidationError as exc:
        _print_issues(exc)
        raise typer.Exit(code=1) from exc
    except VideoIngestError as exc:
        typer.echo(f"video ingest failed: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"video ingest completed successfully in {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
