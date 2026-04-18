from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from importlib.resources import files

from .specs import (
    TEMPLATE_NAME,
    SpecValidationError,
    emit_json_schemas,
    load_bundle,
    validate_bundle,
)

app = typer.Typer(help="Video Task Compiler utilities.")
spec_app = typer.Typer(help="Manage alpha spec bundles.")
app.add_typer(spec_app, name="spec")


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

