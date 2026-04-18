from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import validate as jsonschema_validate
from typer.testing import CliRunner

from video_task_compiler.cli import app


runner = CliRunner()


def test_validate_command_succeeds_for_checked_in_template() -> None:
    result = runner.invoke(app, ["spec", "validate", "--spec-dir", "spec"])
    assert result.exit_code == 0
    assert "spec validation passed" in result.stdout


def test_validate_command_returns_non_zero_for_invalid_bundle(tmp_path: Path) -> None:
    working_dir = tmp_path / "spec"
    working_dir.mkdir()
    for name in ("robot.yaml", "task.yaml", "capture.yaml"):
        (working_dir / name).write_text((Path("spec") / name).read_text(encoding="utf-8"), encoding="utf-8")

    task_data = yaml.safe_load((working_dir / "task.yaml").read_text(encoding="utf-8"))
    task_data["robot_id"] = "wrong_robot"
    (working_dir / "task.yaml").write_text(yaml.safe_dump(task_data, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["spec", "validate", "--spec-dir", str(working_dir)])
    assert result.exit_code == 1
    assert "task.robot_id 'wrong_robot' must match robot.robot_id" in result.stdout


def test_schema_command_emits_schemas_that_accept_the_example_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "schemas"
    result = runner.invoke(app, ["spec", "schema", "--out-dir", str(out_dir)])
    assert result.exit_code == 0

    robot_schema = json.loads((out_dir / "robot.schema.json").read_text(encoding="utf-8"))
    task_schema = json.loads((out_dir / "task.schema.json").read_text(encoding="utf-8"))
    capture_schema = json.loads((out_dir / "capture.schema.json").read_text(encoding="utf-8"))

    robot_data = yaml.safe_load(Path("spec/robot.yaml").read_text(encoding="utf-8"))
    task_data = yaml.safe_load(Path("spec/task.yaml").read_text(encoding="utf-8"))
    capture_data = yaml.safe_load(Path("spec/capture.yaml").read_text(encoding="utf-8"))

    jsonschema_validate(instance=robot_data, schema=robot_schema)
    jsonschema_validate(instance=task_data, schema=task_schema)
    jsonschema_validate(instance=capture_data, schema=capture_schema)


def test_init_command_materializes_template(tmp_path: Path) -> None:
    out_dir = tmp_path / "new-spec"
    result = runner.invoke(
        app,
        [
            "spec",
            "init",
            "--template",
            "ur5e_monocular_pick_place",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "robot.yaml").exists()
    assert (out_dir / "task.yaml").exists()
    assert (out_dir / "capture.yaml").exists()

