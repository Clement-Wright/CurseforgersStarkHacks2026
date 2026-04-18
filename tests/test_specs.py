from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from video_task_compiler.specs import SpecValidationError, load_bundle, validate_bundle


def _write_bundle(target_dir: Path, robot: dict, task: dict, capture: dict) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "robot.yaml").write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    (target_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    (target_dir / "capture.yaml").write_text(yaml.safe_dump(capture, sort_keys=False), encoding="utf-8")


@pytest.fixture()
def example_bundle_dicts() -> tuple[dict, dict, dict]:
    robot = yaml.safe_load(Path("spec/robot.yaml").read_text(encoding="utf-8"))
    task = yaml.safe_load(Path("spec/task.yaml").read_text(encoding="utf-8"))
    capture = yaml.safe_load(Path("spec/capture.yaml").read_text(encoding="utf-8"))
    return robot, task, capture


def test_checked_in_bundle_validates(example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path) -> None:
    robot, task, capture = example_bundle_dicts
    _write_bundle(tmp_path, robot, task, capture)

    bundle = load_bundle(tmp_path)
    validated = validate_bundle(bundle)

    assert validated.robot.robot_id == "ur5e_robotiq_2f85"
    assert validated.task.family == "pick_place"
    assert validated.capture.scale_source == "fiducial"


def test_missing_required_field_reports_targeted_error(
    example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    robot, task, capture = example_bundle_dicts
    broken_task = deepcopy(task)
    del broken_task["language_prompt"]
    _write_bundle(tmp_path, robot, broken_task, capture)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(tmp_path)

    assert any(issue.field_path == "language_prompt" for issue in excinfo.value.issues)


def test_wrong_robot_id_is_rejected(example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path) -> None:
    robot, task, capture = example_bundle_dicts
    broken_task = deepcopy(task)
    broken_task["robot_id"] = "another_robot"
    _write_bundle(tmp_path, robot, broken_task, capture)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(tmp_path))

    assert any("must match robot.robot_id" in issue.reason for issue in excinfo.value.issues)


def test_non_fiducial_monocular_scale_is_rejected(
    example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    robot, task, capture = example_bundle_dicts
    broken_capture = deepcopy(capture)
    broken_capture["scale_source"] = "known_object"
    _write_bundle(tmp_path, robot, task, broken_capture)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(tmp_path))

    assert any(issue.field_path == "scale_source" for issue in excinfo.value.issues)


def test_unsupported_action_space_is_rejected(
    example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    robot, task, capture = example_bundle_dicts
    broken_robot = deepcopy(robot)
    broken_robot["action_space"]["arm_dof"] = 7
    _write_bundle(tmp_path, broken_robot, task, capture)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(tmp_path))

    assert any(issue.field_path == "action_space.arm_dof" for issue in excinfo.value.issues)


def test_task_regions_must_stay_inside_workspace(
    example_bundle_dicts: tuple[dict, dict, dict], tmp_path: Path
) -> None:
    robot, task, capture = example_bundle_dicts
    broken_task = deepcopy(task)
    broken_task["place_region"]["target_region_m"]["max_m"]["x"] = 1.2
    _write_bundle(tmp_path, robot, broken_task, capture)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(tmp_path))

    assert any(issue.field_path == "place_region.target_region_m" for issue in excinfo.value.issues)

