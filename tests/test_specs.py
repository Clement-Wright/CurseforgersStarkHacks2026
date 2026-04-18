from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from video_task_compiler.specs import SpecValidationError, load_bundle, validate_bundle


def _copy_bundle_root(target_root: Path) -> Path:
    for name in ("spec", "env", "checklists"):
        shutil.copytree(Path(name), target_root / name)
    return target_root


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_checked_in_bundle_validates() -> None:
    bundle = validate_bundle(load_bundle(Path("spec")))

    assert bundle.project.capture.preroll_seconds == pytest.approx(10.0)
    assert bundle.project.gamma.primary_backbone == "fourdhumans"
    assert bundle.project.delta.primary_backbone == "grounded_sam2"
    assert bundle.project.epsilon.primary_backbone == "artifact_fusion"
    assert bundle.project.zeta.primary_backbone == "mujoco_safe_assets"
    assert bundle.project.eta.primary_backbone == "heuristic_ik"
    assert bundle.task.pick_object.ontology_id == "target_object"
    assert bundle.capture.beta.normalization.world_frame == "fiducial_center"
    assert bundle.env_specs["sfm"].name == "video-task-compiler-sfm"


def test_missing_project_file_reports_targeted_error(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    (bundle_root / "spec" / "project.yaml").unlink()

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.file_path.endswith("project.yaml") for issue in excinfo.value.issues)
    assert any("missing required file" in issue.reason for issue in excinfo.value.issues)


def test_missing_environment_file_is_rejected(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    (bundle_root / "env" / "human.environment.yml").unlink()

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.file_path.endswith("human.environment.yml") for issue in excinfo.value.issues)


def test_ontology_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    task_path = bundle_root / "spec" / "task.yaml"
    task_data = _load_yaml(task_path)
    task_data["pick_object"]["ontology_id"] = "missing_object"
    _write_yaml(task_path, task_data)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(bundle_root))

    assert any("ontology id 'missing_object' was not defined" in issue.reason for issue in excinfo.value.issues)


def test_project_world_policy_must_match_capture_normalization(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    project_data["world_frame_policy"] = "fiducial_center"
    project_data["metric_scale_policy"] = "fiducial_marker"
    project_data["coord_frames"]["project_world"] = "W"
    _write_yaml(project_path, project_data)

    capture_path = bundle_root / "spec" / "capture.yaml"
    capture_data = _load_yaml(capture_path)
    capture_data["beta"]["normalization"]["world_frame"] = "fiducial_center"
    _write_yaml(capture_path, capture_data)

    validated = validate_bundle(load_bundle(bundle_root))
    assert validated.project.world_frame_policy == "fiducial_center"


def test_env_files_must_pin_python_310(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    env_path = bundle_root / "env" / "sfm.environment.yml"
    env_data = _load_yaml(env_path)
    env_data["dependencies"][0] = "python=3.11"
    _write_yaml(env_path, env_data)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(bundle_root / "spec"))

    assert any(issue.file_path.endswith("sfm.environment.yml") for issue in excinfo.value.issues)
    assert any("python=3.10" in issue.reason for issue in excinfo.value.issues)


def test_missing_gamma_block_reports_targeted_error(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    del project_data["gamma"]
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "gamma" for issue in excinfo.value.issues)


def test_missing_delta_block_reports_targeted_error(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    del project_data["delta"]
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "delta" for issue in excinfo.value.issues)


def test_missing_epsilon_block_reports_targeted_error(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    del project_data["epsilon"]
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "epsilon" for issue in excinfo.value.issues)


def test_invalid_gamma_acceptance_threshold_is_rejected(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    project_data["acceptance"]["gamma_min_median_wrist_confidence"] = 1.2
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "acceptance.gamma_min_median_wrist_confidence" for issue in excinfo.value.issues)


def test_invalid_delta_acceptance_threshold_is_rejected(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    project_data["acceptance"]["delta_max_duplicate_ids_in_review_sample"] = -1
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "acceptance.delta_max_duplicate_ids_in_review_sample" for issue in excinfo.value.issues)


def test_invalid_eta_acceptance_threshold_is_rejected(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    project_data["acceptance"]["eta_max_joint_step_rad"] = 0.0
    _write_yaml(project_path, project_data)

    with pytest.raises(SpecValidationError) as excinfo:
        load_bundle(bundle_root / "spec")

    assert any(issue.field_path == "acceptance.eta_max_joint_step_rad" for issue in excinfo.value.issues)


def test_task_regions_must_stay_inside_workspace(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    task_path = bundle_root / "spec" / "task.yaml"
    task_data = deepcopy(_load_yaml(task_path))
    task_data["place_region"]["target_region_m"]["max_m"]["x"] = 1.2
    _write_yaml(task_path, task_data)

    with pytest.raises(SpecValidationError) as excinfo:
        validate_bundle(load_bundle(bundle_root))

    assert any(issue.field_path == "place_region.target_region_m" for issue in excinfo.value.issues)
