from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from jsonschema import validate as jsonschema_validate
from typer.testing import CliRunner

from video_task_compiler.cli import app
from video_task_compiler.video_ingest import (
    ColmapCamera,
    ColmapImage,
    DecodedFrame,
    FiducialDetection,
    ParsedColmapModel,
    ReprojectionStats,
    extract_frames_from_source,
)


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


def _write_beta_friendly_spec_bundle(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("robot.yaml", "task.yaml", "capture.yaml"):
        (target_dir / name).write_text(Path("spec", name).read_text(encoding="utf-8"), encoding="utf-8")

    capture_data = yaml.safe_load((target_dir / "capture.yaml").read_text(encoding="utf-8"))
    capture_data["beta"]["colmap"]["min_registered_keyframes"] = 3
    (target_dir / "capture.yaml").write_text(yaml.safe_dump(capture_data, sort_keys=False), encoding="utf-8")
    return target_dir


def _build_fake_model() -> ParsedColmapModel:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    camera = ColmapCamera(
        camera_id=1,
        model="OPENCV",
        width_px=8,
        height_px=8,
        params=[8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    )
    image_names = ("frame_000000.png", "frame_000002.png", "frame_000004.png")
    camera_centers = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([1.0, 0.0, 1.0], dtype=float),
        np.array([1.0, 1.0, 1.0], dtype=float),
    )
    images = {
        name: ColmapImage(
            image_id=index + 1,
            qvec_wxyz=q_identity.copy(),
            tvec=-center,
            camera_id=1,
            name=name,
            observations=[],
        )
        for index, (name, center) in enumerate(zip(image_names, camera_centers))
    }
    return ParsedColmapModel(
        model_name="0",
        text_dir=Path("."),
        cameras={1: camera},
        images_by_name=images,
        points3d={},
    )


def _patch_successful_beta_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_task_compiler import video_ingest

    def fake_dependencies(colmap_bin: str | None = None) -> str:
        return "colmap"

    def fake_decode(video_path: Path, frames_dir: Path, image_format: str):
        decoded_frames = [
            DecodedFrame(index=index, timestamp_sec=index * 0.25, rgb=np.full((8, 8, 3), index * 30, dtype=np.uint8))
            for index in range(5)
        ]
        records = extract_frames_from_source(decoded_frames, frames_dir=frames_dir, image_format=image_format)
        metadata = {
            "width_px": 8,
            "height_px": 8,
            "nominal_fps": 4.0,
            "frame_count": len(records),
            "timestamp_source": "synthetic",
        }
        return records, metadata

    def fake_run_colmap(colmap_bin: str, keyframes_dir: Path, work_dir: Path, capture):
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "raw_artifact.txt").write_text("colmap", encoding="utf-8")
        return _build_fake_model()

    def fake_detect_fiducials(capture, model, keyframe_map):
        return [
            FiducialDetection(
                frame_name="frame_000000.png",
                marker_id=1,
                rotation_tc=np.eye(3, dtype=float),
                translation_tc=np.array([1.0, 1.0, 0.0], dtype=float),
            ),
            FiducialDetection(
                frame_name="frame_000002.png",
                marker_id=1,
                rotation_tc=np.eye(3, dtype=float),
                translation_tc=np.array([3.0, 1.0, 0.0], dtype=float),
            ),
            FiducialDetection(
                frame_name="frame_000004.png",
                marker_id=1,
                rotation_tc=np.eye(3, dtype=float),
                translation_tc=np.array([3.0, 3.0, 0.0], dtype=float),
            ),
        ]

    monkeypatch.setattr(video_ingest, "ensure_beta_dependencies", fake_dependencies)
    monkeypatch.setattr(video_ingest, "decode_video_to_frames", fake_decode)
    monkeypatch.setattr(video_ingest, "run_colmap_pipeline", fake_run_colmap)
    monkeypatch.setattr(video_ingest, "detect_fiducials", fake_detect_fiducials)
    monkeypatch.setattr(
        video_ingest,
        "compute_reprojection_statistics",
        lambda model: ReprojectionStats(observation_count=42, mean_error_px=0.5, max_error_px=1.0),
    )


def test_video_ingest_missing_colmap_hard_fails(tmp_path: Path) -> None:
    spec_dir = _write_beta_friendly_spec_bundle(tmp_path / "spec")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-a-real-video")

    result = runner.invoke(
        app,
        ["video", "ingest-monocular", "--spec-dir", str(spec_dir), "--video", str(video_path), "--out-dir", str(tmp_path / "out")],
    )

    assert result.exit_code == 1
    assert "COLMAP binary not found" in result.stdout


def test_video_ingest_success_writes_outputs_and_prunes_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_dir = _write_beta_friendly_spec_bundle(tmp_path / "spec")
    video_path = tmp_path / "demo.mp4"
    out_dir = tmp_path / "out"
    video_path.write_bytes(b"synthetic")
    _patch_successful_beta_pipeline(monkeypatch)

    result = runner.invoke(
        app,
        ["video", "ingest-monocular", "--spec-dir", str(spec_dir), "--video", str(video_path), "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0
    assert (out_dir / "frames").exists()
    assert (out_dir / "timestamps.csv").exists()
    assert (out_dir / "camera_intrinsics.json").exists()
    assert (out_dir / "camera_poses.json").exists()
    assert (out_dir / "reprojection_preview.jpg").exists()
    assert (out_dir / "reconstruction_summary.json").exists()
    assert not (out_dir / "_workdir").exists()


def test_video_ingest_keep_workdir_preserves_raw_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_dir = _write_beta_friendly_spec_bundle(tmp_path / "spec")
    video_path = tmp_path / "demo.mp4"
    out_dir = tmp_path / "out"
    video_path.write_bytes(b"synthetic")
    _patch_successful_beta_pipeline(monkeypatch)

    result = runner.invoke(
        app,
        [
            "video",
            "ingest-monocular",
            "--spec-dir",
            str(spec_dir),
            "--video",
            str(video_path),
            "--out-dir",
            str(out_dir),
            "--keep-workdir",
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "_workdir").exists()
    assert (out_dir / "_workdir" / "raw_artifact.txt").exists()
