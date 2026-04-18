from __future__ import annotations

import json
import shutil
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
    ColmapPipelineResult,
    DecodedFrame,
    FiducialDetection,
    ParsedColmapModel,
    ReprojectionStats,
    extract_frames_from_source,
)


runner = CliRunner()


def _copy_bundle_root(target_root: Path) -> Path:
    for name in ("spec", "env", "checklists"):
        shutil.copytree(Path(name), target_root / name)
    return target_root


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_validate_command_succeeds_for_checked_in_template() -> None:
    result = runner.invoke(app, ["spec", "validate", "--spec-dir", "spec"])
    assert result.exit_code == 0
    assert "spec validation passed" in result.stdout


def test_validate_command_returns_non_zero_for_invalid_bundle(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    task_path = bundle_root / "spec" / "task.yaml"
    task_data = _load_yaml(task_path)
    task_data["robot_id"] = "wrong_robot"
    _write_yaml(task_path, task_data)

    result = runner.invoke(app, ["spec", "validate", "--spec-dir", str(bundle_root / "spec")])
    assert result.exit_code == 1
    assert "task.robot_id 'wrong_robot' must match robot.robot_id" in result.stdout


def test_schema_command_emits_schemas_that_accept_the_example_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "schemas"
    result = runner.invoke(app, ["spec", "schema", "--out-dir", str(out_dir)])
    assert result.exit_code == 0

    for schema_name, data_name in (
        ("project.schema.json", "project.yaml"),
        ("robot.schema.json", "robot.yaml"),
        ("task.schema.json", "task.yaml"),
        ("capture.schema.json", "capture.yaml"),
        ("ontology.schema.json", "ontology.yaml"),
    ):
        schema = json.loads((out_dir / schema_name).read_text(encoding="utf-8"))
        instance = yaml.safe_load((Path("spec") / data_name).read_text(encoding="utf-8"))
        jsonschema_validate(instance=instance, schema=schema)


def test_init_command_materializes_full_template_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "new-bundle"
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
    assert (out_dir / "spec" / "project.yaml").exists()
    assert (out_dir / "spec" / "robot.yaml").exists()
    assert (out_dir / "spec" / "task.yaml").exists()
    assert (out_dir / "spec" / "capture.yaml").exists()
    assert (out_dir / "spec" / "ontology.yaml").exists()
    assert (out_dir / "spec" / "coordinate_frames.md").exists()
    assert (out_dir / "env" / "sfm.environment.yml").exists()
    assert (out_dir / "checklists" / "readiness.md").exists()


def _write_beta_friendly_bundle_root(target_root: Path) -> Path:
    bundle_root = _copy_bundle_root(target_root)
    project_path = bundle_root / "spec" / "project.yaml"
    project_data = _load_yaml(project_path)
    project_data["acceptance"]["beta_min_registered_keyframes"] = 3
    project_data["acceptance"]["beta_min_registered_fraction"] = 0.5
    _write_yaml(project_path, project_data)
    return bundle_root


def _build_fake_model() -> ParsedColmapModel:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    camera = ColmapCamera(
        camera_id=1,
        model="OPENCV",
        width_px=8,
        height_px=8,
        params=[8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    )
    image_centers = {
        "frame_000000.png": np.array([0.0, 0.0, 1.0], dtype=float),
        "frame_000001.png": np.array([1.0, 0.0, 1.0], dtype=float),
        "frame_000002.png": np.array([1.0, 1.0, 1.0], dtype=float),
    }
    images = {
        name: ColmapImage(
            image_id=index + 1,
            qvec_wxyz=q_identity.copy(),
            tvec=-center,
            camera_id=1,
            name=name,
            observations=[],
        )
        for index, (name, center) in enumerate(image_centers.items())
    }
    return ParsedColmapModel(
        model_name="final",
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
            DecodedFrame(index=index, pts=index * 30, pts_sec=float(index) * 5.0, t_ns=index * 5_000_000_000, rgb=np.full((8, 8, 3), index * 30, dtype=np.uint8))
            for index in range(5)
        ]
        records = extract_frames_from_source(decoded_frames, frames_dir=frames_dir, image_format=image_format)
        metadata = {
            "width_px": 8,
            "height_px": 8,
            "nominal_fps": 30.0,
            "frame_count": len(records),
            "timestamp_source": "synthetic",
            "duration_sec": records[-1].pts_sec,
        }
        return records, metadata

    def fake_run_colmap(colmap_bin: str, frames_dir: Path, records, colmap_root: Path, capture):
        database_path = colmap_root / "database.db"
        sparse_root = colmap_root / "sparse"
        sparse_text_root = colmap_root / "sparse_txt"
        staging_dir = colmap_root / "_staging"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_root.mkdir(parents=True, exist_ok=True)
        sparse_text_root.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        (database_path).write_text("db", encoding="utf-8")
        (sparse_root / "base").mkdir(parents=True, exist_ok=True)
        (sparse_root / "final").mkdir(parents=True, exist_ok=True)
        (sparse_text_root / "base").mkdir(parents=True, exist_ok=True)
        (sparse_text_root / "final").mkdir(parents=True, exist_ok=True)
        (staging_dir / "debug.txt").write_text("staging", encoding="utf-8")
        model = _build_fake_model()
        return ColmapPipelineResult(
            base_model=model,
            final_model=model,
            database_path=database_path,
            sparse_root=sparse_root,
            sparse_text_root=sparse_text_root,
            component_count=1,
            staging_dir=staging_dir,
        )

    def fake_detect_fiducials(capture, model, record_map):
        rotation = np.eye(3, dtype=float)
        return [
            FiducialDetection("frame_000000.png", 1, rotation, np.array([1.0, 1.0, 0.0], dtype=float)),
            FiducialDetection("frame_000001.png", 1, rotation, np.array([3.0, 1.0, 0.0], dtype=float)),
            FiducialDetection("frame_000002.png", 1, rotation, np.array([3.0, 3.0, 0.0], dtype=float)),
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
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-a-real-video")

    result = runner.invoke(
        app,
        ["video", "ingest-monocular", "--spec-dir", str(bundle_root / "spec"), "--video", str(video_path), "--out-dir", str(tmp_path / "out")],
    )

    assert result.exit_code == 1
    assert "COLMAP binary not found" in result.stdout


def test_video_ingest_success_writes_canonical_outputs_and_prunes_only_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
    video_path = tmp_path / "demo.mp4"
    out_dir = tmp_path / "out"
    video_path.write_bytes(b"synthetic")
    _patch_successful_beta_pipeline(monkeypatch)

    result = runner.invoke(
        app,
        ["video", "ingest-monocular", "--spec-dir", str(bundle_root / "spec"), "--video", str(video_path), "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0
    assert (out_dir / "frames" / "index.csv").exists()
    assert (out_dir / "calibration" / "intrinsics_opencv.yaml").exists()
    assert (out_dir / "colmap" / "database.db").exists()
    assert (out_dir / "colmap" / "sparse" / "final").exists()
    assert (out_dir / "colmap" / "sparse_txt" / "final").exists()
    assert (out_dir / "camera" / "intrinsics.json").exists()
    assert (out_dir / "camera" / "camera_poses.json").exists()
    assert (out_dir / "scene" / "sparse_points.ply").exists()
    assert (out_dir / "timestamps.csv").exists()
    assert (out_dir / "camera_intrinsics.json").exists()
    assert (out_dir / "camera_poses.json").exists()
    assert (out_dir / "reprojection_preview.jpg").exists()
    assert (out_dir / "reconstruction_summary.json").exists()
    assert not (out_dir / "colmap" / "_staging").exists()


def test_video_ingest_keep_workdir_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
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
            str(bundle_root / "spec"),
            "--video",
            str(video_path),
            "--out-dir",
            str(out_dir),
            "--keep-workdir",
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "colmap" / "_staging").exists()
    assert (out_dir / "colmap" / "_staging" / "debug.txt").exists()
