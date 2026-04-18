from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from jsonschema import validate as jsonschema_validate
from typer.testing import CliRunner

from video_task_compiler.cli import app
from video_task_compiler.human_extract import GammaIntrinsics
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
    project_data["acceptance"]["gamma_min_primary_track_fraction"] = 0.5
    project_data["acceptance"]["gamma_min_median_wrist_confidence"] = 0.2
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


def _write_gamma_beta_artifacts(beta_dir: Path) -> None:
    frames_dir = beta_dir / "frames"
    camera_dir = beta_dir / "camera"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    decoded_frames = [
        DecodedFrame(index=index, pts=index * 30, pts_sec=float(index), t_ns=index * 1_000_000_000, rgb=np.full((10, 10, 3), index * 20, dtype=np.uint8))
        for index in range(4)
    ]
    records = extract_frames_from_source(decoded_frames, frames_dir=frames_dir, image_format="png")
    segments = ["preroll", "demo", "demo", "demo"]
    pose_statuses = ["registered", "registered", "interpolated", "unlocalized"]

    with (frames_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_idx",
                "frame_name",
                "image_path",
                "pts",
                "pts_sec",
                "t_ns",
                "segment",
                "is_keyframe",
                "registered",
                "pose_status",
                "colmap_image_id",
            ],
        )
        writer.writeheader()
        for record, segment, pose_status in zip(records, segments, pose_statuses):
            writer.writerow(
                {
                    "frame_idx": record.frame_index,
                    "frame_name": record.frame_name,
                    "image_path": str(record.image_path.resolve()),
                    "pts": record.pts,
                    "pts_sec": record.pts_sec,
                    "t_ns": record.t_ns,
                    "segment": segment,
                    "is_keyframe": str(record.frame_index == 0).lower(),
                    "registered": str(pose_status in {"registered", "interpolated"}).lower(),
                    "pose_status": pose_status,
                    "colmap_image_id": record.frame_index + 1,
                }
            )

    camera_payload = {
        "schema_version": "0.1.0",
        "video_id": "demo_0001",
        "frames": [],
    }
    for index, pose_status in enumerate(pose_statuses):
        entry = {
            "frame_idx": index,
            "pose_status": pose_status,
        }
        if pose_status != "unlocalized":
            entry["T_wc"] = [
                [1.0, 0.0, 0.0, float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            entry["T_cw"] = [
                [1.0, 0.0, 0.0, -float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        camera_payload["frames"].append(entry)
    (camera_dir / "camera_poses.json").write_text(json.dumps(camera_payload, indent=2) + "\n", encoding="utf-8")
    (camera_dir / "intrinsics.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "camera": {
                    "fx": 500.0,
                    "fy": 500.0,
                    "cx": 5.0,
                    "cy": 5.0,
                    "width_px": 10,
                    "height_px": 10,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_successful_gamma_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_task_compiler import human_extract

    def fake_dependencies(fourdhumans_root: Path | None, smpl_model_path: Path | None):
        return Path("/tmp/4dhumans"), Path("/tmp/SMPL_NEUTRAL.pkl")

    def fake_run_fourdhumans_tracking(
        fourdhumans_root: Path,
        frames_dir: Path,
        work_dir: Path,
        smpl_model_path: Path,
        device: str,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tracks": {
                7: {
                    "track_id": 7,
                    "track_score": 0.9,
                    "shape_betas": np.zeros(10, dtype=np.float32),
                    "frames": [
                        {
                            "frame_idx": 1,
                            "bbox_xyxy": [1.0, 1.0, 8.0, 8.0],
                            "joints2d_xyc": np.array(
                                [[0.0, 0.0, 0.0]] * 17,
                                dtype=np.float32,
                            ),
                            "smpl_global_orient": np.zeros(3, dtype=np.float32),
                            "smpl_body_pose": np.zeros(69, dtype=np.float32),
                            "smpl_betas": np.zeros(10, dtype=np.float32),
                            "transl_cam": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                            "joints3d_cam": np.array(
                                [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.1, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.2, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.3, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            ),
                            "visibility": {"right_wrist": 0.9, "right_elbow": 0.85},
                        },
                        {
                            "frame_idx": 2,
                            "bbox_xyxy": [1.0, 1.0, 8.5, 8.5],
                            "joints2d_xyc": np.array([[0.0, 0.0, 0.0]] * 17, dtype=np.float32),
                            "smpl_global_orient": np.zeros(3, dtype=np.float32),
                            "smpl_body_pose": np.zeros(69, dtype=np.float32),
                            "smpl_betas": np.zeros(10, dtype=np.float32),
                            "transl_cam": np.array([0.1, 0.0, 1.0], dtype=np.float32),
                            "joints3d_cam": np.array(
                                [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.15, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.25, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.35, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            ),
                            "visibility": {"right_wrist": 0.92, "right_elbow": 0.86},
                        },
                        {
                            "frame_idx": 3,
                            "bbox_xyxy": [1.0, 1.0, 9.0, 9.0],
                            "joints2d_xyc": np.array([[0.0, 0.0, 0.0]] * 17, dtype=np.float32),
                            "smpl_global_orient": np.zeros(3, dtype=np.float32),
                            "smpl_body_pose": np.zeros(69, dtype=np.float32),
                            "smpl_betas": np.zeros(10, dtype=np.float32),
                            "transl_cam": np.array([0.2, 0.0, 1.0], dtype=np.float32),
                            "joints3d_cam": np.array(
                                [
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.2, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.3, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.4, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                dtype=np.float32,
                            ),
                            "visibility": {"right_wrist": 0.88, "right_elbow": 0.84},
                        },
                    ],
                }
            }
        }
        native_path = work_dir / "tracks.pkl"
        with native_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return native_path

    monkeypatch.setattr(human_extract, "ensure_gamma_dependencies", fake_dependencies)
    monkeypatch.setattr(human_extract, "run_fourdhumans_tracking", fake_run_fourdhumans_tracking)


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


def test_human_extract_missing_fourdhumans_checkout_hard_fails(tmp_path: Path) -> None:
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    _write_gamma_beta_artifacts(beta_dir)

    result = runner.invoke(
        app,
        [
            "human",
            "extract-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--out-dir",
            str(tmp_path / "gamma"),
        ],
    )

    assert result.exit_code == 1
    assert "4DHumans checkout not found" in result.stdout


def test_human_extract_missing_smpl_model_hard_fails(tmp_path: Path) -> None:
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    _write_gamma_beta_artifacts(beta_dir)
    fourdhumans_root = tmp_path / "4dhumans"
    fourdhumans_root.mkdir()
    (fourdhumans_root / "track.py").write_text("print('stub')\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "human",
            "extract-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--out-dir",
            str(tmp_path / "gamma"),
            "--fourdhumans-root",
            str(fourdhumans_root),
        ],
    )

    assert result.exit_code == 1
    assert "Neutral SMPL model not found" in result.stdout


def test_human_extract_success_writes_gamma_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _write_beta_friendly_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    out_dir = tmp_path / "gamma"
    _write_gamma_beta_artifacts(beta_dir)
    _patch_successful_gamma_pipeline(monkeypatch)

    result = runner.invoke(
        app,
        [
            "human",
            "extract-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "human" / "native" / "4dhumans_tracks.pkl").exists()
    assert (out_dir / "human" / "smpl_tracks.pkl").exists()
    assert (out_dir / "human" / "arm_observables.parquet").exists()
    assert (out_dir / "human" / "reprojection_overlays").exists()
    assert (out_dir / "human" / "summary.json").exists()
    table = pq.read_table(out_dir / "human" / "arm_observables.parquet")
    assert table.num_rows == 4
    summary = json.loads((out_dir / "human" / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_track_id"] == 7
