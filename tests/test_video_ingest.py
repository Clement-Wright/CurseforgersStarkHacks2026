from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_task_compiler.video_ingest import (
    ColmapCamera,
    ColmapImage,
    DecodedFrame,
    FiducialDetection,
    ParsedColmapModel,
    ReprojectionStats,
    build_dense_pose_map,
    compute_reprojection_statistics,
    extract_frames_from_source,
    normalize_registered_poses,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    rotation_matrix_to_quaternion,
    select_keyframe_indices,
)


def test_extract_frames_from_source_writes_dense_frames(tmp_path: Path) -> None:
    decoded_frames = [
        DecodedFrame(index=index, timestamp_sec=index * 0.25, rgb=np.full((6, 6, 3), 40 * index, dtype=np.uint8))
        for index in range(4)
    ]

    records = extract_frames_from_source(decoded_frames, frames_dir=tmp_path / "frames", image_format="png")

    assert [record.frame_name for record in records] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]
    assert records[2].timestamp_sec == pytest.approx(0.5)
    assert all(record.path.exists() for record in records)


def test_select_keyframe_indices_forces_first_and_last() -> None:
    records = [
        type("Frame", (), {"timestamp_sec": ts})()
        for ts in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]

    indices = select_keyframe_indices(records, sample_fps=2.0)

    assert indices[0] == 0
    assert indices[-1] == 4
    assert indices == [0, 3, 4]


def test_parse_colmap_text_files(tmp_path: Path) -> None:
    cameras_txt = tmp_path / "cameras.txt"
    images_txt = tmp_path / "images.txt"
    points_txt = tmp_path / "points3D.txt"
    cameras_txt.write_text("1 OPENCV 640 480 400 401 320 240 0.1 0.01 0.0 0.0\n", encoding="utf-8")
    images_txt.write_text(
        "1 1 0 0 0 0 0 -1 1 frame_000000.png\n"
        "12 34 7 56 78 -1\n",
        encoding="utf-8",
    )
    points_txt.write_text(
        "7 1.0 2.0 3.0 255 255 255 0.25 1 0\n",
        encoding="utf-8",
    )

    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)
    points = parse_points3d_txt(points_txt)

    assert cameras[1].model == "OPENCV"
    assert images["frame_000000.png"].camera_id == 1
    assert images["frame_000000.png"].observations[0].point3d_id == 7
    assert points[7].error == pytest.approx(0.25)


def _make_direct_pose_records() -> tuple[list, dict[str, object]]:
    records = []
    for index, ts in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        records.append(
            type(
                "FrameRecordLike",
                (),
                {
                    "frame_index": index,
                    "frame_name": f"frame_{index:06d}.png",
                    "timestamp_sec": ts,
                    "registered": False,
                    "pose_source": "missing",
                },
            )()
        )
    return records


def test_build_dense_pose_map_interpolates_interior_frames() -> None:
    from video_task_compiler.video_ingest import PoseEstimate

    records = _make_direct_pose_records()
    direct_pose_map = {
        "frame_000000.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([0.0, 0.0, 0.0]), source="colmap"),
        "frame_000002.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([1.0, 0.0, 0.0]), source="colmap"),
        "frame_000004.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([2.0, 0.0, 0.0]), source="colmap"),
    }

    dense = build_dense_pose_map(records, direct_pose_map, max_interpolation_gap_s=1.0)

    assert dense[1].source == "interpolated"
    assert dense[1].translation_wc[0] == pytest.approx(0.5)
    assert dense[3].translation_wc[0] == pytest.approx(1.5)


def test_build_dense_pose_map_rejects_large_gap() -> None:
    from video_task_compiler.video_ingest import PoseEstimate, VideoIngestError

    records = []
    for index, ts in enumerate((0.0, 1.5, 3.0)):
        records.append(
            type(
                "FrameRecordLike",
                (),
                {
                    "frame_index": index,
                    "frame_name": f"frame_{index:06d}.png",
                    "timestamp_sec": ts,
                    "registered": False,
                    "pose_source": "missing",
                },
            )()
        )
    direct_pose_map = {
        "frame_000000.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([0.0, 0.0, 0.0]), source="colmap"),
        "frame_000001.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([1.0, 0.0, 0.0]), source="colmap"),
        "frame_000002.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([2.0, 0.0, 0.0]), source="colmap"),
    }

    with pytest.raises(VideoIngestError):
        build_dense_pose_map(records, direct_pose_map, max_interpolation_gap_s=1.0)


def test_normalize_registered_poses_solves_metric_similarity() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    model = ParsedColmapModel(
        model_name="0",
        text_dir=Path("."),
        cameras={
            1: ColmapCamera(
                camera_id=1,
                model="OPENCV",
                width_px=8,
                height_px=8,
                params=[8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            )
        },
        images_by_name={
            "frame_000000.png": ColmapImage(1, q_identity.copy(), np.array([0.0, 0.0, -1.0]), 1, "frame_000000.png", []),
            "frame_000001.png": ColmapImage(2, q_identity.copy(), np.array([-1.0, 0.0, -1.0]), 1, "frame_000001.png", []),
            "frame_000002.png": ColmapImage(3, q_identity.copy(), np.array([-1.0, -1.0, -1.0]), 1, "frame_000002.png", []),
        },
        points3d={},
    )
    detections = [
        FiducialDetection("frame_000000.png", 1, np.eye(3), np.array([1.0, 1.0, 0.0])),
        FiducialDetection("frame_000001.png", 1, np.eye(3), np.array([3.0, 1.0, 0.0])),
        FiducialDetection("frame_000002.png", 1, np.eye(3), np.array([3.0, 3.0, 0.0])),
    ]

    normalized, similarity = normalize_registered_poses(model, detections)

    assert similarity.scale == pytest.approx(2.0)
    assert normalized["frame_000001.png"].translation_wc.tolist() == pytest.approx([3.0, 1.0, 0.0])


def test_compute_reprojection_statistics_returns_finite_error() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    camera = ColmapCamera(
        camera_id=1,
        model="OPENCV",
        width_px=640,
        height_px=480,
        params=[400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0],
    )
    model = ParsedColmapModel(
        model_name="0",
        text_dir=Path("."),
        cameras={1: camera},
        images_by_name={
            "frame_000000.png": ColmapImage(
                image_id=1,
                qvec_wxyz=q_identity,
                tvec=np.array([0.0, 0.0, 0.0], dtype=float),
                camera_id=1,
                name="frame_000000.png",
                observations=[],
            )
        },
        points3d={},
    )
    point_world = np.array([0.0, 0.0, 2.0], dtype=float)
    model.points3d[7] = type("Point", (), {"xyz": point_world, "error": 0.0})()
    model.images_by_name["frame_000000.png"].observations.append(type("Obs", (), {"x": 320.0, "y": 240.0, "point3d_id": 7})())

    stats = compute_reprojection_statistics(model)

    assert stats.observation_count == 1
    assert stats.mean_error_px == pytest.approx(0.0)


@pytest.mark.skipif(shutil.which("colmap") is None, reason="colmap not installed")
def test_colmap_binary_smoke() -> None:
    result = subprocess.run(["colmap", "help"], capture_output=True, text=True)
    assert result.returncode == 0
