from __future__ import annotations

import importlib.util
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from video_task_compiler.video_ingest import (
    ColmapCamera,
    ColmapImage,
    DecodedFrame,
    FiducialDetection,
    FrameRecord,
    ParsedColmapModel,
    PoseEstimate,
    compute_reprojection_statistics,
    extract_frames_from_source,
    normalize_registered_poses,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    relative_timestamp_ns,
    select_keyframe_indices,
    timestamp_seconds_from_pts,
    assign_frame_segments,
    build_pose_timeline,
)


def test_extract_frames_from_source_writes_dense_frames(tmp_path: Path) -> None:
    decoded_frames = [
        DecodedFrame(index=index, pts=index * 10, pts_sec=index * 0.25, t_ns=index * 250_000_000, rgb=np.full((6, 6, 3), 40 * index, dtype=np.uint8))
        for index in range(4)
    ]

    records = extract_frames_from_source(decoded_frames, frames_dir=tmp_path / "frames", image_format="png")

    assert [record.frame_name for record in records] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]
    assert records[2].pts == 20
    assert records[2].pts_sec == pytest.approx(0.5)
    assert records[2].t_ns == 500_000_000
    assert all(record.image_path.exists() for record in records)


def test_timestamp_helpers_preserve_nanosecond_contract() -> None:
    absolute_sec = timestamp_seconds_from_pts(90, Fraction(1, 30))
    t_ns = relative_timestamp_ns(absolute_sec, 2.0)

    assert absolute_sec == pytest.approx(3.0)
    assert t_ns == 1_000_000_000


def test_assign_frame_segments_splits_preroll_demo_and_postroll() -> None:
    records = [
        FrameRecord(index, f"frame_{index:06d}.png", index, ts, int(ts * 1_000_000_000), Path(f"frame_{index:06d}.png"))
        for index, ts in enumerate((0.0, 5.0, 10.0, 12.0, 14.0, 16.0))
    ]

    assign_frame_segments(records, preroll_seconds=10.0, postroll_seconds=2.0)

    assert [record.segment for record in records] == [
        "preroll",
        "preroll",
        "preroll",
        "demo",
        "postroll",
        "postroll",
    ]


def test_select_keyframe_indices_forces_first_and_last_preroll_frame() -> None:
    records = [
        FrameRecord(index, f"frame_{index:06d}.png", index, ts, int(ts * 1_000_000_000), Path(f"frame_{index:06d}.png"), segment="preroll")
        for index, ts in enumerate((0.0, 0.2, 0.4, 0.6, 0.8))
    ]

    indices = select_keyframe_indices(records, sample_fps=2.0, segment="preroll")

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
    assert points[7].rgb == (255, 255, 255)
    assert points[7].error == pytest.approx(0.25)


def test_build_pose_timeline_interpolates_internal_gaps_and_leaves_edges_unlocalized() -> None:
    records = [
        FrameRecord(index, f"frame_{index:06d}.png", index, ts, int(ts * 1_000_000_000), Path(f"frame_{index:06d}.png"))
        for index, ts in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
    ]
    direct_pose_map = {
        "frame_000001.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([0.0, 0.0, 0.0]), source="colmap", colmap_image_id=11),
        "frame_000003.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([1.0, 0.0, 0.0]), source="colmap", colmap_image_id=13),
    }

    dense = build_pose_timeline(records, direct_pose_map, max_interpolation_gap_s=1.0)

    assert records[0].pose_status == "unlocalized"
    assert records[1].registered is True
    assert records[2].pose_status == "interpolated"
    assert records[4].pose_status == "unlocalized"
    assert dense[2].translation_wc[0] == pytest.approx(0.5)


def test_build_pose_timeline_skips_large_internal_gaps() -> None:
    records = [
        FrameRecord(index, f"frame_{index:06d}.png", index, ts, int(ts * 1_000_000_000), Path(f"frame_{index:06d}.png"))
        for index, ts in enumerate((0.0, 1.0, 2.5, 4.0))
    ]
    direct_pose_map = {
        "frame_000000.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([0.0, 0.0, 0.0]), source="colmap", colmap_image_id=1),
        "frame_000003.png": PoseEstimate(rotation_wc=np.eye(3), translation_wc=np.array([3.0, 0.0, 0.0]), source="colmap", colmap_image_id=4),
    }

    dense = build_pose_timeline(records, direct_pose_map, max_interpolation_gap_s=1.0)

    assert set(dense.keys()) == {0, 3}
    assert records[1].pose_status == "unlocalized"
    assert records[2].pose_status == "unlocalized"


def test_normalize_registered_poses_uses_fiducial_orientation() -> None:
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    rotation_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
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
        FiducialDetection("frame_000000.png", 1, rotation_z_90, np.array([1.0, 1.0, 0.0])),
        FiducialDetection("frame_000001.png", 1, rotation_z_90, np.array([1.0, 3.0, 0.0])),
        FiducialDetection("frame_000002.png", 1, rotation_z_90, np.array([-1.0, 3.0, 0.0])),
    ]

    normalized, similarity = normalize_registered_poses(model, detections, fiducial_size_m=0.06)

    assert similarity.scale == pytest.approx(2.0)
    assert np.allclose(similarity.rotation, rotation_z_90)
    assert normalized["frame_000001.png"].translation_wc.tolist() == pytest.approx([1.0, 3.0, 0.0])
    assert np.allclose(normalized["frame_000001.png"].rotation_wc, rotation_z_90)


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
    model.points3d[7] = type("Point", (), {"xyz": point_world, "rgb": (255, 255, 255), "error": 0.0})()
    model.images_by_name["frame_000000.png"].observations.append(type("Obs", (), {"x": 320.0, "y": 240.0, "point3d_id": 7})())

    stats = compute_reprojection_statistics(model)

    assert stats.observation_count == 1
    assert stats.mean_error_px == pytest.approx(0.0)


@pytest.mark.skipif(
    shutil.which("colmap") is None or importlib.util.find_spec("av") is None,
    reason="colmap or av not installed",
)
def test_colmap_and_pyav_smoke() -> None:
    result = subprocess.run(["colmap", "help"], capture_output=True, text=True)
    assert result.returncode == 0
