from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_task_compiler.specs import load_bundle, validate_bundle
from video_task_compiler.human_extract import (
    GammaCameraPose,
    GammaFrameRecord,
    apply_moving_average_smoothing,
    build_arm_observables_rows,
    normalize_fourdhumans_tracks,
    sample_overlay_frame_indices,
    select_primary_demonstrator_track,
)


def _bundle():
    return validate_bundle(load_bundle(Path("spec")))


def _frame_records(tmp_path: Path) -> list[GammaFrameRecord]:
    frame_records: list[GammaFrameRecord] = []
    for index, segment in enumerate(("preroll", "demo", "demo", "demo")):
        image_path = tmp_path / f"frame_{index:06d}.png"
        image_path.write_bytes(b"frame")
        frame_records.append(
            GammaFrameRecord(
                frame_idx=index,
                frame_name=image_path.name,
                image_path=image_path,
                pts_sec=float(index),
                t_ns=index * 1_000_000_000,
                segment=segment,
                pose_status="registered" if index < 3 else "unlocalized",
            )
        )
    return frame_records


def _camera_pose_map() -> dict[int, GammaCameraPose]:
    identity_wc = np.eye(4, dtype=np.float32)
    identity_wc[2, 3] = 1.0
    identity_cw = np.eye(4, dtype=np.float32)
    identity_cw[2, 3] = -1.0
    return {
        0: GammaCameraPose(0, "registered", identity_wc.copy(), identity_cw.copy()),
        1: GammaCameraPose(1, "registered", identity_wc.copy(), identity_cw.copy()),
        2: GammaCameraPose(2, "interpolated", identity_wc.copy(), identity_cw.copy()),
        3: GammaCameraPose(3, "unlocalized", None, None),
    }


def _native_payload() -> dict:
    joints2d = np.zeros((17, 3), dtype=np.float32)
    joints2d[6] = [10.0, 10.0, 0.8]
    joints2d[8] = [12.0, 11.0, 0.85]
    joints2d[10] = [14.0, 12.0, 0.9]

    def frame(frame_idx: int, wrist_conf: float, wrist_x: float) -> dict:
        joints2d_frame = joints2d.copy()
        joints2d_frame[10, 2] = wrist_conf
        joints3d = np.zeros((17, 3), dtype=np.float32)
        joints3d[6] = [0.1, 0.0, 1.0]
        joints3d[8] = [0.2, 0.0, 1.0]
        joints3d[10] = [wrist_x, 0.0, 1.0]
        return {
            "frame_idx": frame_idx,
            "bbox_xyxy": [1.0, 1.0, 9.0, 9.0],
            "joints2d_xyc": joints2d_frame,
            "smpl_global_orient": np.zeros(3, dtype=np.float32),
            "smpl_body_pose": np.zeros(69, dtype=np.float32),
            "smpl_betas": np.zeros(10, dtype=np.float32),
            "transl_cam": np.array([0.0, 0.0, 1.0 + 0.1 * frame_idx], dtype=np.float32),
            "joints3d_cam": joints3d,
            "visibility": {"right_wrist": wrist_conf, "right_elbow": 0.85},
        }

    return {
        "tracks": {
            7: {
                "track_id": 7,
                "track_score": 0.8,
                "shape_betas": np.zeros(10, dtype=np.float32),
                "frames": [frame(1, 0.92, 0.3), frame(2, 0.88, 0.35), frame(3, 0.86, 0.4)],
            },
            8: {
                "track_id": 8,
                "track_score": 0.7,
                "shape_betas": np.zeros(10, dtype=np.float32),
                "frames": [frame(1, 0.6, 0.25), frame(2, 0.55, 0.28)],
            },
        }
    }


def test_normalize_fourdhumans_tracks_preserves_frame_idx_and_t_ns(tmp_path: Path) -> None:
    normalized = normalize_fourdhumans_tracks(
        native_payload=_native_payload(),
        frame_records=_frame_records(tmp_path),
        camera_pose_map=_camera_pose_map(),
        video_id="demo_0001",
    )

    track_frame = normalized["tracks"][7]["frames"][0]
    assert track_frame["frame_idx"] == 1
    assert track_frame["t_ns"] == 1_000_000_000
    assert track_frame["camera_pose_status"] == "registered"


def test_select_primary_track_prefers_coverage_then_confidence_then_bbox(tmp_path: Path) -> None:
    normalized = normalize_fourdhumans_tracks(
        native_payload=_native_payload(),
        frame_records=_frame_records(tmp_path),
        camera_pose_map=_camera_pose_map(),
        video_id="demo_0001",
    )

    selection = select_primary_demonstrator_track(normalized, _frame_records(tmp_path))

    assert selection.track_id == 7
    assert selection.completeness == pytest.approx(1.0)
    assert selection.median_wrist_confidence == pytest.approx(0.88)


def test_build_arm_observables_rows_adds_world_columns_only_when_camera_poses_exist(tmp_path: Path) -> None:
    bundle = _bundle()
    frame_records = _frame_records(tmp_path)
    normalized = normalize_fourdhumans_tracks(
        native_payload=_native_payload(),
        frame_records=frame_records,
        camera_pose_map=_camera_pose_map(),
        video_id="demo_0001",
    )
    primary_track = normalized["tracks"][7]

    with_world = build_arm_observables_rows(bundle, frame_records, primary_track, _camera_pose_map())
    without_world = build_arm_observables_rows(bundle, frame_records, primary_track, {})

    assert "wrist_r_x_w" in with_world[1]
    assert "wrist_r_x_w" not in without_world[1]
    assert with_world[3]["world_estimate_status"] == "missing_camera_pose"


def test_smoothing_does_not_mutate_normalized_track_payload(tmp_path: Path) -> None:
    bundle = _bundle()
    frame_records = _frame_records(tmp_path)
    normalized = normalize_fourdhumans_tracks(
        native_payload=_native_payload(),
        frame_records=frame_records,
        camera_pose_map=_camera_pose_map(),
        video_id="demo_0001",
    )
    primary_track = normalized["tracks"][7]
    original = copy.deepcopy(primary_track)

    rows = build_arm_observables_rows(bundle, frame_records, primary_track, _camera_pose_map())
    smoothed = apply_moving_average_smoothing(rows, ["wrist_r_x"], window_size=3)

    assert np.array_equal(primary_track["frames"][0]["joints3d_cam"], original["frames"][0]["joints3d_cam"])
    assert np.array_equal(primary_track["frames"][1]["bbox_xyxy"], original["frames"][1]["bbox_xyxy"])
    assert smoothed[1]["wrist_r_x"] is not None


def test_sample_overlay_frames_uses_active_segment_only(tmp_path: Path) -> None:
    frame_records = _frame_records(tmp_path)

    sampled = sample_overlay_frame_indices(frame_records, overlay_count=2)

    assert sampled == [1, 3]


@pytest.mark.skipif(
    not os.environ.get("FOURDHUMANS_ROOT")
    or not os.environ.get("SMPL_MODEL_PATH")
    or importlib.util.find_spec("pyarrow") is None,
    reason="4DHumans checkout or SMPL model not configured",
)
def test_fourdhumans_smoke() -> None:
    track_py = Path(os.environ["FOURDHUMANS_ROOT"]) / "track.py"
    assert track_py.exists()
    result = subprocess.run(["python", str(track_py), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
