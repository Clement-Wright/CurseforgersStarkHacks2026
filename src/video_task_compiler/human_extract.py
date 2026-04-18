from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class HumanExtractError(Exception):
    """Raised when gamma human motion extraction cannot complete successfully."""


class DependencyError(HumanExtractError):
    """Raised when gamma runtime dependencies are missing."""


SMPL_24_RIGHT_ARM = {
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
}

COCO_17_RIGHT_ARM = {
    "right_shoulder": 6,
    "right_elbow": 8,
    "right_wrist": 10,
}


@dataclass(frozen=True)
class GammaFrameRecord:
    frame_idx: int
    frame_name: str
    image_path: Path
    pts_sec: float
    t_ns: int
    segment: str
    pose_status: str


@dataclass(frozen=True)
class GammaCameraPose:
    frame_idx: int
    pose_status: str
    T_wc: np.ndarray | None
    T_cw: np.ndarray | None


@dataclass(frozen=True)
class GammaIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int
    height_px: int


@dataclass(frozen=True)
class PrimaryTrackSelection:
    track_id: int
    completeness: float
    median_wrist_confidence: float
    median_bbox_area: float
    id_switch_count: int


def _path_string(path: Path) -> str:
    return path.resolve().as_posix()


def _as_array(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    return array.copy()


def _as_optional_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        return None
    return matrix.copy()


def _bool_from_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def missing_gamma_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ("numpy", "PIL", "pyarrow"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def resolve_fourdhumans_root(explicit: Path | None) -> Path:
    candidate = explicit or (Path(os.environ["FOURDHUMANS_ROOT"]) if os.environ.get("FOURDHUMANS_ROOT") else None)
    if candidate is None:
        raise DependencyError(
            "4DHumans checkout not found. Provide --fourdhumans-root or set FOURDHUMANS_ROOT."
        )
    root = candidate.resolve()
    if not root.exists() or not root.is_dir():
        raise DependencyError(f"4DHumans checkout path does not exist: {root}")
    if not (root / "track.py").exists():
        raise DependencyError(f"4DHumans checkout must contain track.py: {root}")
    return root


def resolve_smpl_model_path(explicit: Path | None) -> Path:
    candidate = explicit
    if candidate is None and os.environ.get("SMPL_MODEL_PATH"):
        candidate = Path(os.environ["SMPL_MODEL_PATH"])
    if candidate is None and os.environ.get("FOURDHUMANS_SMPL_MODEL"):
        candidate = Path(os.environ["FOURDHUMANS_SMPL_MODEL"])
    if candidate is None:
        raise DependencyError(
            "Neutral SMPL model not found. Provide --smpl-model or set SMPL_MODEL_PATH."
        )
    model_path = candidate.resolve()
    if not model_path.exists() or not model_path.is_file():
        raise DependencyError(f"Neutral SMPL model path does not exist: {model_path}")
    return model_path


def ensure_gamma_dependencies(
    fourdhumans_root: Path | None,
    smpl_model_path: Path | None,
) -> tuple[Path, Path]:
    missing = missing_gamma_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for gamma extract: " + ", ".join(sorted(missing))
        )
    return resolve_fourdhumans_root(fourdhumans_root), resolve_smpl_model_path(smpl_model_path)


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown subprocess failure"
        raise HumanExtractError(f"command failed: {' '.join(command)} :: {detail}")


def discover_native_track_file(*search_roots: Path) -> Path:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists() or not search_root.is_dir():
            continue
        for candidate in search_root.rglob("*.pkl"):
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
    if not candidates:
        raise HumanExtractError("4DHumans did not produce any .pkl track artifact")

    def candidate_rank(path: Path) -> tuple[int, int, float, str]:
        name = path.name.lower()
        parent = path.parent.as_posix().lower()
        if "basicmodel" in name or "/data/" in parent:
            preferred = 3
        elif "track" in name or "phalp" in name:
            preferred = 0
        elif "result" in parent or "demo_frames" in name:
            preferred = 1
        else:
            preferred = 2
        return (preferred, len(path.parts), -path.stat().st_mtime, name)

    return sorted(candidates, key=candidate_rank)[0]


def stage_fourdhumans_source_frames(
    frame_records: Sequence[GammaFrameRecord],
    staging_dir: Path,
) -> Path:
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for frame_record in sorted(frame_records, key=lambda item: item.frame_idx):
        staged_name = f"{Path(frame_record.frame_name).stem}.jpg"
        staged_path = staging_dir / staged_name
        with Image.open(frame_record.image_path) as image:
            image.convert("RGB").save(staged_path, format="JPEG", quality=95)

    return staging_dir


def run_fourdhumans_tracking(
    fourdhumans_root: Path,
    frames_dir: Path,
    work_dir: Path,
    smpl_model_path: Path,
    device: str,
) -> Path:
    python_executable = shutil.which("python") or shutil.which("python3")
    if python_executable is None:
        raise HumanExtractError("python executable not found on PATH for 4DHumans invocation")
    if device not in {"cpu", "cuda"}:
        raise HumanExtractError(f"unsupported device '{device}', expected 'cpu' or 'cuda'")

    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SMPL_MODEL_PATH"] = str(smpl_model_path)
    env["FOURDHUMANS_SMPL_MODEL"] = str(smpl_model_path)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    command = [
        python_executable,
        "track.py",
        f"video.source={frames_dir.as_posix()}",
        f"hydra.run.dir={work_dir.as_posix()}",
    ]
    run_command(command, cwd=fourdhumans_root, env=env)
    return discover_native_track_file(
        work_dir,
        fourdhumans_root / "outputs" / "results",
        fourdhumans_root / "outputs",
    )


def load_frame_index_csv(path: Path) -> list[GammaFrameRecord]:
    if not path.exists():
        raise HumanExtractError(f"missing beta frame index: {path}")

    records: list[GammaFrameRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            records.append(
                GammaFrameRecord(
                    frame_idx=int(row["frame_idx"]),
                    frame_name=row["frame_name"],
                    image_path=image_path,
                    pts_sec=float(row["pts_sec"]),
                    t_ns=int(row["t_ns"]),
                    segment=row["segment"],
                    pose_status=row["pose_status"],
                )
            )
    if not records:
        raise HumanExtractError(f"beta frame index was empty: {path}")
    return records


def load_camera_pose_map(path: Path) -> dict[int, GammaCameraPose]:
    if not path.exists():
        raise HumanExtractError(f"missing beta camera pose export: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise HumanExtractError("camera/camera_poses.json did not contain a 'frames' list")

    pose_map: dict[int, GammaCameraPose] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame["frame_idx"])
        pose_map[frame_idx] = GammaCameraPose(
            frame_idx=frame_idx,
            pose_status=str(frame.get("pose_status", "unlocalized")),
            T_wc=_as_optional_matrix(frame.get("T_wc")),
            T_cw=_as_optional_matrix(frame.get("T_cw")),
        )
    return pose_map


def load_camera_intrinsics(path: Path) -> GammaIntrinsics | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_payload = payload.get("camera", payload)
    required_fields = {"fx", "fy", "cx", "cy", "width_px", "height_px"}
    if not required_fields.issubset(camera_payload):
        return None
    return GammaIntrinsics(
        fx=float(camera_payload["fx"]),
        fy=float(camera_payload["fy"]),
        cx=float(camera_payload["cx"]),
        cy=float(camera_payload["cy"]),
        width_px=int(camera_payload["width_px"]),
        height_px=int(camera_payload["height_px"]),
    )


def load_native_track_payload(path: Path) -> Any:
    payload_bytes = path.read_bytes()
    candidate_buffers = [payload_bytes]
    try:
        candidate_buffers.append(zlib.decompress(payload_bytes))
    except Exception:
        pass

    for candidate in candidate_buffers:
        try:
            return pickle.loads(candidate)
        except Exception:
            pass
        try:
            import joblib

            return joblib.load(BytesIO(candidate))
        except Exception:
            pass

    raise HumanExtractError(f"could not deserialize native 4DHumans track payload: {path}")


def _looks_like_frame_indexed_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    first_value = next(iter(payload.values()))
    if not isinstance(first_value, dict):
        return False
    return "tid" in first_value and "bbox" in first_value


def _frame_idx_from_native_value(frame_key: Any, frame_payload: dict[str, Any]) -> int | None:
    for candidate in (frame_payload.get("frame_idx"), frame_payload.get("frame_index"), frame_payload.get("time")):
        if isinstance(candidate, (int, np.integer)):
            return int(candidate)
    for candidate in (
        frame_payload.get("img_name"),
        frame_payload.get("frame_path"),
        frame_payload.get("img_path"),
        frame_key,
    ):
        if not isinstance(candidate, str):
            continue
        match = re.search(r"(\d+)(?=\.[^.]+$|$)", Path(candidate).name)
        if match:
            return int(match.group(1))
    return None


def _native_list_item(payload: dict[str, Any], key: str, index: int) -> Any:
    value = payload.get(key)
    if not isinstance(value, list):
        return value
    if key == "size" and value and all(isinstance(item, (int, float, np.integer, np.floating)) for item in value[:2]):
        return value
    if index >= len(value):
        return None
    return value[index]


def _native_joints2d_to_xyc(
    joints_payload: Any,
    image_size: Sequence[int] | None,
    confidence: float,
) -> np.ndarray:
    joints = np.asarray(joints_payload, dtype=np.float32)
    if joints.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if joints.ndim == 1:
        if joints.size % 2 == 0:
            joints = joints.reshape(-1, 2)
        elif joints.size % 3 == 0:
            joints = joints.reshape(-1, 3)
        else:
            return np.zeros((0, 3), dtype=np.float32)

    if joints.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32)

    coords = joints[:, :2].astype(np.float32)
    if coords.size and np.nanmax(coords) <= 2.0 and image_size and len(image_size) >= 2:
        height_px = float(image_size[0])
        width_px = float(image_size[1])
        coords[:, 0] *= width_px
        coords[:, 1] *= height_px

    if joints.shape[1] >= 3:
        confidence_column = joints[:, 2:3].astype(np.float32)
    else:
        confidence_column = np.full((coords.shape[0], 1), float(confidence), dtype=np.float32)
    return np.concatenate([coords, confidence_column], axis=1).astype(np.float32)


def _frame_indexed_payload_to_tracks(payload: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    aggregated_tracks: dict[int, dict[str, Any]] = {}

    for frame_key, frame_payload in payload.items():
        if not isinstance(frame_payload, dict):
            continue
        frame_idx = _frame_idx_from_native_value(frame_key, frame_payload)
        if frame_idx is None:
            continue
        track_ids = frame_payload.get("tid")
        if not isinstance(track_ids, list):
            continue

        for index, track_id in enumerate(track_ids):
            track_id_int = int(track_id)
            detection_score = float(np.asarray(_native_list_item(frame_payload, "conf", index), dtype=np.float32))
            bbox_xywh = np.asarray(_native_list_item(frame_payload, "bbox", index), dtype=np.float32).reshape(-1)
            if bbox_xywh.size >= 4:
                bbox_xyxy = np.array(
                    [
                        bbox_xywh[0],
                        bbox_xywh[1],
                        bbox_xywh[0] + bbox_xywh[2],
                        bbox_xywh[1] + bbox_xywh[3],
                    ],
                    dtype=np.float32,
                )
            else:
                bbox_xyxy = np.zeros(4, dtype=np.float32)

            image_size = frame_payload.get("size")
            joints2d_xyc = _native_joints2d_to_xyc(
                _native_list_item(frame_payload, "2d_joints", index),
                image_size if isinstance(image_size, list) else None,
                detection_score,
            )
            joints3d_cam = np.asarray(_native_list_item(frame_payload, "3d_joints", index), dtype=np.float32)
            smpl_payload = _native_list_item(frame_payload, "smpl", index)
            camera_payload = _native_list_item(frame_payload, "camera", index)

            normalized_frame = {
                "frame_idx": frame_idx,
                "bbox_xyxy": bbox_xyxy,
                "score": detection_score,
                "joints2d_xyc": joints2d_xyc,
                "joints3d_cam": joints3d_cam.astype(np.float32),
            }
            if isinstance(smpl_payload, dict):
                normalized_frame["smpl"] = smpl_payload
            if camera_payload is not None:
                normalized_frame["camera"] = np.asarray(camera_payload, dtype=np.float32)

            aggregated_track = aggregated_tracks.setdefault(
                track_id_int,
                {
                    "track_id": track_id_int,
                    "track_score": detection_score,
                    "frames": [],
                },
            )
            aggregated_track["track_score"] = max(float(aggregated_track["track_score"]), detection_score)
            aggregated_track["frames"].append(normalized_frame)

    for track_payload in aggregated_tracks.values():
        track_payload["frames"].sort(key=lambda item: int(item["frame_idx"]))
        yield int(track_payload["track_id"]), track_payload


def _iter_native_tracks(payload: Any) -> Iterable[tuple[int, dict[str, Any]]]:
    if _looks_like_frame_indexed_payload(payload):
        yield from _frame_indexed_payload_to_tracks(payload)
        return
    if isinstance(payload, dict):
        track_container = payload.get("tracks", payload.get("tracklets", payload.get("results", payload)))
        if isinstance(track_container, dict):
            for track_key, track_value in track_container.items():
                if isinstance(track_value, dict):
                    yield int(track_value.get("track_id", track_key)), track_value
        elif isinstance(track_container, list):
            for index, track_value in enumerate(track_container):
                if isinstance(track_value, dict):
                    yield int(track_value.get("track_id", track_value.get("id", index))), track_value
    elif isinstance(payload, list):
        for index, track_value in enumerate(payload):
            if isinstance(track_value, dict):
                yield int(track_value.get("track_id", track_value.get("id", index))), track_value


def _iter_native_frames(track_payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("frames", "track", "results"):
        frames = track_payload.get(key)
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    yield frame
            return


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta <= 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = np.eye(3, dtype=np.float32)
    rotation += math.sin(theta) * skew
    rotation += (1.0 - math.cos(theta)) * (skew @ skew)
    return rotation


def _extract_bbox_xyxy(frame_payload: dict[str, Any]) -> np.ndarray:
    for key in ("bbox_xyxy", "bbox"):
        if key in frame_payload:
            bbox = np.asarray(frame_payload[key], dtype=np.float32).reshape(-1)
            if bbox.size == 4:
                return bbox
    if "bbox_xywh" in frame_payload:
        x, y, w, h = np.asarray(frame_payload["bbox_xywh"], dtype=np.float32).reshape(-1)[:4]
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


def _extract_joints2d(frame_payload: dict[str, Any]) -> np.ndarray:
    for key in ("joints2d_xyc", "keypoints_2d", "pred_keypoints_2d"):
        if key in frame_payload:
            return np.asarray(frame_payload[key], dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def _extract_joints3d(frame_payload: dict[str, Any]) -> np.ndarray:
    for key in ("joints3d_cam", "keypoints_3d", "pred_keypoints_3d"):
        if key in frame_payload:
            return np.asarray(frame_payload[key], dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def _extract_global_orient(frame_payload: dict[str, Any]) -> np.ndarray:
    if "smpl_global_orient" in frame_payload:
        return np.asarray(frame_payload["smpl_global_orient"], dtype=np.float32).reshape(-1)[:3]
    smpl_payload = frame_payload.get("smpl")
    if isinstance(smpl_payload, dict) and "global_orient" in smpl_payload:
        return np.asarray(smpl_payload["global_orient"], dtype=np.float32).reshape(-1)[:3]
    return np.zeros(3, dtype=np.float32)


def _extract_body_pose(frame_payload: dict[str, Any]) -> np.ndarray:
    if "smpl_body_pose" in frame_payload:
        return np.asarray(frame_payload["smpl_body_pose"], dtype=np.float32).reshape(-1)
    smpl_payload = frame_payload.get("smpl")
    if isinstance(smpl_payload, dict) and "body_pose" in smpl_payload:
        return np.asarray(smpl_payload["body_pose"], dtype=np.float32).reshape(-1)
    return np.zeros(69, dtype=np.float32)


def _extract_smpl_betas(track_payload: dict[str, Any], frame_payload: dict[str, Any]) -> np.ndarray:
    for key in ("shape_betas", "smpl_betas", "betas"):
        if key in track_payload:
            return np.asarray(track_payload[key], dtype=np.float32).reshape(-1)
        if key in frame_payload:
            return np.asarray(frame_payload[key], dtype=np.float32).reshape(-1)
    smpl_payload = frame_payload.get("smpl")
    if isinstance(smpl_payload, dict) and "betas" in smpl_payload:
        return np.asarray(smpl_payload["betas"], dtype=np.float32).reshape(-1)
    return np.zeros(10, dtype=np.float32)


def _extract_translation_cam(frame_payload: dict[str, Any], joints3d_cam: np.ndarray) -> np.ndarray:
    for key in ("transl_cam", "transl", "translation"):
        if key in frame_payload:
            return np.asarray(frame_payload[key], dtype=np.float32).reshape(-1)[:3]
    if joints3d_cam.size:
        return np.asarray(joints3d_cam[0], dtype=np.float32).reshape(-1)[:3]
    return np.zeros(3, dtype=np.float32)


def _joint_index_map(joints: np.ndarray) -> dict[str, int]:
    joint_count = joints.shape[0]
    if joint_count >= 24:
        return SMPL_24_RIGHT_ARM
    if joint_count >= 17:
        return COCO_17_RIGHT_ARM
    raise HumanExtractError(f"unsupported joint layout with {joint_count} joints")


def _joint_confidence(joints2d_xyc: np.ndarray, joint_index: int) -> float:
    if joints2d_xyc.ndim != 2 or joints2d_xyc.shape[0] <= joint_index:
        return 0.0
    if joints2d_xyc.shape[1] < 3:
        return 1.0
    return float(joints2d_xyc[joint_index, 2])


def _visibility_map(frame_payload: dict[str, Any], joints2d_xyc: np.ndarray, joint_map: dict[str, int]) -> dict[str, float]:
    visibility = frame_payload.get("visibility")
    if isinstance(visibility, dict):
        return {str(key): float(value) for key, value in visibility.items()}
    return {
        "right_wrist": _joint_confidence(joints2d_xyc, joint_map["right_wrist"]),
        "right_elbow": _joint_confidence(joints2d_xyc, joint_map["right_elbow"]),
    }


def _transform_points_world(points_cam: np.ndarray, pose: GammaCameraPose | None) -> tuple[np.ndarray | None, str]:
    if pose is None or pose.T_wc is None:
        return None, "missing_camera_pose"
    if points_cam.size == 0:
        return np.zeros_like(points_cam), "available"
    rotation = pose.T_wc[:3, :3]
    translation = pose.T_wc[:3, 3]
    transformed = (rotation @ points_cam.T).T + translation
    return transformed.astype(np.float32), "available"


def _root_transform_world(global_orient: np.ndarray, transl_cam: np.ndarray, pose: GammaCameraPose | None) -> np.ndarray | None:
    if pose is None or pose.T_wc is None:
        return None
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = pose.T_wc[:3, :3] @ _axis_angle_to_rotation(global_orient)
    transform[:3, 3] = pose.T_wc[:3, :3] @ transl_cam + pose.T_wc[:3, 3]
    return transform


def normalize_fourdhumans_tracks(
    native_payload: Any,
    frame_records: Sequence[GammaFrameRecord],
    camera_pose_map: dict[int, GammaCameraPose],
    video_id: str,
) -> dict[str, Any]:
    frames_by_idx = {record.frame_idx: record for record in frame_records}
    normalized_tracks: dict[int, dict[str, Any]] = {}

    for track_id, track_payload in _iter_native_tracks(native_payload):
        normalized_frames: list[dict[str, Any]] = []
        track_betas = np.zeros(10, dtype=np.float32)
        track_score = float(track_payload.get("track_score", track_payload.get("score", 0.0)))

        for frame_payload in _iter_native_frames(track_payload):
            if "frame_idx" not in frame_payload and "frame_index" not in frame_payload:
                continue
            frame_idx = int(frame_payload.get("frame_idx", frame_payload.get("frame_index")))
            frame_record = frames_by_idx.get(frame_idx)
            if frame_record is None:
                continue

            joints2d_xyc = _extract_joints2d(frame_payload)
            joints3d_cam = _extract_joints3d(frame_payload)
            joint_map = _joint_index_map(joints3d_cam if joints3d_cam.size else joints2d_xyc)
            global_orient = _extract_global_orient(frame_payload)
            body_pose = _extract_body_pose(frame_payload)
            smpl_betas = _extract_smpl_betas(track_payload, frame_payload)
            transl_cam = _extract_translation_cam(frame_payload, joints3d_cam)
            camera_pose = camera_pose_map.get(frame_idx)
            joints3d_world, world_status = _transform_points_world(joints3d_cam, camera_pose)
            root_transform_world = _root_transform_world(global_orient, transl_cam, camera_pose)
            visibility = _visibility_map(frame_payload, joints2d_xyc, joint_map)
            track_betas = smpl_betas

            normalized_frame = {
                "frame_idx": frame_idx,
                "frame_name": frame_record.frame_name,
                "t_ns": frame_record.t_ns,
                "segment": frame_record.segment,
                "bbox_xyxy": _extract_bbox_xyxy(frame_payload).astype(np.float32),
                "joints2d_xyc": joints2d_xyc.astype(np.float32),
                "smpl_global_orient": global_orient.astype(np.float32),
                "smpl_body_pose": body_pose.astype(np.float32),
                "smpl_betas": smpl_betas.astype(np.float32),
                "transl_cam": transl_cam.astype(np.float32),
                "joints3d_cam": joints3d_cam.astype(np.float32),
                "camera_pose_status": frame_record.pose_status,
                "visibility": visibility,
                "world_estimate_status": world_status,
            }
            if joints3d_world is not None:
                normalized_frame["joints3d_world"] = joints3d_world.astype(np.float32)
            if root_transform_world is not None:
                normalized_frame["T_wh_root"] = root_transform_world.astype(np.float32)
            normalized_frames.append(normalized_frame)

        normalized_frames.sort(key=lambda item: item["frame_idx"])
        if not normalized_frames:
            continue

        normalized_tracks[int(track_id)] = {
            "track_id": int(track_id),
            "is_primary_demonstrator": False,
            "track_score": float(track_score),
            "shape_betas": track_betas.astype(np.float32),
            "frames": normalized_frames,
        }

    if not normalized_tracks:
        raise HumanExtractError("4DHumans did not yield any normalizable tracks")

    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": video_id,
        "source": "4DHumans",
        "body_model": "SMPL",
        "tracks": normalized_tracks,
    }


def _bbox_area(frame_payload: dict[str, Any]) -> float:
    bbox = np.asarray(frame_payload["bbox_xyxy"], dtype=np.float32)
    if bbox.size != 4:
        return 0.0
    width = max(0.0, float(bbox[2] - bbox[0]))
    height = max(0.0, float(bbox[3] - bbox[1]))
    return width * height


def _track_frame_lookup(track_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(frame["frame_idx"]): frame for frame in track_payload["frames"]}


def count_primary_id_switches(
    normalized_payload: dict[str, Any],
    frame_records: Sequence[GammaFrameRecord],
) -> int:
    active_frames = [frame for frame in frame_records if frame.segment != "preroll"]
    tracks = normalized_payload["tracks"]
    last_track_id: int | None = None
    switch_count = 0

    for frame in active_frames:
        best_track_id: int | None = None
        best_score: tuple[float, float] | None = None
        for track_id, track_payload in tracks.items():
            track_frame = _track_frame_lookup(track_payload).get(frame.frame_idx)
            if track_frame is None:
                continue
            score = (
                float(track_frame["visibility"].get("right_wrist", 0.0)),
                _bbox_area(track_frame),
            )
            if best_score is None or score > best_score:
                best_track_id = int(track_id)
                best_score = score
        if best_track_id is None:
            continue
        if last_track_id is not None and best_track_id != last_track_id:
            switch_count += 1
        last_track_id = best_track_id

    return switch_count


def select_primary_demonstrator_track(
    normalized_payload: dict[str, Any],
    frame_records: Sequence[GammaFrameRecord],
) -> PrimaryTrackSelection:
    active_frames = [frame for frame in frame_records if frame.segment != "preroll"]
    if not active_frames:
        raise HumanExtractError("gamma could not find any active non-preroll frames")

    best_track_id: int | None = None
    best_tuple: tuple[float, float, float] | None = None
    tracks = normalized_payload["tracks"]

    for track_id, track_payload in tracks.items():
        track_frames = [
            frame for frame in track_payload["frames"]
            if frame["segment"] != "preroll"
        ]
        coverage = len(track_frames) / len(active_frames)
        wrist_conf_values = [
            float(frame["visibility"].get("right_wrist", 0.0))
            for frame in track_frames
        ]
        median_wrist_conf = float(np.median(wrist_conf_values)) if wrist_conf_values else 0.0
        bbox_areas = [_bbox_area(frame) for frame in track_frames]
        median_bbox_area = float(np.median(bbox_areas)) if bbox_areas else 0.0
        candidate_tuple = (coverage, median_wrist_conf, median_bbox_area)

        if best_tuple is None or candidate_tuple > best_tuple:
            best_track_id = int(track_id)
            best_tuple = candidate_tuple

    if best_track_id is None or best_tuple is None:
        raise HumanExtractError("gamma could not choose a primary demonstrator track")

    selection = PrimaryTrackSelection(
        track_id=best_track_id,
        completeness=best_tuple[0],
        median_wrist_confidence=best_tuple[1],
        median_bbox_area=best_tuple[2],
        id_switch_count=count_primary_id_switches(normalized_payload, frame_records),
    )
    normalized_payload["tracks"][best_track_id]["is_primary_demonstrator"] = True
    normalized_payload["tracks"][best_track_id]["track_score"] = float(selection.completeness)
    return selection


def _root_yaw(global_orient: np.ndarray, frame_payload: dict[str, Any]) -> float:
    if "T_wh_root" in frame_payload:
        rotation = np.asarray(frame_payload["T_wh_root"], dtype=np.float32)[:3, :3]
    else:
        rotation = _axis_angle_to_rotation(global_orient)
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def _moving_average(values: list[float | None], window_size: int) -> list[float | None]:
    radius = max(0, window_size // 2)
    smoothed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            smoothed.append(value)
            continue
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        neighbors = [
            item for item in values[start:end]
            if item is not None and not (isinstance(item, float) and math.isnan(item))
        ]
        if not neighbors:
            smoothed.append(value)
        else:
            smoothed.append(float(sum(neighbors) / len(neighbors)))
    return smoothed


def apply_moving_average_smoothing(
    rows: Sequence[dict[str, Any]],
    column_names: Sequence[str],
    window_size: int,
) -> list[dict[str, Any]]:
    smoothed_rows = [dict(row) for row in rows]
    visibility_mask = [bool(row.get("track_visible", False)) for row in rows]
    for column_name in column_names:
        raw_values: list[float | None] = []
        for row, visible in zip(rows, visibility_mask):
            value = row.get(column_name)
            raw_values.append(float(value) if visible and value is not None else None)
        smoothed_values = _moving_average(raw_values, window_size)
        for row, visible, smoothed_value in zip(smoothed_rows, visibility_mask, smoothed_values):
            if visible:
                row[column_name] = smoothed_value
    return smoothed_rows


def _task_region_interaction(bundle: SpecBundle, wrist_position: np.ndarray) -> bool:
    search_region = bundle.task.pick_object.search_region_m
    target_region = bundle.task.place_region.target_region_m
    in_pick_region = (
        search_region.min_m.x <= wrist_position[0] <= search_region.max_m.x
        and search_region.min_m.y <= wrist_position[1] <= search_region.max_m.y
        and search_region.min_m.z <= wrist_position[2] <= search_region.max_m.z + 0.05
    )
    in_place_region = (
        target_region.min_m.x <= wrist_position[0] <= target_region.max_m.x
        and target_region.min_m.y <= wrist_position[1] <= target_region.max_m.y
        and target_region.min_m.z <= wrist_position[2] <= target_region.max_m.z + 0.05
    )
    return in_pick_region or in_place_region


def build_arm_observables_rows(
    bundle: SpecBundle,
    frame_records: Sequence[GammaFrameRecord],
    primary_track: dict[str, Any],
    camera_pose_map: dict[int, GammaCameraPose],
) -> list[dict[str, Any]]:
    track_frames_by_idx = _track_frame_lookup(primary_track)
    include_world_columns = bool(camera_pose_map)
    rows: list[dict[str, Any]] = []
    previous_root_position: np.ndarray | None = None
    previous_t_ns: int | None = None

    for frame in frame_records:
        track_frame = track_frames_by_idx.get(frame.frame_idx)
        row: dict[str, Any] = {
            "video_id": bundle.project.video_id,
            "track_id": int(primary_track["track_id"]),
            "frame_idx": frame.frame_idx,
            "t_ns": frame.t_ns,
            "hand_side": "right",
            "segment": frame.segment,
            "is_primary_demonstrator": True,
            "camera_pose_status": frame.pose_status,
            "track_visible": track_frame is not None,
        }

        if track_frame is None:
            for column_name in (
                "shoulder_r_x",
                "shoulder_r_y",
                "shoulder_r_z",
                "elbow_r_x",
                "elbow_r_y",
                "elbow_r_z",
                "wrist_r_x",
                "wrist_r_y",
                "wrist_r_z",
                "wrist_conf",
                "elbow_conf",
                "root_yaw",
                "root_speed",
            ):
                row[column_name] = None
            row["is_interaction_candidate"] = False
            row["world_estimate_status"] = "track_missing"
            if include_world_columns:
                for column_name in (
                    "shoulder_r_x_w",
                    "shoulder_r_y_w",
                    "shoulder_r_z_w",
                    "elbow_r_x_w",
                    "elbow_r_y_w",
                    "elbow_r_z_w",
                    "wrist_r_x_w",
                    "wrist_r_y_w",
                    "wrist_r_z_w",
                ):
                    row[column_name] = None
            rows.append(row)
            continue

        joints3d_cam = np.asarray(track_frame["joints3d_cam"], dtype=np.float32)
        joints2d_xyc = np.asarray(track_frame["joints2d_xyc"], dtype=np.float32)
        joint_map = _joint_index_map(joints3d_cam if joints3d_cam.size else joints2d_xyc)
        shoulder_cam = joints3d_cam[joint_map["right_shoulder"]]
        elbow_cam = joints3d_cam[joint_map["right_elbow"]]
        wrist_cam = joints3d_cam[joint_map["right_wrist"]]
        row.update(
            {
                "shoulder_r_x": float(shoulder_cam[0]),
                "shoulder_r_y": float(shoulder_cam[1]),
                "shoulder_r_z": float(shoulder_cam[2]),
                "elbow_r_x": float(elbow_cam[0]),
                "elbow_r_y": float(elbow_cam[1]),
                "elbow_r_z": float(elbow_cam[2]),
                "wrist_r_x": float(wrist_cam[0]),
                "wrist_r_y": float(wrist_cam[1]),
                "wrist_r_z": float(wrist_cam[2]),
                "wrist_conf": float(track_frame["visibility"].get("right_wrist", 0.0)),
                "elbow_conf": float(track_frame["visibility"].get("right_elbow", 0.0)),
                "root_yaw": _root_yaw(np.asarray(track_frame["smpl_global_orient"], dtype=np.float32), track_frame),
                "world_estimate_status": str(track_frame.get("world_estimate_status", "missing_camera_pose")),
            }
        )

        if "T_wh_root" in track_frame:
            current_root_position = np.asarray(track_frame["T_wh_root"], dtype=np.float32)[:3, 3]
        else:
            current_root_position = np.asarray(track_frame["transl_cam"], dtype=np.float32)
        if previous_root_position is None or previous_t_ns is None or frame.t_ns == previous_t_ns:
            row["root_speed"] = 0.0
        else:
            delta_sec = (frame.t_ns - previous_t_ns) / 1_000_000_000.0
            row["root_speed"] = float(np.linalg.norm(current_root_position - previous_root_position) / max(delta_sec, 1e-6))
        previous_root_position = current_root_position
        previous_t_ns = frame.t_ns

        if "joints3d_world" in track_frame:
            joints3d_world = np.asarray(track_frame["joints3d_world"], dtype=np.float32)
            shoulder_world = joints3d_world[joint_map["right_shoulder"]]
            elbow_world = joints3d_world[joint_map["right_elbow"]]
            wrist_world = joints3d_world[joint_map["right_wrist"]]
            row["is_interaction_candidate"] = bool(
                row["wrist_conf"] >= bundle.project.acceptance.gamma_min_median_wrist_confidence
                and _task_region_interaction(bundle, wrist_world)
            )
            if include_world_columns:
                row.update(
                    {
                        "shoulder_r_x_w": float(shoulder_world[0]),
                        "shoulder_r_y_w": float(shoulder_world[1]),
                        "shoulder_r_z_w": float(shoulder_world[2]),
                        "elbow_r_x_w": float(elbow_world[0]),
                        "elbow_r_y_w": float(elbow_world[1]),
                        "elbow_r_z_w": float(elbow_world[2]),
                        "wrist_r_x_w": float(wrist_world[0]),
                        "wrist_r_y_w": float(wrist_world[1]),
                        "wrist_r_z_w": float(wrist_world[2]),
                    }
                )
        else:
            row["is_interaction_candidate"] = bool(
                row["wrist_conf"] >= bundle.project.acceptance.gamma_min_median_wrist_confidence
                and row["wrist_r_z"] is not None
                and float(row["wrist_r_z"]) > 0.0
            )
            if include_world_columns:
                for column_name in (
                    "shoulder_r_x_w",
                    "shoulder_r_y_w",
                    "shoulder_r_z_w",
                    "elbow_r_x_w",
                    "elbow_r_y_w",
                    "elbow_r_z_w",
                    "wrist_r_x_w",
                    "wrist_r_y_w",
                    "wrist_r_z_w",
                ):
                    row[column_name] = None

        rows.append(row)

    smooth_columns = [
        "shoulder_r_x",
        "shoulder_r_y",
        "shoulder_r_z",
        "elbow_r_x",
        "elbow_r_y",
        "elbow_r_z",
        "wrist_r_x",
        "wrist_r_y",
        "wrist_r_z",
        "root_yaw",
        "root_speed",
    ]
    if include_world_columns:
        smooth_columns.extend(
            [
                "shoulder_r_x_w",
                "shoulder_r_y_w",
                "shoulder_r_z_w",
                "elbow_r_x_w",
                "elbow_r_y_w",
                "elbow_r_z_w",
                "wrist_r_x_w",
                "wrist_r_y_w",
                "wrist_r_z_w",
            ]
        )
    return apply_moving_average_smoothing(rows, smooth_columns, bundle.project.gamma.smoothing.window_size)


def sample_overlay_frame_indices(
    frame_records: Sequence[GammaFrameRecord],
    overlay_count: int,
) -> list[int]:
    active_indices = [frame.frame_idx for frame in frame_records if frame.segment != "preroll"]
    if not active_indices:
        return []
    if len(active_indices) <= overlay_count:
        return active_indices
    sampled = np.linspace(0, len(active_indices) - 1, overlay_count)
    deduped: list[int] = []
    for sample in sampled:
        frame_idx = active_indices[int(round(float(sample)))]
        if frame_idx not in deduped:
            deduped.append(frame_idx)
    return deduped


def _project_camera_point(point_camera: np.ndarray, intrinsics: GammaIntrinsics) -> tuple[float, float] | None:
    if point_camera[2] <= 1e-6:
        return None
    u = intrinsics.fx * (point_camera[0] / point_camera[2]) + intrinsics.cx
    v = intrinsics.fy * (point_camera[1] / point_camera[2]) + intrinsics.cy
    return float(u), float(v)


def _project_world_point(point_world: np.ndarray, pose: GammaCameraPose | None, intrinsics: GammaIntrinsics | None) -> tuple[float, float] | None:
    if pose is None or pose.T_cw is None or intrinsics is None:
        return None
    point_camera = pose.T_cw[:3, :3] @ point_world + pose.T_cw[:3, 3]
    return _project_camera_point(point_camera, intrinsics)


def write_reprojection_overlays(
    overlay_dir: Path,
    frame_records: Sequence[GammaFrameRecord],
    primary_track: dict[str, Any],
    camera_pose_map: dict[int, GammaCameraPose],
    intrinsics: GammaIntrinsics | None,
    overlay_count: int,
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    track_frames_by_idx = _track_frame_lookup(primary_track)
    sampled_indices = sample_overlay_frame_indices(frame_records, overlay_count)
    for frame_idx in sampled_indices:
        frame_record = next(frame for frame in frame_records if frame.frame_idx == frame_idx)
        image = Image.open(frame_record.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        track_frame = track_frames_by_idx.get(frame_idx)
        if track_frame is None:
            draw.text((16, 16), "Primary track missing on sampled frame", fill=(255, 64, 64))
            image.save(overlay_dir / f"{frame_record.frame_name}.jpg", quality=90)
            continue

        bbox = np.asarray(track_frame["bbox_xyxy"], dtype=np.float32)
        draw.rectangle(tuple(float(value) for value in bbox), outline=(255, 196, 0), width=3)
        joints2d_xyc = np.asarray(track_frame["joints2d_xyc"], dtype=np.float32)
        joint_map = _joint_index_map(np.asarray(track_frame["joints3d_cam"], dtype=np.float32))
        for joint_name, joint_index in joint_map.items():
            if joints2d_xyc.shape[0] <= joint_index:
                continue
            u, v = float(joints2d_xyc[joint_index, 0]), float(joints2d_xyc[joint_index, 1])
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=(64, 255, 64))
            draw.text((u + 6, v + 6), joint_name, fill=(64, 255, 64))

        if "joints3d_world" in track_frame:
            joints3d_world = np.asarray(track_frame["joints3d_world"], dtype=np.float32)
            pose = camera_pose_map.get(frame_idx)
            for joint_name, joint_index in joint_map.items():
                projected = _project_world_point(joints3d_world[joint_index], pose, intrinsics)
                if projected is None:
                    continue
                u, v = projected
                draw.rectangle((u - 3, v - 3, u + 3, v + 3), outline=(64, 128, 255), width=2)

        image.save(overlay_dir / f"{frame_record.frame_name}.jpg", quality=90)


def write_observables_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def build_gamma_summary(
    bundle: SpecBundle,
    frame_records: Sequence[GammaFrameRecord],
    primary_track: dict[str, Any],
    selection: PrimaryTrackSelection,
    observables_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    active_frames = [frame for frame in frame_records if frame.segment != "preroll"]
    world_estimate_available = sum(
        1 for row in observables_rows
        if row["segment"] != "preroll" and row.get("world_estimate_status") == "available"
    )
    missing_frame_count = sum(
        1 for row in observables_rows
        if row["segment"] != "preroll" and not row.get("track_visible", False)
    )
    summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": "4DHumans",
        "primary_track_id": selection.track_id,
        "primary_track_completeness": selection.completeness,
        "median_wrist_confidence": selection.median_wrist_confidence,
        "id_switch_count": selection.id_switch_count,
        "missing_frame_count": missing_frame_count,
        "world_estimate_coverage": world_estimate_available / len(active_frames) if active_frames else 0.0,
        "track_count": len(primary_track.get("frames", [])),
        "qc_flags": {
            "primary_track_fraction_ok": selection.completeness >= bundle.project.acceptance.gamma_min_primary_track_fraction,
            "wrist_confidence_ok": selection.median_wrist_confidence >= bundle.project.acceptance.gamma_min_median_wrist_confidence,
            "id_switches_ok": selection.id_switch_count <= bundle.project.acceptance.gamma_max_id_switches,
        },
    }
    return summary


def enforce_gamma_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if summary["primary_track_completeness"] < bundle.project.acceptance.gamma_min_primary_track_fraction:
        raise HumanExtractError(
            f"primary track completeness {summary['primary_track_completeness']:.3f} fell below gamma_min_primary_track_fraction={bundle.project.acceptance.gamma_min_primary_track_fraction:.3f}"
        )
    if summary["median_wrist_confidence"] < bundle.project.acceptance.gamma_min_median_wrist_confidence:
        raise HumanExtractError(
            f"median wrist confidence {summary['median_wrist_confidence']:.3f} fell below gamma_min_median_wrist_confidence={bundle.project.acceptance.gamma_min_median_wrist_confidence:.3f}"
        )
    if summary["id_switch_count"] > bundle.project.acceptance.gamma_max_id_switches:
        raise HumanExtractError(
            f"id switch count {summary['id_switch_count']} exceeded gamma_max_id_switches={bundle.project.acceptance.gamma_max_id_switches}"
        )


def extract_monocular_human_motion(
    bundle: SpecBundle,
    beta_dir: Path,
    out_dir: Path,
    fourdhumans_root: Path | None = None,
    smpl_model_path: Path | None = None,
    device: str = "cuda",
    keep_workdir: bool = False,
) -> dict[str, Any]:
    resolved_fourdhumans_root, resolved_smpl_model_path = ensure_gamma_dependencies(
        fourdhumans_root, smpl_model_path
    )

    frames_dir = beta_dir / "frames"
    frame_index_path = frames_dir / "index.csv"
    camera_pose_path = beta_dir / "camera" / "camera_poses.json"
    intrinsics_path = beta_dir / "camera" / "intrinsics.json"
    if not frames_dir.exists():
        raise HumanExtractError(f"missing beta frames directory: {frames_dir}")

    frame_records = load_frame_index_csv(frame_index_path)
    camera_pose_map = load_camera_pose_map(camera_pose_path)
    intrinsics = load_camera_intrinsics(intrinsics_path)

    human_root = out_dir / "human"
    native_dir = human_root / "native"
    overlay_dir = human_root / "reprojection_overlays"
    work_dir = human_root / "_workdir"
    human_root.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)
    staged_frames_dir = stage_fourdhumans_source_frames(frame_records, work_dir / "_source_frames_jpg")

    native_track_path = run_fourdhumans_tracking(
        fourdhumans_root=resolved_fourdhumans_root,
        frames_dir=staged_frames_dir,
        work_dir=work_dir,
        smpl_model_path=resolved_smpl_model_path,
        device=device,
    )
    preserved_native_path = native_dir / "4dhumans_tracks.pkl"
    shutil.copyfile(native_track_path, preserved_native_path)

    native_payload = load_native_track_payload(native_track_path)
    normalized_payload = normalize_fourdhumans_tracks(
        native_payload=native_payload,
        frame_records=frame_records,
        camera_pose_map=camera_pose_map,
        video_id=bundle.project.video_id,
    )
    selection = select_primary_demonstrator_track(normalized_payload, frame_records)
    primary_track = normalized_payload["tracks"][selection.track_id]
    observables_rows = build_arm_observables_rows(
        bundle=bundle,
        frame_records=frame_records,
        primary_track=primary_track,
        camera_pose_map=camera_pose_map,
    )

    with (human_root / "smpl_tracks.pkl").open("wb") as handle:
        pickle.dump(normalized_payload, handle)
    write_observables_parquet(human_root / "arm_observables.parquet", observables_rows)
    write_reprojection_overlays(
        overlay_dir=overlay_dir,
        frame_records=frame_records,
        primary_track=primary_track,
        camera_pose_map=camera_pose_map,
        intrinsics=intrinsics,
        overlay_count=bundle.project.gamma.overlay_sample_count,
    )

    summary = build_gamma_summary(
        bundle=bundle,
        frame_records=frame_records,
        primary_track=primary_track,
        selection=selection,
        observables_rows=observables_rows,
    )
    (human_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not keep_workdir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    enforce_gamma_acceptance(bundle, summary)
    return summary
