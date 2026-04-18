from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .specs import CaptureSpec, SpecBundle


class VideoIngestError(Exception):
    """Raised when beta video ingest cannot complete successfully."""


class DependencyError(VideoIngestError):
    """Raised when runtime dependencies are missing."""


@dataclass
class DecodedFrame:
    index: int
    timestamp_sec: float
    rgb: np.ndarray


@dataclass
class FrameRecord:
    frame_index: int
    frame_name: str
    timestamp_sec: float
    path: Path
    is_keyframe: bool = False
    registered: bool = False
    pose_source: str = "missing"


@dataclass
class ColmapObservation:
    x: float
    y: float
    point3d_id: int


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width_px: int
    height_px: int
    params: list[float]


@dataclass
class ColmapPoint3D:
    point_id: int
    xyz: np.ndarray
    error: float


@dataclass
class ColmapImage:
    image_id: int
    qvec_wxyz: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    observations: list[ColmapObservation]

    @property
    def camera_from_world_rotation(self) -> np.ndarray:
        return quaternion_to_rotation_matrix(self.qvec_wxyz)

    @property
    def world_from_camera_rotation(self) -> np.ndarray:
        return self.camera_from_world_rotation.T

    @property
    def world_from_camera_translation(self) -> np.ndarray:
        return -self.camera_from_world_rotation.T @ self.tvec


@dataclass
class ParsedColmapModel:
    model_name: str
    text_dir: Path
    cameras: dict[int, ColmapCamera]
    images_by_name: dict[str, ColmapImage]
    points3d: dict[int, ColmapPoint3D]


@dataclass
class PoseEstimate:
    rotation_wc: np.ndarray
    translation_wc: np.ndarray
    source: str


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class FiducialDetection:
    frame_name: str
    marker_id: int
    rotation_tc: np.ndarray
    translation_tc: np.ndarray


@dataclass
class ReprojectionStats:
    observation_count: int
    mean_error_px: float
    max_error_px: float


@dataclass
class IngestResult:
    frame_records: list[FrameRecord]
    summary: dict[str, Any]


def _path_string(path: Path) -> str:
    return path.resolve().as_posix()


def resolve_colmap_binary(explicit: str | None = None) -> str:
    for candidate in (explicit, os.environ.get("COLMAP_BIN"), shutil.which("colmap")):
        if candidate:
            return str(candidate)
    raise DependencyError(
        "COLMAP binary not found. Install colmap or provide --colmap-bin / COLMAP_BIN."
    )


def missing_python_dependencies() -> list[str]:
    missing = []
    for module_name in ("numpy", "PIL", "imageio", "imageio_ffmpeg", "cv2"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if "cv2" not in missing:
        import cv2  # type: ignore

        if not hasattr(cv2, "aruco"):
            missing.append("cv2.aruco")

    return missing


def ensure_beta_dependencies(colmap_bin: str | None = None) -> str:
    binary = resolve_colmap_binary(colmap_bin)
    missing = missing_python_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for beta ingest: " + ", ".join(sorted(missing))
        )
    return binary


def extract_frames_from_source(
    decoded_frames: Iterable[DecodedFrame],
    frames_dir: Path,
    image_format: str,
) -> list[FrameRecord]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []
    for decoded in decoded_frames:
        frame_name = f"frame_{decoded.index:06d}.{image_format}"
        target = frames_dir / frame_name
        Image.fromarray(decoded.rgb).save(target)
        records.append(
            FrameRecord(
                frame_index=decoded.index,
                frame_name=frame_name,
                timestamp_sec=float(decoded.timestamp_sec),
                path=target,
            )
        )
    if not records:
        raise VideoIngestError("video decode produced zero frames")
    return records


def decode_video_to_frames(
    video_path: Path,
    frames_dir: Path,
    image_format: str,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoIngestError(f"unable to open video: {video_path}")

    width_px = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames: list[DecodedFrame] = []
    last_timestamp = -math.inf
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            timestamp_sec = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(timestamp_sec):
                raise VideoIngestError("decoder did not expose a usable frame timestamp")
            if timestamp_sec < last_timestamp:
                raise VideoIngestError("decoded frame timestamps were not monotonic")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded_frames.append(
                DecodedFrame(index=frame_index, timestamp_sec=timestamp_sec, rgb=frame_rgb)
            )
            last_timestamp = timestamp_sec
            frame_index += 1
    finally:
        capture.release()

    records = extract_frames_from_source(decoded_frames, frames_dir=frames_dir, image_format=image_format)
    metadata = {
        "width_px": width_px,
        "height_px": height_px,
        "nominal_fps": nominal_fps,
        "frame_count": len(records),
        "timestamp_source": "opencv_pos_msec",
    }
    return records, metadata


def select_keyframe_indices(records: Sequence[FrameRecord], sample_fps: float) -> list[int]:
    if not records:
        raise VideoIngestError("cannot select keyframes from an empty frame set")
    if len(records) == 1:
        return [0]

    interval_sec = 1.0 / sample_fps
    selected = {0, len(records) - 1}
    next_target = records[0].timestamp_sec + interval_sec

    for index, record in enumerate(records[1:-1], start=1):
        if record.timestamp_sec + 1e-9 >= next_target:
            selected.add(index)
            while next_target <= record.timestamp_sec + 1e-9:
                next_target += interval_sec

    return sorted(selected)


def copy_keyframes(records: Sequence[FrameRecord], keyframe_indices: Sequence[int], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index in keyframe_indices:
        record = records[index]
        record.is_keyframe = True
        shutil.copy2(record.path, target_dir / record.frame_name)


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown subprocess failure"
        raise VideoIngestError(f"command failed: {' '.join(command)} :: {detail}")


def parse_cameras_txt(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=parts[1],
            width_px=int(parts[2]),
            height_px=int(parts[3]),
            params=[float(value) for value in parts[4:]],
        )
    return cameras


def parse_points3d_txt(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        error = float(parts[7])
        points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, error=error)
    return points


def parse_images_txt(path: Path) -> dict[str, ColmapImage]:
    lines = [
        raw_line.rstrip("\n")
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if raw_line.strip() and not raw_line.startswith("#")
    ]
    images: dict[str, ColmapImage] = {}

    for offset in range(0, len(lines), 2):
        header = lines[offset].split()
        points_line = lines[offset + 1].split() if offset + 1 < len(lines) else []
        image_id = int(header[0])
        qvec = np.array([float(value) for value in header[1:5]], dtype=float)
        tvec = np.array([float(value) for value in header[5:8]], dtype=float)
        camera_id = int(header[8])
        name = header[9]
        observations = [
            ColmapObservation(
                x=float(points_line[index]),
                y=float(points_line[index + 1]),
                point3d_id=int(points_line[index + 2]),
            )
            for index in range(0, len(points_line), 3)
        ]
        images[name] = ColmapImage(
            image_id=image_id,
            qvec_wxyz=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            observations=observations,
        )

    return images


def load_text_model(text_dir: Path, model_name: str) -> ParsedColmapModel:
    return ParsedColmapModel(
        model_name=model_name,
        text_dir=text_dir,
        cameras=parse_cameras_txt(text_dir / "cameras.txt"),
        images_by_name=parse_images_txt(text_dir / "images.txt"),
        points3d=parse_points3d_txt(text_dir / "points3D.txt"),
    )


def choose_largest_text_model(colmap_bin: str, sparse_root: Path, text_root: Path) -> ParsedColmapModel:
    candidates = sorted(path for path in sparse_root.iterdir() if path.is_dir())
    if not candidates:
        raise VideoIngestError("COLMAP mapper did not produce any sparse models")

    best_model: ParsedColmapModel | None = None
    best_image_count = -1
    text_root.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        text_dir = text_root / candidate.name
        text_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(candidate),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ]
        )
        model = load_text_model(text_dir=text_dir, model_name=candidate.name)
        image_count = len(model.images_by_name)
        if image_count > best_image_count:
            best_model = model
            best_image_count = image_count

    if best_model is None:
        raise VideoIngestError("failed to select a COLMAP sparse model")
    return best_model


def run_colmap_pipeline(
    colmap_bin: str,
    keyframes_dir: Path,
    work_dir: Path,
    capture: CaptureSpec,
) -> ParsedColmapModel:
    database_path = work_dir / "colmap.db"
    sparse_root = work_dir / "sparse"
    text_root = work_dir / "sparse_text"
    sparse_root.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(keyframes_dir),
            "--ImageReader.camera_model",
            capture.beta.colmap.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            "0",
        ]
    )
    run_command(
        [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "0",
        ]
    )
    run_command(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(keyframes_dir),
            "--output_path",
            str(sparse_root),
        ]
    )
    return choose_largest_text_model(colmap_bin=colmap_bin, sparse_root=sparse_root, text_root=text_root)


def camera_matrix_from_colmap(camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray]:
    if camera.model != "OPENCV" or len(camera.params) < 8:
        raise VideoIngestError(
            f"unsupported camera model in COLMAP output: {camera.model}; expected OPENCV"
        )
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params[:8]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    distortion = np.array([k1, k2, p1, p2], dtype=float)
    return camera_matrix, distortion


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = quaternion_wxyz.astype(float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def slerp_quaternion(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start_unit = start / np.linalg.norm(start)
    end_unit = end / np.linalg.norm(end)
    dot = float(np.dot(start_unit, end_unit))
    if dot < 0.0:
        end_unit = -end_unit
        dot = -dot
    if dot > 0.9995:
        blended = start_unit + alpha * (end_unit - start_unit)
        return blended / np.linalg.norm(blended)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    blended = (s0 * start_unit) + (s1 * end_unit)
    return blended / np.linalg.norm(blended)


def solve_similarity_transform(source_points: np.ndarray, target_points: np.ndarray) -> SimilarityTransform:
    if source_points.shape != target_points.shape or source_points.shape[0] < 3:
        raise VideoIngestError("at least three fiducial-normalized camera poses are required")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    covariance = (target_centered.T @ source_centered) / source_points.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if variance <= 0.0:
        raise VideoIngestError("COLMAP camera centers were degenerate; cannot normalize to fiducial world")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def detect_fiducials(
    capture: CaptureSpec,
    model: ParsedColmapModel,
    keyframe_map: dict[str, FrameRecord],
) -> list[FiducialDetection]:
    import cv2  # type: ignore

    dictionary_lookup = {
        "apriltag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "aruco_4x4_50": cv2.aruco.DICT_4X4_50,
    }
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_lookup[capture.fiducial.family])
    parameters = cv2.aruco.DetectorParameters()
    detector = (
        cv2.aruco.ArucoDetector(dictionary, parameters)
        if hasattr(cv2.aruco, "ArucoDetector")
        else None
    )
    size_m = float(capture.fiducial.size_m)
    object_points = np.array(
        [
            [-size_m / 2.0, size_m / 2.0, 0.0],
            [size_m / 2.0, size_m / 2.0, 0.0],
            [size_m / 2.0, -size_m / 2.0, 0.0],
            [-size_m / 2.0, -size_m / 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    detections: list[FiducialDetection] = []
    for frame_name, image in model.images_by_name.items():
        frame_record = keyframe_map.get(frame_name)
        if frame_record is None:
            continue
        rgb = np.array(Image.open(frame_record.path).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(bgr, dictionary, parameters=parameters)
        if ids is None or len(ids) == 0:
            continue
        areas = [cv2.contourArea(corner.reshape(-1, 1, 2).astype(np.float32)) for corner in corners]
        best_index = int(np.argmax(areas))
        image_points = corners[best_index].reshape(4, 2).astype(np.float32)
        colmap_camera = model.cameras[image.camera_id]
        camera_matrix, distortion = camera_matrix_from_colmap(colmap_camera)
        solvepnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=solvepnp_flag,
        )
        if not success:
            continue
        rotation_ct, _ = cv2.Rodrigues(rvec)
        rotation_tc = rotation_ct.T
        translation_tc = (-rotation_ct.T @ tvec.reshape(3)).astype(float)
        detections.append(
            FiducialDetection(
                frame_name=frame_name,
                marker_id=int(ids[best_index][0]),
                rotation_tc=rotation_tc.astype(float),
                translation_tc=translation_tc,
            )
        )
    return detections


def normalize_registered_poses(
    model: ParsedColmapModel,
    detections: Sequence[FiducialDetection],
) -> tuple[dict[str, PoseEstimate], SimilarityTransform]:
    if len(detections) < 3:
        raise VideoIngestError(
            "fiducial normalization requires detections on at least three registered keyframes"
        )

    source_points = []
    target_points = []
    for detection in detections:
        image = model.images_by_name[detection.frame_name]
        source_points.append(image.world_from_camera_translation)
        target_points.append(detection.translation_tc)

    similarity = solve_similarity_transform(
        source_points=np.vstack(source_points),
        target_points=np.vstack(target_points),
    )

    normalized: dict[str, PoseEstimate] = {}
    for frame_name, image in model.images_by_name.items():
        rotation_wc = image.world_from_camera_rotation
        translation_wc = image.world_from_camera_translation
        normalized_rotation = similarity.rotation @ rotation_wc
        normalized_translation = similarity.scale * (similarity.rotation @ translation_wc) + similarity.translation
        normalized[frame_name] = PoseEstimate(
            rotation_wc=normalized_rotation,
            translation_wc=normalized_translation,
            source="colmap",
        )
    return normalized, similarity


def build_dense_pose_map(
    records: Sequence[FrameRecord],
    direct_pose_map: dict[str, PoseEstimate],
    max_interpolation_gap_s: float,
) -> dict[int, PoseEstimate]:
    direct_indices = [
        record.frame_index for record in records if record.frame_name in direct_pose_map
    ]
    if not direct_indices:
        raise VideoIngestError("COLMAP did not register any keyframes")
    if direct_indices[0] != 0 or direct_indices[-1] != records[-1].frame_index:
        raise VideoIngestError(
            "dense pose export requires the first and last extracted frames to be registered"
        )

    dense: dict[int, PoseEstimate] = {}
    for record in records:
        direct = direct_pose_map.get(record.frame_name)
        if direct is not None:
            dense[record.frame_index] = direct
            record.registered = True
            record.pose_source = "colmap"

    for start_index, end_index in zip(direct_indices, direct_indices[1:]):
        start_record = records[start_index]
        end_record = records[end_index]
        gap_sec = end_record.timestamp_sec - start_record.timestamp_sec
        if gap_sec > max_interpolation_gap_s:
            raise VideoIngestError(
                f"interpolation gap of {gap_sec:.3f}s exceeded max_interpolation_gap_s={max_interpolation_gap_s:.3f}s"
            )

        start_pose = dense[start_index]
        end_pose = dense[end_index]
        start_quaternion = rotation_matrix_to_quaternion(start_pose.rotation_wc)
        end_quaternion = rotation_matrix_to_quaternion(end_pose.rotation_wc)
        for frame_index in range(start_index + 1, end_index):
            record = records[frame_index]
            if gap_sec <= 1e-9:
                alpha = 0.0
            else:
                alpha = (record.timestamp_sec - start_record.timestamp_sec) / gap_sec
            quaternion = slerp_quaternion(start_quaternion, end_quaternion, alpha)
            dense[frame_index] = PoseEstimate(
                rotation_wc=quaternion_to_rotation_matrix(quaternion),
                translation_wc=((1.0 - alpha) * start_pose.translation_wc) + (alpha * end_pose.translation_wc),
                source="interpolated",
            )
            record.pose_source = "interpolated"

    missing_frames = [record.frame_name for record in records if record.frame_index not in dense]
    if missing_frames:
        raise VideoIngestError("dense pose interpolation left unresolved frames: " + ", ".join(missing_frames[:5]))

    return dense


def project_point(camera: ColmapCamera, image: ColmapImage, point_world: np.ndarray) -> np.ndarray | None:
    rotation_cw = image.camera_from_world_rotation
    translation_cw = image.tvec
    point_camera = rotation_cw @ point_world + translation_cw
    if point_camera[2] <= 0.0:
        return None

    x = point_camera[0] / point_camera[2]
    y = point_camera[1] / point_camera[2]
    _, params = camera_matrix_from_colmap(camera)
    fx, fy, cx, cy = camera.params[:4]
    k1, k2, p1, p2 = params
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_distorted = x * radial + (2.0 * p1 * x * y) + (p2 * (r2 + 2.0 * x * x))
    y_distorted = y * radial + (p1 * (r2 + 2.0 * y * y)) + (2.0 * p2 * x * y)
    return np.array([fx * x_distorted + cx, fy * y_distorted + cy], dtype=float)


def compute_reprojection_statistics(model: ParsedColmapModel) -> ReprojectionStats:
    errors: list[float] = []
    for image in model.images_by_name.values():
        camera = model.cameras[image.camera_id]
        for observation in image.observations:
            if observation.point3d_id < 0:
                continue
            point = model.points3d.get(observation.point3d_id)
            if point is None:
                continue
            projection = project_point(camera, image, point.xyz)
            if projection is None:
                continue
            observed = np.array([observation.x, observation.y], dtype=float)
            errors.append(float(np.linalg.norm(projection - observed)))

    if not errors:
        return ReprojectionStats(observation_count=0, mean_error_px=math.inf, max_error_px=math.inf)
    return ReprojectionStats(
        observation_count=len(errors),
        mean_error_px=float(np.mean(errors)),
        max_error_px=float(np.max(errors)),
    )


def make_reprojection_preview(
    output_path: Path,
    model: ParsedColmapModel,
    keyframe_map: dict[str, FrameRecord],
    max_images: int = 6,
    max_points: int = 150,
) -> None:
    registered_names = [name for name in keyframe_map if name in model.images_by_name]
    if not registered_names:
        placeholder = Image.new("RGB", (640, 360), color=(245, 245, 245))
        draw = ImageDraw.Draw(placeholder)
        draw.text((24, 24), "No registered keyframes available for reprojection preview.", fill=(20, 20, 20))
        placeholder.save(output_path, quality=90)
        return

    if len(registered_names) > max_images:
        step = max(1, len(registered_names) // max_images)
        preview_names = registered_names[::step][:max_images]
    else:
        preview_names = registered_names

    tiles: list[Image.Image] = []
    for frame_name in preview_names:
        image = model.images_by_name[frame_name]
        frame_record = keyframe_map[frame_name]
        tile = Image.open(frame_record.path).convert("RGB")
        draw = ImageDraw.Draw(tile)
        valid_observations = [obs for obs in image.observations if obs.point3d_id in model.points3d]
        if len(valid_observations) > max_points:
            step = max(1, len(valid_observations) // max_points)
            valid_observations = valid_observations[::step][:max_points]
        camera = model.cameras[image.camera_id]
        for observation in valid_observations:
            point = model.points3d[observation.point3d_id]
            projected = project_point(camera, image, point.xyz)
            if projected is None:
                continue
            u, v = float(projected[0]), float(projected[1])
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(36, 180, 96))
            draw.line((observation.x - 2, observation.y, observation.x + 2, observation.y), fill=(220, 30, 30), width=1)
            draw.line((observation.x, observation.y - 2, observation.x, observation.y + 2), fill=(220, 30, 30), width=1)
        draw.rectangle((0, 0, tile.width, 24), fill=(0, 0, 0))
        draw.text((8, 4), frame_name, fill=(255, 255, 255))
        tile.thumbnail((480, 270))
        tiles.append(tile)

    columns = 2 if len(tiles) > 1 else 1
    rows = math.ceil(len(tiles) / columns)
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(255, 255, 255))

    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(tile, (x, y))

    canvas.save(output_path, quality=90)


def colmap_intrinsics_to_json(camera: ColmapCamera) -> dict[str, Any]:
    if camera.model != "OPENCV" or len(camera.params) < 8:
        raise VideoIngestError("cannot export intrinsics for unsupported COLMAP camera model")
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params[:8]
    return {
        "source": "colmap",
        "camera_model": camera.model,
        "width_px": camera.width_px,
        "height_px": camera.height_px,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_params": {
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
        },
    }


def pose_to_matrix(rotation_wc: np.ndarray, translation_wc: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_wc
    matrix[:3, 3] = translation_wc
    return matrix.tolist()


def write_timestamps_csv(path: Path, records: Sequence[FrameRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "frame_name", "timestamp_sec", "is_keyframe", "registered", "pose_source"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame_index": record.frame_index,
                    "frame_name": record.frame_name,
                    "timestamp_sec": f"{record.timestamp_sec:.9f}",
                    "is_keyframe": str(record.is_keyframe).lower(),
                    "registered": str(record.registered).lower(),
                    "pose_source": record.pose_source,
                }
            )


def write_camera_intrinsics_json(path: Path, camera: ColmapCamera) -> None:
    path.write_text(json.dumps(colmap_intrinsics_to_json(camera), indent=2) + "\n", encoding="utf-8")


def write_camera_poses_json(path: Path, records: Sequence[FrameRecord], dense_pose_map: dict[int, PoseEstimate]) -> None:
    payload = {
        "coordinate_frame": "fiducial_world",
        "units": "meters",
        "frame_count": len(records),
        "poses": [],
    }
    for record in records:
        pose = dense_pose_map[record.frame_index]
        payload["poses"].append(
            {
                "frame_index": record.frame_index,
                "frame_name": record.frame_name,
                "timestamp_sec": record.timestamp_sec,
                "is_keyframe": record.is_keyframe,
                "pose_source": pose.source,
                "translation_m": pose.translation_wc.tolist(),
                "quaternion_wxyz": rotation_matrix_to_quaternion(pose.rotation_wc).tolist(),
                "world_from_camera_4x4": pose_to_matrix(pose.rotation_wc, pose.translation_wc),
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_summary(
    bundle: SpecBundle,
    video_path: Path,
    decode_metadata: dict[str, Any],
    records: Sequence[FrameRecord],
    model: ParsedColmapModel,
    detections: Sequence[FiducialDetection],
    similarity: SimilarityTransform,
    reprojection_stats: ReprojectionStats,
    dense_pose_map: dict[int, PoseEstimate],
) -> dict[str, Any]:
    keyframe_count = sum(1 for record in records if record.is_keyframe)
    registered_keyframe_count = sum(1 for record in records if record.is_keyframe and record.registered)
    registered_ratio = registered_keyframe_count / keyframe_count if keyframe_count else 0.0
    interpolation_count = sum(1 for pose in dense_pose_map.values() if pose.source == "interpolated")
    largest_gap = 0.0
    direct_records = [record for record in records if record.registered]
    for earlier, later in zip(direct_records, direct_records[1:]):
        largest_gap = max(largest_gap, later.timestamp_sec - earlier.timestamp_sec)

    mismatches = []
    if decode_metadata["width_px"] != bundle.capture.resolution.width:
        mismatches.append(
            f"decoded width {decode_metadata['width_px']}px differed from capture contract {bundle.capture.resolution.width}px"
        )
    if decode_metadata["height_px"] != bundle.capture.resolution.height:
        mismatches.append(
            f"decoded height {decode_metadata['height_px']}px differed from capture contract {bundle.capture.resolution.height}px"
        )
    if decode_metadata["nominal_fps"] and abs(decode_metadata["nominal_fps"] - bundle.capture.fps) > 1e-3:
        mismatches.append(
            f"decoded nominal fps {decode_metadata['nominal_fps']:.3f} differed from capture contract {bundle.capture.fps:.3f}"
        )

    return {
        "video": {
            "path": _path_string(video_path),
            "frame_count": len(records),
            "width_px": decode_metadata["width_px"],
            "height_px": decode_metadata["height_px"],
            "nominal_fps": decode_metadata["nominal_fps"],
            "timestamp_source": decode_metadata["timestamp_source"],
        },
        "colmap": {
            "selected_model": model.model_name,
            "registered_keyframes": registered_keyframe_count,
            "total_keyframes": keyframe_count,
            "registered_ratio": registered_ratio,
            "mean_reprojection_error_px": reprojection_stats.mean_error_px,
            "max_reprojection_error_px": reprojection_stats.max_error_px,
            "observation_count": reprojection_stats.observation_count,
        },
        "normalization": {
            "status": "ok",
            "world_frame": bundle.capture.beta.normalization.world_frame,
            "fiducial_family": bundle.capture.fiducial.family,
            "detections_used": len(detections),
            "scale_m_per_colmap_unit": similarity.scale,
        },
        "pose_coverage": {
            "total_frames": len(records),
            "registered_frames": sum(1 for record in records if record.registered),
            "interpolated_frames": interpolation_count,
            "largest_interpolation_gap_s": largest_gap,
            "edge_coverage_ok": records[0].registered and records[-1].registered,
        },
        "capture_contract_mismatches": mismatches,
    }


def enforce_stability_thresholds(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    colmap_summary = summary["colmap"]
    thresholds = bundle.capture.beta.colmap
    if colmap_summary["registered_keyframes"] < thresholds.min_registered_keyframes:
        raise VideoIngestError(
            f"registered keyframes {colmap_summary['registered_keyframes']} fell below min_registered_keyframes={thresholds.min_registered_keyframes}"
        )
    if colmap_summary["registered_ratio"] < thresholds.min_registered_ratio:
        raise VideoIngestError(
            f"registered ratio {colmap_summary['registered_ratio']:.3f} fell below min_registered_ratio={thresholds.min_registered_ratio:.3f}"
        )
    if colmap_summary["mean_reprojection_error_px"] > thresholds.max_mean_reprojection_error_px:
        raise VideoIngestError(
            f"mean reprojection error {colmap_summary['mean_reprojection_error_px']:.3f}px exceeded max_mean_reprojection_error_px={thresholds.max_mean_reprojection_error_px:.3f}px"
        )
    if not summary["pose_coverage"]["edge_coverage_ok"]:
        raise VideoIngestError("pose coverage did not include the first and last extracted frames")
    if summary["pose_coverage"]["largest_interpolation_gap_s"] > thresholds.max_interpolation_gap_s:
        raise VideoIngestError(
            f"largest interpolation gap {summary['pose_coverage']['largest_interpolation_gap_s']:.3f}s exceeded max_interpolation_gap_s={thresholds.max_interpolation_gap_s:.3f}s"
        )


def ingest_monocular_video(
    bundle: SpecBundle,
    video_path: Path,
    out_dir: Path,
    colmap_bin: str | None = None,
    keep_workdir: bool = False,
) -> IngestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    work_dir = out_dir / "_workdir"
    keyframes_dir = work_dir / "keyframes"
    preview_path = out_dir / "reprojection_preview.jpg"
    summary_path = out_dir / "reconstruction_summary.json"

    colmap_binary = ensure_beta_dependencies(colmap_bin)
    records, decode_metadata = decode_video_to_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        image_format=bundle.capture.beta.frame_extraction.image_format,
    )
    keyframe_indices = select_keyframe_indices(
        records=records,
        sample_fps=bundle.capture.beta.frame_extraction.keyframe_sample_fps,
    )
    copy_keyframes(records=records, keyframe_indices=keyframe_indices, target_dir=keyframes_dir)

    summary: dict[str, Any] | None = None
    try:
        model = run_colmap_pipeline(
            colmap_bin=colmap_binary,
            keyframes_dir=keyframes_dir,
            work_dir=work_dir,
            capture=bundle.capture,
        )
        direct_keyframe_map = {record.frame_name: record for record in records if record.is_keyframe}
        detections = detect_fiducials(capture=bundle.capture, model=model, keyframe_map=direct_keyframe_map)
        normalized_pose_map, similarity = normalize_registered_poses(model=model, detections=detections)
        dense_pose_map = build_dense_pose_map(
            records=records,
            direct_pose_map=normalized_pose_map,
            max_interpolation_gap_s=bundle.capture.beta.colmap.max_interpolation_gap_s,
        )
        reprojection_stats = compute_reprojection_statistics(model=model)
        make_reprojection_preview(output_path=preview_path, model=model, keyframe_map=direct_keyframe_map)

        camera = next(iter(model.cameras.values()))
        write_timestamps_csv(out_dir / "timestamps.csv", records)
        write_camera_intrinsics_json(out_dir / "camera_intrinsics.json", camera)
        write_camera_poses_json(out_dir / "camera_poses.json", records, dense_pose_map)

        summary = build_summary(
            bundle=bundle,
            video_path=video_path,
            decode_metadata=decode_metadata,
            records=records,
            model=model,
            detections=detections,
            similarity=similarity,
            reprojection_stats=reprojection_stats,
            dense_pose_map=dense_pose_map,
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        enforce_stability_thresholds(bundle=bundle, summary=summary)
        return IngestResult(frame_records=records, summary=summary)
    finally:
        if not keep_workdir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

