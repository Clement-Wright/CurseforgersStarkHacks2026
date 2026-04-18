from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .specs import PROJECT_SCHEMA_VERSION, CaptureSpec, SpecBundle


class VideoIngestError(Exception):
    """Raised when beta video ingest cannot complete successfully."""


class DependencyError(VideoIngestError):
    """Raised when runtime dependencies are missing."""


@dataclass
class DecodedFrame:
    index: int
    pts: int
    pts_sec: float
    t_ns: int
    rgb: np.ndarray


@dataclass
class FrameRecord:
    frame_index: int
    frame_name: str
    pts: int
    pts_sec: float
    t_ns: int
    image_path: Path
    segment: str = "unassigned"
    is_keyframe: bool = False
    registered: bool = False
    pose_status: str = "unlocalized"
    colmap_image_id: int | None = None


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
    rgb: tuple[int, int, int]
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
class ColmapPipelineResult:
    base_model: ParsedColmapModel
    final_model: ParsedColmapModel
    database_path: Path
    sparse_root: Path
    sparse_text_root: Path
    component_count: int
    staging_dir: Path


@dataclass
class PoseEstimate:
    rotation_wc: np.ndarray
    translation_wc: np.ndarray
    source: str
    colmap_image_id: int | None = None


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    orientation_spread_deg: float
    inlier_count: int


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
    missing: list[str] = []
    for module_name in ("av", "numpy", "PIL", "cv2", "yaml"):
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


def timestamp_seconds_from_pts(pts: int, time_base: Any) -> float:
    return float(pts * time_base)


def relative_timestamp_ns(absolute_sec: float, origin_sec: float) -> int:
    return int(round((absolute_sec - origin_sec) * 1_000_000_000))


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
                pts=decoded.pts,
                pts_sec=float(decoded.pts_sec),
                t_ns=int(decoded.t_ns),
                image_path=target,
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
    import av  # type: ignore

    decoded_frames: list[DecodedFrame] = []
    width_px = 0
    height_px = 0
    nominal_fps = 0.0
    first_absolute_sec: float | None = None
    last_relative_sec = -math.inf

    try:
        with av.open(str(video_path)) as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                raise VideoIngestError(f"no video stream found in {video_path}")
            if video_stream.average_rate is not None:
                nominal_fps = float(video_stream.average_rate)

            frame_index = 0
            for frame in container.decode(video_stream):
                if frame.pts is None:
                    raise VideoIngestError("decoded frame did not expose a container PTS")
                time_base = frame.time_base or video_stream.time_base
                if time_base is None:
                    raise VideoIngestError("decoded frame did not expose a usable time_base")

                absolute_sec = timestamp_seconds_from_pts(int(frame.pts), time_base)
                if first_absolute_sec is None:
                    first_absolute_sec = absolute_sec
                relative_sec = absolute_sec - first_absolute_sec
                if relative_sec + 1e-12 < last_relative_sec:
                    raise VideoIngestError("decoded frame timestamps were not monotonic")

                rgb = frame.to_ndarray(format="rgb24")
                width_px = int(frame.width)
                height_px = int(frame.height)
                decoded_frames.append(
                    DecodedFrame(
                        index=frame_index,
                        pts=int(frame.pts),
                        pts_sec=relative_sec,
                        t_ns=relative_timestamp_ns(absolute_sec, first_absolute_sec),
                        rgb=rgb,
                    )
                )
                last_relative_sec = relative_sec
                frame_index += 1
    except VideoIngestError:
        raise
    except Exception as exc:  # pragma: no cover - exercised in live runs
        raise VideoIngestError(f"unable to decode video with PyAV: {exc}") from exc

    records = extract_frames_from_source(decoded_frames, frames_dir=frames_dir, image_format=image_format)
    metadata = {
        "width_px": width_px,
        "height_px": height_px,
        "nominal_fps": nominal_fps,
        "frame_count": len(records),
        "timestamp_source": "pyav_pts_time_base",
        "duration_sec": records[-1].pts_sec if records else 0.0,
    }
    return records, metadata


def assign_frame_segments(
    records: Sequence[FrameRecord],
    preroll_seconds: float,
    postroll_seconds: float = 0.0,
) -> None:
    if not records:
        raise VideoIngestError("cannot segment an empty frame sequence")

    clip_end_sec = records[-1].pts_sec
    for record in records:
        if record.pts_sec <= preroll_seconds + 1e-9:
            record.segment = "preroll"
        elif postroll_seconds > 0.0 and (clip_end_sec - record.pts_sec) <= postroll_seconds + 1e-9:
            record.segment = "postroll"
        else:
            record.segment = "demo"


def select_keyframe_indices(
    records: Sequence[FrameRecord],
    sample_fps: float,
    segment: str = "preroll",
) -> list[int]:
    segment_indices = [index for index, record in enumerate(records) if record.segment == segment]
    if not segment_indices:
        raise VideoIngestError(f"cannot select keyframes because no frames were tagged as '{segment}'")
    if len(segment_indices) == 1:
        return segment_indices

    interval_sec = 1.0 / sample_fps
    selected = {segment_indices[0], segment_indices[-1]}
    next_target = records[segment_indices[0]].pts_sec + interval_sec

    for index in segment_indices[1:-1]:
        record = records[index]
        if record.pts_sec + 1e-9 >= next_target:
            selected.add(index)
            while next_target <= record.pts_sec + 1e-9:
                next_target += interval_sec

    return sorted(selected)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def stage_frame_subset(records: Sequence[FrameRecord], target_dir: Path, segment_names: set[str]) -> list[FrameRecord]:
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = [record for record in records if record.segment in segment_names or record.is_keyframe and "keyframes" in segment_names]
    for record in selected:
        _link_or_copy(record.image_path, target_dir / record.frame_name)
    return selected


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
        rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
        error = float(parts[7])
        points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error)
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


def convert_model_to_text(colmap_bin: str, input_dir: Path, output_dir: Path) -> ParsedColmapModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(input_dir),
            "--output_path",
            str(output_dir),
            "--output_type",
            "TXT",
        ]
    )
    return load_text_model(output_dir, input_dir.name)


def choose_largest_sparse_component(
    colmap_bin: str,
    sparse_candidates_root: Path,
    text_candidates_root: Path,
) -> tuple[ParsedColmapModel, Path, int]:
    candidates = sorted(path for path in sparse_candidates_root.iterdir() if path.is_dir())
    if not candidates:
        raise VideoIngestError("COLMAP mapper did not produce any sparse models")

    best_model: ParsedColmapModel | None = None
    best_sparse_dir: Path | None = None
    best_image_count = -1

    for candidate in candidates:
        model = convert_model_to_text(colmap_bin, candidate, text_candidates_root / candidate.name)
        image_count = len(model.images_by_name)
        if image_count > best_image_count:
            best_model = model
            best_sparse_dir = candidate
            best_image_count = image_count

    if best_model is None or best_sparse_dir is None:
        raise VideoIngestError("failed to choose a COLMAP sparse component")
    return best_model, best_sparse_dir, len(candidates)


def run_colmap_pipeline(
    colmap_bin: str,
    frames_dir: Path,
    records: Sequence[FrameRecord],
    colmap_root: Path,
    capture: CaptureSpec,
) -> ColmapPipelineResult:
    database_path = colmap_root / "database.db"
    sparse_root = colmap_root / "sparse"
    sparse_text_root = colmap_root / "sparse_txt"
    staging_dir = colmap_root / "_staging"
    mapper_candidates_dir = staging_dir / "mapper_candidates"
    mapper_text_dir = staging_dir / "mapper_text"
    preroll_dir = staging_dir / "preroll_images"
    localization_dir = staging_dir / "localization_images"
    base_model_dir = sparse_root / "base"
    final_model_dir = sparse_root / "final"
    base_text_dir = sparse_text_root / "base"
    final_text_dir = sparse_text_root / "final"

    for path in (sparse_root, sparse_text_root, staging_dir):
        path.mkdir(parents=True, exist_ok=True)

    preroll_records = [record for record in records if record.is_keyframe]
    localization_records = [record for record in records if record.segment != "preroll"]
    if not preroll_records:
        raise VideoIngestError("no pre-roll keyframes were available for COLMAP reconstruction")

    stage_frame_subset(preroll_records, preroll_dir, {"preroll", "keyframes"})
    stage_frame_subset(localization_records, localization_dir, {"demo", "postroll"})

    run_command(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(frames_dir),
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
            str(preroll_dir),
            "--output_path",
            str(mapper_candidates_dir),
        ]
    )

    base_model, base_sparse_candidate, component_count = choose_largest_sparse_component(
        colmap_bin=colmap_bin,
        sparse_candidates_root=mapper_candidates_dir,
        text_candidates_root=mapper_text_dir,
    )
    if base_model_dir.exists():
        shutil.rmtree(base_model_dir)
    shutil.copytree(base_sparse_candidate, base_model_dir)
    if base_text_dir.exists():
        shutil.rmtree(base_text_dir)
    shutil.copytree(base_model.text_dir, base_text_dir)
    base_model = load_text_model(base_text_dir, "base")

    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)

    if any(localization_dir.iterdir()):
        run_command(
            [
                colmap_bin,
                "image_registrator",
                "--database_path",
                str(database_path),
                "--image_path",
                str(localization_dir),
                "--input_path",
                str(base_model_dir),
                "--output_path",
                str(final_model_dir),
            ]
        )
    else:
        shutil.copytree(base_model_dir, final_model_dir)

    final_model = convert_model_to_text(colmap_bin, final_model_dir, final_text_dir)
    return ColmapPipelineResult(
        base_model=base_model,
        final_model=final_model,
        database_path=database_path,
        sparse_root=sparse_root,
        sparse_text_root=sparse_text_root,
        component_count=component_count,
        staging_dir=staging_dir,
    )


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


def rotation_geodesic_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = lhs @ rhs.T
    trace = float(np.trace(delta))
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((3, 3), dtype=float)
    for rotation in rotations:
        accumulator += rotation
    u, _, vt = np.linalg.svd(accumulator)
    averaged = u @ vt
    if np.linalg.det(averaged) < 0.0:
        u[:, -1] *= -1.0
        averaged = u @ vt
    return averaged


def solve_scale_and_translation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray]:
    rotated_source = (rotation @ source_points.T).T
    source_mean = rotated_source.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = rotated_source - source_mean
    target_centered = target_points - target_mean
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 0.0:
        raise VideoIngestError("COLMAP camera centers were degenerate; cannot solve metric scale")
    scale = float(np.sum(source_centered * target_centered) / denominator)
    translation = target_mean - scale * source_mean
    return scale, translation


def detect_fiducials(
    capture: CaptureSpec,
    model: ParsedColmapModel,
    record_map: dict[str, FrameRecord],
) -> list[FiducialDetection]:
    import cv2  # type: ignore

    dictionary_lookup = {
        "apriltag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "aruco_4x4_50": cv2.aruco.DICT_4X4_50,
    }
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_lookup[capture.fiducial.family])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None
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
        frame_record = record_map.get(frame_name)
        if frame_record is None:
            continue
        rgb = np.array(Image.open(frame_record.image_path).convert("RGB"))
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
        rotation_tc = rotation_ct.T.astype(float)
        translation_tc = (-rotation_ct.T @ tvec.reshape(3)).astype(float)
        detections.append(
            FiducialDetection(
                frame_name=frame_name,
                marker_id=int(ids[best_index][0]),
                rotation_tc=rotation_tc,
                translation_tc=translation_tc,
            )
        )
    return detections


def normalize_registered_poses(
    model: ParsedColmapModel,
    detections: Sequence[FiducialDetection],
    fiducial_size_m: float,
) -> tuple[dict[str, PoseEstimate], SimilarityTransform]:
    if len(detections) < 3:
        raise VideoIngestError(
            "fiducial normalization requires detections on at least three registered frames"
        )

    observation_rows: list[tuple[FiducialDetection, ColmapImage, np.ndarray]] = []
    for detection in detections:
        image = model.images_by_name.get(detection.frame_name)
        if image is None:
            continue
        candidate_rotation = detection.rotation_tc @ image.world_from_camera_rotation.T
        observation_rows.append((detection, image, candidate_rotation))

    if len(observation_rows) < 3:
        raise VideoIngestError(
            "fiducial normalization requires at least three detections with COLMAP poses"
        )

    orientation_tolerance_deg = 5.0
    residual_tolerance_m = max(fiducial_size_m * 0.25, 0.02)

    candidate_rotations = [row[2] for row in observation_rows]
    average_rotation = average_rotations(candidate_rotations)
    orientation_spreads = [rotation_geodesic_deg(rotation, average_rotation) for rotation in candidate_rotations]
    rotation_inliers = [
        row for row, spread in zip(observation_rows, orientation_spreads) if spread <= orientation_tolerance_deg
    ]
    if len(rotation_inliers) < 3:
        raise VideoIngestError("fiducial orientation estimates were too inconsistent to define the world frame")

    average_rotation = average_rotations([row[2] for row in rotation_inliers])
    source_points = np.vstack([row[1].world_from_camera_translation for row in rotation_inliers])
    target_points = np.vstack([row[0].translation_tc for row in rotation_inliers])
    scale, translation = solve_scale_and_translation(source_points, target_points, average_rotation)
    residuals = np.linalg.norm(
        ((scale * (average_rotation @ source_points.T)).T + translation) - target_points,
        axis=1,
    )
    residual_inliers = [row for row, residual in zip(rotation_inliers, residuals) if residual <= residual_tolerance_m]
    if len(residual_inliers) < 3:
        raise VideoIngestError("fiducial normalization rejected too many detections during residual filtering")

    average_rotation = average_rotations([row[2] for row in residual_inliers])
    source_points = np.vstack([row[1].world_from_camera_translation for row in residual_inliers])
    target_points = np.vstack([row[0].translation_tc for row in residual_inliers])
    scale, translation = solve_scale_and_translation(source_points, target_points, average_rotation)
    final_spread = max(
        rotation_geodesic_deg(row[2], average_rotation)
        for row in residual_inliers
    )

    normalized: dict[str, PoseEstimate] = {}
    for frame_name, image in model.images_by_name.items():
        rotation_wc = image.world_from_camera_rotation
        translation_wc = image.world_from_camera_translation
        normalized_rotation = average_rotation @ rotation_wc
        normalized_translation = scale * (average_rotation @ translation_wc) + translation
        normalized[frame_name] = PoseEstimate(
            rotation_wc=normalized_rotation,
            translation_wc=normalized_translation,
            source="colmap",
            colmap_image_id=image.image_id,
        )

    similarity = SimilarityTransform(
        scale=scale,
        rotation=average_rotation,
        translation=translation,
        orientation_spread_deg=final_spread,
        inlier_count=len(residual_inliers),
    )
    return normalized, similarity


def build_pose_timeline(
    records: Sequence[FrameRecord],
    direct_pose_map: dict[str, PoseEstimate],
    max_interpolation_gap_s: float,
) -> dict[int, PoseEstimate]:
    dense: dict[int, PoseEstimate] = {}
    direct_indices: list[int] = []

    for record in records:
        record.registered = False
        record.pose_status = "unlocalized"
        record.colmap_image_id = None
        direct = direct_pose_map.get(record.frame_name)
        if direct is None:
            continue
        dense[record.frame_index] = direct
        direct_indices.append(record.frame_index)
        record.registered = True
        record.pose_status = "registered"
        record.colmap_image_id = direct.colmap_image_id

    for start_index, end_index in zip(direct_indices, direct_indices[1:]):
        if end_index - start_index <= 1:
            continue
        start_record = records[start_index]
        end_record = records[end_index]
        gap_sec = end_record.pts_sec - start_record.pts_sec
        if gap_sec > max_interpolation_gap_s:
            continue

        start_pose = dense[start_index]
        end_pose = dense[end_index]
        start_quaternion = rotation_matrix_to_quaternion(start_pose.rotation_wc)
        end_quaternion = rotation_matrix_to_quaternion(end_pose.rotation_wc)

        for frame_index in range(start_index + 1, end_index):
            record = records[frame_index]
            if record.frame_name in direct_pose_map:
                continue
            if gap_sec <= 1e-9:
                alpha = 0.0
            else:
                alpha = (record.pts_sec - start_record.pts_sec) / gap_sec
            quaternion = slerp_quaternion(start_quaternion, end_quaternion, alpha)
            dense[frame_index] = PoseEstimate(
                rotation_wc=quaternion_to_rotation_matrix(quaternion),
                translation_wc=((1.0 - alpha) * start_pose.translation_wc) + (alpha * end_pose.translation_wc),
                source="interpolated",
                colmap_image_id=None,
            )
            record.pose_status = "interpolated"

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
    record_map: dict[str, FrameRecord],
    max_images: int = 6,
    max_points: int = 150,
) -> None:
    registered_names = [name for name in record_map if name in model.images_by_name]
    if not registered_names:
        placeholder = Image.new("RGB", (640, 360), color=(245, 245, 245))
        draw = ImageDraw.Draw(placeholder)
        draw.text((24, 24), "No registered frames available for reprojection preview.", fill=(20, 20, 20))
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
        frame_record = record_map[frame_name]
        tile = Image.open(frame_record.image_path).convert("RGB")
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


def camera_from_world_matrix(rotation_wc: np.ndarray, translation_wc: np.ndarray) -> list[list[float]]:
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_wc.T @ translation_wc
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_cw
    matrix[:3, 3] = translation_cw
    return matrix.tolist()


def write_frame_index_csv(path: Path, records: Sequence[FrameRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
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
        for record in records:
            writer.writerow(
                {
                    "frame_idx": record.frame_index,
                    "frame_name": record.frame_name,
                    "image_path": _path_string(record.image_path),
                    "pts": record.pts,
                    "pts_sec": f"{record.pts_sec:.9f}",
                    "t_ns": record.t_ns,
                    "segment": record.segment,
                    "is_keyframe": str(record.is_keyframe).lower(),
                    "registered": str(record.registered).lower(),
                    "pose_status": record.pose_status,
                    "colmap_image_id": "" if record.colmap_image_id is None else record.colmap_image_id,
                }
            )


def write_legacy_timestamps_csv(path: Path, records: Sequence[FrameRecord]) -> None:
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
                    "timestamp_sec": f"{record.pts_sec:.9f}",
                    "is_keyframe": str(record.is_keyframe).lower(),
                    "registered": str(record.registered).lower(),
                    "pose_source": record.pose_status,
                }
            )


def write_camera_intrinsics_json(path: Path, bundle: SpecBundle, camera: ColmapCamera) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "camera": colmap_intrinsics_to_json(camera),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_intrinsics_opencv_yaml(path: Path, camera: ColmapCamera) -> None:
    intrinsics = colmap_intrinsics_to_json(camera)
    payload = {
        "camera_model": intrinsics["camera_model"],
        "image_width": intrinsics["width_px"],
        "image_height": intrinsics["height_px"],
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                intrinsics["fx"],
                0.0,
                intrinsics["cx"],
                0.0,
                intrinsics["fy"],
                intrinsics["cy"],
                0.0,
                0.0,
                1.0,
            ],
        },
        "distortion_model": "opencv",
        "distortion_coefficients": {
            "rows": 1,
            "cols": 4,
            "data": [
                intrinsics["distortion_params"]["k1"],
                intrinsics["distortion_params"]["k2"],
                intrinsics["distortion_params"]["p1"],
                intrinsics["distortion_params"]["p2"],
            ],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_camera_poses_json(
    path: Path,
    bundle: SpecBundle,
    records: Sequence[FrameRecord],
    pose_map: dict[int, PoseEstimate],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "coordinate_frame": "fiducial_world",
        "camera_frame_convention": "COLMAP/OpenCV optical frame: +x right, +y down, +z forward",
        "time_base": "ns_from_first_decoded_frame",
        "units": "meters",
        "frames": [],
    }
    for record in records:
        pose = pose_map.get(record.frame_index)
        entry: dict[str, Any] = {
            "frame_idx": record.frame_index,
            "image_path": _path_string(record.image_path),
            "pts": record.pts,
            "pts_sec": record.pts_sec,
            "t_ns": record.t_ns,
            "segment": record.segment,
            "is_keyframe": record.is_keyframe,
            "registered": record.registered,
            "pose_status": record.pose_status,
            "colmap_image_id": record.colmap_image_id,
        }
        if pose is not None:
            entry["T_cw"] = camera_from_world_matrix(pose.rotation_wc, pose.translation_wc)
            entry["T_wc"] = pose_to_matrix(pose.rotation_wc, pose.translation_wc)
            entry["camera_center_w"] = pose.translation_wc.tolist()
            entry["quaternion_wxyz"] = rotation_matrix_to_quaternion(pose.rotation_wc).tolist()
        payload["frames"].append(entry)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_sparse_points_ply(path: Path, model: ParsedColmapModel) -> None:
    points = list(model.points3d.values())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(
                f"{point.xyz[0]} {point.xyz[1]} {point.xyz[2]} {point.rgb[0]} {point.rgb[1]} {point.rgb[2]}\n"
            )


def build_summary(
    bundle: SpecBundle,
    video_path: Path,
    decode_metadata: dict[str, Any],
    records: Sequence[FrameRecord],
    pipeline: ColmapPipelineResult,
    detections: Sequence[FiducialDetection],
    similarity: SimilarityTransform,
    reprojection_stats: ReprojectionStats,
    pose_map: dict[int, PoseEstimate],
) -> dict[str, Any]:
    preroll_keyframes = [record for record in records if record.segment == "preroll" and record.is_keyframe]
    registered_preroll_keyframes = [record for record in preroll_keyframes if record.registered]
    localized_non_preroll = [record for record in records if record.segment != "preroll" and record.registered]
    interpolated_frames = [record for record in records if record.pose_status == "interpolated"]
    unlocalized_frames = [record for record in records if record.pose_status == "unlocalized"]
    direct_records = [record for record in records if record.registered]

    largest_interpolation_gap_s = 0.0
    for earlier, later in zip(direct_records, direct_records[1:]):
        if later.frame_index - earlier.frame_index > 1:
            largest_interpolation_gap_s = max(
                largest_interpolation_gap_s,
                later.pts_sec - earlier.pts_sec,
            )

    mismatches: list[str] = []
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

    registered_fraction = (
        len(registered_preroll_keyframes) / len(preroll_keyframes) if preroll_keyframes else 0.0
    )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video": {
            "id": bundle.project.video_id,
            "path": _path_string(video_path),
            "frame_count": len(records),
            "width_px": decode_metadata["width_px"],
            "height_px": decode_metadata["height_px"],
            "nominal_fps": decode_metadata["nominal_fps"],
            "timestamp_source": decode_metadata["timestamp_source"],
        },
        "segments": {
            "preroll_frames": sum(1 for record in records if record.segment == "preroll"),
            "demo_frames": sum(1 for record in records if record.segment == "demo"),
            "postroll_frames": sum(1 for record in records if record.segment == "postroll"),
        },
        "colmap": {
            "component_count": pipeline.component_count,
            "base_registered_keyframes": len(registered_preroll_keyframes),
            "base_total_keyframes": len(preroll_keyframes),
            "base_registered_fraction": registered_fraction,
            "localized_non_preroll_frames": len(localized_non_preroll),
            "mean_reprojection_error_px": reprojection_stats.mean_error_px,
            "max_reprojection_error_px": reprojection_stats.max_error_px,
            "observation_count": reprojection_stats.observation_count,
        },
        "normalization": {
            "status": "ok",
            "fiducial_family": bundle.capture.fiducial.family,
            "detections_total": len(detections),
            "detections_used": similarity.inlier_count,
            "scale_m_per_colmap_unit": similarity.scale,
            "orientation_spread_deg": similarity.orientation_spread_deg,
        },
        "pose_coverage": {
            "direct_registered_frames": len(direct_records),
            "interpolated_frames": len(interpolated_frames),
            "unlocalized_frames": len(unlocalized_frames),
            "pose_entries": len(pose_map),
            "largest_interpolation_gap_s": largest_interpolation_gap_s,
        },
        "capture_contract_mismatches": mismatches,
    }


def enforce_stability_thresholds(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    acceptance = bundle.project.acceptance
    colmap_summary = summary["colmap"]
    if colmap_summary["base_registered_keyframes"] < acceptance.beta_min_registered_keyframes:
        raise VideoIngestError(
            f"registered pre-roll keyframes {colmap_summary['base_registered_keyframes']} fell below beta_min_registered_keyframes={acceptance.beta_min_registered_keyframes}"
        )
    if colmap_summary["base_registered_fraction"] < acceptance.beta_min_registered_fraction:
        raise VideoIngestError(
            f"registered pre-roll fraction {colmap_summary['base_registered_fraction']:.3f} fell below beta_min_registered_fraction={acceptance.beta_min_registered_fraction:.3f}"
        )
    if colmap_summary["mean_reprojection_error_px"] > acceptance.beta_max_mean_reprojection_error_px:
        raise VideoIngestError(
            f"mean reprojection error {colmap_summary['mean_reprojection_error_px']:.3f}px exceeded beta_max_mean_reprojection_error_px={acceptance.beta_max_mean_reprojection_error_px:.3f}px"
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
    calibration_dir = out_dir / "calibration"
    colmap_root = out_dir / "colmap"
    camera_dir = out_dir / "camera"
    scene_dir = out_dir / "scene"
    preview_path = out_dir / "reprojection_preview.jpg"
    summary_path = out_dir / "reconstruction_summary.json"

    for path in (calibration_dir, colmap_root, camera_dir, scene_dir):
        path.mkdir(parents=True, exist_ok=True)

    colmap_binary = ensure_beta_dependencies(colmap_bin)
    records, decode_metadata = decode_video_to_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        image_format=bundle.capture.beta.frame_extraction.image_format,
    )
    assign_frame_segments(
        records=records,
        preroll_seconds=bundle.project.capture.preroll_seconds,
        postroll_seconds=bundle.project.capture.postroll_seconds,
    )
    keyframe_indices = select_keyframe_indices(
        records=records,
        sample_fps=bundle.capture.beta.frame_extraction.keyframe_sample_fps,
        segment="preroll",
    )
    for index in keyframe_indices:
        records[index].is_keyframe = True

    pipeline = run_colmap_pipeline(
        colmap_bin=colmap_binary,
        frames_dir=frames_dir,
        records=records,
        colmap_root=colmap_root,
        capture=bundle.capture,
    )
    record_map = {record.frame_name: record for record in records}
    detections = detect_fiducials(capture=bundle.capture, model=pipeline.final_model, record_map=record_map)
    normalized_pose_map, similarity = normalize_registered_poses(
        model=pipeline.final_model,
        detections=detections,
        fiducial_size_m=bundle.capture.fiducial.size_m,
    )
    pose_map = build_pose_timeline(
        records=records,
        direct_pose_map=normalized_pose_map,
        max_interpolation_gap_s=bundle.capture.beta.colmap.max_interpolation_gap_s,
    )
    reprojection_stats = compute_reprojection_statistics(model=pipeline.final_model)
    make_reprojection_preview(output_path=preview_path, model=pipeline.final_model, record_map=record_map)

    camera = next(iter(pipeline.final_model.cameras.values()))
    frame_index_path = frames_dir / "index.csv"
    write_frame_index_csv(frame_index_path, records)
    write_legacy_timestamps_csv(out_dir / "timestamps.csv", records)
    write_intrinsics_opencv_yaml(calibration_dir / "intrinsics_opencv.yaml", camera)
    write_camera_intrinsics_json(camera_dir / "intrinsics.json", bundle, camera)
    write_camera_intrinsics_json(out_dir / "camera_intrinsics.json", bundle, camera)
    write_camera_poses_json(camera_dir / "camera_poses.json", bundle, records, pose_map)
    write_camera_poses_json(out_dir / "camera_poses.json", bundle, records, pose_map)
    write_sparse_points_ply(scene_dir / "sparse_points.ply", pipeline.final_model)

    summary = build_summary(
        bundle=bundle,
        video_path=video_path,
        decode_metadata=decode_metadata,
        records=records,
        pipeline=pipeline,
        detections=detections,
        similarity=similarity,
        reprojection_stats=reprojection_stats,
        pose_map=pose_map,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    enforce_stability_thresholds(bundle=bundle, summary=summary)

    if not keep_workdir and pipeline.staging_dir.exists():
        shutil.rmtree(pipeline.staging_dir, ignore_errors=True)

    return IngestResult(frame_records=list(records), summary=summary)
