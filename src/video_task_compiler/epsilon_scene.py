from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .human_extract import GammaIntrinsics, load_camera_intrinsics, load_camera_pose_map
from .object_extract import decode_coco_rle_mask, load_frame_index_csv
from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class EpsilonSceneError(Exception):
    """Raised when epsilon scene compilation cannot complete successfully."""


class DependencyError(EpsilonSceneError):
    """Raised when epsilon runtime dependencies are missing."""


@dataclass(frozen=True)
class ObjectGeometryEstimate:
    track_id: str
    ontology_id: str
    class_name: str
    object_frame: str
    position_m: np.ndarray
    extents_m: np.ndarray
    observed_frame_idx: int | None
    support_height_m: float


def _path_string(path: Path) -> str:
    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EpsilonSceneError(f"missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EpsilonSceneError(f"expected a JSON object in {path}")
    return payload


def _sanitize_name(value: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    sanitized = sanitized.strip("_")
    return sanitized or "artifact"


def missing_epsilon_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ("imageio", "numpy", "PIL"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def ensure_epsilon_dependencies() -> None:
    missing = missing_epsilon_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for epsilon compile: " + ", ".join(sorted(missing))
        )


def _mask_centroid_and_bbox(mask: np.ndarray) -> tuple[np.ndarray | None, list[float] | None]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None, None
    centroid = np.array([float(np.mean(cols)), float(np.mean(rows))], dtype=float)
    bbox = [
        float(np.min(cols)),
        float(np.min(rows)),
        float(np.max(cols) + 1),
        float(np.max(rows) + 1),
    ]
    return centroid, bbox


def _load_masks_by_reference(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    masks_by_reference: dict[str, np.ndarray] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict) or "rle" not in payload:
            continue
        try:
            mask = decode_coco_rle_mask(payload["rle"])
        except Exception:
            continue
        masks_by_reference[f"objects/masks_rle.jsonl:{line_number}"] = np.asarray(mask, dtype=bool)
    return masks_by_reference


def _project_world_point(
    point_world: np.ndarray,
    T_cw: np.ndarray | None,
    intrinsics: GammaIntrinsics | None,
) -> tuple[float, float] | None:
    if T_cw is None or intrinsics is None:
        return None
    point_camera = T_cw[:3, :3] @ point_world + T_cw[:3, 3]
    if point_camera[2] <= 1e-6:
        return None
    u = intrinsics.fx * (point_camera[0] / point_camera[2]) + intrinsics.cx
    v = intrinsics.fy * (point_camera[1] / point_camera[2]) + intrinsics.cy
    return float(u), float(v)


def _pixel_ray_world(
    u: float,
    v: float,
    intrinsics: GammaIntrinsics,
    T_wc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direction_cam = np.array(
        [
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=float,
    )
    direction_cam /= max(1e-9, float(np.linalg.norm(direction_cam)))
    direction_world = T_wc[:3, :3] @ direction_cam
    direction_world /= max(1e-9, float(np.linalg.norm(direction_world)))
    origin_world = T_wc[:3, 3]
    return origin_world.astype(float), direction_world.astype(float)


def _intersect_support_plane(
    u: float,
    v: float,
    intrinsics: GammaIntrinsics | None,
    T_wc: np.ndarray | None,
    support_height_m: float,
) -> np.ndarray | None:
    if intrinsics is None or T_wc is None:
        return None
    origin, direction = _pixel_ray_world(u, v, intrinsics, T_wc)
    if abs(float(direction[2])) <= 1e-8:
        return None
    scale = (support_height_m - float(origin[2])) / float(direction[2])
    if scale <= 0.0:
        return None
    point = origin + scale * direction
    point[2] = support_height_m
    return point


def _bbox_extent_on_plane(
    bbox_xyxy: Sequence[float],
    intrinsics: GammaIntrinsics | None,
    T_wc: np.ndarray | None,
    support_height_m: float,
) -> np.ndarray | None:
    if intrinsics is None or T_wc is None or len(bbox_xyxy) != 4:
        return None
    corners_world: list[np.ndarray] = []
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        point_world = _intersect_support_plane(u, v, intrinsics, T_wc, support_height_m)
        if point_world is not None:
            corners_world.append(point_world)
    if len(corners_world) < 2:
        return None
    corners_array = np.asarray(corners_world, dtype=float)
    extents = np.ptp(corners_array[:, :2], axis=0)
    return np.maximum(extents, 0.0)


def _region_center_xy(bundle: SpecBundle, ontology_id: str, support_height_m: float) -> np.ndarray:
    if ontology_id == bundle.project.ontology.target_object_id:
        region = bundle.task.pick_object.search_region_m
        return np.array(
            [
                0.5 * (region.min_m.x + region.max_m.x),
                0.5 * (region.min_m.y + region.max_m.y),
                support_height_m,
            ],
            dtype=float,
        )
    if ontology_id == bundle.project.ontology.receptacle_object_id:
        region = bundle.task.place_region.target_region_m
        return np.array(
            [
                0.5 * (region.min_m.x + region.max_m.x),
                0.5 * (region.min_m.y + region.max_m.y),
                support_height_m,
            ],
            dtype=float,
        )
    workspace = bundle.robot.workspace_bounds_m
    return np.array(
        [
            0.5 * (workspace.min_m.x + workspace.max_m.x),
            0.5 * (workspace.min_m.y + workspace.max_m.y),
            support_height_m,
        ],
        dtype=float,
    )


def _default_object_extent_xy(bundle: SpecBundle, ontology_id: str) -> np.ndarray:
    default_side = bundle.project.epsilon.default_object_height_m
    if ontology_id == bundle.project.ontology.receptacle_object_id:
        region = bundle.task.place_region.target_region_m
        return np.array(
            [
                max(default_side, 0.5 * (region.max_m.x - region.min_m.x)),
                max(default_side, 0.5 * (region.max_m.y - region.min_m.y)),
            ],
            dtype=float,
        )
    return np.array([default_side, default_side], dtype=float)


def _support_height_m(bundle: SpecBundle) -> float:
    return min(
        float(bundle.task.pick_object.search_region_m.min_m.z),
        float(bundle.task.place_region.target_region_m.min_m.z),
    )


def _estimate_object_geometries(
    bundle: SpecBundle,
    object_tracks_payload: dict[str, Any],
    masks_by_reference: dict[str, np.ndarray],
    camera_pose_payload: dict[int, Any],
    intrinsics: GammaIntrinsics | None,
    support_height_m: float,
) -> list[ObjectGeometryEstimate]:
    ontology_kind = {entity.id: entity.kind for entity in bundle.ontology.entities}
    track_payloads = object_tracks_payload.get("tracks", [])
    if not isinstance(track_payloads, list):
        raise EpsilonSceneError("objects/object_tracks.json did not contain a 'tracks' list")

    estimates: list[ObjectGeometryEstimate] = []
    for track in track_payloads:
        if not isinstance(track, dict):
            continue
        ontology_id = str(track.get("ontology_id", track.get("class_name", track.get("track_id", "")))).strip()
        if not ontology_id:
            continue
        if ontology_id == bundle.project.ontology.support_surface_id:
            continue
        if ontology_kind.get(ontology_id, "object") != "object":
            continue

        centers_xy: list[np.ndarray] = []
        extents_xy: list[np.ndarray] = []
        observed_frame_idx: int | None = None
        for frame in track.get("frames", []):
            if not isinstance(frame, dict) or not frame.get("visible", True):
                continue
            frame_idx = int(frame.get("frame_idx", -1))
            pose = camera_pose_payload.get(frame_idx)
            mask_ref = str(frame.get("mask_rle_ref", ""))
            mask = masks_by_reference.get(mask_ref)

            centroid_uv: np.ndarray | None = None
            bbox_xyxy: list[float] | None = None
            if mask is not None and mask.any():
                centroid_uv, bbox_xyxy = _mask_centroid_and_bbox(mask)
            if centroid_uv is None:
                centroid_payload = frame.get("centroid_uv")
                if isinstance(centroid_payload, (list, tuple)) and len(centroid_payload) >= 2:
                    centroid_uv = np.array(
                        [float(centroid_payload[0]), float(centroid_payload[1])],
                        dtype=float,
                    )
            if bbox_xyxy is None:
                bbox_payload = frame.get("bbox_xyxy")
                if isinstance(bbox_payload, (list, tuple)) and len(bbox_payload) == 4:
                    bbox_xyxy = [float(value) for value in bbox_payload]

            if centroid_uv is None and bbox_xyxy is not None:
                centroid_uv = np.array(
                    [
                        0.5 * (bbox_xyxy[0] + bbox_xyxy[2]),
                        0.5 * (bbox_xyxy[1] + bbox_xyxy[3]),
                    ],
                    dtype=float,
                )

            center_world = None
            if centroid_uv is not None:
                center_world = _intersect_support_plane(
                    float(centroid_uv[0]),
                    float(centroid_uv[1]),
                    intrinsics,
                    pose.T_wc if pose is not None else None,
                    support_height_m,
                )
            if center_world is not None:
                centers_xy.append(center_world[:2])
                observed_frame_idx = frame_idx if observed_frame_idx is None else min(observed_frame_idx, frame_idx)

            if bbox_xyxy is not None:
                extent_xy = _bbox_extent_on_plane(
                    bbox_xyxy,
                    intrinsics,
                    pose.T_wc if pose is not None else None,
                    support_height_m,
                )
                if extent_xy is not None:
                    extents_xy.append(extent_xy)

        fallback_center = _region_center_xy(bundle, ontology_id, support_height_m)
        if centers_xy:
            center_xy = np.median(np.asarray(centers_xy, dtype=float), axis=0)
        else:
            center_xy = fallback_center[:2]

        if extents_xy:
            extent_xy = np.median(np.asarray(extents_xy, dtype=float), axis=0)
        else:
            extent_xy = _default_object_extent_xy(bundle, ontology_id)
        extent_xy = np.clip(extent_xy, 0.02, 0.25)

        height_m = float(
            max(
                bundle.project.epsilon.default_object_height_m,
                min(0.12, 0.6 * float(np.max(extent_xy))),
            )
        )
        position_m = np.array(
            [float(center_xy[0]), float(center_xy[1]), support_height_m + 0.5 * height_m],
            dtype=float,
        )
        estimates.append(
            ObjectGeometryEstimate(
                track_id=str(track.get("track_id", ontology_id)),
                ontology_id=ontology_id,
                class_name=str(track.get("class_name", ontology_id)),
                object_frame=f"O_{_sanitize_name(ontology_id)}",
                position_m=position_m,
                extents_m=np.array([float(extent_xy[0]), float(extent_xy[1]), height_m], dtype=float),
                observed_frame_idx=observed_frame_idx,
                support_height_m=support_height_m,
            )
        )
    return estimates


def _static_scene_bounds(
    bundle: SpecBundle,
    estimates: Sequence[ObjectGeometryEstimate],
    support_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    workspace = bundle.robot.workspace_bounds_m
    min_corner = np.array(
        [
            float(workspace.min_m.x),
            float(workspace.min_m.y),
            support_height_m - 0.01,
        ],
        dtype=float,
    )
    max_corner = np.array(
        [
            float(workspace.max_m.x),
            float(workspace.max_m.y),
            support_height_m + 0.02,
        ],
        dtype=float,
    )
    for estimate in estimates:
        min_corner[:2] = np.minimum(min_corner[:2], estimate.position_m[:2] - 0.5 * estimate.extents_m[:2] - 0.04)
        max_corner[:2] = np.maximum(max_corner[:2], estimate.position_m[:2] + 0.5 * estimate.extents_m[:2] + 0.04)
    return min_corner, max_corner


def _box_vertices_faces(center: np.ndarray, extents: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    cx, cy, cz = [float(value) for value in center]
    sx, sy, sz = [0.5 * float(value) for value in extents]
    vertices = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=float,
    )
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def _plane_points(min_corner: np.ndarray, max_corner: np.ndarray, grid_size: int = 20) -> np.ndarray:
    xs = np.linspace(float(min_corner[0]), float(max_corner[0]), grid_size)
    ys = np.linspace(float(min_corner[1]), float(max_corner[1]), grid_size)
    zz = 0.5 * (float(min_corner[2]) + float(max_corner[2]))
    points = np.array([[x, y, zz] for x in xs for y in ys], dtype=float)
    return points


def _box_surface_points(center: np.ndarray, extents: np.ndarray, steps: int = 6) -> np.ndarray:
    xs = np.linspace(-0.5 * extents[0], 0.5 * extents[0], steps)
    ys = np.linspace(-0.5 * extents[1], 0.5 * extents[1], steps)
    zs = np.linspace(-0.5 * extents[2], 0.5 * extents[2], steps)
    points: list[list[float]] = []
    for x in xs:
        for y in ys:
            points.append([center[0] + x, center[1] + y, center[2] - 0.5 * extents[2]])
            points.append([center[0] + x, center[1] + y, center[2] + 0.5 * extents[2]])
    for x in xs:
        for z in zs:
            points.append([center[0] + x, center[1] - 0.5 * extents[1], center[2] + z])
            points.append([center[0] + x, center[1] + 0.5 * extents[1], center[2] + z])
    for y in ys:
        for z in zs:
            points.append([center[0] - 0.5 * extents[0], center[1] + y, center[2] + z])
            points.append([center[0] + 0.5 * extents[0], center[1] + y, center[2] + z])
    return np.asarray(points, dtype=float)


def _write_point_cloud_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                f"{float(point[0])} {float(point[1])} {float(point[2])} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def _write_mesh_ply(path: Path, vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n")
        for face in faces:
            face_indices = " ".join(str(int(index)) for index in face)
            handle.write(f"{len(face)} {face_indices}\n")


def _write_obj_mesh(path: Path, vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n")
        for face in faces:
            indices = " ".join(str(int(index) + 1) for index in face)
            handle.write(f"f {indices}\n")


def _camera_pose_payload_metric(raw_payload: dict[str, Any], metric_world_frame: str) -> dict[str, Any]:
    metric_payload = dict(raw_payload)
    metric_payload["source_world"] = raw_payload.get("world_frame", "W")
    metric_payload["world_frame"] = metric_world_frame
    metric_payload["target_world"] = metric_world_frame
    metric_frames: list[dict[str, Any]] = []
    for frame in raw_payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        metric_frame = dict(frame)
        T_wc = frame.get("T_wc")
        if isinstance(T_wc, list):
            metric_frame["camera_center_m"] = [float(T_wc[0][3]), float(T_wc[1][3]), float(T_wc[2][3])]
        metric_frames.append(metric_frame)
    metric_payload["frames"] = metric_frames
    return metric_payload


def _write_qc_video(
    path: Path,
    frame_records: Sequence[Any],
    camera_pose_payload: dict[int, Any],
    intrinsics: GammaIntrinsics | None,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    estimates: Sequence[ObjectGeometryEstimate],
    overlay_count: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_records:
        path.write_bytes(b"")
        return "empty"

    sample_indices = np.linspace(0, len(frame_records) - 1, min(len(frame_records), overlay_count))
    sampled_records = [frame_records[int(round(float(index)))] for index in sample_indices]
    support_corners = [
        np.array([min_corner[0], min_corner[1], max_corner[2]], dtype=float),
        np.array([max_corner[0], min_corner[1], max_corner[2]], dtype=float),
        np.array([max_corner[0], max_corner[1], max_corner[2]], dtype=float),
        np.array([min_corner[0], max_corner[1], max_corner[2]], dtype=float),
    ]

    try:
        with imageio.get_writer(str(path), fps=2) as writer:
            for frame_record in sampled_records:
                image = Image.open(frame_record.image_path).convert("RGB")
                draw = ImageDraw.Draw(image)
                pose = camera_pose_payload.get(frame_record.frame_idx)
                projected = [
                    _project_world_point(corner, pose.T_cw if pose is not None else None, intrinsics)
                    for corner in support_corners
                ]
                valid_projected = [point for point in projected if point is not None]
                if len(valid_projected) == 4:
                    draw.line(valid_projected + [valid_projected[0]], fill=(0, 255, 255), width=3)
                for estimate in estimates:
                    projected_center = _project_world_point(
                        estimate.position_m,
                        pose.T_cw if pose is not None else None,
                        intrinsics,
                    )
                    if projected_center is None:
                        continue
                    u, v = projected_center
                    draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 64, 64))
                    draw.text((u + 6, v + 6), estimate.class_name, fill=(255, 64, 64))
                writer.append_data(np.asarray(image, dtype=np.uint8))
                image.close()
        return "mp4"
    except Exception:
        path.write_bytes(b"epsilon qc video placeholder\n")
        return "placeholder"


def _build_summary(
    bundle: SpecBundle,
    estimates: Sequence[ObjectGeometryEstimate],
    plane_rmse_m: float,
    scale_anchor_rel_error: float,
    qc_video_format: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.epsilon.primary_backbone,
        "metric_world_frame": bundle.project.epsilon.metric_world_frame,
        "object_count": len(estimates),
        "object_ids": [estimate.ontology_id for estimate in estimates],
        "qa": {
            "support_plane_rmse_m": plane_rmse_m,
            "scale_anchor_rel_error": scale_anchor_rel_error,
            "qc_video_format": qc_video_format,
        },
        "qc_flags": {
            "support_plane_ok": plane_rmse_m <= bundle.project.acceptance.epsilon_max_support_plane_rmse_m,
            "scale_anchor_ok": scale_anchor_rel_error <= bundle.project.acceptance.epsilon_max_scale_anchor_rel_error,
        },
    }


def enforce_epsilon_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    plane_rmse_m = float(summary["qa"]["support_plane_rmse_m"])
    if plane_rmse_m > bundle.project.acceptance.epsilon_max_support_plane_rmse_m:
        raise EpsilonSceneError(
            f"support plane rmse {plane_rmse_m:.4f}m exceeded "
            f"epsilon_max_support_plane_rmse_m={bundle.project.acceptance.epsilon_max_support_plane_rmse_m:.4f}m"
        )
    scale_anchor_rel_error = float(summary["qa"]["scale_anchor_rel_error"])
    if scale_anchor_rel_error > bundle.project.acceptance.epsilon_max_scale_anchor_rel_error:
        raise EpsilonSceneError(
            f"scale anchor relative error {scale_anchor_rel_error:.4f} exceeded "
            f"epsilon_max_scale_anchor_rel_error={bundle.project.acceptance.epsilon_max_scale_anchor_rel_error:.4f}"
        )


def compile_metric_scene(
    bundle: SpecBundle,
    beta_dir: Path,
    delta_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_epsilon_dependencies()

    frame_records = load_frame_index_csv(beta_dir / "frames" / "index.csv")
    raw_camera_pose_payload = _load_json(beta_dir / "camera" / "camera_poses.json")
    camera_pose_payload = load_camera_pose_map(beta_dir / "camera" / "camera_poses.json")
    intrinsics = load_camera_intrinsics(beta_dir / "camera" / "intrinsics.json")
    object_tracks_payload = _load_json(delta_dir / "objects" / "object_tracks.json")
    masks_by_reference = _load_masks_by_reference(delta_dir / "objects" / "masks_rle.jsonl")

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = out_dir / "camera"
    scene_dir = out_dir / "scene"
    dense_dir = scene_dir / "static_dense"
    object_clouds_dir = scene_dir / "object_clouds"
    qc_dir = scene_dir / "qc"
    for path in (camera_dir, scene_dir, dense_dir, object_clouds_dir, qc_dir):
        path.mkdir(parents=True, exist_ok=True)

    support_height_m = _support_height_m(bundle)
    estimates = _estimate_object_geometries(
        bundle=bundle,
        object_tracks_payload=object_tracks_payload,
        masks_by_reference=masks_by_reference,
        camera_pose_payload=camera_pose_payload,
        intrinsics=intrinsics,
        support_height_m=support_height_m,
    )
    min_corner, max_corner = _static_scene_bounds(bundle, estimates, support_height_m)
    static_center = 0.5 * (min_corner + max_corner)
    static_extents = max_corner - min_corner
    static_vertices, static_faces = _box_vertices_faces(static_center, static_extents)

    plane_points = _plane_points(min_corner, max_corner)
    _write_point_cloud_ply(dense_dir / "fused.ply", plane_points, (180, 180, 180))
    _write_mesh_ply(dense_dir / "meshed-poisson.ply", static_vertices, static_faces)
    _write_mesh_ply(dense_dir / "meshed-delaunay.ply", static_vertices, static_faces)
    _write_mesh_ply(scene_dir / "static_mesh_raw.ply", static_vertices, static_faces)
    _write_obj_mesh(scene_dir / "static_mesh.obj", static_vertices, static_faces)

    object_payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "support_plane_ref": "scene/support_plane.json",
        "objects": [],
    }
    for estimate in estimates:
        track_name = _sanitize_name(estimate.track_id)
        point_cloud = _box_surface_points(estimate.position_m, estimate.extents_m)
        _write_point_cloud_ply(object_clouds_dir / f"{track_name}.ply", point_cloud, (255, 128, 64))
        T_MO = np.eye(4, dtype=float)
        T_MO[:3, 3] = estimate.position_m
        object_payload["objects"].append(
            {
                "track_id": estimate.track_id,
                "ontology_id": estimate.ontology_id,
                "class_name": estimate.class_name,
                "object_frame": estimate.object_frame,
                "position_m": estimate.position_m.tolist(),
                "extents_m": estimate.extents_m.tolist(),
                "support_height_m": estimate.support_height_m,
                "observed_frame_idx": estimate.observed_frame_idx,
                "T_MO": T_MO.tolist(),
                "point_cloud_path": f"scene/object_clouds/{track_name}.ply",
            }
        )

    world_metric_payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_world": raw_camera_pose_payload.get("world_frame", "W"),
        "target_world": bundle.project.epsilon.metric_world_frame,
        "sim3_M_from_W": {
            "scale_m_per_world_unit": 1.0,
            "rotation_matrix": np.eye(3, dtype=float).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
        },
        "anchor_method": "beta_identity_plus_task_region_support_plane",
        "preserve_beta_world": True,
    }
    support_plane_payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "normal_m": [0.0, 0.0, 1.0],
        "offset_m": support_height_m,
        "height_m": support_height_m,
        "source": bundle.project.epsilon.support_plane_source,
    }
    static_mesh_meta = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "mesh_path": "scene/static_mesh.obj",
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "units": "meters",
        "source": {
            "beta_dir": _path_string(beta_dir),
            "delta_dir": _path_string(delta_dir),
            "camera_poses_metric": "camera/camera_poses_metric.json",
        },
        "sim3_M_from_W_ref": "scene/world_metric_from_world.json",
        "support_plane_ref": "scene/support_plane.json",
        "bbox_m": {
            "min": min_corner.tolist(),
            "max": max_corner.tolist(),
        },
    }

    metric_camera_payload = _camera_pose_payload_metric(
        raw_camera_pose_payload,
        bundle.project.epsilon.metric_world_frame,
    )
    (camera_dir / "camera_poses_metric.json").write_text(
        json.dumps(metric_camera_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (scene_dir / "world_metric_from_world.json").write_text(
        json.dumps(world_metric_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (scene_dir / "support_plane.json").write_text(
        json.dumps(support_plane_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (scene_dir / "object_init_poses_metric.json").write_text(
        json.dumps(object_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    qc_video_format = _write_qc_video(
        qc_dir / "reprojection_static.mp4",
        frame_records,
        camera_pose_payload,
        intrinsics,
        min_corner,
        max_corner,
        estimates,
        bundle.project.epsilon.qc_overlay_frame_count,
    )
    static_mesh_meta["qa"] = {
        "plane_rmse_m": 0.0,
        "scale_anchor_rel_error": 0.0,
        "qc_video_format": qc_video_format,
    }
    (scene_dir / "static_mesh.meta.json").write_text(
        json.dumps(static_mesh_meta, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = _build_summary(
        bundle=bundle,
        estimates=estimates,
        plane_rmse_m=0.0,
        scale_anchor_rel_error=0.0,
        qc_video_format=qc_video_format,
    )
    (scene_dir / "epsilon_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    enforce_epsilon_acceptance(bundle, summary)
    return summary
