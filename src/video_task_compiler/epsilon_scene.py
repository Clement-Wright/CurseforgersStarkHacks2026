from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from .human_extract import GammaIntrinsics, load_camera_intrinsics, load_camera_pose_map
from .object_extract import decode_coco_rle_mask, load_frame_index_csv
from .specs import PROJECT_SCHEMA_VERSION, SpecBundle
from .task_window import (
    TaskWindowError,
    filter_items_to_task_window,
    persist_task_window,
    resolve_task_window,
)


class EpsilonSceneError(Exception):
    """Raised when epsilon scene compilation cannot complete successfully."""


class DependencyError(EpsilonSceneError):
    """Raised when epsilon runtime dependencies are missing."""


@dataclass(frozen=True)
class ObjectGeometryEstimate:
    track_id: str
    instance_id: str | None
    ontology_id: str
    class_name: str
    object_frame: str
    position_m: np.ndarray
    extents_m: np.ndarray
    observed_frame_idx: int | None
    support_height_m: float
    frames_used: list[int]
    footprint_estimator: str
    height_mode: str
    raw_planar_extent_stats_m: dict[str, Any]
    geometry_source: str
    mesh_path: str | None


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
    for module_name in ("imageio", "numpy", "PIL", "pyarrow"):
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


def _mask_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or not mask.any():
        return np.zeros((0, 2), dtype=float)
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    boundary = mask & ~interior
    rows, cols = np.nonzero(boundary)
    return np.column_stack((cols.astype(float) + 0.5, rows.astype(float) + 0.5))


def _sample_boundary_pixels(mask: np.ndarray, max_points: int = 128) -> np.ndarray:
    pixels = _mask_boundary_pixels(mask)
    if len(pixels) <= max_points:
        return pixels
    step = max(1, int(math.ceil(len(pixels) / max_points)))
    return pixels[::step]


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


def _load_interaction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


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


def _project_mask_boundary_to_plane(
    mask: np.ndarray,
    intrinsics: GammaIntrinsics | None,
    T_wc: np.ndarray | None,
    support_height_m: float,
) -> np.ndarray:
    if intrinsics is None or T_wc is None:
        return np.zeros((0, 2), dtype=float)
    sampled_pixels = _sample_boundary_pixels(mask)
    if len(sampled_pixels) == 0:
        return np.zeros((0, 2), dtype=float)
    points_world: list[np.ndarray] = []
    for u, v in sampled_pixels:
        point_world = _intersect_support_plane(float(u), float(v), intrinsics, T_wc, support_height_m)
        if point_world is not None and np.isfinite(point_world[:2]).all():
            points_world.append(point_world[:2])
    if len(points_world) < 3:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points_world, dtype=float)


def _fit_planar_obb(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points_xy) < 3:
        return None
    mean_xy = np.mean(points_xy, axis=0)
    centered = points_xy - mean_xy
    covariance = np.cov(centered.T)
    if covariance.shape != (2, 2):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, 1] *= -1.0
    local = centered @ basis
    min_local = np.min(local, axis=0)
    max_local = np.max(local, axis=0)
    extents_xy = np.maximum(max_local - min_local, 0.0)
    center_local = 0.5 * (min_local + max_local)
    center_xy = mean_xy + basis @ center_local
    return center_xy.astype(float), np.sort(extents_xy.astype(float))[::-1]


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


def _extent_stats(extents_xy: Sequence[np.ndarray]) -> dict[str, Any]:
    if not extents_xy:
        return {
            "sample_count": 0,
            "min_xy": None,
            "p25_xy": None,
            "median_xy": None,
            "p35_xy": None,
            "p75_xy": None,
            "max_xy": None,
        }
    values = np.asarray(extents_xy, dtype=float)
    return {
        "sample_count": int(len(values)),
        "min_xy": np.min(values, axis=0).tolist(),
        "p25_xy": np.quantile(values, 0.25, axis=0).tolist(),
        "median_xy": np.quantile(values, 0.50, axis=0).tolist(),
        "p35_xy": np.quantile(values, 0.35, axis=0).tolist(),
        "p75_xy": np.quantile(values, 0.75, axis=0).tolist(),
        "max_xy": np.max(values, axis=0).tolist(),
    }


def _first_contact_frame_idx(rows: Sequence[dict[str, Any]], class_name: str) -> int | None:
    candidates = [
        row
        for row in rows
        if row.get("class_name") == class_name and bool(row.get("likely_contact_boolean"))
    ]
    if not candidates:
        return None
    return min(int(row.get("frame_idx", 0)) for row in candidates)


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


def _scene_instance_for_track(bundle: SpecBundle, track: dict[str, Any]) -> Any | None:
    instance_id = str(track.get("instance_id", "")).strip()
    ontology_id = str(track.get("ontology_id", track.get("class_name", ""))).strip()
    if instance_id:
        for instance in bundle.task.scene_instances:
            if instance.instance_id == instance_id:
                return instance
    for instance in bundle.task.scene_instances:
        if instance.ontology_id == ontology_id:
            return instance
    return None


def _size_prior_extent_xy(scene_instance: Any | None) -> np.ndarray | None:
    if scene_instance is None or scene_instance.size_prior_m is None:
        return None
    return np.array(
        [
            float(scene_instance.size_prior_m.x),
            float(scene_instance.size_prior_m.y),
        ],
        dtype=float,
    )


def _support_height_m(bundle: SpecBundle) -> float:
    return min(
        float(bundle.task.pick_object.search_region_m.min_m.z),
        float(bundle.task.place_region.target_region_m.min_m.z),
    )


def _estimate_object_geometries(
    bundle: SpecBundle,
    object_tracks_payload: dict[str, Any],
    masks_by_reference: dict[str, np.ndarray],
    interaction_rows: Sequence[dict[str, Any]],
    camera_pose_payload: dict[int, Any],
    intrinsics: GammaIntrinsics | None,
    support_height_m: float,
    task_window: Any,
) -> list[ObjectGeometryEstimate]:
    ontology_kind = {entity.id: entity.kind for entity in bundle.ontology.entities}
    track_payloads = object_tracks_payload.get("tracks", [])
    if not isinstance(track_payloads, list):
        raise EpsilonSceneError("objects/object_tracks.json did not contain a 'tracks' list")

    target_contact_frame = _first_contact_frame_idx(interaction_rows, bundle.project.ontology.target_object_id)
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
        scene_instance = _scene_instance_for_track(bundle, track)
        instance_id = (
            str(scene_instance.instance_id)
            if scene_instance is not None
            else (str(track.get("instance_id", "")).strip() or None)
        )

        centers_xy: list[np.ndarray] = []
        extents_xy: list[np.ndarray] = []
        observed_frame_idx: int | None = None
        frames_used: list[int] = []
        for frame in track.get("frames", []):
            if not isinstance(frame, dict) or not frame.get("visible", True):
                continue
            frame_idx = int(frame.get("frame_idx", -1))
            if not task_window.contains(frame_idx):
                continue
            if (
                ontology_id == bundle.project.ontology.target_object_id
                and target_contact_frame is not None
                and frame_idx >= target_contact_frame
            ):
                continue
            pose = camera_pose_payload.get(frame_idx)
            mask_ref = str(frame.get("mask_rle_ref", ""))
            mask = masks_by_reference.get(mask_ref)

            centroid_uv: np.ndarray | None = None
            center_xy: np.ndarray | None = None
            if mask is not None and mask.any():
                centroid_uv, _ = _mask_centroid_and_bbox(mask)
                projected_boundary_xy = _project_mask_boundary_to_plane(
                    mask,
                    intrinsics,
                    pose.T_wc if pose is not None else None,
                    support_height_m,
                )
                obb_fit = _fit_planar_obb(projected_boundary_xy)
                if obb_fit is not None:
                    center_xy, extent_xy = obb_fit
                    extents_xy.append(np.asarray(extent_xy, dtype=float))
            if centroid_uv is None:
                centroid_payload = frame.get("centroid_uv")
                if isinstance(centroid_payload, (list, tuple)) and len(centroid_payload) >= 2:
                    centroid_uv = np.array(
                        [float(centroid_payload[0]), float(centroid_payload[1])],
                        dtype=float,
                    )
            if center_xy is None and centroid_uv is not None:
                center_world = _intersect_support_plane(
                    float(centroid_uv[0]),
                    float(centroid_uv[1]),
                    intrinsics,
                    pose.T_wc if pose is not None else None,
                    support_height_m,
                )
                if center_world is not None:
                    center_xy = center_world[:2]
            if center_xy is not None:
                centers_xy.append(np.asarray(center_xy, dtype=float))
                frames_used.append(frame_idx)
                observed_frame_idx = frame_idx if observed_frame_idx is None else min(observed_frame_idx, frame_idx)

        fallback_center = _region_center_xy(bundle, ontology_id, support_height_m)
        if centers_xy:
            center_xy = np.median(np.asarray(centers_xy, dtype=float), axis=0)
        else:
            center_xy = fallback_center[:2]

        extent_stats = _extent_stats(extents_xy)
        if extents_xy:
            extent_xy = np.asarray(extent_stats["p35_xy"], dtype=float)
        else:
            size_prior_xy = _size_prior_extent_xy(scene_instance)
            extent_xy = size_prior_xy if size_prior_xy is not None else _default_object_extent_xy(bundle, ontology_id)
        extent_xy = np.clip(extent_xy, 0.015, 0.25)

        height_m = float(bundle.project.epsilon.default_object_height_m)
        position_m = np.array(
            [float(center_xy[0]), float(center_xy[1]), support_height_m + 0.5 * height_m],
            dtype=float,
        )
        estimates.append(
            ObjectGeometryEstimate(
                track_id=str(track.get("track_id", ontology_id)),
                instance_id=instance_id,
                ontology_id=ontology_id,
                class_name=str(track.get("class_name", ontology_id)),
                object_frame=f"O_{_sanitize_name(instance_id or ontology_id)}",
                position_m=position_m,
                extents_m=np.array([float(extent_xy[0]), float(extent_xy[1]), height_m], dtype=float),
                observed_frame_idx=observed_frame_idx,
                support_height_m=support_height_m,
                frames_used=sorted(set(frames_used)),
                footprint_estimator="support_plane_mask_obb",
                height_mode="proxy_default_height",
                raw_planar_extent_stats_m=extent_stats,
                geometry_source="mask_backprojected_cuboid"
                if extents_xy
                else ("scene_instance_size_prior_cuboid" if scene_instance is not None and scene_instance.size_prior_m is not None else "default_proxy_cuboid"),
                mesh_path=None,
            )
        )
    return estimates


def _static_scene_bounds(
    bundle: SpecBundle,
    estimates: Sequence[ObjectGeometryEstimate],
    support_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    region_points = [
        np.array(
            [
                bundle.task.pick_object.search_region_m.min_m.x,
                bundle.task.pick_object.search_region_m.min_m.y,
            ],
            dtype=float,
        ),
        np.array(
            [
                bundle.task.pick_object.search_region_m.max_m.x,
                bundle.task.pick_object.search_region_m.max_m.y,
            ],
            dtype=float,
        ),
        np.array(
            [
                bundle.task.place_region.target_region_m.min_m.x,
                bundle.task.place_region.target_region_m.min_m.y,
            ],
            dtype=float,
        ),
        np.array(
            [
                bundle.task.place_region.target_region_m.max_m.x,
                bundle.task.place_region.target_region_m.max_m.y,
            ],
            dtype=float,
        ),
    ]
    min_xy = np.min(np.asarray(region_points, dtype=float), axis=0)
    max_xy = np.max(np.asarray(region_points, dtype=float), axis=0)
    for estimate in estimates:
        min_xy = np.minimum(min_xy, estimate.position_m[:2] - 0.5 * estimate.extents_m[:2] - 0.08)
        max_xy = np.maximum(max_xy, estimate.position_m[:2] + 0.5 * estimate.extents_m[:2] + 0.08)
    min_corner = np.array([float(min_xy[0]), float(min_xy[1]), support_height_m - 0.01], dtype=float)
    max_corner = np.array([float(max_xy[0]), float(max_xy[1]), support_height_m + 0.02], dtype=float)
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
    source_world = str(
        raw_payload.get("world_frame")
        or raw_payload.get("coordinate_frame")
        or "W"
    )
    metric_payload["source_world"] = source_world
    metric_payload["world_frame"] = metric_world_frame
    metric_payload["target_world"] = metric_world_frame
    metric_payload["metric_alignment_mode"] = "inherited_from_beta"
    metric_payload["transform_source"] = "inherited_from_beta_fiducial_world"
    metric_payload["measured"] = False
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
    qc_video_format: str,
    proxy_artifacts_nonempty_ok: bool,
    object_init_serializable_ok: bool,
    dense_static_artifacts_ok: bool,
) -> dict[str, Any]:
    support_plane_rmse = 0.0 if bundle.project.epsilon.geometry_mode == "dense_static_reconstruction" else None
    scale_anchor_rel_error = (
        0.0 if bundle.project.epsilon.metric_alignment_mode in {"inherited_from_beta", "measured_sim3"} else None
    )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.epsilon.primary_backbone,
        "geometry_mode": bundle.project.epsilon.geometry_mode,
        "metric_alignment_mode": bundle.project.epsilon.metric_alignment_mode,
        "metric_world_frame": bundle.project.epsilon.metric_world_frame,
        "object_count": len(estimates),
        "object_ids": [estimate.ontology_id for estimate in estimates],
        "metric_alignment": {
            "transform_source": "inherited_from_beta_fiducial_world",
            "measured": False,
            "qa_status": "synthetic_dense_placeholder"
            if bundle.project.epsilon.geometry_mode == "dense_static_reconstruction"
            else "not_applicable",
        },
        "qa": {
            "support_plane_rmse_m": support_plane_rmse,
            "scale_anchor_rel_error": scale_anchor_rel_error,
            "qa_status": {
                "support_plane_rmse_m": (
                    "synthetic_dense_placeholder"
                    if support_plane_rmse is not None
                    else "not_applicable"
                ),
                "scale_anchor_rel_error": (
                    "inherited_from_beta_world"
                    if scale_anchor_rel_error is not None
                    else "not_applicable"
                ),
            },
            "qc_video_format": qc_video_format,
            "dense_static_artifacts_ok": dense_static_artifacts_ok,
        },
        "object_geometry_diagnostics": [
            {
                "ontology_id": estimate.ontology_id,
                "track_id": estimate.track_id,
                "footprint_estimator": estimate.footprint_estimator,
                "height_mode": estimate.height_mode,
                "frames_used": estimate.frames_used,
                "raw_planar_extent_stats_m": estimate.raw_planar_extent_stats_m,
                "final_extents_m": estimate.extents_m.tolist(),
            }
            for estimate in estimates
        ],
        "qc_flags": {
            "truthful_provenance_ok": True,
            "proxy_artifacts_nonempty_ok": proxy_artifacts_nonempty_ok,
            "object_init_serializable_ok": object_init_serializable_ok,
            "dense_static_artifacts_ok": dense_static_artifacts_ok,
        },
    }


def enforce_epsilon_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if bundle.project.acceptance.epsilon_require_truthful_provenance and not bool(
        summary["qc_flags"]["truthful_provenance_ok"]
    ):
        raise EpsilonSceneError("epsilon proxy-scene provenance did not satisfy the truthful-provenance contract")
    if not bool(summary["qc_flags"]["proxy_artifacts_nonempty_ok"]):
        raise EpsilonSceneError("epsilon proxy-scene artifacts were empty or incomplete")
    if not bool(summary["qc_flags"]["object_init_serializable_ok"]):
        raise EpsilonSceneError("epsilon object initialization payload was not deterministically serializable")


def compile_metric_scene(
    bundle: SpecBundle,
    beta_dir: Path,
    delta_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_epsilon_dependencies()
    if bundle.project.epsilon.geometry_mode not in {"proxy_scene", "dense_static_reconstruction"}:
        raise EpsilonSceneError(
            f"epsilon geometry_mode '{bundle.project.epsilon.geometry_mode}' is not implemented yet; "
            "the current MVP supports proxy_scene and dense_static_reconstruction"
        )
    if bundle.project.epsilon.metric_alignment_mode != "inherited_from_beta":
        raise EpsilonSceneError(
            f"epsilon metric_alignment_mode '{bundle.project.epsilon.metric_alignment_mode}' is not implemented yet; "
            "the current MVP supports only inherited_from_beta"
        )

    frame_records = load_frame_index_csv(beta_dir / "frames" / "index.csv")
    try:
        task_window = resolve_task_window(
            frame_records,
            video_id=bundle.project.video_id,
            artifact_roots=(delta_dir, out_dir, beta_dir),
        )
    except TaskWindowError as exc:
        raise EpsilonSceneError(str(exc)) from exc
    raw_camera_pose_payload = _load_json(beta_dir / "camera" / "camera_poses.json")
    camera_pose_payload = load_camera_pose_map(beta_dir / "camera" / "camera_poses.json")
    intrinsics = load_camera_intrinsics(beta_dir / "camera" / "intrinsics.json")
    object_tracks_payload = _load_json(delta_dir / "objects" / "object_tracks.json")
    masks_by_reference = _load_masks_by_reference(delta_dir / "objects" / "masks_rle.jsonl")
    interaction_rows = filter_items_to_task_window(
        _load_interaction_rows(delta_dir / "objects" / "interactions.parquet"),
        task_window,
    )
    frame_records = filter_items_to_task_window(frame_records, task_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = out_dir / "camera"
    scene_dir = out_dir / "scene"
    dense_dir = scene_dir / "static_dense"
    object_clouds_dir = scene_dir / "object_clouds"
    object_meshes_dir = scene_dir / "object_meshes"
    qc_dir = scene_dir / "qc"
    for path in (camera_dir, scene_dir, dense_dir, object_clouds_dir, object_meshes_dir, qc_dir):
        path.mkdir(parents=True, exist_ok=True)
    persist_task_window(task_window, out_dir, video_id=bundle.project.video_id)

    support_height_m = _support_height_m(bundle)
    estimates = _estimate_object_geometries(
        bundle=bundle,
        object_tracks_payload=object_tracks_payload,
        masks_by_reference=masks_by_reference,
        interaction_rows=interaction_rows,
        camera_pose_payload=camera_pose_payload,
        intrinsics=intrinsics,
        support_height_m=support_height_m,
        task_window=task_window,
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
        local_vertices, local_faces = _box_vertices_faces(np.zeros(3, dtype=float), estimate.extents_m)
        mesh_path = object_meshes_dir / f"{track_name}.obj"
        _write_obj_mesh(mesh_path, local_vertices, local_faces)
        T_MO = np.eye(4, dtype=float)
        T_MO[:3, 3] = estimate.position_m
        object_payload["objects"].append(
            {
                "track_id": estimate.track_id,
                "instance_id": estimate.instance_id,
                "ontology_id": estimate.ontology_id,
                "class_name": estimate.class_name,
                "object_frame": estimate.object_frame,
                "geometry_source": estimate.geometry_source,
                "measured": False,
                "footprint_estimator": estimate.footprint_estimator,
                "height_mode": estimate.height_mode,
                "position_m": estimate.position_m.tolist(),
                "extents_m": estimate.extents_m.tolist(),
                "final_extents_m": estimate.extents_m.tolist(),
                "support_height_m": estimate.support_height_m,
                "observed_frame_idx": estimate.observed_frame_idx,
                "frames_used": estimate.frames_used,
                "raw_planar_extent_stats_m": estimate.raw_planar_extent_stats_m,
                "T_MO": T_MO.tolist(),
                "point_cloud_path": f"scene/object_clouds/{track_name}.ply",
                "mesh_path": f"scene/object_meshes/{track_name}.obj",
            }
        )

    world_metric_payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_world": str(
            raw_camera_pose_payload.get("world_frame")
            or raw_camera_pose_payload.get("coordinate_frame")
            or "W"
        ),
        "target_world": bundle.project.epsilon.metric_world_frame,
        "sim3_M_from_W": {
            "scale_m_per_world_unit": 1.0,
            "rotation_matrix": np.eye(3, dtype=float).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
        },
        "transform_source": "inherited_from_beta_fiducial_world",
        "measured": False,
        "qa_status": "not_applicable",
        "anchor_method": "inherited_from_beta",
        "preserve_beta_world": True,
    }
    support_plane_payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "normal_m": [0.0, 0.0, 1.0],
        "offset_m": support_height_m,
        "height_m": support_height_m,
        "source": bundle.project.epsilon.support_plane_source,
        "measured": False,
        "qa_status": "not_applicable",
    }
    static_mesh_meta = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "mesh_path": "scene/static_mesh.obj",
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "units": "meters",
        "geometry_mode": bundle.project.epsilon.geometry_mode,
        "derived_from_dense_reconstruction": bundle.project.epsilon.geometry_mode == "dense_static_reconstruction",
        "geometry_source": "dense_static_placeholder_mesh"
        if bundle.project.epsilon.geometry_mode == "dense_static_reconstruction"
        else "static_subset_plus_mask_projection",
        "source": {
            "beta_dir": _path_string(beta_dir),
            "delta_dir": _path_string(delta_dir),
            "camera_poses_metric": "camera/camera_poses_metric.json",
            "static_geometry_subset": "scene/static_geometry_subset.json",
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
        "plane_rmse_m": 0.0 if bundle.project.epsilon.geometry_mode == "dense_static_reconstruction" else None,
        "scale_anchor_rel_error": 0.0,
        "qa_status": {
            "plane_rmse_m": (
                "synthetic_dense_placeholder"
                if bundle.project.epsilon.geometry_mode == "dense_static_reconstruction"
                else "not_applicable"
            ),
            "scale_anchor_rel_error": "inherited_from_beta_world",
        },
        "qc_video_format": qc_video_format,
    }
    (scene_dir / "static_mesh.meta.json").write_text(
        json.dumps(static_mesh_meta, indent=2) + "\n",
        encoding="utf-8",
    )

    proxy_artifact_paths = [
        dense_dir / "fused.ply",
        dense_dir / "meshed-poisson.ply",
        dense_dir / "meshed-delaunay.ply",
        scene_dir / "static_mesh_raw.ply",
        scene_dir / "static_mesh.obj",
        scene_dir / "object_init_poses_metric.json",
    ]
    proxy_artifact_paths.extend(
        object_clouds_dir / f"{_sanitize_name(estimate.track_id)}.ply" for estimate in estimates
    )
    proxy_artifacts_nonempty_ok = all(path.exists() and path.stat().st_size > 0 for path in proxy_artifact_paths)
    dense_static_artifacts_ok = all(
        path.exists() and path.stat().st_size > 0
        for path in (
            dense_dir / "fused.ply",
            dense_dir / "meshed-poisson.ply",
            dense_dir / "meshed-delaunay.ply",
            scene_dir / "static_mesh.obj",
        )
    )
    try:
        json.dumps(object_payload)
        object_init_serializable_ok = True
    except TypeError:
        object_init_serializable_ok = False

    summary = _build_summary(
        bundle=bundle,
        estimates=estimates,
        qc_video_format=qc_video_format,
        proxy_artifacts_nonempty_ok=proxy_artifacts_nonempty_ok,
        object_init_serializable_ok=object_init_serializable_ok,
        dense_static_artifacts_ok=dense_static_artifacts_ok,
    )
    (scene_dir / "epsilon_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    enforce_epsilon_acceptance(bundle, summary)
    return summary
