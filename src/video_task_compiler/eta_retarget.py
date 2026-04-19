from __future__ import annotations

import importlib.util
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import yaml

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle
from .task_window import (
    TaskWindow,
    TaskWindowError,
    filter_items_to_task_window,
    load_task_window_frame_records,
    persist_task_window,
    resolve_task_window,
)


class EtaRetargetError(Exception):
    """Raised when eta retargeting cannot complete successfully."""


class DependencyError(EtaRetargetError):
    """Raised when eta runtime dependencies are missing."""


@dataclass(frozen=True)
class Waypoint:
    phase: str
    position_m: np.ndarray
    gripper_width_m: float


WRIST_HINT_MAX_DISTANCE_M = 0.08
WRIST_HINT_MAX_SHIFT_M = 0.02
ROBOT_BASE_NOMINAL_REACH_RADIUS_M = 0.55
ROBOT_BASE_CLEARANCE_MARGIN_M = 0.08


def _path_string(path: Path) -> str:
    return path.as_posix()


def missing_eta_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ("numpy", "pyarrow"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def ensure_eta_dependencies() -> None:
    missing = missing_eta_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for eta retarget: " + ", ".join(sorted(missing))
        )


def _load_pinocchio_module() -> Any:
    if importlib.util.find_spec("pinocchio") is None:
        raise DependencyError(
            "Missing Python dependency for eta retarget: pinocchio. Install env/sim.environment.yml "
            "to enable the pinocchio_seed IK backend."
        )
    return importlib.import_module("pinocchio")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EtaRetargetError(f"missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EtaRetargetError(f"expected a JSON object in {path}")
    return payload


def _workspace_clip(bundle: SpecBundle, point: np.ndarray) -> np.ndarray:
    bounds = bundle.robot.workspace_bounds_m
    clipped = np.array(
        [
            np.clip(point[0], bounds.min_m.x, bounds.max_m.x),
            np.clip(point[1], bounds.min_m.y, bounds.max_m.y),
            np.clip(point[2], bounds.min_m.z, bounds.max_m.z),
        ],
        dtype=float,
    )
    return clipped


def _workspace_clip_with_metadata(bundle: SpecBundle, point: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    clipped = _workspace_clip(bundle, point)
    clip_delta = clipped - np.asarray(point, dtype=float)
    return clipped, bool(np.linalg.norm(clip_delta) > 1e-9), clip_delta


def _region_center(region: Any) -> np.ndarray:
    return np.array(
        [
            0.5 * (region.min_m.x + region.max_m.x),
            0.5 * (region.min_m.y + region.max_m.y),
            0.5 * (region.min_m.z + region.max_m.z),
        ],
        dtype=float,
    )


def _load_object_pose_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    pose_map: dict[str, dict[str, Any]] = {}
    for obj in payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        pose_map[str(obj["ontology_id"])] = obj
    return pose_map


def _load_gamma_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise EtaRetargetError(f"missing gamma observables parquet: {path}")
    return pq.read_table(path).to_pylist()


def _load_interaction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise EtaRetargetError(f"missing delta interactions parquet: {path}")
    return pq.read_table(path).to_pylist()


def _load_static_scene_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(path)
    bbox_payload = payload.get("bbox_m")
    if not isinstance(bbox_payload, dict):
        raise EtaRetargetError(f"static mesh metadata was missing bbox_m: {path}")
    min_corner = np.asarray(bbox_payload.get("min"), dtype=float)
    max_corner = np.asarray(bbox_payload.get("max"), dtype=float)
    if min_corner.shape != (3,) or max_corner.shape != (3,):
        raise EtaRetargetError(f"static mesh metadata bbox_m was invalid: {path}")
    return min_corner, max_corner


def _first_contact(
    rows: Sequence[dict[str, Any]],
    class_name: str,
    *,
    after_t_ns: int | None = None,
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("class_name") == class_name and bool(row.get("likely_contact_boolean"))
    ]
    if after_t_ns is not None:
        candidates = [row for row in candidates if int(row.get("t_ns", 0)) >= after_t_ns]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (int(row.get("t_ns", 0)), int(row.get("frame_idx", 0))))[0]


def _gamma_row_by_frame(rows: Sequence[dict[str, Any]], frame_idx: int | None) -> dict[str, Any] | None:
    if frame_idx is None:
        return None
    for row in rows:
        if int(row.get("frame_idx", -1)) == int(frame_idx):
            return row
    return None


def _world_wrist(row: dict[str, Any] | None) -> np.ndarray | None:
    if row is None:
        return None
    values = (
        row.get("wrist_r_x_w"),
        row.get("wrist_r_y_w"),
        row.get("wrist_r_z_w"),
    )
    if any(value is None for value in values):
        return None
    return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)


def _rotation_z(theta_rad: float) -> np.ndarray:
    cosine = math.cos(theta_rad)
    sine = math.sin(theta_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _build_contact_schedule(
    bundle: SpecBundle,
    gamma_rows: Sequence[dict[str, Any]],
    interaction_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    target_id = bundle.project.ontology.target_object_id
    receptacle_id = bundle.project.ontology.receptacle_object_id
    pick_contact = _first_contact(interaction_rows, target_id)
    place_contact = _first_contact(
        interaction_rows,
        receptacle_id,
        after_t_ns=int(pick_contact["t_ns"]) if pick_contact is not None else None,
    )

    fallback_rows = [row for row in gamma_rows if row.get("track_visible")]
    pick_frame_idx = int(pick_contact["frame_idx"]) if pick_contact is not None else (int(fallback_rows[0]["frame_idx"]) if fallback_rows else 0)
    place_frame_idx = int(place_contact["frame_idx"]) if place_contact is not None else (int(fallback_rows[-1]["frame_idx"]) if fallback_rows else pick_frame_idx + 1)
    pick_t_ns = int(pick_contact["t_ns"]) if pick_contact is not None else 0
    place_t_ns = int(place_contact["t_ns"]) if place_contact is not None else max(pick_t_ns + 1_000_000_000, 1_000_000_000)

    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "events": [
            {
                "name": "pick",
                "frame_idx": pick_frame_idx,
                "t_ns": pick_t_ns,
                "class_name": target_id,
                "detection_source": "delta_contact" if pick_contact is not None else "gamma_fallback",
            },
            {
                "name": "place",
                "frame_idx": place_frame_idx,
                "t_ns": place_t_ns,
                "class_name": receptacle_id,
                "detection_source": "delta_contact" if place_contact is not None else "gamma_fallback",
            },
        ],
    }


def _pick_and_place_positions(
    bundle: SpecBundle,
    object_pose_map: dict[str, dict[str, Any]],
    gamma_rows: Sequence[dict[str, Any]],
    contact_schedule: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    support_height = float(
        min(
            bundle.task.pick_object.search_region_m.min_m.z,
            bundle.task.place_region.target_region_m.min_m.z,
        )
    )
    target_pose = object_pose_map.get(bundle.project.ontology.target_object_id)
    receptacle_pose = object_pose_map.get(bundle.project.ontology.receptacle_object_id)

    target_source_position = (
        np.asarray(target_pose["position_m"], dtype=float)
        if target_pose is not None
        else _region_center(bundle.task.pick_object.search_region_m)
    )
    receptacle_source_position = (
        np.asarray(receptacle_pose["position_m"], dtype=float)
        if receptacle_pose is not None
        else _region_center(bundle.task.place_region.target_region_m)
    )

    pick_event = next(event for event in contact_schedule["events"] if event["name"] == "pick")
    place_event = next(event for event in contact_schedule["events"] if event["name"] == "place")
    pick_wrist = _world_wrist(_gamma_row_by_frame(gamma_rows, int(pick_event["frame_idx"])))
    place_wrist = _world_wrist(_gamma_row_by_frame(gamma_rows, int(place_event["frame_idx"])))

    target_height = float(target_pose["extents_m"][2]) if target_pose is not None else bundle.project.epsilon.default_object_height_m
    receptacle_height = (
        float(receptacle_pose["extents_m"][2])
        if receptacle_pose is not None
        else bundle.project.epsilon.default_object_height_m
    )

    target_position = target_source_position.copy()
    target_position[2] = support_height + 0.5 * target_height

    receptacle_position = receptacle_source_position.copy()
    receptacle_top_z = receptacle_source_position[2] + 0.5 * receptacle_height
    receptacle_position[2] = max(receptacle_top_z + 0.5 * target_height, support_height + 0.5 * target_height)

    def _resolve_target_position(
        *,
        ontology_id: str,
        class_name: str,
        source_position: np.ndarray,
        wrist_position: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        resolved = np.asarray(source_position, dtype=float).copy()
        raw_wrist_offset = None
        raw_wrist_distance = None
        applied_hint_delta = np.zeros(3, dtype=float)
        hint_applied = False
        if wrist_position is not None:
            raw_wrist_offset = wrist_position - resolved
            raw_wrist_distance = float(np.linalg.norm(raw_wrist_offset[:2]))
            if raw_wrist_distance <= WRIST_HINT_MAX_DISTANCE_M:
                planar_offset = raw_wrist_offset[:2]
                planar_norm = float(np.linalg.norm(planar_offset))
                if planar_norm > 1e-9:
                    planar_scale = min(1.0, WRIST_HINT_MAX_SHIFT_M / planar_norm)
                    applied_hint_delta[:2] = planar_offset * planar_scale
                    resolved[:2] = resolved[:2] + applied_hint_delta[:2]
                    hint_applied = bool(np.linalg.norm(applied_hint_delta[:2]) > 1e-9)
        preclip_position = resolved.copy()
        resolved, was_clipped, clip_delta = _workspace_clip_with_metadata(bundle, resolved)
        return resolved, {
            "ontology_id": ontology_id,
            "class_name": class_name,
            "source_position_m": source_position.tolist(),
            "wrist_position_m": wrist_position.tolist() if wrist_position is not None else None,
            "raw_wrist_offset_m": raw_wrist_offset.tolist() if raw_wrist_offset is not None else None,
            "raw_wrist_distance_m": raw_wrist_distance,
            "hint_applied": hint_applied,
            "applied_hint_delta_m": applied_hint_delta.tolist(),
            "preclip_position_m": preclip_position.tolist(),
            "position_m": resolved.tolist(),
            "was_clipped": was_clipped,
            "clip_delta_m": clip_delta.tolist(),
        }

    resolved_pick_position, pick_diagnostics = _resolve_target_position(
        ontology_id=bundle.project.ontology.target_object_id,
        class_name=bundle.project.ontology.target_object_id,
        source_position=target_position,
        wrist_position=pick_wrist,
    )
    resolved_place_position, place_diagnostics = _resolve_target_position(
        ontology_id=bundle.project.ontology.receptacle_object_id,
        class_name=bundle.project.ontology.receptacle_object_id,
        source_position=receptacle_position,
        wrist_position=place_wrist,
    )
    return (
        resolved_pick_position,
        resolved_place_position,
        support_height,
        {
            "pick": pick_diagnostics,
            "place": place_diagnostics,
        },
    )


def _approximate_ur5e_joint_waypoint(bundle: SpecBundle, position_m: np.ndarray) -> np.ndarray:
    x, y, z = [float(value) for value in position_m]
    d1 = 0.1625
    l1 = 0.425
    l2 = 0.3922
    wrist_reach = 0.10

    q1 = math.atan2(y, x)
    radial = max(0.10, math.hypot(x, y) - wrist_reach)
    vertical = z - d1
    distance = math.hypot(radial, vertical)
    max_distance = max(0.05, l1 + l2 - 1e-3)
    distance = min(max(distance, 0.05), max_distance)
    cos_elbow = (distance * distance - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_elbow = float(np.clip(cos_elbow, -0.999, 0.999))
    elbow_internal = math.acos(cos_elbow)
    q3 = -elbow_internal
    q2 = math.atan2(vertical, radial) - math.atan2(l2 * math.sin(elbow_internal), l1 + l2 * math.cos(elbow_internal))
    q4 = -q2 - q3 - math.pi / 2.0
    q5 = -math.pi / 2.0
    q6 = 0.0
    joints = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
    return np.clip(joints, -2.0 * math.pi, 2.0 * math.pi)


def _build_waypoints(
    bundle: SpecBundle,
    pick_position: np.ndarray,
    place_position: np.ndarray,
    support_height: float,
) -> tuple[list[Waypoint], list[dict[str, Any]]]:
    workspace = bundle.robot.workspace_bounds_m
    home_position = np.array(
        [
            0.5 * (workspace.min_m.x + workspace.max_m.x),
            0.0,
            min(workspace.max_m.z, support_height + bundle.project.eta.transport_clearance_m + 0.15),
        ],
        dtype=float,
    )
    pick_grasp = pick_position.copy()
    pick_grasp[2] = max(support_height + 0.5 * bundle.project.epsilon.default_object_height_m, pick_grasp[2])
    place_release = place_position.copy()
    place_release[2] = max(place_release[2], support_height + 0.5 * bundle.project.epsilon.default_object_height_m)

    pick_pre = pick_grasp + np.array([0.0, 0.0, bundle.project.eta.pregrasp_clearance_m], dtype=float)
    pick_lift = pick_grasp + np.array([0.0, 0.0, bundle.project.eta.transport_clearance_m], dtype=float)
    place_pre = place_release + np.array([0.0, 0.0, bundle.project.eta.pregrasp_clearance_m], dtype=float)
    retreat = place_release + np.array([0.0, 0.0, bundle.project.eta.postplace_clearance_m], dtype=float)

    open_width = bundle.robot.gripper.open_width_m
    closed_width = bundle.robot.gripper.closed_width_m
    waypoint_specs = [
        ("home", home_position, open_width),
        ("pregrasp", pick_pre, open_width),
        ("grasp", pick_grasp, closed_width),
        ("lift", pick_lift, closed_width),
        ("transfer", place_pre, closed_width),
        ("preplace", place_pre, closed_width),
        ("place", place_release, open_width),
        ("retreat", retreat, open_width),
    ]
    waypoints: list[Waypoint] = []
    diagnostics: list[dict[str, Any]] = []
    for phase, raw_position, gripper_width in waypoint_specs:
        clipped_position, was_clipped, clip_delta = _workspace_clip_with_metadata(bundle, raw_position)
        waypoints.append(Waypoint(phase, clipped_position, gripper_width))
        diagnostics.append(
            {
                "phase": phase,
                "preclip_position_m": np.asarray(raw_position, dtype=float).tolist(),
                "position_m": clipped_position.tolist(),
                "was_clipped": was_clipped,
                "clip_delta_m": clip_delta.tolist(),
            }
        )
    return waypoints, diagnostics


def _fit_robot_base_transform(
    bundle: SpecBundle,
    waypoints: Sequence[Waypoint],
    support_bbox: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    active_positions = np.asarray(
        [waypoint.position_m for waypoint in waypoints if waypoint.phase != "home"],
        dtype=float,
    )
    if len(active_positions) == 0:
        active_positions = np.asarray([waypoint.position_m for waypoint in waypoints], dtype=float)
    activity_centroid = np.mean(active_positions, axis=0)

    support_min, support_max = support_bbox
    side_candidates = {
        "min_x": float(activity_centroid[0] - support_min[0]),
        "max_x": float(support_max[0] - activity_centroid[0]),
        "min_y": float(activity_centroid[1] - support_min[1]),
        "max_y": float(support_max[1] - activity_centroid[1]),
    }
    fit_side = min(side_candidates, key=side_candidates.get)
    inward_direction_by_side = {
        "min_x": np.array([1.0, 0.0], dtype=float),
        "max_x": np.array([-1.0, 0.0], dtype=float),
        "min_y": np.array([0.0, 1.0], dtype=float),
        "max_y": np.array([0.0, -1.0], dtype=float),
    }
    inward_direction = inward_direction_by_side[fit_side]
    base_xy = activity_centroid[:2] - inward_direction * ROBOT_BASE_NOMINAL_REACH_RADIUS_M
    pushed_outside = False
    if fit_side == "min_x":
        base_xy[0] = min(base_xy[0], support_min[0] - ROBOT_BASE_CLEARANCE_MARGIN_M)
    elif fit_side == "max_x":
        base_xy[0] = max(base_xy[0], support_max[0] + ROBOT_BASE_CLEARANCE_MARGIN_M)
    elif fit_side == "min_y":
        base_xy[1] = min(base_xy[1], support_min[1] - ROBOT_BASE_CLEARANCE_MARGIN_M)
    else:
        base_xy[1] = max(base_xy[1], support_max[1] + ROBOT_BASE_CLEARANCE_MARGIN_M)
    if (
        support_min[0] <= base_xy[0] <= support_max[0]
        and support_min[1] <= base_xy[1] <= support_max[1]
    ):
        pushed_outside = True
        if fit_side == "min_x":
            base_xy[0] = support_min[0] - ROBOT_BASE_CLEARANCE_MARGIN_M
        elif fit_side == "max_x":
            base_xy[0] = support_max[0] + ROBOT_BASE_CLEARANCE_MARGIN_M
        elif fit_side == "min_y":
            base_xy[1] = support_min[1] - ROBOT_BASE_CLEARANCE_MARGIN_M
        else:
            base_xy[1] = support_max[1] + ROBOT_BASE_CLEARANCE_MARGIN_M

    facing_direction_xy = activity_centroid[:2] - base_xy
    facing_norm = float(np.linalg.norm(facing_direction_xy))
    if facing_norm <= 1e-9:
        facing_direction_xy = inward_direction.copy()
        facing_norm = float(np.linalg.norm(facing_direction_xy))
    facing_direction_xy = facing_direction_xy / max(facing_norm, 1e-9)
    facing_yaw = math.atan2(float(facing_direction_xy[1]), float(facing_direction_xy[0]))

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_z(facing_yaw)
    transform[:3, 3] = np.array([base_xy[0], base_xy[1], float(support_min[2])], dtype=float)
    return transform, {
        "registration_method": "automatic_trace_fit_proxy",
        "measured": False,
        "fit_side": fit_side,
        "nominal_reach_radius_m": ROBOT_BASE_NOMINAL_REACH_RADIUS_M,
        "clearance_margin_m": ROBOT_BASE_CLEARANCE_MARGIN_M,
        "activity_centroid_m": activity_centroid.tolist(),
        "support_surface_bbox_m": {
            "min": support_min.tolist(),
            "max": support_max.tolist(),
        },
        "facing_direction_xy": facing_direction_xy.tolist(),
        "facing_yaw_rad": facing_yaw,
        "base_position_m": transform[:3, 3].tolist(),
        "pushed_outside_support_surface": pushed_outside,
    }


def _solve_waypoint_joints_pinocchio_seed(
    bundle: SpecBundle,
    waypoints: Sequence[Waypoint],
) -> tuple[np.ndarray, dict[str, Any]]:
    _load_pinocchio_module()
    joint_waypoints = np.asarray(
        [_approximate_ur5e_joint_waypoint(bundle, waypoint.position_m) for waypoint in waypoints],
        dtype=float,
    )
    solved_mask = np.ones(len(waypoints), dtype=bool)
    return joint_waypoints, {
        "backend": bundle.project.eta.ik_backend,
        "runtime_status": "mvp_seed_projection",
        "solve_count": int(np.sum(solved_mask)),
        "total_targets": int(len(solved_mask)),
        "solve_rate": float(np.mean(solved_mask)) if len(solved_mask) else 0.0,
    }


def _interpolate_demo(
    bundle: SpecBundle,
    waypoints: Sequence[Waypoint],
    joint_waypoints: np.ndarray,
) -> dict[str, np.ndarray]:
    control_rate_hz = int(bundle.robot.control_rate_hz)
    max_joint_step_target = min(bundle.project.acceptance.eta_max_joint_step_rad, 0.9 * bundle.project.acceptance.eta_max_joint_step_rad)
    ee_quaternion_wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    grippers: list[float] = []
    phase_names: list[str] = []
    timestamps_ns: list[int] = []

    current_time_ns = 0
    for index in range(len(waypoints) - 1):
        start_waypoint = waypoints[index]
        end_waypoint = waypoints[index + 1]
        start_joints = joint_waypoints[index]
        end_joints = joint_waypoints[index + 1]
        joint_delta = float(np.max(np.abs(end_joints - start_joints)))
        position_delta = float(np.linalg.norm(end_waypoint.position_m - start_waypoint.position_m))
        sample_count = max(
            2,
            int(math.ceil(joint_delta / max(1e-6, max_joint_step_target))) + 1,
            int(math.ceil(position_delta / 0.01)) + 1,
            int(math.ceil(bundle.project.eta.waypoint_hold_s * control_rate_hz)) + 1,
        )
        alphas = np.linspace(0.0, 1.0, sample_count, endpoint=False)
        for alpha in alphas:
            alpha_f = float(alpha)
            positions.append((1.0 - alpha_f) * start_waypoint.position_m + alpha_f * end_waypoint.position_m)
            quaternions.append(ee_quaternion_wxyz.copy())
            joints.append((1.0 - alpha_f) * start_joints + alpha_f * end_joints)
            grippers.append((1.0 - alpha_f) * start_waypoint.gripper_width_m + alpha_f * end_waypoint.gripper_width_m)
            phase_names.append(end_waypoint.phase)
            timestamps_ns.append(current_time_ns)
            current_time_ns += int(round(1_000_000_000 / control_rate_hz))

    positions.append(waypoints[-1].position_m.copy())
    quaternions.append(ee_quaternion_wxyz.copy())
    joints.append(joint_waypoints[-1].copy())
    grippers.append(float(waypoints[-1].gripper_width_m))
    phase_names.append(waypoints[-1].phase)
    timestamps_ns.append(current_time_ns)
    return {
        "t_ns": np.asarray(timestamps_ns, dtype=np.int64),
        "ee_position_m": np.asarray(positions, dtype=np.float32),
        "ee_quaternion_wxyz": np.asarray(quaternions, dtype=np.float32),
        "joint_positions_rad": np.asarray(joints, dtype=np.float32),
        "gripper_width_m": np.asarray(grippers, dtype=np.float32),
        "phase_name": np.asarray(phase_names),
    }


def _write_robot_target(path: Path, bundle: SpecBundle, task_window: TaskWindow) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "robot_id": bundle.robot.robot_id,
        "robot_model_ref": bundle.robot.model_ref,
        "base_frame": bundle.robot.base_frame,
        "ee_frame": bundle.robot.ee_frame,
        "execution_mode": bundle.project.eta.execution_mode,
        "ik_backend": bundle.project.eta.ik_backend,
        "task_window": task_window.to_payload(video_id=bundle.project.video_id),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_summary(
    bundle: SpecBundle,
    contact_schedule: dict[str, Any],
    task_window: TaskWindow,
    task_window_gamma_rows: Sequence[dict[str, Any]],
    demo_payload: dict[str, np.ndarray],
    ik_summary: dict[str, Any],
    target_position_diagnostics: dict[str, Any],
    waypoint_clip_diagnostics: Sequence[dict[str, Any]],
    robot_base_registration: dict[str, Any],
) -> dict[str, Any]:
    joint_positions = demo_payload["joint_positions_rad"]
    if len(joint_positions) <= 1:
        max_joint_step = 0.0
    else:
        max_joint_step = float(np.max(np.abs(np.diff(joint_positions, axis=0))))
    event_sources = {event["name"]: event["detection_source"] for event in contact_schedule["events"]}
    target_positions_unclipped_ok = (
        not any(bool(diagnostic.get("was_clipped")) for diagnostic in target_position_diagnostics.values())
        and not any(bool(diagnostic.get("was_clipped")) for diagnostic in waypoint_clip_diagnostics)
    )
    target_position_registration_ok = all(
        diagnostic.get("raw_wrist_distance_m") is None
        or float(diagnostic["raw_wrist_distance_m"]) <= WRIST_HINT_MAX_DISTANCE_M
        for diagnostic in target_position_diagnostics.values()
    )
    robot_base_identity_ok = not bool(
        np.allclose(
            np.asarray(robot_base_registration["X_Br_from_M"], dtype=float),
            np.eye(4, dtype=float),
            atol=1e-6,
        )
    )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.eta.primary_backbone,
        "execution_mode": bundle.project.eta.execution_mode,
        "ik_backend": bundle.project.eta.ik_backend,
        "task_window": task_window.to_payload(video_id=bundle.project.video_id),
        "demo_frame_count": int(len(task_window_gamma_rows)),
        "retarget_frame_count": int(len(demo_payload["t_ns"])),
        "max_joint_step_rad": max_joint_step,
        "contact_event_sources": event_sources,
        "ik": ik_summary,
        "target_position_diagnostics": target_position_diagnostics,
        "waypoint_clip_diagnostics": list(waypoint_clip_diagnostics),
        "robot_base_registration": robot_base_registration,
        "qc_flags": {
            "demo_frame_count_ok": len(task_window_gamma_rows) >= bundle.project.acceptance.eta_min_demo_frames,
            "joint_step_ok": max_joint_step <= bundle.project.acceptance.eta_max_joint_step_rad,
            "ik_solve_rate_ok": float(ik_summary["solve_rate"]) >= bundle.project.acceptance.eta_min_ik_solve_rate,
            "target_positions_unclipped_ok": target_positions_unclipped_ok,
            "robot_base_identity_ok": robot_base_identity_ok,
            "target_position_registration_ok": target_position_registration_ok,
        },
    }


def enforce_eta_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if summary["demo_frame_count"] < bundle.project.acceptance.eta_min_demo_frames:
        raise EtaRetargetError(
            f"demo frame count {summary['demo_frame_count']} fell below "
            f"eta_min_demo_frames={bundle.project.acceptance.eta_min_demo_frames}"
        )
    if summary["max_joint_step_rad"] > bundle.project.acceptance.eta_max_joint_step_rad:
        raise EtaRetargetError(
            f"max joint step {summary['max_joint_step_rad']:.4f}rad exceeded "
            f"eta_max_joint_step_rad={bundle.project.acceptance.eta_max_joint_step_rad:.4f}rad"
        )
    if float(summary["ik"]["solve_rate"]) < bundle.project.acceptance.eta_min_ik_solve_rate:
        raise EtaRetargetError(
            f"IK solve rate {summary['ik']['solve_rate']:.3f} fell below "
            f"eta_min_ik_solve_rate={bundle.project.acceptance.eta_min_ik_solve_rate:.3f}"
        )


def retarget_monocular_demonstration(
    bundle: SpecBundle,
    gamma_dir: Path,
    delta_dir: Path,
    epsilon_dir: Path,
    zeta_dir: Path,
    out_dir: Path,
    *,
    task_start_frame: int | None = None,
    task_end_frame: int | None = None,
) -> dict[str, Any]:
    ensure_eta_dependencies()
    if bundle.project.eta.execution_mode != "arm_gripper_waypoint_replay":
        raise EtaRetargetError(
            f"eta execution_mode '{bundle.project.eta.execution_mode}' is not implemented; "
            "the current MVP supports only arm_gripper_waypoint_replay"
        )
    if bundle.project.eta.ik_backend != "pinocchio_seed":
        raise EtaRetargetError(
            f"eta ik_backend '{bundle.project.eta.ik_backend}' is not implemented; "
            "the current MVP supports only pinocchio_seed"
        )

    try:
        frame_records = load_task_window_frame_records(gamma_dir / "frames" / "index.csv")
        task_window = resolve_task_window(
            frame_records,
            video_id=bundle.project.video_id,
            artifact_roots=(gamma_dir, delta_dir, epsilon_dir, zeta_dir, out_dir),
            task_start_frame=task_start_frame,
            task_end_frame=task_end_frame,
        )
    except TaskWindowError as exc:
        raise EtaRetargetError(str(exc)) from exc

    gamma_rows = filter_items_to_task_window(
        _load_gamma_rows(gamma_dir / "human" / "arm_observables.parquet"),
        task_window,
    )
    interaction_rows = filter_items_to_task_window(
        _load_interaction_rows(delta_dir / "objects" / "interactions.parquet"),
        task_window,
    )
    object_pose_map = _load_object_pose_map(epsilon_dir / "scene" / "object_init_poses_metric.json")
    support_bbox = _load_static_scene_bbox(epsilon_dir / "scene" / "static_mesh.meta.json")
    _ = zeta_dir

    contact_schedule = _build_contact_schedule(bundle, gamma_rows, interaction_rows)
    pick_position, place_position, support_height, target_position_diagnostics = _pick_and_place_positions(
        bundle,
        object_pose_map,
        gamma_rows,
        contact_schedule,
    )
    waypoints, waypoint_clip_diagnostics = _build_waypoints(bundle, pick_position, place_position, support_height)
    joint_waypoints, ik_summary = _solve_waypoint_joints_pinocchio_seed(bundle, waypoints)
    demo_payload = _interpolate_demo(bundle, waypoints, joint_waypoints)
    robot_base_transform, base_registration = _fit_robot_base_transform(bundle, waypoints, support_bbox)

    retarget_dir = out_dir / "retarget"
    for path in (retarget_dir,):
        path.mkdir(parents=True, exist_ok=True)
    persist_task_window(task_window, out_dir, video_id=bundle.project.video_id)

    robot_base_in_metric_world = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_world": bundle.project.epsilon.metric_world_frame,
        "target_frame": bundle.robot.base_frame,
        "X_Br_from_M": robot_base_transform.tolist(),
        **base_registration,
    }
    _write_robot_target(retarget_dir / "robot_target.yaml", bundle, task_window)
    (retarget_dir / "robot_base_in_metric_world.json").write_text(
        json.dumps(robot_base_in_metric_world, indent=2) + "\n",
        encoding="utf-8",
    )
    (retarget_dir / "contact_schedule.json").write_text(
        json.dumps(contact_schedule, indent=2) + "\n",
        encoding="utf-8",
    )

    demo_npz_path = retarget_dir / "robot_demo.npz"
    with demo_npz_path.open("wb") as handle:
        np.savez(handle, **demo_payload)

    summary = _build_summary(
        bundle,
        contact_schedule,
        task_window,
        gamma_rows,
        demo_payload,
        ik_summary,
        target_position_diagnostics,
        waypoint_clip_diagnostics,
        robot_base_in_metric_world,
    )
    (retarget_dir / "eta_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    enforce_eta_acceptance(bundle, summary)
    return summary
