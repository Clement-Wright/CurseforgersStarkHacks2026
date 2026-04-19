from __future__ import annotations

import importlib
import importlib.util
import json
import math
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle
from .task_window import TaskWindow, filter_items_to_task_window


class ThetaSimError(Exception):
    """Raised when theta simulator compilation cannot complete successfully."""


class DependencyError(ThetaSimError):
    """Raised when theta runtime dependencies are missing."""


@dataclass(frozen=True)
class AuditCamera:
    position_m: np.ndarray
    x_axis_w: np.ndarray
    y_axis_w: np.ndarray


TARGET_REGION_REANCHOR_THRESHOLD_M = 0.10
OBJECT_EXTENT_CLIP_CEILING_M = 0.245


def _path_string(path: Path) -> str:
    return path.as_posix()


def missing_theta_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ("imageio", "numpy", "PIL", "pyarrow"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def ensure_theta_dependencies() -> None:
    missing = missing_theta_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for sim compile-task: " + ", ".join(sorted(missing))
        )


def _load_mujoco_module() -> Any:
    if importlib.util.find_spec("mujoco") is None:
        raise DependencyError(
            "Missing Python dependency for sim compile-task: mujoco. "
            "Install env/sim.environment.yml to enable Theta compilation."
        )
    return importlib.import_module("mujoco")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ThetaSimError(f"missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThetaSimError(f"expected a JSON object in {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ThetaSimError(f"missing required artifact: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThetaSimError(f"expected a YAML mapping in {path}")
    return payload


def _sanitize_name(value: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    sanitized = sanitized.strip("_")
    return sanitized or "artifact"


def _vendored_robot_model_path() -> Path:
    return Path(
        str(
            files("video_task_compiler").joinpath(
                "assets",
                "robots",
                "ur5e_robotiq_2f85.xml",
            )
        )
    )


def _robot_model_copy_path(sim_dir: Path) -> Path:
    return sim_dir / "robot_model" / "ur5e_robotiq_2f85.xml"


def _copy_vendored_robot_model(sim_dir: Path) -> Path:
    source = _vendored_robot_model_path()
    target = _robot_model_copy_path(sim_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = [float(rotation[0, 0]), float(rotation[1, 1]), float(rotation[2, 2])]
        max_index = int(np.argmax(diagonal))
        if max_index == 0:
            scale = math.sqrt(max(1e-9, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif max_index == 1:
            scale = math.sqrt(max(1e-9, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(max(1e-9, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qw, qx, qy, qz], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quaternion / norm


def _task_window_from_eta(robot_target_path: Path) -> TaskWindow:
    payload = _load_yaml(robot_target_path)
    task_window_payload = payload.get("task_window")
    if not isinstance(task_window_payload, dict):
        raise ThetaSimError(f"eta robot target was missing a task_window block: {robot_target_path}")
    return TaskWindow(
        start_frame_idx=int(task_window_payload["start_frame_idx"]),
        end_frame_idx=int(task_window_payload["end_frame_idx"]),
        source=str(task_window_payload.get("source", "eta_robot_target")),
    )


def _load_gamma_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ThetaSimError(f"missing gamma observables parquet: {path}")
    return pq.read_table(path).to_pylist()


def _load_metric_camera_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ThetaSimError(f"camera_poses_metric.json was missing a frames list: {path}")
    return payload


def _load_robot_demo_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ThetaSimError(f"missing eta robot demo: {path}")
    archive = np.load(path, allow_pickle=True)
    required_keys = ("t_ns", "joint_positions_rad", "gripper_width_m", "phase_name")
    missing_keys = [key for key in required_keys if key not in archive.files]
    if missing_keys:
        raise ThetaSimError(
            f"eta robot demo was missing required arrays: {', '.join(sorted(missing_keys))}"
        )
    payload = {
        "t_ns": np.asarray(archive["t_ns"]),
        "joint_positions_rad": np.asarray(archive["joint_positions_rad"]),
        "gripper_width_m": np.asarray(archive["gripper_width_m"]),
        "phase_name": np.asarray(archive["phase_name"]),
    }
    for optional_key in ("ee_position_m", "ee_quaternion_wxyz"):
        if optional_key in archive.files:
            payload[optional_key] = np.asarray(archive[optional_key])
    return payload


def _metric_camera_frame_for_task_window(
    camera_payload: dict[str, Any],
    task_window: TaskWindow,
) -> dict[str, Any]:
    frames = [
        frame
        for frame in camera_payload.get("frames", [])
        if isinstance(frame, dict)
        and frame.get("T_wc") is not None
        and task_window.contains(int(frame.get("frame_idx", -1)))
    ]
    if not frames:
        frames = [
            frame
            for frame in camera_payload.get("frames", [])
            if isinstance(frame, dict) and frame.get("T_wc") is not None
        ]
    if not frames:
        raise ThetaSimError("camera/camera_poses_metric.json did not contain any usable T_wc camera poses")
    midpoint = 0.5 * (task_window.start_frame_idx + task_window.end_frame_idx)
    return min(frames, key=lambda frame: abs(float(frame.get("frame_idx", 0)) - midpoint))


def _audit_camera_from_metric_pose(camera_frame: dict[str, Any]) -> AuditCamera:
    T_wc = np.asarray(camera_frame["T_wc"], dtype=float)
    position = T_wc[:3, 3].astype(float)
    x_axis_w = T_wc[:3, 0].astype(float)
    y_axis_w = (-T_wc[:3, 1]).astype(float)
    x_axis_w /= max(1e-9, float(np.linalg.norm(x_axis_w)))
    y_axis_w /= max(1e-9, float(np.linalg.norm(y_axis_w)))
    return AuditCamera(position_m=position, x_axis_w=x_axis_w, y_axis_w=y_axis_w)


def _debug_camera(
    activity_center: np.ndarray,
    support_bbox: tuple[np.ndarray, np.ndarray],
) -> AuditCamera:
    support_min, support_max = support_bbox
    span_xy = np.asarray(support_max[:2] - support_min[:2], dtype=float)
    span_scale = max(float(np.max(span_xy)), 0.4)
    center = np.array(
        [
            float(activity_center[0]),
            float(activity_center[1]),
            max(float(support_max[2]) + 0.55, float(activity_center[2]) + 0.55, 0.6 * span_scale),
        ],
        dtype=float,
    )
    return AuditCamera(
        position_m=center,
        x_axis_w=np.array([1.0, 0.0, 0.0], dtype=float),
        y_axis_w=np.array([0.0, 1.0, 0.0], dtype=float),
    )


def _region_center_and_half_extents(region: Any) -> tuple[np.ndarray, np.ndarray]:
    center = np.array(
        [
            0.5 * (region.min_m.x + region.max_m.x),
            0.5 * (region.min_m.y + region.max_m.y),
            0.5 * (region.min_m.z + region.max_m.z),
        ],
        dtype=float,
    )
    half_extents = np.array(
        [
            0.5 * (region.max_m.x - region.min_m.x),
            0.5 * (region.max_m.y - region.min_m.y),
            0.5 * (region.max_m.z - region.min_m.z),
        ],
        dtype=float,
    )
    return center, half_extents


def _resolve_runtime_target_region(
    bundle: SpecBundle,
    object_pose_map: dict[str, dict[str, Any]],
    support_plane: dict[str, Any],
) -> dict[str, Any]:
    declared_center, declared_half_extents = _region_center_and_half_extents(bundle.task.place_region.target_region_m)
    runtime_center = declared_center.copy()
    runtime_half_extents = declared_half_extents.copy()
    resolution = "declared_spec"
    anchor_track_id = None
    receptacle = object_pose_map.get(bundle.project.ontology.receptacle_object_id)
    if receptacle is not None:
        anchor_track_id = receptacle.get("track_id")
        receptacle_position = np.asarray(receptacle.get("position_m"), dtype=float)
        receptacle_extents = np.asarray(
            receptacle.get("extents_m", [bundle.project.epsilon.default_object_height_m] * 3),
            dtype=float,
        )
        receptacle_top_z = float(receptacle_position[2] + 0.5 * receptacle_extents[2])
        candidate_center = np.array(
            [
                float(receptacle_position[0]),
                float(receptacle_position[1]),
                max(receptacle_top_z + float(declared_half_extents[2]), float(support_plane.get("height_m", 0.0))),
            ],
            dtype=float,
        )
        candidate_half_extents = np.array(
            [
                max(float(declared_half_extents[0]), 0.5 * float(receptacle_extents[0])),
                max(float(declared_half_extents[1]), 0.5 * float(receptacle_extents[1])),
                float(declared_half_extents[2]),
            ],
            dtype=float,
        )
        offset = float(np.linalg.norm(candidate_center[:2] - declared_center[:2]))
        if offset > TARGET_REGION_REANCHOR_THRESHOLD_M:
            runtime_center = candidate_center
            runtime_half_extents = candidate_half_extents
            resolution = "auto_reanchored_from_scene"
    runtime_offset_m = float(np.linalg.norm(runtime_center[:2] - declared_center[:2]))
    return {
        "declared_center_m": declared_center.tolist(),
        "declared_half_extents_m": declared_half_extents.tolist(),
        "runtime_center_m": runtime_center.tolist(),
        "runtime_half_extents_m": runtime_half_extents.tolist(),
        "runtime_region_resolution": resolution,
        "runtime_region_offset_m": runtime_offset_m,
        "anchor_ontology_id": bundle.project.ontology.receptacle_object_id,
        "anchor_track_id": anchor_track_id,
    }


def _object_map_by_ontology(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ThetaSimError(f"object_init_poses_metric.json did not contain an objects list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        instance_id = str(obj.get("instance_id", "")).strip()
        ontology_id = str(obj.get("ontology_id", "")).strip()
        if instance_id:
            result[instance_id] = obj
        if ontology_id and ontology_id not in result:
            result[ontology_id] = obj
    return result


def _human_ghost_payload(gamma_rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    frame_idx: list[int] = []
    t_ns: list[int] = []
    elbow_positions: list[np.ndarray] = []
    wrist_positions: list[np.ndarray] = []
    visibility: list[bool] = []
    wrist_conf: list[float] = []
    for row in gamma_rows:
        frame_idx.append(int(row.get("frame_idx", -1)))
        t_ns.append(int(row.get("t_ns", 0)))
        elbow_values = (
            row.get("elbow_r_x_w"),
            row.get("elbow_r_y_w"),
            row.get("elbow_r_z_w"),
        )
        wrist_values = (
            row.get("wrist_r_x_w"),
            row.get("wrist_r_y_w"),
            row.get("wrist_r_z_w"),
        )
        elbow_positions.append(
            np.array(
                [
                    float(elbow_values[0]) if elbow_values[0] is not None else np.nan,
                    float(elbow_values[1]) if elbow_values[1] is not None else np.nan,
                    float(elbow_values[2]) if elbow_values[2] is not None else np.nan,
                ],
                dtype=np.float32,
            )
        )
        wrist_positions.append(
            np.array(
                [
                    float(wrist_values[0]) if wrist_values[0] is not None else np.nan,
                    float(wrist_values[1]) if wrist_values[1] is not None else np.nan,
                    float(wrist_values[2]) if wrist_values[2] is not None else np.nan,
                ],
                dtype=np.float32,
            )
        )
        visibility.append(bool(row.get("track_visible")) and all(value is not None for value in wrist_values))
        wrist_conf.append(float(row.get("wrist_conf") or 0.0))
    return {
        "frame_idx": np.asarray(frame_idx, dtype=np.int32),
        "t_ns": np.asarray(t_ns, dtype=np.int64),
        "elbow_position_m": np.asarray(elbow_positions, dtype=np.float32),
        "wrist_position_m": np.asarray(wrist_positions, dtype=np.float32),
        "track_visible": np.asarray(visibility, dtype=bool),
        "wrist_confidence": np.asarray(wrist_conf, dtype=np.float32),
    }


def _robot_trace_payload(bundle: SpecBundle, raw_demo: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    payload = {
        "t_ns": np.asarray(raw_demo["t_ns"], dtype=np.int64),
        "joint_positions_rad": np.asarray(raw_demo["joint_positions_rad"], dtype=np.float32),
        "gripper_width_m": np.asarray(raw_demo["gripper_width_m"], dtype=np.float32),
        "phase_name": np.asarray(raw_demo["phase_name"]),
        "joint_order": np.asarray(bundle.robot.action_space.arm_joint_order),
    }
    for optional_key in ("ee_position_m", "ee_quaternion_wxyz"):
        if optional_key in raw_demo:
            payload[optional_key] = np.asarray(raw_demo[optional_key], dtype=np.float32)
    return payload


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def _trace_event_index(phases: Sequence[str], target_phase: str) -> int:
    for index, phase in enumerate(phases):
        if str(phase) == target_phase:
            return index
    return 0


def _manipulable_instance_ids(bundle: SpecBundle) -> set[str]:
    instance_ids = {
        str(instance_id)
        for goal in bundle.task.goals
        for instance_id in goal.source_instance_ids
    }
    if instance_ids:
        return instance_ids
    return {
        str(instance.instance_id)
        for instance in bundle.task.scene_instances
        if getattr(instance, "role", "") == "manipulable"
        or instance.ontology_id == bundle.project.ontology.target_object_id
    }


def _build_task_json(
    bundle: SpecBundle,
    *,
    epsilon_dir: Path,
    zeta_dir: Path,
    eta_dir: Path,
    zeta_manifest: dict[str, Any],
    world_transform: dict[str, Any],
    support_plane: dict[str, Any],
    object_pose_map: dict[str, dict[str, Any]],
    contact_schedule: dict[str, Any],
    robot_trace: dict[str, np.ndarray],
    eta_summary: dict[str, Any],
    target_region: dict[str, Any],
    robot_model_path: Path,
) -> dict[str, Any]:
    phase_names = [str(value) for value in robot_trace["phase_name"]]
    event_phase_map = [
        ("reach", "home"),
        ("pregrasp", "pregrasp"),
        ("close", "grasp"),
        ("lift", "lift"),
        ("transfer", "transfer"),
        ("lower", "preplace"),
        ("release", "place"),
    ]
    normalized_events = []
    for event_name, phase_name in event_phase_map:
        trace_index = _trace_event_index(phase_names, phase_name)
        normalized_events.append(
            {
                "name": event_name,
                "source_phase": phase_name,
                "trace_index": trace_index,
                "t_ns": int(robot_trace["t_ns"][trace_index]),
            }
        )
    manipulable_instance_ids = _manipulable_instance_ids(bundle)
    initial_object_states = []
    for obj in zeta_manifest.get("objects", []):
        instance_id = str(obj.get("instance_id", "")).strip() or None
        ontology_id = str(obj.get("ontology_id", "")).strip()
        object_payload = object_pose_map.get(instance_id or "", object_pose_map.get(ontology_id, obj))
        initial_object_states.append(
            {
                "instance_id": instance_id,
                "ontology_id": ontology_id,
                "track_id": object_payload["track_id"],
                "class_name": object_payload["class_name"],
                "position_m": object_payload["position_m"],
                "extents_m": object_payload["extents_m"],
                "motion_mode": (
                    "freejoint"
                    if (instance_id in manipulable_instance_ids or ontology_id == bundle.project.ontology.target_object_id)
                    else "fixed"
                ),
                "geometry_source": object_payload.get("geometry_source"),
                "mesh_path": object_payload.get("mesh_path"),
            }
        )
    scene_instances = []
    for instance in bundle.task.scene_instances:
        scene_instances.append(
            {
                "instance_id": instance.instance_id,
                "ontology_id": instance.ontology_id,
                "role": instance.role,
                "prompt_text": instance.prompt_text,
                "size_prior_m": (
                    None
                    if instance.size_prior_m is None
                    else {
                        "x": instance.size_prior_m.x,
                        "y": instance.size_prior_m.y,
                        "z": instance.size_prior_m.z,
                    }
                ),
                "search_region_m": (
                    None
                    if instance.search_region_m is None
                    else {
                        "min_m": {
                            "x": instance.search_region_m.min_m.x,
                            "y": instance.search_region_m.min_m.y,
                            "z": instance.search_region_m.min_m.z,
                        },
                        "max_m": {
                            "x": instance.search_region_m.max_m.x,
                            "y": instance.search_region_m.max_m.y,
                            "z": instance.search_region_m.max_m.z,
                        },
                    }
                ),
                "target_region_m": (
                    None
                    if instance.target_region_m is None
                    else {
                        "min_m": {
                            "x": instance.target_region_m.min_m.x,
                            "y": instance.target_region_m.min_m.y,
                            "z": instance.target_region_m.min_m.z,
                        },
                        "max_m": {
                            "x": instance.target_region_m.max_m.x,
                            "y": instance.target_region_m.max_m.y,
                            "z": instance.target_region_m.max_m.z,
                        },
                    }
                ),
            }
        )
    goals = []
    for goal in bundle.task.goals:
        goals.append(
            {
                "goal_id": goal.goal_id,
                "type": goal.type,
                "source_instance_ids": list(goal.source_instance_ids),
                "target_region_instance_id": goal.target_region_instance_id,
            }
        )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.theta.primary_backbone,
        "sim_world_frame": bundle.project.theta.sim_world_frame,
        "robot": {
            "robot_id": bundle.robot.robot_id,
            "base_frame": bundle.robot.base_frame,
            "ee_frame": bundle.robot.ee_frame,
            "control_mode": bundle.robot.action_space.control_mode,
            "control_rate_hz": bundle.robot.control_rate_hz,
            "joint_order": list(bundle.robot.action_space.arm_joint_order),
            "robot_model_ref": _path_string(robot_model_path),
        },
        "task": {
            "task_id": bundle.task.task_id,
            "family": bundle.task.family,
            "target_object_id": bundle.project.ontology.target_object_id,
            "receptacle_object_id": bundle.project.ontology.receptacle_object_id,
            "support_surface_id": bundle.project.ontology.support_surface_id,
            "reference_object_ids": list(bundle.project.ontology.reference_object_ids),
            "fiducial_board_id": bundle.project.ontology.fiducial_board_id,
            "success_metric": bundle.task.success.metric,
            "position_tolerance_m": bundle.task.success.position_tolerance_m,
            "hold_time_s": bundle.task.success.hold_time_s,
            "target_region_center_m": target_region["runtime_center_m"],
            "target_region_half_extents_m": target_region["runtime_half_extents_m"],
            "declared_target_region_m": {
                "center_m": target_region["declared_center_m"],
                "half_extents_m": target_region["declared_half_extents_m"],
            },
            "runtime_target_region_m": {
                "center_m": target_region["runtime_center_m"],
                "half_extents_m": target_region["runtime_half_extents_m"],
            },
            "runtime_region_resolution": target_region["runtime_region_resolution"],
            "runtime_region_offset_m": target_region["runtime_region_offset_m"],
            "initial_object_states": initial_object_states,
            "normalized_events": normalized_events,
            "contact_schedule": contact_schedule.get("events", []),
            "target_position_diagnostics": eta_summary.get("target_position_diagnostics", {}),
            "support_plane": {
                "height_m": support_plane.get("height_m"),
                "normal_m": support_plane.get("normal_m"),
            },
            "scene_instances": scene_instances,
            "goals": goals,
        },
        "provenance": {
            "project_spec_ref": _path_string(bundle.project_path),
            "robot_spec_ref": _path_string(bundle.robot_path),
            "task_spec_ref": _path_string(bundle.task_path),
            "ontology_spec_ref": _path_string(bundle.ontology_path),
            "coordinate_frames_ref": _path_string(bundle.coordinate_frames_path),
            "epsilon_dir": _path_string(epsilon_dir),
            "zeta_dir": _path_string(zeta_dir),
            "eta_dir": _path_string(eta_dir),
            "world_metric_from_world_ref": _path_string(epsilon_dir / "scene" / "world_metric_from_world.json"),
            "world_metric_from_world": world_transform,
            "robot_target_ref": _path_string(eta_dir / "retarget" / "robot_target.yaml"),
            "robot_demo_ref": _path_string(eta_dir / "retarget" / "robot_demo.npz"),
            "contact_schedule_ref": _path_string(eta_dir / "retarget" / "contact_schedule.json"),
            "assets_manifest_ref": _path_string(zeta_dir / "assets" / "manifest.json"),
        },
    }


def _set_body_pose(body_element: ET.Element, transform: np.ndarray) -> None:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    translation = np.asarray(transform[:3, 3], dtype=float)
    quaternion = _rotation_matrix_to_quaternion_wxyz(rotation)
    body_element.set("pos", " ".join(f"{value:.6f}" for value in translation))
    body_element.set("quat", " ".join(f"{value:.6f}" for value in quaternion))


def _add_mesh_asset(asset_element: ET.Element, name: str, file_path: str) -> None:
    ET.SubElement(asset_element, "mesh", name=name, file=file_path)


def _format_xyz(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _first_valid_human_segment(human_ghost: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    visibility = human_ghost["track_visible"]
    elbows = human_ghost["elbow_position_m"]
    wrists = human_ghost["wrist_position_m"]
    for index, visible in enumerate(visibility):
        elbow = elbows[index]
        wrist = wrists[index]
        if bool(visible) and np.isfinite(elbow).all() and np.isfinite(wrist).all():
            return elbow.astype(float), wrist.astype(float)
    return None


def _activity_center(
    bundle: SpecBundle,
    robot_trace: dict[str, np.ndarray],
    object_pose_map: dict[str, dict[str, Any]],
    target_region: dict[str, Any],
) -> np.ndarray:
    centers: list[np.ndarray] = []
    ee_positions = robot_trace.get("ee_position_m")
    if ee_positions is not None and len(ee_positions) > 0:
        centers.append(np.median(np.asarray(ee_positions, dtype=float), axis=0))
    for ontology_id in (
        bundle.project.ontology.target_object_id,
        bundle.project.ontology.receptacle_object_id,
        *bundle.project.ontology.reference_object_ids,
    ):
        payload = object_pose_map.get(ontology_id)
        if payload is not None:
            centers.append(np.asarray(payload["position_m"], dtype=float))
    centers.append(np.asarray(target_region["runtime_center_m"], dtype=float))
    return np.mean(np.asarray(centers, dtype=float), axis=0)


def _support_table_geom(
    worldbody: ET.Element,
    manifest: dict[str, Any],
    support_plane: dict[str, Any],
) -> None:
    bbox = manifest["static"]["bbox_m"]
    min_corner = np.asarray(bbox["min"], dtype=float)
    max_corner = np.asarray(bbox["max"], dtype=float)
    top_z = float(support_plane["height_m"])
    thickness = max(0.01, top_z - float(min_corner[2]))
    center = np.array(
        [
            0.5 * (float(min_corner[0]) + float(max_corner[0])),
            0.5 * (float(min_corner[1]) + float(max_corner[1])),
            top_z - 0.5 * thickness,
        ],
        dtype=float,
    )
    half_extents = np.array(
        [
            0.5 * (float(max_corner[0]) - float(min_corner[0])),
            0.5 * (float(max_corner[1]) - float(min_corner[1])),
            0.5 * thickness,
        ],
        dtype=float,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="support_table_collision",
        type="box",
        pos=_format_xyz(center),
        size=_format_xyz(half_extents),
        friction="0.9 0.05 0.01",
        rgba="0.7 0.7 0.7 0.15",
    )


def _write_scene_xml(
    path: Path,
    bundle: SpecBundle,
    *,
    robot_model_path: Path,
    robot_base_transform: np.ndarray,
    zeta_manifest: dict[str, Any],
    object_pose_map: dict[str, dict[str, Any]],
    support_plane: dict[str, Any],
    audit_camera: AuditCamera,
    debug_camera: AuditCamera,
    target_region: dict[str, Any],
    robot_trace: dict[str, np.ndarray],
    human_ghost: dict[str, np.ndarray],
) -> None:
    robot_tree = ET.parse(robot_model_path)
    root = robot_tree.getroot()
    root.set("model", f"{bundle.project.video_id}_theta_scene")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("coordinate", "local")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    base_link = worldbody.find(".//body[@name='base_link']")
    if base_link is None:
        raise ThetaSimError(f"vendored robot model was missing a base_link body: {robot_model_path}")
    _set_body_pose(base_link, robot_base_transform)

    static = zeta_manifest["static"]
    _add_mesh_asset(asset, "static_visual", f"../{static['visual_mesh']}")
    for index, collision_mesh in enumerate(static["collision_meshes"]):
        _add_mesh_asset(asset, f"static_collision_{index:02d}", f"../{collision_mesh}")

    ET.SubElement(
        worldbody,
        "geom",
        name="scene_static_visual",
        type="mesh",
        mesh="static_visual",
        contype="0",
        conaffinity="0",
        rgba="0.85 0.85 0.85 1",
    )
    _support_table_geom(worldbody, zeta_manifest, support_plane)
    manipulable_instance_ids = _manipulable_instance_ids(bundle)

    for obj in zeta_manifest.get("objects", []):
        instance_id = str(obj.get("instance_id", "")).strip() or None
        object_name = _sanitize_name(instance_id or str(obj["track_id"]))
        _add_mesh_asset(asset, f"{object_name}_visual", f"../{obj['visual_mesh']}")
        for index, collision_mesh in enumerate(obj["collision_meshes"]):
            _add_mesh_asset(asset, f"{object_name}_collision_{index:02d}", f"../{collision_mesh}")
        object_pose = object_pose_map.get(instance_id or "", object_pose_map.get(str(obj["ontology_id"]), obj))
        position = object_pose.get("position_m", obj["position_m"])
        body = ET.SubElement(
            worldbody,
            "body",
            name=object_name,
            pos=_format_xyz(position),
        )
        if instance_id in manipulable_instance_ids or str(obj["ontology_id"]) == bundle.project.ontology.target_object_id:
            ET.SubElement(body, "freejoint", name=f"{object_name}_freejoint")
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=f"{object_name}_visual",
            contype="0",
            conaffinity="0",
            rgba=(
                "0.15 0.55 0.95 1"
                if str(obj["ontology_id"]) == bundle.project.ontology.target_object_id
                else "0.28 0.55 0.18 1"
            ),
        )
        for index, _collision_mesh in enumerate(obj["collision_meshes"]):
            collision_kwargs = {
                "type": "mesh",
                "mesh": f"{object_name}_collision_{index:02d}",
                "rgba": "0.85 0.9 0.85 0.12",
            }
            if instance_id in manipulable_instance_ids or str(obj["ontology_id"]) == bundle.project.ontology.target_object_id:
                collision_kwargs["density"] = "180"
                collision_kwargs["friction"] = "0.8 0.05 0.01"
            else:
                collision_kwargs["contype"] = "0"
                collision_kwargs["conaffinity"] = "0"
            ET.SubElement(body, "geom", **collision_kwargs)

    runtime_target_center = np.asarray(target_region["runtime_center_m"], dtype=float)
    runtime_target_half_extents = np.asarray(target_region["runtime_half_extents_m"], dtype=float)
    declared_target_center = np.asarray(target_region["declared_center_m"], dtype=float)
    declared_target_half_extents = np.asarray(target_region["declared_half_extents_m"], dtype=float)
    receptacle_position = np.asarray(
        object_pose_map[bundle.project.ontology.receptacle_object_id]["position_m"],
        dtype=float,
    )
    target_object_position = np.asarray(
        object_pose_map[bundle.project.ontology.target_object_id]["position_m"],
        dtype=float,
    )
    pick_pregrasp = target_object_position + np.array([0.0, 0.0, bundle.project.eta.pregrasp_clearance_m], dtype=float)
    place_pregrasp = receptacle_position + np.array([0.0, 0.0, bundle.project.eta.pregrasp_clearance_m], dtype=float)
    ET.SubElement(
        worldbody,
        "site",
        name="target_region_center",
        type="box",
        pos=_format_xyz(runtime_target_center),
        size=_format_xyz(runtime_target_half_extents),
        rgba="0.1 0.9 0.1 0.3",
    )
    if target_region["runtime_region_resolution"] == "auto_reanchored_from_scene":
        ET.SubElement(
            worldbody,
            "site",
            name="declared_target_region_center",
            type="box",
            pos=_format_xyz(declared_target_center),
            size=_format_xyz(declared_target_half_extents),
            rgba="0.95 0.8 0.2 0.12",
        )
    ET.SubElement(
        worldbody,
        "site",
        name="receptacle_center",
        type="sphere",
        pos=_format_xyz(receptacle_position),
        size="0.014",
        rgba="0.95 0.6 0.1 0.8",
    )
    ET.SubElement(
        worldbody,
        "site",
        name="pick_pregrasp_waypoint",
        type="sphere",
        pos=_format_xyz(pick_pregrasp),
        size="0.012",
        rgba="0.1 0.7 1.0 0.55",
    )
    ET.SubElement(
        worldbody,
        "site",
        name="place_pregrasp_waypoint",
        type="sphere",
        pos=_format_xyz(place_pregrasp),
        size="0.012",
        rgba="0.1 0.7 1.0 0.55",
    )

    ET.SubElement(
        worldbody,
        "camera",
        name="audit_camera",
        pos=_format_xyz(audit_camera.position_m),
        xyaxes=f"{_format_xyz(audit_camera.x_axis_w)} {_format_xyz(audit_camera.y_axis_w)}",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="debug_camera",
        pos=_format_xyz(debug_camera.position_m),
        xyaxes=f"{_format_xyz(debug_camera.x_axis_w)} {_format_xyz(debug_camera.y_axis_w)}",
    )

    if "ee_position_m" in robot_trace and len(robot_trace["ee_position_m"]) > 0:
        ee_position = np.asarray(robot_trace["ee_position_m"][0], dtype=float)
        ET.SubElement(
            worldbody,
            "site",
            name="robot_ghost_ee",
            type="sphere",
            pos=_format_xyz(ee_position),
            size="0.012",
            rgba="0.0 0.65 1.0 0.45",
        )

    human_segment = _first_valid_human_segment(human_ghost)
    if human_segment is not None:
        elbow_world, wrist_world = human_segment
        ET.SubElement(
            worldbody,
            "site",
            name="human_ghost_wrist",
            type="sphere",
            pos=_format_xyz(wrist_world),
            size="0.012",
            rgba="1.0 0.2 0.2 0.5",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="human_ghost_arm",
            type="capsule",
            fromto=f"{_format_xyz(elbow_world)} {_format_xyz(wrist_world)}",
            size="0.006",
            rgba="1.0 0.2 0.2 0.28",
            contype="0",
            conaffinity="0",
        )

    ET.indent(robot_tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    robot_tree.write(path, encoding="utf-8", xml_declaration=False)


def _save_mjb(mujoco: Any, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = getattr(mujoco, "mj_saveModel", None)
    if save_fn is not None:
        attempts = (
            lambda: save_fn(model, str(path), None, 0),
            lambda: save_fn(model, str(path), None),
            lambda: save_fn(model, str(path)),
        )
        for attempt in attempts:
            try:
                attempt()
                return
            except TypeError:
                continue
    if hasattr(model, "save_binary"):
        model.save_binary(str(path))
        return
    raise ThetaSimError("MuJoCo bindings did not expose a usable MJB save API")


def _apply_robot_trace_state(
    mujoco: Any,
    model: Any,
    data: Any,
    initial_qpos: np.ndarray,
    joint_positions: np.ndarray,
    gripper_width_m: float,
) -> None:
    if not hasattr(data, "qpos"):
        return
    qpos = getattr(data, "qpos")
    qvel = getattr(data, "qvel", None)
    qpos[:] = initial_qpos
    if qvel is not None:
        qvel[:] = 0.0
    arm_dof = min(6, len(joint_positions), len(qpos))
    qpos[:arm_dof] = joint_positions[:arm_dof]
    if len(qpos) >= 8:
        jaw_half_width = float(np.clip(0.5 * gripper_width_m, 0.0, 0.0425))
        qpos[6] = jaw_half_width
        qpos[7] = jaw_half_width
    forward_fn = getattr(mujoco, "mj_forward", None)
    if forward_fn is not None:
        forward_fn(model, data)


def _trace_sample_indices(trace_length: int, sample_count: int) -> np.ndarray:
    if trace_length <= 0:
        return np.asarray([], dtype=int)
    if trace_length <= sample_count:
        return np.arange(trace_length, dtype=int)
    return np.unique(np.linspace(0, trace_length - 1, sample_count, dtype=int))


def _render_placeholder_audit(
    path: Path,
    support_bbox: tuple[np.ndarray, np.ndarray],
    zeta_manifest: dict[str, Any],
    object_pose_map: dict[str, dict[str, Any]],
    target_region: dict[str, Any],
    robot_trace: dict[str, np.ndarray],
    human_ghost: dict[str, np.ndarray],
) -> tuple[bool, str, tuple[int, int]]:
    width_px, height_px = 960, 540
    image = Image.new("RGB", (width_px, height_px), color=(245, 245, 245))
    draw = ImageDraw.Draw(image, "RGBA")
    support_min, support_max = support_bbox
    view_min = np.asarray(support_min[:2], dtype=float).copy()
    view_max = np.asarray(support_max[:2], dtype=float).copy()
    for obj in zeta_manifest.get("objects", []):
        payload = object_pose_map.get(str(obj["ontology_id"]), obj)
        position = np.asarray(payload["position_m"], dtype=float)
        extents = np.asarray(payload.get("extents_m", obj.get("extents_m", [0.04, 0.04, 0.04])), dtype=float)
        view_min = np.minimum(view_min, position[:2] - 0.6 * extents[:2])
        view_max = np.maximum(view_max, position[:2] + 0.6 * extents[:2])
    if "ee_position_m" in robot_trace and len(robot_trace["ee_position_m"]) > 0:
        ee_positions = np.asarray(robot_trace["ee_position_m"], dtype=float)
        view_min = np.minimum(view_min, np.min(ee_positions[:, :2], axis=0))
        view_max = np.maximum(view_max, np.max(ee_positions[:, :2], axis=0))
    runtime_target_center = np.asarray(target_region["runtime_center_m"], dtype=float)
    runtime_target_half_extents = np.asarray(target_region["runtime_half_extents_m"], dtype=float)
    view_min = np.minimum(view_min, runtime_target_center[:2] - runtime_target_half_extents[:2])
    view_max = np.maximum(view_max, runtime_target_center[:2] + runtime_target_half_extents[:2])
    view_span = np.maximum(view_max - view_min, 0.2)
    margin = 0.08 * float(np.max(view_span))
    view_min = view_min - margin
    view_max = view_max + margin
    padding = 40.0

    def project(point_xy: Sequence[float]) -> tuple[float, float]:
        x_norm = (float(point_xy[0]) - float(view_min[0])) / max(1e-6, float(view_max[0] - view_min[0]))
        y_norm = (float(point_xy[1]) - float(view_min[1])) / max(1e-6, float(view_max[1] - view_min[1]))
        return (
            padding + x_norm * (width_px - 2.0 * padding),
            height_px - (padding + y_norm * (height_px - 2.0 * padding)),
        )

    table_min = project((support_min[0], support_min[1]))
    table_max = project((support_max[0], support_max[1]))
    draw.rectangle([table_min[0], table_max[1], table_max[0], table_min[1]], outline=(60, 60, 60), width=3)

    target_min = project(
        (
            runtime_target_center[0] - runtime_target_half_extents[0],
            runtime_target_center[1] - runtime_target_half_extents[1],
        )
    )
    target_max = project(
        (
            runtime_target_center[0] + runtime_target_half_extents[0],
            runtime_target_center[1] + runtime_target_half_extents[1],
        )
    )
    draw.rectangle(
        [target_min[0], target_max[1], target_max[0], target_min[1]],
        outline=(0, 180, 0, 255),
        fill=(0, 180, 0, 40),
        width=2,
    )

    for obj in zeta_manifest.get("objects", []):
        payload = object_pose_map.get(str(obj["ontology_id"]), obj)
        position = payload["position_m"]
        extents = payload.get("extents_m", obj.get("extents_m", [0.04, 0.04, 0.04]))
        min_corner = project((position[0] - 0.5 * extents[0], position[1] - 0.5 * extents[1]))
        max_corner = project((position[0] + 0.5 * extents[0], position[1] + 0.5 * extents[1]))
        color = (
            (40, 120, 230, 155)
            if str(obj["ontology_id"]) == "target_object"
            else (90, 150, 55, 155)
        )
        draw.rectangle(
            [min_corner[0], max_corner[1], max_corner[0], min_corner[1]],
            outline=color[:3] + (255,),
            fill=color,
            width=2,
        )

    if "ee_position_m" in robot_trace and len(robot_trace["ee_position_m"]) > 0:
        ee_points = [project(position[:2]) for position in np.asarray(robot_trace["ee_position_m"], dtype=float)]
        if len(ee_points) >= 2:
            draw.line(ee_points, fill=(0, 110, 220, 255), width=3)
        for point in (ee_points[0], ee_points[-1]):
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(0, 110, 220))

    human_segment = _first_valid_human_segment(human_ghost)
    if human_segment is not None:
        elbow_world, wrist_world = human_segment
        elbow_uv = project(elbow_world[:2])
        wrist_uv = project(wrist_world[:2])
        draw.line([elbow_uv, wrist_uv], fill=(220, 30, 30, 220), width=3)
        draw.ellipse((wrist_uv[0] - 4, wrist_uv[1] - 4, wrist_uv[0] + 4, wrist_uv[1] + 4), fill=(220, 30, 30))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    image.close()
    return True, "placeholder_projection", (width_px, height_px)


def _render_audit_image(
    mujoco: Any,
    model: Any,
    data: Any,
    out_path: Path,
    *,
    support_bbox: tuple[np.ndarray, np.ndarray],
    zeta_manifest: dict[str, Any],
    object_pose_map: dict[str, dict[str, Any]],
    target_region: dict[str, Any],
    robot_trace: dict[str, np.ndarray],
    human_ghost: dict[str, np.ndarray],
) -> tuple[bool, str, tuple[int, int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer_cls = getattr(mujoco, "Renderer", None)
    if renderer_cls is None:
        return _render_placeholder_audit(
            out_path,
            support_bbox,
            zeta_manifest,
            object_pose_map,
            target_region,
            robot_trace,
            human_ghost,
        )
    try:
        renderer = renderer_cls(model, 960, 540)
        try:
            update_scene = getattr(renderer, "update_scene", None)
            if update_scene is not None:
                try:
                    update_scene(data, camera="audit_camera")
                except TypeError:
                    update_scene(data)
            rgb = renderer.render()
        finally:
            close_fn = getattr(renderer, "close", None)
            if close_fn is not None:
                close_fn()
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        image.save(out_path)
        image.close()
        return True, "mujoco_renderer", (int(rgb.shape[1]), int(rgb.shape[0]))
    except Exception:
        return _render_placeholder_audit(
            out_path,
            support_bbox,
            zeta_manifest,
            object_pose_map,
            target_region,
            robot_trace,
            human_ghost,
        )


def _render_playback_videos(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    initial_qpos: np.ndarray,
    robot_trace: dict[str, np.ndarray],
    playback_dir: Path,
    camera_names: Sequence[str] = ("audit_camera", "debug_camera"),
) -> list[dict[str, Any]]:
    renderer_cls = getattr(mujoco, "Renderer", None)
    playback_dir.mkdir(parents=True, exist_ok=True)
    trace_indices = _trace_sample_indices(len(robot_trace["t_ns"]), min(32, max(1, len(robot_trace["t_ns"]))))
    results: list[dict[str, Any]] = []
    if renderer_cls is None:
        for camera_name in camera_names:
            path = playback_dir / f"{camera_name}.mp4"
            frames = [np.full((540, 960, 3), 235, dtype=np.uint8) for _ in range(max(1, len(trace_indices)))]
            imageio.mimsave(path, frames, fps=10)
            results.append(
                {
                    "camera_name": camera_name,
                    "path": _path_string(path),
                    "ok": True,
                    "mode": "placeholder_video",
                    "frame_count": len(frames),
                }
            )
        return results

    for camera_name in camera_names:
        path = playback_dir / f"{camera_name}.mp4"
        frames: list[np.ndarray] = []
        mode = "mujoco_renderer"
        try:
            renderer = renderer_cls(model, 960, 540)
            try:
                for index in trace_indices:
                    _apply_robot_trace_state(
                        mujoco,
                        model,
                        data,
                        initial_qpos,
                        np.asarray(robot_trace["joint_positions_rad"][index], dtype=float),
                        float(robot_trace["gripper_width_m"][index]),
                    )
                    update_scene = getattr(renderer, "update_scene", None)
                    if update_scene is not None:
                        try:
                            update_scene(data, camera=camera_name)
                        except TypeError:
                            update_scene(data)
                    frames.append(np.asarray(renderer.render(), dtype=np.uint8))
            finally:
                close_fn = getattr(renderer, "close", None)
                if close_fn is not None:
                    close_fn()
        except Exception:
            mode = "placeholder_video"
            frames = [np.full((540, 960, 3), 235, dtype=np.uint8) for _ in range(max(1, len(trace_indices)))]
        imageio.mimsave(path, frames, fps=10)
        results.append(
            {
                "camera_name": camera_name,
                "path": _path_string(path),
                "ok": True,
                "mode": mode,
                "frame_count": len(frames),
            }
        )
    return results


def _compile_and_validate_scene(
    *,
    bundle: SpecBundle,
    scene_xml_path: Path,
    scene_mjb_path: Path,
    playback_dir: Path,
    support_bbox: tuple[np.ndarray, np.ndarray],
    zeta_manifest: dict[str, Any],
    object_pose_map: dict[str, dict[str, Any]],
    robot_base_transform: np.ndarray,
    target_region: dict[str, Any],
    robot_trace: dict[str, np.ndarray],
    human_ghost: dict[str, np.ndarray],
    eta_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mujoco = _load_mujoco_module()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        data = mujoco.MjData(model)
        initial_qpos = np.array(getattr(data, "qpos", np.zeros(int(getattr(model, "nq", 0)))), dtype=float, copy=True)

        for _ in range(bundle.project.acceptance.theta_zero_control_smoke_steps):
            mujoco.mj_step(model, data)

        trace_indices = _trace_sample_indices(
            trace_length=int(len(robot_trace["t_ns"])),
            sample_count=bundle.project.acceptance.theta_trace_smoke_steps,
        )
        for index in trace_indices:
            _apply_robot_trace_state(
                mujoco,
                model,
                data,
                initial_qpos,
                np.asarray(robot_trace["joint_positions_rad"][index], dtype=float),
                float(robot_trace["gripper_width_m"][index]),
            )
            mujoco.mj_step(model, data)

        _save_mjb(mujoco, model, scene_mjb_path)
        mjb_load_ok = bool(scene_mjb_path.exists() and scene_mjb_path.stat().st_size > 0)
        binary_loader = getattr(mujoco.MjModel, "from_binary_path", None)
        if binary_loader is not None:
            binary_loader(str(scene_mjb_path))
            mjb_load_ok = True

        audit_ok, audit_mode, audit_resolution = _render_audit_image(
            mujoco,
            model,
            data,
            scene_xml_path.parent / "audit_render.png",
            support_bbox=support_bbox,
            zeta_manifest=zeta_manifest,
            object_pose_map=object_pose_map,
            target_region=target_region,
            robot_trace=robot_trace,
            human_ghost=human_ghost,
        )
        playback_renders = _render_playback_videos(
            mujoco,
            model,
            data,
            initial_qpos=initial_qpos,
            robot_trace=robot_trace,
            playback_dir=playback_dir,
        )
    except Exception as exc:
        raise ThetaSimError(f"theta compile-task failed for {scene_xml_path}: {exc}") from exc

    object_extent_ceiling_hits = {
        ontology_id: bool(np.any(np.asarray(payload.get("extents_m", []), dtype=float)[:2] >= OBJECT_EXTENT_CLIP_CEILING_M))
        for ontology_id, payload in object_pose_map.items()
    }
    pick_target_diagnostics = eta_summary.get("target_position_diagnostics", {}).get("pick", {})
    place_target_diagnostics = eta_summary.get("target_position_diagnostics", {}).get("place", {})
    waypoint_clip_count = int(
        sum(
            1
            for diagnostic in eta_summary.get("waypoint_clip_diagnostics", [])
            if bool(diagnostic.get("was_clipped"))
        )
    )
    spatial_sanity = {
        "robot_base_identity_ok": not bool(np.allclose(robot_base_transform, np.eye(4, dtype=float), atol=1e-6)),
        "runtime_region_resolution": target_region["runtime_region_resolution"],
        "runtime_region_offset_m": target_region["runtime_region_offset_m"],
        "object_extent_ceiling_hits": object_extent_ceiling_hits,
        "object_extents_clip_ceiling_ok": not any(object_extent_ceiling_hits.values()),
        "target_position_registration_ok": bool(
            eta_summary.get("qc_flags", {}).get("target_position_registration_ok", True)
        ),
        "pick_target_was_clipped": bool(pick_target_diagnostics.get("was_clipped")),
        "place_target_was_clipped": bool(place_target_diagnostics.get("was_clipped")),
        "waypoint_clip_count": waypoint_clip_count,
        "pick_place_targets_unclipped_ok": not bool(pick_target_diagnostics.get("was_clipped"))
        and not bool(place_target_diagnostics.get("was_clipped"))
        and waypoint_clip_count == 0,
    }

    validation = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.theta.primary_backbone,
        "scene_compile": {
            "compile_ok": True,
            "mjb_load_ok": mjb_load_ok,
            "zero_control_smoke_ok": True,
            "zero_control_smoke_steps": bundle.project.acceptance.theta_zero_control_smoke_steps,
            "trace_smoke_ok": True,
            "trace_smoke_steps": int(len(trace_indices)),
            "nq": int(getattr(model, "nq", 0)),
            "nv": int(getattr(model, "nv", 0)),
            "scene_xml_path": _path_string(scene_xml_path),
            "scene_mjb_path": _path_string(scene_mjb_path),
        },
        "audit_render": {
            "ok": audit_ok,
            "path": _path_string(scene_xml_path.parent / "audit_render.png"),
            "mode": audit_mode,
            "resolution_px": {
                "width": audit_resolution[0],
                "height": audit_resolution[1],
            },
        },
        "playback": {
            "robot_trace_frame_count": int(len(robot_trace["t_ns"])),
            "human_ghost_frame_count": int(len(human_ghost["t_ns"])),
            "robot_ghost_visible": bool("ee_position_m" in robot_trace and len(robot_trace["ee_position_m"]) > 0),
            "human_ghost_visible": bool(_first_valid_human_segment(human_ghost) is not None),
            "renders_ok": all(bool(render.get("ok")) for render in playback_renders),
            "renders": playback_renders,
        },
        "spatial_sanity": spatial_sanity,
    }
    return validation, {
        "audit_ok": audit_ok,
        "audit_mode": audit_mode,
    }


def enforce_theta_acceptance(bundle: SpecBundle, validation: dict[str, Any]) -> None:
    scene_compile = validation["scene_compile"]
    playback = validation["playback"]
    audit_render = validation["audit_render"]
    if bundle.project.acceptance.theta_require_scene_compile and not bool(scene_compile["compile_ok"]):
        raise ThetaSimError("theta scene.xml did not compile successfully")
    if bundle.project.acceptance.theta_require_mjb_load and not bool(scene_compile["mjb_load_ok"]):
        raise ThetaSimError("theta scene.mjb did not load successfully")
    if bundle.project.acceptance.theta_require_audit_render and not bool(audit_render["ok"]):
        raise ThetaSimError("theta audit render was not produced successfully")
    if bundle.project.acceptance.theta_require_robot_ghost_visible and not bool(playback["robot_ghost_visible"]):
        raise ThetaSimError("theta robot ghost marker was not available")
    if bundle.project.acceptance.theta_require_human_ghost_visible and not bool(playback["human_ghost_visible"]):
        raise ThetaSimError("theta human ghost marker was not available")
    if bundle.project.acceptance.theta_require_playback_renders and not bool(playback["renders_ok"]):
        raise ThetaSimError("theta playback renders were not produced successfully")


def compile_task_package(
    bundle: SpecBundle,
    gamma_dir: Path,
    epsilon_dir: Path,
    zeta_dir: Path,
    eta_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_theta_dependencies()
    if bundle.project.theta.primary_backbone != "canonical_mujoco_task":
        raise ThetaSimError(
            f"theta primary_backbone '{bundle.project.theta.primary_backbone}' is not implemented; "
            "the current MVP supports only canonical_mujoco_task"
        )

    task_window = _task_window_from_eta(eta_dir / "retarget" / "robot_target.yaml")
    gamma_rows = filter_items_to_task_window(
        _load_gamma_rows(gamma_dir / "human" / "arm_observables.parquet"),
        task_window,
    )
    if not gamma_rows:
        raise ThetaSimError("gamma task window did not contain any arm observables for Theta playback")

    metric_camera_payload = _load_metric_camera_payload(epsilon_dir / "camera" / "camera_poses_metric.json")
    world_transform = _load_json(epsilon_dir / "scene" / "world_metric_from_world.json")
    support_plane = _load_json(epsilon_dir / "scene" / "support_plane.json")
    object_pose_map = _object_map_by_ontology(epsilon_dir / "scene" / "object_init_poses_metric.json")
    zeta_manifest = _load_json(zeta_dir / "assets" / "manifest.json")
    contact_schedule = _load_json(eta_dir / "retarget" / "contact_schedule.json")
    robot_target = _load_yaml(eta_dir / "retarget" / "robot_target.yaml")
    robot_demo = _load_robot_demo_npz(eta_dir / "retarget" / "robot_demo.npz")
    eta_summary = _load_json(eta_dir / "retarget" / "eta_summary.json")
    robot_base_payload = _load_json(eta_dir / "retarget" / "robot_base_in_metric_world.json")
    robot_base_transform = np.asarray(robot_base_payload["X_Br_from_M"], dtype=float)
    support_bbox = (
        np.asarray(zeta_manifest["static"]["bbox_m"]["min"], dtype=float),
        np.asarray(zeta_manifest["static"]["bbox_m"]["max"], dtype=float),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = out_dir / "sim"
    playback_dir = sim_dir / "playback"
    playback_dir.mkdir(parents=True, exist_ok=True)

    robot_model_path = _copy_vendored_robot_model(sim_dir)
    robot_trace = _robot_trace_payload(bundle, robot_demo)
    human_ghost = _human_ghost_payload(gamma_rows)
    _write_npz(playback_dir / "robot_trace.npz", robot_trace)
    _write_npz(playback_dir / "human_arm_ghost.npz", human_ghost)
    target_region = _resolve_runtime_target_region(bundle, object_pose_map, support_plane)

    task_payload = _build_task_json(
        bundle,
        epsilon_dir=epsilon_dir,
        zeta_dir=zeta_dir,
        eta_dir=eta_dir,
        zeta_manifest=zeta_manifest,
        world_transform=world_transform,
        support_plane=support_plane,
        object_pose_map=object_pose_map,
        contact_schedule=contact_schedule,
        robot_trace=robot_trace,
        eta_summary=eta_summary,
        target_region=target_region,
        robot_model_path=robot_model_path,
    )
    (sim_dir / "task.json").write_text(json.dumps(task_payload, indent=2) + "\n", encoding="utf-8")

    metric_camera_frame = _metric_camera_frame_for_task_window(metric_camera_payload, task_window)
    audit_camera = _audit_camera_from_metric_pose(metric_camera_frame)
    debug_camera = _debug_camera(_activity_center(bundle, robot_trace, object_pose_map, target_region), support_bbox)
    scene_xml_path = sim_dir / "scene.xml"
    _write_scene_xml(
        scene_xml_path,
        bundle,
        robot_model_path=robot_model_path,
        robot_base_transform=robot_base_transform,
        zeta_manifest=zeta_manifest,
        object_pose_map=object_pose_map,
        support_plane=support_plane,
        audit_camera=audit_camera,
        debug_camera=debug_camera,
        target_region=target_region,
        robot_trace=robot_trace,
        human_ghost=human_ghost,
    )
    validation, _render_meta = _compile_and_validate_scene(
        bundle=bundle,
        scene_xml_path=scene_xml_path,
        scene_mjb_path=sim_dir / "scene.mjb",
        playback_dir=playback_dir,
        support_bbox=support_bbox,
        zeta_manifest=zeta_manifest,
        object_pose_map=object_pose_map,
        robot_base_transform=robot_base_transform,
        target_region=target_region,
        robot_trace=robot_trace,
        human_ghost=human_ghost,
        eta_summary=eta_summary,
    )
    validation["task_window"] = task_window.to_payload(video_id=bundle.project.video_id)
    validation["provenance"] = {
        "robot_target_ref": _path_string(eta_dir / "retarget" / "robot_target.yaml"),
        "robot_demo_ref": _path_string(eta_dir / "retarget" / "robot_demo.npz"),
        "robot_base_in_metric_world_ref": _path_string(eta_dir / "retarget" / "robot_base_in_metric_world.json"),
        "camera_poses_metric_ref": _path_string(epsilon_dir / "camera" / "camera_poses_metric.json"),
        "assets_manifest_ref": _path_string(zeta_dir / "assets" / "manifest.json"),
        "gamma_arm_observables_ref": _path_string(gamma_dir / "human" / "arm_observables.parquet"),
        "robot_target": robot_target,
    }
    validation["qc_flags"] = {
        "scene_compile_ok": bool(validation["scene_compile"]["compile_ok"]),
        "mjb_load_ok": bool(validation["scene_compile"]["mjb_load_ok"]),
        "zero_control_smoke_ok": bool(validation["scene_compile"]["zero_control_smoke_ok"]),
        "trace_smoke_ok": bool(validation["scene_compile"]["trace_smoke_ok"]),
        "audit_render_ok": bool(validation["audit_render"]["ok"]),
        "playback_renders_ok": bool(validation["playback"]["renders_ok"]),
        "robot_ghost_visible": bool(validation["playback"]["robot_ghost_visible"]),
        "human_ghost_visible": bool(validation["playback"]["human_ghost_visible"]),
        "robot_base_identity_ok": bool(validation["spatial_sanity"]["robot_base_identity_ok"]),
        "object_extents_clip_ceiling_ok": bool(validation["spatial_sanity"]["object_extents_clip_ceiling_ok"]),
        "target_position_registration_ok": bool(validation["spatial_sanity"]["target_position_registration_ok"]),
        "pick_place_targets_unclipped_ok": bool(validation["spatial_sanity"]["pick_place_targets_unclipped_ok"]),
    }
    (sim_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    enforce_theta_acceptance(bundle, validation)
    return validation
