from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

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
) -> tuple[np.ndarray, np.ndarray, float]:
    support_height = float(
        min(
            bundle.task.pick_object.search_region_m.min_m.z,
            bundle.task.place_region.target_region_m.min_m.z,
        )
    )
    target_pose = object_pose_map.get(bundle.project.ontology.target_object_id)
    receptacle_pose = object_pose_map.get(bundle.project.ontology.receptacle_object_id)

    target_position = (
        np.asarray(target_pose["position_m"], dtype=float)
        if target_pose is not None
        else _region_center(bundle.task.pick_object.search_region_m)
    )
    receptacle_position = (
        np.asarray(receptacle_pose["position_m"], dtype=float)
        if receptacle_pose is not None
        else _region_center(bundle.task.place_region.target_region_m)
    )

    pick_event = next(event for event in contact_schedule["events"] if event["name"] == "pick")
    place_event = next(event for event in contact_schedule["events"] if event["name"] == "place")
    pick_wrist = _world_wrist(_gamma_row_by_frame(gamma_rows, int(pick_event["frame_idx"])))
    place_wrist = _world_wrist(_gamma_row_by_frame(gamma_rows, int(place_event["frame_idx"])))

    if pick_wrist is not None:
        target_position[:2] = pick_wrist[:2]
    if place_wrist is not None:
        receptacle_position[:2] = place_wrist[:2]

    target_height = float(target_pose["extents_m"][2]) if target_pose is not None else bundle.project.epsilon.default_object_height_m
    target_position[2] = support_height + 0.5 * target_height
    receptacle_position[2] = max(receptacle_position[2], support_height + 0.5 * target_height)
    return (
        _workspace_clip(bundle, target_position),
        _workspace_clip(bundle, receptacle_position),
        support_height,
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
) -> list[Waypoint]:
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
    return [
        Waypoint("home", _workspace_clip(bundle, home_position), open_width),
        Waypoint("pregrasp", _workspace_clip(bundle, pick_pre), open_width),
        Waypoint("grasp", _workspace_clip(bundle, pick_grasp), closed_width),
        Waypoint("lift", _workspace_clip(bundle, pick_lift), closed_width),
        Waypoint("transfer", _workspace_clip(bundle, place_pre), closed_width),
        Waypoint("preplace", _workspace_clip(bundle, place_pre), closed_width),
        Waypoint("place", _workspace_clip(bundle, place_release), open_width),
        Waypoint("retreat", _workspace_clip(bundle, retreat), open_width),
    ]


def _interpolate_demo(
    bundle: SpecBundle,
    waypoints: Sequence[Waypoint],
) -> dict[str, np.ndarray]:
    control_rate_hz = int(bundle.robot.control_rate_hz)
    max_joint_step_target = min(bundle.project.acceptance.eta_max_joint_step_rad, 0.9 * bundle.project.acceptance.eta_max_joint_step_rad)
    joint_waypoints = np.asarray(
        [_approximate_ur5e_joint_waypoint(bundle, waypoint.position_m) for waypoint in waypoints],
        dtype=float,
    )
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


def _write_scene_xml(
    path: Path,
    bundle: SpecBundle,
    zeta_manifest: dict[str, Any],
) -> None:
    lines = [
        f'<mujoco model="{bundle.project.video_id}_compiled_scene">',
        '  <compiler angle="radian" coordinate="local"/>',
        '  <option gravity="0 0 -9.81"/>',
        f'  <!-- robot model_ref: {bundle.robot.model_ref} -->',
        "  <asset>",
        '    <mesh name="static_visual" file="../assets/static_visual.obj"/>',
        '    <mesh name="static_collision_00" file="../assets/static_collision_00.obj"/>',
    ]
    for obj in zeta_manifest.get("objects", []):
        asset_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(obj["track_id"])).strip("_")
        lines.append(
            f'    <mesh name="{asset_name}_visual" file="../assets/{Path(obj["visual_mesh"]).name}"/>'
        )
        lines.append(
            f'    <mesh name="{asset_name}_collision_00" file="../assets/{Path(obj["collision_meshes"][0]).name}"/>'
        )
    lines.extend(
        [
            "  </asset>",
            "  <worldbody>",
            '    <geom name="table_surface" type="mesh" mesh="static_collision_00" rgba="0.85 0.85 0.85 1"/>',
        ]
    )
    for obj in zeta_manifest.get("objects", []):
        asset_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(obj["track_id"])).strip("_")
        position = obj["position_m"]
        lines.extend(
            [
                f'    <body name="{asset_name}" pos="{position[0]} {position[1]} {position[2]}">',
                f'      <geom type="mesh" mesh="{asset_name}_visual" contype="0" conaffinity="0" rgba="0.7 0.3 0.2 1"/>',
                f'      <geom type="mesh" mesh="{asset_name}_collision_00" group="3" rgba="0.2 0.8 0.2 0.4"/>',
                "    </body>",
            ]
        )
    lines.extend(["  </worldbody>", "</mujoco>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_container(
    path: Path,
    bundle: SpecBundle,
    demo_payload: dict[str, np.ndarray],
) -> str:
    if importlib.util.find_spec("h5py") is not None:
        import h5py  # type: ignore

        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = PROJECT_SCHEMA_VERSION
            handle.attrs["video_id"] = bundle.project.video_id
            for key, value in demo_payload.items():
                if key == "phase_name":
                    encoded = np.asarray([str(item).encode("utf-8") for item in value], dtype="S32")
                    handle.create_dataset(key, data=encoded)
                else:
                    handle.create_dataset(key, data=value)
        return "hdf5"

    with path.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray([PROJECT_SCHEMA_VERSION]),
            video_id=np.asarray([bundle.project.video_id]),
            **demo_payload,
        )
    return "npz_fallback_named_hdf5"


def _write_ros2_handoff(path: Path, bundle: SpecBundle, demo_path: Path, scene_xml_path: Path) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "robot_id": bundle.robot.robot_id,
        "robot_model_ref": bundle.robot.model_ref,
        "base_frame": bundle.robot.base_frame,
        "ee_frame": bundle.robot.ee_frame,
        "joint_order": list(bundle.robot.action_space.arm_joint_order),
        "demo_ref": _path_string(demo_path),
        "scene_xml_ref": _path_string(scene_xml_path),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_summary(
    bundle: SpecBundle,
    contact_schedule: dict[str, Any],
    task_window: TaskWindow,
    task_window_gamma_rows: Sequence[dict[str, Any]],
    demo_payload: dict[str, np.ndarray],
    container_format: str,
) -> dict[str, Any]:
    joint_positions = demo_payload["joint_positions_rad"]
    if len(joint_positions) <= 1:
        max_joint_step = 0.0
    else:
        max_joint_step = float(np.max(np.abs(np.diff(joint_positions, axis=0))))
    event_sources = {event["name"]: event["detection_source"] for event in contact_schedule["events"]}
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.eta.primary_backbone,
        "task_window": task_window.to_payload(video_id=bundle.project.video_id),
        "demo_frame_count": int(len(task_window_gamma_rows)),
        "retarget_frame_count": int(len(demo_payload["t_ns"])),
        "max_joint_step_rad": max_joint_step,
        "container_format": container_format,
        "contact_event_sources": event_sources,
        "qc_flags": {
            "demo_frame_count_ok": len(task_window_gamma_rows) >= bundle.project.acceptance.eta_min_demo_frames,
            "joint_step_ok": max_joint_step <= bundle.project.acceptance.eta_max_joint_step_rad,
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
    zeta_manifest = _load_json(zeta_dir / "assets" / "manifest.json")

    contact_schedule = _build_contact_schedule(bundle, gamma_rows, interaction_rows)
    pick_position, place_position, support_height = _pick_and_place_positions(
        bundle,
        object_pose_map,
        gamma_rows,
        contact_schedule,
    )
    waypoints = _build_waypoints(bundle, pick_position, place_position, support_height)
    demo_payload = _interpolate_demo(bundle, waypoints)

    retarget_dir = out_dir / "retarget"
    sim_dir = out_dir / "sim"
    data_dir = out_dir / "data"
    deployment_dir = out_dir / "deployment"
    for path in (retarget_dir, sim_dir, data_dir, deployment_dir):
        path.mkdir(parents=True, exist_ok=True)
    persist_task_window(task_window, out_dir, video_id=bundle.project.video_id)

    world_to_robot_base = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_world": bundle.project.epsilon.metric_world_frame,
        "target_frame": bundle.robot.base_frame,
        "X_Br_from_M": np.eye(4, dtype=float).tolist(),
        "registration_method": "identity_task_region_assumption",
    }
    (retarget_dir / "world_to_robot_base.json").write_text(
        json.dumps(world_to_robot_base, indent=2) + "\n",
        encoding="utf-8",
    )
    (retarget_dir / "contact_schedule.json").write_text(
        json.dumps(contact_schedule, indent=2) + "\n",
        encoding="utf-8",
    )

    demo_npz_path = retarget_dir / "robot_demo.npz"
    with demo_npz_path.open("wb") as handle:
        np.savez(handle, **demo_payload)

    scene_xml_path = sim_dir / "scene.xml"
    _write_scene_xml(scene_xml_path, bundle, zeta_manifest)
    demo_container_path = data_dir / "demos.hdf5"
    container_format = _write_demo_container(demo_container_path, bundle, demo_payload)
    _write_ros2_handoff(deployment_dir / "ros2_handoff.json", bundle, demo_npz_path, scene_xml_path)

    summary = _build_summary(bundle, contact_schedule, task_window, gamma_rows, demo_payload, container_format)
    (retarget_dir / "eta_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    enforce_eta_acceptance(bundle, summary)
    return summary
