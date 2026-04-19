from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class KappaDeployError(Exception):
    """Raised when Kappa deployment export cannot complete successfully."""


def _path_string(path: Path) -> str:
    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise KappaDeployError(f"missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KappaDeployError(f"expected a JSON object in {path}")
    return payload


def export_ros2_workspace(
    bundle: SpecBundle,
    theta_dir: Path,
    out_dir: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    task_payload = _load_json(theta_dir / "sim" / "task.json")
    validation_payload = _load_json(theta_dir / "sim" / "validation.json")
    ros2_dir = out_dir / "ros2"
    config_dir = ros2_dir / "config"
    launch_dir = ros2_dir / "launch"
    for path in (config_dir, launch_dir):
        path.mkdir(parents=True, exist_ok=True)

    controllers = {
        "controller_manager": {
            "ros__parameters": {
                "update_rate": 100,
                "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"},
                "arm_controller": {"type": "position_controllers/JointGroupPositionController"},
                "gripper_controller": {"type": "position_controllers/GripperActionController"},
            }
        }
    }
    safety = {
        "workspace_bounds_m": {
            "min": {
                "x": bundle.robot.workspace_bounds_m.min_m.x,
                "y": bundle.robot.workspace_bounds_m.min_m.y,
                "z": bundle.robot.workspace_bounds_m.min_m.z,
            },
            "max": {
                "x": bundle.robot.workspace_bounds_m.max_m.x,
                "y": bundle.robot.workspace_bounds_m.max_m.y,
                "z": bundle.robot.workspace_bounds_m.max_m.z,
            },
        },
        "control_rate_hz": bundle.robot.control_rate_hz,
        "action_abstraction": bundle.project.kappa.action_abstraction,
        "safety_supervisor": bundle.project.kappa.safety_supervisor,
    }
    parity = {
        "theta_task_json_ref": _path_string(theta_dir / "sim" / "task.json"),
        "theta_scene_xml_ref": _path_string(theta_dir / "sim" / "scene.xml"),
        "theta_validation_ref": _path_string(theta_dir / "sim" / "validation.json"),
        "mujoco_parity_backend": bundle.project.kappa.mujoco_parity_backend,
        "measured_robot_base_required": bool(validation_payload["spatial_sanity"]["robot_base_measured_ok"]),
    }
    task_package = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "task": task_payload["task"],
        "robot": task_payload["robot"],
        "policy_path": None if policy_path is None else _path_string(policy_path),
        "control_plane": bundle.project.kappa.control_plane,
    }
    (config_dir / "controllers.yaml").write_text(
        yaml.safe_dump(controllers, sort_keys=False),
        encoding="utf-8",
    )
    (config_dir / "safety_supervisor.yaml").write_text(
        yaml.safe_dump(safety, sort_keys=False),
        encoding="utf-8",
    )
    (config_dir / "theta_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    (config_dir / "task_package.json").write_text(json.dumps(task_package, indent=2) + "\n", encoding="utf-8")
    (launch_dir / "theta_parity.launch.py").write_text(
        "\n".join(
            [
                "from launch import LaunchDescription",
                "",
                "",
                "def generate_launch_description():",
                "    return LaunchDescription([])",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.kappa.primary_backbone,
        "control_plane": bundle.project.kappa.control_plane,
        "workspace_root": _path_string(ros2_dir),
        "controllers_ref": _path_string(config_dir / "controllers.yaml"),
        "safety_ref": _path_string(config_dir / "safety_supervisor.yaml"),
        "theta_parity_ref": _path_string(config_dir / "theta_parity.json"),
        "task_package_ref": _path_string(config_dir / "task_package.json"),
        "policy_path": None if policy_path is None else _path_string(policy_path),
        "qc_flags": {
            "robot_base_measured_ok": bool(validation_payload["spatial_sanity"]["robot_base_measured_ok"]),
            "theta_renderer_backed_ok": bool(validation_payload["playback"]["renderer_backed_renders_ok"]),
            "scene_compile_ok": bool(validation_payload["scene_compile"]["compile_ok"]),
        },
    }
    (ros2_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
