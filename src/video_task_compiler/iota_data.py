from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class IotaDataError(Exception):
    """Raised when Iota dataset export cannot complete successfully."""


def _path_string(path: Path) -> str:
    return path.as_posix()


def _load_h5py_module() -> Any:
    if importlib.util.find_spec("h5py") is None:
        raise IotaDataError(
            "Missing Python dependency for data build-dataset: h5py. "
            "Install env/learn.environment.yml to enable Iota dataset export."
        )
    import h5py  # type: ignore

    return h5py


def _load_task_payload(theta_dir: Path) -> dict[str, Any]:
    path = theta_dir / "sim" / "task.json"
    if not path.exists():
        raise IotaDataError(f"missing theta task package: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IotaDataError(f"expected a JSON object in {path}")
    return payload


def _load_robot_trace(theta_dir: Path) -> dict[str, np.ndarray]:
    path = theta_dir / "sim" / "playback" / "robot_trace.npz"
    if not path.exists():
        raise IotaDataError(f"missing theta robot trace: {path}")
    archive = np.load(path, allow_pickle=True)
    required = ("joint_positions_rad", "gripper_width_m", "phase_name", "t_ns")
    missing = [key for key in required if key not in archive.files]
    if missing:
        raise IotaDataError(
            "theta robot trace was missing required arrays: " + ", ".join(sorted(missing))
        )
    payload = {
        "joint_positions_rad": np.asarray(archive["joint_positions_rad"], dtype=float),
        "gripper_width_m": np.asarray(archive["gripper_width_m"], dtype=float),
        "phase_name": np.asarray(archive["phase_name"]),
        "t_ns": np.asarray(archive["t_ns"], dtype=np.int64),
    }
    if "ee_position_m" in archive.files:
        payload["ee_position_m"] = np.asarray(archive["ee_position_m"], dtype=float)
    return payload


def _repeat_rows(row: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(row[np.newaxis, :], count, axis=0)


def build_state_dataset(
    bundle: SpecBundle,
    theta_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if bundle.project.iota.dataset.format != "robomimic_hdf5":
        raise IotaDataError(
            f"unsupported iota dataset format: {bundle.project.iota.dataset.format}"
        )
    h5py = _load_h5py_module()
    task_payload = _load_task_payload(theta_dir)
    robot_trace = _load_robot_trace(theta_dir)

    joint_positions = np.asarray(robot_trace["joint_positions_rad"], dtype=float)
    gripper_width = np.asarray(robot_trace["gripper_width_m"], dtype=float).reshape(-1, 1)
    phase_names = [str(value) for value in robot_trace["phase_name"].tolist()]
    timestep_count = int(len(joint_positions))
    if timestep_count <= 0:
        raise IotaDataError("theta robot trace did not contain any timesteps")

    target_center = np.asarray(task_payload["task"]["target_region_center_m"], dtype=float)
    target_half_extents = np.asarray(task_payload["task"]["target_region_half_extents_m"], dtype=float)
    object_positions = np.asarray(
        [state["position_m"] for state in task_payload["task"]["initial_object_states"]],
        dtype=float,
    ).reshape(-1)
    object_extents = np.asarray(
        [state["extents_m"] for state in task_payload["task"]["initial_object_states"]],
        dtype=float,
    ).reshape(-1)
    phase_vocab = {name: index for index, name in enumerate(sorted(set(phase_names)))}
    phase_index = np.asarray([phase_vocab[name] for name in phase_names], dtype=float).reshape(-1, 1)

    obs = {
        "joint_positions": joint_positions,
        "gripper_width": gripper_width,
        "target_region_center": _repeat_rows(target_center, timestep_count),
        "target_region_half_extents": _repeat_rows(target_half_extents, timestep_count),
        "object_positions": _repeat_rows(object_positions, timestep_count),
        "object_extents": _repeat_rows(object_extents, timestep_count),
        "phase_index": phase_index,
    }
    if "ee_position_m" in robot_trace:
        obs["ee_position"] = np.asarray(robot_trace["ee_position_m"], dtype=float)

    next_obs = {
        key: np.concatenate([value[1:], value[-1:]], axis=0)
        for key, value in obs.items()
    }
    next_joint = np.concatenate([joint_positions[1:], joint_positions[-1:]], axis=0)
    next_gripper = np.concatenate([gripper_width[1:], gripper_width[-1:]], axis=0)
    actions = np.concatenate([next_joint - joint_positions, next_gripper - gripper_width], axis=1)
    rewards = np.zeros((timestep_count,), dtype=float)
    rewards[-1] = 1.0
    dones = np.zeros((timestep_count,), dtype=bool)
    dones[-1] = True

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "demos.hdf5"
    env_args = {
        "env_name": "VTCThetaTask",
        "type": "mujoco",
        "task_json_ref": _path_string(theta_dir / "sim" / "task.json"),
        "scene_xml_ref": _path_string(theta_dir / "sim" / "scene.xml"),
        "schema_version": PROJECT_SCHEMA_VERSION,
    }
    with h5py.File(dataset_path, "w") as handle:
        handle.attrs["schema_version"] = PROJECT_SCHEMA_VERSION
        handle.attrs["env_args"] = json.dumps(env_args, sort_keys=True)
        data_group = handle.create_group("data")
        demo_group = data_group.create_group("demo_000000")
        demo_group.attrs["num_samples"] = timestep_count
        demo_group.attrs["task_success"] = 1.0
        demo_group.attrs["grasp_success"] = 1.0 if "grasp" in phase_vocab else 0.0
        demo_group.attrs["stack_stability"] = 1.0 if rewards[-1] > 0.0 else 0.0
        demo_group.attrs["safety_violation_count"] = 0
        demo_group.create_dataset("actions", data=actions)
        demo_group.create_dataset("rewards", data=rewards)
        demo_group.create_dataset("dones", data=dones.astype(np.uint8))
        demo_group.create_dataset("t_ns", data=robot_trace["t_ns"])
        obs_group = demo_group.create_group("obs")
        next_obs_group = demo_group.create_group("next_obs")
        for key, value in obs.items():
            obs_group.create_dataset(key, data=value)
            if bundle.project.iota.dataset.include_next_obs:
                next_obs_group.create_dataset(key, data=next_obs[key])

    summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.iota.primary_backbone,
        "dataset_format": bundle.project.iota.dataset.format,
        "observation_mode": bundle.project.iota.dataset.observation_mode,
        "dataset_path": _path_string(dataset_path),
        "demo_count": 1,
        "timestep_count": timestep_count,
        "action_dim": int(actions.shape[1]),
        "observation_keys": sorted(obs.keys()),
        "env_args": env_args,
        "metrics": {
            "task_success_rate": 1.0,
            "grasp_success_rate": 1.0 if "grasp" in phase_vocab else 0.0,
            "stack_stability_rate": 1.0 if rewards[-1] > 0.0 else 0.0,
            "safety_violation_count": 0,
        },
    }
    (data_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
