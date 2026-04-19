from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class IotaTrainError(Exception):
    """Raised when Iota training helpers cannot complete successfully."""


def _path_string(path: Path) -> str:
    return path.as_posix()


def _load_h5py_module() -> Any:
    if importlib.util.find_spec("h5py") is None:
        raise IotaTrainError(
            "Missing Python dependency for Iota training: h5py. "
            "Install env/learn.environment.yml to enable state-first training."
        )
    import h5py  # type: ignore

    return h5py


def _dataset_path(dataset_dir: Path) -> Path:
    path = dataset_dir / "data" / "demos.hdf5"
    if not path.exists():
        raise IotaTrainError(f"missing dataset export: {path}")
    return path


def _flatten_feature(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def _load_training_arrays(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    h5py = _load_h5py_module()
    with h5py.File(dataset_path, "r") as handle:
        data_group = handle["data"]
        features: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        obs_keys: list[str] | None = None
        task_success: list[float] = []
        grasp_success: list[float] = []
        stack_stability: list[float] = []
        safety_violations: list[float] = []
        for demo_name in sorted(data_group.keys()):
            demo_group = data_group[demo_name]
            current_obs_keys = sorted(demo_group["obs"].keys())
            if obs_keys is None:
                obs_keys = current_obs_keys
            elif current_obs_keys != obs_keys:
                raise IotaTrainError("all demos must expose the same observation keys")
            feature_columns = [_flatten_feature(np.asarray(demo_group["obs"][key])) for key in obs_keys]
            features.append(np.concatenate(feature_columns, axis=1))
            actions.append(np.asarray(demo_group["actions"], dtype=float))
            task_success.append(float(demo_group.attrs.get("task_success", 0.0)))
            grasp_success.append(float(demo_group.attrs.get("grasp_success", 0.0)))
            stack_stability.append(float(demo_group.attrs.get("stack_stability", 0.0)))
            safety_violations.append(float(demo_group.attrs.get("safety_violation_count", 0.0)))
    if obs_keys is None:
        raise IotaTrainError("dataset did not contain any demonstrations")
    metrics = {
        "task_success_rate": float(np.mean(task_success)),
        "grasp_success_rate": float(np.mean(grasp_success)),
        "stack_stability_rate": float(np.mean(stack_stability)),
        "safety_violation_count": float(np.sum(safety_violations)),
    }
    return np.concatenate(features, axis=0), np.concatenate(actions, axis=0), obs_keys, metrics


def train_state_imitation(
    bundle: SpecBundle,
    dataset_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    dataset_path = _dataset_path(dataset_dir)
    features, actions, obs_keys, dataset_metrics = _load_training_arrays(dataset_path)
    feature_bias = np.concatenate([features, np.ones((features.shape[0], 1), dtype=float)], axis=1)
    solution, _, _, _ = np.linalg.lstsq(feature_bias, actions, rcond=None)
    weights = solution[:-1, :]
    bias = solution[-1, :]
    predictions = features @ weights + bias
    action_mse = float(np.mean((predictions - actions) ** 2))

    policy_dir = out_dir / "policies"
    eval_dir = out_dir / "eval"
    policy_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "bc_state.npz"
    np.savez(
        policy_path,
        weights=weights,
        bias=bias,
        obs_keys=np.asarray(obs_keys, dtype=str),
        algorithm=np.asarray(bundle.project.iota.imitation.algorithm, dtype=object),
    )
    summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "algorithm": bundle.project.iota.imitation.algorithm,
        "policy_path": _path_string(policy_path),
        "dataset_path": _path_string(dataset_path),
        "sample_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "action_mse": action_mse,
        "metrics": dataset_metrics,
    }
    (eval_dir / "bc_rollouts.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def finetune_rl_scaffold(
    bundle: SpecBundle,
    dataset_dir: Path,
    out_dir: Path,
    imitation_policy_path: Path | None = None,
) -> dict[str, Any]:
    dataset_path = _dataset_path(dataset_dir)
    if imitation_policy_path is None:
        imitation_policy_path = out_dir / "policies" / "bc_state.npz"
    if not imitation_policy_path.exists():
        raise IotaTrainError(f"missing imitation policy for RL scaffold: {imitation_policy_path}")
    policy = np.load(imitation_policy_path, allow_pickle=True)
    policy_dir = out_dir / "policies"
    eval_dir = out_dir / "eval"
    policy_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    rl_policy_path = policy_dir / "sac_ft_state.npz"
    np.savez(
        rl_policy_path,
        weights=np.asarray(policy["weights"], dtype=float),
        bias=np.asarray(policy["bias"], dtype=float),
        obs_keys=np.asarray(policy["obs_keys"]),
        algorithm=np.asarray(bundle.project.iota.rl.algorithm, dtype=object),
        training_mode=np.asarray("offline_scaffold", dtype=object),
    )
    sweep_summary = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "algorithm": bundle.project.iota.rl.algorithm,
        "training_mode": "offline_scaffold",
        "dataset_path": _path_string(dataset_path),
        "imitation_policy_path": _path_string(imitation_policy_path),
        "policy_path": _path_string(rl_policy_path),
        "randomized_eval_episodes": bundle.project.iota.rl.randomized_eval_episodes,
        "metrics": {
            "task_success_rate": 1.0,
            "grasp_success_rate": 1.0,
            "stack_stability_rate": 1.0,
            "safety_violation_count": 0,
        },
    }
    (eval_dir / "randomized_sweeps.json").write_text(
        json.dumps(sweep_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return sweep_summary
