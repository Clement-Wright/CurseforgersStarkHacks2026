from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("pycocotools")

from video_task_compiler.cli import app
from video_task_compiler.epsilon_scene import compile_metric_scene
from video_task_compiler.eta_retarget import retarget_monocular_demonstration
from video_task_compiler.object_extract import encode_coco_rle_mask
from video_task_compiler.specs import load_bundle, validate_bundle
from video_task_compiler.theta_sim import compile_task_package
from video_task_compiler.zeta_assets import assetize_metric_scene
import video_task_compiler.eta_retarget as eta_retarget_module
import video_task_compiler.theta_sim as theta_sim_module
import video_task_compiler.zeta_assets as zeta_assets_module


runner = CliRunner()


class _FakeMjModel:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.nq = 15
        self.nv = 15

    @classmethod
    def from_xml_path(cls, path: str) -> "_FakeMjModel":
        assert Path(path).exists()
        return cls(path)


class _FakeMjData:
    def __init__(self, model: _FakeMjModel):
        self.model = model
        self.qpos = np.zeros(model.nq, dtype=float)
        self.qvel = np.zeros(model.nv, dtype=float)


class _FakeMujocoModule:
    MjModel = _FakeMjModel
    MjData = _FakeMjData

    @staticmethod
    def mj_step(model: _FakeMjModel, data: _FakeMjData) -> None:
        assert data.model is model

    @staticmethod
    def mj_saveModel(model: _FakeMjModel, path: str, *args) -> None:
        Path(path).write_bytes(b"fake-mjb")


def _install_fake_sim_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zeta_assets_module, "_load_mujoco_module", lambda: _FakeMujocoModule())
    monkeypatch.setattr(eta_retarget_module, "_load_pinocchio_module", lambda: SimpleNamespace(__name__="pinocchio"))
    monkeypatch.setattr(theta_sim_module, "_load_mujoco_module", lambda: _FakeMujocoModule())


def _copy_bundle_root(target_root: Path) -> Path:
    for name in ("spec", "env", "checklists"):
        shutil.copytree(Path(name), target_root / name)
    return target_root


def _square_mask(x_min: int, y_min: int, x_max: int, y_max: int, *, size: int = 64) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def _write_beta_artifacts(beta_dir: Path) -> None:
    frames_dir = beta_dir / "frames"
    camera_dir = beta_dir / "camera"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    with (frames_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
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
        for frame_idx in range(3):
            image_path = frames_dir / f"frame_{frame_idx:06d}.png"
            Image.fromarray(np.full((64, 64, 3), 40 + frame_idx * 30, dtype=np.uint8), mode="RGB").save(image_path)
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "frame_name": image_path.name,
                    "image_path": image_path.resolve().as_posix(),
                    "pts": frame_idx * 30,
                    "pts_sec": float(frame_idx),
                    "t_ns": frame_idx * 1_000_000_000,
                    "segment": "preroll" if frame_idx == 0 else "demo",
                    "is_keyframe": str(frame_idx == 0).lower(),
                    "registered": "true",
                    "pose_status": "registered",
                    "colmap_image_id": frame_idx + 1,
                }
            )

    T_wc = [
        [1.0, 0.0, 0.0, 0.45],
        [0.0, -1.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 1.00],
        [0.0, 0.0, 0.0, 1.00],
    ]
    T_cw = [
        [1.0, 0.0, 0.0, -0.45],
        [0.0, -1.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 1.00],
        [0.0, 0.0, 0.0, 1.00],
    ]
    (camera_dir / "camera_poses.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "world_frame": "W",
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "pose_status": "registered",
                        "T_wc": T_wc,
                        "T_cw": T_cw,
                    }
                    for frame_idx in range(3)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (camera_dir / "intrinsics.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "camera": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 32.0,
                    "cy": 32.0,
                    "width_px": 64,
                    "height_px": 64,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_delta_artifacts(delta_dir: Path) -> None:
    objects_dir = delta_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    masks = [
        ("obj_target", 1, "target_object", _square_mask(28, 28, 36, 36)),
        ("obj_target", 2, "target_object", _square_mask(29, 28, 37, 36)),
        ("obj_receptacle", 1, "receptacle", _square_mask(42, 24, 54, 36)),
        ("obj_receptacle", 2, "receptacle", _square_mask(42, 24, 54, 36)),
        ("obj_tabletop", 1, "tabletop", _square_mask(4, 16, 60, 56)),
        ("obj_tabletop", 2, "tabletop", _square_mask(4, 16, 60, 56)),
    ]
    mask_ref_by_pair: dict[tuple[str, int], str] = {}
    with (objects_dir / "masks_rle.jsonl").open("w", encoding="utf-8") as handle:
        for line_number, (track_id, frame_idx, ontology_id, mask) in enumerate(masks, start=1):
            handle.write(
                json.dumps(
                    {
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "ontology_id": ontology_id,
                        "rle": encode_coco_rle_mask(mask),
                    }
                )
                + "\n"
            )
            mask_ref_by_pair[(track_id, frame_idx)] = f"objects/masks_rle.jsonl:{line_number}"

    object_tracks = {
        "schema_version": "0.1.0",
        "video_id": "demo_0001",
        "tracks": [
            {
                "track_id": "obj_target",
                "ontology_id": "target_object",
                "class_name": "target_object",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [28.0, 28.0, 36.0, 36.0],
                        "centroid_uv": [32.0, 32.0],
                        "visible": True,
                        "score": 0.95,
                        "mask_rle_ref": mask_ref_by_pair[("obj_target", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [29.0, 28.0, 37.0, 36.0],
                        "centroid_uv": [33.0, 32.0],
                        "visible": True,
                        "score": 0.94,
                        "mask_rle_ref": mask_ref_by_pair[("obj_target", 2)],
                    },
                ],
            },
            {
                "track_id": "obj_receptacle",
                "ontology_id": "receptacle",
                "class_name": "receptacle",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [42.0, 24.0, 54.0, 36.0],
                        "centroid_uv": [48.0, 30.0],
                        "visible": True,
                        "score": 0.97,
                        "mask_rle_ref": mask_ref_by_pair[("obj_receptacle", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [42.0, 24.0, 54.0, 36.0],
                        "centroid_uv": [48.0, 30.0],
                        "visible": True,
                        "score": 0.97,
                        "mask_rle_ref": mask_ref_by_pair[("obj_receptacle", 2)],
                    },
                ],
            },
            {
                "track_id": "obj_tabletop",
                "ontology_id": "tabletop",
                "class_name": "tabletop",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [4.0, 16.0, 60.0, 56.0],
                        "centroid_uv": [32.0, 36.0],
                        "visible": True,
                        "score": 0.99,
                        "mask_rle_ref": mask_ref_by_pair[("obj_tabletop", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [4.0, 16.0, 60.0, 56.0],
                        "centroid_uv": [32.0, 36.0],
                        "visible": True,
                        "score": 0.99,
                        "mask_rle_ref": mask_ref_by_pair[("obj_tabletop", 2)],
                    },
                ],
            },
        ],
    }
    (objects_dir / "object_tracks.json").write_text(json.dumps(object_tracks, indent=2) + "\n", encoding="utf-8")

    interaction_rows = [
        {
            "video_id": "demo_0001",
            "frame_idx": 2,
            "t_ns": 2_000_000_000,
            "track_id": "obj_target",
            "class_name": "target_object",
            "centroid_u": 33.0,
            "centroid_v": 32.0,
            "visible": True,
            "wrist_u": 36.0,
            "wrist_v": 36.0,
            "wrist_conf": 0.9,
            "pixel_distance_wrist_to_mask": 0.0,
            "likely_contact_boolean": True,
        },
        {
            "video_id": "demo_0001",
            "frame_idx": 2,
            "t_ns": 2_000_000_000,
            "track_id": "obj_receptacle",
            "class_name": "receptacle",
            "centroid_u": 48.0,
            "centroid_v": 30.0,
            "visible": True,
            "wrist_u": 56.0,
            "wrist_v": 20.0,
            "wrist_conf": 0.92,
            "pixel_distance_wrist_to_mask": 0.0,
            "likely_contact_boolean": True,
        },
    ]
    pq.write_table(pa.Table.from_pylist(interaction_rows), objects_dir / "interactions.parquet")


def _write_gamma_artifacts(gamma_dir: Path) -> None:
    human_dir = gamma_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "video_id": "demo_0001",
            "track_id": 7,
            "frame_idx": 0,
            "t_ns": 0,
            "segment": "preroll",
            "is_primary_demonstrator": True,
            "camera_pose_status": "registered",
            "track_visible": False,
            "wrist_conf": None,
            "elbow_r_x_w": None,
            "elbow_r_y_w": None,
            "elbow_r_z_w": None,
            "wrist_r_x_w": None,
            "wrist_r_y_w": None,
            "wrist_r_z_w": None,
        },
        {
            "video_id": "demo_0001",
            "track_id": 7,
            "frame_idx": 1,
            "t_ns": 1_000_000_000,
            "segment": "demo",
            "is_primary_demonstrator": True,
            "camera_pose_status": "registered",
            "track_visible": True,
            "wrist_conf": 0.9,
            "elbow_r_x_w": 0.38,
            "elbow_r_y_w": -0.04,
            "elbow_r_z_w": 0.18,
            "wrist_r_x_w": 0.53,
            "wrist_r_y_w": 0.08,
            "wrist_r_z_w": 0.06,
        },
        {
            "video_id": "demo_0001",
            "track_id": 7,
            "frame_idx": 2,
            "t_ns": 2_000_000_000,
            "segment": "demo",
            "is_primary_demonstrator": True,
            "camera_pose_status": "registered",
            "track_visible": True,
            "wrist_conf": 0.92,
            "elbow_r_x_w": 0.52,
            "elbow_r_y_w": 0.09,
            "elbow_r_z_w": 0.19,
            "wrist_r_x_w": 0.68,
            "wrist_r_y_w": 0.26,
            "wrist_r_z_w": 0.06,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), human_dir / "arm_observables.parquet")


def _prepare_back_half_inputs(tmp_path: Path):
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    project_path = bundle_root / "spec" / "project.yaml"
    project_payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    project_payload["acceptance"]["eta_min_demo_frames"] = 2
    project_path.write_text(yaml.safe_dump(project_payload, sort_keys=False), encoding="utf-8")
    beta_dir = tmp_path / "beta"
    delta_dir = tmp_path / "delta"
    gamma_dir = tmp_path / "gamma"
    run_dir = tmp_path / "run"
    _write_beta_artifacts(beta_dir)
    _write_delta_artifacts(delta_dir)
    _write_gamma_artifacts(gamma_dir)
    shutil.copytree(beta_dir / "frames", gamma_dir / "frames")
    bundle = validate_bundle(load_bundle(bundle_root / "spec"))
    return bundle_root, beta_dir, delta_dir, gamma_dir, run_dir, bundle


def test_compile_metric_scene_writes_truthful_proxy_epsilon_outputs(tmp_path: Path) -> None:
    _, beta_dir, delta_dir, _, run_dir, bundle = _prepare_back_half_inputs(tmp_path)

    summary = compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)

    assert summary["geometry_mode"] == "proxy_scene"
    assert summary["metric_alignment_mode"] == "inherited_from_beta"
    assert summary["qa"]["support_plane_rmse_m"] is None
    assert summary["qc_flags"]["truthful_provenance_ok"] is True
    assert (run_dir / "scene" / "world_metric_from_world.json").exists()
    assert (run_dir / "scene" / "support_plane.json").exists()
    assert (run_dir / "scene" / "static_dense" / "fused.ply").exists()
    assert (run_dir / "scene" / "static_mesh.obj").exists()
    assert (run_dir / "camera" / "camera_poses_metric.json").exists()
    world_metric = json.loads((run_dir / "scene" / "world_metric_from_world.json").read_text(encoding="utf-8"))
    assert world_metric["transform_source"] == "inherited_from_beta_fiducial_world"
    assert world_metric["measured"] is False
    object_payload = json.loads((run_dir / "scene" / "object_init_poses_metric.json").read_text(encoding="utf-8"))
    assert {obj["ontology_id"] for obj in object_payload["objects"]} == {"target_object", "receptacle"}
    assert all(obj["geometry_source"] == "proxy_box_surface" for obj in object_payload["objects"])
    assert all(obj["footprint_estimator"] == "support_plane_mask_obb" for obj in object_payload["objects"])
    assert all(obj["height_mode"] == "proxy_default_height" for obj in object_payload["objects"])
    assert all(len(obj["frames_used"]) >= 1 for obj in object_payload["objects"])
    assert all(obj["extents_m"][2] == pytest.approx(bundle.project.epsilon.default_object_height_m) for obj in object_payload["objects"])
    assert all(max(obj["extents_m"][:2]) < 0.25 for obj in object_payload["objects"])
    assert summary["object_geometry_diagnostics"]


def test_assetize_metric_scene_writes_proxy_assets_and_validates_mujoco(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    _, beta_dir, delta_dir, _, run_dir, bundle = _prepare_back_half_inputs(tmp_path)
    compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)

    summary = assetize_metric_scene(bundle=bundle, epsilon_dir=run_dir, out_dir=run_dir)

    assert summary["qc_flags"]["collision_geom_count_ok"] is True
    assert summary["qc_flags"]["mujoco_compile_ok"] is True
    assert summary["qc_flags"]["passive_smoke_ok"] is True
    assert (run_dir / "assets" / "manifest.json").exists()
    assert (run_dir / "assets" / "mujoco_assets.xml").exists()
    assert (run_dir / "sim" / "assets_only.xml").exists()
    assert (run_dir / "sim" / "assets_only.mjb").exists()
    manifest = json.loads((run_dir / "assets" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["objects"]) == 2
    assert manifest["asset_mode"] == "proxy_visual_and_collision"
    assert all(Path(run_dir / obj["visual_mesh"]).exists() for obj in manifest["objects"])
    assert all(obj["source_kind"] == "proxy_box_asset" for obj in manifest["objects"])
    assert all(Path(run_dir / obj["metadata_ref"]).exists() for obj in manifest["objects"])


def test_retarget_demonstration_writes_eta_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    _, beta_dir, delta_dir, gamma_dir, run_dir, bundle = _prepare_back_half_inputs(tmp_path)
    compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)
    assetize_metric_scene(bundle=bundle, epsilon_dir=run_dir, out_dir=run_dir)

    summary = retarget_monocular_demonstration(
        bundle=bundle,
        gamma_dir=gamma_dir,
        delta_dir=delta_dir,
        epsilon_dir=run_dir,
        zeta_dir=run_dir,
        out_dir=run_dir,
    )

    assert summary["qc_flags"]["demo_frame_count_ok"] is True
    assert summary["qc_flags"]["joint_step_ok"] is True
    assert summary["qc_flags"]["ik_solve_rate_ok"] is True
    assert summary["demo_frame_count"] == 2
    assert summary["retarget_frame_count"] >= bundle.project.acceptance.eta_min_demo_frames
    assert summary["ik"]["solve_rate"] == pytest.approx(1.0)
    assert (run_dir / "retarget" / "robot_demo.npz").exists()
    assert (run_dir / "retarget" / "robot_target.yaml").exists()
    assert (run_dir / "retarget" / "robot_base_in_metric_world.json").exists()
    assert (run_dir / "retarget" / "contact_schedule.json").exists()
    demo_payload = np.load(run_dir / "retarget" / "robot_demo.npz", allow_pickle=True)
    assert len(demo_payload["t_ns"]) == summary["retarget_frame_count"]
    assert "replay" not in summary
    robot_base_payload = json.loads((run_dir / "retarget" / "robot_base_in_metric_world.json").read_text(encoding="utf-8"))
    assert robot_base_payload["registration_method"] == "automatic_trace_fit_proxy"
    assert not np.allclose(np.asarray(robot_base_payload["X_Br_from_M"], dtype=float), np.eye(4), atol=1e-6)
    assert summary["qc_flags"]["robot_base_identity_ok"] is True
    assert "target_position_diagnostics" in summary
    assert "waypoint_clip_diagnostics" in summary


def test_eta_uses_persisted_task_window_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    _, beta_dir, delta_dir, gamma_dir, run_dir, bundle = _prepare_back_half_inputs(tmp_path)
    bundle.project.acceptance.eta_min_demo_frames = 1
    compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)
    assetize_metric_scene(bundle=bundle, epsilon_dir=run_dir, out_dir=run_dir)
    (gamma_dir / "task_window.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": bundle.project.video_id,
                "source": "cli_override",
                "start_frame_idx": 2,
                "end_frame_idx": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = retarget_monocular_demonstration(
        bundle=bundle,
        gamma_dir=gamma_dir,
        delta_dir=delta_dir,
        epsilon_dir=run_dir,
        zeta_dir=run_dir,
        out_dir=run_dir,
    )

    assert summary["task_window"]["start_frame_idx"] == 2
    assert summary["demo_frame_count"] == 1


def test_epsilon_zeta_eta_cli_pipeline_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    bundle_root, beta_dir, delta_dir, gamma_dir, run_dir, _ = _prepare_back_half_inputs(tmp_path)

    epsilon_result = runner.invoke(
        app,
        [
            "epsilon",
            "compile-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--delta-dir",
            str(delta_dir),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert epsilon_result.exit_code == 0

    zeta_result = runner.invoke(
        app,
        [
            "zeta",
            "assetize-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--epsilon-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert zeta_result.exit_code == 0

    eta_result = runner.invoke(
        app,
        [
            "eta",
            "retarget-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--gamma-dir",
            str(gamma_dir),
            "--delta-dir",
            str(delta_dir),
            "--epsilon-dir",
            str(run_dir),
            "--zeta-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
            "--task-start-frame",
            "1",
            "--task-end-frame",
            "2",
        ],
    )
    assert eta_result.exit_code == 0
    assert (run_dir / "retarget" / "eta_summary.json").exists()


def test_compile_task_package_writes_theta_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    _, beta_dir, delta_dir, gamma_dir, run_dir, bundle = _prepare_back_half_inputs(tmp_path)
    compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)
    assetize_metric_scene(bundle=bundle, epsilon_dir=run_dir, out_dir=run_dir)
    retarget_monocular_demonstration(
        bundle=bundle,
        gamma_dir=gamma_dir,
        delta_dir=delta_dir,
        epsilon_dir=run_dir,
        zeta_dir=run_dir,
        out_dir=run_dir,
    )

    validation = compile_task_package(
        bundle=bundle,
        gamma_dir=gamma_dir,
        epsilon_dir=run_dir,
        zeta_dir=run_dir,
        eta_dir=run_dir,
        out_dir=run_dir,
    )

    assert validation["qc_flags"]["scene_compile_ok"] is True
    assert validation["qc_flags"]["mjb_load_ok"] is True
    assert validation["qc_flags"]["trace_smoke_ok"] is True
    assert validation["qc_flags"]["audit_render_ok"] is True
    assert validation["qc_flags"]["robot_ghost_visible"] is True
    assert validation["qc_flags"]["human_ghost_visible"] is True
    assert validation["qc_flags"]["robot_base_identity_ok"] is True
    assert validation["qc_flags"]["object_extents_clip_ceiling_ok"] is True
    assert (run_dir / "sim" / "scene.xml").exists()
    assert (run_dir / "sim" / "scene.mjb").exists()
    assert (run_dir / "sim" / "task.json").exists()
    assert (run_dir / "sim" / "validation.json").exists()
    assert (run_dir / "sim" / "playback" / "robot_trace.npz").exists()
    assert (run_dir / "sim" / "playback" / "human_arm_ghost.npz").exists()
    task_payload = json.loads((run_dir / "sim" / "task.json").read_text(encoding="utf-8"))
    assert task_payload["task"]["success_metric"] == "object_in_target_region"
    assert len(task_payload["task"]["normalized_events"]) == 7
    assert "declared_target_region_m" in task_payload["task"]
    assert "runtime_target_region_m" in task_payload["task"]
    assert task_payload["task"]["runtime_region_resolution"] == "auto_reanchored_from_scene"
    assert validation["spatial_sanity"]["runtime_region_offset_m"] > 0.0


def test_cli_theta_pipeline_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    bundle_root, beta_dir, delta_dir, gamma_dir, run_dir, _ = _prepare_back_half_inputs(tmp_path)

    assert runner.invoke(
        app,
        [
            "epsilon",
            "compile-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--delta-dir",
            str(delta_dir),
            "--out-dir",
            str(run_dir),
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "zeta",
            "assetize-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--epsilon-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "eta",
            "retarget-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--gamma-dir",
            str(gamma_dir),
            "--delta-dir",
            str(delta_dir),
            "--epsilon-dir",
            str(run_dir),
            "--zeta-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
            "--task-start-frame",
            "1",
            "--task-end-frame",
            "2",
        ],
    ).exit_code == 0

    theta_result = runner.invoke(
        app,
        [
            "sim",
            "compile-task",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--gamma-dir",
            str(gamma_dir),
            "--epsilon-dir",
            str(run_dir),
            "--zeta-dir",
            str(run_dir),
            "--eta-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    )

    assert theta_result.exit_code == 0
    assert (run_dir / "sim" / "validation.json").exists()
