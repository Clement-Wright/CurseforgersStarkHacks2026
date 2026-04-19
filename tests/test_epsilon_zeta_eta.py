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


class _FakeRenderer:
    def __init__(self, model: _FakeMjModel, width: int, height: int):
        self.model = model
        self.width = width
        self.height = height
        self.camera = "unknown"

    def update_scene(self, data: _FakeMjData, camera: str | None = None) -> None:
        _ = data
        if camera is not None:
            self.camera = camera

    def render(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
        frame[40:120, 40:240, :] = np.array([40, 180, 90], dtype=np.uint8)
        if self.camera == "debug_camera":
            frame[180:320, 280:520, :] = np.array([70, 90, 220], dtype=np.uint8)
        else:
            frame[180:320, 280:520, :] = np.array([220, 90, 70], dtype=np.uint8)
        return frame

    def close(self) -> None:
        return None


class _FakeMujocoModule:
    MjModel = _FakeMjModel
    MjData = _FakeMjData
    Renderer = _FakeRenderer

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
    monkeypatch.setattr(theta_sim_module, "ensure_theta_dependencies", lambda: None)


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
    scene_dir = beta_dir / "scene"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

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
    (camera_dir / "robot_base_in_metric_world.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "source_world": "M",
                "target_frame": "Br",
                "X_Br_from_M": [
                    [0.0, -1.0, 0.0, 0.18],
                    [1.0, 0.0, 0.0, -0.34],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "registration_method": "measured_fiducial_anchor",
                "measured": True,
                "anchor_kind": "fiducial_board",
                "anchor_board_id": "fiducial_board",
                "anchor_frame": "Ft",
                "X_Br_from_anchor": [
                    [0.0, -1.0, 0.0, 0.18],
                    [1.0, 0.0, 0.0, -0.34],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "legacy_debug_fallback_used": False,
                "anchor_selection_policy": "nearest_visible_observation_for_provenance",
                "anchor_marker_id": 7,
                "anchor_frame_name": "Ft_7",
                "anchor_translation_tc_m": [0.08, -0.03, 0.55],
                "anchor_rotation_tc": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "anchor_distance_m": 0.556,
                "quality": {
                    "visible_observation_count": 3,
                    "nearest_visible_anchor_distance_m": 0.556,
                    "confidence": 0.93,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (scene_dir / "static_geometry_subset.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "source": "beta_preroll_static_subset",
                "selection_policy": "registered_preroll_frames",
                "frames": [
                    {
                        "frame_idx": 0,
                        "frame_name": "frame_000000.png",
                        "segment": "preroll",
                        "registered": True,
                    }
                ],
                "keyframe_frame_indices": [0],
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
        ("eraser_01_track", "eraser_01", 1, "eraser", _square_mask(18, 30, 24, 34)),
        ("eraser_01_track", "eraser_01", 2, "eraser", _square_mask(19, 30, 25, 34)),
        ("eraser_02_track", "eraser_02", 1, "eraser", _square_mask(24, 25, 30, 29)),
        ("eraser_02_track", "eraser_02", 2, "eraser", _square_mask(25, 25, 31, 29)),
        ("toothpaste_box_01_track", "toothpaste_box_01", 1, "toothpaste_box", _square_mask(34, 24, 50, 34)),
        ("toothpaste_box_01_track", "toothpaste_box_01", 2, "toothpaste_box", _square_mask(34, 24, 50, 34)),
        ("toothpaste_box_02_track", "toothpaste_box_02", 1, "toothpaste_box", _square_mask(36, 10, 52, 20)),
        ("toothpaste_box_02_track", "toothpaste_box_02", 2, "toothpaste_box", _square_mask(36, 10, 52, 20)),
        ("tabletop_01_track", "tabletop_01", 1, "tabletop", _square_mask(4, 16, 60, 56)),
        ("tabletop_01_track", "tabletop_01", 2, "tabletop", _square_mask(4, 16, 60, 56)),
    ]
    mask_ref_by_pair: dict[tuple[str, int], str] = {}
    with (objects_dir / "masks_rle.jsonl").open("w", encoding="utf-8") as handle:
        for line_number, (track_id, instance_id, frame_idx, ontology_id, mask) in enumerate(masks, start=1):
            handle.write(
                json.dumps(
                    {
                        "track_id": track_id,
                        "instance_id": instance_id,
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
                "track_id": "eraser_01_track",
                "instance_id": "eraser_01",
                "ontology_id": "eraser",
                "class_name": "eraser",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [18.0, 30.0, 24.0, 34.0],
                        "centroid_uv": [21.0, 32.0],
                        "visible": True,
                        "score": 0.95,
                        "mask_rle_ref": mask_ref_by_pair[("eraser_01_track", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [19.0, 30.0, 25.0, 34.0],
                        "centroid_uv": [22.0, 32.0],
                        "visible": True,
                        "score": 0.94,
                        "mask_rle_ref": mask_ref_by_pair[("eraser_01_track", 2)],
                    },
                ],
            },
            {
                "track_id": "eraser_02_track",
                "instance_id": "eraser_02",
                "ontology_id": "eraser",
                "class_name": "eraser",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [24.0, 25.0, 30.0, 29.0],
                        "centroid_uv": [27.0, 27.0],
                        "visible": True,
                        "score": 0.93,
                        "mask_rle_ref": mask_ref_by_pair[("eraser_02_track", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [25.0, 25.0, 31.0, 29.0],
                        "centroid_uv": [28.0, 27.0],
                        "visible": True,
                        "score": 0.92,
                        "mask_rle_ref": mask_ref_by_pair[("eraser_02_track", 2)],
                    },
                ],
            },
            {
                "track_id": "toothpaste_box_01_track",
                "instance_id": "toothpaste_box_01",
                "ontology_id": "toothpaste_box",
                "class_name": "toothpaste_box",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [34.0, 24.0, 50.0, 34.0],
                        "centroid_uv": [42.0, 29.0],
                        "visible": True,
                        "score": 0.97,
                        "mask_rle_ref": mask_ref_by_pair[("toothpaste_box_01_track", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [34.0, 24.0, 50.0, 34.0],
                        "centroid_uv": [42.0, 29.0],
                        "visible": True,
                        "score": 0.97,
                        "mask_rle_ref": mask_ref_by_pair[("toothpaste_box_01_track", 2)],
                    },
                ],
            },
            {
                "track_id": "toothpaste_box_02_track",
                "instance_id": "toothpaste_box_02",
                "ontology_id": "toothpaste_box",
                "class_name": "toothpaste_box",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [36.0, 10.0, 52.0, 20.0],
                        "centroid_uv": [44.0, 15.0],
                        "visible": True,
                        "score": 0.96,
                        "mask_rle_ref": mask_ref_by_pair[("toothpaste_box_02_track", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [36.0, 10.0, 52.0, 20.0],
                        "centroid_uv": [44.0, 15.0],
                        "visible": True,
                        "score": 0.96,
                        "mask_rle_ref": mask_ref_by_pair[("toothpaste_box_02_track", 2)],
                    },
                ],
            },
            {
                "track_id": "tabletop_01_track",
                "instance_id": "tabletop_01",
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
                        "mask_rle_ref": mask_ref_by_pair[("tabletop_01_track", 1)],
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "bbox_xyxy": [4.0, 16.0, 60.0, 56.0],
                        "centroid_uv": [32.0, 36.0],
                        "visible": True,
                        "score": 0.99,
                        "mask_rle_ref": mask_ref_by_pair[("tabletop_01_track", 2)],
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
            "track_id": "eraser_01_track",
            "instance_id": "eraser_01",
            "ontology_id": "eraser",
            "class_name": "eraser",
            "centroid_u": 22.0,
            "centroid_v": 32.0,
            "visible": True,
            "wrist_u": 24.0,
            "wrist_v": 36.0,
            "wrist_conf": 0.9,
            "pixel_distance_wrist_to_mask": 0.0,
            "likely_contact_boolean": True,
        },
        {
            "video_id": "demo_0001",
            "frame_idx": 2,
            "t_ns": 2_000_000_000,
            "track_id": "toothpaste_box_01_track",
            "instance_id": "toothpaste_box_01",
            "ontology_id": "toothpaste_box",
            "class_name": "toothpaste_box",
            "centroid_u": 42.0,
            "centroid_v": 29.0,
            "visible": True,
            "wrist_u": 44.0,
            "wrist_v": 24.0,
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

    assert summary["geometry_mode"] == "dense_static_reconstruction"
    assert summary["metric_alignment_mode"] == "inherited_from_beta"
    assert summary["qa"]["support_plane_rmse_m"] is not None
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
    assert {obj["instance_id"] for obj in object_payload["objects"]} == {
        "eraser_01",
        "eraser_02",
        "toothpaste_box_01",
        "toothpaste_box_02",
    }
    assert {obj["ontology_id"] for obj in object_payload["objects"]} == {"eraser", "toothpaste_box"}
    assert all(obj["geometry_source"] in {"mask_backprojected_cuboid", "scene_instance_size_prior_cuboid"} for obj in object_payload["objects"])
    assert all(obj["footprint_estimator"] == "support_plane_mask_obb" for obj in object_payload["objects"])
    assert all(obj["height_mode"] == "proxy_default_height" for obj in object_payload["objects"])
    assert all(len(obj["frames_used"]) >= 1 for obj in object_payload["objects"])
    assert all(obj["extents_m"][2] == pytest.approx(bundle.project.epsilon.default_object_height_m) for obj in object_payload["objects"])
    assert all(max(obj["extents_m"][:2]) < 0.25 for obj in object_payload["objects"])
    assert all((run_dir / obj["mesh_path"]).exists() for obj in object_payload["objects"])
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
    assert len(manifest["objects"]) == 4
    assert manifest["asset_mode"] == "geometry_backed_assets"
    assert all(Path(run_dir / obj["visual_mesh"]).exists() for obj in manifest["objects"])
    assert all(obj["visual_geometry_mode"] in {"cuboid_visual", "geometry_backed_visual"} for obj in manifest["objects"])
    assert all(Path(run_dir / obj["collision_meshes"][0]).exists() for obj in manifest["objects"])
    assert all(Path(run_dir / obj["metadata_ref"]).exists() for obj in manifest["objects"])


def test_retarget_demonstration_writes_eta_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sim_dependencies(monkeypatch)
    _, beta_dir, delta_dir, gamma_dir, run_dir, bundle = _prepare_back_half_inputs(tmp_path)
    compile_metric_scene(bundle=bundle, beta_dir=beta_dir, delta_dir=delta_dir, out_dir=run_dir)
    assetize_metric_scene(bundle=bundle, epsilon_dir=run_dir, out_dir=run_dir)

    summary = retarget_monocular_demonstration(
        bundle=bundle,
        beta_dir=beta_dir,
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
    assert robot_base_payload["registration_method"] == "measured_fiducial_anchor"
    assert robot_base_payload["anchor_selection_policy"] == "nearest_visible_observation_for_provenance"
    assert robot_base_payload["measured"] is True
    assert not np.allclose(np.asarray(robot_base_payload["X_Br_from_M"], dtype=float), np.eye(4), atol=1e-6)
    assert summary["qc_flags"]["robot_base_identity_ok"] is True
    assert summary["robot_base_registration"]["consumed_by_eta"] is True
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
        beta_dir=beta_dir,
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
            "--beta-dir",
            str(beta_dir),
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
        beta_dir=beta_dir,
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
    assert validation["qc_flags"]["playback_renders_ok"] is True
    assert validation["qc_flags"]["renderer_backed_renders_ok"] is True
    assert validation["qc_flags"]["robot_ghost_visible"] is True
    assert validation["qc_flags"]["human_ghost_visible"] is True
    assert validation["qc_flags"]["robot_base_identity_ok"] is True
    assert validation["qc_flags"]["robot_base_measured_ok"] is True
    assert validation["qc_flags"]["object_extents_clip_ceiling_ok"] is True
    assert validation["qc_flags"]["instance_contract_ok"] is True
    assert (run_dir / "sim" / "scene.xml").exists()
    assert (run_dir / "sim" / "scene.mjb").exists()
    assert (run_dir / "sim" / "task.json").exists()
    assert (run_dir / "sim" / "validation.json").exists()
    assert (run_dir / "sim" / "playback" / "robot_trace.npz").exists()
    assert (run_dir / "sim" / "playback" / "human_arm_ghost.npz").exists()
    assert (run_dir / "sim" / "playback" / "audit_camera.mp4").exists()
    assert (run_dir / "sim" / "playback" / "debug_camera.mp4").exists()
    task_payload = json.loads((run_dir / "sim" / "task.json").read_text(encoding="utf-8"))
    assert task_payload["task"]["success_metric"] == "object_in_target_region"
    assert task_payload["task"]["family"] == "tabletop_stacking"
    assert len(task_payload["task"]["normalized_events"]) == 7
    assert len(task_payload["task"]["scene_instances"]) >= 5
    assert task_payload["task"]["goals"][0]["goal_id"] == "stack_white_erasers"
    assert "declared_target_region_m" in task_payload["task"]
    assert "runtime_target_region_m" in task_payload["task"]
    assert task_payload["task"]["runtime_region_resolution"] == "auto_reanchored_from_scene"
    assert validation["spatial_sanity"]["runtime_region_offset_m"] > 0.0
    assert validation["audit_render"]["renderer_backed"] is True
    assert validation["audit_render"]["mode"] == "mujoco_renderer"
    assert validation["instance_contract"]["count_ok"] is True


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
            "--beta-dir",
            str(beta_dir),
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


def test_iota_and_kappa_cli_pipeline_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("h5py")
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
            "--beta-dir",
            str(beta_dir),
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
    assert runner.invoke(
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
    ).exit_code == 0

    dataset_result = runner.invoke(
        app,
        [
            "data",
            "build-dataset",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--theta-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert dataset_result.exit_code == 0
    assert (run_dir / "data" / "demos.hdf5").exists()

    bc_result = runner.invoke(
        app,
        [
            "train",
            "imitation",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--dataset-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert bc_result.exit_code == 0
    assert (run_dir / "policies" / "bc_state.npz").exists()
    assert (run_dir / "eval" / "bc_rollouts.json").exists()

    rl_result = runner.invoke(
        app,
        [
            "train",
            "finetune-rl",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--dataset-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert rl_result.exit_code == 0
    assert (run_dir / "policies" / "sac_ft_state.npz").exists()
    assert (run_dir / "eval" / "randomized_sweeps.json").exists()

    deploy_result = runner.invoke(
        app,
        [
            "deploy",
            "export-ros2",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--theta-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir),
            "--policy-path",
            str(run_dir / "policies" / "bc_state.npz"),
        ],
    )
    assert deploy_result.exit_code == 0
    assert (run_dir / "ros2" / "config" / "controllers.yaml").exists()
    assert (run_dir / "ros2" / "config" / "safety_supervisor.yaml").exists()
    assert (run_dir / "ros2" / "config" / "theta_parity.json").exists()
    assert (run_dir / "ros2" / "export_summary.json").exists()
