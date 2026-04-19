from __future__ import annotations

import importlib.util
import json
import importlib
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .specs import PROJECT_SCHEMA_VERSION, SpecBundle


class ZetaAssetError(Exception):
    """Raised when zeta assetization cannot complete successfully."""


class DependencyError(ZetaAssetError):
    """Raised when zeta runtime dependencies are missing."""


def _path_string(path: Path) -> str:
    return path.as_posix()


def _sanitize_name(value: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    sanitized = sanitized.strip("_")
    return sanitized or "asset"


def missing_zeta_dependencies() -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    return missing


def ensure_zeta_dependencies() -> None:
    missing = missing_zeta_dependencies()
    if missing:
        raise DependencyError(
            "Missing Python dependencies for zeta assetize: " + ", ".join(sorted(missing))
        )


def _load_mujoco_module() -> Any:
    if importlib.util.find_spec("mujoco") is None:
        raise DependencyError(
            "Missing Python dependency for zeta assetize: mujoco. Install env/sim.environment.yml "
            "to enable MuJoCo compile validation."
        )
    return importlib.import_module("mujoco")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ZetaAssetError(f"missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ZetaAssetError(f"expected a JSON object in {path}")
    return payload


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


def _write_obj_mesh(path: Path, vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n")
        for face in faces:
            indices = " ".join(str(int(index) + 1) for index in face)
            handle.write(f"f {indices}\n")


def _write_asset_xml(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f'<mujoco model="{manifest["video_id"]}_assets">',
        "  <asset>",
    ]
    static = manifest["static"]
    lines.append(
        f'    <mesh name="static_visual" file="{Path(static["visual_mesh"]).name}"/>'
    )
    for collision_index, collision_mesh in enumerate(static["collision_meshes"]):
        lines.append(
            f'    <mesh name="static_collision_{collision_index:02d}" file="{Path(collision_mesh).name}"/>'
        )
    for obj in manifest["objects"]:
        mesh_prefix = _sanitize_name(str(obj["track_id"]))
        lines.append(
            f'    <mesh name="{mesh_prefix}_visual" file="{Path(obj["visual_mesh"]).name}"/>'
        )
        for collision_index, collision_mesh in enumerate(obj["collision_meshes"]):
            lines.append(
                f'    <mesh name="{mesh_prefix}_collision_{collision_index:02d}" '
                f'file="{Path(collision_mesh).name}"/>'
            )
    lines.extend(["  </asset>", "</mujoco>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_assets_only_xml(path: Path, manifest: dict[str, Any]) -> None:
    static = manifest["static"]
    lines = [
        f'<mujoco model="{manifest["video_id"]}_assets_only">',
        '  <compiler angle="radian" coordinate="local" meshdir="../assets"/>',
        '  <option timestep="0.002" gravity="0 0 -9.81"/>',
        "  <asset>",
        f'    <mesh name="static_visual" file="{Path(static["visual_mesh"]).name}"/>',
    ]
    for collision_index, collision_mesh in enumerate(static["collision_meshes"]):
        lines.append(
            f'    <mesh name="static_collision_{collision_index:02d}" file="{Path(collision_mesh).name}"/>'
        )
    for obj in manifest["objects"]:
        mesh_prefix = _sanitize_name(str(obj["track_id"]))
        lines.append(
            f'    <mesh name="{mesh_prefix}_visual" file="{Path(obj["visual_mesh"]).name}"/>'
        )
        for collision_index, collision_mesh in enumerate(obj["collision_meshes"]):
            lines.append(
                f'    <mesh name="{mesh_prefix}_collision_{collision_index:02d}" file="{Path(collision_mesh).name}"/>'
            )
    lines.extend(
        [
            "  </asset>",
            "  <worldbody>",
            '    <geom name="world_table" type="mesh" mesh="static_collision_00" rgba="0.85 0.85 0.85 1"/>',
        ]
    )
    for obj in manifest["objects"]:
        mesh_prefix = _sanitize_name(str(obj["track_id"]))
        position = obj["position_m"]
        lines.extend(
            [
                f'    <body name="{mesh_prefix}" pos="{position[0]} {position[1]} {position[2]}">',
                f'      <geom type="mesh" mesh="{mesh_prefix}_visual" contype="0" conaffinity="0" rgba="0.7 0.3 0.2 1"/>',
            ]
        )
        for collision_index, _ in enumerate(obj["collision_meshes"]):
            lines.append(
                f'      <geom type="mesh" mesh="{mesh_prefix}_collision_{collision_index:02d}" '
                'density="250" friction="0.8 0.01 0.001"/>'
            )
        lines.append("    </body>")
    lines.extend(["  </worldbody>", "</mujoco>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_object_metadata(path: Path, manifest_object: dict[str, Any], *, world_frame: str) -> None:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "track_id": manifest_object["track_id"],
        "instance_id": manifest_object.get("instance_id"),
        "ontology_id": manifest_object["ontology_id"],
        "class_name": manifest_object["class_name"],
        "world_frame": world_frame,
        "source_kind": manifest_object["source_kind"],
        "visual_geometry_mode": manifest_object["visual_geometry_mode"],
        "collision_geometry_mode": manifest_object["collision_geometry_mode"],
        "initial_pose_ref": manifest_object["initial_pose_ref"],
        "visual_mesh": manifest_object["visual_mesh"],
        "collision_meshes": manifest_object["collision_meshes"],
        "bbox_m": manifest_object["bbox_m"],
        "position_m": manifest_object["position_m"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    raise ZetaAssetError("MuJoCo bindings did not expose a usable MJB save API")


def _compile_and_smoke_assets(xml_path: Path, mjb_path: Path) -> dict[str, Any]:
    mujoco = _load_mujoco_module()
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        for _ in range(8):
            mujoco.mj_step(model, data)
        _save_mjb(mujoco, model, mjb_path)
    except Exception as exc:
        raise ZetaAssetError(f"MuJoCo compile/smoke failed for {xml_path}: {exc}") from exc
    return {
        "compile_ok": True,
        "passive_smoke_ok": True,
        "step_count": 8,
        "nq": int(getattr(model, "nq", 0)),
        "nv": int(getattr(model, "nv", 0)),
        "mjb_path": _path_string(mjb_path),
    }


def _build_summary(bundle: SpecBundle, manifest: dict[str, Any], smoke_summary: dict[str, Any]) -> dict[str, Any]:
    max_collision_geoms = max(
        [len(obj["collision_meshes"]) for obj in manifest["objects"]] or [len(manifest["static"]["collision_meshes"])]
    )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.zeta.primary_backbone,
        "asset_mode": bundle.project.zeta.asset_mode,
        "object_asset_count": len(manifest["objects"]),
        "static_collision_mesh_count": len(manifest["static"]["collision_meshes"]),
        "max_collision_geoms_per_object": max_collision_geoms,
        "mujoco": smoke_summary,
        "qc_flags": {
            "collision_geom_count_ok": max_collision_geoms <= bundle.project.acceptance.zeta_max_collision_geoms_per_object,
            "mujoco_compile_ok": bool(smoke_summary.get("compile_ok")),
            "passive_smoke_ok": bool(smoke_summary.get("passive_smoke_ok")),
        },
    }


def enforce_zeta_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if summary["max_collision_geoms_per_object"] > bundle.project.acceptance.zeta_max_collision_geoms_per_object:
        raise ZetaAssetError(
            f"collision geom count {summary['max_collision_geoms_per_object']} exceeded "
            f"zeta_max_collision_geoms_per_object={bundle.project.acceptance.zeta_max_collision_geoms_per_object}"
        )
    if bundle.project.acceptance.zeta_require_mujoco_compile and not bool(summary["qc_flags"]["mujoco_compile_ok"]):
        raise ZetaAssetError("MuJoCo compile validation did not succeed for zeta assets")
    if bundle.project.acceptance.zeta_require_passive_smoke and not bool(summary["qc_flags"]["passive_smoke_ok"]):
        raise ZetaAssetError("MuJoCo passive smoke validation did not succeed for zeta assets")


def assetize_metric_scene(
    bundle: SpecBundle,
    epsilon_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_zeta_dependencies()
    if bundle.project.zeta.asset_mode not in {"proxy_visual_and_collision", "geometry_backed_assets"}:
        raise ZetaAssetError(
            f"zeta asset_mode '{bundle.project.zeta.asset_mode}' is not implemented yet; "
            "the current MVP supports proxy_visual_and_collision and geometry_backed_assets"
        )

    scene_dir = epsilon_dir / "scene"
    scene_meta = _load_json(scene_dir / "static_mesh.meta.json")
    object_payload = _load_json(scene_dir / "object_init_poses_metric.json")
    static_mesh_path = scene_dir / "static_mesh.obj"
    if not static_mesh_path.exists():
        raise ZetaAssetError(f"missing epsilon static mesh: {static_mesh_path}")

    assets_dir = out_dir / "assets"
    sim_dir = out_dir / "sim"
    assets_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    static_visual_path = assets_dir / "static_visual.obj"
    shutil.copyfile(static_mesh_path, static_visual_path)
    bbox = scene_meta.get("bbox_m", {})
    min_corner = np.asarray(bbox.get("min", [0.25, -0.25, 0.0]), dtype=float)
    max_corner = np.asarray(bbox.get("max", [0.75, 0.25, 0.05]), dtype=float)
    static_vertices, static_faces = _box_vertices_faces(0.5 * (min_corner + max_corner), max_corner - min_corner)
    static_collision_path = assets_dir / "static_collision_00.obj"
    _write_obj_mesh(static_collision_path, static_vertices, static_faces)

    manifest = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "world_frame": bundle.project.epsilon.metric_world_frame,
        "source": bundle.project.zeta.primary_backbone,
        "asset_mode": bundle.project.zeta.asset_mode,
        "static": {
            "visual_mesh": "assets/static_visual.obj",
            "collision_meshes": ["assets/static_collision_00.obj"],
            "source_kind": "proxy_scene_mesh",
            "visual_geometry_mode": scene_meta.get("geometry_mode", "proxy_scene"),
            "collision_geometry_mode": "primitive_box",
            "initial_pose_ref": "scene/static_mesh.meta.json",
            "bbox_m": {
                "min": min_corner.tolist(),
                "max": max_corner.tolist(),
            },
        },
        "objects": [],
    }
    for obj in object_payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        asset_name = _sanitize_name(str(obj["track_id"]))
        extents = np.asarray(obj.get("extents_m", [0.04, 0.04, 0.04]), dtype=float)
        vertices, faces = _box_vertices_faces(np.zeros(3, dtype=float), extents)
        source_mesh_rel = str(obj.get("mesh_path", "")).strip()
        visual_path = assets_dir / f"{asset_name}_visual.obj"
        collision_path = assets_dir / f"{asset_name}_collision_00.obj"
        if source_mesh_rel:
            source_mesh_path = epsilon_dir / source_mesh_rel
            if source_mesh_path.exists():
                shutil.copyfile(source_mesh_path, visual_path)
            else:
                _write_obj_mesh(visual_path, vertices, faces)
        else:
            _write_obj_mesh(visual_path, vertices, faces)
        _write_obj_mesh(collision_path, vertices, faces)
        position = [float(value) for value in obj["position_m"]]
        extents = [float(value) for value in obj["extents_m"]]
        bbox_min = [position[0] - 0.5 * extents[0], position[1] - 0.5 * extents[1], position[2] - 0.5 * extents[2]]
        bbox_max = [position[0] + 0.5 * extents[0], position[1] + 0.5 * extents[1], position[2] + 0.5 * extents[2]]
        manifest_object = {
            "track_id": obj["track_id"],
            "instance_id": obj.get("instance_id"),
            "ontology_id": obj["ontology_id"],
            "class_name": obj["class_name"],
            "position_m": position,
            "extents_m": extents,
            "visual_mesh": f"assets/{visual_path.name}",
            "collision_meshes": [f"assets/{collision_path.name}"],
            "collision_geom_count": 1,
            "source_kind": obj.get("geometry_source", "proxy_box_asset"),
            "visual_geometry_mode": "geometry_backed_visual"
            if source_mesh_rel
            else "proxy_box_visual",
            "collision_geometry_mode": "proxy_box_collision",
            "initial_pose_ref": "scene/object_init_poses_metric.json",
            "bbox_m": {
                "min": bbox_min,
                "max": bbox_max,
            },
            "metadata_ref": f"assets/{asset_name}.meta.json",
        }
        _write_object_metadata(
            assets_dir / f"{asset_name}.meta.json",
            manifest_object,
            world_frame=bundle.project.epsilon.metric_world_frame,
        )
        manifest["objects"].append(manifest_object)

    _write_asset_xml(assets_dir / "mujoco_assets.xml", manifest)
    _write_assets_only_xml(sim_dir / "assets_only.xml", manifest)
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    smoke_summary = _compile_and_smoke_assets(sim_dir / "assets_only.xml", sim_dir / "assets_only.mjb")
    (sim_dir / "assets_smoke.json").write_text(json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8")

    summary = _build_summary(bundle, manifest, smoke_summary)
    (assets_dir / "zeta_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    enforce_zeta_acceptance(bundle, summary)
    return summary
