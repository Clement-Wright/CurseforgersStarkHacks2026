from __future__ import annotations

import importlib.util
import json
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


def _build_summary(bundle: SpecBundle, manifest: dict[str, Any]) -> dict[str, Any]:
    max_collision_geoms = max(
        [len(obj["collision_meshes"]) for obj in manifest["objects"]] or [len(manifest["static"]["collision_meshes"])]
    )
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "video_id": bundle.project.video_id,
        "source": bundle.project.zeta.primary_backbone,
        "object_asset_count": len(manifest["objects"]),
        "static_collision_mesh_count": len(manifest["static"]["collision_meshes"]),
        "max_collision_geoms_per_object": max_collision_geoms,
        "qc_flags": {
            "collision_geom_count_ok": max_collision_geoms <= bundle.project.acceptance.zeta_max_collision_geoms_per_object,
        },
    }


def enforce_zeta_acceptance(bundle: SpecBundle, summary: dict[str, Any]) -> None:
    if summary["max_collision_geoms_per_object"] > bundle.project.acceptance.zeta_max_collision_geoms_per_object:
        raise ZetaAssetError(
            f"collision geom count {summary['max_collision_geoms_per_object']} exceeded "
            f"zeta_max_collision_geoms_per_object={bundle.project.acceptance.zeta_max_collision_geoms_per_object}"
        )


def assetize_metric_scene(
    bundle: SpecBundle,
    epsilon_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_zeta_dependencies()

    scene_dir = epsilon_dir / "scene"
    scene_meta = _load_json(scene_dir / "static_mesh.meta.json")
    object_payload = _load_json(scene_dir / "object_init_poses_metric.json")
    static_mesh_path = scene_dir / "static_mesh.obj"
    if not static_mesh_path.exists():
        raise ZetaAssetError(f"missing epsilon static mesh: {static_mesh_path}")

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

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
        "static": {
            "visual_mesh": "assets/static_visual.obj",
            "collision_meshes": ["assets/static_collision_00.obj"],
        },
        "objects": [],
    }
    for obj in object_payload.get("objects", []):
        if not isinstance(obj, dict):
            continue
        asset_name = _sanitize_name(str(obj["track_id"]))
        extents = np.asarray(obj.get("extents_m", [0.04, 0.04, 0.04]), dtype=float)
        vertices, faces = _box_vertices_faces(np.zeros(3, dtype=float), extents)
        visual_path = assets_dir / f"{asset_name}_visual.obj"
        collision_path = assets_dir / f"{asset_name}_collision_00.obj"
        _write_obj_mesh(visual_path, vertices, faces)
        _write_obj_mesh(collision_path, vertices, faces)
        manifest["objects"].append(
            {
                "track_id": obj["track_id"],
                "ontology_id": obj["ontology_id"],
                "class_name": obj["class_name"],
                "position_m": obj["position_m"],
                "extents_m": obj["extents_m"],
                "visual_mesh": f"assets/{visual_path.name}",
                "collision_meshes": [f"assets/{collision_path.name}"],
                "collision_geom_count": 1,
            }
        )

    _write_asset_xml(assets_dir / "mujoco_assets.xml", manifest)
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = _build_summary(bundle, manifest)
    (assets_dir / "zeta_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    enforce_zeta_acceptance(bundle, summary)
    return summary
