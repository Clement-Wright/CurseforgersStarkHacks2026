from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

pytest.importorskip("pycocotools")

from video_task_compiler.cli import app
from video_task_compiler import object_extract
from video_task_compiler.object_extract import (
    PromptSpec,
    SeedFrameProbe,
    WristObservation,
    build_interaction_rows,
    build_object_prompt_specs,
    choose_seed_frame,
    normalize_grounded_sam2_tracks,
)
from video_task_compiler.specs import load_bundle, validate_bundle
from video_task_compiler.task_window import TaskWindow


runner = CliRunner()


def _copy_bundle_root(target_root: Path) -> Path:
    for name in ("spec", "env", "checklists"):
        shutil.copytree(Path(name), target_root / name)
    return target_root


def _square_mask(x_min: int, y_min: int, x_max: int, y_max: int, *, size: int = 10) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def _write_delta_beta_artifacts(beta_dir: Path) -> None:
    frames_dir = beta_dir / "frames"
    camera_dir = beta_dir / "camera"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for frame_idx in range(4):
        image_path = frames_dir / f"frame_{frame_idx:06d}.png"
        pixels = np.full((10, 10, 3), frame_idx * 40, dtype=np.uint8)
        object_extract.Image.fromarray(pixels, mode="RGB").save(image_path)
        frame_paths.append(image_path.resolve())

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
        for frame_idx, image_path in enumerate(frame_paths):
            segment = "preroll" if frame_idx == 0 else "demo"
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "frame_name": image_path.stem,
                    "image_path": image_path.as_posix(),
                    "pts": frame_idx * 30,
                    "pts_sec": float(frame_idx),
                    "t_ns": frame_idx * 1_000_000_000,
                    "segment": segment,
                    "is_keyframe": str(frame_idx == 0).lower(),
                    "registered": "true",
                    "pose_status": "registered",
                    "colmap_image_id": frame_idx + 1,
                }
            )

    (camera_dir / "camera_poses.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "pose_status": "registered",
                        "T_wc": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                    for frame_idx in range(4)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_gamma_smpl_tracks(gamma_dir: Path) -> None:
    human_dir = gamma_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[21] = np.array([2.0, 2.0, 0.9], dtype=np.float32)
    payload = {
        "tracks": {
            7: {
                "track_id": 7,
                "is_primary_demonstrator": True,
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "joints2d_xyc": joints,
                        "visibility": {"right_wrist": 0.9},
                    },
                    {
                        "frame_idx": 2,
                        "t_ns": 2_000_000_000,
                        "joints2d_xyc": joints,
                        "visibility": {"right_wrist": 0.9},
                    },
                ],
            }
        }
    }
    with (human_dir / "smpl_tracks.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def _mock_tracking_payload() -> dict[str, object]:
    return {
        "source": "Grounded-SAM-2",
        "tracks": [
            {
                "track_id": "obj_target",
                "ontology_id": "target_object",
                "frames": [
                    {
                        "frame_idx": 1,
                        "score": 0.95,
                        "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
                        "mask": _square_mask(1, 1, 4, 4),
                    },
                    {
                        "frame_idx": 2,
                        "score": 0.93,
                        "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
                        "mask": _square_mask(1, 1, 4, 4),
                    },
                ],
            },
            {
                "track_id": "obj_receptacle",
                "ontology_id": "receptacle",
                "frames": [
                    {
                        "frame_idx": 1,
                        "score": 0.97,
                        "bbox_xyxy": [6.0, 6.0, 9.0, 9.0],
                        "mask": _square_mask(6, 6, 9, 9),
                    },
                    {
                        "frame_idx": 2,
                        "score": 0.96,
                        "bbox_xyxy": [6.0, 6.0, 9.0, 9.0],
                        "mask": _square_mask(6, 6, 9, 9),
                    },
                ],
            },
            {
                "track_id": "obj_tabletop",
                "ontology_id": "tabletop",
                "frames": [
                    {
                        "frame_idx": 1,
                        "score": 0.99,
                        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                        "mask": _square_mask(0, 0, 10, 10),
                    },
                    {
                        "frame_idx": 2,
                        "score": 0.99,
                        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                        "mask": _square_mask(0, 0, 10, 10),
                    },
                ],
            },
        ],
    }


def test_build_object_prompt_specs_uses_ontology_ids_labels_and_task_references() -> None:
    bundle = validate_bundle(load_bundle(Path("spec")))

    prompts = build_object_prompt_specs(bundle)

    assert [prompt.ontology_id for prompt in prompts] == ["target_object", "receptacle", "tabletop"]
    assert prompts[0].prompt_text == "red cube"
    assert prompts[0].task_reference_prompt == "red cube"
    assert prompts[1].task_reference_prompt == "black tray center"
    assert prompts[2].task_reference_prompt is None


def test_choose_seed_frame_prefers_preroll_then_falls_back_to_active() -> None:
    frame_records = [
        object_extract.DeltaFrameRecord(0, "frame_000000", Path("/tmp/f0.png"), 0.0, 0, "preroll", "registered"),
        object_extract.DeltaFrameRecord(1, "frame_000001", Path("/tmp/f1.png"), 1.0, 1, "demo", "registered"),
    ]
    prompts = [
        PromptSpec("target_object", "target_object", "object", "red cube", "red cube", "red cube"),
        PromptSpec("receptacle", "receptacle", "object", "black tray", "black tray", "black tray center"),
        PromptSpec("tabletop", "tabletop", "surface", "tabletop", "tabletop", None),
    ]
    preroll_probes = [
        SeedFrameProbe(
            frame_idx=0,
            frame_name="frame_000000",
            segment="preroll",
            detected_ontology_ids=("target_object",),
            detection_scores={"target_object": 0.7},
        )
    ]
    active_probes = [
        SeedFrameProbe(
            frame_idx=1,
            frame_name="frame_000001",
            segment="demo",
            detected_ontology_ids=("receptacle", "tabletop", "target_object"),
            detection_scores={"target_object": 0.9, "receptacle": 0.8, "tabletop": 0.75},
        )
    ]

    seed_frame = choose_seed_frame(frame_records, prompts, preroll_probes, active_probes)

    assert seed_frame.frame_idx == 1
    assert seed_frame.candidate_pool == "active"


def test_normalize_grounded_sam2_tracks_round_trips_masks() -> None:
    frame_records = [
        object_extract.DeltaFrameRecord(1, "frame_000001", Path(__file__), 1.0, 1_000_000_000, "demo", "registered"),
    ]
    prompts = [
        PromptSpec("target_object", "target_object", "object", "red cube", "red cube", "red cube"),
    ]
    raw_payload = {
        "video_id": "demo_0001",
        "source": "Grounded-SAM-2",
        "tracks": [
            {
                "track_id": "obj_target",
                "ontology_id": "target_object",
                "frames": [
                    {
                        "frame_idx": 1,
                        "score": 0.91,
                        "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
                        "mask": _square_mask(1, 1, 4, 4),
                    }
                ],
            }
        ],
    }
    image_path = Path(__file__).parent / "tmp_object_extract_mask.png"
    object_extract.Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), mode="RGB").save(image_path)
    frame_records[0] = object_extract.DeltaFrameRecord(1, "frame_000001", image_path, 1.0, 1_000_000_000, "demo", "registered")

    normalized_payload, mask_entries, masks_by_track_frame, rle_decode_success = normalize_grounded_sam2_tracks(
        raw_payload,
        frame_records,
        prompts,
    )

    assert rle_decode_success is True
    assert normalized_payload["tracks"][0]["frames"][0]["mask_rle_ref"] == "objects/masks_rle.jsonl:1"
    assert mask_entries[0]["track_id"] == "obj_target"
    assert masks_by_track_frame[("obj_target", 1)].shape == (10, 10)
    image_path.unlink()


def test_build_interaction_rows_obeys_distance_and_confidence_thresholds() -> None:
    bundle = validate_bundle(load_bundle(Path("spec")))
    normalized_payload = {
        "tracks": [
            {
                "track_id": "obj_target",
                "ontology_id": "target_object",
                "class_name": "target_object",
                "frames": [
                    {
                        "frame_idx": 1,
                        "t_ns": 1_000_000_000,
                        "bbox_xyxy": [1.0, 1.0, 4.0, 4.0],
                        "score": 0.9,
                        "mask_rle_ref": "objects/masks_rle.jsonl:1",
                        "centroid_uv": [2.0, 2.0],
                        "visible": True,
                    }
                ],
            }
        ]
    }
    mask = _square_mask(1, 1, 4, 4)
    interaction_rows = build_interaction_rows(
        bundle=bundle,
        normalized_payload=normalized_payload,
        masks_by_track_frame={("obj_target", 1): mask},
        wrist_observations={1: WristObservation(1, 1_000_000_000, 2.0, 2.0, 0.9)},
        task_window=TaskWindow(start_frame_idx=1, end_frame_idx=1, source="cli_override"),
    )

    assert interaction_rows[0]["pixel_distance_wrist_to_mask"] == pytest.approx(0.0)
    assert interaction_rows[0]["likely_contact_boolean"] is True


def test_extract_monocular_objects_success_writes_delta_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    out_dir = tmp_path / "delta"
    _write_delta_beta_artifacts(beta_dir)
    _write_gamma_smpl_tracks(beta_dir)
    bundle = validate_bundle(load_bundle(bundle_root / "spec"))

    monkeypatch.setattr(object_extract, "ensure_delta_dependencies", lambda grounded_sam2_root: tmp_path / "Grounded-SAM-2")
    monkeypatch.setattr(
        object_extract,
        "probe_seed_frame_candidates",
        lambda grounded_sam2_root, frame_candidates, prompts, device, work_dir: [
            SeedFrameProbe(
                frame_idx=frame_candidates[0].frame_idx,
                frame_name=frame_candidates[0].frame_name,
                segment=frame_candidates[0].segment,
                detected_ontology_ids=("target_object", "receptacle", "tabletop"),
                detection_scores={"target_object": 0.9, "receptacle": 0.85, "tabletop": 0.8},
            )
        ],
    )
    monkeypatch.setattr(object_extract, "run_grounded_sam2_tracking", lambda **_: _mock_tracking_payload())

    summary = object_extract.extract_monocular_objects(
        bundle=bundle,
        beta_dir=beta_dir,
        out_dir=out_dir,
        grounded_sam2_root=tmp_path / "Grounded-SAM-2",
    )

    assert summary["qc_flags"]["all_ontology_entities_tracked"] is True
    assert (out_dir / "task_window.json").exists()
    assert (out_dir / "objects" / "prompts.yaml").exists()
    assert (out_dir / "objects" / "object_tracks.json").exists()
    assert (out_dir / "objects" / "masks_rle.jsonl").exists()
    assert (out_dir / "objects" / "interactions.parquet").exists()
    assert (out_dir / "objects" / "summary.json").exists()
    assert any(path.name.startswith("frame_000001") for path in (out_dir / "objects" / "overlays").iterdir())

    object_tracks = json.loads((out_dir / "objects" / "object_tracks.json").read_text(encoding="utf-8"))
    assert len(object_tracks["tracks"]) == 3
    interaction_table = pq.read_table(out_dir / "objects" / "interactions.parquet")
    interaction_rows = interaction_table.to_pylist()
    assert len(interaction_rows) == 6
    assert any(row["likely_contact_boolean"] for row in interaction_rows if row["class_name"] == "target_object")
    assert all(not row["likely_contact_boolean"] for row in interaction_rows if row["class_name"] == "tabletop")

    seen_pairs = {(row["track_id"], row["frame_idx"]) for row in interaction_rows}
    assert len(seen_pairs) == len(interaction_rows)


def test_extract_monocular_objects_uses_persisted_task_window_for_interactions_and_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    out_dir = tmp_path / "delta"
    _write_delta_beta_artifacts(beta_dir)
    _write_gamma_smpl_tracks(beta_dir)
    (beta_dir / "task_window.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "video_id": "demo_0001",
                "source": "cli_override",
                "start_frame_idx": 2,
                "end_frame_idx": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = validate_bundle(load_bundle(bundle_root / "spec"))

    monkeypatch.setattr(object_extract, "ensure_delta_dependencies", lambda grounded_sam2_root: tmp_path / "Grounded-SAM-2")
    monkeypatch.setattr(
        object_extract,
        "probe_seed_frame_candidates",
        lambda grounded_sam2_root, frame_candidates, prompts, device, work_dir: [
            SeedFrameProbe(
                frame_idx=frame_candidates[0].frame_idx,
                frame_name=frame_candidates[0].frame_name,
                segment=frame_candidates[0].segment,
                detected_ontology_ids=("target_object", "receptacle", "tabletop"),
                detection_scores={"target_object": 0.9, "receptacle": 0.85, "tabletop": 0.8},
            )
        ],
    )
    monkeypatch.setattr(object_extract, "run_grounded_sam2_tracking", lambda **_: _mock_tracking_payload())

    summary = object_extract.extract_monocular_objects(
        bundle=bundle,
        beta_dir=beta_dir,
        out_dir=out_dir,
        grounded_sam2_root=tmp_path / "Grounded-SAM-2",
    )

    assert summary["task_window"]["start_frame_idx"] == 2
    interaction_rows = pq.read_table(out_dir / "objects" / "interactions.parquet").to_pylist()
    assert {row["frame_idx"] for row in interaction_rows} == {2}
    overlay_names = {path.stem for path in (out_dir / "objects" / "overlays").iterdir()}
    assert overlay_names == {"frame_000002"}


def test_extract_monocular_objects_without_gamma_writes_null_wrist_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    out_dir = tmp_path / "delta"
    _write_delta_beta_artifacts(beta_dir)
    bundle = validate_bundle(load_bundle(bundle_root / "spec"))

    monkeypatch.setattr(object_extract, "ensure_delta_dependencies", lambda grounded_sam2_root: tmp_path / "Grounded-SAM-2")
    monkeypatch.setattr(
        object_extract,
        "probe_seed_frame_candidates",
        lambda grounded_sam2_root, frame_candidates, prompts, device, work_dir: [
            SeedFrameProbe(
                frame_idx=frame_candidates[0].frame_idx,
                frame_name=frame_candidates[0].frame_name,
                segment=frame_candidates[0].segment,
                detected_ontology_ids=("target_object", "receptacle", "tabletop"),
                detection_scores={"target_object": 0.9, "receptacle": 0.85, "tabletop": 0.8},
            )
        ],
    )
    monkeypatch.setattr(object_extract, "run_grounded_sam2_tracking", lambda **_: _mock_tracking_payload())

    object_extract.extract_monocular_objects(
        bundle=bundle,
        beta_dir=beta_dir,
        out_dir=out_dir,
        grounded_sam2_root=tmp_path / "Grounded-SAM-2",
    )

    interaction_table = pq.read_table(out_dir / "objects" / "interactions.parquet")
    interaction_rows = interaction_table.to_pylist()
    assert all(row["wrist_u"] is None for row in interaction_rows)
    assert all(row["likely_contact_boolean"] is False for row in interaction_rows)


def test_objects_cli_missing_grounded_sam2_checkout_hard_fails(tmp_path: Path) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    _write_delta_beta_artifacts(beta_dir)

    result = runner.invoke(
        app,
        [
            "objects",
            "extract-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--out-dir",
            str(tmp_path / "delta"),
        ],
    )

    assert result.exit_code == 1
    assert "Grounded-SAM-2 checkout not found" in result.stdout


def test_objects_cli_missing_beta_artifacts_hard_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = _copy_bundle_root(tmp_path / "bundle")
    beta_dir = tmp_path / "beta"
    beta_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(object_extract, "ensure_delta_dependencies", lambda grounded_sam2_root: tmp_path / "Grounded-SAM-2")

    result = runner.invoke(
        app,
        [
            "objects",
            "extract-monocular",
            "--spec-dir",
            str(bundle_root / "spec"),
            "--beta-dir",
            str(beta_dir),
            "--out-dir",
            str(tmp_path / "delta"),
        ],
    )

    assert result.exit_code == 1
    assert "missing beta frames directory" in result.stdout


@pytest.mark.skipif(not os.environ.get("GROUNDED_SAM2_ROOT"), reason="requires a local Grounded-SAM-2 checkout")
def test_optional_grounded_sam2_smoke_check() -> None:
    root = object_extract.resolve_grounded_sam2_root(None)
    assert root.exists()
